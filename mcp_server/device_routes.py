"""Device Authorization Grant (RFC 8628) for `synapse login` — a browser-free CLI login.

The loopback authorization-code flow needs an interactive browser on the SAME host that runs
the login script. That's wrong for servers and headless boxes. This proxies GitHub's native
device flow instead: the box prints a short code, the human approves at github.com/login/device
on ANY device, and the box polls until GitHub confirms — no same-host browser, no loopback
redirect, no redirect URIs at all.

Two unauthenticated bootstrap routes (custom routes bypass FastMCP's auth middleware by design —
these are the pre-token handshake). Security rests entirely on GitHub: the machine token is
handed back ONLY after GitHub confirms the device was approved AND the approving user's login is
in ALLOWED_GITHUB_USERS (the same gate as the web/MCP leg). Stateless — the device_code lives on
the client and is replayed on each poll; we keep no server-side state.

  POST /device/code   {}              -> {user_code, verification_uri, device_code, interval, ...}
  POST /device/token  {device_code}   -> {token} | {error: authorization_pending|access_denied|...}
"""

from __future__ import annotations

import logging

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

_GH_DEVICE_CODE = "https://github.com/login/device/code"
_GH_DEVICE_TOKEN = "https://github.com/login/oauth/access_token"
_GH_USER = "https://api.github.com/user"
_DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
# read:user is enough to read the login for the allowlist check; matches the web leg's "user".
_SCOPE = "read:user"
_UA = "synapse-device-login/1.0"
_TIMEOUT = 15.0


def _err(error: str, description: str, status: int) -> JSONResponse:
    return JSONResponse({"error": error, "error_description": description}, status_code=status)


def register(
    mcp,
    client_id: str,
    client_secret: str,
    allowed: set[str],
    machine_token: str,
) -> None:
    """Wire the device-flow routes. No-op unless a GitHub app AND a machine token are set —
    without GitHub there's no identity to gate on, without a token there's nothing to hand back."""
    if not (client_id and machine_token):
        logger.info("device-login routes disabled (need GITHUB_CLIENT_ID + SYNAPSE_MACHINE_TOKEN)")
        return

    @mcp.custom_route("/device/code", methods=["POST"])
    async def device_code(request: Request) -> JSONResponse:
        """Start a device login: ask GitHub for a device + user code, pass them to the client."""
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.post(
                    _GH_DEVICE_CODE,
                    data={"client_id": client_id, "scope": _SCOPE},
                    headers={"Accept": "application/json", "User-Agent": _UA},
                )
            data = resp.json()
        except Exception as e:
            logger.warning("device/code: GitHub call failed: %s", e)
            return _err("server_error", str(e), 502)

        if "device_code" not in data:
            # e.g. {"error":"device_flow_disabled"} — the OAuth App hasn't enabled device flow.
            logger.warning("device/code: GitHub returned %s", data)
            return JSONResponse(data, status_code=400)

        return JSONResponse(
            {
                "device_code": data["device_code"],
                "user_code": data["user_code"],
                "verification_uri": data.get("verification_uri", "https://github.com/login/device"),
                "verification_uri_complete": data.get("verification_uri_complete"),
                "expires_in": data.get("expires_in", 900),
                "interval": data.get("interval", 5),
            }
        )

    @mcp.custom_route("/device/token", methods=["POST"])
    async def device_token(request: Request) -> JSONResponse:
        """Poll: exchange the device_code at GitHub; on approval, gate by allowlist and return
        the machine token. Pending/slow_down pass back so the client keeps polling."""
        try:
            body = await request.json()
        except Exception:
            body = {}
        device = body.get("device_code")
        if not device:
            return _err("invalid_request", "device_code required", 400)

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                tok_resp = await client.post(
                    _GH_DEVICE_TOKEN,
                    data={
                        "client_id": client_id,
                        "device_code": device,
                        "grant_type": _DEVICE_GRANT,
                    },
                    headers={"Accept": "application/json", "User-Agent": _UA},
                )
                tok = tok_resp.json()
        except Exception as e:
            logger.warning("device/token: GitHub token poll failed: %s", e)
            return _err("server_error", str(e), 502)

        access = tok.get("access_token")
        if not access:
            # authorization_pending / slow_down (poll on) or expired_token / access_denied (done).
            return JSONResponse({"error": tok.get("error", "authorization_pending")})

        # Approved by GitHub — now enforce OUR allowlist before handing back the token.
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                user_resp = await client.get(
                    _GH_USER,
                    headers={
                        "Authorization": f"Bearer {access}",
                        "Accept": "application/vnd.github+json",
                        "User-Agent": _UA,
                    },
                )
                login = str(user_resp.json().get("login", "")).lower()
        except Exception as e:
            logger.warning("device/token: GitHub /user lookup failed: %s", e)
            return _err("server_error", str(e), 502)

        if not login or login not in allowed:
            logger.warning("device/token: github user %r not in allowlist", login)
            return _err("access_denied", f"github user {login!r} not in allowlist", 403)

        logger.info("device-login: issued machine token to github user %r", login)
        return JSONResponse({"token": machine_token, "login": login})
