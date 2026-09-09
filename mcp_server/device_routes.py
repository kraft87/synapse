"""Device Authorization Grant (RFC 8628) for `synapse login` — a browser-free CLI login.

The loopback authorization-code flow needs an interactive browser on the SAME host that
runs the login script. That's wrong for servers and headless boxes. This proxies the
configured IdP's device flow instead: the box prints a short code, the human approves it
in a browser on ANY device, and the box polls until the IdP confirms — no same-host
browser, no loopback redirect, no redirect URIs at all.

Two unauthenticated bootstrap routes (custom routes bypass FastMCP's auth middleware by
design — these are the pre-token handshake). Security rests entirely on the IdP: the
machine token is handed back ONLY after the IdP confirms the device was approved AND the
approving user's identity is in the allowlist (the same gate as the web/MCP leg).
Stateless — the device_code lives on the client and is replayed on each poll; we keep no
server-side state.

  POST /device/code   {}              -> {user_code, verification_uri, device_code, interval, ...}
  POST /device/token  {device_code}   -> {token} | {error: authorization_pending|access_denied|...}
"""

from __future__ import annotations

import logging
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


def _err(error: str, description: str, status: int) -> JSONResponse:
    return JSONResponse({"error": error, "error_description": description}, status_code=status)


def register(
    mcp: Any,
    idp: Any,
    machine_token: str,
) -> None:
    """Wire the device-flow routes. Needs an identity provider AND a machine token —
    without an IdP there's no identity to gate on, without a token nothing to hand back.

    When either is missing the routes still EXIST and answer 503 with a machine-readable
    ``error`` (``no_idp`` / ``no_machine_token``). They used to be absent, so the client
    got a bare 404 and printed "enrollment unavailable: unknown" — which reads like a bug
    on a deployment where it is simply the truth that there is nobody to sign in to. The
    remedy in that case is ``synapse-admin bootstrap``, and the client can only say so if
    the server tells it which case this is.
    """
    if not (idp and machine_token):
        reason = "no_idp" if machine_token else "no_machine_token"
        logger.info(
            "device-login routes disabled (%s) — /device/* will report it to clients", reason
        )
        description = (
            "no identity provider is configured on this server, so there is no sign-in to "
            "approve; run 'docker compose exec mcp-server synapse-admin bootstrap \"<label>\"' "
            "on the server and paste the token into SYNAPSE_INGEST_TOKEN"
            if reason == "no_idp"
            else "this server has no SYNAPSE_MACHINE_TOKEN set, so it has no credential to issue"
        )

        @mcp.custom_route("/device/code", methods=["POST"])  # type: ignore[misc]
        async def device_code_unavailable(request: Request) -> JSONResponse:
            return _err(reason, description, 503)

        @mcp.custom_route("/device/token", methods=["POST"])  # type: ignore[misc]
        async def device_token_unavailable(request: Request) -> JSONResponse:
            return _err(reason, description, 503)

        return

    @mcp.custom_route("/device/code", methods=["POST"])  # type: ignore[misc]
    async def device_code(request: Request) -> JSONResponse:
        """Start a device login: ask the IdP for a device + user code, pass them to the client."""
        try:
            data = await idp.device_start()
        except Exception as e:
            logger.warning("device/code: %s call failed: %s", idp.label, e)
            return _err("server_error", str(e), 502)

        if "device_code" not in data:
            # e.g. {"error":"device_flow_disabled"} — the IdP has no device grant enabled.
            logger.warning("device/code: %s returned %s", idp.label, data)
            data.setdefault("error", "device_flow_disabled")
            data.setdefault("error_description", idp.device_disabled_hint)
            return JSONResponse(data, status_code=400)

        return JSONResponse(
            {
                "device_code": data["device_code"],
                "user_code": data["user_code"],
                "verification_uri": data.get("verification_uri", ""),
                "verification_uri_complete": data.get("verification_uri_complete"),
                "expires_in": data.get("expires_in", 900),
                "interval": data.get("interval", 5),
            }
        )

    @mcp.custom_route("/device/token", methods=["POST"])  # type: ignore[misc]
    async def device_token(request: Request) -> JSONResponse:
        """Poll: exchange the device_code at the IdP; on approval, gate by allowlist and return
        the machine token. Pending/slow_down pass back so the client keeps polling."""
        try:
            body = await request.json()
        except Exception:
            body = {}
        device = body.get("device_code")
        if not device:
            return _err("invalid_request", "device_code required", 400)

        try:
            tok = await idp.device_poll(device)
        except Exception as e:
            logger.warning("device/token: %s token poll failed: %s", idp.label, e)
            return _err("server_error", str(e), 502)

        access = tok.get("access_token")
        if not access:
            # authorization_pending / slow_down (poll on) or expired_token / access_denied (done).
            return JSONResponse({"error": tok.get("error", "authorization_pending")})

        # Approved by the IdP — now enforce OUR allowlist before handing back the token.
        try:
            identity = await idp.fetch_identity(access)
        except Exception as e:
            logger.warning("device/token: %s identity lookup failed: %s", idp.label, e)
            return _err("server_error", str(e), 502)

        if not identity or identity not in idp.allowed:
            logger.warning("device/token: %s user %r not in allowlist", idp.label, identity)
            return _err("access_denied", f"{idp.label} user {identity!r} not in allowlist", 403)

        logger.info("device-login: issued machine token to %s user %r", idp.label, identity)
        return JSONResponse({"token": machine_token, "login": identity})
