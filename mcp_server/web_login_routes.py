"""Browser (authorization-code) login for the dashboard — the redirect UX.

The dashboard is served on hosts the IdP can't redirect to (the registered callback
lives on the PUBLIC base URL), so the flow bounces:

  LAN /dash → GET /dash/oauth/start            (302 to the IdP's authorize endpoint;
                                                 signed state carries the return origin)
  → IdP approves → PUBLIC /auth/callback/dash  (code→token exchange, allowlist gate)
  → 302 {origin}/dash#token=<machine token>    (the app's existing fragment bootstrap)

GitHub accepts redirect URIs that are SUBPATHS of the registered callback, which is
how /auth/callback/dash coexists with the MCP leg on one OAuth App; OIDC providers
match exactly, so register both /auth/callback and /auth/callback/dash on the client.

Identity + authorization are the SAME gate as the MCP leg and `synapse login`: the
IdP confirms the account, the allowlist authorizes it, and what's handed back is the
machine token. The state is HMAC-signed (key = machine token, never sent) with a
10-minute TTL, and the return origin must be in SYNAPSE_DASH_ORIGINS (plus the public
base) — without that check the cross-host bounce would be an open redirect that
exfiltrates the token to an attacker-supplied Host header.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import time
from typing import Any
from urllib.parse import quote

from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response

logger = logging.getLogger(__name__)

_STATE_TTL_S = 600
# Subpath of the registered callback (the FastMCP OAuthProxy's {base_url}/auth/callback).
_CALLBACK_PATH = "/auth/callback/dash"


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _sign_state(key: str, origin: str) -> str:
    payload = _b64(json.dumps({"o": origin, "t": int(time.time())}).encode())
    sig = _b64(hmac.new(key.encode(), payload.encode(), hashlib.sha256).digest())
    return f"{payload}.{sig}"


def _verify_state(key: str, state: str) -> str | None:
    """Return the embedded origin, or None if the state is malformed/forged/expired."""
    try:
        payload, sig = state.split(".", 1)
        want = _b64(hmac.new(key.encode(), payload.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, want):
            return None
        data = json.loads(_unb64(payload))
        if time.time() - float(data["t"]) > _STATE_TTL_S:
            return None
        return str(data["o"])
    except Exception:
        return None


def register(
    mcp: Any,
    idp: Any,
    machine_token: str,
    public_url: str,
) -> None:
    """Wire the browser-login routes. Same enablement condition as the device flow."""
    if not (idp and machine_token):
        logger.info("web-login routes disabled (need an identity provider + SYNAPSE_MACHINE_TOKEN)")
        return

    public_origin = public_url.rstrip("/")
    # Origins the flow may RETURN the token to. The public base is always allowed;
    # operators add their LAN/other dashboard origins (comma-separated, scheme://host[:port]).
    extra = os.environ.get("SYNAPSE_DASH_ORIGINS", "")
    allowed_origins = {public_origin} | {
        o.strip().rstrip("/") for o in extra.split(",") if o.strip()
    }
    redirect_uri = public_origin + _CALLBACK_PATH

    def _request_origin(request: Request) -> str:
        proto = request.headers.get("x-forwarded-proto") or request.url.scheme
        host = request.headers.get("x-forwarded-host") or request.headers.get("host") or ""
        return f"{proto}://{host}".rstrip("/")

    @mcp.custom_route("/dash/oauth/start", methods=["GET"])  # type: ignore[misc]
    async def oauth_start(request: Request) -> Response:
        origin = _request_origin(request)
        if origin not in allowed_origins:
            # Not redirected anywhere attacker-chosen — just refused.
            return JSONResponse(
                {"status": "error", "detail": f"origin {origin!r} not allowed for dashboard login"},
                status_code=403,
            )
        try:
            target = await idp.authorize_url(redirect_uri, _sign_state(machine_token, origin))
        except Exception as e:  # e.g. OIDC discovery unreachable
            logger.warning("web-login: could not build authorize URL: %s", e)
            return JSONResponse(
                {"status": "error", "detail": "identity provider unavailable"}, status_code=502
            )
        return RedirectResponse(target, status_code=302)

    @mcp.custom_route(_CALLBACK_PATH, methods=["GET"])  # type: ignore[misc]
    async def oauth_callback(request: Request) -> Response:
        state = request.query_params.get("state") or ""
        origin = _verify_state(machine_token, state)
        if origin is None or origin not in allowed_origins:
            # No trustworthy return target -> plain error, never a redirect.
            return JSONResponse(
                {"status": "error", "detail": "invalid or expired login state"}, status_code=400
            )

        def fail(msg: str) -> Response:
            return RedirectResponse(f"{origin}/dash#login_error={quote(msg)}", status_code=302)

        if request.query_params.get("error"):
            return fail(request.query_params.get("error_description") or "sign-in cancelled")
        code = request.query_params.get("code") or ""
        if not code:
            return fail("missing authorization code")

        try:
            access = await idp.exchange_code(code, redirect_uri)
        except Exception as e:
            logger.warning("web-login: code exchange failed: %s", e)
            return fail(f"{idp.label} token exchange failed")
        if not access:
            return fail(f"{idp.label} rejected the authorization code")

        try:
            identity = await idp.fetch_identity(access)
        except Exception as e:
            logger.warning("web-login: identity lookup failed: %s", e)
            return fail(f"{idp.label} user lookup failed")
        if not identity or identity not in idp.allowed:
            logger.warning("web-login: %s user %r not in allowlist", idp.label, identity)
            return fail(f"{idp.label} user {identity!r} not in allowlist")

        logger.info(
            "web-login: issued machine token to %s user %r via %s", idp.label, identity, origin
        )
        return RedirectResponse(f"{origin}/dash#token={quote(machine_token)}", status_code=302)
