"""``GET /auth/whoami`` — what THIS bearer is, answered for the bearer itself.

The MCP gateway (``mcp_gateway``) authenticates its callers by presenting each caller's own
bearer here, so it needs neither a Synapse credential of its own nor a database DSN, and
cannot disagree with Synapse about who a device is. The answer is the caller's own trust
verdict — no token, hash, project allowlist, or anything about other surfaces:

  root token        -> {"kind": "root", "trust": "restricted", "surface_id": null}
  approved device   -> {"kind": "device", "trust": "full"|"restricted", "surface_id": "..."}
  anything else     -> 401 (pending, revoked and garbage are indistinguishable, as on /mcp)

Same resolution as :class:`~mcp_server.auth_tokens.SynapseTokenVerifier`. A server with no
machine token has no credential system to describe, so it answers 503 rather than "open".
"""

from __future__ import annotations

import asyncio
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from ingestion.surfaces import resolve_caller, token_hash
from mcp_server.http_auth import bearer, is_root

_NO_STORE = {"Cache-Control": "no-store"}


def register(mcp: Any, DB_URL: str, MACHINE_TOKEN: str) -> None:
    @mcp.custom_route("/auth/whoami", methods=["GET"])  # type: ignore[misc]
    async def auth_whoami(request: Request) -> JSONResponse:
        if not MACHINE_TOKEN:
            return JSONResponse({"error": "no_machine_token"}, status_code=503, headers=_NO_STORE)
        tok = bearer(request)
        if not tok:
            return JSONResponse({"error": "unauthorized"}, status_code=401, headers=_NO_STORE)
        if is_root(request, MACHINE_TOKEN):
            body: dict[str, Any] = {"kind": "root", "trust": "restricted", "surface_id": None}
            return JSONResponse(body, headers=_NO_STORE)
        st = await asyncio.to_thread(resolve_caller, DB_URL, token_hash_hex=token_hash(tok))
        if not st.known:
            return JSONResponse({"error": "unauthorized"}, status_code=401, headers=_NO_STORE)
        body = {"kind": "device", "trust": st.trust, "surface_id": st.surface_id}
        return JSONResponse(body, headers=_NO_STORE)
