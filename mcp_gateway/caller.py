"""Who is calling the gateway, answered by Synapse itself.

The gateway holds no Synapse credential and no database DSN. It authenticates a caller by
presenting the caller's OWN bearer to Synapse's ``GET /auth/whoami``; Synapse's answer
(``kind``, ``trust``, ``surface_id``) becomes the gateway-side claims. Consequences:

* one source of truth for device approval, trust and revocation — the gateway cannot
  disagree with Synapse about who a caller is;
* every Synapse MCP call is still re-authenticated upstream with that same bearer, so a
  revoked device loses memory access on its very next call regardless of the short
  identity cache below (which only bounds how long research access can outlive it);
* only APPROVED DEVICE tokens are admitted. The root/enrollment token identifies a
  deployment rather than a device, and the gateway has no business relaying it.

Fail closed: Synapse unreachable, a non-200, or a malformed body all mean "unauthenticated".
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

import httpx
from fastmcp.server.auth.auth import AccessToken, TokenVerifier
from fastmcp.server.dependencies import get_access_token

logger = logging.getLogger(__name__)

CLIENT_ID = "synapse-gateway-caller"
KIND_DEVICE = "device"
_MAX_CACHED = 1024


def identity_key(token: str) -> str:
    """Stable, non-reversible key for per-identity caches and logs."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class SynapseCallerVerifier(TokenVerifier):
    """Admit a bearer iff Synapse's ``/auth/whoami`` says it is an approved device."""

    def __init__(
        self,
        whoami_url: str,
        *,
        cache_ttl: float = 30.0,
        timeout: float = 5.0,
        base_url: str | None = None,
    ) -> None:
        super().__init__(base_url=base_url or None)
        self._whoami_url = whoami_url
        self._ttl = cache_ttl
        self._timeout = timeout
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}

    async def verify_token(self, token: str) -> AccessToken | None:
        if not token:
            return None
        key = identity_key(token)
        now = time.monotonic()
        hit = self._cache.get(key)
        if hit and now - hit[0] < self._ttl:
            claims = hit[1]
        else:
            fresh = await self._whoami(token)
            if fresh is None:
                self._cache.pop(key, None)
                return None
            claims = fresh
            if self._ttl > 0:
                if len(self._cache) >= _MAX_CACHED:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[key] = (now, claims)
        return AccessToken(token=token, client_id=CLIENT_ID, scopes=[], claims=dict(claims))

    async def _whoami(self, token: str) -> dict[str, Any] | None:
        # A fresh client per lookup: a shared httpx client keeps a cookie jar, and nothing
        # one caller's response sets may ride along on another caller's request.
        try:
            async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=False) as c:
                r = await c.get(self._whoami_url, headers={"Authorization": f"Bearer {token}"})
        except httpx.HTTPError as e:
            logger.warning("caller verification unavailable (%s)", type(e).__name__)
            return None
        if r.status_code != 200:
            if r.status_code not in (401, 403):
                logger.warning("caller verification returned HTTP %d", r.status_code)
            return None
        try:
            body = r.json()
        except ValueError:
            return None
        if not isinstance(body, dict) or body.get("kind") != KIND_DEVICE:
            return None
        trust = body.get("trust")
        if trust not in ("full", "restricted"):
            return None
        return {"kind": KIND_DEVICE, "trust": trust, "surface_id": body.get("surface_id")}


def current_token() -> AccessToken | None:
    try:
        return get_access_token()
    except Exception:  # pragma: no cover - no request context at all
        return None


def current_bearer() -> str:
    """The authenticated caller's raw bearer; raises when there is no verified caller."""
    tok = current_token()
    if tok is None or tok.client_id != CLIENT_ID or not tok.token:
        raise PermissionError("no authenticated gateway caller")
    return tok.token


def caller_trust() -> str | None:
    tok = current_token()
    if tok is None or tok.client_id != CLIENT_ID:
        return None
    return (tok.claims or {}).get("trust")


def trust_allows(required: str) -> bool:
    """``required='full'`` admits only full-trust devices; ``'restricted'`` admits any."""
    trust = caller_trust()
    if trust is None:
        return False
    return trust == "full" or required == "restricted"
