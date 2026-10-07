"""Resolve the trust scope attached to an MCP or plain-HTTP caller.

This module deliberately knows nothing about server configuration or FastMCP globals.
The server passes request state and configuration in, which keeps credential precedence
testable without booting the MCP application.

A caller's read scope comes from exactly two places, and neither is something the caller
says about itself:

1. **Device token**: the verified claims (MCP) or the token's hash (plain HTTP).
2. **OAuth/OIDC identity**: the server-derived ``oauth:<login>`` id on the MCP lane.

Everything else resolves to :data:`~ingestion.surfaces.UNKNOWN_SURFACE`, which is
restricted with an empty allowlist. That covers the shared root machine token, an open
(tokenless) server, and an OAuth token with no identity claim. The root token in
particular identifies a deployment, not a machine. Every machine that ever ran the plugin
has held it, so letting it name a ``surface`` id would let any of them borrow any row's
trust, ``full`` included. There is no parameter left here for a caller to supply.
"""

from __future__ import annotations

import hmac
import logging
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from ingestion.surfaces import UNKNOWN_SURFACE, SurfaceTrust, resolve_caller, token_hash
from mcp_server.auth_tokens import KIND_DEVICE, claims_of

logger = logging.getLogger(__name__)

OAUTH_SURFACE_PREFIX = "oauth:"

#: Call sites that have already logged an ignored ``surface`` param this process.
_IGNORED_SURFACE_SITES: set[str] = set()


def trust_from_claims(claims: dict[str, Any]) -> SurfaceTrust:
    """Turn a verified device credential's claims into its trust verdict."""
    return SurfaceTrust(
        surface_id=claims.get("surface_id"),
        trust=str(claims.get("trust") or "restricted"),
        allowed_projects=tuple(claims.get("allowed_projects") or ()),
        known=True,
        credential_bound=True,
    )


def caller_trust(
    *,
    db_url: str,
    access_token: Any,
    identity_claims: tuple[str, ...],
    machine_client_ids: set[str],
    claims_identity: Callable[[dict[str, Any], tuple[str, ...]], str],
) -> SurfaceTrust:
    """Resolve the trust for an MCP call from the credential alone.

    Device claims are already authenticated by FastMCP and therefore need no second
    database read. OAuth identities map to a server-derived ``oauth:<identity>``
    surface. A root-token caller, a call with no token context, and an OAuth token that
    carries no identity claim all resolve to :data:`UNKNOWN_SURFACE`.
    """
    claims = claims_of(access_token)
    if claims.get("kind") == KIND_DEVICE:
        return trust_from_claims(claims)
    if access_token is not None and access_token.client_id not in machine_client_ids:
        identity = claims_identity(access_token.claims or {}, identity_claims)
        if identity:
            st = resolve_caller(db_url, legacy_surface_id=f"{OAUTH_SURFACE_PREFIX}{identity}")
            # The id was derived by the server from a VERIFIED identity, so it is as
            # credential-bound as a device token (schema 057 provenance rides on it).
            # Only a matched, approved row counts.
            return replace(st, credential_bound=True) if st.known else st
    return UNKNOWN_SURFACE


def request_trust(*, db_url: str, machine_token: str, bearer: str) -> SurfaceTrust:
    """Resolve trust for a custom HTTP route, which has no FastMCP token context.

    A bearer that is not the root token is looked up as a device token by its hash. The
    root token, a missing bearer, and an open server (no machine token configured) all
    resolve to :data:`UNKNOWN_SURFACE`.
    """
    if machine_token and bearer and not hmac.compare_digest(bearer, machine_token):
        return resolve_caller(db_url, token_hash_hex=token_hash(bearer))
    return UNKNOWN_SURFACE


def note_ignored_surface(surface: str | None, site: str) -> None:
    """Log, once per call site per process, that a client sent the retired ``surface``.

    The value plays no part in trust. The log line exists so an operator can find
    clients that still send it (a pre-0.17 plugin, a hand-rolled script) without
    access-log archaeology. The value itself is not logged.
    """
    if not surface or site in _IGNORED_SURFACE_SITES:
        return
    _IGNORED_SURFACE_SITES.add(site)
    logger.warning(
        "%s: ignoring the retired 'surface' param; a caller's scope comes from its "
        "device token or OAuth identity only (enroll the machine with `synapse login`)",
        site,
    )
