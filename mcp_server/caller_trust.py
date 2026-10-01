"""Resolve the trust scope attached to an MCP or plain-HTTP caller.

This module deliberately knows nothing about server configuration or FastMCP globals.
The server passes request state and configuration in, which keeps credential precedence
testable without booting the MCP application.
"""

from __future__ import annotations

import hmac
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import Any

from ingestion.surfaces import SurfaceTrust, resolve_caller, token_hash
from mcp_server.auth_tokens import KIND_DEVICE, claims_of

OAUTH_SURFACE_PREFIX = "oauth:"


def trust_from_claims(claims: dict[str, Any]) -> SurfaceTrust:
    """Turn a verified device credential's claims into its trust verdict."""
    return SurfaceTrust(
        surface_id=claims.get("surface_id"),
        trust=str(claims.get("trust") or "restricted"),
        allowed_projects=tuple(claims.get("allowed_projects") or ()),
        known=True,
        credential_bound=True,
    )


#: A verdict an HTTP route already resolved from its request's bearer, handed to code it
#: calls that would otherwise re-resolve from FastMCP's token context (which custom
#: routes do not have). The /remember/spool replay is the one user: it calls the
#: remember tool, and without this the device would fall back to the self-reported id
#: lane and lose its credential binding (schema 055 provenance).
_ROUTE_TRUST: ContextVar[SurfaceTrust | None] = ContextVar("synapse_route_trust", default=None)


@contextmanager
def route_trust(st: SurfaceTrust | None) -> Iterator[None]:
    """Bind ``st`` as the caller verdict for the duration of the block.

    Only a credential-bound verdict is bound; anything else leaves resolution exactly as
    it was, so this can never widen what an unknown caller is treated as.
    """
    if st is None or not st.credential_bound:
        yield
        return
    tok = _ROUTE_TRUST.set(st)
    try:
        yield
    finally:
        _ROUTE_TRUST.reset(tok)


def caller_trust(
    *,
    db_url: str,
    surface: str | None,
    access_token: Any,
    identity_claims: tuple[str, ...],
    machine_client_ids: set[str],
    claims_identity: Callable[[dict[str, Any], tuple[str, ...]], str],
) -> SurfaceTrust:
    """Resolve the trust for an MCP call, with credential identity taking precedence.

    Device claims are already authenticated by FastMCP and therefore need no second
    database read. OAuth identities map to a server-derived ``oauth:<identity>``
    surface. The legacy surface parameter is considered only for the root-token lane.
    """
    bound = _ROUTE_TRUST.get()
    if bound is not None:
        return bound
    claims = claims_of(access_token)
    if claims.get("kind") == KIND_DEVICE:
        return trust_from_claims(claims)
    if access_token is not None and access_token.client_id not in machine_client_ids:
        identity = claims_identity(access_token.claims or {}, identity_claims)
        if identity:
            st = resolve_caller(db_url, legacy_surface_id=f"{OAUTH_SURFACE_PREFIX}{identity}")
            # The id was derived by the server from a VERIFIED identity, so it is as
            # credential-bound as a device token. Only a matched, approved row counts.
            return replace(st, credential_bound=True) if st.known else st
    return resolve_caller(db_url, legacy_surface_id=surface)


def request_trust(
    *,
    db_url: str,
    machine_token: str,
    bearer: str,
    surface: str | None,
) -> SurfaceTrust:
    """Resolve trust for a custom HTTP route, which has no FastMCP token context."""
    if machine_token and bearer and not hmac.compare_digest(bearer, machine_token):
        return resolve_caller(db_url, token_hash_hex=token_hash(bearer))
    return resolve_caller(db_url, legacy_surface_id=surface)
