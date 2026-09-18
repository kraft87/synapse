"""Resolve the trust scope attached to an MCP or plain-HTTP caller.

This module deliberately knows nothing about server configuration or FastMCP globals.
The server passes request state and configuration in, which keeps credential precedence
testable without booting the MCP application.
"""

from __future__ import annotations

import hmac
from collections.abc import Callable
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
    )


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
    claims = claims_of(access_token)
    if claims.get("kind") == KIND_DEVICE:
        return trust_from_claims(claims)
    if access_token is not None and access_token.client_id not in machine_client_ids:
        identity = claims_identity(access_token.claims or {}, identity_claims)
        if identity:
            return resolve_caller(db_url, legacy_surface_id=f"{OAUTH_SURFACE_PREFIX}{identity}")
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
