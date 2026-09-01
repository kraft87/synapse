"""Surface-registration helpers for tests (schema 053, audience scoping).

Serving is fail-closed: a call that carries no ``surface``, or one naming a host with
no ``surfaces`` row, is RESTRICTED with an empty project allowlist and sees nothing.
That is correct behaviour, and it means any test that wants to exercise ordinary
full-corpus serving has to say so — hence :data:`FULL_SURFACE` and :func:`register_full`.

Use :func:`register_full` (or the ``full_surface`` fixture in conftest) for tests about
something OTHER than audience scoping, and :func:`register_restricted` for the ones that
are. Tests deliberately probing the fail-closed path just pass no surface at all.
"""

from __future__ import annotations

import psycopg

#: A surface registered trust='full' — the "this host sees everything" test identity.
FULL_SURFACE = "test-full-surface"

#: A surface registered trust='restricted' — allowlist supplied per test.
RESTRICTED_SURFACE = "test-restricted-surface"


def _upsert(
    conn: psycopg.Connection,
    surface_id: str,
    trust: str,
    projects: list[str],
    *,
    status: str = "approved",
    token_hash: str | None = None,
) -> str:
    """Write one surfaces row. ``status='approved'`` by default — schema 054's column
    default is the fail-closed 'revoked', so a helper that omitted it would silently
    produce rows that serve nothing and every unrelated test would go mysteriously
    empty."""
    conn.execute(
        "INSERT INTO surfaces (surface_id, trust, allowed_projects, status, token_hash) "
        "VALUES (%s, %s, %s, %s, %s) "
        "ON CONFLICT (surface_id) DO UPDATE SET trust = EXCLUDED.trust, "
        "  allowed_projects = EXCLUDED.allowed_projects, status = EXCLUDED.status, "
        "  token_hash = EXCLUDED.token_hash, updated_at = now()",
        (surface_id, trust, projects, status, token_hash),
    )
    return surface_id


def register_device(
    conn: psycopg.Connection,
    token: str,
    *,
    trust: str = "full",
    projects: list[str] | None = None,
    status: str = "approved",
    surface_id: str | None = None,
    label: str | None = None,
) -> str:
    """Register a CREDENTIAL-bound surface (schema 054) and return its id.

    The token is stored hashed, exactly as the routes store it, so tests authenticate
    with the plaintext the same way a real client does.
    """
    from ingestion.surfaces import token_hash as _hash

    sid = surface_id or f"dev-test-{token[:8]}"
    conn.execute(
        "INSERT INTO surfaces (surface_id, trust, allowed_projects, status, token_hash, label) "
        "VALUES (%s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (surface_id) DO UPDATE SET trust = EXCLUDED.trust, "
        "  allowed_projects = EXCLUDED.allowed_projects, status = EXCLUDED.status, "
        "  token_hash = EXCLUDED.token_hash, label = EXCLUDED.label, updated_at = now()",
        (sid, trust, projects or [], status, _hash(token), label),
    )
    return sid


def register_full(conn: psycopg.Connection, surface_id: str = FULL_SURFACE) -> str:
    """Register a full-trust surface and return its id, ready to pass as ``surface=``."""
    return _upsert(conn, surface_id, "full", [])


def register_restricted(
    conn: psycopg.Connection,
    allowed_projects: list[str],
    surface_id: str = RESTRICTED_SURFACE,
) -> str:
    """Register a restricted surface with an explicit project allowlist."""
    return _upsert(conn, surface_id, "restricted", allowed_projects)


def clear_surfaces(conn: psycopg.Connection) -> None:
    """Drop every registration — back to "no host is trusted", the default state."""
    conn.execute("DELETE FROM surfaces")
