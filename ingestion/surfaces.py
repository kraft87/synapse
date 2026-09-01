"""Per-host trust lookup + the audience derivation rule (schema 053).

The serving-side half of audience scoping. Every serving path (board, recall,
recall_full_turns, fetch, fetch_session, plain-HTTP /recall) resolves the calling
surface through :func:`lookup_surface` and filters what it serves accordingly; the
write path (remember → reconcile_note) resolves the audience tag through
:func:`derive_audience`.

A surface id is usually a HOST (the plugin's ``SYNAPSE_SURFACE`` / hostname constant),
but not always: MCP callers on the OAuth/OIDC lane run no hook and send no id, so
``mcp_server/server._caller_surface`` derives ``oauth:<login>`` from their verified
identity instead. Both kinds resolve through the one lookup below and both are equally
fail-closed — an unregistered ``oauth:<login>`` is as restricted as an unknown host.

Fail-closed is the whole point, so it is concentrated in ONE place: every failure mode
of :func:`lookup_surface` — no surface id, no row, missing table, unreachable database,
a malformed row — returns :data:`UNKNOWN_SURFACE`, which is ``restricted`` with an empty
allowlist. There is no code path that turns an error into ``full``. Callers must not
"handle" a lookup failure by skipping the filter; they get a restricted verdict and
serve the empty intersection.

``known`` distinguishes "a surfaces row said restricted" from "we could not tell". Both
restrict SERVING identically. They differ on the WRITE side only: rule 2 of the audience
precedence (a remember() from a restricted surface defaults ``work-safe``) fires on a
REGISTERED restricted surface, never on an unknown one — defaulting an unclassified note
to ``work-safe`` because a hostname went unrecognised would invert the fail-closed
posture into a leak.

Lookups are uncached on purpose. A cache would serve a stale ``full`` verdict after a
surface is demoted, which is the one staleness this system cannot afford; the read is a
primary-key hit on a table with a handful of rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)

AUDIENCES = ("personal", "work-safe")
TRUST_LEVELS = ("full", "restricted")

#: The default audience for an unclassified note — never leaves a trusted host.
DEFAULT_AUDIENCE = "personal"

#: What a restricted surface is allowed to read from the notes store.
RESTRICTED_AUDIENCE = "work-safe"


@dataclass(frozen=True)
class SurfaceTrust:
    """The resolved trust verdict for one calling surface.

    ``allowed_projects`` is meaningful only when :attr:`restricted` — a full-trust
    surface serves everything and never consults it.
    """

    surface_id: str | None = None
    trust: str = "restricted"
    allowed_projects: tuple[str, ...] = ()
    known: bool = False

    @property
    def restricted(self) -> bool:
        """True unless a surfaces row explicitly granted full trust."""
        return self.trust != "full"

    @property
    def project_filter(self) -> list[str] | None:
        """Project allowlist to apply to episode/timeline reads, or None for no filter.

        A restricted surface always gets a LIST — empty for an unknown surface, which
        makes ``project = ANY('{}')`` false for every row (NULL project included). That
        is the intended fail-closed serve: nothing, rather than everything.
        """
        return None if not self.restricted else list(self.allowed_projects)

    @property
    def audience_filter(self) -> str | None:
        """Notes audience to filter on, or None for no filter (full trust)."""
        return None if not self.restricted else RESTRICTED_AUDIENCE


#: The verdict for "no surface id", "no such surface", or any lookup failure.
UNKNOWN_SURFACE = SurfaceTrust()

#: Convenience for callers that already hold a full-trust decision (tests, CLI tools).
FULL_TRUST = SurfaceTrust(trust="full", known=True)


def _row_to_trust(surface_id: str, row: Any) -> SurfaceTrust:
    trust = str(row["trust"] or "restricted")
    if trust not in TRUST_LEVELS:  # pragma: no cover - CHECK constraint prevents it
        logger.warning("surface %r has unknown trust %r; treating as restricted", surface_id, trust)
        trust = "restricted"
    projects = tuple(str(p) for p in (row["allowed_projects"] or ()))
    return SurfaceTrust(surface_id=surface_id, trust=trust, allowed_projects=projects, known=True)


def lookup_surface(db_url: str, surface_id: str | None) -> SurfaceTrust:
    """Resolve one surface id to its trust verdict. NEVER raises, never fails open.

    Missing id, missing row, missing table (a deployment behind schema/053), or an
    unreachable database all yield :data:`UNKNOWN_SURFACE` — restricted, empty allowlist.
    """
    sid = (surface_id or "").strip()
    if not sid or not db_url:
        return UNKNOWN_SURFACE
    try:
        conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
        try:
            row = conn.execute(
                "SELECT trust, allowed_projects FROM surfaces WHERE surface_id = %s", (sid,)
            ).fetchone()
        finally:
            conn.close()
    except Exception as e:
        # Fail CLOSED and say so: an operator staring at an empty board needs the reason.
        logger.warning("surface lookup failed for %r (%s); treating as restricted", sid, e)
        return UNKNOWN_SURFACE
    if row is None:
        return SurfaceTrust(surface_id=sid)
    return _row_to_trust(sid, row)


# ---------------------------------------------------------------------------
# Write side: audience derivation
# ---------------------------------------------------------------------------


def restricted_project_union(db: Any) -> set[str]:
    """Every project any REGISTERED restricted surface is allowed to read.

    This is the provenance rule behind audience precedence #3: a note filed under a
    project that some work host can already read episodes from is work-safe by
    construction, so the work board keeps receiving new notes without anyone inventing
    a second classification step. Fail-closed (empty set) on any error — the caller then
    falls through to ``personal``.
    """
    try:
        return set(db.restricted_surface_projects())
    except Exception as e:
        logger.warning("restricted-project union failed (%s); deriving audience as personal", e)
        return set()


def derive_audience(
    db: Any,
    *,
    explicit: str | None,
    caller_restricted: bool,
    project: str | None,
) -> str:
    """The audience tag for a note being written, in spec precedence order.

    1. ``explicit`` — an ``audience`` argument on remember() wins outright.
    2. ``caller_restricted`` — a write from a REGISTERED restricted surface defaults
       ``work-safe``, symmetric with what that host is allowed to read (otherwise notes
       written at work vanish from the work board on the next session).
    3. ``project`` in the union of restricted surfaces' allowlists ⇒ ``work-safe``.
    4. Otherwise ``personal``.

    Note that an UNKNOWN surface must not set ``caller_restricted``: unknown restricts
    reads, but it must never widen a write.
    """
    if explicit:
        if explicit not in AUDIENCES:
            raise ValueError(f"invalid audience {explicit!r} — expected one of {AUDIENCES}")
        return explicit
    if caller_restricted:
        return RESTRICTED_AUDIENCE
    if project and project in restricted_project_union(db):
        return RESTRICTED_AUDIENCE
    return DEFAULT_AUDIENCE


# ---------------------------------------------------------------------------
# Registration CRUD (the machine-token routes in mcp_server/surface_routes.py)
# ---------------------------------------------------------------------------


class SchemaMissing(Exception):
    """schema/053 not applied on this deployment — reported as 503, never as success."""


def list_surfaces(db_url: str) -> list[dict[str, Any]]:
    """Every registered surface, id order. Operator read — not a serving path."""
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    try:
        rows = conn.execute(
            "SELECT surface_id, trust, allowed_projects, created_at, updated_at "
            "FROM surfaces ORDER BY surface_id"
        ).fetchall()
    except psycopg.errors.UndefinedTable as e:
        raise SchemaMissing from e
    finally:
        conn.close()
    return [
        {
            "surface_id": r["surface_id"],
            "trust": r["trust"],
            "allowed_projects": list(r["allowed_projects"] or []),
            "created_at": r["created_at"].isoformat(),
            "updated_at": r["updated_at"].isoformat(),
        }
        for r in rows
    ]


def upsert_surface(
    db_url: str, surface_id: str, trust: str, allowed_projects: list[str]
) -> dict[str, Any]:
    """Register or re-register one surface. Full replacement of trust + allowlist.

    Deliberately not a PATCH: a partial update of a security allowlist is how a stale
    entry survives a demotion. The caller states the whole intended state every time.
    """
    if trust not in TRUST_LEVELS:
        raise ValueError(f"invalid trust {trust!r} — expected one of {TRUST_LEVELS}")
    projects = [str(p).strip() for p in allowed_projects if str(p).strip()]
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    try:
        row = conn.execute(
            "INSERT INTO surfaces (surface_id, trust, allowed_projects) VALUES (%s, %s, %s) "
            "ON CONFLICT (surface_id) DO UPDATE SET trust = EXCLUDED.trust, "
            "  allowed_projects = EXCLUDED.allowed_projects, updated_at = now() "
            "RETURNING surface_id, trust, allowed_projects, created_at, updated_at",
            (surface_id, trust, projects),
        ).fetchone()
    except psycopg.errors.UndefinedTable as e:
        raise SchemaMissing from e
    finally:
        conn.close()
    assert row is not None, "INSERT ... RETURNING returned nothing"
    return {
        "surface_id": row["surface_id"],
        "trust": row["trust"],
        "allowed_projects": list(row["allowed_projects"] or []),
        "created_at": row["created_at"].isoformat(),
        "updated_at": row["updated_at"].isoformat(),
    }


def delete_surface(db_url: str, surface_id: str) -> int:
    """Unregister a surface. It reverts to restricted/empty — a safe direction."""
    conn = psycopg.connect(db_url, autocommit=True)
    try:
        cur = conn.execute("DELETE FROM surfaces WHERE surface_id = %s", (surface_id,))
        return cur.rowcount
    except psycopg.errors.UndefinedTable as e:
        raise SchemaMissing from e
    finally:
        conn.close()
