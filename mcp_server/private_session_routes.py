"""Private-mode toggle routes — the server seam for `private_mode.py on|off`.

Private mode takes a Claude Code session off the record. The toggle CLI writes a local
marker file (the Stop hook then posts nothing for that session) AND flips the durable
flag here, because the marker is host-local and expires while the transcript on disk
does not — see ``ingestion.private_sessions`` for the enforcement side.

The plugin is a THIN CLIENT with no DSN (Postgres lives behind the server, often on
another host), so the flag is written over the same client-gate seam as the skills /
config / timeline routes.

Routes (client-gate):
  PUT    /private-sessions/{session_id}[?purge=1]
         -> {"status": "ok", "private": true, "stored_turns": N[, "purged": {...}]}
  DELETE /private-sessions/{session_id}  -> {"status": "ok", "private": false, "deleted": N}
  GET    /private-sessions/{session_id}  -> {"status": "ok", "private": bool}

PUT is idempotent (ON CONFLICT DO NOTHING) so a retried toggle is a no-op. GET exists so
the CLI can READ BACK what it wrote and refuse to tell the user "off the record" on an
unverified write. DELETE is the deliberate un-private escape hatch: it re-exposes that
session to a future backfill, which is why the CLI only sends it behind an explicit flag
and never on session-end cleanup.

The flag covers the session FROM NOW ON. Turns ingested before it was set are already
memory, so PUT reports how many are stored (``stored_turns``) and deletes them only when
asked by name: ``?purge=1`` runs the dashboard's episode-delete cascade over every stored
turn of the session, plus the preferences those turns gave birth to and every web page or
archived thread captured in the session (a sub-agent's included), in one transaction.
That is wider than the dashboard's single-turn delete, which leaves preferences alone
because a preference's ``source_ref`` names only its first turn: a privacy purge errs
toward removing what the session said, and a preference the user states again elsewhere
comes back on its own. Notes are left alone; ``remember()`` is an explicit act.

Deleting memory is not a client-gate act elsewhere (the dashboard's deletes need the admin
gate), and the root enrollment token sits on every machine that ever ran the plugin, so a
purge is allowed only to a FULL-trust device, or to a restricted device purging a session
whose every stored turn carries that device's own provenance stamp (schema 057). Anyone
else gets 403; the flag, which is committed first, stays set and nothing is deleted.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Json
from starlette.requests import Request

from ingestion import pg_pool
from ingestion.surfaces import OWN_SURFACE_PROBE, SurfaceTrust
from mcp_server.dash.curation import delete_episode_content, retire_flags
from mcp_server.http_helpers import RouteError, route

logger = logging.getLogger(__name__)

_MAX_ID_LEN = 200


#: The cascade counts a session purge reports. A chunk's "mixed" count is left out: a
#: session's chunk windows only span that session's own turns.
_PURGE_COUNTS = (
    "chunks_deleted",
    "facts_deleted",
    "facts_unlinked",
    "events_deleted",
    "web_deleted",
)


class PurgeRefused(RouteError):
    """The caller's trust does not cover deleting this session's turns (the route's 403)."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail, 403)


def _mark_private(db_url: str, session_id: str, purge_as: SurfaceTrust | None) -> dict[str, Any]:
    """Flag the session, purge what it already stored for ``purge_as`` (None: don't),
    report what is left.

    The flag commits first and on its own, so a purge that fails or is refused still leaves
    the session private, and every ingest that starts after the flag is refused rather than
    racing the purge. ``stored_turns`` is read last: it is what is still stored when PUT
    returns."""
    with pg_pool.connection(db_url) as conn:
        conn.execute(
            "INSERT INTO private_sessions (session_id) VALUES (%s) ON CONFLICT DO NOTHING",
            (session_id,),
        )
    out: dict[str, Any] = {}
    if purge_as is not None:
        out["purged"] = _purge_session(db_url, session_id, purge_as)
    with pg_pool.connection(db_url) as conn:
        row = conn.execute(
            "SELECT count(*) FROM episodes WHERE session_id = %s", (session_id,)
        ).fetchone()
    out["stored_turns"] = int(row[0]) if row else 0
    return out


def _may_purge(conn: Any, db_url: str, session_id: str, trust: SurfaceTrust) -> bool:
    """Whether ``trust`` may delete ``session_id``'s stored turns (module docstring).

    Full trust always may. A restricted device may only when schema 057 is applied and
    no stored turn of the session lacks its own stamp: a turn another surface wrote, or
    one stamped before 057, is not its to delete. Unknown and root-token callers carry
    no stamp, so they never may."""
    if trust.known and not trust.restricted:
        return True
    own = trust.stamp_surface_id
    if own is None or not OWN_SURFACE_PROBE.ready(conn, db_url):
        return False
    foreign = conn.execute(
        "SELECT 1 FROM episodes WHERE session_id = %s AND surface_id IS DISTINCT FROM %s LIMIT 1",
        (session_id, own),
    ).fetchone()
    return foreign is None


def _purge_session(db_url: str, session_id: str, trust: SurfaceTrust) -> dict[str, int]:
    """Delete every stored turn of ``session_id`` and what was derived from it, in one
    transaction, and audit it (``dashboard_audit``, like the dashboard's own deletes,
    naming the surface that asked). Raises :class:`PurgeRefused` before deleting anything
    when ``trust`` does not cover the session."""
    with pg_pool.connection(db_url, autocommit=False, row_factory=dict_row) as conn:
        turns = [
            r["id"]
            for r in conn.execute(
                "SELECT id FROM episodes WHERE session_id = %s ORDER BY id FOR UPDATE",
                (session_id,),
            ).fetchall()
        ]
        if not _may_purge(conn, db_url, session_id, trust):
            raise PurgeRefused(
                "purge refused: deleting a session's stored turns needs a full-trust "
                "device, or a restricted one purging only turns it wrote itself; the "
                "session is flagged private and nothing was deleted"
            )
        counts = delete_episode_content(conn, turns, session_id)
        prefs = [
            r["id"]
            for r in conn.execute(
                "SELECT id FROM preferences WHERE source_ref = ANY(%s) FOR UPDATE",
                ([f"ep:{episode_id}" for episode_id in turns],),
            ).fetchall()
        ]
        if prefs:
            # An older preference these superseded stays retired; only its pointer goes.
            conn.execute(
                "UPDATE preferences SET superseded_by = NULL WHERE superseded_by = ANY(%s)",
                (prefs,),
            )
            retire_flags(conn, "preference", [str(p) for p in prefs])
            conn.execute("DELETE FROM preferences WHERE id = ANY(%s)", (prefs,))
        summary = {
            "episodes_deleted": len(turns),
            **{k: counts[k] for k in _PURGE_COUNTS},
            "preferences_deleted": len(prefs),
        }
        conn.execute(
            "INSERT INTO dashboard_audit (action, kind, item_id, detail) "
            "VALUES ('purge_private_session', 'session', %s, %s)",
            (session_id, Json({**summary, "surface_id": trust.surface_id})),
        )
    logger.info("private session %s purged by %s: %s", session_id, trust.surface_id, summary)
    return summary


def _unmark_private(db_url: str, session_id: str) -> int:
    with pg_pool.connection(db_url) as conn:
        cur = conn.execute("DELETE FROM private_sessions WHERE session_id = %s", (session_id,))
        return cur.rowcount


def _is_private(db_url: str, session_id: str) -> bool:
    with pg_pool.connection(db_url) as conn:
        row = conn.execute(
            "SELECT 1 FROM private_sessions WHERE session_id = %s", (session_id,)
        ).fetchone()
        return row is not None


def register(
    mcp: Any,
    db_url: str,
    authorized: Callable[[Request], bool],
    resolve_trust: Callable[[Request], SurfaceTrust],
) -> None:
    """``resolve_trust`` maps a request to its bearer-resolved verdict (schema 054); only
    ``?purge=1`` consults it."""
    if not db_url:
        logger.info("private-session routes disabled (no DB_URL)")
        return

    def _session_id(request: Request) -> str:
        session_id = (request.path_params.get("session_id") or "").strip()
        if not session_id or len(session_id) > _MAX_ID_LEN:
            raise RouteError("session_id required")
        return session_id

    @mcp.custom_route("/private-sessions/{session_id}", methods=["PUT"])  # type: ignore[misc]
    @route(authorized)
    def private_session_put(request: Request) -> dict[str, Any]:
        """Take a session off the record from now on. Idempotent; the row is permanent by
        design. ``?purge=1`` also deletes the turns it already stored, for a caller whose
        trust covers them (module docstring)."""
        purge = request.query_params.get("purge", "0")
        if purge not in ("0", "1"):
            raise RouteError("purge must be 0 or 1")
        session_id = _session_id(request)
        out = _mark_private(db_url, session_id, resolve_trust(request) if purge == "1" else None)
        return {"status": "ok", "session_id": session_id, "private": True, **out}

    @mcp.custom_route("/private-sessions/{session_id}", methods=["DELETE"])  # type: ignore[misc]
    @route(authorized)
    def private_session_delete(request: Request) -> dict[str, Any]:
        """Un-private a session. Re-exposes its on-disk transcript to future backfills."""
        session_id = _session_id(request)
        return {
            "status": "ok",
            "session_id": session_id,
            "private": False,
            "deleted": int(_unmark_private(db_url, session_id)),
        }

    @mcp.custom_route("/private-sessions/{session_id}", methods=["GET"])  # type: ignore[misc]
    @route(authorized)
    def private_session_get(request: Request) -> dict[str, Any]:
        """Read back the flag — how the CLI verifies its write actually landed."""
        session_id = _session_id(request)
        return {
            "status": "ok",
            "session_id": session_id,
            "private": bool(_is_private(db_url, session_id)),
        }
