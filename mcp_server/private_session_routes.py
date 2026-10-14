"""Private-mode toggle routes — the server seam for `private_mode.py on|off`.

Private mode takes a Claude Code session off the record. The toggle CLI writes a local
marker file (the Stop hook then posts nothing for that session) AND flips the durable
flag here, because the marker is host-local and expires while the transcript on disk
does not — see ``ingestion.private_sessions`` for the enforcement side.

The plugin is a THIN CLIENT with no DSN (Postgres lives behind the server, often on
another host), so the flag is written over the same machine-token seam as the skills /
config / timeline routes.

Routes (machine-token gated):
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
turn of the session, plus the preferences those turns gave birth to, in one transaction.
That is wider than the dashboard's single-turn delete, which leaves preferences alone
because a preference's ``source_ref`` names only its first turn: a privacy purge errs
toward removing what the session said, and a preference the user states again elsewhere
comes back on its own. Notes are left alone; ``remember()`` is an explicit act.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse

from ingestion import pg_pool
from mcp_server.dashboard_routes import delete_episode_content
from mcp_server.http_helpers import err, unauthorized

logger = logging.getLogger(__name__)

_MAX_ID_LEN = 200


class _SchemaMissing(Exception):
    """schema/050 not applied on this deployment — reported as 503, never as success."""


#: The cascade counts a session purge reports, summed over its turns. A chunk's "mixed"
#: count is left out: a session's chunk windows only span that session's own turns.
_PURGE_COUNTS = ("chunks_deleted", "facts_deleted", "facts_unlinked", "events_deleted")


def _mark_private(db_url: str, session_id: str, purge: bool) -> dict[str, Any]:
    """Flag the session, purge what it already stored when asked, report what is left.

    The flag commits first and on its own, so a purge that fails still leaves the session
    private, and every ingest that starts after the flag is refused rather than racing the
    purge. ``stored_turns`` is read last: it is what is still stored when PUT returns."""
    try:
        with pg_pool.connection(db_url) as conn:
            conn.execute(
                "INSERT INTO private_sessions (session_id) VALUES (%s) ON CONFLICT DO NOTHING",
                (session_id,),
            )
    except psycopg.errors.UndefinedTable as e:
        raise _SchemaMissing from e
    out: dict[str, Any] = {}
    if purge:
        out["purged"] = _purge_session(db_url, session_id)
    with pg_pool.connection(db_url) as conn:
        row = conn.execute(
            "SELECT count(*) FROM episodes WHERE session_id = %s", (session_id,)
        ).fetchone()
    out["stored_turns"] = int(row[0]) if row else 0
    return out


def _purge_session(db_url: str, session_id: str) -> dict[str, int]:
    """Delete every stored turn of ``session_id`` and what was derived from it, in one
    transaction, and audit it (``dashboard_audit``, like the dashboard's own deletes)."""
    with pg_pool.connection(db_url, autocommit=False, row_factory=dict_row) as conn:
        turns = [
            r["id"]
            for r in conn.execute(
                "SELECT id FROM episodes WHERE session_id = %s ORDER BY id FOR UPDATE",
                (session_id,),
            ).fetchall()
        ]
        totals = dict.fromkeys(_PURGE_COUNTS, 0)
        for episode_id in turns:
            counts = delete_episode_content(conn, episode_id)
            for key in _PURGE_COUNTS:
                totals[key] += counts[key]
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
            conn.execute(
                "UPDATE dashboard_flags SET removed_at = now() WHERE kind = 'preference' "
                "AND item_id = ANY(%s) AND removed_at IS NULL",
                ([str(p) for p in prefs],),
            )
            conn.execute("DELETE FROM preferences WHERE id = ANY(%s)", (prefs,))
        summary = {"episodes_deleted": len(turns), **totals, "preferences_deleted": len(prefs)}
        conn.execute(
            "INSERT INTO dashboard_audit (action, kind, item_id, detail) "
            "VALUES ('purge_private_session', 'session', %s, %s)",
            (session_id, Json(summary)),
        )
    logger.info("private session %s purged: %s", session_id, summary)
    return summary


def _unmark_private(db_url: str, session_id: str) -> int:
    try:
        with pg_pool.connection(db_url) as conn:
            cur = conn.execute("DELETE FROM private_sessions WHERE session_id = %s", (session_id,))
            return cur.rowcount
    except psycopg.errors.UndefinedTable as e:
        raise _SchemaMissing from e


def _is_private(db_url: str, session_id: str) -> bool:
    try:
        with pg_pool.connection(db_url) as conn:
            row = conn.execute(
                "SELECT 1 FROM private_sessions WHERE session_id = %s", (session_id,)
            ).fetchone()
            return row is not None
    except psycopg.errors.UndefinedTable as e:
        raise _SchemaMissing from e


def register(mcp: Any, db_url: str, authorized: Callable[[Request], bool]) -> None:
    if not db_url:
        logger.info("private-session routes disabled (no DB_URL)")
        return

    def _session_id(request: Request) -> str:
        return (request.path_params.get("session_id") or "").strip()

    async def _run(request: Request, work: Callable[[str], Any]) -> JSONResponse | Any:
        """Auth + session-id validation + threadpool, shared by the three verbs.

        Returns a JSONResponse on any failure, or the raw work() result on success —
        callers wrap that in their own envelope."""
        if not authorized(request):
            return unauthorized()
        session_id = _session_id(request)
        if not session_id or len(session_id) > _MAX_ID_LEN:
            return err("session_id required", 400)
        try:
            return await run_in_threadpool(work, session_id)
        except _SchemaMissing:
            return err("private_sessions missing (apply schema/050)", 503)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("private-session route failed: %s", e)
            return err(str(e)[:200], 500)

    @mcp.custom_route("/private-sessions/{session_id}", methods=["PUT"])  # type: ignore[misc]
    async def private_session_put(request: Request) -> JSONResponse:
        """Take a session off the record from now on. Idempotent; the row is permanent by
        design. ``?purge=1`` also deletes the turns it already stored (module docstring)."""
        purge = request.query_params.get("purge", "0")
        if purge not in ("0", "1"):
            return err("purge must be 0 or 1", 400)
        out = await _run(request, lambda sid: _mark_private(db_url, sid, purge == "1"))
        if isinstance(out, JSONResponse):
            return out
        return JSONResponse(
            {"status": "ok", "session_id": _session_id(request), "private": True, **out}
        )

    @mcp.custom_route("/private-sessions/{session_id}", methods=["DELETE"])  # type: ignore[misc]
    async def private_session_delete(request: Request) -> JSONResponse:
        """Un-private a session. Re-exposes its on-disk transcript to future backfills."""
        out = await _run(request, lambda sid: _unmark_private(db_url, sid))
        if isinstance(out, JSONResponse):
            return out
        return JSONResponse(
            {
                "status": "ok",
                "session_id": _session_id(request),
                "private": False,
                "deleted": int(out),
            }
        )

    @mcp.custom_route("/private-sessions/{session_id}", methods=["GET"])  # type: ignore[misc]
    async def private_session_get(request: Request) -> JSONResponse:
        """Read back the flag — how the CLI verifies its write actually landed."""
        out = await _run(request, lambda sid: _is_private(db_url, sid))
        if isinstance(out, JSONResponse):
            return out
        return JSONResponse(
            {"status": "ok", "session_id": _session_id(request), "private": bool(out)}
        )
