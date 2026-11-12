"""Private-mode toggle routes — the server seam for `private_mode.py on|off`.

Private mode takes a Claude Code session off the record. The toggle CLI writes a local
marker file (the Stop hook then posts nothing for that session) AND flips the durable
flag here, because the marker is host-local and expires while the transcript on disk
does not — see ``ingestion.private_sessions`` for the enforcement side.

The plugin is a THIN CLIENT with no DSN (Postgres lives behind the server, often on
another host), so the flag is written over the same machine-token seam as the skills /
config / timeline routes.

Routes:
  PUT    /private-sessions/{session_id}  client  -> {"status": "ok", "private": true}
  GET    /private-sessions/{session_id}  client  -> {"status": "ok", "private": bool}
  DELETE /private-sessions/{session_id}  admin   -> {"status": "ok", "private": false, "deleted": N}

PUT is idempotent (ON CONFLICT DO NOTHING) so a retried toggle is a no-op. GET exists so
the CLI can READ BACK what it wrote and refuse to tell the user "off the record" on an
unverified write. DELETE is the deliberate un-private escape hatch: it re-exposes that
session to a future backfill, which is why the CLI only sends it behind an explicit flag
and never on session-end cleanup. It is behind the admin gate (a full-trust device; the
root token and a restricted device are refused with a 403) because a session one device
took off the record must not be put back on it by a device the owner trusts less.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import psycopg
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse

from ingestion import pg_pool
from mcp_server.http_helpers import err, unauthorized

logger = logging.getLogger(__name__)

_MAX_ID_LEN = 200


class _SchemaMissing(Exception):
    """schema/050 not applied on this deployment — reported as 503, never as success."""


def _mark_private(db_url: str, session_id: str) -> None:
    try:
        with pg_pool.connection(db_url) as conn:
            conn.execute(
                "INSERT INTO private_sessions (session_id) VALUES (%s) ON CONFLICT DO NOTHING",
                (session_id,),
            )
    except psycopg.errors.UndefinedTable as e:
        raise _SchemaMissing from e


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


def register(
    mcp: Any,
    db_url: str,
    authorized: Callable[[Request], bool],
    admin: Callable[[Request], bool],
) -> None:
    if not db_url:
        logger.info("private-session routes disabled (no DB_URL)")
        return

    def _session_id(request: Request) -> str:
        return (request.path_params.get("session_id") or "").strip()

    async def _run(
        request: Request, work: Callable[[str], Any], *, admin_only: bool = False
    ) -> JSONResponse | Any:
        """Auth + session-id validation + threadpool, shared by the three verbs.

        A caller with no valid credential gets a 401. With ``admin_only``, a caller that
        clears the client gate but not the admin gate gets a 403 instead.

        Returns a JSONResponse on any failure, or the raw work() result on success —
        callers wrap that in their own envelope."""
        if not authorized(request):
            return unauthorized()
        if admin_only and not admin(request):
            return err("un-privating a session needs a full-trust device", 403)
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
        """Take a session off the record. Idempotent; the row is permanent by design."""
        out = await _run(request, lambda sid: _mark_private(db_url, sid))
        if isinstance(out, JSONResponse):
            return out
        return JSONResponse({"status": "ok", "session_id": _session_id(request), "private": True})

    @mcp.custom_route("/private-sessions/{session_id}", methods=["DELETE"])  # type: ignore[misc]
    async def private_session_delete(request: Request) -> JSONResponse:
        """Un-private a session. Re-exposes its on-disk transcript to future backfills.

        Admin gate: only a full-trust device may take a session back on the record."""
        out = await _run(request, lambda sid: _unmark_private(db_url, sid), admin_only=True)
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
