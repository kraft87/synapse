"""The board — a small always-injected index of explicit memories (schema 041).

Converts recall into recognition: instead of hoping the agent thinks to search, the
board puts one bounded text block in front of it — curated note hooks (rules/feedback,
user facts, project state, references), the last week's milestones, and a banner saying
what memory exists at all. Bodies stay behind ids; absence from the board means SEARCH
(recall), not doesn't-exist.

Served two ways, same block (mirrors preferences_routes: the plugin hook is a THIN
CLIENT with no DSN, the server owns the DB):
  - MCP tool ``get_context(project)`` in server.py
  - GET /context?project=X — machine-token auth (custom routes bypass FastMCP's auth
    middleware by design), PG work in a threadpool, fail-soft JSON.

Pure SQL, no embedding calls, target <100ms: the note list (ingestion/db.py's
list_board_notes), the timeline milestones (timeline_routes._recent_events — reused,
not duplicated), and two cheap banner aggregates. Each helper owns its own short-lived
connection; the board adds exactly one more for the banner.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from typing import Any

import psycopg
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse

from ingestion.db import Database
from mcp_server.timeline_routes import _recent_events

logger = logging.getLogger(__name__)

# Single-owner constant, mirroring preferences_routes/notes — one env axis.
_OWNER = os.environ.get("SYNAPSE_KG_OWNER_ID", "default")

# HARD CAP on the rendered block: whichever hits first. The board must stay a cheap
# always-injected index, never a context tax. est tokens = chars // 4.
_MAX_LINES = 80
_MAX_EST_TOKENS = 2000

# Timeline facts are hook-length lines on the board, never full prose: POST
# /timeline/events accepts unbounded fact text, and the cap loop can only drop NOTES —
# without this clamp one verbose feeder would both bust the hard cap and evict every
# curated note for zero benefit. 10 events x ~200 chars stays well under the caps.
_EVENT_FACT_MAX = 200

# Overflow drop priority (lower = dropped first): project + reference notes are
# stale-managed here, user notes go next, feedback/rules are NEVER dropped before
# the others — a standing correction is the board's whole point.
_DROP_CLASS = {"project": 0, "reference": 0, "user": 1, "feedback": 2}

_SECTION_TITLES = {
    "feedback": "## Rules & feedback",
    "user": "## User",
    "reference": "## References",
}


def _banner_stats(db_url: str) -> tuple[int, list[str]]:
    """(total episodes, project names newest-activity first, capped at 12).

    If count(*) ever gets slow at scale, switch to pg_class.reltuples (approximate
    but O(1)); at current volumes the exact count is milliseconds.
    """
    conn = psycopg.connect(db_url, autocommit=True)
    try:
        row = conn.execute("SELECT count(*) FROM episodes").fetchone()
        n_episodes = int(row[0]) if row else 0
        # Unlabeled episodes (NULL — and '', defensively) are excluded in SQL so they
        # never consume one of the 12 LIMIT slots or inflate/deflate the project count.
        rows = conn.execute(
            "SELECT project, max(created_at)::date FROM episodes "
            "WHERE project IS NOT NULL AND project <> '' "
            "GROUP BY project ORDER BY 2 DESC LIMIT 12"
        ).fetchall()
        return n_episodes, [r[0] for r in rows]
    finally:
        conn.close()


def _note_line(note: dict[str, Any]) -> str:
    upd = note["updated_at"].strftime("%m-%d")
    return f"- {note['hook']} (n:{note['id']}, upd {upd})"


def _fits(text: str) -> bool:
    return text.count("\n") + 1 <= _MAX_LINES and len(text) // 4 <= _MAX_EST_TOKENS


def _render(
    project: str | None,
    n_episodes: int,
    project_names: list[str],
    notes: list[dict[str, Any]],
    dropped: int,
    events: list[dict[str, Any]],
) -> str:
    lines = [f"[Synapse board — project: {project or 'all'}]"]
    recent = f" (most recent: {', '.join(project_names)})" if project_names else ""
    lines.append(f"{n_episodes} episodes across {len(project_names)} projects{recent}.")
    lines.append(
        "Absence from this board means SEARCH (recall), not doesn't-exist. "
        "Note bodies: fetch by id."
    )

    by_type: dict[str, list[dict[str, Any]]] = {}
    for n in notes:
        by_type.setdefault(n["type"], []).append(n)
    if notes or dropped:  # `dropped` alone: the overflow line still gets its blank line
        lines.append("")
    for t in ("feedback", "user", "project", "reference"):  # empty sections omitted
        if t not in by_type:
            continue
        lines.append(_SECTION_TITLES.get(t) or f"## Project: {project or 'all'}")
        lines.extend(_note_line(n) for n in by_type[t])
    if dropped:
        lines.append(f"(+ {dropped} more notes behind recall)")

    if events:
        lines.append("")
        lines.append("## Last 7 days")
        for e in events:
            proj = f" ({e['project']})" if e.get("project") else ""
            fact = e["fact"]
            if len(fact) > _EVENT_FACT_MAX:
                fact = fact[: _EVENT_FACT_MAX - 1] + "…"
            lines.append(f"- {str(e['date'])[5:]}{proj}: {fact}")
    return "\n".join(lines)


def build_board(db_url: str, project: str | None) -> dict[str, Any]:
    """Build the rendered board block. Pure SQL, no embedding calls.

    Returns ``{"status": "ok", "text", "n_notes", "overflow", "note_ids"}`` —
    ``note_ids`` is the telemetry envelope's serve list; serve paths pop it before
    returning the block to callers (the ids the caller needs are inline as ``n:ID``).

    Missing tables (a deployment behind migration 033/041) degrade that section to
    empty rather than failing the whole board — same posture as preferences_routes.
    """
    db = Database(db_url)
    try:
        notes = db.list_board_notes(_OWNER, project)
    except psycopg.errors.UndefinedTable:
        notes = []
    finally:
        db.close()

    try:
        events = _recent_events(db_url, days=7, min_salience=2, limit=10, project=None)
    except psycopg.errors.UndefinedTable:
        events = []

    n_episodes, project_names = _banner_stats(db_url)

    # Cap loop: drop one note at a time (drop class, then oldest updated_at) and
    # re-render until under both caps. Dozens of notes at most — O(n^2) is fine.
    # Guard first: if the fixed portion (banner + clamped events) alone busts the caps,
    # no amount of note-dropping can reach them — keep every note rather than draining
    # the board for zero benefit. With event facts clamped this is a residual guard.
    kept = list(notes)
    dropped = 0
    floor = _render(project, n_episodes, project_names, [], len(notes), events)
    can_reach_cap = _fits(floor)
    while True:
        text = _render(project, n_episodes, project_names, kept, dropped, events)
        if not kept or not can_reach_cap or _fits(text):
            break
        victim = min(kept, key=lambda n: (_DROP_CLASS.get(n["type"], 0), n["updated_at"], n["id"]))
        kept.remove(victim)
        dropped += 1

    return {
        "status": "ok",
        "text": text,
        "n_notes": len(kept),
        "overflow": dropped,
        "note_ids": [n["id"] for n in kept],
    }


def record_board_metrics(engine: Any, source: str, ms_total: float, board: dict[str, Any]) -> None:
    """One recall_metrics row (kind='board') per serve, through Recall's fire-and-forget
    writer (record_event). served_ids is the existing free-form JSONB envelope — no new
    DDL. Fail-soft: telemetry must never break a serve."""
    try:
        text = board.get("text") or ""
        engine.record_event(
            "board",
            source=source,
            ms_total=round(ms_total, 2),
            chars=len(text),
            est_tokens=len(text) // 4,
            served_ids={
                "notes": board.get("note_ids") or [],
                "n_notes": board.get("n_notes", 0),
                "overflow": board.get("overflow", 0),
            },
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("board metrics record failed: %s", e)


def register(
    mcp: Any,
    db_url: str,
    authorized: Callable[[Request], bool],
    get_recall: Callable[[], Any] | None = None,
) -> None:
    """Mount GET /context. ``get_recall`` lazily yields the process's Recall engine so
    board serves share its telemetry writer; None (dev/stdio) skips telemetry."""
    if not db_url:
        logger.info("board routes disabled (no DB_URL)")
        return

    @mcp.custom_route("/context", methods=["GET"])  # type: ignore[misc]
    async def board_context(request: Request) -> JSONResponse:
        """The rendered board for the plugin's SessionStart hook (follow-up PR): one
        bounded always-relevant index block, not query-blind recall injection."""
        if not authorized(request):
            return JSONResponse({"status": "error", "detail": "unauthorized"}, status_code=401)
        project = request.query_params.get("project") or None
        t0 = time.perf_counter()
        try:
            board = await run_in_threadpool(build_board, db_url, project)
        except Exception as e:
            logger.warning("board build failed: %s", e)
            return JSONResponse({"status": "error", "detail": str(e)[:200]}, status_code=500)
        if get_recall is not None:
            record_board_metrics(get_recall(), "http", (time.perf_counter() - t0) * 1000.0, board)
        board.pop("note_ids", None)
        return JSONResponse(board)
