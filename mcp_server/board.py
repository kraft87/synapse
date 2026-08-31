"""The board — a small always-injected index of explicit memories (schema 041).

Converts recall into recognition: instead of hoping the agent thinks to search, the
board puts one bounded text block in front of it — curated note hooks (rules/feedback,
user facts, project state, references), the last week's milestones, and a banner saying
what memory exists at all. Bodies stay behind ids; absence from the board means SEARCH
(recall), not doesn't-exist.

Served ONE way: GET /context?project=X — machine-token auth (custom routes bypass
FastMCP's auth middleware by design), PG work in a threadpool, fail-soft JSON. The
plugin's SessionStart hook is the thin client (no DSN, the server owns the DB —
mirrors preferences_routes). There is deliberately no MCP board tool: the hook
already injects the block, and a listed tool invites a double-inject (Hermes
pattern — when injection covers the read, ship no read tool).

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
from ingestion.surfaces import SurfaceTrust, lookup_surface
from mcp_server.http_helpers import err, unauthorized
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


def _banner_stats(db_url: str, allowed_projects: list[str] | None = None) -> tuple[int, list[str]]:
    """(total episodes, project names newest-activity first, capped at 12).

    The episode total comes from pg_class.reltuples (approximate but O(1)) rather
    than count(*): at ~40K rows / 1.5GB the exact count degrades to ~650ms of
    index-only scan on every session start, and a banner headline doesn't need the
    precise figure. reltuples is -1 until the table's first ANALYZE/VACUUM (fresh
    deploy, or a truncated test fixture), so fall back to an exact count while it's
    unpopulated — which also keeps the exact-count board tests green.

    ``allowed_projects`` (a restricted surface, schema 053) narrows BOTH numbers to the
    allowlist: the project names are the leak that matters ("most recent: family-health"
    on a work laptop is exactly the disclosure this feature exists to stop), and a
    corpus-wide episode count next to an allowlisted project list would misdescribe what
    the caller can actually reach. The reltuples shortcut is skipped in that branch —
    it cannot answer a filtered question — but the filtered count is over a small slice.
    """
    conn = psycopg.connect(db_url, autocommit=True)
    try:
        if allowed_projects is not None:
            row = conn.execute(
                "SELECT count(*) FROM episodes WHERE project = ANY(%s)", (allowed_projects,)
            ).fetchone()
            n_episodes = int(row[0]) if row else 0
            rows = conn.execute(
                "SELECT project, max(created_at)::date FROM episodes "
                "WHERE project = ANY(%s) AND project IS NOT NULL AND project <> '' "
                "GROUP BY project ORDER BY 2 DESC LIMIT 12",
                (allowed_projects,),
            ).fetchall()
            return n_episodes, [r[0] for r in rows]
        row = conn.execute(
            "SELECT reltuples::bigint FROM pg_class WHERE oid = 'episodes'::regclass"
        ).fetchone()
        n_episodes = int(row[0]) if row and row[0] is not None else 0
        if n_episodes <= 0:  # never analyzed (reltuples = -1) or genuinely empty
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
        # "not shown", NOT "behind recall": recall() searches the episode archive, not
        # the notes store, and fetch() is by-id only — an overflowed note's id appears
        # nowhere, so pointing at recall would overpromise.
        lines.append(f"(+ {dropped} older notes not shown)")

    if events:
        lines.append("")
        lines.append("## Last 7 days")
        for e in events:
            proj = f" ({e['project']})" if e.get("project") else ""
            fact = e["fact"]
            if len(fact) > _EVENT_FACT_MAX:
                fact = fact[: _EVENT_FACT_MAX - 1] + "…"
            tid = f" (t:{e['id']})" if e.get("id") is not None else ""
            lines.append(f"- {str(e['date'])[5:]}{proj}: {fact}{tid}")
    return "\n".join(lines)


def build_board(
    db_url: str, project: str | None, surface: str | None = None, trust: SurfaceTrust | None = None
) -> dict[str, Any]:
    """Build the rendered board block. Pure SQL, no embedding calls.

    Returns ``{"status": "ok", "text", "n_notes", "overflow", "note_ids", "trust"}`` —
    ``note_ids`` is the telemetry envelope's serve list; serve paths pop it before
    returning the block to callers (the ids the caller needs are inline as ``n:ID``).

    ``surface`` is the calling host's id (schema 053). It is resolved to a trust verdict
    that is FAIL-CLOSED at every step: absent, unregistered, or unreadable all mean
    restricted with an empty allowlist. On a restricted surface the notes section is
    filtered to ``audience='work-safe'``, and the digest + banner are filtered to the
    surface's project allowlist. ``trust`` is a pre-resolved verdict (tests, and callers
    that already looked it up); passing it skips the lookup.

    Missing tables (a deployment behind migration 033/041) degrade that section to
    empty rather than failing the whole board — same posture as preferences_routes.
    Degrading to EMPTY is the only acceptable degradation here: a filtered section that
    errors must never come back unfiltered, which is why the filters are applied inside
    the queries rather than as a post-pass a failure could skip.
    """
    st = trust if trust is not None else lookup_surface(db_url, surface)
    audience = st.audience_filter
    allowed = st.project_filter

    db = Database(db_url)
    try:
        notes = db.list_board_notes(_OWNER, project, audience=audience)
    except (psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn):
        notes = []
    finally:
        db.close()

    try:
        events = _recent_events(
            db_url,
            days=7,
            min_salience=2,
            limit=10,
            project=None,
            allowed_projects=allowed,
        )
    except psycopg.errors.UndefinedTable:
        events = []

    n_episodes, project_names = _banner_stats(db_url, allowed_projects=allowed)

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
        # Told to the caller so an unexpectedly thin board is diagnosable as "this host
        # isn't registered" instead of "memory is empty". It reveals nothing filtered.
        "trust": st.trust,
    }


def record_board_metrics(engine: Any, source: str, ms_total: float, board: dict[str, Any]) -> None:
    """One recall_metrics row (kind='board') per serve, through Recall's fire-and-forget
    writer (record_event). served_ids is the existing free-form JSONB envelope — no new
    DDL. Note ids are recorded in served form ("n:N") so board serves join directly
    against recall_feedback's helpful/noise ids (item 6: notes must be measurable).
    Fail-soft: telemetry must never break a serve."""
    try:
        text = board.get("text") or ""
        engine.record_event(
            "board",
            source=source,
            ms_total=round(ms_total, 2),
            chars=len(text),
            est_tokens=len(text) // 4,
            served_ids={
                "notes": [f"n:{i}" for i in (board.get("note_ids") or [])],
                "n_notes": board.get("n_notes", 0),
                "overflow": board.get("overflow", 0),
                # Restricted serves are expected to be thinner; without this the metrics
                # read as an unexplained drop in board size after the 053 rollout.
                "trust": board.get("trust"),
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
            return unauthorized()
        project = request.query_params.get("project") or None
        surface = request.query_params.get("surface") or None
        t0 = time.perf_counter()
        try:
            board = await run_in_threadpool(build_board, db_url, project, surface)
        except Exception as e:
            logger.warning("board build failed: %s", e)
            return err(str(e)[:200], 500)
        if get_recall is not None:
            record_board_metrics(get_recall(), "http", (time.perf_counter() - t0) * 1000.0, board)
        board.pop("note_ids", None)
        return JSONResponse(board)
