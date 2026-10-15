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

Pure SQL, no embedding calls, target <100ms: the note list (ingestion/db_notes.py's
select_board_notes), the timeline milestones (timeline_routes._recent_events — reused,
not duplicated), and two cheap banner aggregates. Each read borrows its own connection
through ingestion/pg_pool, one after another, never two at once.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from typing import Any

import psycopg
from fastmcp import FastMCP
from starlette.requests import Request

from ingestion import pg_pool
from ingestion.db_notes import select_board_notes
from ingestion.schema_check import ColumnProbe
from ingestion.surfaces import (
    UNKNOWN_SURFACE,
    SurfaceTrust,
    episode_scope_sql,
    lookup_surface,
    own_surface_if_ready,
    own_surface_served,
)
from mcp_server.caller_trust import note_ignored_surface
from mcp_server.http_helpers import route
from mcp_server.recall_warnings import serving_notice
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


# How many project names the banner lists, newest activity first.
_RECENT_PROJECTS = 12

# Schema 061: (project, created_at) on episodes, which the skip scan below needs. Without
# it the per-project max(created_at) is far slower than one pass over every episode, so
# the GROUP BY stays until a valid index exists (an interrupted CONCURRENTLY build leaves
# an invalid one).
PROJECT_RECENCY_INDEX = ColumnProbe(
    "SELECT 1 FROM pg_index "
    "WHERE indexrelid = to_regclass('episodes_project_created_idx') AND indisvalid",
    "schema 061 (episodes (project, created_at) index)",
)

# Every labeled project with its latest episode, as a skip scan: walk the distinct
# projects through the index, then read each one's newest created_at from it. Two index
# probes per project instead of a pass over the whole heap (~3 ms vs ~19 ms on a copy of
# prod's 52K episodes and 157 projects; 61 ms in prod). `project > ''` is both the start
# of the walk and the "labeled" test: NULL and '' fail it.
_RECENT_PROJECTS_SKIP_SCAN = """
WITH RECURSIVE labeled(project) AS (
  (SELECT project FROM episodes WHERE project > '' ORDER BY project LIMIT 1)
  UNION ALL
  SELECT (SELECT e.project FROM episodes e WHERE e.project > labeled.project
          ORDER BY e.project LIMIT 1)
  FROM labeled WHERE labeled.project IS NOT NULL
), latest AS (
  SELECT labeled.project,
         (SELECT max(e.created_at) FROM episodes e WHERE e.project = labeled.project) AS last_at
  FROM labeled WHERE labeled.project IS NOT NULL
)
SELECT project, count(*) OVER () FROM latest ORDER BY last_at DESC, project LIMIT %s
"""


def _banner_stats(
    db_url: str, allowed_projects: list[str] | None = None, own_surface: str | None = None
) -> tuple[int, int, list[str]]:
    """(total episodes, labeled projects, the newest-activity ones' names, capped at 12).

    The episode total comes from pg_class.reltuples (approximate but O(1)) rather
    than count(*): at ~40K rows / 1.5GB the exact count degrades to ~650ms of
    index-only scan on every session start, and a banner headline doesn't need the
    precise figure. reltuples is -1 until the table's first ANALYZE/VACUUM (fresh
    deploy, or a truncated test fixture), so fall back to an exact count while it's
    unpopulated — which also keeps the exact-count board tests green.

    The project count is every labeled project, not the length of the capped list: the
    banner said "across 12 projects" with 157 in the corpus. Projects order by their
    newest episode, then by name, so same-day projects no longer come out in whatever
    order the aggregate produced.

    ``allowed_projects`` (a restricted surface, schema 053) narrows ALL three numbers to
    the allowlist: the project names are the leak that matters ("most recent:
    family-health" on a work laptop is exactly the disclosure this feature exists to
    stop), and corpus-wide counts next to an allowlisted project list would misdescribe
    what the caller can actually reach. The reltuples shortcut and the skip scan are
    skipped in that branch — neither can answer a filtered question — but the filtered
    queries run over a small slice. ``own_surface`` (schema 057) widens them to the
    episodes the calling surface ingested itself, matching what recall will serve it.
    """
    with pg_pool.connection(db_url) as conn:
        own = own_surface_if_ready(conn, db_url, own_surface)  # None before schema 057
        scope = episode_scope_sql(allowed_projects, own)
        if scope is not None:
            pred, args = scope
            row = conn.execute(
                f"SELECT count(*) FROM episodes WHERE {pred}",  # nosec B608 — pred is a literal from episode_scope_sql; values are bound
                args,
            ).fetchone()
            n_episodes = int(row[0]) if row else 0
            rows = conn.execute(
                "SELECT project, count(*) OVER () FROM episodes "  # nosec B608 — as above
                f"WHERE {pred} AND project IS NOT NULL AND project <> '' "
                "GROUP BY project ORDER BY max(created_at) DESC, project LIMIT %s",
                [*args, _RECENT_PROJECTS],
            ).fetchall()
        else:
            row = conn.execute(
                "SELECT reltuples::bigint FROM pg_class WHERE oid = 'episodes'::regclass"
            ).fetchone()
            n_episodes = int(row[0]) if row and row[0] is not None else 0
            if n_episodes <= 0:  # never analyzed (reltuples = -1) or genuinely empty
                row = conn.execute("SELECT count(*) FROM episodes").fetchone()
                n_episodes = int(row[0]) if row else 0
            # Unlabeled episodes (NULL — and '', defensively) are excluded in SQL so they
            # never consume one of the LIMIT slots or inflate the project count.
            if PROJECT_RECENCY_INDEX.ready(conn, db_url):
                rows = conn.execute(_RECENT_PROJECTS_SKIP_SCAN, (_RECENT_PROJECTS,)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT project, count(*) OVER () FROM episodes "
                    "WHERE project IS NOT NULL AND project <> '' "
                    "GROUP BY project ORDER BY max(created_at) DESC, project LIMIT %s",
                    (_RECENT_PROJECTS,),
                ).fetchall()
    n_projects = int(rows[0][1]) if rows else 0
    return n_episodes, n_projects, [r[0] for r in rows]


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
    notice: str | None = None,
    n_projects: int | None = None,
) -> str:
    lines = [f"[Synapse board — project: {project or 'all'}]"]
    if notice:
        # Directly under the header, ahead of the "0 episodes" banner it explains.
        lines.append(f"WARNING: {notice}")
    recent = f" (most recent: {', '.join(project_names)})" if project_names else ""
    n_projects = len(project_names) if n_projects is None else n_projects
    lines.append(f"{n_episodes} episodes across {n_projects} projects{recent}.")
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


def _fit_notes(
    project: str | None,
    n_episodes: int,
    project_names: list[str],
    notes: list[dict[str, Any]],
    events: list[dict[str, Any]],
    notice: str | None,
    n_projects: int | None = None,
) -> tuple[str, list[dict[str, Any]], int]:
    """Drop the fewest notes that brings the board under both caps: ``(text, kept,
    dropped)``. Notes leave in drop order (drop class, then oldest updated_at, then id);
    the survivors keep their list order.

    Guard first: if the fixed portion (banner + clamped events) alone busts the caps, no
    amount of note-dropping can reach them, so every note is kept rather than draining
    the board for zero benefit. With event facts clamped this is a residual guard.

    The fewest drops is found by binary search over the drop count k, which renders
    O(log n) boards instead of one per dropped note (a few hundred live notes made that
    a ~300 ms CPU tax on every /context). Binary search is exact because ``_fits`` only
    ever flips once from k=1 upward: each further drop removes one note line, at least
    19 chars ("- " + hook + " (n:N, upd MM-DD)") and one line, and, if it was the last
    of its type, the section header too, while the overflow line gains at most one digit.
    Every note renders a line because notes_type_chk limits ``type`` to the four rendered
    sections. k=0 is checked on its own: the first drop adds the 27-char overflow line,
    so a short note line can make k=1 longer than k=0. A ``_render`` change that breaks
    this argument fails tests/test_board_cap.py, which pins the result to the old
    one-drop-at-a-time loop.
    """

    def render(kept: list[dict[str, Any]], dropped: int) -> str:
        return _render(
            project, n_episodes, project_names, kept, dropped, events, notice, n_projects
        )

    text = render(notes, 0)
    if not notes or _fits(text):
        return text, list(notes), 0
    n = len(notes)
    floor = render([], n)
    if not _fits(floor):
        return text, list(notes), 0

    # A stable sort reproduces the old loop's tie-break: min() took the first of equals.
    order = sorted(
        range(n),
        key=lambda i: (
            _DROP_CLASS.get(notes[i]["type"], 0),
            notes[i]["updated_at"],
            notes[i]["id"],
        ),
    )
    drop_rank = [0] * n
    for rank, i in enumerate(order):
        drop_rank[i] = rank

    def kept_after(k: int) -> list[dict[str, Any]]:
        return [note for i, note in enumerate(notes) if drop_rank[i] >= k]

    # Smallest k in [1, n] that fits; k=n (the floor) is known to.
    lo, hi, hi_text = 1, n, floor
    while lo < hi:
        mid = (lo + hi) // 2
        mid_text = render(kept_after(mid), mid)
        if _fits(mid_text):
            hi, hi_text = mid, mid_text
        else:
            lo = mid + 1
    return hi_text, kept_after(hi), hi


def build_board(
    db_url: str,
    project: str | None,
    surface: str | None = None,
    trust: SurfaceTrust | None = None,
    notice: str | None = None,
) -> dict[str, Any]:
    """Build the rendered board block. Pure SQL, no embedding calls.

    Returns ``{"status": "ok", "text", "n_notes", "overflow", "note_ids", "event_ids",
    "trust"}`` — ``note_ids`` and ``event_ids`` are the telemetry envelope's serve lists;
    serve paths pop them before returning the block to callers (the ids the caller needs
    are inline as ``n:ID`` and ``t:ID``).

    ``notice`` is a serve-nothing explanation (recall_warnings.serving_notice) that the
    route computed for an authenticated caller. It renders as a ``WARNING:`` line right
    under the header and comes back as ``warnings: [notice]``; the key exists only when
    there is one, the same contract as recall's ``warnings``.

    ``trust`` is the caller's verdict, resolved from its credential by the route. On a
    restricted verdict the notes section is filtered to ``audience='work-safe'``, and
    the digest + banner are filtered to the surface's project allowlist plus, for a
    credential-bound surface, what that surface ingested itself (schema 057).
    ``surface`` is the in-process alternative for server-side callers and tests: a
    surface id resolved here, FAIL-CLOSED at every step (absent, unregistered, or
    unreadable all mean restricted with an empty allowlist). It must never carry a
    caller-supplied value; the /context route does not pass one.

    Missing tables (a deployment behind migration 033/041) degrade that section to
    empty rather than failing the whole board — same posture as preferences_routes.
    Degrading to EMPTY is the only acceptable degradation here: a filtered section that
    errors must never come back unfiltered, which is why the filters are applied inside
    the queries rather than as a post-pass a failure could skip.
    """
    st = trust if trust is not None else lookup_surface(db_url, surface)
    audience = st.audience_filter
    allowed = st.project_filter
    own = st.own_surface

    try:
        with pg_pool.connection(db_url) as conn:
            notes = select_board_notes(conn, _OWNER, project, audience=audience)
    except (psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn):
        notes = []

    try:
        # Every project the caller may see, not just the requested one: the board is the
        # one place a session learns what else moved this week.
        events = _recent_events(
            db_url,
            days=7,
            min_salience=2,
            limit=10,
            project=None,
            allowed_projects=allowed,
            own_surface=own,
        )
    except psycopg.errors.UndefinedTable:
        events = []

    n_episodes, n_projects, project_names = _banner_stats(
        db_url, allowed_projects=allowed, own_surface=own
    )

    text, kept, dropped = _fit_notes(
        project, n_episodes, project_names, notes, events, notice, n_projects
    )

    out: dict[str, Any] = {
        "status": "ok",
        "text": text,
        "n_notes": len(kept),
        "overflow": dropped,
        "note_ids": [n["id"] for n in kept],
        "event_ids": [e["id"] for e in events],
        # Told to the caller so an unexpectedly thin board is diagnosable as "this host
        # isn't registered" instead of "memory is empty". It reveals nothing filtered.
        "trust": st.trust,
    }
    if notice:
        out["warnings"] = [notice]
    return out


def record_board_metrics(engine: Any, source: str, ms_total: float, board: dict[str, Any]) -> None:
    """One recall_metrics row (kind='board') per serve, through Recall's fire-and-forget
    writer (record_event). served_ids is the existing free-form JSONB envelope — no new
    DDL. Note and timeline-event ids are recorded in served form ("n:N", "t:N") so board
    serves join directly against recall_feedback's helpful/noise ids (notes and the "Last
    7 days" block must both be measurable).
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
                "events": [f"t:{i}" for i in (board.get("event_ids") or [])],
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
    mcp: FastMCP,
    db_url: str,
    authorized: Callable[[Request], bool],
    get_recall: Callable[[], Any] | None = None,
    resolve_trust: Callable[[Request], SurfaceTrust] | None = None,
    authenticated: Callable[[Request], bool] | None = None,
) -> None:
    """Mount GET /context.

    ``get_recall`` lazily yields the process's Recall engine so board serves share its
    telemetry writer; None (dev/stdio) skips telemetry. ``resolve_trust`` is the
    server's single caller-resolution point: a device token resolves to its surface,
    anything else (the root token included) to unknown. None means unknown for every
    request. A ``?surface=`` query param is ignored: it used to select the trust row for
    a root-token caller, which let any holder of the shared token borrow any row's trust.

    ``authenticated`` says whether the request presented a credential the server
    checked (http_auth.authenticated). Only then, and only with a resolved verdict, does
    a serve-nothing verdict put its explanation at the top of the board. None (an open
    server, or a caller that does not wire it) keeps the board silent, as before.
    """
    if not db_url:
        logger.info("board routes disabled (no DB_URL)")
        return

    @mcp.custom_route("/context", methods=["GET"])
    @route(authorized)
    def board_context(request: Request) -> dict[str, Any]:
        """The rendered board for the plugin's SessionStart hook (follow-up PR): one
        bounded always-relevant index block, not query-blind recall injection."""
        project = request.query_params.get("project") or None
        note_ignored_surface(request.query_params.get("surface"), "http.context")
        trust = resolve_trust(request) if resolve_trust is not None else UNKNOWN_SURFACE
        # Explain only a verdict that was actually resolved for an authenticated caller.
        told = resolve_trust is not None and authenticated is not None and authenticated(request)
        notice = None
        if told:
            notice = serving_notice(trust, own_surface_served(db_url, trust))
        t0 = time.perf_counter()
        board = build_board(db_url, project, None, trust, notice)
        if get_recall is not None:
            record_board_metrics(get_recall(), "http", (time.perf_counter() - t0) * 1000.0, board)
        board.pop("note_ids", None)
        board.pop("event_ids", None)
        return board
