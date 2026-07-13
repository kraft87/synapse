"""Synapse MCP server — recall, get_context, recall_episodes, fetch_episode,
recall_timeline, remember, list_projects, and query_graph as MCP tools.

Run with:
    uv run python -m mcp_server.server

Or via stdio for Claude Code:
    uv run python -m mcp_server.server --stdio
"""

from __future__ import annotations

import hmac
import logging
import os
from pathlib import Path

import logfire
from fastmcp import FastMCP
from fastmcp.exceptions import AuthorizationError
from fastmcp.server.auth import MultiAuth
from fastmcp.server.auth.providers.github import GitHubProvider
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier
from fastmcp.server.dependencies import get_access_token
from fastmcp.server.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse

# Logfire spans for the MCP server. Each tool invocation (recall, recall_episodes,
# remember, query_graph, list_projects) gets a top-level span; auto-instrument
# picks up any LLM/HTTP/FastMCP work underneath. Emits to whichever project the
# LOGFIRE_TOKEN env points at (matched to poller/dream via compose override).
logfire.configure(
    service_name=os.environ.get("LOGFIRE_SERVICE_NAME", "synapse-mcp"),
    send_to_logfire="if-token-present",
)
logfire.instrument_httpx()
logfire.instrument_mcp()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (from env / .env file)
# ---------------------------------------------------------------------------


def _load_env() -> dict[str, str]:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    env: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


_env = _load_env()

DB_URL = os.environ.get("SYNAPSE_DB_URL") or _env.get("SYNAPSE_DB_URL", "")
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY") or _env.get("VOYAGE_API_KEY", "")


def _cfg(key: str, default: str = "") -> str:
    return os.environ.get(key) or _env.get(key, default)


# --- Auth (env-gated; absent machine token => OPEN server, the pre-cutover / dev default) ---
# One opaque bearer gates /mcp (StaticTokenVerifier) AND /ingest + /recall (manual check below).
# Set GITHUB_CLIENT_ID to additionally stand up the claude.ai-web OAuth leg via MultiAuth.
MACHINE_TOKEN = _cfg("SYNAPSE_MACHINE_TOKEN")
GITHUB_CLIENT_ID = _cfg("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = _cfg("GITHUB_CLIENT_SECRET")
PUBLIC_URL = _cfg("SYNAPSE_PUBLIC_URL", "https://synapse.example.net")
OAUTH_SIGNING_KEY = _cfg(
    "SYNAPSE_OAUTH_SIGNING_KEY"
)  # stable => issued OAuth tokens survive restart
ALLOWED_GITHUB_USERS = {
    u.strip().lower() for u in _cfg("ALLOWED_GITHUB_USERS").split(",") if u.strip()
}
_MACHINE_CLIENT_ID = "synapse-machine"  # marks the bearer leg so the GitHub allowlist skips it

# ---------------------------------------------------------------------------
# Server bootstrap
# ---------------------------------------------------------------------------

import sys as _sys  # noqa: E402

_use_http = "--stdio" not in _sys.argv
_mcp_host = "0.0.0.0" if _use_http else "127.0.0.1"
_mcp_port = int(os.environ.get("MCP_PORT", "8765"))


class _GitHubAllowlist(Middleware):
    """Gate tool calls so the GitHub OAuth leg can't expose memory to the world.

    GitHubProvider admits ANY GitHub account by default; without this, anyone who
    completes the OAuth flow could read this instance's memory. The machine bearer
    leg carries client_id ``synapse-machine`` and skips the login check.
    """

    def __init__(self, allowed_logins: set[str]) -> None:
        self._allowed = allowed_logins

    async def on_call_tool(self, context, call_next):
        token = get_access_token()
        if token is not None and token.client_id != _MACHINE_CLIENT_ID:
            login = str((token.claims or {}).get("login", "")).lower()
            if login not in self._allowed:
                raise AuthorizationError(f"github user {login!r} not in allowlist")
        return await call_next(context)


def _build_auth():
    """(auth_provider, middleware). No machine token => open server (dev/stdio/pre-cutover)."""
    if not MACHINE_TOKEN:
        return None, []
    # scopes=["user"] so the machine token clears the GitHub leg's required scope on /mcp
    # (GitHubProvider defaults required_scopes=["user"], and MultiAuth applies it to /mcp).
    bearer = StaticTokenVerifier(
        {MACHINE_TOKEN: {"client_id": _MACHINE_CLIENT_ID, "scopes": ["user"]}}
    )
    if not GITHUB_CLIENT_ID:
        return bearer, []  # bearer-only: hooks + Claude Code --header; no claude.ai-web connector
    github = GitHubProvider(
        client_id=GITHUB_CLIENT_ID,
        client_secret=GITHUB_CLIENT_SECRET,
        base_url=PUBLIC_URL,
        jwt_signing_key=OAUTH_SIGNING_KEY or None,
        allowed_client_redirect_uris=[
            "https://claude.ai/api/mcp/auth_callback",
            "https://claude.com/api/mcp/auth_callback",
            # `synapse login` (RFC 8252): an ephemeral loopback redirect on a random
            # port. Without these patterns the OAuthProxy 400s the authorize step
            # ("does not match allowed patterns") and the CLI login can never complete.
            "http://localhost:*",
            "http://127.0.0.1:*",
        ],
    )
    if not ALLOWED_GITHUB_USERS:
        logger.warning(
            "GitHub OAuth on but ALLOWED_GITHUB_USERS empty -> all human logins DENIED (fail-closed)"
        )
    return MultiAuth(server=github, verifiers=[bearer]), [_GitHubAllowlist(ALLOWED_GITHUB_USERS)]


_auth, _auth_mw = _build_auth()
mcp = FastMCP("synapse", auth=_auth, middleware=_auth_mw)

# Serve skills_lane skills as skill:// MCP resources (PG-backed). Clients materialize
# them into ~/.claude/skills via sync_skills. Guarded on DB_URL so dev/stdio boots open.
if DB_URL:
    from mcp_server.skills_provider import PgSkillsProvider

    mcp.add_provider(PgSkillsProvider(DB_URL))


def _machine_authorized(request: Request) -> bool:
    """Custom routes bypass FastMCP's auth middleware (by design, issue #3704), so gate
    them here. Open when no machine token is set (dev / pre-cutover)."""
    if not MACHINE_TOKEN:
        return True
    authz = request.headers.get("authorization", "")
    if not authz.startswith("Bearer "):
        return False
    return hmac.compare_digest(authz[len("Bearer ") :].strip(), MACHINE_TOKEN)


# Skill sync + review over plain HTTP — lets the Claude Code plugin stay DSN-free (it talks to
# these machine-token-gated routes instead of reaching Postgres directly). No-op without DB_URL.
from mcp_server.skill_sync_routes import register as _register_skill_routes  # noqa: E402

_register_skill_routes(mcp, DB_URL, _machine_authorized)

# Config lane — the plugin mirrors each surface's opted-in config files here (machine-token gated)
# so the dream pipeline can read them and propose edits. Same DSN-free seam as skills. No-op w/o DB.
from mcp_server.config_sync_routes import register as _register_config_routes  # noqa: E402

_register_config_routes(mcp, DB_URL, _machine_authorized)

# Timeline event ingest — feeders (the plugin's git feeder, later calendar) POST plain event
# rows here; the server embeds + upserts. Same DSN-free machine-token seam. No-op w/o DB.
from mcp_server.timeline_routes import register as _register_timeline_routes  # noqa: E402

_register_timeline_routes(mcp, DB_URL, _machine_authorized, VOYAGE_API_KEY)

# Preferences read route — the plugin's SessionStart block GETs the top standing user
# preferences here (schema 035). Same machine-token seam; server owns the DB. No-op w/o DB.
from mcp_server.preferences_routes import register as _register_preferences_routes  # noqa: E402

_register_preferences_routes(mcp, DB_URL, _machine_authorized)

# Board read route — GET /context?project=X serves the rendered explicit-memory board
# (the same block the get_context tool returns) for the plugin's SessionStart hook.
# Same machine-token seam. No-op w/o DB. get_recall is a lazy thunk: _get_recall is
# defined below and only resolved at request time (telemetry shares its writer).
from mcp_server.board import register as _register_board_routes  # noqa: E402

_register_board_routes(mcp, DB_URL, _machine_authorized, get_recall=lambda: _get_recall())

# Device-login lane — RFC 8628 device flow so `synapse login` works browser-free on servers /
# headless boxes. Proxies GitHub's device flow and gates the machine token by the same GitHub
# allowlist as the web leg. No-op unless GITHUB_CLIENT_ID + SYNAPSE_MACHINE_TOKEN are set.
from mcp_server.device_routes import register as _register_device_routes  # noqa: E402

_register_device_routes(
    mcp, GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET, ALLOWED_GITHUB_USERS, MACHINE_TOKEN
)


# Lazy-init recall engine (one per process)
_recall_engine: Recall | None = None  # type: ignore[name-defined,unused-ignore]  # noqa: F821


def _get_recall() -> Recall:  # type: ignore[name-defined,unused-ignore]  # noqa: F821
    global _recall_engine
    if _recall_engine is None:
        from mcp_server.recall import Recall

        _recall_engine = Recall(db_url=DB_URL, voyage_api_key=VOYAGE_API_KEY)
    return _recall_engine


# Lazy-init timeline engine (one per process)
_timeline_engine: TimelineRecall | None = None  # type: ignore[name-defined,unused-ignore]  # noqa: F821


def _get_timeline() -> TimelineRecall:  # type: ignore[name-defined,unused-ignore]  # noqa: F821
    global _timeline_engine
    if _timeline_engine is None:
        from mcp_server.timeline import TimelineRecall

        _timeline_engine = TimelineRecall(db_url=DB_URL, voyage_api_key=VOYAGE_API_KEY)
    return _timeline_engine


def app_health_check() -> dict:
    """Health check callable — used by /health endpoint and container probes."""
    return {"status": "ok", "service": "synapse"}


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    """Unauthenticated liveness probe for the container healthcheck.

    Deliberately no auth and no DB touch: it answers "is the server process
    serving HTTP", nothing more. The previous container healthcheck imported
    this module in a fresh interpreter per probe (Logfire init and all) and
    routinely exceeded its own 5s timeout, flagging a healthy server unhealthy.
    """
    return JSONResponse(app_health_check())


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def recall(
    query: str,
    project: str | None = None,
    session_focus: list[str] | None = None,
    group_id: str = "technical",
) -> dict:
    """Search the user's long-term memory: tens of thousands of reranked
    past-conversation turns, the knowledge-graph facts extracted from them, and a
    dated event timeline.

    BEFORE answering anything that references past work — a prior decision or
    discussion ("what did we decide", "last time", "have we tried"), any device,
    purchase, tool, project, or person the user names, their preferences, or
    history this session alone can't supply — call this first. WHEN the topic
    shifts to something plausibly discussed before, call it again. Assume memory
    has the context; the failure mode is not checking, not over-checking.

    Do NOT call for facts already visible in the current conversation or context
    (read those directly), nor for generic-knowledge questions with no
    user-history angle (definitions, math, general how-tos).

    Query in plain language carrying the message's distinctive nouns; leave
    `project` unset unless results come back noisy from another domain. Served
    passages carry `role` (user / assistant / mixed) and a `date`: for "current
    state of X", weight newer user-stated content over older assistant-role text,
    which may be speculation or a plan that never happened.

    Follow-ups: fetch_episode(id) expands a truncated passage; recall_timeline()
    answers when-did / how-long; recall_episodes() returns raw turn text.

    Args:
        query: Natural language search query.
        project: Optional project slug to filter results (e.g. "synapse").
        session_focus: Entity names active in current conversation for KG bias.
        group_id: Knowledge graph scope — "technical" (default) or "personal".
    """
    with logfire.span(
        "mcp.recall {query!r}",
        query=query[:80],
        project=project,
        group_id=group_id,
    ):
        engine = _get_recall()
        return engine.recall(
            query=query,
            project=project,
            session_focus=session_focus or [],
            group_id=group_id,
            source="mcp-tool",
        )


# Registered DIRECTLY after recall on purpose: tool-list position biases tool choice,
# and the board is the session-start companion to recall — the order is deliberate.
@mcp.tool()
def get_context(project: str | None = None) -> dict:
    """The memory board: a small always-current index of the user's explicit
    memories — curated note hooks (rules & feedback, user facts, project state,
    references), the last week's milestones, and what memory exists at all.

    Call ONCE at session start, or when disoriented about what memory exists /
    what was recently worked on — it answers "what should I already know here?"
    before any work begins. Each line is a hook carrying its note id (n:12);
    when a hook is relevant, fetch the full note body by that id.

    Do NOT call per-question — recall() covers search; the board is recognition,
    not retrieval. Do NOT call more than about once per session — its content
    changes slowly. Absence from the board means SEARCH (recall), not
    doesn't-exist.

    Args:
        project: Optional project slug — scopes the board's project-notes section.
    """
    import time as _time

    from mcp_server.board import build_board, record_board_metrics

    with logfire.span("mcp.get_context", project=project):
        t0 = _time.perf_counter()
        board = build_board(DB_URL, project)
        record_board_metrics(_get_recall(), "mcp-tool", (_time.perf_counter() - t0) * 1000.0, board)
        board.pop("note_ids", None)
        return board


@mcp.tool()
def recall_episodes(
    query: str,
    project: str | None = None,
    limit: int = 5,
) -> dict:
    """Raw episode drill-down: individual conversation turns.

    Use this when you need the exact text of a specific exchange —
    "what exactly did we say about X?", "show me that debug session".
    Returns full episode content ranked by relevance + recency.

    For a blended overview (top episodes + KG facts), use recall() instead.

    Args:
        query: Natural language search query.
        project: Optional project slug to filter results.
        limit: Max episodes to return (default 5, max ~10 before it gets noisy).
    """
    with logfire.span(
        "mcp.recall_episodes {query!r}",
        query=query[:80],
        project=project,
        limit=limit,
    ):
        engine = _get_recall()
        return engine.recall_episodes(
            query=query,
            project=project,
            limit=limit,
            source="mcp-tool",
        )


@mcp.tool()
def recall_timeline(
    query: str | None = None,
    since: str | None = None,
    until: str | None = None,
    project: str | None = None,
    min_salience: int = 0,
    limit: int = 20,
    group_id: str | None = None,
) -> dict:
    """The personal timeline: dated events that HAPPENED (commits, decisions, ships).

    Use this for "when did I..." / "what did I do last week" / "how long between X and Y"
    questions — anything about the ORDER or DATES of past work. recall() stays the tool
    for what-is-true facts; this is the tool for what-happened events.

    Two query shapes:
      - Topical (query set): "the login work" — hybrid-searches events, returns them
        in chronological order.
      - Pure-time (query empty, since/until set): "last week" — returns the window's
        events, milestones individually and dense runs collapsed to one line with
        count + first/last anchors (so durations stay auditable).

    Never compute a bare day-count from this yourself without citing the two anchor
    events' dates the payload gives you.

    Args:
        query: Optional topical search ("browser-free login work"). Empty = pure time-range.
        since: Optional ISO date/timestamp lower bound (inclusive).
        until: Optional ISO date/timestamp upper bound (exclusive).
        project: Optional project slug filter (e.g. "synapse").
        min_salience: 0 (all), 1 (skip routine), 2 (milestones only).
        limit: Max items for pure-time queries (post-collapse, default 20).
        group_id: Scope — pass "personal" for questions about the user's own life
            (health, appointments, purchases, career) to exclude technical/work
            events; unset or "technical" serves everything.
    """
    with logfire.span(
        "mcp.recall_timeline {query!r}",
        query=(query or "")[:80],
        since=since,
        until=until,
        project=project,
    ):
        res = _get_timeline().recall_timeline(
            query=query,
            since=since,
            until=until,
            project=project,
            min_salience=min_salience,
            limit=limit,
            group_id=group_id,
        )
        for it in res.get("items") or []:
            it.pop("_id", None)  # internal telemetry key (recall served_ids), not for callers
        return res


@mcp.tool()
def fetch_episode(episode_ids: list[str]) -> dict:
    """Drill down to the FULL text of specific episodes by id.

    recall() serves compact passages, each tagged with its parent episode ``id`` (e.g. "e:227168").
    When a passage looks relevant but you need the whole untruncated turn, pass its id(s) here.
    Accepts the "e:N" ids from recall()/recall_episodes() results.

    Args:
        episode_ids: Episode ids to expand (the ``id`` field from recall results), max ~20.
    """
    with logfire.span("mcp.fetch_episode", n=len(episode_ids)):
        return _get_recall().fetch_episodes(episode_ids, source="mcp-tool")


def _notes_deps() -> tuple:
    """(embedder, llm) for the notes reconcile path — a seam so tests can stub both.

    BOTH constructions degrade to ``None`` on failure — remember() must never
    surface a raw exception for a config problem (the episode is already written
    by the time reconcile runs). Embedder ``None`` (keyless dev/test):
    reconcile_note stores a NULL embedding and skips dedup rather than failing
    the write (same degrade the timeline ingest route uses). LLM ``None`` (e.g.
    a bad SYNAPSE_LLM_PROVIDER): the confirm call fails inside reconcile_note's
    blanket except and collapses to "same" -> UPDATE (the fail-open design)."""
    from ingestion.embedding import create_embedder
    from ingestion.llm_client import create_llm_client

    try:
        embedder = create_embedder(voyage_api_key=VOYAGE_API_KEY, db_url=DB_URL)
    except Exception as e:
        logger.warning("notes embedder unavailable (%s); note dedup will be skipped", e)
        embedder = None
    try:
        llm = create_llm_client()
    except Exception as e:
        logger.warning("notes LLM unavailable (%s); confirm will collapse to update", e)
        llm = None
    return embedder, llm


def _derive_hook(content: str) -> str:
    """Legacy-form board line: first sentence of the content, hard-truncated to 120."""
    import re

    stripped = content.strip()
    first_line = stripped.splitlines()[0] if stripped else ""
    m = re.search(r"[.!?](?:\s|$)", first_line)
    first = first_line[: m.end()].strip() if m else first_line
    return first[:120]


@mcp.tool()
async def remember(
    content: str | None = None,
    hook: str | None = None,
    body: str | None = None,
    type: str = "project",
    project: str | None = None,
    session_id: str | None = None,
) -> dict:
    """Write to the user's curated long-term memory: reconciles a NOTE into the
    explicit notes store (deduped against the live set, superseded on
    contradiction) AND archives the text as an episode with knowledge-graph
    extraction.

    WHEN the user states a durable fact, preference, or decision, corrects
    something you had wrong, or explicitly asks you to remember → call this. If
    you are about to reply "noted" / "got it" / "I'll remember that," call this
    FIRST, then reply. Also use it to bank a 2-3 sentence summary of what was
    decided or built before a session ends or is cleared.

    Do NOT call for transient task state (what you're mid-doing, "running late"),
    to restate something already stored, or to save your own speculation or an
    unconfirmed plan. Routine conversation is ingested automatically.

    PREFERRED form — pass hook + body (+ type). `hook` is the one-line index
    entry (target ~120 chars, hard cap 200; internal newlines/whitespace runs
    are collapsed to single spaces); `body` is the full note, fetched on demand
    by id. Passing EITHER hook or body selects this form: both must then be
    non-empty — there is no silent fallback to legacy (a lone hook would be
    discarded). If `content` is also given alongside a full hook + body pair it
    becomes the archived episode text (the note itself stays hook + body);
    otherwise the episode is "hook\\n\\nbody". The legacy content-only form is
    kept for compatibility: it derives the hook from the first sentence and
    files the note as type 'project'.

    WRITE CONTRACT — what a note must look like:
    - Decisions WITH reasons, not changelog lines: "chose X because Y" survives;
      "merged the fix" / "updated the file" is noise.
    - DECLARATIVE, not imperative: "User prefers concise responses", never
      "Always respond concisely" — imperative memory re-reads as a standing
      directive and overrides live requests.
    - 7-day staleness blocklist: PR numbers, commit SHAs, "phase N done", file
      counts — anything stale within a week belongs in the episode archive
      (written automatically), never in a note.
    - Absolute dates ("2026-07-12"), never "yesterday" or "last week".
    - Self-contained body: it must stand alone months later with zero
      surrounding conversation.

    Type semantics:
      user      — durable facts about the user (global — every session).
      feedback  — corrections to agent behavior (global — every session).
      project   — scoped to `project`, staleness-managed (the default).
      reference — pointers to canonical sources (repos, docs, runbooks).

    Args:
        content: Legacy form — full text to remember (hook and type derived).
        hook: Preferred form — one-line note index entry (target ~120 chars).
        body: Preferred form — the full, self-contained note text.
        type: Note type — 'user' | 'feedback' | 'project' (default) | 'reference'.
        project: Optional project slug (scopes 'project' notes and the episode).
        session_id: Optional session ID to attach the episode to.
    """
    import time as _time
    import uuid as _uuid

    import anyio.to_thread

    from ingestion.db import Database
    from ingestion.models import Episode, ExtractionItem
    from ingestion.notes import _VALID_TYPES, reconcile_note

    structured = hook is not None or body is not None
    if not structured and not (content or "").strip():
        return {
            "status": "error",
            "detail": "provide hook + body (preferred) or content (legacy)",
        }
    if type not in _VALID_TYPES:
        return {
            "status": "error",
            "detail": f"invalid type {type!r} — expected one of {_VALID_TYPES}",
        }

    if structured:
        # Passing either hook or body commits to the structured form — a lone
        # hook (even with content also present) must NOT silently fall back to
        # legacy, which would discard the caller's hook. Board lines are
        # single-line by contract: collapse whitespace runs/newlines BEFORE the
        # hard cap so a multi-line hook can't smuggle newlines under it.
        note_hook = " ".join((hook or "").split())[:200]  # hard cap; targets ~120
        note_body = (body or "").strip()
        if not note_hook or not note_body:
            missing = "hook" if not note_hook else "body"
            return {
                "status": "error",
                "detail": f"structured form requires hook + body — {missing} is missing or blank",
            }
        note_type = type
        # content alongside a full pair serves as the archived episode text.
        ep_content = (content or "").strip() or f"{note_hook}\n\n{note_body}"
    else:
        ep_content = (content or "").strip()  # non-empty per the form check above
        note_hook = _derive_hook(ep_content)
        note_body = ep_content
        note_type = "project"

    sid = session_id or str(_uuid.uuid4())

    def _work() -> dict:
        t0 = _time.perf_counter()
        # Use a dedicated short-lived connection for writes — keeps the shared
        # recall engine's read connection clean and avoids transaction leakage.
        db = Database(DB_URL)
        try:
            existing = db.get_session_episodes(sid)
            seq = (max(e["sequence"] for e in existing) + 1) if existing else 1

            ep = Episode(
                session_id=sid,
                sequence=seq,
                project=project,
                content=ep_content,
                source="manual",
            )
            episode_id = db.upsert_episode(ep)

            db.enqueue_extraction(
                ExtractionItem(
                    episode_id=episode_id,
                    session_id=sid,
                    content=ep_content,
                    content_type="manual",
                    project=project,
                )
            )

            embedder, llm = _notes_deps()
            res = reconcile_note(
                db,
                embedder,
                llm,
                hook=note_hook,
                body=note_body,
                type=note_type,
                project=project,
                source_ref=f"ep:{episode_id}",
            )
        finally:
            db.close()

        _get_recall().record_event(
            "remember",
            source="mcp-tool",
            ms_total=(_time.perf_counter() - t0) * 1000.0,
            served_ids={"note": res["note_id"], "outcome": res["outcome"], "type": note_type},
        )
        return {
            "status": "ok",
            "note_id": res["note_id"],
            "outcome": res["outcome"],
            "episode_id": episode_id,
            "session_id": sid,
        }

    # reconcile_note does blocking DB I/O + possibly a sync LLM call that runs
    # asyncio.run() internally — it must live on a worker thread, never on the
    # FastMCP event loop (same trap query_graph documents).
    return await anyio.to_thread.run_sync(_work)


@mcp.custom_route("/ingest", methods=["POST"])
async def ingest_turns(request: Request) -> JSONResponse:
    """Direct-push ingest endpoint — replaces the Logfire poll for Claude Code.

    A Claude Code ``Stop`` hook POSTs a bounded TAIL of the session transcript
    (raw JSONL records) here on every turn. We parse with the SAME ``JSONLParser``
    the disk sweep uses, then dedup by span_id (a turn's stable last-record uuid):
    already-stored turns are skipped, new turns append at ``max(sequence)+1``. The
    parser's positional sequence is NOT used as the key — a tail would renumber it
    from 1 and collide — so identity rides on span_id and the sweep/push still
    converge idempotently. A full-transcript POST stays correct too (all old turns
    skip), so this is backward compatible with the pre-tail hook.

    Any client that runs the real ``claude`` CLI fires this same hook.
    Body: {"records": [...], "project": optional, "source": optional}.
    """
    if not _machine_authorized(request):
        return JSONResponse({"status": "error", "detail": "unauthorized"}, status_code=401)

    from starlette.concurrency import run_in_threadpool

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"status": "error", "detail": "invalid JSON body"}, status_code=400)

    records = body.get("records")
    if not isinstance(records, list):
        return JSONResponse(
            {"status": "error", "detail": "body must contain a 'records' list"},
            status_code=400,
        )
    source_label = body.get("source") or "hook"
    project_override = body.get("project")

    def _work() -> int:
        from ingestion.contamination import is_harness_call, is_transcript_contamination
        from ingestion.db import Database
        from ingestion.jsonl_client import JSONLParser
        from ingestion.models import ExtractionItem

        episodes = JSONLParser().parse_records(records, source_label, project_override)
        if not episodes:
            return 0
        db = Database(DB_URL)
        try:
            # The hook ships a bounded TAIL of the transcript, so the parser's
            # positional sequence is meaningless here — turn 50 arrives numbered 1
            # and would collide with the real turn 1. Identity is the span_id (a
            # turn's last record uuid), stable across full and tail parses. Per
            # session, load the stored span_ids + max sequence ONCE, then:
            #   * span_id already stored -> skip wholesale. Idempotent no-op; also
            #     stops a tail's truncated leading fragment (same span_id as the
            #     full turn it tails) from overwriting that turn.
            #   * new span_id            -> append at max_seq+1 and enqueue for KG.
            # Backward compatible with a full-transcript POST (old hook): every old
            # turn skips, only genuinely-new tail turns append at the same numbers
            # they had positionally — same final state, minus the O(turns^2)
            # re-upsert/re-enqueue churn the full re-ship used to cause.
            index: dict[str, tuple[set[str], int]] = {}
            written = 0
            dropped = 0
            content_dups = 0
            for ep in episodes:
                # Reject transcribe_ai deposition payloads before they ever land — third-party
                # PII must not enter memory. Dev conversations about the domain still flow in.
                # Likewise reject Synapse's own extraction/judge calls (the eval must not
                # eat itself — see contamination.is_harness_call).
                if is_transcript_contamination(ep.content) or is_harness_call(ep.content):
                    dropped += 1
                    continue
                if ep.session_id not in index:
                    index[ep.session_id] = db.get_session_span_index(ep.session_id)
                seen, max_seq = index[ep.session_id]
                if not ep.span_id or ep.span_id in seen:
                    continue  # no identity key, or already stored — skip
                # Cross-session replay guard (schema 036): a retried session ships the
                # same turns under a new session id + new span ids, so the span index
                # above can't see them. SYNAPSE_CONTENT_DEDUP=0 is the kill switch.
                if os.environ.get("SYNAPSE_CONTENT_DEDUP", "1") != "0" and db.content_dup_exists(
                    ep.project, ep.content
                ):
                    content_dups += 1
                    continue
                max_seq += 1
                ep.sequence = max_seq
                seen.add(ep.span_id)
                index[ep.session_id] = (seen, max_seq)
                eid = db.upsert_episode(ep)
                if ep.content and ep.content.strip():
                    db.enqueue_extraction(
                        ExtractionItem(
                            episode_id=eid,
                            session_id=ep.session_id,
                            content=ep.content,
                            content_type="episode",
                            project=ep.project,
                        )
                    )
                written += 1
            if dropped:
                logger.info("ingest dropped %d transcribe_ai transcript-payload turn(s)", dropped)
            if content_dups:
                logger.info(
                    "ingest skipped %d cross-session content-duplicate turn(s)", content_dups
                )
            return written
        finally:
            db.close()

    try:
        n = await run_in_threadpool(_work)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("ingest failed")
        return JSONResponse({"status": "error", "detail": str(exc)[:200]}, status_code=500)
    return JSONResponse({"status": "ok", "ingested": n})


@mcp.custom_route("/recall", methods=["POST"])
async def recall_http(request: Request) -> JSONResponse:
    """Plain-HTTP recall for non-MCP callers — the auto-recall memory hook.

    A shell/Python hook (UserPromptSubmit) has no MCP client: command hooks talk
    stdout/exit-code only and can't invoke an MCP tool. This route gives them the
    same recall() over plain HTTP, reusing the warm process-singleton engine
    (loaded embedder + pooled PG connections + warm HNSW cache) so a per-turn hook
    stays fast instead of cold-starting the pipeline each call.

    Body: {"query": str, "group_id"?: str, "write_feedback"?: bool}.
    write_feedback defaults FALSE here: automatic recalls must not bump the
    retrieval-count feedback signal (bench-grade discipline). Fail-soft like
    /ingest — never raises past the JSONResponse boundary.
    """
    if not _machine_authorized(request):
        return JSONResponse({"status": "error", "detail": "unauthorized"}, status_code=401)

    from starlette.concurrency import run_in_threadpool

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"status": "error", "detail": "invalid JSON body"}, status_code=400)

    query = (body.get("query") or "").strip()
    if not query:
        return JSONResponse({"status": "error", "detail": "missing 'query'"}, status_code=400)
    group_id = body.get("group_id") or "technical"
    write_feedback = bool(body.get("write_feedback", False))
    source = body.get("source") or "http"

    def _work() -> dict:
        return _get_recall().recall(
            query=query, group_id=group_id, write_feedback=write_feedback, source=source
        )

    try:
        with logfire.span("http.recall {query!r}", query=query[:80], group_id=group_id):
            out = await run_in_threadpool(_work)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("http recall failed")
        return JSONResponse({"status": "error", "detail": str(exc)[:200]}, status_code=500)
    return JSONResponse(out)


@mcp.tool()
def list_projects() -> dict:
    """List all projects in Synapse with episode counts, summary counts, and last activity.

    Use this to discover what's in the system before running recall() with a project filter.
    """
    import psycopg
    from psycopg.rows import dict_row

    pg = psycopg.connect(DB_URL, row_factory=dict_row, autocommit=True)
    try:
        rows = pg.execute("""
            SELECT
                e.project,
                COUNT(DISTINCT e.id)                                        AS episodes,
                COUNT(DISTINCT sd.id) FILTER (WHERE sd.doc_type = 'summary') AS summaries,
                COUNT(DISTINCT sd.id) FILTER (WHERE sd.doc_type = 'dream')   AS dreams,
                MAX(e.created_at)::date                                     AS last_activity
            FROM episodes e
            LEFT JOIN synth_documents sd ON sd.project = e.project
            GROUP BY e.project
            ORDER BY MAX(e.created_at) DESC
        """).fetchall()
    finally:
        pg.close()

    return {"projects": [dict(r) for r in rows]}


@mcp.tool()
def query_graph(
    query: str,
    group_id: str = "technical",
    limit: int = 20,
) -> dict:
    """Experimental: translate a natural language query to SQL over the KG tables.

    WARNING: This is an experimental tool for exploratory use only. Results may be
    incorrect or slow. Never use in automated pipelines — use recall() instead.

    Args:
        query: Natural language description of what to find in the graph.
        group_id: Graph scope — "technical" or "personal".
        limit: Max results to return.
    """
    import psycopg

    from ingestion.llm_client import create_llm_client

    llm = create_llm_client()
    prompt = f"""You are a SQL query generator for a knowledge graph stored in Postgres.

Generate a single SELECT statement for this request: {query}

The graph lives in two tables:
- kg_entities(uuid, owner_id, group_id, name, normalized_name, entity_type, summary)
- kg_relationships(uuid, owner_id, group_id, src_uuid, tgt_uuid, name, fact,
  t_created, t_valid, t_invalid, t_expired)
  src_uuid / tgt_uuid join to kg_entities.uuid.

Rules:
- Always filter: owner_id = 'default' AND group_id = '{group_id}' (on every table referenced)
- Active facts only: kg_relationships.t_invalid IS NULL (unless the request asks for history)
- LIMIT {limit}
- SELECT only — never modify data.

Return ONLY the SQL, no explanation."""

    sql = ""
    try:
        # ClaudeCLIClient.messages.create calls asyncio.run() internally, which
        # blows up inside FastMCP's running event loop ("asyncio.run() cannot be
        # called from a running event loop" — true of the old Cypher version of
        # this tool too). Run it in a worker thread, where no loop is running.
        from concurrent.futures import ThreadPoolExecutor

        def _generate() -> str:
            from ingestion.llm_client import stage_model

            response = llm.messages.create(
                model=stage_model("QUERY_GRAPH"),
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            return str(response.content[0].text).strip()

        with ThreadPoolExecutor(max_workers=1) as pool:
            sql = pool.submit(_generate).result(timeout=120)
        # Strip markdown fences
        if sql.startswith("```"):
            sql = sql.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        sql = sql.rstrip(";").strip()

        # Guard: single read-only statement, belt (prefix check) and braces
        # (read-only transaction) both — the model is untrusted input here.
        if not sql.lower().startswith("select") or ";" in sql:
            return {"error": "generated statement is not a single SELECT", "sql": sql}

        with psycopg.connect(DB_URL) as conn:
            conn.read_only = True
            with conn.cursor() as cur:
                cur.execute(sql)
                cols = [d.name for d in cur.description] if cur.description else []
                rows = [list(r) for r in cur.fetchmany(limit)]

        return {
            "sql": sql,
            "columns": cols,
            "results": rows,
            "count": len(rows),
        }
    except Exception as e:
        return {"error": str(e), "sql": sql}


@mcp.tool()
def issue_machine_token() -> dict:
    """Return this Synapse's shared machine bearer token (auth-gated).

    Lets ``synapse login`` fetch the token over OAuth instead of a manual copy-paste:
    the caller authenticates to /mcp (GitHub OAuth or an existing bearer), the
    on_call_tool allowlist gates it to permitted identities, and we hand back the
    token the hooks send to /ingest, /recall, and /mcp. Empty if auth is disabled.
    """
    return {"token": MACHINE_TOKEN}


if __name__ == "__main__":
    import sys

    from ingestion.schema_check import check_schema_version

    logging.basicConfig(level=logging.INFO)
    check_schema_version(DB_URL)

    if "--stdio" not in sys.argv:
        _mode = "authenticated" if MACHINE_TOKEN else "OPEN (no auth)"
        logger.info("Starting Synapse MCP server (http, %s) on %s:%d", _mode, _mcp_host, _mcp_port)
        mcp.run(transport="http", host=_mcp_host, port=_mcp_port, stateless_http=True, path="/mcp")
    else:
        mcp.run(transport="stdio")
