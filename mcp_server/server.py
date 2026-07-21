"""Synapse MCP server — recall, fetch, remember, recall_timeline, recall_episodes,
and recall_feedback as MCP tools (plus issue_machine_token, hidden from listings).

Run with:
    uv run python -m mcp_server.server

Or via stdio for Claude Code:
    uv run python -m mcp_server.server --stdio
"""

from __future__ import annotations

import hmac
import logging
import os
import re
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

# Logfire spans for the MCP server. Each tool invocation (recall, fetch,
# remember, ...) gets a top-level span; auto-instrument
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


# Tools that stay fully callable via tools/call but never appear in tools/list:
# plumbing that a specific client invokes by name (synapse_login's raw tools/call
# on issue_machine_token) and that a model browsing the tool list must not see.
_HIDDEN_TOOLS = {"issue_machine_token"}


class _HiddenToolsList(Middleware):
    """Filter hidden tools out of tools/list responses.

    Listing and calling are separate request paths in FastMCP, so dropping a tool
    here leaves tools/call untouched — `synapse login` keeps working while the
    model-facing surface stays the six deliberate tools registered below."""

    async def on_list_tools(self, context, call_next):
        tools = await call_next(context)
        return [t for t in tools if t.name not in _HIDDEN_TOOLS]


def _oauth_client_storage():
    """Where the OAuth proxy keeps its state: DCR client registrations, upstream GitHub
    tokens, and JTI mappings.

    FastMCP's default is a FileTree store under ~/.local/share/fastmcp/oauth-proxy. In a
    container with no volume that path is on the ephemeral layer, so every recreate
    (watchtower redeploy, reboot) wipes it — the claude.ai connector's registered client
    then vanishes and its next token refresh fails, forcing a full re-auth. Persist in
    Postgres instead (the DB already survives on its own volume).

    FastMCP only Fernet-wraps the state in its disk-default branch; a bare backend stores
    the upstream GitHub tokens as plaintext. Since this is the same DB recall() serves, we
    wrap it ourselves with a key deterministically derived from the signing key — matching
    the default's encryption-at-rest. Returns None (=> FastMCP's encrypted disk default)
    when there's no DB or signing key, e.g. dev/stdio.
    """
    if not (DB_URL and OAUTH_SIGNING_KEY):
        return None
    import base64
    import hashlib

    from cryptography.fernet import Fernet
    from key_value.aio.stores.postgresql import PostgreSQLStore
    from key_value.aio.wrappers.encryption import FernetEncryptionWrapper

    # Deterministic 32-byte Fernet key from the (stable) signing key. Self-contained: it
    # does not have to match FastMCP's internal derivation, only be stable across restarts
    # so the same ciphertext decrypts after a redeploy.
    fernet_key = base64.urlsafe_b64encode(
        hashlib.sha256(f"synapse-oauth-store::{OAUTH_SIGNING_KEY}".encode()).digest()
    )
    store = PostgreSQLStore(url=DB_URL, table_name="oauth_proxy_kv")
    return FernetEncryptionWrapper(
        key_value=store, fernet=Fernet(fernet_key), raise_on_decryption_error=False
    )


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
        client_storage=_oauth_client_storage(),
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

# Server instructions: with tool search on (Claude Code's default) only tool NAMES and
# this string load at session start — it is the always-loaded orientation surface that
# tells the model these tools exist and when to reach for them. Claude Code truncates
# it at 2KB (test_tool_surface.py pins the cap); most other clients ignore the field,
# which costs nothing.
_INSTRUCTIONS = (
    "Synapse is the user's persistent cross-session memory: tens of thousands of "
    "past conversation turns, knowledge-graph facts extracted from them, a dated "
    "event timeline, and curated notes. A board of note hooks is injected at "
    "session start where the client supports it; note bodies expand by id. "
    "BEFORE answering anything that references past work — a prior decision, "
    "device, purchase, tool, project, person, or preference — search with "
    "recall(query) first. fetch(ids) expands episode ids (e:N) and note ids "
    "(n:N) from earlier results. recall_timeline answers when-did / how-long "
    "questions; recall_episodes returns raw turn text. WHEN the user states a "
    "durable fact or correction, or you are about to say 'noted', call remember "
    "FIRST, then reply. AFTER a recall whose results you used, recall_feedback "
    "reports which served ids helped, which were noise, and what was missing. "
    "Absence from results means unknown, not false."
)

mcp = FastMCP(
    "synapse", instructions=_INSTRUCTIONS, auth=_auth, middleware=[*_auth_mw, _HiddenToolsList()]
)

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
# for the plugin's SessionStart hook (the ONLY serve path — see the Tools comment).
# Same machine-token seam. No-op w/o DB. get_recall is a lazy thunk: _get_recall is
# defined below and only resolved at request time (telemetry shares its writer).
from mcp_server.board import register as _register_board_routes  # noqa: E402

_register_board_routes(mcp, DB_URL, _machine_authorized, get_recall=lambda: _get_recall())

# Operator dashboard — static React bundle at /dash + read/flag API at /dash/api/* (issue #12,
# contract docs/dashboard-contract.md). Static routes are unauthenticated (public bundle, no
# data); every api route rides the same machine-token seam. No-op w/o DB_URL, like the siblings.
from mcp_server.dashboard_routes import register as _register_dashboard_routes  # noqa: E402

_register_dashboard_routes(mcp, DB_URL, _machine_authorized)

# Device-login lane — RFC 8628 device flow so `synapse login` works browser-free on servers /
# headless boxes. Proxies GitHub's device flow and gates the machine token by the same GitHub
# allowlist as the web leg. No-op unless GITHUB_CLIENT_ID + SYNAPSE_MACHINE_TOKEN are set.
from mcp_server.device_routes import register as _register_device_routes  # noqa: E402

_register_device_routes(
    mcp, GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET, ALLOWED_GITHUB_USERS, MACHINE_TOKEN
)

# Browser-login lane — authorization-code flow for the dashboard login screen (redirect UX;
# the device flow stays for `synapse login`). Same GitHub identity + allowlist gate; return
# origins restricted via SYNAPSE_DASH_ORIGINS. Same enablement condition as the device flow.
from mcp_server.web_login_routes import register as _register_web_login_routes  # noqa: E402

_register_web_login_routes(
    mcp, GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET, ALLOWED_GITHUB_USERS, MACHINE_TOKEN, PUBLIC_URL
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
# Tools — REGISTRATION ORDER IS DELIBERATE. Tool-list position biases which tool
# a model reaches for (first-listed wins most ties), so the surface reads in
# intended-use order: recall (the workhorse), fetch (id expansion), remember (the
# write), the two specialist reads — recall_timeline, recall_episodes — then
# recall_feedback (the after-the-fact quality report). Hidden plumbing
# (issue_machine_token, see _HIDDEN_TOOLS) registers last.
# test_tool_surface.py pins this order.
#
# There is deliberately NO board tool: the board is push-only (the plugin's
# SessionStart hook injects it via GET /context). A listed get_context tool told the
# model to "call this once at session start" when the hook had already injected the
# block — a compliant model double-spends ~2K tokens. The Hermes pattern: when
# injection covers the read, ship no read tool. Clients without the hook (claude.ai
# connector) never spontaneously orient anyway; re-registering is a small revert if
# that changes.
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

    Follow-ups: fetch(ids) expands a truncated passage or note body;
    recall_timeline() answers when-did / how-long; recall_episodes() returns raw
    turn text.

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


@mcp.tool()
def fetch(ids: list[str]) -> dict:
    """Expand memory ids into full records: episode ids from recall results
    ("e:227168") and note ids from the board ("n:12"), mixed freely in one call.

    WHEN a recall() passage is relevant but truncated and you need the whole
    turn, or WHEN a hook on the session-start board block (its n:ID lines)
    matters and you need the note body — pass the ids here. Bare numeric ids
    are treated as episode ids.

    Do NOT search with this — it only expands ids you already hold; recall()
    finds things. Do NOT re-fetch ids already expanded this session. Unknown or
    unparseable ids are returned under "skipped"; at most 20 ids per call.

    Args:
        ids: Ids to expand — "e:N" episodes, "n:N" notes, bare N = episode.
    """
    with logfire.span("mcp.fetch", n=len(ids)):
        return _get_recall().fetch(ids, source="mcp-tool")


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


# Docstring formatting is load-bearing: a bare "Word:" line is parsed as a docstring
# SECTION and everything from it on is silently dropped from the wire description
# (that once cost this tool its entire type-semantics block). Em-dash headers
# survive; test_tool_surface.py pins the tail phrases of every description.
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
    something you had wrong, or explicitly asks you to remember → call this.
    If you are about to reply "noted" / "got it" / "I'll remember that," call
    this FIRST, then reply. Also bank a 2-3 sentence summary of what was
    decided before a session ends or is cleared.

    Do NOT call for transient task state, to restate something already stored,
    or to save your own speculation or an unconfirmed plan. Routine
    conversation is ingested automatically.

    Form — pass hook + body (+ type). `hook` is the one-line index entry
    (target ~120 chars, hard cap 200, whitespace collapsed); `body` is the
    full note, fetched on demand by id. Passing either selects this form and
    both must then be non-empty — a lone hook never falls back to legacy.
    The legacy content-only form stays for compatibility (hook derived from
    the first sentence, type 'project'); content alongside a full hook + body
    pair becomes the archived episode text.

    Type semantics — user: durable facts about the user, and feedback:
    corrections to agent behavior (both global — every session); project:
    scoped to `project`, staleness-managed (the default);
    reference: pointers to canonical sources.

    The write contract — a good note:
    - States decisions WITH reasons: "chose X because Y" survives; "merged
      the fix" is noise.
    - Is DECLARATIVE, not imperative: "User prefers concise responses",
      never "Always respond concisely" — imperative memory re-reads as a
      standing directive.
    - Has no PR numbers, commit SHAs, "phase N done", file counts — anything
      week-stale belongs in the episode archive (written automatically),
      never in a note.
    - Uses absolute dates ("2026-07-12"), never "yesterday" or "last week".
    - Is self-contained: the body must stand alone months later.

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
    # asyncio.run() internally — asyncio.run() cannot be called from a running
    # event loop, so it must live on a worker thread, never on FastMCP's loop.
    return await anyio.to_thread.run_sync(_work)


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
    """The personal timeline: dated events that happened (commits, decisions,
    ships), served in chronological order.

    WHEN the question is about dates, order, or duration — "when did I...",
    "what did I do last week", "how long between X and Y" — call this. Two query
    shapes: topical (query set: "the login work") hybrid-searches events and
    returns them chronologically; pure-time (query empty, since/until set:
    "last week") returns the window's events, milestones individually and dense
    runs collapsed to one line with count + first/last anchors.

    Do NOT use this for what-is-true facts or preferences — recall() owns those;
    this tool owns what-happened events. Do NOT compute a bare day-count from
    the results without citing the two anchor events' dates the payload gives
    you.

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
    import time as _time

    with logfire.span(
        "mcp.recall_timeline {query!r}",
        query=(query or "")[:80],
        since=since,
        until=until,
        project=project,
    ):
        t0 = _time.perf_counter()
        res = _get_timeline().recall_timeline(
            query=query,
            since=since,
            until=until,
            project=project,
            min_salience=min_salience,
            limit=limit,
            group_id=group_id,
        )
        items = res.get("items") or []
        for it in items:
            it.pop("_id", None)  # internal telemetry key (recall served_ids), not for callers
        # One recall_metrics row (kind='timeline') per serve, through the recall
        # engine's fire-and-forget writer — same seam as remember/board telemetry.
        _get_recall().record_event(
            "timeline",
            source="mcp-tool",
            ms_total=round((_time.perf_counter() - t0) * 1000.0, 1),
            served_ids={"n_events": len(items)},
        )
        return res


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

    Do NOT use this for a blended overview — recall() serves the top episodes
    plus KG facts and is the right first call.

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


# --- recall_feedback: offline labeled retrieval-quality capture (schema 046) ------
# One row per rated recall. Deliberately NOT wired into live scoring — no ranking
# boost, no retrieval_count bump, nothing feeds _merge_rrf. The rows are goldens
# for offline eval + reranker tuning, so the id validation is strict: downstream
# tooling must be able to trust "e:N"/"n:N" without re-parsing.

_FEEDBACK_ID_RE = re.compile(r"^[en]:\d+$")


def _feedback_ids_error(field: str, ids: list[str] | None) -> str | None:
    """Validation error for recall_feedback's helpful/noise lists, or None if valid.

    Accepts None or a list of served-id strings — "e:N" (episode) / "n:N" (note),
    exactly as recall() serves them. Anything else is rejected."""
    if ids is None:
        return None
    if not isinstance(ids, list):
        return f"{field} must be a list of served ids like ['e:123', 'n:45']"
    bad = [i for i in ids if not (isinstance(i, str) and _FEEDBACK_ID_RE.fullmatch(i))]
    if bad:
        return (
            f"{field} contains invalid ids {bad!r} — expected recall-served ids "
            'matching "e:N" (episode) or "n:N" (note)'
        )
    return None


def _file_recall_feedback(
    query: str,
    helpful: list[str] | None,
    noise: list[str] | None,
    missing: str | None,
    note: str | None,
    session_id: str | None,
    project: str | None,
) -> dict:
    """Validate + INSERT one recall_feedback row — shared by the MCP tool and
    POST /feedback. Returns the tool-shaped result dict (never raises for bad
    input; DB errors propagate to the caller's boundary)."""
    from ingestion.db import Database

    q = (query or "").strip()
    if not q:
        return {"status": "error", "detail": "missing 'query' — pass the recall query being rated"}
    for field, ids in (("helpful", helpful), ("noise", noise)):
        err = _feedback_ids_error(field, ids)
        if err:
            return {"status": "error", "detail": err}

    db = Database(DB_URL)
    try:
        feedback_id = db.insert_recall_feedback(
            query=q,
            helpful=helpful or [],
            noise=noise or [],
            missing=missing,
            note=note,
            session_id=session_id,
            project=project,
        )
    finally:
        db.close()
    return {"status": "ok", "feedback_id": feedback_id}


@mcp.tool()
def recall_feedback(
    query: str,
    helpful: list[str] | None = None,
    noise: list[str] | None = None,
    missing: str | None = None,
    note: str | None = None,
    session_id: str | None = None,
    project: str | None = None,
) -> dict:
    """Report retrieval quality after a recall() whose results you used: which
    served ids genuinely helped, which were noise, and what was missing.

    AFTER acting on a recall's results — you answered from them, or found they
    lacked what you needed — call this ONCE with that recall's query string.
    `helpful` = served ids that were load-bearing for your answer; `noise` =
    served ids that were irrelevant or distracting; `missing` = one line on
    what you needed but were not served. Ids come from the recall response —
    "e:N" episodes, "n:N" notes. A report with only `missing` set is still
    valuable; file it when a recall came back empty-handed.

    This is offline labeled data (eval goldens, reranker tuning). It never
    changes live ranking, so honest negatives are safe and wanted.

    Do NOT file more than one report per recall query, do NOT rate recalls
    whose results you never used, and do NOT invent ids — report only ids the
    recall actually served.

    Args:
        query: The recall query being rated, verbatim.
        helpful: Served ids that were load-bearing ("e:123", "n:45").
        noise: Served ids that were irrelevant or distracting.
        missing: What you needed that the recall did not return.
        note: Free-form idea for improving this retrieval.
        session_id: Optional session id for grouping reports.
        project: Optional project slug the recall was scoped to.
    """
    with logfire.span("mcp.recall_feedback {query!r}", query=query[:80], project=project):
        return _file_recall_feedback(query, helpful, noise, missing, note, session_id, project)


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

    Body: {"query": str, "project"?: str, "group_id"?: str, "write_feedback"?: bool,
           "source"?: str, "debug"?: bool}.
    write_feedback defaults FALSE here: automatic recalls must not bump the
    retrieval-count feedback signal (bench-grade discipline) — and the phase-2
    dashboard debug console relies on this default staying false so its diagnostic
    recalls never pollute the feedback signal. ``debug`` attaches the per-leg timing /
    pool-size / rerank envelope the engine already measures (see recall(debug=...)).
    Fail-soft like /ingest — never raises past the JSONResponse boundary.
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
    project = body.get("project") or None
    group_id = body.get("group_id") or "technical"
    write_feedback = bool(body.get("write_feedback", False))
    source = body.get("source") or "http"
    debug = bool(body.get("debug", False))

    def _work() -> dict:
        return _get_recall().recall(
            query=query,
            project=project,
            group_id=group_id,
            write_feedback=write_feedback,
            source=source,
            debug=debug,
        )

    try:
        with logfire.span("http.recall {query!r}", query=query[:80], group_id=group_id):
            out = await run_in_threadpool(_work)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("http recall failed")
        return JSONResponse({"status": "error", "detail": str(exc)[:200]}, status_code=500)
    return JSONResponse(out)


@mcp.custom_route("/feedback", methods=["POST"])
async def feedback_http(request: Request) -> JSONResponse:
    """Plain-HTTP recall_feedback for non-MCP callers — the /recall sibling.

    Hooks and the dashboard talk to /recall over plain HTTP (no MCP client), so
    the labeled-feedback write needs the same seam or those callers could never
    file a report. Same validation + insert as the recall_feedback tool
    (_file_recall_feedback); same machine-token gate; fail-soft like /ingest.

    Body: {"query": str, "helpful"?: [..], "noise"?: [..], "missing"?: str,
           "note"?: str, "session_id"?: str, "project"?: str}.
    """
    if not _machine_authorized(request):
        return JSONResponse({"status": "error", "detail": "unauthorized"}, status_code=401)

    from starlette.concurrency import run_in_threadpool

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"status": "error", "detail": "invalid JSON body"}, status_code=400)

    def _work() -> dict:
        return _file_recall_feedback(
            query=body.get("query") or "",
            helpful=body.get("helpful"),
            noise=body.get("noise"),
            missing=body.get("missing"),
            note=body.get("note"),
            session_id=body.get("session_id"),
            project=body.get("project"),
        )

    try:
        with logfire.span("http.feedback"):
            out = await run_in_threadpool(_work)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("http feedback failed")
        return JSONResponse({"status": "error", "detail": str(exc)[:200]}, status_code=500)
    return JSONResponse(out, status_code=200 if out.get("status") == "ok" else 400)


# Registered LAST and hidden from tools/list (_HiddenToolsList): infra plumbing,
# not part of the model-facing surface. The `synapse login` CLI invokes it by
# name via a raw tools/call, which the listing filter deliberately leaves intact.
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
