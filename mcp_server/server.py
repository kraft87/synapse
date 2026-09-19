"""Synapse MCP server — recall, fetch, remember, and
recall_feedback as MCP tools (plus issue_machine_token, hidden from listings).

Run with:
    uv run python -m mcp_server.server

Or via stdio for Claude Code:
    uv run python -m mcp_server.server --stdio
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import logfire
from fastmcp import FastMCP
from fastmcp.exceptions import AuthorizationError
from fastmcp.server.dependencies import get_access_token
from fastmcp.server.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from ingestion.scope import personal_scope_enabled
from ingestion.surfaces import SurfaceTrust
from mcp_server.auth_tokens import (
    DEVICE_CLIENT_ID as _DEVICE_CLIENT_ID,
)
from mcp_server.auth_tokens import (
    KIND_DEVICE as _KIND_DEVICE,
)
from mcp_server.auth_tokens import (
    ROOT_CLIENT_ID as _MACHINE_CLIENT_ID,
)
from mcp_server.auth_tokens import (
    claims_of,
)

# The credential verifier and the constants it stamps, from ONE module so what WRITES a
# claim and what READS it cannot drift: the client_ids decide whether the human-login
# allowlist applies, and `kind` decides which trust lane a call resolves through.
from mcp_server.auxiliary_routes import register as _register_auxiliary_routes
from mcp_server.caller_trust import caller_trust as _resolve_caller_trust
from mcp_server.caller_trust import request_trust as _resolve_request_trust
from mcp_server.feedback_tools import register as _register_feedback_tools
from mcp_server.http_auth import admin_authorized, is_root, machine_authorized
from mcp_server.http_auth import bearer as _bearer
from mcp_server.ingest_route import register as _register_ingest_route
from mcp_server.recall_route import register as _register_recall_route
from mcp_server.remember_tool import register as _register_remember_tool
from mcp_server.retrieval_tools import register as _register_retrieval_tools
from mcp_server.server_auth import AuthSettings, build_auth, build_idp, oauth_client_storage

#: Both machine lanes. The human-login allowlist skips these client_ids — the credential
#: is their whole gate, and there is no identity on them to match against a list.
_MACHINE_CLIENT_IDS = {_MACHINE_CLIENT_ID, _DEVICE_CLIENT_ID}

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
    """Parse the dotenv fallback layer. Real env vars always win over this (see _cfg).

    SYNAPSE_ENV_FILE overrides the path. Tests that reload this module point it at a
    nonexistent path so a developer's repo-root .env can't leak into an assertion about
    unset config — `monkeypatch.setenv(k, "")` alone does NOT neutralize this layer,
    because _cfg treats an empty env var as absent and falls through to it.
    """
    override = os.environ.get("SYNAPSE_ENV_FILE")
    env_path = Path(override) if override else Path(__file__).resolve().parent.parent / ".env"
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


# --- Auth (machine token REQUIRED to start; SYNAPSE_ALLOW_OPEN=1 is the dev/stdio hatch) ---
# The ROOT bearer. It gates /mcp (via SynapseTokenVerifier) AND the internal write lanes
# (/ingest and friends, manual check below), and it is the ENROLLMENT credential every new
# device pastes once to obtain its own token. It is deliberately NOT sufficient for the
# privileged operations — approving a device, minting a token, the dashboard — because the
# credential a machine has to hold in order to join must not also be the credential that
# decides who may join. Set GITHUB_CLIENT_ID to additionally stand up the claude.ai-web
# OAuth leg via MultiAuth.
MACHINE_TOKEN = _cfg("SYNAPSE_MACHINE_TOKEN")
# Placeholders .env.example ships. Treated as "not set", so a copied-but-unedited .env
# fails the startup check instead of standing up a server with a guessable root bearer.
_TOKEN_PLACEHOLDERS = {"changeme", "change-me", "changeme!", "todo", "xxx"}
GITHUB_CLIENT_ID = _cfg("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = _cfg("GITHUB_CLIENT_SECRET")
PUBLIC_URL = _cfg("SYNAPSE_PUBLIC_URL", "https://synapse.example.net")
OAUTH_SIGNING_KEY = _cfg(
    "SYNAPSE_OAUTH_SIGNING_KEY"
)  # stable => issued OAuth tokens survive restart
ALLOWED_GITHUB_USERS = {
    u.strip().lower() for u in _cfg("ALLOWED_GITHUB_USERS").split(",") if u.strip()
}
# Alternative interactive IdP: any OIDC-compliant provider (e.g. self-hosted Authelia or
# Keycloak). When set it REPLACES GitHub for all three interactive flows (MCP OAuth,
# dashboard login, device login) — MCP discovery can only advertise one authorization
# server, so "alternative" is a per-deployment choice, not a login picker. Leave unset to
# keep the GitHub default; the machine-bearer leg is unaffected either way.
OIDC_CONFIG_URL = _cfg("OIDC_CONFIG_URL")  # .../.well-known/openid-configuration
OIDC_CLIENT_ID = _cfg("OIDC_CLIENT_ID")
OIDC_CLIENT_SECRET = _cfg("OIDC_CLIENT_SECRET")
# Matched against the identity claims below (lowercased).
ALLOWED_OIDC_USERS = {u.strip().lower() for u in _cfg("ALLOWED_OIDC_USERS").split(",") if u.strip()}
# Scopes requested/advertised/enforced on the MCP leg (space- or comma-separated).
# offline_access is what makes the IdP issue refresh tokens (long-lived connector
# sessions) — trim it if the IdP rejects the scope (e.g. Google).
OIDC_SCOPES = [
    s for s in re.split(r"[ ,]+", _cfg("OIDC_SCOPES", "openid profile email offline_access")) if s
]
# Claims consulted (in order) for the user's identity, in both the id_token (MCP leg)
# and userinfo (dashboard/device flows). Adjust per IdP, e.g. "email" for Google.
OIDC_USER_CLAIMS = tuple(
    c for c in re.split(r"[ ,]+", _cfg("OIDC_USER_CLAIMS", "preferred_username,email")) if c
)
# ---------------------------------------------------------------------------
# Server bootstrap
# ---------------------------------------------------------------------------
import sys as _sys  # noqa: E402

_use_http = "--stdio" not in _sys.argv
_mcp_host = "0.0.0.0" if _use_http else "127.0.0.1"
_mcp_port = int(os.environ.get("MCP_PORT", "8765"))


def _claims_identity(claims: dict, claim_keys: tuple[str, ...]) -> str:
    """First non-empty identity claim, lowercased; "" when none is present.

    Deliberately ONE definition: the allowlist gate below and the audience-scoping
    surface derivation (:func:`_caller_surface`) must never disagree about who a caller
    is, or a login could clear the gate as one identity and be served as another.
    """
    return next((str(claims[k]).lower() for k in claim_keys if claims.get(k)), "")


class _UserAllowlist(Middleware):
    """Gate tool calls so the interactive OAuth leg can't expose memory to the world.

    The OAuth proxies admit ANY upstream account by default; without this, anyone who
    completes the OAuth flow could read this instance's memory. The machine bearer legs
    (root ``synapse-machine`` and per-device ``synapse-device``) carry no human identity
    and skip the identity check — their gate is the credential itself. The identity claim
    differs by provider: GitHub tokens carry ``login``, OIDC id_tokens carry
    ``preferred_username``/``email`` (the IdP's claims config must put them in the
    id_token — e.g. an Authelia claims policy).
    """

    def __init__(self, allowed_users: set[str], claim_keys: tuple[str, ...], label: str) -> None:
        self._allowed = allowed_users
        self._claim_keys = claim_keys
        self._label = label

    async def on_call_tool(self, context, call_next):
        token = get_access_token()
        if token is not None and token.client_id not in _MACHINE_CLIENT_IDS:
            user = _claims_identity(token.claims or {}, self._claim_keys)
            if user not in self._allowed:
                raise AuthorizationError(f"{self._label} user {user!r} not in allowlist")
        return await call_next(context)


# Tools that stay fully callable via tools/call but never appear in tools/list:
# plumbing that a specific client invokes by name (synapse_login's raw tools/call
# on issue_machine_token) and that a model browsing the tool list must not see.
_HIDDEN_TOOLS = {"issue_machine_token"}


class _HiddenToolsList(Middleware):
    """Filter hidden tools out of tools/list responses.

    Listing and calling are separate request paths in FastMCP, so dropping a tool
    here leaves tools/call untouched — `synapse login` keeps working while the
    model-facing surface stays the deliberate tools registered below."""

    async def on_list_tools(self, context, call_next):
        tools = await call_next(context)
        return [t for t in tools if t.name not in _HIDDEN_TOOLS]


def _oauth_client_storage():
    return oauth_client_storage(DB_URL, OAUTH_SIGNING_KEY)


def _auth_settings() -> AuthSettings:
    return AuthSettings(
        DB_URL=DB_URL,
        MACHINE_TOKEN=MACHINE_TOKEN,
        OIDC_SCOPES=OIDC_SCOPES,
        OIDC_CONFIG_URL=OIDC_CONFIG_URL,
        OIDC_CLIENT_ID=OIDC_CLIENT_ID,
        OIDC_CLIENT_SECRET=OIDC_CLIENT_SECRET,
        PUBLIC_URL=PUBLIC_URL,
        OAUTH_SIGNING_KEY=OAUTH_SIGNING_KEY,
        ALLOWED_OIDC_USERS=ALLOWED_OIDC_USERS,
        OIDC_USER_CLAIMS=OIDC_USER_CLAIMS,
        GITHUB_CLIENT_ID=GITHUB_CLIENT_ID,
        GITHUB_CLIENT_SECRET=GITHUB_CLIENT_SECRET,
        ALLOWED_GITHUB_USERS=ALLOWED_GITHUB_USERS,
    )


def _build_auth():
    return build_auth(_auth_settings(), _oauth_client_storage, _UserAllowlist)


def _build_idp():
    return build_idp(_auth_settings())


_auth, _auth_mw = _build_auth()
_idp = _build_idp()

# Identity claims of whichever interactive leg _build_auth just selected, read off the
# allowlist middleware it built rather than re-derived from env — one source, no drift.
# Empty when there is no OAuth leg at all (open dev/stdio, bearer-only): those servers
# have no OAuth callers to identify, so the derivation below stays inert.
_IDENTITY_CLAIMS: tuple[str, ...] = _auth_mw[0]._claim_keys if _auth_mw else ()


def _access_token() -> Any:
    """The request's AccessToken, or None. Never raises (no auth context at all)."""
    try:
        return get_access_token()
    except Exception:  # pragma: no cover - defensive
        return None


def _caller_surface(surface: str | None) -> str | None:
    """The surface ID for this call — display/telemetry, and the legacy id lane.

    See :func:`_caller_trust` for the verdict itself; this is the id that goes with it.
    """
    return _caller_trust(surface).surface_id


def _caller_trust(surface: str | None) -> SurfaceTrust:
    """Resolve authenticated caller scope; see caller_trust for credential precedence."""
    return _resolve_caller_trust(
        db_url=DB_URL,
        surface=surface,
        access_token=_access_token(),
        identity_claims=_IDENTITY_CLAIMS,
        machine_client_ids=_MACHINE_CLIENT_IDS,
        claims_identity=_claims_identity,
    )


def _request_trust(request: Request, surface: str | None = None) -> SurfaceTrust:
    """:func:`_caller_trust` for the plain-HTTP custom routes.

    Custom routes bypass FastMCP's auth middleware by design (issue #3704), so there is
    no token context to read — the bearer has to be re-resolved from the header. Same
    precedence, minus the OAuth lane, which never reaches these routes: browsers and
    hooks send a bearer, not an OIDC token.
    """
    return _resolve_request_trust(
        db_url=DB_URL,
        machine_token=MACHINE_TOKEN,
        bearer=_bearer(request),
        surface=surface,
    )


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
    "(n:N) from earlier results. "
    "questions; recall_full_turns searches complete raw turns — the retry when an "
    "overview recall comes back thin. WHEN the user states a "
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


def _is_root(request: Request) -> bool:
    return is_root(request, MACHINE_TOKEN)


def _machine_authorized(request: Request) -> bool:
    return machine_authorized(request, MACHINE_TOKEN, DB_URL)


def _admin_authorized(request: Request) -> bool:
    return admin_authorized(request, MACHINE_TOKEN, DB_URL)


def _issue_dash_token(identity: str) -> str:
    """Mint the browser's full-trust device token (schema 054) after a dashboard login."""
    from ingestion.surfaces import issue_dash_token

    return issue_dash_token(DB_URL, identity)


_register_auxiliary_routes(
    mcp,
    DB_URL,
    VOYAGE_API_KEY,
    MACHINE_TOKEN,
    PUBLIC_URL,
    _idp,
    _machine_authorized,
    _admin_authorized,
    lambda: _get_recall(),
    _request_trust,
    _issue_dash_token,
)


# Lazy-init recall engine (one per process)
_recall_engine: Recall | None = None  # type: ignore[name-defined,unused-ignore]  # noqa: F821


def _get_recall() -> Recall:  # type: ignore[name-defined,unused-ignore]  # noqa: F821
    global _recall_engine
    if _recall_engine is None:
        from mcp_server.recall import Recall

        _recall_engine = Recall(db_url=DB_URL, voyage_api_key=VOYAGE_API_KEY)
    return _recall_engine


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


# Registration order is part of the public tool surface (test_tool_surface.py):
# recall, recall_full_turns, fetch, fetch_session, remember, recall_feedback.
# Hidden issue_machine_token registers last. The board stays push-only via /context;
# exposing a read tool would duplicate the SessionStart hook's injected context.


# Personal scope off (SYNAPSE_PERSONAL_SCOPE=0): the tool description must not
# advertise a graph this deployment does not keep. A model told about a scope
# that no longer exists spends a call asking for it. The sentence is swapped at
# import, before FastMCP reads the docstring into the tool description.
_GROUP_ID_DOC_SPLIT = 'group_id: Knowledge graph scope — "technical" (default) or "personal".'
_GROUP_ID_DOC_SINGLE = (
    "group_id: Knowledge graph scope. This deployment keeps one graph;\n            leave it unset."
)


def _scope_doc(fn):  # type: ignore[no-untyped-def]
    """Rewrite the personal-scope line out of a tool docstring when the scope is off."""
    if not personal_scope_enabled() and fn.__doc__:
        fn.__doc__ = fn.__doc__.replace(_GROUP_ID_DOC_SPLIT, _GROUP_ID_DOC_SINGLE)
    return fn


recall, recall_full_turns, fetch, fetch_session = _register_retrieval_tools(
    mcp, lambda: _get_recall(), lambda surface: _caller_trust(surface), _scope_doc
)


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
remember = _register_remember_tool(
    mcp,
    lambda: DB_URL,
    lambda: _get_recall(),
    lambda surface: _caller_trust(surface),
    lambda: _notes_deps(),
    _derive_hook,
)


# Spooled-remember replay — the plugin queues a memory write to local disk whenever the
# remember() MCP tool is unavailable (OAuth down, server never connected) and replays it
# here over the machine-token lane, which is a different transport and stayed up through
# the 2026-08-25 outage. Registered HERE, not with the sibling routes above, because it
# takes `remember` itself as its writer — same code path as the tool, no drift possible.
# Idempotent on the client's intent id (schema 052). No-op w/o DB_URL, like the siblings.
from mcp_server.remember_routes import register as _register_remember_routes  # noqa: E402

_register_remember_routes(
    mcp,
    DB_URL,
    _machine_authorized,
    remember,
    caller_surface=lambda request: _request_trust(request).surface_id,
)


# Offline retrieval-quality capture, shared by MCP and HTTP.
_feedback_ids_error, _file_recall_feedback, recall_feedback, feedback_http = (
    _register_feedback_tools(mcp, lambda: DB_URL, lambda request: _machine_authorized(request))
)


ingest_turns = _register_ingest_route(
    mcp, lambda: DB_URL, lambda request: _machine_authorized(request)
)


recall_http = _register_recall_route(
    mcp,
    lambda: _get_recall(),
    lambda request: _machine_authorized(request),
    lambda request, surface: _request_trust(request, surface),
)


# Registered LAST and hidden from tools/list (_HiddenToolsList): infra plumbing,
# not part of the model-facing surface. The `synapse login` CLI invokes it by
# name via a raw tools/call, which the listing filter deliberately leaves intact.
@mcp.tool()
def issue_machine_token() -> dict:
    """Return this Synapse's ROOT bearer — the enrollment credential (auth-gated).

    Lets ``synapse login`` fetch it over OAuth instead of a manual copy-paste: the
    caller authenticates to /mcp, the on_call_tool allowlist gates it to permitted
    identities, and we hand back the token a fresh machine uses to call
    ``POST /surfaces/enroll`` and obtain its OWN device token. Empty if auth is disabled.

    A DEVICE token is refused here (schema 054). Not because the root token grants more
    reading — a full-trust device already reads everything — but because handing the
    long-lived, un-revocable, shared enrollment credential to a per-device credential
    would collapse the two back into one and make revocation meaningless. Device tokens
    are the leaves; they do not get to fetch the root.
    """
    if claims_of(_access_token()).get("kind") == _KIND_DEVICE:
        raise AuthorizationError(
            "device tokens cannot fetch the root enrollment credential — "
            "log in with `synapse login` on the machine that needs to enroll"
        )
    return {"token": MACHINE_TOKEN}


_MISSING_TOKEN_ERROR = """
SYNAPSE_MACHINE_TOKEN is not set — refusing to start.

Since schema 054 a caller is served according to the credential it presents. With
no machine token there is no credential to check, so EVERY caller resolves to an
unknown surface and is served nothing: an empty board, empty recall, no error.

Fix it in one minute:

  1. Generate a token:      openssl rand -hex 32
  2. Put it in .env:        SYNAPSE_MACHINE_TOKEN=<that value>
  3. Restart:               docker compose up -d
  4. Mint this machine its own device token (printed once):
         docker compose exec mcp-server synapse-admin bootstrap "<label>"
     and paste that token into the plugin's Synapse token prompt.

Really want an open server (dev / stdio only)? Set SYNAPSE_ALLOW_OPEN=1. It still
serves every caller `restricted` — this flag suppresses the check, not the effect.
"""

_OPEN_BANNER = (
    "OPEN server: every caller is served restricted; "
    "set SYNAPSE_MACHINE_TOKEN and bootstrap a device"
)


def _startup_auth_mode() -> str:
    """``"authenticated"`` or ``"open"``, refusing to start when the token is missing.

    The open server is a legitimate dev/stdio shape, but it is NOT a working install:
    it serves nothing to everyone, silently. So the token is required by default and
    open mode has to be asked for by name (SYNAPSE_ALLOW_OPEN=1) — and says what it is
    every time it boots.
    """
    if MACHINE_TOKEN and MACHINE_TOKEN.strip().lower() not in _TOKEN_PLACEHOLDERS:
        return "authenticated"
    if _cfg("SYNAPSE_ALLOW_OPEN", "0") not in ("", "0"):
        logger.warning(_OPEN_BANNER)
        return "open"
    raise SystemExit(_MISSING_TOKEN_ERROR)


if __name__ == "__main__":
    import sys

    from ingestion.schema_check import check_schema_version

    logging.basicConfig(level=logging.INFO)
    _mode = _startup_auth_mode()
    check_schema_version(DB_URL)

    if "--stdio" not in sys.argv:
        logger.info("Starting Synapse MCP server (http, %s) on %s:%d", _mode, _mcp_host, _mcp_port)
        mcp.run(transport="http", host=_mcp_host, port=_mcp_port, stateless_http=True, path="/mcp")
    else:
        mcp.run(transport="stdio")
