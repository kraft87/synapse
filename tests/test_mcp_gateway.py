"""End-to-end gateway behaviour against local stub upstreams (no network, no database).

Real HTTP MCP servers run in-process: a stub Synapse (the REAL ``/auth/whoami`` route and
the REAL ``PgSkillsProvider`` over in-memory storage) and two fictional deployment services
named only in a registry file — a project ``tracker`` (bearer secret from an env var, full
trust) and a team ``wiki`` (secret in its URL path from a secret file, restricted trust).
The gateway is built from ``load_settings`` + ``build_gateway`` exactly as
``python -m mcp_gateway`` builds it. Nothing here is specific to any real third-party service.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client, FastMCP
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.exceptions import AuthorizationError, ToolError
from fastmcp.server.auth.auth import AccessToken, TokenVerifier
from fastmcp.server.dependencies import get_access_token, get_http_headers
from fastmcp.server.middleware import Middleware
from fastmcp.utilities.tests import run_server_async
from mcp.shared.exceptions import McpError
from pydantic import BaseModel

import mcp_server.skills_provider as skills_mod
import mcp_server.whoami_route as whoami_mod
from ingestion.surfaces import SurfaceTrust, token_hash
from mcp_gateway.app import build_gateway
from mcp_gateway.config import load_settings

ROOT = "root-enrollment-token"
DEVICES = {
    "tok-alice-full": SurfaceTrust("dev-alice", "full", (), True),
    "tok-bob-work": SurfaceTrust("dev-bob", "restricted", ("proj-a",), True),
    "tok-carol-full": SurfaceTrust("dev-carol", "full", (), True),
}
TRACKER_SECRET = "tracker-secret-111"
WIKI_SECRET = "wiki-secret-222"
ALL_SECRETS = (TRACKER_SECRET, WIKI_SECRET, ROOT, *DEVICES)

SKILLS = {"demo-skill": ("Demo skill", "---\nname: demo-skill\ndescription: d\n---\n# Demo\n")}
SKILL_FILES = {("demo-skill", "scripts/run.py"): b"print('hi')\n"}


# --------------------------------------------------------------------------- stubs


class _DeviceVerifier(TokenVerifier):
    """Stub of Synapse's verifier: root token + approved device tokens."""

    async def verify_token(self, token: str) -> AccessToken | None:
        if token == ROOT:
            return AccessToken(
                token=token, client_id="synapse-machine", scopes=[], claims={"kind": "root"}
            )
        st = DEVICES.get(token)
        if st is None:
            return None
        return AccessToken(
            token=token,
            client_id="synapse-device",
            scopes=[],
            claims={"kind": "device", "surface_id": st.surface_id, "trust": st.trust},
        )


class _Recorder(Middleware):
    """Record the raw inbound headers of every MCP request an upstream receives."""

    def __init__(self, seen: list[dict[str, str]]) -> None:
        self.seen = seen

    async def on_request(self, context, call_next):
        self.seen.append(get_http_headers(include_all=True))
        return await call_next(context)


class _FullOnly(Middleware):
    """Synapse-side policy stand-in: `full_only_note` exists only for full-trust devices."""

    @staticmethod
    def _full() -> bool:
        tok = get_access_token()
        return bool(tok and (tok.claims or {}).get("trust") == "full")

    async def on_list_tools(self, context, call_next):
        tools = await call_next(context)
        # issue_machine_token: hidden from every listing yet callable by name, exactly like
        # the real server's _HiddenToolsList.
        return [
            t
            for t in tools
            if t.name != "issue_machine_token" and (t.name != "full_only_note" or self._full())
        ]

    async def on_call_tool(self, context, call_next):
        if context.message.name == "full_only_note" and not self._full():
            raise AuthorizationError("restricted")
        return await call_next(context)


class RecallHit(BaseModel):
    caller: str
    query: str


def _stub_synapse(seen: list[dict[str, str]]) -> FastMCP:
    mcp = FastMCP("stub-synapse", auth=_DeviceVerifier(), middleware=[_Recorder(seen), _FullOnly()])

    @mcp.tool()
    async def recall(query: str, limit: int = 5) -> RecallHit:
        """Search memory."""
        await asyncio.sleep(0.01)  # widen the interleaving window for the concurrency test
        tok = get_access_token()
        return RecallHit(caller=(tok.claims or {}).get("surface_id") or "root", query=query)

    @mcp.tool()
    def fetch(ids: list[str]) -> str:
        """Expand ids."""
        return ",".join(ids)

    @mcp.tool()
    def remember(content: str) -> str:
        """Store a memory."""
        return "stored"

    @mcp.tool()
    def full_only_note() -> str:
        """Visible to full-trust devices only."""
        return "personal"

    @mcp.tool()
    def broken() -> str:
        """Always fails with a domain error."""
        raise ToolError("recall index offline")

    @mcp.tool()
    def issue_machine_token() -> str:
        """Hidden plumbing."""
        return ROOT

    # The REAL provider, and as permissive as the real one: every active skill to ANY
    # authenticated bearer, restricted devices included. The gateway must do the withholding.
    mcp.add_provider(skills_mod.PgSkillsProvider("postgresql://unused"))
    whoami_mod.register(mcp, "postgresql://unused", ROOT)
    return mcp


def _slow_listing(seconds: float) -> list[Middleware]:
    class _Slow(Middleware):
        async def on_list_tools(self, context, call_next):
            await asyncio.sleep(seconds)
            return await call_next(context)

    return [_Slow()] if seconds else []


def _stub_tracker(seen: list[dict[str, str]], slow_list: float = 0.0) -> FastMCP:
    """A fictional project tracker that demands its own bearer secret."""
    mcp = FastMCP("stub-tracker", middleware=[_Recorder(seen), *_slow_listing(slow_list)])

    def _authorized() -> None:
        auth = get_http_headers(include={"authorization"}).get("authorization")
        if auth != f"Bearer {TRACKER_SECRET}":
            raise ToolError("tracker: bad credential")

    @mcp.tool()
    def list_issues(project: str, state: str = "open") -> list[str]:
        """List issues in a project."""
        _authorized()
        return [f"{project}-1", f"{project}-2"]

    @mcp.tool()
    def create_issue(project: str, title: str) -> str:
        """Create an issue."""
        _authorized()
        return f"{project}-3"

    return mcp


def _stub_wiki(seen: list[dict[str, str]]) -> FastMCP:
    """A fictional team wiki; its secret is part of its URL path (see _start)."""
    mcp = FastMCP("stub-wiki", middleware=[_Recorder(seen)])

    @mcp.tool()
    def get_page(slug: str) -> str:
        """Fetch a wiki page."""
        return f"# {slug}"

    @mcp.resource("docs://home")
    def home() -> str:
        return "wiki home"

    return mcp


@pytest.fixture(autouse=True)
def _in_memory_skills(monkeypatch):
    monkeypatch.setattr(
        skills_mod, "_fetch_active", lambda db: sorted((n, d) for n, (d, _) in SKILLS.items())
    )
    monkeypatch.setattr(skills_mod, "_skill_active", lambda db, n: n in SKILLS)
    monkeypatch.setattr(
        skills_mod, "_fetch_body", lambda db, n: SKILLS[n][1] if n in SKILLS else None
    )
    monkeypatch.setattr(
        skills_mod,
        "_fetch_files_meta",
        lambda db, n: [(p, len(b), "0" * 64) for (s, p), b in SKILL_FILES.items() if s == n],
    )
    monkeypatch.setattr(skills_mod, "_fetch_file", lambda db, n, p: SKILL_FILES.get((n, p)))

    def resolve(db_url, *, token_hash_hex=None, legacy_surface_id=None):
        for tok, st in DEVICES.items():
            if token_hash_hex == token_hash(tok):
                return st
        return SurfaceTrust()

    monkeypatch.setattr(whoami_mod, "resolve_caller", resolve)


class Stack:
    def __init__(self) -> None:
        self.synapse_seen: list[dict[str, str]] = []
        self.tracker_seen: list[dict[str, str]] = []
        self.wiki_seen: list[dict[str, str]] = []
        self.gateway_url = ""
        self.gateway: FastMCP | None = None

    def client(self, token: str | None) -> Client:
        return Client(StreamableHttpTransport(self.gateway_url, auth=token), timeout=10)


async def _start(
    stack: AsyncExitStack,
    tmp_path: Path,
    *,
    upstreams: bool = True,
    tracker_slow: float = 0.0,
    wiki_down: bool = False,
    skills_trust: str = "full",
    skills_dirs: list[dict[str, Any]] | None = None,
) -> Stack:
    """Start stub Synapse (+ tracker and wiki), write a registry file, build the gateway."""
    s = Stack()
    syn_url = await stack.enter_async_context(run_server_async(_stub_synapse(s.synapse_seen)))
    registry: dict[str, Any] = {"upstreams": [], "skills_dirs": skills_dirs or []}
    if upstreams:
        tracker_url = await stack.enter_async_context(
            run_server_async(_stub_tracker(s.tracker_seen, tracker_slow))
        )
        if wiki_down:
            wiki_template = "http://127.0.0.1:9/{secret}/mcp"  # discard port: refused
        else:
            wiki_url = await stack.enter_async_context(
                run_server_async(_stub_wiki(s.wiki_seen), path=f"/{WIKI_SECRET}/mcp")
            )
            wiki_template = wiki_url.replace(WIKI_SECRET, "{secret}")
        (tmp_path / "wiki.secret").write_text(WIKI_SECRET + "\n")
        registry["upstreams"] = [
            {
                "namespace": "tracker",
                "description": "Project tracker: issues",
                "url": tracker_url,
                "auth": {"type": "bearer", "secret_env": "TRACKER_MCP_TOKEN"},
                "min_trust": "full",
                "discovery_timeout": 1.0,
                "call_timeout": 10,
            },
            {
                "namespace": "wiki",
                "url": wiki_template,
                "auth": {"type": "url", "secret_file": "wiki.secret"},
                "min_trust": "restricted",
                "discovery_timeout": 1.0,
                "call_timeout": 10,
            },
        ]
    config = tmp_path / "gateway.json"
    config.write_text(json.dumps(registry))
    settings = load_settings(
        {
            "SYNAPSE_GATEWAY_SYNAPSE_URL": syn_url,
            "SYNAPSE_GATEWAY_CONFIG_FILE": str(config),
            "SYNAPSE_GATEWAY_SKILLS_TRUST": skills_trust,
            "TRACKER_MCP_TOKEN": TRACKER_SECRET,
        }
    )
    s.gateway = build_gateway(settings)
    s.gateway_url = await stack.enter_async_context(run_server_async(s.gateway))
    return s


def _text(result) -> str:
    return result.content[0].text


# --------------------------------------------------------------------------- authentication

CORE = {
    "recall",
    "fetch",
    "remember",
    "full_only_note",
    "broken",
    "list_resources",
    "read_resource",
}
TRACKER_TOOLS = {"tracker_list_issues", "tracker_create_issue"}
WIKI_TOOLS = {"wiki_get_page"}


@pytest.mark.parametrize("token", [None, "not-a-token", ROOT])
async def test_unauthenticated_root_and_unknown_callers_are_rejected(tmp_path, token):
    async with AsyncExitStack() as stack:
        s = await _start(stack, tmp_path)
        with pytest.raises(Exception, match=r"401|Unauthorized"):
            async with s.client(token) as c:
                await c.list_tools()
        # Nothing reached a configured upstream on behalf of a refused caller.
        assert not s.tracker_seen and not s.wiki_seen


async def test_revoked_device_loses_memory_on_its_next_call(tmp_path):
    """The identity cache never outlives Synapse for memory: every call re-authenticates."""
    async with AsyncExitStack() as stack:
        s = await _start(stack, tmp_path)
        async with s.client("tok-carol-full") as c:
            assert (await c.call_tool("recall", {"query": "q"})).structured_content[
                "caller"
            ] == "dev-carol"
            revoked = DEVICES.pop("tok-carol-full")
            try:
                with pytest.raises(ToolError):
                    await c.call_tool("recall", {"query": "q"})
            finally:
                DEVICES["tok-carol-full"] = revoked


# --------------------------------------------------------------------------- identity


async def test_each_request_reaches_synapse_as_its_own_caller(tmp_path):
    async with AsyncExitStack() as stack:
        s = await _start(stack, tmp_path)
        tokens = ["tok-alice-full", "tok-bob-work", "tok-carol-full"] * 6

        async def one(tok: str, i: int) -> tuple[str, str]:
            async with s.client(tok) as c:
                r = await c.call_tool("recall", {"query": f"q{i}"})
            return tok, r.structured_content["caller"]

        results = await asyncio.gather(*(one(t, i) for i, t in enumerate(tokens)))
        for tok, caller in results:
            assert caller == DEVICES[tok].surface_id

        seen_bearers = {h.get("authorization") for h in s.synapse_seen}
        assert seen_bearers == {f"Bearer {t}" for t in DEVICES}


async def test_credentials_stay_with_their_own_upstream(tmp_path):
    async with AsyncExitStack() as stack:
        s = await _start(stack, tmp_path)
        async with s.client("tok-alice-full") as c:
            await c.call_tool("recall", {"query": "x"})
            listed = await c.call_tool("tracker_list_issues", {"project": "web"})
            assert listed.structured_content == {"result": ["web-1", "web-2"]}
            assert _text(await c.call_tool("wiki_get_page", {"slug": "home"})) == "# home"
        assert s.tracker_seen and s.wiki_seen and s.synapse_seen

        def blob(seen: list[dict[str, str]]) -> str:
            return json.dumps(seen)

        # The tracker sees exactly its own configured bearer, never a caller's ...
        assert {h.get("authorization") for h in s.tracker_seen} == {f"Bearer {TRACKER_SECRET}"}
        # ... the wiki sees no Authorization at all (its secret rides only in its URL path) ...
        assert not any("authorization" in h for h in s.wiki_seen)
        for seen in (s.tracker_seen, s.wiki_seen):
            assert not any(tok in blob(seen) for tok in (*DEVICES, ROOT))
        assert WIKI_SECRET not in blob(s.tracker_seen) and TRACKER_SECRET not in blob(s.wiki_seen)
        # ... and Synapse never sees an upstream secret.
        assert TRACKER_SECRET not in blob(s.synapse_seen) and WIKI_SECRET not in blob(
            s.synapse_seen
        )


# --------------------------------------------------------------------------- restricted callers


async def test_min_trust_is_enforced_per_upstream(tmp_path):
    """tracker requires full trust, wiki admits restricted devices: each device sees exactly
    the namespaces it qualifies for, and cannot call the others by name."""
    async with AsyncExitStack() as stack:
        s = await _start(stack, tmp_path)
        async with s.client("tok-alice-full") as full:
            assert {t.name for t in await full.list_tools()} == CORE | TRACKER_TOOLS | WIKI_TOOLS
            assert "docs://wiki/home" in {str(r.uri) for r in await full.list_resources()}

        tracker_calls_before = len(s.tracker_seen)
        async with s.client("tok-bob-work") as work:
            assert {t.name for t in await work.list_tools()} == (
                CORE - {"full_only_note"}
            ) | WIKI_TOOLS
            with pytest.raises(ToolError, match="Unknown tool"):
                await work.call_tool("tracker_create_issue", {"project": "web", "title": "x"})
            assert _text(await work.call_tool("wiki_get_page", {"slug": "a"})) == "# a"
            # Namespaced upstream resources follow the same per-upstream trust.
            assert (await work.read_resource("docs://wiki/home"))[0].text == "wiki home"
        # The refused tracker call never reached the tracker.
        assert len(s.tracker_seen) == tracker_calls_before


PERSONAL_SKILL_URIS = (
    "skill://demo-skill/SKILL.md",
    "skill://demo-skill/_manifest",
    "skill://demo-skill/scripts/run.py",  # template path
)


async def test_restricted_device_cannot_reach_synapse_skills_by_any_path(tmp_path):
    """The upstream skills provider is permissive (see _stub_synapse); the gateway is the
    gate. Listing, templates, direct reads and the tools bridge must all come up empty —
    while the restricted device keeps its (Synapse-scoped) memory tools."""
    async with AsyncExitStack() as stack:
        s = await _start(stack, tmp_path)
        async with s.client("tok-bob-work") as work:
            assert not [r for r in await work.list_resources() if str(r.uri).startswith("skill://")]
            assert not await work.list_resource_templates()
            bridged = json.loads(_text(await work.call_tool("list_resources", {})))
            assert not [
                r for r in bridged if "skill://" in (r.get("uri") or r.get("uri_template") or "")
            ]
            for uri in PERSONAL_SKILL_URIS:
                with pytest.raises(McpError, match="Unknown resource"):
                    await work.read_resource(uri)
                r = await work.call_tool("read_resource", {"uri": uri}, raise_on_error=False)
                assert r.is_error and "Demo" not in _text(r) and "print" not in _text(r)
            hit = await work.call_tool("recall", {"query": "q"})
            assert hit.structured_content["caller"] == "dev-bob"
        # A full-trust device right after sees them: nothing cached across identities.
        async with s.client("tok-alice-full") as full:
            assert _text(await full.call_tool("read_resource", {"uri": PERSONAL_SKILL_URIS[0]}))
        # And the restricted device is still refused after the full one warmed any cache.
        async with s.client("tok-bob-work") as work:
            with pytest.raises(McpError, match="Unknown resource"):
                await work.read_resource(PERSONAL_SKILL_URIS[0])


async def test_skills_trust_can_be_widened_explicitly(tmp_path):
    async with AsyncExitStack() as stack:
        s = await _start(stack, tmp_path, skills_trust="restricted")
        async with s.client("tok-bob-work") as work:
            body = (await work.read_resource("skill://demo-skill/SKILL.md"))[0].text
            assert body == SKILLS["demo-skill"][1]
            # Widening skills does not widen a full-trust upstream.
            assert not TRACKER_TOOLS & {t.name for t in await work.list_tools()}


async def test_configured_skills_dir_uses_native_provider_and_trust_guard(tmp_path):
    team = tmp_path / "team-skills" / "release-checklist"
    team.mkdir(parents=True)
    (team / "SKILL.md").write_text(
        "---\nname: release-checklist\ndescription: Steps before tagging a release\n---\n# Go\n"
    )
    uri = "skill://release-checklist/SKILL.md"
    async with AsyncExitStack() as stack:
        s = await _start(
            stack, tmp_path, skills_dirs=[{"path": "team-skills", "min_trust": "full"}]
        )
        async with s.client("tok-alice-full") as full:
            assert uri in {str(r.uri) for r in await full.list_resources()}
            assert "# Go" in _text(await full.call_tool("read_resource", {"uri": uri}))
        async with s.client("tok-bob-work") as work:
            assert uri not in {str(r.uri) for r in await work.list_resources()}
            with pytest.raises(McpError, match="Unknown resource"):
                await work.read_resource(uri)


async def test_synapse_side_filtering_is_not_leaked_through_the_cache(tmp_path):
    async with AsyncExitStack() as stack:
        s = await _start(stack, tmp_path)
        async with s.client("tok-alice-full") as full:
            assert "full_only_note" in {t.name for t in await full.list_tools()}
            assert _text(await full.call_tool("full_only_note", {})) == "personal"
        async with s.client("tok-bob-work") as work:
            assert "full_only_note" not in {t.name for t in await work.list_tools()}
            with pytest.raises(ToolError, match="Unknown tool"):
                await work.call_tool("full_only_note", {})


# --------------------------------------------------------------------------- core only


async def test_core_only_config_is_synapse_memory_and_skills(tmp_path):
    """No configured upstreams: the gateway is exactly Synapse memory + skills + bridge."""
    async with AsyncExitStack() as stack:
        s = await _start(stack, tmp_path, upstreams=False)
        async with s.client("tok-alice-full") as c:
            assert {t.name for t in await c.list_tools()} == CORE
            assert "skill://demo-skill/SKILL.md" in {str(r.uri) for r in await c.list_resources()}
            text = c.initialize_result.instructions or ""
        assert "recall" in text and "skill://<name>/SKILL.md" in text
        assert "namespace" not in text  # no services advertised when none are configured


async def test_core_only_without_any_config_file(tmp_path, monkeypatch):
    async with AsyncExitStack() as stack:
        syn_url = await stack.enter_async_context(run_server_async(_stub_synapse([])))
        settings = load_settings({"SYNAPSE_GATEWAY_SYNAPSE_URL": syn_url})
        assert settings.registry.upstreams == () and settings.registry.skills_dirs == ()
        url = await stack.enter_async_context(run_server_async(build_gateway(settings)))
        async with Client(StreamableHttpTransport(url, auth="tok-alice-full")) as c:
            assert {t.name for t in await c.list_tools()} == CORE


# --------------------------------------------------------------------------- degradation


async def test_upstream_outage_does_not_take_memory_or_skills_down(tmp_path, caplog):
    # FastMCP's loggers do not propagate to root; attach caplog's handler directly so the
    # aggregate provider's "Error during list_tools" warnings are captured too.
    fastmcp_log = logging.getLogger("fastmcp")
    fastmcp_log.addHandler(caplog.handler)
    caplog.set_level(logging.DEBUG)
    try:
        await _outage_scenario(tmp_path)
    finally:
        fastmcp_log.removeHandler(caplog.handler)
    assert "upstream" in caplog.text  # the outage was logged ...
    for secret in ALL_SECRETS:  # ... without the keyed wiki URL or any bearer
        assert secret not in caplog.text


async def _outage_scenario(tmp_path: Path) -> None:
    async with AsyncExitStack() as stack:
        s = await _start(stack, tmp_path, tracker_slow=5.0, wiki_down=True)
        async with s.client("tok-alice-full") as c:
            start = time.monotonic()
            names = {t.name for t in await c.list_tools()}
            elapsed = time.monotonic() - start
            assert names == CORE
            assert elapsed < 3.0, f"discovery not bounded: {elapsed:.1f}s"
            r = await c.call_tool("recall", {"query": "still here"})
            assert r.structured_content["caller"] == "dev-alice"
            uris = {str(r.uri) for r in await c.list_resources()}
            assert "skill://demo-skill/SKILL.md" in uris
            # Backoff: the next listing does not wait on the dead upstreams again.
            start = time.monotonic()
            await c.list_tools()
            assert time.monotonic() - start < 1.0


# --------------------------------------------------------------------------- surface


async def test_tool_names_and_schemas_are_stable_and_preserved(tmp_path):
    async with AsyncExitStack() as stack:
        s = await _start(stack, tmp_path)
        async with Client(StreamableHttpTransport(s.gateway_url, auth="tok-alice-full")) as c:
            gw = {t.name: t for t in await c.list_tools()}
        # Memory tools keep their Synapse names (hooks match mcp__<server>__recall etc.);
        # configured upstreams are namespaced; the hidden plumbing tool stays hidden.
        assert set(gw) == CORE | TRACKER_TOOLS | WIKI_TOOLS
        upstream = {t.name: t.to_mcp_tool() for t in await _stub_synapse([]).list_tools()}
        assert gw["recall"].inputSchema == upstream["recall"].inputSchema
        assert gw["recall"].outputSchema == upstream["recall"].outputSchema
        assert gw["recall"].description == "Search memory."
        tracker = {t.name: t.to_mcp_tool() for t in await _stub_tracker([]).list_tools()}
        for name, tool in tracker.items():
            mine = gw[f"tracker_{name}"]
            assert (mine.inputSchema, mine.outputSchema, mine.description) == (
                tool.inputSchema,
                tool.outputSchema,
                tool.description,
            )


async def test_upstream_tool_errors_pass_through(tmp_path):
    async with AsyncExitStack() as stack:
        s = await _start(stack, tmp_path)
        async with s.client("tok-alice-full") as c:
            r = await c.call_tool("broken", {}, raise_on_error=False)
            assert r.is_error and "recall index offline" in _text(r)
            r = await c.call_tool(
                "recall", {"limit": "not-a-number", "query": "q"}, raise_on_error=False
            )
            assert r.is_error
            r = await c.call_tool("tracker_list_issues", {}, raise_on_error=False)
            assert r.is_error  # upstream schema validation error, passed through


# --------------------------------------------------------------------------- skills


async def test_skill_resources_and_tools_bridge_end_to_end(tmp_path):
    async with AsyncExitStack() as stack:
        s = await _start(stack, tmp_path)
        async with s.client("tok-alice-full") as c:
            uris = {str(r.uri) for r in await c.list_resources()}
            assert {"skill://demo-skill/SKILL.md", "skill://demo-skill/_manifest"} <= uris

            body = (await c.read_resource("skill://demo-skill/SKILL.md"))[0].text
            assert body == SKILLS["demo-skill"][1]
            manifest = json.loads((await c.read_resource("skill://demo-skill/_manifest"))[0].text)
            assert {f["path"] for f in manifest["files"]} == {"SKILL.md", "scripts/run.py"}
            nested = (await c.read_resource("skill://demo-skill/scripts/run.py"))[0].text
            assert nested == "print('hi')\n"

            listed = json.loads(_text(await c.call_tool("list_resources", {})))
            assert "skill://demo-skill/SKILL.md" in {r.get("uri") for r in listed}
            assert (
                _text(await c.call_tool("read_resource", {"uri": "skill://demo-skill/SKILL.md"}))
                == body
            )
            assert (
                _text(
                    await c.call_tool("read_resource", {"uri": "skill://demo-skill/scripts/run.py"})
                )
                == "print('hi')\n"
            )


async def test_no_bundled_skill_or_workflow(tmp_path):
    """The gateway ships no skill of its own: only Synapse's (and configured dirs')."""
    async with AsyncExitStack() as stack:
        s = await _start(stack, tmp_path)
        async with s.client("tok-alice-full") as c:
            skills = {
                str(r.uri) for r in await c.list_resources() if str(r.uri).startswith("skill://")
            }
    assert skills == {"skill://demo-skill/SKILL.md", "skill://demo-skill/_manifest"}


async def test_hidden_synapse_tools_are_not_reachable_by_name(tmp_path):
    """A tool Synapse leaves out of tools/list (issue_machine_token) is not forwarded."""
    async with AsyncExitStack() as stack:
        s = await _start(stack, tmp_path)
        async with s.client("tok-alice-full") as c:
            with pytest.raises(ToolError, match="Unknown tool"):
                await c.call_tool("issue_machine_token", {})


async def test_instructions_describe_memory_services_and_skill_discovery(tmp_path):
    async with AsyncExitStack() as stack:
        s = await _start(stack, tmp_path)
        async with s.client("tok-alice-full") as c:
            text = c.initialize_result.instructions or ""
    assert "call recall" in text and "skill://<name>/SKILL.md" in text
    assert "tracker_* (Project tracker: issues)" in text and "wiki_*" in text
    assert "http" not in text  # namespaces and descriptions only, never URLs
    assert len(text.encode()) <= 2048


async def test_probe_reports_structure_without_printing_the_token(tmp_path, capsys):
    from mcp_gateway.probe import _probe

    async with AsyncExitStack() as stack:
        s = await _start(stack, tmp_path)
        call = ("tracker_list_issues", {"project": "web"})
        assert await _probe(s.gateway_url, "tok-alice-full", ["tracker", "wiki"], call) == 0
        full = capsys.readouterr().out
        # A restricted device does not see the tracker; expecting it fails the probe.
        assert await _probe(s.gateway_url, "tok-bob-work", ["tracker"], None) == 1
        work = capsys.readouterr().out
    assert "core tools: ok" in full and "namespace tracker: ok" in full
    assert "tools[tracker]: 2" in full and "tools[wiki]: 1" in full
    assert "skill read (native == read_resource tool): True" in full
    assert "call tracker_list_issues: ok" in full
    assert "namespace tracker: NOT LISTED" in work and "skill resources: 0" in work
    assert "web-1" not in full  # results are sized, never printed
    assert not any(tok in full + work for tok in (*DEVICES, TRACKER_SECRET, WIKI_SECRET))


# --------------------------------------------------------------------------- hook compatibility

_ROOT = Path(__file__).resolve().parents[1]


def _hook_patterns() -> list[str]:
    """Every tool-name regex the Claude and Codex Synapse hooks use."""
    pats = re.findall(r'"matcher": "(mcp__[^"]+)"', (_ROOT / "plugin/hooks/hooks.json").read_text())
    pats += re.findall(r'matcher = "(mcp__[^"]+)"', (_ROOT / "plugin-codex/install.py").read_text())
    for script in (
        "plugin/scripts/self_session_inject.py",
        "plugin-codex/hooks/pre_tool_use.py",
        "plugin-codex/hooks/post_tool_use.py",
    ):
        pats += re.findall(r'r"(\^mcp__[^"]+)"', (_ROOT / script).read_text())
    return pats


async def test_existing_hooks_see_gateway_memory_tools_like_direct_ones(tmp_path):
    """Session injection, the remember spool and the feedback nudge key on tool names.
    Through a gateway named anything, each hook must fire for exactly the memory tools it
    fires for on the plugin's own direct connection."""
    patterns = _hook_patterns()
    assert len(patterns) >= 7
    async with AsyncExitStack() as stack:
        s = await _start(stack, tmp_path)
        async with s.client("tok-alice-full") as c:
            gateway = {t.name for t in await c.list_tools()}
    direct = {t.name for t in await _stub_synapse([]).list_tools()} - {"issue_machine_token"}
    for pat in patterns:
        via_plugin = {n for n in direct if re.search(pat, f"mcp__plugin_synapse_synapse__{n}")}
        via_gateway = {n for n in gateway if re.search(pat, f"mcp__synapse-gateway__{n}")}
        assert via_gateway == via_plugin, pat
        assert via_plugin, f"{pat} matches no memory tool"
