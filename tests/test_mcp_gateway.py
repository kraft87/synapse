"""End-to-end gateway behaviour against local stub upstreams (no network, no database).

Three real HTTP MCP servers run in-process: a stub Synapse (the REAL ``/auth/whoami`` route
and the REAL ``PgSkillsProvider`` over in-memory storage), a stub Exa (header-keyed) and a
stub Firecrawl (URL-keyed). The gateway is built from ``build_gateway`` exactly as
``python -m mcp_gateway`` builds it. Nothing here proves anything about the live Exa or
Firecrawl services — only about what the gateway does with whatever answers.
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
from mcp_gateway.config import GatewaySettings, ResearchUpstream

ROOT = "root-enrollment-token"
DEVICES = {
    "tok-alice-full": SurfaceTrust("dev-alice", "full", (), True),
    "tok-bob-work": SurfaceTrust("dev-bob", "restricted", ("proj-a",), True),
    "tok-carol-full": SurfaceTrust("dev-carol", "full", (), True),
}
EXA_KEY = "exa-test-key-111"
FIRECRAWL_KEY = "fc-test-key-222"
ALL_SECRETS = (EXA_KEY, FIRECRAWL_KEY, ROOT, *DEVICES)

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


def _stub_research(name: str, seen: list[dict[str, str]], slow_list: float = 0.0) -> FastMCP:
    class _Slow(Middleware):
        async def on_list_tools(self, context, call_next):
            await asyncio.sleep(slow_list)
            return await call_next(context)

    mw: list[Middleware] = [_Recorder(seen)] + ([_Slow()] if slow_list else [])
    mcp = FastMCP(f"stub-{name}", middleware=mw)

    if name == "exa":

        @mcp.tool()
        def web_search_exa(query: str, objective: str, numResults: int = 3) -> str:
            """Search the web."""
            if get_http_headers(include={"x-api-key"}).get("x-api-key") != EXA_KEY:
                raise ToolError("exa: bad key")
            return f"results for {query}"

    else:

        @mcp.tool()
        def firecrawl_scrape(url: str) -> str:
            """Scrape a page."""
            return f"# page {url}"

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
        self.exa_seen: list[dict[str, str]] = []
        self.firecrawl_seen: list[dict[str, str]] = []
        self.gateway_url = ""
        self.gateway: FastMCP | None = None

    def client(self, token: str | None) -> Client:
        return Client(StreamableHttpTransport(self.gateway_url, auth=token), timeout=10)


async def _start(
    stack: AsyncExitStack,
    *,
    exa_slow: float = 0.0,
    firecrawl_down: bool = False,
    research_trust: str = "full",
    skills_trust: str = "full",
    **overrides: Any,
) -> Stack:
    s = Stack()
    syn_url = await stack.enter_async_context(run_server_async(_stub_synapse(s.synapse_seen)))
    exa_url = await stack.enter_async_context(
        run_server_async(_stub_research("exa", s.exa_seen, exa_slow))
    )
    if firecrawl_down:
        fc_template = "http://127.0.0.1:9/{api_key}/mcp"  # discard port: connection refused
    else:
        fc_url = await stack.enter_async_context(
            run_server_async(
                _stub_research("firecrawl", s.firecrawl_seen), path=f"/{FIRECRAWL_KEY}/mcp"
            )
        )
        fc_template = fc_url.replace(FIRECRAWL_KEY, "{api_key}")
    settings = GatewaySettings(
        synapse_url=syn_url.removesuffix("/mcp"),
        research_trust=research_trust,
        skills_trust=skills_trust,
        research=(
            ResearchUpstream(
                "exa", exa_url, EXA_KEY, "x-api-key", discovery_timeout=1.0, call_timeout=10
            ),
            ResearchUpstream(
                "firecrawl", fc_template, FIRECRAWL_KEY, "", discovery_timeout=1.0, call_timeout=10
            ),
        ),
        **overrides,
    )
    s.gateway = build_gateway(settings)
    s.gateway_url = await stack.enter_async_context(run_server_async(s.gateway))
    return s


def _text(result) -> str:
    return result.content[0].text


# --------------------------------------------------------------------------- authentication


@pytest.mark.parametrize("token", [None, "not-a-token", ROOT])
async def test_unauthenticated_root_and_unknown_callers_are_rejected(token):
    async with AsyncExitStack() as stack:
        s = await _start(stack)
        with pytest.raises(Exception, match=r"401|Unauthorized"):
            async with s.client(token) as c:
                await c.list_tools()


async def test_revoked_device_loses_memory_on_its_next_call():
    """The identity cache never outlives Synapse for memory: every call re-authenticates."""
    async with AsyncExitStack() as stack:
        s = await _start(stack)
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


async def test_each_request_reaches_synapse_as_its_own_caller():
    async with AsyncExitStack() as stack:
        s = await _start(stack)
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


async def test_credentials_stay_with_their_own_upstream():
    async with AsyncExitStack() as stack:
        s = await _start(stack)
        async with s.client("tok-alice-full") as c:
            await c.call_tool("recall", {"query": "x"})
            await c.call_tool("exa_web_search_exa", {"query": "x", "objective": "x"})
            await c.call_tool("firecrawl_firecrawl_scrape", {"url": "https://example.org"})
        assert s.exa_seen and s.firecrawl_seen and s.synapse_seen

        def blob(seen: list[dict[str, str]]) -> str:
            return json.dumps(seen)

        # Research upstreams never see a Synapse bearer (or any Authorization at all) ...
        for seen in (s.exa_seen, s.firecrawl_seen):
            assert not any("authorization" in h for h in seen)
            assert not any(tok in blob(seen) for tok in (*DEVICES, ROOT))
        # ... Exa sees only its own key, Firecrawl's key rides only in its own URL path ...
        assert all(h.get("x-api-key") == EXA_KEY for h in s.exa_seen)
        assert FIRECRAWL_KEY not in blob(s.exa_seen) and EXA_KEY not in blob(s.firecrawl_seen)
        # ... and Synapse never sees a research key.
        assert EXA_KEY not in blob(s.synapse_seen) and FIRECRAWL_KEY not in blob(s.synapse_seen)


# --------------------------------------------------------------------------- restricted callers


async def test_restricted_caller_gets_no_research_tools_or_research_skill():
    async with AsyncExitStack() as stack:
        s = await _start(stack)
        async with s.client("tok-alice-full") as full:
            names = {t.name for t in await full.list_tools()}
            assert {"exa_web_search_exa", "firecrawl_firecrawl_scrape"} <= names
            uris = {str(r.uri) for r in await full.list_resources()}
            assert "skill://gateway-research/SKILL.md" in uris

        async with s.client("tok-bob-work") as work:
            names = {t.name for t in await work.list_tools()}
            assert not any(n.startswith(("exa_", "firecrawl_")) for n in names)
            assert "recall" in names
            with pytest.raises(ToolError, match="Unknown tool"):
                await work.call_tool("exa_web_search_exa", {"query": "x", "objective": "x"})
            with pytest.raises(McpError, match="Unknown resource"):
                await work.read_resource("skill://gateway-research/SKILL.md")
            with pytest.raises(ToolError):
                await work.call_tool("read_resource", {"uri": "skill://gateway-research/SKILL.md"})


PERSONAL_SKILL_URIS = (
    "skill://demo-skill/SKILL.md",
    "skill://demo-skill/_manifest",
    "skill://demo-skill/scripts/run.py",  # template path
)


async def test_restricted_device_cannot_reach_synapse_skills_by_any_path():
    """The upstream skills provider is permissive (see _stub_synapse); the gateway is the
    gate. Listing, templates, direct reads and the tools bridge must all come up empty —
    while the restricted device keeps its (Synapse-scoped) memory tools."""
    async with AsyncExitStack() as stack:
        s = await _start(stack)
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


async def test_skills_trust_can_be_widened_explicitly():
    async with AsyncExitStack() as stack:
        s = await _start(stack, skills_trust="restricted")
        async with s.client("tok-bob-work") as work:
            body = (await work.read_resource("skill://demo-skill/SKILL.md"))[0].text
            assert body == SKILLS["demo-skill"][1]
            # Widening skills does not widen research.
            uris = {str(r.uri) for r in await work.list_resources()}
            assert "skill://gateway-research/SKILL.md" not in uris


async def test_research_trust_can_be_widened_explicitly():
    async with AsyncExitStack() as stack:
        s = await _start(stack, research_trust="restricted")
        async with s.client("tok-bob-work") as work:
            assert "exa_web_search_exa" in {t.name for t in await work.list_tools()}


async def test_synapse_side_filtering_is_not_leaked_through_the_cache():
    async with AsyncExitStack() as stack:
        s = await _start(stack)
        async with s.client("tok-alice-full") as full:
            assert "full_only_note" in {t.name for t in await full.list_tools()}
            assert _text(await full.call_tool("full_only_note", {})) == "personal"
        async with s.client("tok-bob-work") as work:
            assert "full_only_note" not in {t.name for t in await work.list_tools()}
            with pytest.raises(ToolError, match="Unknown tool"):
                await work.call_tool("full_only_note", {})


# --------------------------------------------------------------------------- degradation


async def test_research_outage_does_not_take_memory_or_skills_down(caplog):
    # FastMCP's loggers do not propagate to root; attach caplog's handler directly so the
    # aggregate provider's "Error during list_tools" warnings are captured too.
    fastmcp_log = logging.getLogger("fastmcp")
    fastmcp_log.addHandler(caplog.handler)
    caplog.set_level(logging.DEBUG)
    try:
        await _outage_scenario()
    finally:
        fastmcp_log.removeHandler(caplog.handler)
    assert "upstream" in caplog.text  # the outage was logged ...
    for secret in ALL_SECRETS:  # ... without the keyed Firecrawl URL or any bearer
        assert secret not in caplog.text


async def _outage_scenario() -> None:
    async with AsyncExitStack() as stack:
        s = await _start(stack, exa_slow=5.0, firecrawl_down=True)
        async with s.client("tok-alice-full") as c:
            start = time.monotonic()
            names = {t.name for t in await c.list_tools()}
            elapsed = time.monotonic() - start
            assert "recall" in names and "read_resource" in names
            assert not any(n.startswith(("exa_", "firecrawl_")) for n in names)
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


async def test_tool_names_and_schemas_are_stable_and_preserved():
    async with AsyncExitStack() as stack:
        s = await _start(stack)
        async with Client(StreamableHttpTransport(s.gateway_url, auth="tok-alice-full")) as c:
            gw = {t.name: t for t in await c.list_tools()}
        # Memory tools keep their Synapse names (hooks match mcp__<server>__recall etc.);
        # research tools are namespaced; the hidden plumbing tool stays hidden.
        assert set(gw) == {
            "recall",
            "remember",
            "full_only_note",
            "broken",
            "exa_web_search_exa",
            "firecrawl_firecrawl_scrape",
            "list_resources",
            "read_resource",
        }
        upstream = {t.name: t for t in await _stub_synapse([]).list_tools()}
        assert gw["recall"].inputSchema == upstream["recall"].to_mcp_tool().inputSchema
        assert gw["recall"].outputSchema == upstream["recall"].to_mcp_tool().outputSchema
        assert gw["recall"].description == "Search memory."


async def test_upstream_tool_errors_pass_through():
    async with AsyncExitStack() as stack:
        s = await _start(stack)
        async with s.client("tok-alice-full") as c:
            r = await c.call_tool("broken", {}, raise_on_error=False)
            assert r.is_error and "recall index offline" in _text(r)
            r = await c.call_tool(
                "recall", {"limit": "not-a-number", "query": "q"}, raise_on_error=False
            )
            assert r.is_error


# --------------------------------------------------------------------------- skills


async def test_skill_resources_and_tools_bridge_end_to_end():
    async with AsyncExitStack() as stack:
        s = await _start(stack)
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

        async with s.client("tok-alice-full") as c:
            research = _text(
                await c.call_tool("read_resource", {"uri": "skill://gateway-research/SKILL.md"})
            )
            assert "name: gateway-research" in research
            # The skill names only tool prefixes / memory names the gateway really serves.
            names = {t.name for t in await c.list_tools()}
            for prefix in ("exa_", "firecrawl_"):
                assert f"`{prefix}*`" in research and any(n.startswith(prefix) for n in names)
            assert "`recall`" in research and "recall" in names


async def test_hidden_synapse_tools_are_not_reachable_by_name():
    """A tool Synapse leaves out of tools/list (issue_machine_token) is not forwarded."""
    async with AsyncExitStack() as stack:
        s = await _start(stack)
        async with s.client("tok-alice-full") as c:
            with pytest.raises(ToolError, match="Unknown tool"):
                await c.call_tool("issue_machine_token", {})


async def test_instructions_fit_claude_code_cap():
    async with AsyncExitStack() as stack:
        s = await _start(stack)
        async with s.client("tok-alice-full") as c:
            text = c.initialize_result.instructions or ""
        assert "skill://gateway-research/SKILL.md" in text
        assert len(text.encode()) <= 2048


async def test_probe_reports_surface_without_printing_the_token(capsys):
    from mcp_gateway.probe import _probe

    async with AsyncExitStack() as stack:
        s = await _start(stack)
        assert await _probe(s.gateway_url, "tok-alice-full", "gateway pilot") == 0
        full = capsys.readouterr().out
        assert await _probe(s.gateway_url, "tok-bob-work", None) == 0
        work = capsys.readouterr().out
    assert "tools[exa]: exa_web_search_exa" in full
    assert "resources/read == read_resource tool: True" in full
    assert "search via exa_web_search_exa (args: objective, query): ok" in full
    assert "tools[exa]" not in work and "not served to this device" in work
    assert "tools[memory+bridge]: broken, list_resources, read_resource, recall, remember" in work
    assert "skill resources: 0" in work
    assert not any(tok in full + work for tok in DEVICES)


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


async def test_existing_hooks_see_gateway_memory_tools_like_direct_ones():
    """Session injection, the remember spool and the feedback nudge key on tool names.
    Through a gateway named anything, each hook must fire for exactly the memory tools it
    fires for on the plugin's own direct connection."""
    patterns = _hook_patterns()
    assert len(patterns) >= 7
    async with AsyncExitStack() as stack:
        s = await _start(stack)
        async with s.client("tok-alice-full") as c:
            gateway = {t.name for t in await c.list_tools()}
    direct = {t.name for t in await _stub_synapse([]).list_tools()} - {"issue_machine_token"}
    for pat in patterns:
        via_plugin = {n for n in direct if re.search(pat, f"mcp__plugin_synapse_synapse__{n}")}
        via_gateway = {n for n in gateway if re.search(pat, f"mcp__synapse-gateway__{n}")}
        assert via_gateway == via_plugin, pat
        assert via_plugin, f"{pat} matches no memory tool"
