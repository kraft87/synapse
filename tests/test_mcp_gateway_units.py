"""Unit coverage for the gateway's config, redaction and error hygiene, and /auth/whoami."""

from __future__ import annotations

import asyncio
import logging

import httpx
import pytest
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData
from starlette.testclient import TestClient

import mcp_server.whoami_route as whoami_mod
from ingestion.surfaces import SurfaceTrust, token_hash
from mcp_gateway.config import (
    ConfigError,
    SecretRedactor,
    install_log_redaction,
    load_settings,
    redact_url,
)
from mcp_gateway.upstream import _guard

P = "SYNAPSE_GATEWAY_"

# --------------------------------------------------------------------------- config


def test_defaults_enable_no_research_upstream():
    s = load_settings({})
    assert s.research == ()
    assert s.synapse_mcp_url == "http://127.0.0.1:8765/mcp"
    assert s.whoami_url == "http://127.0.0.1:8765/auth/whoami"
    assert s.research_trust == "full"
    assert s.skills_trust == "full"  # Synapse skills withheld from restricted devices by default
    assert (s.host, s.port) == ("127.0.0.1", 8766)


def test_synapse_url_may_be_given_with_mcp_suffix():
    assert (
        load_settings({P + "SYNAPSE_URL": "https://syn.example/mcp"}).synapse_url
        == "https://syn.example"
    )


def test_keyless_upstream_needs_only_a_url():
    (exa,) = load_settings({P + "EXA_URL": "https://exa.example/mcp"}).research
    assert exa.url() == "https://exa.example/mcp" and exa.headers() == {}
    assert load_settings({P + "EXA_URL": "https://exa.example/mcp"}).secrets() == []


def test_url_keyed_upstream_substitutes_quoted_key():
    s = load_settings(
        {P + "FIRECRAWL_URL": "https://fc.example/{api_key}/mcp", P + "FIRECRAWL_API_KEY": "a/b c"}
    )
    (fc,) = s.research
    assert fc.url() == "https://fc.example/a%2Fb%20c/mcp"
    assert fc.headers() == {}
    assert "a/b" not in repr(fc) and fc.display_url == "https://fc.example/…"


def test_header_keyed_upstream_and_authorization_bearer():
    s = load_settings(
        {
            P + "EXA_URL": "https://exa.example/mcp",
            P + "EXA_API_KEY": "k1",
            P + "EXA_AUTH_HEADER": "x-api-key",
            P + "FIRECRAWL_URL": "https://fc.example/mcp",
            P + "FIRECRAWL_API_KEY": "k2",
            P + "FIRECRAWL_AUTH_HEADER": "Authorization",
        }
    )
    exa, fc = s.research
    assert exa.headers() == {"x-api-key": "k1"} and exa.url() == "https://exa.example/mcp"
    assert fc.headers() == {"Authorization": "Bearer k2"}
    assert s.secrets() == ["k1", "k2"]


def test_secret_file(tmp_path):
    f = tmp_path / "exa.key"
    f.write_text("from-file\n")
    s = load_settings(
        {P + "EXA_URL": "https://e.example/mcp?k={api_key}", P + "EXA_API_KEY_FILE": str(f)}
    )
    assert s.research[0].url() == "https://e.example/mcp?k=from-file"


@pytest.mark.parametrize(
    "env, match",
    [
        ({P + "EXA_URL": "https://e/mcp", P + "EXA_API_KEY": "k"}, "never be sent"),
        ({P + "EXA_URL": "https://e/{api_key}"}, "required"),
        (
            {
                P + "EXA_URL": "https://e/{api_key}",
                P + "EXA_API_KEY": "k",
                P + "EXA_AUTH_HEADER": "x",
            },
            "not both",
        ),
        ({P + "EXA_URL": "ftp://e"}, "http"),
        (
            {
                P + "EXA_URL": "https://e/{api_key}",
                P + "EXA_API_KEY": "k",
                P + "EXA_API_KEY_FILE": "/x",
            },
            "only one",
        ),
        ({P + "RESEARCH_TRUST": "everyone"}, "RESEARCH_TRUST"),
        ({P + "SKILLS_TRUST": "anyone"}, "SKILLS_TRUST"),
        ({P + "DISCOVERY_TIMEOUT": "-1"}, ">= 0"),
    ],
)
def test_invalid_config_is_refused_at_startup(env, match):
    with pytest.raises(ConfigError, match=match):
        load_settings(env)


def test_config_errors_never_echo_the_key():
    with pytest.raises(ConfigError) as e:
        load_settings({P + "EXA_URL": "https://e/mcp", P + "EXA_API_KEY": "super-secret-value"})
    assert "super-secret-value" not in str(e.value)


def test_redact_url_keeps_only_origin():
    assert (
        redact_url("https://mcp.example.dev/KEY123/v2/mcp?x=KEY123") == "https://mcp.example.dev/…"
    )
    assert redact_url("http://127.0.0.1:8765/mcp") == "http://127.0.0.1:8765/…"
    assert redact_url("not a url") == "<invalid url>"


# --------------------------------------------------------------------------- redaction


def test_redactor_scrubs_raw_and_quoted_forms():
    r = SecretRedactor(["k/1", ""])
    assert r("url https://x/k%2F1/mcp and k/1") == "url https://x/***/mcp and ***"


def test_log_redaction_covers_messages_and_tracebacks(monkeypatch):
    import mcp_gateway.config as cfg

    records: list[logging.LogRecord] = []
    rendered: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)
            rendered.append(self.format(record))

    original = logging.getLogRecordFactory()
    monkeypatch.setattr(cfg, "_BASE_FACTORY", None)
    # A fastmcp.* logger: FastMCP's loggers do not propagate to root and render tracebacks
    # with rich, so a Formatter-based scrubber would miss exactly these.
    log = logging.getLogger("fastmcp.gateway.redaction-test")
    handler = _Capture()
    log.addHandler(handler)
    try:
        install_log_redaction(["topsecret"])
        try:
            raise RuntimeError("GET https://fc.example/topsecret/mcp failed")
        except RuntimeError:
            log.exception("upstream said %s", "topsecret")
        log.warning("clean %s", "args")
    finally:
        log.removeHandler(handler)
        logging.setLogRecordFactory(original)
    assert all("topsecret" not in r for r in rendered)
    assert "***" in rendered[0] and "RuntimeError" in rendered[0]  # traceback kept, scrubbed
    assert records[0].exc_info is None
    assert records[1].args == ("args",)  # records without secrets are untouched
    assert logging.getLogger("httpx").level == logging.WARNING


async def _fail(exc: BaseException):
    raise exc


async def test_guard_hides_transport_errors_that_quote_keyed_urls():
    req = httpx.Request("POST", "https://fc.example/fc-KEY/mcp")
    err = httpx.HTTPStatusError(
        "Server error for url 'https://fc.example/fc-KEY/mcp'",
        request=req,
        response=httpx.Response(500, request=req),
    )
    with pytest.raises(ToolError) as e:
        await _guard("firecrawl", _fail(err), ToolError)
    assert "fc-KEY" not in str(e.value) and "HTTP 500" in str(e.value)
    assert e.value.__suppress_context__

    with pytest.raises(ToolError, match="firecrawl upstream request failed \\(ConnectError\\)"):
        await _guard("firecrawl", _fail(httpx.ConnectError("https://fc.example/fc-KEY")), ToolError)


async def test_guard_passes_upstream_protocol_errors_verbatim():
    with pytest.raises(ToolError, match=r"^Invalid params: query$"):
        await _guard(
            "exa",
            _fail(McpError(ErrorData(code=-32602, message="Invalid params: query"))),
            ToolError,
        )


# --------------------------------------------------------------------------- /auth/whoami

ROOT = "root-tok"
DEVICES = {
    "dev-full": SurfaceTrust("s-full", "full", (), True),
    "dev-work": SurfaceTrust("s-work", "restricted", ("p",), True),
}


@pytest.fixture()
def whoami(monkeypatch):
    def resolve(db_url, *, token_hash_hex=None, legacy_surface_id=None):
        return next(
            (st for t, st in DEVICES.items() if token_hash(t) == token_hash_hex), SurfaceTrust()
        )

    monkeypatch.setattr(whoami_mod, "resolve_caller", resolve)

    def client(machine_token: str = ROOT) -> TestClient:
        mcp = FastMCP("whoami-test")
        whoami_mod.register(mcp, "postgresql://unused", machine_token)
        return TestClient(mcp.http_app())

    return client


@pytest.mark.parametrize(
    "tok, trust, sid", [("dev-full", "full", "s-full"), ("dev-work", "restricted", "s-work")]
)
def test_whoami_describes_an_approved_device(whoami, tok, trust, sid):
    with whoami() as c:
        r = c.get("/auth/whoami", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    assert r.json() == {"kind": "device", "trust": trust, "surface_id": sid}
    assert r.headers["cache-control"] == "no-store"
    # The caller's own verdict only: no project allowlist, no token material.
    assert "p" not in r.json().values() and tok not in r.text


def test_whoami_root_is_restricted_and_has_no_surface(whoami):
    with whoami() as c:
        r = c.get("/auth/whoami", headers={"Authorization": f"Bearer {ROOT}"})
    assert r.json() == {"kind": "root", "trust": "restricted", "surface_id": None}


@pytest.mark.parametrize(
    "headers", [{}, {"Authorization": "Bearer nope"}, {"Authorization": "Basic x"}]
)
def test_whoami_rejects_unknown_callers(whoami, headers):
    with whoami() as c:
        assert c.get("/auth/whoami", headers=headers).status_code == 401


def test_whoami_without_machine_token_is_unavailable_not_open(whoami):
    with whoami(machine_token="") as c:
        r = c.get("/auth/whoami", headers={"Authorization": "Bearer dev-full"})
    assert r.status_code == 503 and r.json() == {"error": "no_machine_token"}


async def test_guard_describes_wrapped_and_timeout_failures():
    try:
        try:
            raise httpx.ConnectError("https://fc.example/fc-KEY")
        except httpx.ConnectError as inner:
            raise RuntimeError("Client failed to connect") from inner
    except RuntimeError as wrapped:
        err = wrapped
    with pytest.raises(ToolError, match=r"request failed \(ConnectError\)$"):
        await _guard("firecrawl", _fail(err), ToolError)
    try:
        try:
            raise asyncio.CancelledError()
        except asyncio.CancelledError:
            raise TimeoutError() from None
    except TimeoutError as t:
        timeout_err = t
    timeout_err.__context__ = asyncio.CancelledError()
    with pytest.raises(ToolError, match="exa upstream timed out"):
        await _guard("exa", _fail(timeout_err), ToolError)


# --------------------------------------------------------------------------- provider cache


def _provider(ident: list[str], allowed: list[bool] | None = None, **kw):
    from mcp_gateway.upstream import UpstreamProvider

    return UpstreamProvider(
        "stub",
        lambda: None,  # type: ignore[arg-type,return-value]
        identity=lambda: ident[0],
        allow=lambda kind: (allowed or [True])[0] and kind not in kw.get("denied_kinds", ()),
        cache_ttl=kw.get("ttl", 10.0),
        discovery_timeout=1.0,
        failure_backoff=kw.get("backoff", 0.0),
    )


async def test_provider_caches_per_identity_and_serves_bounded_stale(monkeypatch):
    import mcp_gateway.upstream as up

    clock = [1000.0]
    monkeypatch.setattr(up.time, "monotonic", lambda: clock[0])
    ident = ["alice"]
    p = _provider(ident)
    calls: list[str] = []
    fail = [False]

    async def fetch(kind):
        calls.append(ident[0])
        if fail[0]:
            raise httpx.ConnectError("https://secret.example/KEY")
        return [f"{ident[0]}-{kind}"]

    monkeypatch.setattr(p, "_fetch", fetch)
    assert await p._components("tools") == ["alice-tools"]
    ident[0] = "bob"
    assert await p._components("tools") == ["bob-tools"]  # never alice's snapshot
    assert calls == ["alice", "bob"]

    ident[0] = "alice"
    clock[0] += 60  # past the TTL: refresh fails, the recent snapshot is still served
    fail[0] = True
    assert await p._components("tools") == ["alice-tools"]
    clock[0] += up._MAX_STALE  # too old to fall back on
    with pytest.raises(
        up.UpstreamUnavailable, match=r"^stub upstream request failed \(ConnectError\)$"
    ):
        await p._components("tools")


async def test_provider_checks_access_before_any_upstream_io(monkeypatch):
    allowed = [False]
    p = _provider(["alice"], allowed)

    async def fetch(kind):
        raise AssertionError("upstream contacted for a caller who may not see it")

    monkeypatch.setattr(p, "_fetch", fetch)
    assert await p._list_tools() == []
    assert await p._get_tool("anything") is None

    def no_caller():
        raise PermissionError

    p._identity = no_caller
    allowed[0] = True
    assert await p._list_resources() == []


async def test_provider_withholds_a_denied_kind_without_upstream_io(monkeypatch):
    p = _provider(["bob"], denied_kinds=("resources", "templates"))
    fetched: list[str] = []

    async def fetch(kind):
        fetched.append(kind)
        return [f"bob-{kind}"]

    monkeypatch.setattr(p, "_fetch", fetch)
    assert await p._components("tools") == ["bob-tools"]
    assert await p._list_resources() == []
    assert await p._get_resource("skill://x/SKILL.md") is None
    assert await p._get_resource_template("skill://x/a.py") is None
    assert fetched == ["tools"]


def test_probe_search_fills_required_args_and_falls_back_between_providers():
    from types import SimpleNamespace

    from mcp_gateway.probe import _search_call

    def tool(props, required):
        return SimpleNamespace(inputSchema={"properties": props, "required": required})

    exa = tool(
        {"query": {"type": "string"}, "objective": {"type": "string"}}, ["query", "objective"]
    )
    fc = tool({"query": {"type": "string"}, "limit": {"type": "integer"}}, ["query"])
    odd = tool({"query": {"type": "string"}, "n": {"type": "integer"}}, ["query", "n"])
    assert _search_call({"exa_web_search_exa": exa, "firecrawl_firecrawl_search": fc}, "q") == (
        "exa_web_search_exa",
        {"query": "q", "objective": "q"},
    )
    assert _search_call({"firecrawl_firecrawl_search": fc, "recall": fc}, "q") == (
        "firecrawl_firecrawl_search",
        {"query": "q"},
    )
    assert _search_call({"exa_search_odd": odd}, "q") is None
