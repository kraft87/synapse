"""Unit coverage for the gateway's config, redaction and error hygiene, and /auth/whoami."""

from __future__ import annotations

import asyncio
import json
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

# --------------------------------------------------------------------------- core settings


def test_defaults_are_synapse_only():
    s = load_settings({})
    assert s.registry.upstreams == () and s.registry.skills_dirs == ()
    assert s.synapse_mcp_url == "http://127.0.0.1:8765/mcp"
    assert s.whoami_url == "http://127.0.0.1:8765/auth/whoami"
    assert s.skills_trust == "full"  # Synapse skills withheld from restricted devices by default
    assert (s.host, s.port) == ("127.0.0.1", 8766)
    assert s.secrets() == []


def test_synapse_url_may_be_given_with_mcp_suffix():
    assert (
        load_settings({P + "SYNAPSE_URL": "https://syn.example/mcp"}).synapse_url
        == "https://syn.example"
    )


@pytest.mark.parametrize(
    "env, match",
    [
        ({P + "SKILLS_TRUST": "anyone"}, "SKILLS_TRUST"),
        ({P + "AUTH_CACHE_TTL": "-1"}, ">= 0"),
        ({P + "SYNAPSE_URL": "ftp://syn"}, "http"),
        ({P + "PORT": "eighty"}, "PORT"),
    ],
)
def test_invalid_core_settings_are_refused(env, match):
    with pytest.raises(ConfigError, match=match):
        load_settings(env)


# --------------------------------------------------------------------------- registry

SECRET = "s3cr3t-value-that-must-never-appear"


def _load(tmp_path, registry, env=None):
    cfg = tmp_path / "gateway.json"
    cfg.write_text(registry if isinstance(registry, str) else json.dumps(registry))
    return load_settings({P + "CONFIG_FILE": str(cfg), **(env or {})})


def _up(**over):
    return {"namespace": "tracker", "url": "https://tracker.example/mcp", **over}


def test_registry_auth_shapes(tmp_path):
    (tmp_path / "docs.key").write_text("from-file\n")
    s = _load(
        tmp_path,
        {
            "upstreams": [
                _up(auth={"type": "bearer", "secret_env": "TRACKER_TOKEN"}, min_trust="full"),
                {
                    "namespace": "docs-search",
                    "url": "https://docs.example/mcp",
                    "auth": {"type": "header", "header": "X-Api-Key", "secret_file": "docs.key"},
                    "min_trust": "restricted",
                    "discovery_timeout": 2,
                    "description": "Team docs",
                },
                {
                    "namespace": "files",
                    "url": "https://files.example/{secret}/mcp",
                    "auth": {"type": "url", "secret_env": "FILES_KEY"},
                },
                {"namespace": "status", "url": "http://status.internal:9000/mcp"},
            ]
        },
        {"TRACKER_TOKEN": "t1", "FILES_KEY": "a/b c"},
    )
    tracker, docs, files, status = s.registry.upstreams
    assert tracker.headers() == {"Authorization": "Bearer t1"} and tracker.min_trust == "full"
    assert docs.headers() == {"X-Api-Key": "from-file"} and docs.min_trust == "restricted"
    assert docs.discovery_timeout == 2.0 and docs.description == "Team docs"
    assert files.url() == "https://files.example/a%2Fb%20c/mcp" and files.headers() == {}
    assert files.display_url == "https://files.example/…" and "a/b" not in repr(files)
    assert status.auth_type == "none" and status.headers() == {} and status.min_trust == "full"
    assert sorted(s.secrets()) == ["a/b c", "from-file", "t1"]


def test_registry_skills_dirs_resolve_relative_to_the_file(tmp_path):
    (tmp_path / "team").mkdir()
    s = _load(tmp_path, {"skills_dirs": [{"path": "team", "min_trust": "restricted"}]})
    (d,) = s.registry.skills_dirs
    assert d.path == (tmp_path / "team").resolve() and d.min_trust == "restricted"


@pytest.mark.parametrize(
    "registry, match",
    [
        ({"upstreams": [_up(), _up()]}, "duplicate namespace"),
        ({"upstreams": [_up(namespace="recall")]}, "reserved"),
        ({"upstreams": [_up(namespace="synapse")]}, "reserved"),
        ({"upstreams": [_up(namespace="list")]}, "reserved"),
        ({"upstreams": [_up(namespace="Tracker")]}, "namespace must be"),
        ({"upstreams": [_up(namespace="my_tracker")]}, "namespace must be"),
        ({"upstreams": [_up(namespace="../x")]}, "namespace must be"),
        ({"upstreams": [_up(namespace="")]}, "namespace must be"),
        ({"upstreams": [_up(namespace="a" * 33)]}, "namespace must be"),
        ({"upstreams": [_up(namespace="-x")]}, "namespace must be"),
        ({"upstreams": [_up(url="ftp://tracker.example/mcp")]}, "http"),
        ({"upstreams": [_up(url="stdio:tracker")]}, "http"),
        ({"upstreams": [_up(url="https://u:p@tracker.example/mcp")]}, "must not embed"),
        ({"upstreams": [_up(url="https://t.example/{secret}")]}, "placeholder"),
        ({"upstreams": [_up(auth={"type": "url", "secret_env": "K"})]}, "placeholder"),
        ({"upstreams": [_up(auth={"type": "oauth"})]}, "auth.type"),
        ({"upstreams": [_up(auth={"type": "bearer"})]}, "exactly one"),
        (
            {"upstreams": [_up(auth={"type": "bearer", "secret_env": "K", "secret_file": "f"})]},
            "exactly one",
        ),
        ({"upstreams": [_up(auth={"type": "bearer", "secret_env": "UNSET_VAR_X"})]}, "unset"),
        (
            {"upstreams": [_up(auth={"type": "bearer", "secret_file": "missing.key"})]},
            "cannot read",
        ),
        ({"upstreams": [_up(auth={"type": "header", "secret_env": "K"})]}, "header name"),
        (
            {"upstreams": [_up(auth={"type": "header", "header": "Host", "secret_env": "K"})]},
            "not allowed",
        ),
        (
            {"upstreams": [_up(auth={"type": "bearer", "header": "X", "secret_env": "K"})]},
            "only applies",
        ),
        ({"upstreams": [_up(min_trust="everyone")]}, "min_trust"),
        # An inline credential next to a valid reference must not ride along silently.
        (
            {"upstreams": [_up(auth={"type": "bearer", "secret_env": "K", "token": "inline"})]},
            "unknown key",
        ),
        ({"upstreams": [_up(call_timeout=-1)]}, "non-negative"),
        ({"upstreams": [_up(call_timeout=True)]}, "non-negative"),
        ({"upstreams": [_up(cache="big")]}, "unknown key"),
        ({"upstreams": "tracker"}, "must be lists"),
        ({"skills_dirs": [{"path": "nope"}]}, "not a directory"),
        ({"extra": 1}, "unknown key"),
        ([], "JSON object"),
    ],
)
def test_malformed_registry_is_refused_at_startup(tmp_path, registry, match):
    with pytest.raises(ConfigError, match=match):
        _load(tmp_path, registry, {"K": "v"})


@pytest.mark.parametrize(
    "registry",
    [
        # An inline credential — in an unknown field or where a reference belongs — is refused
        # and never echoed back.
        {"upstreams": [_up(auth={"type": "bearer", "token": SECRET})]},
        {"upstreams": [_up(auth={"type": "bearer", "secret_env": SECRET})]},
        {"upstreams": [_up(api_key=SECRET)]},
        {"upstreams": [_up(url=f"https://user:{SECRET}@t.example/mcp")]},
        {"upstreams": [_up(namespace=SECRET)]},
        {"upstreams": [_up(description=SECRET * 10)]},
        f'{{"upstreams": [{{"namespace": "{SECRET}", }}]}}',  # malformed JSON quoting it
    ],
)
def test_registry_errors_never_echo_values(tmp_path, registry):
    with pytest.raises(ConfigError) as e:
        _load(tmp_path, registry)
    assert SECRET not in str(e.value)


def test_registry_secret_values_never_in_errors(tmp_path):
    (tmp_path / "k").write_text(SECRET)
    bad = {"upstreams": [_up(auth={"type": "header", "header": "Cookie", "secret_file": "k"})]}
    with pytest.raises(ConfigError) as e:
        _load(tmp_path, bad, {"TRACKER_TOKEN": SECRET})
    assert SECRET not in str(e.value)


def test_missing_or_unreadable_registry_file_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="cannot read registry"):
        load_settings({P + "CONFIG_FILE": str(tmp_path / "absent.json")})


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
            raise RuntimeError("GET https://files.example/topsecret/mcp failed")
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
    req = httpx.Request("POST", "https://files.example/fc-KEY/mcp")
    err = httpx.HTTPStatusError(
        "Server error for url 'https://files.example/fc-KEY/mcp'",
        request=req,
        response=httpx.Response(500, request=req),
    )
    with pytest.raises(ToolError) as e:
        await _guard("wiki", _fail(err), ToolError)
    assert "fc-KEY" not in str(e.value) and "HTTP 500" in str(e.value)
    assert e.value.__suppress_context__

    with pytest.raises(ToolError, match="wiki upstream request failed \\(ConnectError\\)"):
        await _guard("wiki", _fail(httpx.ConnectError("https://files.example/fc-KEY")), ToolError)


async def test_guard_passes_upstream_protocol_errors_verbatim():
    with pytest.raises(ToolError, match=r"^Invalid params: query$"):
        await _guard(
            "tracker",
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
            raise httpx.ConnectError("https://files.example/fc-KEY")
        except httpx.ConnectError as inner:
            raise RuntimeError("Client failed to connect") from inner
    except RuntimeError as wrapped:
        err = wrapped
    with pytest.raises(ToolError, match=r"request failed \(ConnectError\)$"):
        await _guard("wiki", _fail(err), ToolError)
    try:
        try:
            raise asyncio.CancelledError()
        except asyncio.CancelledError:
            raise TimeoutError() from None
    except TimeoutError as t:
        timeout_err = t
    timeout_err.__context__ = asyncio.CancelledError()
    with pytest.raises(ToolError, match="tracker upstream timed out"):
        await _guard("tracker", _fail(timeout_err), ToolError)


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


def test_probe_call_arguments_must_be_an_explicit_json_object(monkeypatch, capsys):
    import mcp_gateway.probe as probe

    called = []

    async def fake(*a, **k):
        called.append(a)
        return 0

    monkeypatch.setattr(probe, "_probe", fake)
    monkeypatch.setenv("SYNAPSE_INGEST_TOKEN", "device-token")
    assert probe.main(["--call", "tracker_list_issues", "--args", "[1]"]) == 2
    assert probe.main(["--call", "tracker_list_issues", "--args", "not json"]) == 2
    assert not called
    assert probe.main(["--call", "tracker_list_issues", "--args", '{"project": "web"}']) == 0
    assert called[0][3] == ("tracker_list_issues", {"project": "web"})
    assert "device-token" not in capsys.readouterr().err


def test_probe_explains_a_refused_credential(monkeypatch, capsys):
    import mcp_gateway.probe as probe

    async def refused(*a, **k):
        raise httpx.HTTPStatusError(
            "Client error '401 Unauthorized'",
            request=httpx.Request("POST", "http://gw/mcp"),
            response=httpx.Response(401),
        )

    monkeypatch.setattr(probe, "_probe", refused)
    monkeypatch.setenv("SYNAPSE_INGEST_TOKEN", "root-like-token")
    assert probe.main([]) == 3
    err = capsys.readouterr().err
    assert "only approved Synapse DEVICE tokens" in err and "root-like-token" not in err


def test_shipped_example_registry_is_valid(tmp_path):
    """examples/gateway/gateway.example.json must stay loadable: only its secret references
    are swapped for test ones (a secret file here, the env var it names)."""
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "examples/gateway/gateway.example.json"
    data = json.loads(example.read_text())
    (tmp_path / "wiki.key").write_text("k")
    for up in data["upstreams"]:
        if "secret_file" in up.get("auth", {}):
            up["auth"]["secret_file"] = "wiki.key"
    s = _load(tmp_path, data, {"TRACKER_MCP_TOKEN": "t"})
    assert [u.namespace for u in s.registry.upstreams] == ["tracker", "wiki"]
