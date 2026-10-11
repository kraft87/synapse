"""`synapse-login` (plugin/scripts/synapse_login.py): one approval per machine.

The plugin keeps ONE credential slot, SYNAPSE_INGEST_TOKEN, and what ends up in it is the
device token enrollment mints. The enrollment device flow is itself the sign-in: the
server reads the approving identity and checks the allowlist before it mints. So the
default (device) login runs exactly ONE device flow and never fetches the shared root
token at all. It used to run two, a sign-in whose root token was overwritten moments
later, then the enrollment: two codes to approve for one result.

  * an enrolled machine keeps its device token and record, and no sign-in is started;
  * a first login asks for one approval and stores the device token;
  * a refused or abandoned approval leaves the machine exactly as it was;
  * ``--reenroll`` replaces the device credential on purpose, with one approval, and a
    failure leaves the old one in place;
  * a server without the enroll route falls back to the root-token sign-in on the SAME
    device code, so it is still one approval.

Stdlib-only script loaded by path. No live server: the HTTP helpers are monkeypatched.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import urllib.error
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "plugin" / "scripts"
_SCRIPT = _SCRIPTS / "synapse_login.py"
_MODULES = ("config", "enroll", "synapse_login")

_ENV_VARS = (
    "SYNAPSE_URL",
    "SYNAPSE_INGEST_URL",
    "SYNAPSE_INGEST_TOKEN",
    "SYNAPSE_MCP_URL",
    "SYNAPSE_MACHINE_ROLE",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_URL",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_TOKEN",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_MACHINE_ROLE",
)

_ROOT = "root-token-shared"
_DEVICE = "device-token-old"
_RECORD = {
    "surface_id": "dev-old123",
    "trust": "full",
    "allowed_projects": [],
    "label": "test-laptop",
    "login": "owner",
}

_LOGIN_START = {
    "device_code": "login-dc",
    "user_code": "LOGN-0001",
    "verification_uri": "https://idp.example.net/device",
    "interval": 0,
    "expires_in": 5,
}
_ENROLL_START = {
    "device_code": "enroll-dc",
    "user_code": "ENRL-0002",
    "verification_uri": "https://idp.example.net/device",
    "interval": 0,
    "expires_in": 5,
}
_MINTED = {
    "status": "ok",
    "token": "device-token-new",
    "login": "owner",
    "surface": {"surface_id": "dev-new456", "trust": "full", "allowed_projects": []},
}


def _load() -> ModuleType:
    """Load synapse_login.py (and the config/enroll it imports) fresh against the env."""
    for name in _MODULES:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(_SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location("synapse_login", _SCRIPT)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules["synapse_login"] = mod
        spec.loader.exec_module(mod)
        import enroll  # the same module object the script's lazy `import enroll` returns

        mod.enroll_mod = enroll
    finally:
        sys.path.remove(str(_SCRIPTS))
    return mod


@pytest.fixture()
def login(monkeypatch, tmp_path) -> ModuleType:
    """Scratch config + data dirs, no real env, no sleeping, no browser."""
    cfg_dir = tmp_path / "claude"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(cfg_dir))
    monkeypatch.setenv("SYNAPSE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SYNAPSE_SURFACE", "test-laptop")
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    mod = _load()
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
    monkeypatch.setattr(mod.enroll_mod.time, "sleep", lambda _s: None)
    monkeypatch.setattr(mod.webbrowser, "open", lambda *_a, **_k: True)
    yield mod
    for name in _MODULES:
        sys.modules.pop(name, None)


def _slot(tmp_path) -> str | None:
    """What settings.json holds in SYNAPSE_INGEST_TOKEN — what the MCP header sends."""
    settings = tmp_path / "claude" / "settings.json"
    if not settings.exists():
        return None
    data = json.loads(settings.read_text())
    opts = next(iter(data.get("pluginConfigs", {}).values()), {}).get("options", {})
    return opts.get("SYNAPSE_INGEST_TOKEN")


def _enrolled(mod) -> None:
    mod.config.write_user_config("SYNAPSE_INGEST_TOKEN", _DEVICE)
    mod.config.write_device_state(dict(_RECORD))
    assert mod.enroll_mod.is_enrolled()


def _script_login(monkeypatch, mod, token_reply: dict | None = None) -> list[str]:
    """Script the login device flow (/device/code then /device/token). Returns the paths
    hit, so a test can assert that no sign-in was started at all."""
    calls: list[str] = []
    reply = token_reply if token_reply is not None else {"token": _ROOT, "login": "owner"}

    def fake_post_json(url, payload):
        calls.append(url)
        if url.endswith("/device/code"):
            return dict(_LOGIN_START)
        return dict(reply)

    def fake_get_json(url):
        calls.append(url)
        raise AssertionError("the browser flow should not have started")

    monkeypatch.setattr(mod, "_post_json", fake_post_json)
    monkeypatch.setattr(mod, "_get_json", fake_get_json)
    return calls


def _script_enroll(
    monkeypatch, mod, replies: list | None = None, payloads: list[dict] | None = None
) -> list[str]:
    """Script enroll.py's own device flow (config.post_json). ``replies`` are the
    /surfaces/enroll answers, one per poll (the last repeats); an Exception is raised.
    ``payloads`` collects the bodies sent, so a test can read the declared trust."""
    calls: list[str] = []
    queue = list(replies) if replies is not None else [_MINTED]

    def fake_post(path, payload, timeout=30.0):
        calls.append(path)
        if payloads is not None:
            payloads.append(dict(payload))
        if path == "/device/code":
            return dict(_ENROLL_START)
        reply = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(reply, Exception):
            raise reply
        return dict(reply)

    monkeypatch.setattr(mod.config, "post_json", fake_post)
    return calls


def _approvals(*call_lists: list[str]) -> int:
    """How many codes the human was asked to approve: one per device flow started."""
    return sum(1 for calls in call_lists for c in calls if c.endswith("/device/code"))


class _Body:
    """File-like for HTTPError's fp (``read`` for the code, ``close`` for GC)."""

    def __init__(self, payload: dict) -> None:
        self._data = json.dumps(payload).encode()

    def read(self, *_a: object) -> bytes:
        return self._data

    def close(self) -> None:
        pass


def _http_error(code: int, payload: dict | None) -> urllib.error.HTTPError:
    body = _Body(payload) if payload is not None else None
    return urllib.error.HTTPError("http://x/surfaces/enroll", code, "err", None, body)  # type: ignore[arg-type]


def _run(monkeypatch, mod, *args: str) -> int:
    monkeypatch.setattr(sys, "argv", ["synapse_login.py", *args])
    return mod.main()


# ---------------------------------------------------------------------------
# The bug: re-login on an enrolled machine
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("args", [(), ("--browser",)], ids=["device", "browser"])
def test_relogin_on_an_enrolled_machine_keeps_its_device_token(
    monkeypatch, login, tmp_path, capsys, args
):
    """The root token in the slot would authenticate as no surface at all, which the
    server serves as restricted with an empty allowlist: recall goes quiet, no error."""
    _enrolled(login)
    login_calls = _script_login(monkeypatch, login)
    enroll_calls = _script_enroll(monkeypatch, login)

    assert _run(monkeypatch, login, *args) == 0

    assert _slot(tmp_path) == _DEVICE
    assert login.config.read_device_state() == _RECORD
    assert login_calls == [] and enroll_calls == []  # nothing to approve, nothing fetched
    out = capsys.readouterr().out
    assert "already enrolled" in out and "dev-old123" in out and "--reenroll" in out


# ---------------------------------------------------------------------------
# First login: one approval
# ---------------------------------------------------------------------------


def test_first_login_needs_exactly_one_approval(monkeypatch, login, tmp_path):
    """The headline: one code to approve, and the device token is what gets stored.
    The shared root token is never fetched, so it never sits in the slot even briefly."""
    login_calls = _script_login(monkeypatch, login)
    enroll_calls = _script_enroll(monkeypatch, login)
    written: list[str] = []
    real_write = login.config.write_user_config

    def spy(key, value):
        written.append(value)
        real_write(key, value)

    monkeypatch.setattr(login.config, "write_user_config", spy)

    assert _run(monkeypatch, login) == 0

    assert _approvals(login_calls, enroll_calls) == 1
    assert login_calls == []  # no separate sign-in flow, no /device/token poll
    assert enroll_calls == ["/device/code", "/surfaces/enroll"]
    assert written == ["device-token-new"]  # the root token never touched the slot
    assert _slot(tmp_path) == "device-token-new"
    record = login.config.read_device_state()
    assert record["surface_id"] == "dev-new456"
    assert record["token_sha256"] == hashlib.sha256(b"device-token-new").hexdigest()


def test_first_login_prints_one_code_and_who_signed_in(monkeypatch, login, capsys):
    _script_login(monkeypatch, login)
    _script_enroll(monkeypatch, login)

    assert _run(monkeypatch, login) == 0

    captured = capsys.readouterr()
    assert captured.err.count("code:  ") == 1
    assert "ENRL-0002" in captured.err and "LOGN-0001" not in captured.err
    assert "owner" in captured.out and "dev-new456" in captured.out


@pytest.mark.parametrize(
    ("role", "trust"), [("personal", "full"), ("work", "restricted")], ids=["personal", "work"]
)
def test_the_machine_role_is_declared_in_the_single_flow(monkeypatch, tmp_path, login, role, trust):
    """The personal/work answer still reaches the server as the trust to enroll at."""
    monkeypatch.setenv("SYNAPSE_MACHINE_ROLE", role)
    mod = _load()  # MACHINE_ROLE resolves at import
    monkeypatch.setattr(mod.enroll_mod.time, "sleep", lambda _s: None)
    login_calls = _script_login(monkeypatch, mod)
    payloads: list[dict] = []
    enroll_calls = _script_enroll(monkeypatch, mod, payloads=payloads)

    assert _run(monkeypatch, mod) == 0

    assert _approvals(login_calls, enroll_calls) == 1
    assert payloads[1] == {"device_code": "enroll-dc", "label": "test-laptop", "trust": trust}


def test_login_backs_off_on_an_idp_rate_limit_instead_of_failing(monkeypatch, login, tmp_path):
    """``temporarily_unavailable`` is the IdP's 429. Like slow_down: the interval grows
    and the loop keeps asking, rather than printing a failure over an approval already
    given. (With one flow there is no second poller to collide with, but an IdP can
    still throttle a single one.)"""
    throttled = {"status": "pending", "error": "temporarily_unavailable"}
    enroll_calls = _script_enroll(monkeypatch, login, replies=[throttled, throttled, _MINTED])
    _script_login(monkeypatch, login)
    sleeps: list[float] = []
    monkeypatch.setattr(login.enroll_mod.time, "sleep", sleeps.append)

    assert _run(monkeypatch, login) == 0
    assert enroll_calls.count("/surfaces/enroll") == 3
    assert sleeps == sorted(sleeps) and sleeps[-1] > sleeps[0]  # backed off, not hammered
    assert _slot(tmp_path) == "device-token-new"


@pytest.mark.parametrize(
    "reply",
    [
        _http_error(403, {"status": "pending", "error": "access_denied"}),
        _http_error(403, {"status": "error", "detail": "fake user 'stranger' not in allowlist"}),
        {"status": "pending", "error": "expired_token"},
    ],
    ids=["refused", "not-allowlisted", "expired"],
)
def test_a_failed_first_login_leaves_the_machine_as_it_was(monkeypatch, login, tmp_path, reply):
    login_calls = _script_login(monkeypatch, login)
    enroll_calls = _script_enroll(monkeypatch, login, replies=[reply])

    assert _run(monkeypatch, login) == 1

    assert _approvals(login_calls, enroll_calls) == 1
    assert _slot(tmp_path) is None  # nothing written: no root token, no device token
    assert login.config.read_device_state() == {}


def test_a_network_blip_mid_poll_keeps_waiting(monkeypatch, login, tmp_path):
    """The approval may land while the connection is down; one failed poll must not
    throw away the code the human is about to approve."""
    blip = urllib.error.URLError("connection reset")
    enroll_calls = _script_enroll(monkeypatch, login, replies=[blip, _MINTED])

    assert _run(monkeypatch, login) == 0
    assert enroll_calls == ["/device/code", "/surfaces/enroll", "/surfaces/enroll"]
    assert _slot(tmp_path) == "device-token-new"


def test_an_unreachable_server_leaves_the_machine_as_it_was(monkeypatch, login, tmp_path):
    def down(path, payload, timeout=30.0):
        raise urllib.error.URLError("server down")

    monkeypatch.setattr(login.config, "post_json", down)

    assert _run(monkeypatch, login) == 1
    assert _slot(tmp_path) is None
    assert login.config.read_device_state() == {}


# ---------------------------------------------------------------------------
# Older server: no /surfaces/enroll route
# ---------------------------------------------------------------------------


def test_a_server_without_the_enroll_route_signs_in_on_the_same_code(
    monkeypatch, login, tmp_path, capsys
):
    """A server that predates enrollment 404s /surfaces/enroll without polling the IdP,
    so the device code is still unspent: redeem it at /device/token instead of asking
    for a second approval. Such a server has only the root token to give."""
    login_calls = _script_login(monkeypatch, login)
    enroll_calls = _script_enroll(monkeypatch, login, replies=[_http_error(404, None)])

    assert _run(monkeypatch, login) == 0

    assert _approvals(login_calls, enroll_calls) == 1
    assert login_calls == ["http://localhost:8765/device/token"]
    assert _slot(tmp_path) == _ROOT
    assert login.config.read_device_state() == {}
    assert "owner" in capsys.readouterr().out


def test_the_legacy_fallback_reports_a_refusal(monkeypatch, login, tmp_path):
    _script_login(monkeypatch, login, token_reply={"error": "access_denied"})
    _script_enroll(monkeypatch, login, replies=[_http_error(404, None)])

    assert _run(monkeypatch, login) == 1
    assert _slot(tmp_path) is None


# ---------------------------------------------------------------------------
# --browser: the legacy loopback sign-in, then enrollment
# ---------------------------------------------------------------------------


def test_browser_login_stores_the_root_token_then_enrolls(monkeypatch, login, tmp_path):
    """The browser flow's OAuth token cannot enroll by itself, so it still hands over
    to the device flow; the device token is what ends up stored."""
    seen_by_enroll: list[str | None] = []
    monkeypatch.setattr(login, "_get_json", lambda url: _OAUTH_META)
    monkeypatch.setattr(login, "_post_json", lambda url, payload: {"client_id": "c1"})
    monkeypatch.setattr(login, "_post_form", lambda url, fields: {"access_token": "at"})
    monkeypatch.setattr(login, "_mcp_call_token", lambda access: _ROOT)
    monkeypatch.setattr(login.secrets, "token_urlsafe", lambda _n: "fixed-state")
    monkeypatch.setattr(login.http.server.HTTPServer, "handle_request", _fake_callback)
    enroll_calls = _script_enroll(monkeypatch, login)
    real_enroll = login.enroll_mod.enroll

    def spy(*a, **k):
        seen_by_enroll.append(_slot(tmp_path))
        return real_enroll(*a, **k)

    monkeypatch.setattr(login.enroll_mod, "enroll", spy)

    assert _run(monkeypatch, login, "--browser") == 0

    assert seen_by_enroll == [_ROOT]
    assert enroll_calls == ["/device/code", "/surfaces/enroll"]
    assert _slot(tmp_path) == "device-token-new"


_OAUTH_META = {
    "registration_endpoint": "https://synapse.example.net/register",
    "authorization_endpoint": "https://synapse.example.net/authorize",
    "token_endpoint": "https://synapse.example.net/token",
}


def _fake_callback(self) -> None:
    """Stand in for the browser redirect: hand the handler a code and the right state."""
    self.RequestHandlerClass.code = "auth-code"
    self.RequestHandlerClass.state = "fixed-state"


# ---------------------------------------------------------------------------
# --reenroll: replacing the device credential on purpose
# ---------------------------------------------------------------------------


def test_reenroll_replaces_the_device_credential_with_one_approval(monkeypatch, login, tmp_path):
    _enrolled(login)
    login_calls = _script_login(monkeypatch, login)
    enroll_calls = _script_enroll(monkeypatch, login)

    assert _run(monkeypatch, login, "--reenroll") == 0

    assert _approvals(login_calls, enroll_calls) == 1
    assert enroll_calls == ["/device/code", "/surfaces/enroll"]
    assert _slot(tmp_path) == "device-token-new"
    record = login.config.read_device_state()
    assert record["surface_id"] == "dev-new456"
    # The record now pins the NEW token (a legacy record gains its hash here).
    assert record["token_sha256"] == hashlib.sha256(b"device-token-new").hexdigest()
    assert "device-token-new" not in json.dumps(record)


def test_reenroll_whose_approval_fails_leaves_the_old_credential(monkeypatch, login, tmp_path):
    """The record is only dropped once the approval succeeded, so a refused or abandoned
    approval leaves the machine exactly as it was."""
    _enrolled(login)
    _script_login(monkeypatch, login)
    denied = _http_error(403, {"status": "pending", "error": "access_denied"})
    _script_enroll(monkeypatch, login, replies=[denied])

    assert _run(monkeypatch, login, "--reenroll") == 1

    assert _slot(tmp_path) == _DEVICE
    assert login.config.read_device_state() == _RECORD


def test_reenroll_drops_the_old_record_before_writing_the_new_token(monkeypatch, login):
    """Crash ordering: old record gone first, so a crash mid-write leaves a token with
    no record (re-enrolls harmlessly), never the new token under the old record."""
    _enrolled(login)
    _script_login(monkeypatch, login)
    _script_enroll(monkeypatch, login)
    order: list[str] = []
    real_write = login.config.write_user_config

    def spy(key, value):
        order.append("record" if login.config.DEVICE_FILE.exists() else "no-record")
        real_write(key, value)

    monkeypatch.setattr(login.config, "write_user_config", spy)

    assert _run(monkeypatch, login, "--reenroll") == 0
    assert order == ["no-record"]
