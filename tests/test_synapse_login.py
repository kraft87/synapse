"""`synapse-login` on a machine that is already enrolled (plugin/scripts/synapse_login.py).

The plugin keeps ONE credential slot, SYNAPSE_INGEST_TOKEN. Login fetches the shared root
token into it; enrollment then replaces it with a token minted for this machine. So the
slot only ever moves forward: a re-run of login on an enrolled machine must not put the
root token back, because the root token resolves server-side to a restricted surface
with no projects and recall silently serves nothing.

  * an enrolled machine keeps its device token and record, and no sign-in is started;
  * a first login still stores the root token and then enrolls;
  * ``--reenroll`` replaces the device credential on purpose, and a sign-in that fails
    part-way leaves the old one in place.

Stdlib-only script loaded by path. No live server: the HTTP helpers are monkeypatched.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
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


def _script_enroll(monkeypatch, mod) -> list[str]:
    """Script enroll.py's own device flow (config.post_json)."""
    calls: list[str] = []

    def fake_post(path, payload, timeout=30.0):
        calls.append(path)
        return dict(_ENROLL_START) if path == "/device/code" else dict(_MINTED)

    monkeypatch.setattr(mod.config, "post_json", fake_post)
    return calls


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
# First login is unchanged
# ---------------------------------------------------------------------------


def test_first_login_stores_the_root_token_then_enrolls(monkeypatch, login, tmp_path):
    seen_by_enroll: list[str | None] = []
    _script_login(monkeypatch, login)
    enroll_calls = _script_enroll(monkeypatch, login)
    real_enroll = login.enroll_mod.enroll

    def spy(*a, **k):
        seen_by_enroll.append(_slot(tmp_path))
        return real_enroll(*a, **k)

    monkeypatch.setattr(login.enroll_mod, "enroll", spy)

    assert _run(monkeypatch, login) == 0

    assert seen_by_enroll == [_ROOT]  # login wrote the root token, enrollment traded it
    assert enroll_calls == ["/device/code", "/surfaces/enroll"]
    assert _slot(tmp_path) == "device-token-new"
    assert login.config.read_device_state()["surface_id"] == "dev-new456"


# ---------------------------------------------------------------------------
# --reenroll: replacing the device credential on purpose
# ---------------------------------------------------------------------------


def test_reenroll_replaces_the_device_credential(monkeypatch, login, tmp_path):
    _enrolled(login)
    _script_login(monkeypatch, login)
    enroll_calls = _script_enroll(monkeypatch, login)

    assert _run(monkeypatch, login, "--reenroll") == 0

    assert enroll_calls == ["/device/code", "/surfaces/enroll"]
    assert _slot(tmp_path) == "device-token-new"
    record = login.config.read_device_state()
    assert record["surface_id"] == "dev-new456"
    # The record now pins the NEW token (a legacy record gains its hash here).
    assert record["token_sha256"] == hashlib.sha256(b"device-token-new").hexdigest()
    assert "device-token-new" not in json.dumps(record)


def test_reenroll_whose_sign_in_fails_leaves_the_old_credential(monkeypatch, login, tmp_path):
    """The record is only dropped once the sign-in succeeded, so a refused or abandoned
    approval leaves the machine exactly as it was."""
    _enrolled(login)
    _script_login(monkeypatch, login, token_reply={"error": "access_denied"})
    enroll_calls = _script_enroll(monkeypatch, login)

    assert _run(monkeypatch, login, "--reenroll") == 1

    assert enroll_calls == []
    assert _slot(tmp_path) == _DEVICE
    assert login.config.read_device_state() == _RECORD
