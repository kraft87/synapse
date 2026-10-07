"""An enrolled machine's device token wins over SYNAPSE_INGEST_TOKEN from the environment.

The plugin keeps one credential slot. The MCP server reads it only from the plugin config,
while the hooks used plain precedence (env var first). A token left in a shell profile or
in settings.json's "env" block therefore shadowed the device token for every hook while
recall used the device token: uploads attributed to one credential, recall served by
another, and nothing said so.

  * enrolled + stale env -> the device token, plus one session-start line naming the source
  * enrolled, no conflict -> the device token, no line
  * unenrolled -> plain precedence, unchanged (bootstrap carries the root token in env)
  * legacy record without a hash -> the plugin-config token wins (enrollment wrote it there)
  * a hash that matches nothing, or an env URL pointing at another server -> unchanged
  * enrollment and --reenroll record the hash; no token material in device.json or output
  * the Codex mirror (plugin-codex/scripts/common.py) resolves the same way

Stdlib-only scripts loaded by path. No live server.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "plugin" / "scripts"
_CODEX = _REPO / "plugin-codex"
_MODULES = ("config", "enroll", "board_block", "synapse_login")

_ENV_VARS = (
    "SYNAPSE_URL",
    "SYNAPSE_INGEST_URL",
    "SYNAPSE_INGEST_TOKEN",
    "SYNAPSE_BOARD",
    "SYNAPSE_MACHINE_ROLE",
    "SYNAPSE_MCP_URL",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_URL",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_URL",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_TOKEN",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_BOARD",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_MACHINE_ROLE",
)

# Distinctive values, so "no token material in the output" can check prefixes too.
_DEVICE = "tkDEV9f3q7ZxL2mNw4"
_STALE = "tkENVstale0Qw8RpY6"
_OTHER = "tkOPTloaded5Hj1KcT"
_URL = "https://synapse.example.net"
_SURFACE_ID = "dev-abc123"


def _sha(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _record(*, hashed: bool = True, server: str | None = _URL) -> dict:
    rec = {
        "surface_id": _SURFACE_ID,
        "trust": "full",
        "allowed_projects": [],
        "label": "test-laptop",
        "login": "owner",
    }
    if hashed:
        rec["token_sha256"] = _sha(_DEVICE)
    if server:
        rec["server"] = server
    return rec


def _setup(
    monkeypatch,
    tmp_path,
    *,
    saved: str | None = _DEVICE,
    record: dict | None = None,
    env: dict | None = None,
    env_block: dict | None = None,
) -> None:
    """Scratch config + data dirs. ``saved`` is the plugin-config token, ``record`` the
    enrollment record (None = unenrolled), ``env`` the process environment and
    ``env_block`` the "env" block of the user settings.json."""
    cfg_dir = tmp_path / "claude"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    options = {"SYNAPSE_URL": _URL}
    if saved:
        options["SYNAPSE_INGEST_TOKEN"] = saved
    settings: dict = {"pluginConfigs": {"synapse@synapse": {"options": options}}}
    if env_block:
        settings["env"] = env_block
    (cfg_dir / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    if record is not None:
        (data_dir / "device.json").write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(cfg_dir))
    monkeypatch.setenv("SYNAPSE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("SYNAPSE_SURFACE", "test-laptop")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)


def _load(name: str) -> ModuleType:
    for mod in _MODULES:
        sys.modules.pop(mod, None)
    sys.path.insert(0, str(_SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(_SCRIPTS))
    return mod


@pytest.fixture(autouse=True)
def _forget_modules():
    yield
    for mod in _MODULES:
        sys.modules.pop(mod, None)


def _no_token_material(text: str) -> None:
    for tok in (_DEVICE, _STALE, _OTHER):
        assert tok not in text
        assert tok[:6] not in text and tok[-6:] not in text


def _board_payload(monkeypatch, capsys) -> dict:
    """Run the SessionStart board hook against a canned board; return its JSON output."""
    hook = _load("board_block")
    monkeypatch.setattr(hook, "get_json", lambda *_a, **_k: {"status": "ok", "text": "BOARD"})
    monkeypatch.setattr(sys, "stdin", io.StringIO("{}"))
    hook.main()
    out = capsys.readouterr().out
    _no_token_material(out)
    return json.loads(out) if out else {}


def _board(monkeypatch, capsys) -> str:
    payload = _board_payload(monkeypatch, capsys)
    return payload["hookSpecificOutput"]["additionalContext"] if payload else ""


# ---------------------------------------------------------------------------
# Enrolled: the device token wins, and says so
# ---------------------------------------------------------------------------


def test_enrolled_with_a_stale_shell_token_uses_the_device_token(monkeypatch, tmp_path, capsys):
    _setup(monkeypatch, tmp_path, record=_record(), env={"SYNAPSE_INGEST_TOKEN": _STALE})
    cfg = _load("config")
    assert cfg.INGEST_TOKEN == _DEVICE
    assert cfg.TOKEN_OVERRIDE == {"surface_id": _SURFACE_ID, "sources": ["env"]}

    payload = _board_payload(monkeypatch, capsys)
    first, rest = payload["hookSpecificOutput"]["additionalContext"].split("\n", 1)
    assert first == (
        "[Synapse] SYNAPSE_INGEST_TOKEN from your environment is ignored: this machine is "
        f"enrolled as {_SURFACE_ID}. Remove it from your shell profile."
    )
    assert rest.strip() == "BOARD"
    # Shown to the user too, not only to the model; the board itself stays model-only.
    assert payload["systemMessage"] == first


def test_every_hook_sends_the_device_token(monkeypatch, tmp_path):
    """The shared HTTP helpers carry it, so ingest, device admin, private mode, the
    remember spool and the sync lanes all agree with recall."""
    _setup(monkeypatch, tmp_path, record=_record(), env={"SYNAPSE_INGEST_TOKEN": _STALE})
    cfg = _load("config")
    seen: list[str] = []

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    def fake_urlopen(req, timeout=None):
        seen.append(req.get_header("Authorization"))
        return _Resp(b"{}")

    monkeypatch.setattr(cfg.urllib.request, "urlopen", fake_urlopen)
    cfg.post_json("/x", {})
    cfg.get_json("/x")
    cfg.request_json("PUT", "/x")
    assert seen == [f"Bearer {_DEVICE}"] * 3


def test_the_notice_names_the_settings_env_block(monkeypatch, tmp_path, capsys):
    """Claude Code exports settings.json's "env" block to every hook — where pre-054
    installs usually left the token — so the line points there rather than the shell."""
    _setup(
        monkeypatch,
        tmp_path,
        record=_record(),
        env={"SYNAPSE_INGEST_TOKEN": _STALE},
        env_block={"SYNAPSE_INGEST_TOKEN": _STALE},
    )
    ctx = _board(monkeypatch, capsys)
    line = ctx.split("\n", 1)[0]
    assert "is ignored" in line and _SURFACE_ID in line
    assert 'the "env" block of ' in line and line.endswith("settings.json.")
    assert "shell profile" not in line


def test_a_stale_plugin_option_is_overridden_too(monkeypatch, tmp_path, capsys):
    """CLAUDE_PLUGIN_OPTION_* is what Claude Code loaded at startup. After an in-session
    `synapse-login` it still holds the old token until the plugins reload."""
    _setup(
        monkeypatch,
        tmp_path,
        record=_record(),
        env={"CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_TOKEN": _OTHER},
    )
    cfg = _load("config")
    assert cfg.INGEST_TOKEN == _DEVICE
    assert cfg.TOKEN_OVERRIDE["sources"] == ["plugin_option"]
    ctx = _board(monkeypatch, capsys)
    assert "/reload-plugins" in ctx.split("\n", 1)[0]


def test_the_notice_survives_a_disabled_board_and_a_down_server(monkeypatch, tmp_path, capsys):
    _setup(
        monkeypatch,
        tmp_path,
        record=_record(),
        env={"SYNAPSE_INGEST_TOKEN": _STALE, "SYNAPSE_BOARD": "0"},
    )
    hook = _load("board_block")
    monkeypatch.setattr(hook, "get_json", lambda *_a, **_k: pytest.fail("board disabled"))
    hook.main()
    out = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]
    assert out.startswith("[Synapse] SYNAPSE_INGEST_TOKEN from your environment is ignored")

    monkeypatch.delenv("SYNAPSE_BOARD")
    hook = _load("board_block")

    def down(*_a, **_k):
        raise ConnectionError("server down")

    monkeypatch.setattr(hook, "get_json", down)
    monkeypatch.setattr(sys, "stdin", io.StringIO("{}"))
    hook.main()
    out = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]
    assert out.startswith("[Synapse] SYNAPSE_INGEST_TOKEN from your environment is ignored")


def test_enrolled_without_a_conflict_prints_no_notice(monkeypatch, tmp_path, capsys):
    _setup(
        monkeypatch,
        tmp_path,
        record=_record(),
        env={
            "SYNAPSE_INGEST_TOKEN": _DEVICE,  # the same token in env is not a conflict
            "CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_TOKEN": _DEVICE,
        },
    )
    cfg = _load("config")
    assert cfg.INGEST_TOKEN == _DEVICE
    assert cfg.TOKEN_OVERRIDE == {}
    assert cfg.token_override_notice() == ""
    payload = _board_payload(monkeypatch, capsys)
    assert payload["hookSpecificOutput"]["additionalContext"] == "BOARD"
    assert "systemMessage" not in payload


# ---------------------------------------------------------------------------
# Unenrolled and legacy machines
# ---------------------------------------------------------------------------


def test_unenrolled_keeps_plain_precedence(monkeypatch, tmp_path, capsys):
    """Before enrollment the slot carries the root token, wherever the user put it, and
    `synapse-login` needs it. Env still wins and nothing is printed."""
    _setup(monkeypatch, tmp_path, saved=_OTHER, env={"SYNAPSE_INGEST_TOKEN": _STALE})
    cfg = _load("config")
    assert cfg.INGEST_TOKEN == _STALE
    assert cfg.TOKEN_OVERRIDE == {}
    assert _board(monkeypatch, capsys) == "BOARD"


def test_unenrolled_without_env_reads_the_plugin_config(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, saved=_OTHER)
    assert _load("config").INGEST_TOKEN == _OTHER


def test_legacy_record_without_a_hash_trusts_the_plugin_config(monkeypatch, tmp_path, capsys):
    """Enrolled before the record carried a hash. Enrollment wrote the device token into
    the plugin config and nothing replaces it while the record exists, so it wins, with
    the same notice. The record is not rewritten by a hook."""
    rec = _record(hashed=False, server=None)
    _setup(monkeypatch, tmp_path, record=rec, env={"SYNAPSE_INGEST_TOKEN": _STALE})
    cfg = _load("config")
    assert cfg.INGEST_TOKEN == _DEVICE
    assert cfg.TOKEN_OVERRIDE == {"surface_id": _SURFACE_ID, "sources": ["env"]}
    assert "is ignored" in _board(monkeypatch, capsys)
    assert json.loads((tmp_path / "data" / "device.json").read_text()) == rec


def test_legacy_record_with_an_empty_plugin_config_changes_nothing(monkeypatch, tmp_path):
    _setup(
        monkeypatch,
        tmp_path,
        saved=None,
        record=_record(hashed=False),
        env={"SYNAPSE_INGEST_TOKEN": _STALE},
    )
    cfg = _load("config")
    assert cfg.INGEST_TOKEN == _STALE and cfg.TOKEN_OVERRIDE == {}


def test_a_hash_matching_nothing_changes_nothing(monkeypatch, tmp_path):
    """The configured token was replaced by hand after enrolling: the record no longer
    describes it, so no value is singled out as the device token."""
    _setup(
        monkeypatch,
        tmp_path,
        saved=_OTHER,
        record=_record(),
        env={"SYNAPSE_INGEST_TOKEN": _STALE},
    )
    cfg = _load("config")
    assert cfg.INGEST_TOKEN == _STALE and cfg.TOKEN_OVERRIDE == {}


@pytest.mark.parametrize("var", ["SYNAPSE_URL", "SYNAPSE_INGEST_URL"])
def test_the_device_token_never_goes_to_another_server(monkeypatch, tmp_path, var):
    """An env URL pointing elsewhere means the env token is that server's. Sending the
    device token there would hand a credential to an origin that never had it."""
    _setup(
        monkeypatch,
        tmp_path,
        record=_record(),
        env={"SYNAPSE_INGEST_TOKEN": _STALE, var: "https://other.example.org"},
    )
    cfg = _load("config")
    assert cfg.INGEST_TOKEN == _STALE and cfg.TOKEN_OVERRIDE == {}


def test_the_same_server_spelled_differently_still_overrides(monkeypatch, tmp_path):
    _setup(
        monkeypatch,
        tmp_path,
        record=_record(),
        env={"SYNAPSE_INGEST_TOKEN": _STALE, "SYNAPSE_URL": "HTTPS://Synapse.Example.net/mcp"},
    )
    assert _load("config").INGEST_TOKEN == _DEVICE


# ---------------------------------------------------------------------------
# Enrollment writes the fingerprint, never the token
# ---------------------------------------------------------------------------

_MINTED = {
    "status": "ok",
    "token": _DEVICE,
    "login": "owner",
    "surface": {"surface_id": _SURFACE_ID, "trust": "full", "allowed_projects": []},
}


def test_enrollment_records_the_hash_and_server_but_not_the_token(monkeypatch, tmp_path, capsys):
    _setup(monkeypatch, tmp_path, saved=None, env={"SYNAPSE_INGEST_TOKEN": _STALE})
    enroll = _load("enroll")
    state = enroll._persist(dict(_MINTED))
    assert state["token_sha256"] == _sha(_DEVICE)
    assert state["server"] == _URL
    raw = (tmp_path / "data" / "device.json").read_text()
    _no_token_material(raw)
    _no_token_material(capsys.readouterr().out)

    # The next hook process prefers the minted token over the stale env value.
    assert _load("config").INGEST_TOKEN == _DEVICE


def test_reenroll_replaces_the_hash(monkeypatch, tmp_path):
    """`synapse-login --reenroll` drops the record, stores the root token, enrolls again:
    the new record pins the NEW token, and the old one is no longer preferred."""
    new = "tkNEWdev7Lp0Zs3Vb1"
    _setup(monkeypatch, tmp_path, record=_record(), env={"SYNAPSE_INGEST_TOKEN": _STALE})
    login = _load("synapse_login")
    import enroll  # the module object the script's lazy import returns

    monkeypatch.setattr(login.time, "sleep", lambda _s: None)
    monkeypatch.setattr(enroll.time, "sleep", lambda _s: None)
    monkeypatch.setattr(login.webbrowser, "open", lambda *_a, **_k: True)
    start = {
        "device_code": "dc",
        "user_code": "CODE-0001",
        "verification_uri": "https://idp.example.net/device",
        "interval": 0,
        "expires_in": 5,
    }

    def fake_login_post(url, payload):
        return dict(start) if url.endswith("/device/code") else {"token": "root-token"}

    def fake_enroll_post(path, payload, timeout=30.0):
        if path == "/device/code":
            return dict(start)
        return {**_MINTED, "token": new}

    monkeypatch.setattr(login, "_post_json", fake_login_post)
    monkeypatch.setattr(enroll.config, "post_json", fake_enroll_post)
    monkeypatch.setattr(sys, "argv", ["synapse_login.py", "--reenroll"])
    assert login.main() == 0

    record = json.loads((tmp_path / "data" / "device.json").read_text())
    assert record["token_sha256"] == _sha(new) != _sha(_DEVICE)
    _no_token_material(json.dumps(record))
    assert new not in json.dumps(record)
    assert _load("config").INGEST_TOKEN == new


# ---------------------------------------------------------------------------
# Codex mirror
# ---------------------------------------------------------------------------


def _codex_common(monkeypatch, tmp_path, *, record, env_token):
    """plugin-codex/scripts/common.py reads ~/.claude/settings.json and the Claude
    plugin's device.json; HOME points both at the scratch dir."""
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True, exist_ok=True)
    (home / ".claude" / "settings.json").write_text(
        json.dumps(
            {
                "pluginConfigs": {
                    "synapse@synapse": {
                        "options": {"SYNAPSE_URL": _URL, "SYNAPSE_INGEST_TOKEN": _DEVICE}
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    if record is not None:
        (data / "device.json").write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SYNAPSE_DATA_DIR", str(data))
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    if env_token:
        monkeypatch.setenv("SYNAPSE_INGEST_TOKEN", env_token)
    sys.modules.pop("common", None)
    spec = importlib.util.spec_from_file_location("common", _CODEX / "scripts" / "common.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_codex_enrolled_prefers_the_saved_device_token(monkeypatch, tmp_path, capsys):
    common = _codex_common(monkeypatch, tmp_path, record=_record(), env_token=_STALE)
    assert common.TOKEN == _DEVICE
    notice = common.token_override_notice()
    assert notice.startswith("[Synapse] SYNAPSE_INGEST_TOKEN from your environment is ignored")
    assert _SURFACE_ID in notice
    _no_token_material(notice)

    # The MCP header helper, the Stop hook and the board all read common.TOKEN, so
    # Codex no longer needs `env -u SYNAPSE_INGEST_TOKEN` for the saved token to win.
    monkeypatch.setitem(sys.modules, "common", common)
    spec = importlib.util.spec_from_file_location(
        "mcp_headers_t", _CODEX / "scripts/mcp_headers.py"
    )
    assert spec and spec.loader
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    monkeypatch.setattr(sys, "argv", ["mcp_headers.py", "--url", f"{_URL}/mcp"])
    assert helper.main() == 0
    assert json.loads(capsys.readouterr().out) == {"Authorization": f"Bearer {_DEVICE}"}


def test_codex_legacy_record_and_no_conflict(monkeypatch, tmp_path):
    legacy = _codex_common(
        monkeypatch, tmp_path, record=_record(hashed=False, server=None), env_token=_STALE
    )
    assert legacy.TOKEN == _DEVICE and legacy.token_override_notice()
    calm = _codex_common(monkeypatch, tmp_path, record=_record(), env_token=None)
    assert calm.TOKEN == _DEVICE and calm.token_override_notice() == ""


def test_codex_unenrolled_keeps_env_first(monkeypatch, tmp_path):
    common = _codex_common(monkeypatch, tmp_path, record=None, env_token=_STALE)
    assert common.TOKEN == _STALE
    assert common.token_override_notice() == ""


def test_codex_stop_hook_uses_the_shared_resolution(monkeypatch, tmp_path):
    _codex_common(monkeypatch, tmp_path, record=_record(), env_token=_STALE)
    sys.modules.pop("common", None)
    spec = importlib.util.spec_from_file_location(
        "codex_stop_hook_t", _CODEX / "hooks" / "synapse_stop_hook.py"
    )
    assert spec and spec.loader
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    assert hook.TOKEN == _DEVICE
    sys.modules.pop("common", None)
