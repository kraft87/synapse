"""~/.synapse/config.json is the client's own config file, ranked above the Claude plugin's.

Until 0.20 every client setting lived in Claude Code's settings.json (plugin options) and
the enrollment record in the Claude plugin's data dir, so a machine without Claude Code had
no URL, no credential and no way to log in. config.json (plugin/scripts/synapse_home.py)
is now the one file both plugins and the `synapse` CLI read and write; the legacy places
stay readable underneath it, so an existing install keeps working.

  * synapse_home: load() never raises; save() is atomic and owner-only; update() merges
    and deletes on None; option() renders values as the env var would carry them;
    plugin_options() merges the legacy synapse@<any> blocks
  * config.py resolves env -> config.json -> CLAUDE_PLUGIN_OPTION_* -> settings.json ->
    default. config.json must outrank the plugin options because Claude Code keeps
    exporting answers to options the plugin no longer declares: a stale "1" must not
    beat an explicit `synapse config skills_sync off`
  * the device token and its record live in config.json; the record still pins the
    token by hash, still only toward an origin that already has it, and a legacy
    device.json is read only when config.json has no record
  * store_token() writes the token and its record in one save, and touches settings.json
    only to keep an existing plaintext slot from going stale
  * machine_role() sees a role set earlier in the same process (login asks, then enrolls)
  * the Codex mirror (plugin-codex/scripts/common.py) resolves the same way

Stdlib-only scripts loaded by path; HOME, SYNAPSE_HOME, CLAUDE_CONFIG_DIR, SYNAPSE_DATA_DIR
and the cwd all point into tmp_path. No network: HTTP goes to a fake urlopen.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import stat
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "plugin" / "scripts"
_CODEX_SCRIPTS = _REPO / "plugin-codex" / "scripts"
_MODULES = (
    "config",
    "enroll",
    "synapse_login",
    "harness",
    "synapse_cli",
    "synapse_home",
    "common",
)

_KEYS = (
    "SYNAPSE_URL",
    "SYNAPSE_INGEST_URL",
    "SYNAPSE_RECALL_URL",
    "SYNAPSE_MCP_URL",
    "SYNAPSE_INGEST_TOKEN",
    "SYNAPSE_MACHINE_ROLE",
    "SYNAPSE_SKILLS_SYNC",
    "SYNAPSE_CONFIG_SYNC",
    "SYNAPSE_CONFIG_PATHS",
)
_ENV_VARS = _KEYS + tuple(f"CLAUDE_PLUGIN_OPTION_{k}" for k in _KEYS)

# Distinctive values, so "no token material in the output" can check prefixes too.
_DEVICE = "tkDEV4c8Rw2YhN0p"
_STALE = "tkENV7m1Qz5Tb3Ks"
_OTHER = "tkOPT2v9Jx6Lc8Fd"
_URL = "https://synapse.example.net"
_FOREIGN = "https://other.example.org"
_SURFACE_ID = "dev-abc123"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Everything the client reads lives under tmp_path: HOME (the Codex mirror's
    ~/.claude/settings.json), SYNAPSE_HOME (config.json), CLAUDE_CONFIG_DIR (the Claude
    plugin's settings.json), SYNAPSE_DATA_DIR (legacy device.json) and a cwd with no
    project .claude. No setting arrives through the real environment."""
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SYNAPSE_HOME", str(home / ".synapse"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home / ".claude"))
    monkeypatch.setenv("SYNAPSE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.chdir(tmp_path)
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    yield
    _forget_modules()


def _forget_modules() -> None:
    for mod in _MODULES:
        sys.modules.pop(mod, None)


def _exec(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_home() -> ModuleType:
    _forget_modules()
    return _exec("synapse_home", _SCRIPTS / "synapse_home.py")


def _load_config() -> ModuleType:
    """config.py as a fresh hook process would import it: it reads both config sources
    and resolves the token at import."""
    _forget_modules()
    return _exec("config", _SCRIPTS / "config.py")


def _load_common() -> ModuleType:
    _forget_modules()
    return _exec("common", _CODEX_SCRIPTS / "common.py")


def _codex_skills_sync(monkeypatch, common: ModuleType) -> ModuleType:
    """plugin-codex/scripts/skills_sync.py, which does `from common import _cfg` after
    prepending its own dir to sys.path; both are undone after the test."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setitem(sys.modules, "common", common)
    return _exec("synapse_codex_skills_sync_cfgfile", _CODEX_SCRIPTS / "skills_sync.py")


def _config_json() -> Path:
    return Path(os.environ["SYNAPSE_HOME"]) / "config.json"


def _write_home(data: dict) -> None:
    path = _config_json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _read_home() -> dict:
    return json.loads(_config_json().read_text(encoding="utf-8"))


def _settings_path() -> Path:
    return Path(os.environ["CLAUDE_CONFIG_DIR"]) / "settings.json"


def _write_settings(options: dict, **top_level) -> None:
    """The user settings.json, with the Claude plugin's saved answers under a
    non-default marketplace name (synapse@<any> must match)."""
    path = _settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"pluginConfigs": {"synapse@some-marketplace": {"options": options}}, **top_level}
    path.write_text(json.dumps(data), encoding="utf-8")


def _legacy_device_file() -> Path:
    return Path(os.environ["SYNAPSE_DATA_DIR"]) / "device.json"


def _write_legacy_record(record: dict) -> None:
    path = _legacy_device_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")


def _sha(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _record(token: str = _DEVICE, *, surface: str = _SURFACE_ID, server: str = _URL) -> dict:
    return {"surface_id": surface, "trust": "full", "token_sha256": _sha(token), "server": server}


def _no_token_material(text: str) -> None:
    for tok in (_DEVICE, _STALE, _OTHER):
        assert tok not in text
        assert tok[:6] not in text and tok[-6:] not in text


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


def _capture_requests(monkeypatch, cfg: ModuleType) -> list[tuple[str, str | None]]:
    """(url, Authorization header) for every request the module sends."""
    seen: list[tuple[str, str | None]] = []

    def fake_urlopen(req, timeout=None):
        seen.append((req.full_url, req.get_header("Authorization")))
        return _Resp(b"{}")

    monkeypatch.setattr(cfg.urllib.request, "urlopen", fake_urlopen)
    return seen


# ---------------------------------------------------------------------------
# synapse_home: the file itself
# ---------------------------------------------------------------------------


def test_load_returns_an_empty_dict_for_a_missing_malformed_or_non_object_file(tmp_path):
    """Hooks call load() at import, so a bad file must read as "nothing configured"
    rather than kill every hook with a traceback."""
    home = _load_home()
    assert home.load() == {}  # SYNAPSE_HOME has no config.json yet
    assert home.load(tmp_path / "missing.json") == {}
    for name, raw in (
        ("malformed.json", b"{not json"),
        ("list.json", b"[1, 2]"),
        ("string.json", b'"text"'),
        ("binary.json", b"\xff\xfe\x00"),
    ):
        path = tmp_path / name
        path.write_bytes(raw)
        assert home.load(path) == {}, name


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_save_creates_the_home_dir_owner_only_and_writes_the_file_0600():
    home = _load_home()
    home.save({"url": _URL, "token": _DEVICE})
    path = _config_json()
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert home.load() == {"url": _URL, "token": _DEVICE}
    assert path.read_text(encoding="utf-8").endswith("\n")
    # The temp file was renamed over the target, not left beside it.
    assert [p.name for p in path.parent.iterdir()] == ["config.json"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_save_replaces_a_world_readable_file_with_an_owner_only_one():
    """A file someone created by hand (or with a loose umask) holds a bearer token once
    the client writes it, so the rewrite must not inherit its mode."""
    _write_home({"url": _URL})
    _config_json().chmod(0o644)
    _load_home().save({"url": _URL, "token": _DEVICE})
    assert stat.S_IMODE(_config_json().stat().st_mode) == 0o600


def test_a_failed_save_leaves_the_previous_file_and_no_temp_file():
    """json.dump writes in chunks, so an unserialisable value fails midway through the
    temp file. The live file must still be the previous one, whole."""
    home = _load_home()
    home.save({"token": "old"})
    with pytest.raises(TypeError):
        home.save({"token": "new", "broken": object()})
    assert home.load() == {"token": "old"}
    assert [p.name for p in _config_json().parent.iterdir()] == ["config.json"]


def test_log_appends_timestamped_lines_under_the_synapse_home():
    """Hook logs used to default to /tmp, which Windows doesn't have (the work laptop
    logged nothing) and POSIX shares between users."""
    home = _load_home()
    home.log("ingest-hook", "OK s.jsonl")
    home.log("ingest-hook", "ERR boom")
    lines = (Path(os.environ["SYNAPSE_HOME"]) / "logs" / "ingest-hook.log").read_text().splitlines()
    assert [line.split(" ", 1)[1] for line in lines] == ["OK s.jsonl", "ERR boom"]
    assert lines[0].split(" ", 1)[0].startswith("20")  # ISO timestamp


def test_log_restarts_a_file_past_64kb():
    home = _load_home()
    path = Path(os.environ["SYNAPSE_HOME"]) / "logs" / "spool.log"
    path.parent.mkdir(parents=True)
    path.write_text("x" * (64 * 1024 + 1), encoding="utf-8")
    home.log("spool", "fresh")
    assert path.read_text().endswith("fresh\n") and len(path.read_text()) < 200


def test_log_never_raises(tmp_path, monkeypatch):
    """The Synapse home can't be created (it is a file) or the message can't be encoded:
    a hook must not die over its log."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    monkeypatch.setenv("SYNAPSE_HOME", str(blocker))
    home = _load_home()
    home.log("x", "msg")
    monkeypatch.setenv("SYNAPSE_HOME", str(tmp_path / "ok"))
    home.log("x", "lone surrogate \ud800")


def test_update_merges_changes_and_a_none_value_deletes_the_key():
    home = _load_home()
    assert home.update({"url": _URL}) == {"url": _URL}  # creates the file when absent
    home.save({"url": _URL, "token": _DEVICE, "role": "work"})
    new = home.update({"role": None, "skills_sync": True, "never_set": None})
    assert new == {"url": _URL, "token": _DEVICE, "skills_sync": True}
    assert home.load() == new


def test_option_renders_values_as_the_env_var_would_carry_them():
    home = _load_home()
    data = {
        "url": _URL,
        "role": "work",
        "skills_sync": True,
        "config_sync": False,
        "config_paths": ["output-styles/*.md", "agents/*.md"],
    }
    assert home.option(data, "SYNAPSE_URL") == _URL
    assert home.option(data, "SYNAPSE_MACHINE_ROLE") == "work"
    assert home.option(data, "SYNAPSE_SKILLS_SYNC") == "1"
    assert home.option(data, "SYNAPSE_CONFIG_SYNC") == "0"
    assert home.option(data, "SYNAPSE_CONFIG_PATHS") == "output-styles/*.md,agents/*.md"
    assert home.option(data, "SYNAPSE_INGEST_TOKEN") == ""  # known key, not set
    # Names are env-style: the config.json key itself and unknown names read as unset.
    assert home.option(data, "url") == ""
    assert home.option({**data, "SYNAPSE_SURFACE": "x"}, "SYNAPSE_SURFACE") == ""


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1", True),
        ("true", True),
        ("TRUE", True),
        (" yes ", True),
        ("y", True),
        ("On", True),
        ("0", False),
        ("false", False),
        ("No", False),
        ("n", False),
        ("OFF", False),
        ("", None),
        ("maybe", None),
        ("2", None),
        ("enabled", None),
    ],
)
def test_parse_bool_reads_yes_no_on_off_one_zero_and_nothing_else(text, expected):
    assert _load_home().parse_bool(text) is expected


def test_plugin_options_merges_every_synapse_block_with_later_files_winning(tmp_path):
    first, second = tmp_path / "user.json", tmp_path / "project.json"
    first.write_text(
        json.dumps(
            {
                "pluginConfigs": {
                    "synapse@marketplace-a": {
                        "options": {
                            "SYNAPSE_URL": "https://first.example.net",
                            "SYNAPSE_INGEST_TOKEN": _DEVICE,
                            "SYNAPSE_SKILLS_SYNC": "1",
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "pluginConfigs": {
                    "synapse@marketplace-b": {
                        "options": {
                            "SYNAPSE_URL": _URL,
                            # Empty answers are "not answered": they don't erase the first.
                            "SYNAPSE_INGEST_TOKEN": "",
                            "SYNAPSE_SKILLS_SYNC": None,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    assert _load_home().plugin_options([first, second]) == {
        "SYNAPSE_URL": _URL,
        "SYNAPSE_INGEST_TOKEN": _DEVICE,
        "SYNAPSE_SKILLS_SYNC": "1",
    }


def test_plugin_options_ignores_other_plugins_and_unreadable_files(tmp_path):
    good = tmp_path / "settings.json"
    good.write_text(
        json.dumps(
            {
                "pluginConfigs": {
                    "other@synapse": {"options": {"SYNAPSE_URL": "https://wrong.example"}},
                    "synapse-extra@x": {"options": {"SYNAPSE_URL": "https://wrong.example"}},
                    "synapse@not-a-dict": "options",
                    "synapse@synapse": {"options": {"SYNAPSE_URL": _URL}},
                }
            }
        ),
        encoding="utf-8",
    )
    malformed, non_object, no_configs = (
        tmp_path / "malformed.json",
        tmp_path / "list.json",
        tmp_path / "plain.json",
    )
    malformed.write_text("{not json", encoding="utf-8")
    non_object.write_text("[1, 2]", encoding="utf-8")
    no_configs.write_text('{"theme": "dark", "pluginConfigs": null}', encoding="utf-8")
    files = [good, malformed, non_object, no_configs, tmp_path / "missing.json"]
    assert _load_home().plugin_options(files) == {"SYNAPSE_URL": _URL}


@pytest.mark.parametrize(
    "settings",
    [
        {"pluginConfigs": ["synapse@synapse"]},
        {"pluginConfigs": {"synapse@synapse": {"options": ["SYNAPSE_URL"]}}},
    ],
    ids=["plugin-configs-a-list", "options-a-list"],
)
def test_plugin_options_skips_a_settings_file_whose_plugin_configs_has_the_wrong_shape(
    tmp_path, settings
):
    bad, good = tmp_path / "bad.json", tmp_path / "good.json"
    bad.write_text(json.dumps(settings), encoding="utf-8")
    good.write_text(
        json.dumps({"pluginConfigs": {"synapse@synapse": {"options": {"SYNAPSE_URL": _URL}}}}),
        encoding="utf-8",
    )
    assert _load_home().plugin_options([bad, good]) == {"SYNAPSE_URL": _URL}


# ---------------------------------------------------------------------------
# config.py: where config.json ranks
# ---------------------------------------------------------------------------


def test_the_config_json_url_drives_every_endpoint():
    _write_home({"url": "https://syn.example.net/"})
    cfg = _load_config()
    assert cfg.BASE_URL == "https://syn.example.net"
    assert cfg.INGEST_URL == "https://syn.example.net/ingest"
    assert cfg.RECALL_URL == "https://syn.example.net/recall"
    assert cfg.MCP_URL == "https://syn.example.net/mcp"
    assert cfg.SKILLS_URL == "https://syn.example.net/skills"


def test_config_json_beats_settings_json_and_the_plugin_options_claude_code_exports(
    monkeypatch,
):
    _write_settings(
        {
            "SYNAPSE_URL": "https://settings.example.net",
            "SYNAPSE_MACHINE_ROLE": "personal",
            "SYNAPSE_SKILLS_SYNC": "0",
            "SYNAPSE_CONFIG_SYNC": "0",
        }
    )
    monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_SYNAPSE_URL", "https://option.example.net")
    monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_SYNAPSE_MACHINE_ROLE", "personal")
    _write_home({"url": _URL, "role": "work", "skills_sync": True, "config_sync": True})
    cfg = _load_config()
    assert cfg.BASE_URL == _URL and cfg.INGEST_URL == f"{_URL}/ingest"
    assert cfg.MACHINE_ROLE == "work"
    assert cfg.SKILLS_SYNC is True
    assert cfg.CONFIG_SYNC is True


def test_an_env_var_beats_config_json(monkeypatch):
    _write_home({"url": _URL, "role": "work", "skills_sync": True})
    monkeypatch.setenv("SYNAPSE_URL", "http://envhost:1111")
    monkeypatch.setenv("SYNAPSE_MACHINE_ROLE", "personal")
    monkeypatch.setenv("SYNAPSE_SKILLS_SYNC", "0")
    cfg = _load_config()
    assert cfg.BASE_URL == "http://envhost:1111"
    assert cfg.MACHINE_ROLE == "personal"
    assert cfg.SKILLS_SYNC is False


@pytest.mark.parametrize("stale_in", ["plugin_option", "settings_json", "both"])
def test_an_explicit_off_in_config_json_beats_a_stale_on_from_the_legacy_options(
    monkeypatch, stale_in
):
    """The four options 0.20 stopped declaring are still exported by Claude Code as
    CLAUDE_PLUGIN_OPTION_* and still sit in settings.json until `synapse login` drops
    them. `synapse config skills_sync off` writes a JSON false, which must win."""
    if stale_in in ("plugin_option", "both"):
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_SYNAPSE_SKILLS_SYNC", "1")
        monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_SYNAPSE_CONFIG_SYNC", "1")
    if stale_in in ("settings_json", "both"):
        _write_settings({"SYNAPSE_SKILLS_SYNC": "1", "SYNAPSE_CONFIG_SYNC": "1"})
    stale = _load_config()
    assert stale.SKILLS_SYNC is True and stale.CONFIG_SYNC is True  # the stale value is live

    _write_home({"skills_sync": False, "config_sync": False})
    cfg = _load_config()
    assert cfg.SKILLS_SYNC is False
    assert cfg.CONFIG_SYNC is False


def test_config_paths_from_config_json_reach_config_paths(monkeypatch):
    _write_settings({"SYNAPSE_CONFIG_PATHS": "stale/*.md"})
    monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_SYNAPSE_CONFIG_PATHS", "stale/*.md")
    _write_home({"config_paths": ["output-styles/*.md", "agents/*.md"]})
    assert _load_config().CONFIG_PATHS == ["output-styles/*.md", "agents/*.md"]


def test_an_explicitly_empty_config_paths_list_beats_stale_legacy_globs(monkeypatch):
    _write_settings({"SYNAPSE_CONFIG_PATHS": "stale/*.md"})
    monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_SYNAPSE_CONFIG_PATHS", "stale/*.md")
    _write_home({"config_paths": []})
    assert _load_config().CONFIG_PATHS == []


def test_the_role_in_config_json_sets_the_machine_role_and_requested_trust():
    _write_home({"role": "work"})
    cfg = _load_config()
    assert cfg.MACHINE_ROLE == "work"
    assert cfg.machine_role() == "work"
    assert cfg.requested_trust() == "restricted"


def test_a_role_set_in_this_process_is_seen_without_reloading():
    """`synapse login` asks the role, writes it, then enrolls in the same process: the
    enrollment must declare the answer just given, not the import-time value."""
    cfg = _load_config()
    assert (cfg.machine_role(), cfg.requested_trust()) == ("personal", "full")

    cfg.set_home_values({"role": "work"})
    assert (cfg.machine_role(), cfg.requested_trust()) == ("work", "restricted")
    assert _read_home() == {"role": "work"}

    cfg.set_home_values({"role": None})
    assert (cfg.machine_role(), cfg.requested_trust()) == ("personal", "full")
    assert _read_home() == {}


# ---------------------------------------------------------------------------
# config.py: the token
# ---------------------------------------------------------------------------


def test_the_config_json_token_is_used_when_nothing_else_is_set(monkeypatch):
    _write_home({"url": _URL, "token": _DEVICE})
    cfg = _load_config()
    assert cfg.INGEST_TOKEN == _DEVICE
    assert cfg.TOKEN_OVERRIDE == {}
    seen = _capture_requests(monkeypatch, cfg)
    cfg.post_json("/skills/list", {})
    assert seen == [(f"{_URL}/skills/list", f"Bearer {_DEVICE}")]


def test_unenrolled_env_beats_config_json_which_beats_both_plugin_option_sources(
    monkeypatch,
):
    """Before enrollment the slot carries whatever token the user put there, so plain
    precedence applies, with config.json above the legacy plugin options."""
    _write_settings({"SYNAPSE_URL": _URL, "SYNAPSE_INGEST_TOKEN": "tkSETTINGSjson01"})
    monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_TOKEN", "tkPLUGINoption01")
    _write_home({"token": _OTHER})
    monkeypatch.setenv("SYNAPSE_INGEST_TOKEN", _STALE)
    assert _load_config().INGEST_TOKEN == _STALE

    monkeypatch.delenv("SYNAPSE_INGEST_TOKEN")
    cfg = _load_config()
    assert cfg.INGEST_TOKEN == _OTHER
    assert cfg.TOKEN_OVERRIDE == {}


def test_an_enrolled_machine_prefers_its_config_json_device_token_over_a_stale_env_token(
    monkeypatch,
):
    _write_home({"url": _URL, "token": _DEVICE, "device": _record()})
    monkeypatch.setenv("SYNAPSE_INGEST_TOKEN", _STALE)
    cfg = _load_config()
    assert cfg.INGEST_TOKEN == _DEVICE
    assert cfg.TOKEN_OVERRIDE == {"surface_id": _SURFACE_ID, "sources": ["env"]}
    notice = cfg.token_override_notice()
    assert notice.startswith("[Synapse] SYNAPSE_INGEST_TOKEN from your environment is ignored")
    assert _SURFACE_ID in notice
    _no_token_material(notice)

    seen = _capture_requests(monkeypatch, cfg)
    cfg.post_json(cfg.INGEST_URL, {})
    assert seen == [(f"{_URL}/ingest", f"Bearer {_DEVICE}")]


def test_a_config_json_token_the_record_does_not_describe_is_overridden_and_named():
    """The record pins the settings.json token; config.json holds another (a hand edit).
    The device token wins, and the session-start line points at config.json and the two
    commands that resolve it, without a byte of either token."""
    _write_settings({"SYNAPSE_URL": _URL, "SYNAPSE_INGEST_TOKEN": _DEVICE})
    _write_home({"token": _OTHER, "device": _record(_DEVICE)})
    cfg = _load_config()
    assert cfg.INGEST_TOKEN == _DEVICE
    assert cfg.TOKEN_OVERRIDE == {"surface_id": _SURFACE_ID, "sources": ["config_file"]}
    notice = cfg.token_override_notice()
    assert notice.startswith("[Synapse] The token in ~/.synapse/config.json is ignored")
    assert _SURFACE_ID in notice
    assert "synapse login --reenroll" in notice and "synapse config token --unset" in notice
    _no_token_material(notice)


@pytest.mark.parametrize(
    ("var", "value"),
    [("SYNAPSE_URL", _FOREIGN), ("SYNAPSE_INGEST_URL", f"{_FOREIGN}/ingest")],
)
def test_the_device_token_never_goes_to_a_foreign_origin(monkeypatch, var, value):
    """An env URL pointing at another server means the env token is that server's.
    The device token must not go to an origin that never had it."""
    _write_home({"url": _URL, "token": _DEVICE, "device": _record()})
    monkeypatch.setenv("SYNAPSE_INGEST_TOKEN", _STALE)
    monkeypatch.setenv(var, value)
    cfg = _load_config()
    assert cfg.INGEST_TOKEN == _STALE
    assert cfg.TOKEN_OVERRIDE == {}
    assert cfg.token_override_notice() == ""

    seen = _capture_requests(monkeypatch, cfg)
    cfg.post_json(cfg.INGEST_URL, {})
    assert seen == [(f"{_FOREIGN}/ingest", f"Bearer {_STALE}")]


def test_a_legacy_device_json_is_still_read_when_config_json_has_no_record(monkeypatch):
    """A pre-0.20 install that hasn't run `synapse login` since keeps its enrollment."""
    record = _record()
    _write_legacy_record(record)
    _write_home({"url": _URL, "token": _DEVICE})
    monkeypatch.setenv("SYNAPSE_INGEST_TOKEN", _STALE)
    cfg = _load_config()
    assert cfg.read_device_state() == record
    assert cfg.INGEST_TOKEN == _DEVICE
    assert cfg.TOKEN_OVERRIDE == {"surface_id": _SURFACE_ID, "sources": ["env"]}


def test_a_config_json_record_wins_over_a_legacy_device_json(monkeypatch):
    """The old record pins a token nothing holds any more; were it read, nothing would
    match and the stale env token would win."""
    _write_legacy_record(_record(_OTHER, surface="dev-old"))
    _write_home({"url": _URL, "token": _DEVICE, "device": _record(surface="dev-new")})
    monkeypatch.setenv("SYNAPSE_INGEST_TOKEN", _STALE)
    cfg = _load_config()
    assert cfg.read_device_state()["surface_id"] == "dev-new"
    assert cfg.INGEST_TOKEN == _DEVICE
    assert cfg.TOKEN_OVERRIDE == {"surface_id": "dev-new", "sources": ["env"]}


# ---------------------------------------------------------------------------
# config.py: storing and forgetting the credential
# ---------------------------------------------------------------------------


def test_store_token_writes_the_token_and_its_record_in_one_save(monkeypatch):
    """A crash between two writes could leave a record describing a token that isn't
    there, which locks the machine out of a credential it can't obtain again."""
    _write_home({"url": _URL, "role": "work"})
    cfg = _load_config()
    writes: list[dict] = []
    real_save = cfg.synapse_home.save

    def spy(data, path=None):
        writes.append(json.loads(json.dumps(data)))
        real_save(data, path)

    monkeypatch.setattr(cfg.synapse_home, "save", spy)
    record = _record()
    cfg.store_token(_DEVICE, record)

    expected = {"url": _URL, "role": "work", "token": _DEVICE, "device": record}
    assert writes == [expected]
    assert _read_home() == expected
    _no_token_material(json.dumps(record))
    assert not _settings_path().exists()
    # The next hook process resolves to it.
    assert _load_config().INGEST_TOKEN == _DEVICE


def test_store_token_keeps_a_legacy_plaintext_settings_slot_in_step():
    """settings.json's token feeds the Claude plugin's static Authorization header, the
    fallback when the headers helper can't run; a stale copy there would be sent."""
    _write_settings({"SYNAPSE_URL": _URL, "SYNAPSE_INGEST_TOKEN": "old"}, theme="dark")
    cfg = _load_config()
    cfg.store_token(_DEVICE, _record())
    settings = json.loads(_settings_path().read_text(encoding="utf-8"))
    assert settings == {
        "pluginConfigs": {
            "synapse@some-marketplace": {
                "options": {"SYNAPSE_URL": _URL, "SYNAPSE_INGEST_TOKEN": _DEVICE}
            }
        },
        "theme": "dark",
    }
    assert _read_home()["token"] == _DEVICE


@pytest.mark.parametrize("settings", ["absent", "without-a-token"])
def test_store_token_never_creates_a_settings_token_slot(settings):
    if settings == "without-a-token":
        _write_settings({"SYNAPSE_URL": _URL})
        before = _settings_path().read_bytes()
    cfg = _load_config()
    cfg.store_token(_DEVICE, _record())
    if settings == "absent":
        assert not _settings_path().exists()
    else:
        assert _settings_path().read_bytes() == before
    assert _read_home()["token"] == _DEVICE


def test_write_user_config_changes_only_its_key_and_leaves_no_temp_file():
    _write_settings({"SYNAPSE_URL": _URL, "other": "kept"}, theme="dark")
    cfg = _load_config()
    cfg.write_user_config("SYNAPSE_URL", _FOREIGN)
    assert json.loads(_settings_path().read_text(encoding="utf-8")) == {
        "pluginConfigs": {
            "synapse@some-marketplace": {"options": {"SYNAPSE_URL": _FOREIGN, "other": "kept"}}
        },
        "theme": "dark",
    }
    assert cfg._FILE_OPTIONS["SYNAPSE_URL"] == _FOREIGN
    assert [p.name for p in _settings_path().parent.iterdir()] == ["settings.json"]


def test_write_user_config_refuses_a_settings_file_it_cannot_parse():
    """The user's Claude Code settings must never be replaced by a file holding only the
    keys this code knows about."""
    _settings_path().parent.mkdir(parents=True, exist_ok=True)
    _settings_path().write_text("{oops", encoding="utf-8")
    cfg = _load_config()
    with pytest.raises(RuntimeError, match="isn't valid JSON"):
        cfg.write_user_config("SYNAPSE_URL", _URL)
    assert _settings_path().read_text(encoding="utf-8") == "{oops"


def test_a_crash_while_writing_settings_leaves_the_previous_file(monkeypatch):
    _write_settings({"SYNAPSE_URL": _URL})
    before = _settings_path().read_bytes()
    cfg = _load_config()

    def crash(src, dst):
        raise OSError("disk went away")

    monkeypatch.setattr(cfg.synapse_home.os, "replace", crash)
    with pytest.raises(OSError, match="disk went away"):
        cfg.write_user_config("SYNAPSE_URL", _FOREIGN)
    assert _settings_path().read_bytes() == before
    assert [p.name for p in _settings_path().parent.iterdir()] == ["settings.json"]


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_settings_written_through_a_symlink_keep_the_link(tmp_path):
    """A dotfiles repo often owns settings.json; replacing the link would orphan it."""
    real = tmp_path / "dotfiles" / "settings.json"
    real.parent.mkdir()
    real.write_text(json.dumps({"theme": "dark"}), encoding="utf-8")
    _settings_path().parent.mkdir(parents=True, exist_ok=True)
    _settings_path().symlink_to(real)
    _load_config().write_user_config("SYNAPSE_URL", _URL)
    assert _settings_path().is_symlink()
    assert json.loads(real.read_text(encoding="utf-8"))["theme"] == "dark"
    assert "SYNAPSE_URL" in real.read_text(encoding="utf-8")


def test_store_token_with_a_record_removes_the_legacy_device_json():
    """Otherwise the old record would resurface the moment config.json lost its own."""
    _write_legacy_record(_record(_OTHER, surface="dev-old"))
    cfg = _load_config()
    record = _record()
    cfg.store_token(_DEVICE, record)
    assert not _legacy_device_file().exists()
    assert cfg.read_device_state() == record


def test_clear_device_state_forgets_the_record_in_both_places():
    _write_legacy_record(_record(_OTHER, surface="dev-old"))
    _write_home({"url": _URL, "token": _DEVICE, "device": _record()})
    cfg = _load_config()
    cfg.clear_device_state()
    assert _read_home() == {"url": _URL, "token": _DEVICE}
    assert not _legacy_device_file().exists()
    assert cfg.read_device_state() == {}


def test_clear_device_state_on_an_unenrolled_machine_writes_nothing():
    cfg = _load_config()
    cfg.clear_device_state()
    assert not _config_json().exists()
    assert not _legacy_device_file().parent.exists()


# ---------------------------------------------------------------------------
# The Codex mirror: plugin-codex/scripts/common.py
# ---------------------------------------------------------------------------


def test_codex_reads_url_token_and_skills_sync_from_config_json(monkeypatch):
    """A Codex-only machine has no ~/.claude at all: `synapse login` alone configures it."""
    _write_home({"url": "https://syn.example.net/", "token": _DEVICE, "skills_sync": True})
    common = _load_common()
    assert common.BASE_URL == "https://syn.example.net"
    assert common.INGEST_URL == "https://syn.example.net/ingest"
    assert common.TOKEN == _DEVICE
    assert common.token_override_notice() == ""
    assert _codex_skills_sync(monkeypatch, common).SKILLS_SYNC is True


def test_codex_config_json_beats_the_claude_settings_json(monkeypatch):
    _write_settings(
        {
            "SYNAPSE_URL": "https://settings.example.net",
            "SYNAPSE_INGEST_TOKEN": _OTHER,
            "SYNAPSE_SKILLS_SYNC": "1",
        }
    )
    _write_home({"url": _URL, "token": _DEVICE, "skills_sync": False})
    common = _load_common()
    assert common.BASE_URL == _URL
    assert common.TOKEN == _DEVICE
    assert _codex_skills_sync(monkeypatch, common).SKILLS_SYNC is False


def test_codex_enrolled_machine_uses_the_config_json_device_token_over_env(monkeypatch):
    # A legacy record pinning another token is ignored once config.json has one.
    _write_legacy_record(_record(_OTHER, surface="dev-old"))
    _write_home({"url": _URL, "token": _DEVICE, "device": _record()})
    monkeypatch.setenv("SYNAPSE_INGEST_TOKEN", _STALE)
    common = _load_common()
    assert common.TOKEN == _DEVICE
    notice = common.token_override_notice()
    assert notice.startswith("[Synapse] SYNAPSE_INGEST_TOKEN from your environment is ignored")
    assert _SURFACE_ID in notice and "dev-old" not in notice
    _no_token_material(notice)

    seen = _capture_requests(monkeypatch, common)
    common.post_json("/ingest", {})
    assert seen == [(f"{_URL}/ingest", f"Bearer {_DEVICE}")]


def test_codex_falls_back_to_the_claude_settings_json_without_config_json(monkeypatch):
    _write_settings(
        {"SYNAPSE_URL": _URL, "SYNAPSE_INGEST_TOKEN": _DEVICE, "SYNAPSE_SKILLS_SYNC": "1"}
    )
    common = _load_common()
    assert not _config_json().exists()
    assert common.BASE_URL == _URL
    assert common.TOKEN == _DEVICE
    assert _codex_skills_sync(monkeypatch, common).SKILLS_SYNC is True
