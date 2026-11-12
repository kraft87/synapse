"""The core's settings (plugin/synapse_client/settings.py) and package metadata.

``Settings.load`` resolves env -> ~/.synapse/config.json -> extra sources -> default from
one snapshot, the order plugin/scripts/config.py's ``_cfg`` uses, and never writes. The
flag test below loads config.py by path over the same scratch home and asserts the same
string for each name, so the two cannot drift while the hooks still use config.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.helpers.client_core import REPO, drop_scripts, load_script, scratch_user, use_core

use_core()
import synapse_client  # noqa: E402
from synapse_client import settings as settings_mod  # noqa: E402
from synapse_client.settings import Settings  # noqa: E402

_TOKEN = "tkSET4r9Wq2Lm8Zx"


def _write_config(home: Path, data: object) -> Path:
    path = home / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture()
def scratch(monkeypatch, tmp_path):
    """A scratch user for loading the hook modules (tests/helpers/client_core.py)."""
    yield scratch_user(monkeypatch, tmp_path)
    drop_scripts()


# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------


def test_version_matches_the_plugin_manifest():
    manifest = json.loads((REPO / "plugin" / ".claude-plugin" / "plugin.json").read_text())
    assert synapse_client.__version__ == manifest["version"]


def test_config_keys_mirror_synapse_home(scratch):
    home = load_script("synapse_home")
    assert dict(settings_mod.KEYS) == home.KEYS


# ---------------------------------------------------------------------------
# Resolution order
# ---------------------------------------------------------------------------


def test_nothing_configured_is_empty_not_localhost(tmp_path):
    s = Settings.load(env={}, home=tmp_path / "h")
    assert (s.base_url, s.ingest_url, s.token, dict(s.token_note)) == ("", "", "", {})
    assert not s.enrolled and dict(s.device) == {} and s.device_origins == frozenset()


def test_env_beats_config_json_beats_extras_beats_default(tmp_path):
    home = tmp_path / "h"
    _write_config(home, {"role": "work"})
    extras = [("plugin_option", {"SYNAPSE_MACHINE_ROLE": "x", "SYNAPSE_BOARD": "0"})]
    s = Settings.load(env={}, home=home, extra_candidates=extras)
    assert s.flag("SYNAPSE_MACHINE_ROLE", "personal") == "work"
    assert s.flag("SYNAPSE_BOARD", "1") == "0"
    assert s.flag("SYNAPSE_PREFS_BLOCK", "1") == "1"
    s = Settings.load(env={"SYNAPSE_MACHINE_ROLE": "personal"}, home=home, extra_candidates=extras)
    assert s.flag("SYNAPSE_MACHINE_ROLE") == "personal"


def test_an_empty_env_value_counts_as_unset(tmp_path):
    home = tmp_path / "h"
    _write_config(home, {"role": "work"})
    assert Settings.load(env={"SYNAPSE_MACHINE_ROLE": ""}, home=home).flag(
        "SYNAPSE_MACHINE_ROLE"
    ) == ("work")


def test_a_config_json_key_set_empty_stops_the_fall_through(tmp_path):
    """A cleared list of globs must not fall through to a stale legacy value."""
    home = tmp_path / "h"
    _write_config(home, {"config_paths": []})
    extras = [("plugin_config", {"SYNAPSE_CONFIG_PATHS": "~/stale/*"})]
    s = Settings.load(env={}, home=home, extra_candidates=extras)
    assert s.flag("SYNAPSE_CONFIG_PATHS", "dflt") == "dflt"


def test_config_json_values_render_as_env_strings(tmp_path):
    home = tmp_path / "h"
    _write_config(home, {"skills_sync": True, "config_sync": False, "config_paths": ["a", "b"]})
    s = Settings.load(env={}, home=home)
    assert s.flag("SYNAPSE_SKILLS_SYNC") == "1"
    assert s.flag("SYNAPSE_CONFIG_SYNC") == "0"
    assert s.flag("SYNAPSE_CONFIG_PATHS") == "a,b"


def test_extras_drop_empty_values_and_stringify_the_rest(tmp_path):
    extras = [
        ("a", {"SYNAPSE_BOARD": "", "SYNAPSE_SURFACE": None}),
        ("b", {"SYNAPSE_BOARD": 0, "SYNAPSE_SURFACE": "laptop"}),
    ]
    s = Settings.load(env={}, home=tmp_path / "h", extra_candidates=extras)
    assert s.flag("SYNAPSE_BOARD") == "0"
    assert s.flag("SYNAPSE_SURFACE") == "laptop"


def test_only_synapse_names_are_kept_from_the_environment(tmp_path):
    env = {"SYNAPSE_SURFACE": "box", "PATH": "/bin", "OTHER_TOKEN": "x"}
    s = Settings.load(env=env, home=tmp_path / "h")
    assert dict(s.environ) == {"SYNAPSE_SURFACE": "box"}
    assert s.flag("PATH") == ""


def test_a_malformed_config_json_reads_as_nothing_configured(tmp_path):
    home = tmp_path / "h"
    home.mkdir()
    (home / "config.json").write_text("{not json", encoding="utf-8")
    assert Settings.load(env={}, home=home).base_url == ""
    (home / "config.json").write_text("[1, 2]", encoding="utf-8")
    assert Settings.load(env={}, home=home).base_url == ""


# ---------------------------------------------------------------------------
# URL, home and paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "base", "ingest"),
    [
        ("https://s.example/", "https://s.example", "https://s.example/ingest"),
        ("https://s.example/mcp/", "https://s.example", "https://s.example/ingest"),
        ("https://s.example/sub/ingest", "https://s.example/sub", "https://s.example/sub/ingest"),
        ("http://mcp", "http://mcp", "http://mcp/ingest"),
    ],
)
def test_the_base_url_drops_a_slash_or_pasted_endpoint(tmp_path, raw, base, ingest):
    s = Settings.load(env={"SYNAPSE_URL": raw}, home=tmp_path / "h")
    assert (s.base_url, s.ingest_url) == (base, ingest)


def test_a_legacy_ingest_url_still_names_the_server_and_the_ingest_endpoint(tmp_path):
    s = Settings.load(env={"SYNAPSE_INGEST_URL": "https://s.example/ingest"}, home=tmp_path)
    assert (s.base_url, s.ingest_url) == ("https://s.example", "https://s.example/ingest")


@pytest.mark.parametrize(
    "raw",
    [
        "https://s.example/install/",
        " http://10.0.0.5:8765/mcp ",
        "http://[::1]:8765",
        "https://s.example/a/b",
    ],
)
def test_normalize_url_matches_the_cli(scratch, raw):
    harness = load_script("harness")
    assert settings_mod.normalize_url(raw) == harness.normalize_url(raw)


@pytest.mark.parametrize("raw", ["", "ftp://x", "https://s.example/?q=1", "https://a b"])
def test_normalize_url_rejects_what_the_cli_rejects(scratch, raw):
    harness = load_script("harness")
    with pytest.raises(ValueError):
        harness.normalize_url(raw)
    with pytest.raises(ValueError):
        settings_mod.normalize_url(raw)


def test_home_comes_from_the_argument_then_synapse_home_then_the_default(tmp_path):
    env = {"HOME": str(tmp_path / "u"), "SYNAPSE_HOME": str(tmp_path / "alt")}
    assert Settings.load(env=env, home=tmp_path / "arg").home == tmp_path / "arg"
    assert Settings.load(env=env).home == tmp_path / "alt"
    assert Settings.load(env={"HOME": str(tmp_path / "u")}).home == tmp_path / "u" / ".synapse"
    s = Settings.load(env={"HOME": str(tmp_path / "u"), "SYNAPSE_HOME": "~/elsewhere"})
    assert s.home == tmp_path / "u" / "elsewhere"
    assert s.config_file == s.home / "config.json"


def test_the_private_dir_is_under_the_user_home_not_synapse_home(tmp_path):
    """As config.PRIVATE_DIR: SYNAPSE_HOME moves config.json, not the private markers."""
    env = {"HOME": str(tmp_path / "u"), "SYNAPSE_HOME": str(tmp_path / "alt")}
    assert Settings.load(env=env).private_dir == tmp_path / "u" / ".synapse" / "private"
    env["SYNAPSE_PRIVATE_DIR"] = str(tmp_path / "p")
    assert Settings.load(env=env).private_dir == tmp_path / "p"


# ---------------------------------------------------------------------------
# Read-only, immutable, no token in repr
# ---------------------------------------------------------------------------


def test_load_never_writes(tmp_path):
    home = tmp_path / "h"
    Settings.load(env={"SYNAPSE_URL": "https://s.example"}, home=home)
    assert not home.exists()
    path = _write_config(home, {"url": "https://s.example", "token": _TOKEN})
    before = (path.read_bytes(), path.stat().st_mtime_ns, sorted(home.iterdir()))
    Settings.load(env={}, home=home)
    assert (path.read_bytes(), path.stat().st_mtime_ns, sorted(home.iterdir())) == before


def test_settings_are_frozen_and_hide_the_token(tmp_path):
    home = tmp_path / "h"
    _write_config(home, {"url": "https://s.example", "token": _TOKEN})
    s = Settings.load(env={"SYNAPSE_INGEST_TOKEN": _TOKEN}, home=home)
    assert s.token == _TOKEN
    assert _TOKEN not in repr(s) and _TOKEN[:6] not in repr(s)
    with pytest.raises(AttributeError):
        s.token = "x"  # type: ignore[misc]
    with pytest.raises(TypeError):
        s.config["url"] = "x"  # type: ignore[index]
    with pytest.raises(TypeError):
        s.environ["SYNAPSE_URL"] = "x"  # type: ignore[index]


def test_two_homes_load_side_by_side(tmp_path):
    """Nothing is cached at module level: one process can hold two settings."""
    a, b = tmp_path / "a", tmp_path / "b"
    _write_config(a, {"url": "https://a.example"})
    _write_config(b, {"url": "https://b.example"})
    assert Settings.load(env={}, home=a).base_url == "https://a.example"
    assert Settings.load(env={}, home=b).base_url == "https://b.example"


# ---------------------------------------------------------------------------
# flag() equals config.py's _cfg
# ---------------------------------------------------------------------------

_FLAG_NAMES = (
    ("SYNAPSE_MACHINE_ROLE", "personal"),
    ("SYNAPSE_SKILLS_SYNC", "0"),
    ("SYNAPSE_CONFIG_SYNC", "0"),
    ("SYNAPSE_CONFIG_PATHS", ""),
    ("SYNAPSE_BOARD", "1"),
    ("SYNAPSE_PREFS_BLOCK", "1"),
    ("SYNAPSE_SURFACE", ""),
)

_FLAG_MATRIX = {
    "nothing": ({}, {}, {}),
    "config-json": (
        {"role": "work", "skills_sync": True, "config_paths": ["~/a/*", "~/b"]},
        {},
        {},
    ),
    "config-json-empty-beats-legacy": (
        {"config_paths": [], "role": ""},
        {"SYNAPSE_CONFIG_PATHS": "~/stale", "SYNAPSE_MACHINE_ROLE": "work"},
        {},
    ),
    "env-wins": (
        {"role": "work"},
        {"SYNAPSE_BOARD": "0"},
        {"SYNAPSE_MACHINE_ROLE": "personal", "SYNAPSE_BOARD": "1", "SYNAPSE_SURFACE": "box"},
    ),
    "empty-env-falls-through": ({"config_sync": True}, {}, {"SYNAPSE_CONFIG_SYNC": ""}),
    "legacy-options-only": (
        {},
        {"SYNAPSE_PREFS_BLOCK": "0", "SYNAPSE_SURFACE": "legacy"},
        {"CLAUDE_PLUGIN_OPTION_SYNAPSE_SURFACE": "opt"},
    ),
}


@pytest.mark.parametrize("case", sorted(_FLAG_MATRIX))
def test_flag_returns_what_config_py_returns(scratch, monkeypatch, case):
    home_json, options, env = _FLAG_MATRIX[case]
    home = scratch
    _write_config(home / ".synapse", home_json)
    settings_json = home / ".claude" / "settings.json"
    settings_json.parent.mkdir(parents=True)
    settings_json.write_text(
        json.dumps({"pluginConfigs": {"synapse@synapse": {"options": options}}})
    )
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    cfg = load_script("config")
    prefix = "CLAUDE_PLUGIN_OPTION_"
    extras = [
        ("plugin_option", {k[len(prefix) :]: v for k, v in env.items() if k.startswith(prefix)}),
        ("plugin_config", options),
    ]
    core = Settings.load(extra_candidates=extras)
    for name, default in _FLAG_NAMES:
        assert core.flag(name, default) == cfg._cfg(name, default), name
