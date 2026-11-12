"""The core's credential rules (plugin/synapse_client/credentials.py) and their wiring.

credentials.py is a pure port of plugin/scripts/config.py's token resolution: an
unenrolled machine keeps plain precedence; an enrolled one sends its device token, and
only toward an origin that already has it; overridden sources are named, never their
values. The hooks still use config.py, so the EQUIVALENCE test below builds the same
scratch home for both and asserts the same ``(token, note)``: that is what keeps the two
implementations from drifting until the hooks move onto the core.

Stdlib only, no network: config.py is loaded by path, as the other client tests do.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parents[1]
_PLUGIN = str(_REPO / "plugin")
if _PLUGIN not in sys.path:
    sys.path.append(_PLUGIN)

from synapse_client import credentials  # noqa: E402
from synapse_client.settings import Settings  # noqa: E402

_SCRIPTS = _REPO / "plugin" / "scripts"
_MODULES = ("config", "synapse_home", "enroll")

# Distinctive values, so "no token material in the output" can check prefixes too.
_DEVICE = "tkDEV3w8Pq1ZmK7r"
_STALE = "tkENV5n2Xc9Lb4Hd"
_OTHER = "tkOPT6t0Gy3Vf8Js"
_URL = "https://synapse.example.net"
_FOREIGN = "https://other.example.org"
_SURFACE_ID = "dev-abc123"


def _sha(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _record(token: str = _DEVICE, *, server: str | None = _URL, hashed: bool = True) -> dict:
    rec: dict = {"surface_id": _SURFACE_ID, "trust": "full", "allowed_projects": []}
    if hashed:
        rec["token_sha256"] = _sha(token)
    if server:
        rec["server"] = server
    return rec


def _no_token_material(text: str) -> None:
    for tok in (_DEVICE, _STALE, _OTHER):
        assert tok not in text
        assert tok[:6] not in text and tok[-6:] not in text


# ---------------------------------------------------------------------------
# credentials.resolve: the rules on explicit inputs
# ---------------------------------------------------------------------------


def _cands(env: str = "", config_file: str = "", **legacy: str) -> list[tuple[str, str]]:
    return [
        ("env", env),
        ("config_file", config_file),
        ("plugin_option", legacy.get("plugin_option", "")),
        ("plugin_config", legacy.get("plugin_config", "")),
    ]


def test_token_sha256_is_the_hex_sha256_of_the_token():
    assert credentials.token_sha256(_DEVICE) == _sha(_DEVICE)


def test_unenrolled_keeps_plain_precedence_then_the_legacy_credential():
    record: dict = {}
    assert credentials.resolve(_cands(_STALE, _OTHER), record, [_URL]) == (_STALE, {})
    assert credentials.resolve(_cands("", _OTHER), record, [_URL]) == (_OTHER, {})
    assert credentials.resolve(_cands(), record, [_URL], legacy_token="old") == ("old", {})


def test_enrolled_with_a_stale_env_token_sends_the_device_token_and_names_the_source():
    token, note = credentials.resolve(_cands(_STALE, _DEVICE), _record(), [_URL, f"{_URL}/ingest"])
    assert token == _DEVICE
    assert note == {"surface_id": _SURFACE_ID, "sources": ["env"]}


def test_the_same_token_in_env_is_not_a_conflict():
    assert credentials.resolve(_cands(_DEVICE, _DEVICE), _record(), [_URL]) == (_DEVICE, {})


def test_a_hash_matching_nothing_changes_nothing():
    """The configured token was replaced by hand after enrolling: the record no longer
    describes it, so no value is singled out as the device token."""
    assert credentials.resolve(_cands(_STALE, _OTHER), _record(), [_URL]) == (_STALE, {})


@pytest.mark.parametrize("urls", [[_FOREIGN], [_URL, f"{_FOREIGN}/ingest"]])
def test_the_device_token_never_goes_to_another_origin(urls):
    """An env URL pointing elsewhere means the env token is that server's."""
    assert credentials.resolve(_cands(_STALE, _DEVICE), _record(), urls) == (_STALE, {})


def test_with_nothing_else_to_send_the_withheld_note_names_the_url_and_the_server():
    token, note = credentials.resolve(_cands("", _DEVICE), _record(), [f"{_FOREIGN}/x"])
    assert token == ""
    assert note == {
        "surface_id": _SURFACE_ID,
        "sources": [],
        "withheld": _FOREIGN,
        "server": _URL,
    }


def test_the_same_server_spelled_differently_still_gets_the_device_token():
    token, _ = credentials.resolve(
        _cands(_STALE, _DEVICE), _record(), ["HTTPS://Synapse.Example.net/mcp"]
    )
    assert token == _DEVICE


def test_a_record_without_a_server_goes_by_the_saved_url():
    rec = _record(server=None)
    assert credentials.resolve(_cands(_STALE, _DEVICE), rec, [_URL], saved_url=_URL)[0] == _DEVICE
    assert credentials.resolve(_cands(_STALE, _DEVICE), rec, [_FOREIGN], saved_url=_URL)[0] == (
        _STALE
    )


def test_a_legacy_record_without_a_hash_trusts_the_plugin_config():
    rec = _record(hashed=False, server=None)
    cands = _cands(_STALE, plugin_config=_DEVICE)
    token, note = credentials.resolve(cands, rec, [_URL], plugin_origins={credentials.origin(_URL)})
    assert (token, note) == (_DEVICE, {"surface_id": _SURFACE_ID, "sources": ["env"]})


def test_plugin_origins_widen_only_for_a_device_token_the_plugin_config_holds():
    rec = _record(server="http://localhost:8765")
    held = _cands(plugin_config=_DEVICE)
    plugin = {credentials.origin(_URL)}
    assert credentials.resolve(held, rec, [_URL], plugin_origins=plugin)[0] == _DEVICE
    not_held = _cands(config_file=_DEVICE)
    assert credentials.resolve(not_held, rec, [_URL], plugin_origins=plugin)[0] == ""


def test_with_no_url_at_all_the_origin_rule_has_nothing_to_check():
    """config.py would compare against its http://localhost:8765 default; the core has no
    default URL, so there is nowhere a token could go and nothing to withhold."""
    assert credentials.resolve(_cands(_STALE, _DEVICE), _record(), ["", ""])[0] == _DEVICE


def test_enrolled_origins_are_empty_for_an_unenrolled_machine():
    assert credentials.enrolled_origins({}, _cands(config_file=_DEVICE)) == set()
    assert credentials.enrolled_origins(_record(), _cands(config_file=_DEVICE)) == {
        credentials.origin(_URL)
    }


# ---------------------------------------------------------------------------
# The notices
# ---------------------------------------------------------------------------


def test_override_notice_is_empty_for_an_empty_note():
    assert credentials.override_notice({}) == ""


def test_override_notice_names_each_source_without_token_material():
    env = credentials.override_notice(
        {"surface_id": _SURFACE_ID, "sources": ["env"]}, env_location="~/.hermes/.env"
    )
    assert env == (
        "[Synapse] SYNAPSE_INGEST_TOKEN from your environment is ignored: this machine is "
        f"enrolled as {_SURFACE_ID}. Remove it from ~/.hermes/.env."
    )
    cfg = credentials.override_notice({"surface_id": _SURFACE_ID, "sources": ["config_file"]})
    assert cfg.startswith("[Synapse] The token in ~/.synapse/config.json is ignored")
    assert "synapse login --reenroll" in cfg and "synapse config token --unset" in cfg
    other = credentials.override_notice(
        {"surface_id": _SURFACE_ID, "sources": ["plugin_option"]}, hint="Restart the harness."
    )
    assert other.endswith("uses its device token. Restart the harness.")
    assert "/reload-plugins" not in env + cfg + other and "Claude" not in env + cfg + other
    for text in (env, cfg, other):
        _no_token_material(text)
        assert "\n" not in text


def test_a_withheld_note_renders_the_withheld_notice():
    note = {"surface_id": _SURFACE_ID, "sources": [], "withheld": _FOREIGN, "server": _URL}
    assert credentials.override_notice(note) == credentials.withheld_notice(_FOREIGN, _URL)
    text = credentials.withheld_notice(_FOREIGN, "")
    assert "where this machine's device token was enrolled" in text
    assert f"synapse config url {_FOREIGN} --same-server" in text


# ---------------------------------------------------------------------------
# Equivalence with plugin/scripts/config.py
# ---------------------------------------------------------------------------


@pytest.fixture()
def scratch(monkeypatch, tmp_path):
    """Everything both implementations read lives under tmp_path: HOME, SYNAPSE_HOME
    (config.json), CLAUDE_CONFIG_DIR (the Claude plugin's settings.json), SYNAPSE_DATA_DIR
    (the legacy device.json and credentials file, never written here) and a cwd with no
    project .claude. No setting arrives through the real environment."""
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SYNAPSE_HOME", str(home / ".synapse"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home / ".claude"))
    monkeypatch.setenv("SYNAPSE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.chdir(tmp_path)
    keep = ("SYNAPSE_HOME", "SYNAPSE_DATA_DIR")
    for var in list(os.environ):
        if var.startswith(("SYNAPSE_", "CLAUDE_PLUGIN_OPTION_")) and var not in keep:
            monkeypatch.delenv(var)
    yield home
    for mod in _MODULES:
        sys.modules.pop(mod, None)


def _load_config() -> ModuleType:
    """config.py as a fresh hook process imports it: it resolves the token at import."""
    for mod in _MODULES:
        sys.modules.pop(mod, None)
    spec = importlib.util.spec_from_file_location("config", _SCRIPTS / "config.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


_ENROLLED = {"url": _URL, "token": _DEVICE, "device": _record()}

# (config.json, environment). Every row names a URL: config.py falls back to
# http://localhost:8765 when none is set and the core does not (tested above).
_MATRIX = {
    "unenrolled-env-wins": ({"url": _URL, "token": _OTHER}, {"SYNAPSE_INGEST_TOKEN": _STALE}),
    "unenrolled-config-json": ({"url": _URL, "token": _OTHER}, {}),
    "unenrolled-nothing": ({"url": _URL}, {}),
    "unenrolled-env-url-only": ({}, {"SYNAPSE_URL": f"{_URL}/mcp/"}),
    "enrolled-with-hash": (_ENROLLED, {}),
    "enrolled-env-token-differs": (_ENROLLED, {"SYNAPSE_INGEST_TOKEN": _STALE}),
    "enrolled-env-token-same": (_ENROLLED, {"SYNAPSE_INGEST_TOKEN": _DEVICE}),
    "enrolled-device-token-in-env": (
        {"url": _URL, "token": _OTHER, "device": _record()},
        {"SYNAPSE_INGEST_TOKEN": _DEVICE},
    ),
    "enrolled-hash-matches-nothing": (
        {"url": _URL, "token": _OTHER, "device": _record()},
        {"SYNAPSE_INGEST_TOKEN": _STALE},
    ),
    "env-url-at-another-origin": (
        _ENROLLED,
        {"SYNAPSE_INGEST_TOKEN": _STALE, "SYNAPSE_URL": _FOREIGN},
    ),
    "env-url-at-another-origin-nothing-else": (_ENROLLED, {"SYNAPSE_URL": _FOREIGN}),
    "env-ingest-url-at-another-origin": (
        _ENROLLED,
        {"SYNAPSE_INGEST_TOKEN": _STALE, "SYNAPSE_INGEST_URL": f"{_FOREIGN}/ingest"},
    ),
    "env-ingest-url-at-another-origin-nothing-else": (
        _ENROLLED,
        {"SYNAPSE_INGEST_URL": f"{_FOREIGN}/ingest"},
    ),
    "saved-url-moved": ({**_ENROLLED, "url": _FOREIGN}, {}),
    "saved-url-moved-env-token": ({**_ENROLLED, "url": _FOREIGN}, {"SYNAPSE_INGEST_TOKEN": _STALE}),
    "same-server-spelled-differently": (
        _ENROLLED,
        {"SYNAPSE_INGEST_TOKEN": _STALE, "SYNAPSE_URL": "HTTPS://Synapse.Example.net/mcp"},
    ),
    "record-without-server": (
        {"url": _URL, "token": _DEVICE, "device": _record(server=None)},
        {"SYNAPSE_INGEST_TOKEN": _STALE},
    ),
    "record-without-server-url-elsewhere": (
        {"url": _URL, "token": _DEVICE, "device": _record(server=None)},
        {"SYNAPSE_INGEST_TOKEN": _STALE, "SYNAPSE_URL": _FOREIGN},
    ),
    "record-without-server-or-saved-url": (
        {"token": _DEVICE, "device": _record(server=None)},
        {"SYNAPSE_URL": _URL},
    ),
    "record-without-surface": (
        {"url": _URL, "token": _DEVICE, "device": {"token_sha256": _sha(_DEVICE)}},
        {"SYNAPSE_INGEST_TOKEN": _STALE},
    ),
    "record-not-an-object": (
        {"url": _URL, "token": _DEVICE, "device": "nonsense"},
        {"SYNAPSE_INGEST_TOKEN": _STALE},
    ),
}


@pytest.mark.parametrize("case", sorted(_MATRIX))
def test_the_core_resolves_exactly_as_config_py(scratch, monkeypatch, case):
    home_json, env = _MATRIX[case]
    _write_json(scratch / ".synapse" / "config.json", home_json)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    cfg = _load_config()
    core = Settings.load()

    assert (core.token, dict(core.token_note)) == (cfg.INGEST_TOKEN, cfg.TOKEN_OVERRIDE)
    assert (core.base_url, core.ingest_url) == (cfg.BASE_URL, cfg.INGEST_URL)
    assert core.device_origins == cfg.device_origins()
    assert core.enrolled == bool(cfg.read_device_state().get("surface_id"))
    # No settings.json "env" block exists here, so config.py names the shell profile,
    # which is the core's default too.
    assert credentials.override_notice(core.token_note) == cfg.token_override_notice()
    _no_token_material(repr(core))


def _plugin_extras() -> list[tuple[str, dict]]:
    """The Claude plugin's two legacy stores, as its adapter will pass them: the
    CLAUDE_PLUGIN_OPTION_* environment and the settings.json options block."""
    prefix = "CLAUDE_PLUGIN_OPTION_"
    loaded = {k[len(prefix) :]: v for k, v in os.environ.items() if k.startswith(prefix)}
    settings = Path(os.environ["CLAUDE_CONFIG_DIR"]) / "settings.json"
    data = json.loads(settings.read_text()) if settings.exists() else {}
    saved = ((data.get("pluginConfigs") or {}).get("synapse@synapse") or {}).get("options") or {}
    return [("plugin_option", loaded), ("plugin_config", saved)]


# (config.json, settings.json options, environment): the legacy stores a Claude adapter
# will hand over as extra_candidates.
_LEGACY_MATRIX = {
    "plugin-config-device-token-over-env": (
        {"device": _record()},
        {"SYNAPSE_URL": _URL, "SYNAPSE_INGEST_TOKEN": _DEVICE},
        {"SYNAPSE_INGEST_TOKEN": _STALE},
    ),
    "legacy-record-without-hash": (
        {"device": _record(hashed=False, server=None)},
        {"SYNAPSE_URL": _URL, "SYNAPSE_INGEST_TOKEN": _DEVICE},
        {"SYNAPSE_INGEST_TOKEN": _STALE},
    ),
    "stale-plugin-option-overridden": (
        {"device": _record()},
        {"SYNAPSE_URL": _URL, "SYNAPSE_INGEST_TOKEN": _DEVICE},
        {"CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_TOKEN": _OTHER},
    ),
    "plugin-url-holding-the-device-token": (
        {"device": _record(server="http://localhost:8765")},
        {"SYNAPSE_URL": _URL, "SYNAPSE_INGEST_TOKEN": _DEVICE},
        {},
    ),
    "plugin-url-not-holding-the-device-token": (
        {"token": _DEVICE, "device": _record(server="http://localhost:8765")},
        {"SYNAPSE_URL": _URL},
        {},
    ),
    "config-json-outranks-plugin-options": (
        {"url": _URL, "token": _OTHER},
        {"SYNAPSE_URL": _FOREIGN, "SYNAPSE_INGEST_TOKEN": _STALE},
        {"CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_TOKEN": _DEVICE},
    ),
}


@pytest.mark.parametrize("case", sorted(_LEGACY_MATRIX))
def test_extra_candidates_reproduce_the_claude_plugin_sources(scratch, monkeypatch, case):
    home_json, options, env = _LEGACY_MATRIX[case]
    _write_json(scratch / ".synapse" / "config.json", home_json)
    _write_json(
        scratch / ".claude" / "settings.json",
        {"pluginConfigs": {"synapse@synapse": {"options": options}}},
    )
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    cfg = _load_config()
    core = Settings.load(extra_candidates=_plugin_extras())

    assert (core.token, dict(core.token_note)) == (cfg.INGEST_TOKEN, cfg.TOKEN_OVERRIDE)
    assert core.base_url == cfg.BASE_URL
    assert core.device_origins == cfg.device_origins()
