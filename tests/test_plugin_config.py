"""Plugin config layer (plugin/scripts/config.py): resolution order + privacy defaults.

Resolution order (per _cfg): explicit env var → CLAUDE_PLUGIN_OPTION_* → the /plugin install
answers persisted in settings.json (pluginConfigs."synapse@<marketplace>".options) → default.

Privacy defaults: config mirroring — the user's global CLAUDE.md + rules/*.md — is
OFF unless explicitly opted in (SYNAPSE_CONFIG_SYNC=1); skills sync is OFF unless
explicitly opted in (SYNAPSE_SKILLS_SYNC=1, issue #9 — a hook that writes into
~/.claude/skills at session start must never be a surprise).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

_CONFIG_PY = Path(__file__).resolve().parents[1] / "plugin" / "scripts" / "config.py"

_ENV_VARS = (
    "SYNAPSE_URL",
    "SYNAPSE_INGEST_URL",
    "SYNAPSE_INGEST_TOKEN",
    "SYNAPSE_CONFIG_SYNC",
    "SYNAPSE_SKILLS_SYNC",
    "SYNAPSE_CONFIG_PATHS",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_URL",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_URL",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_TOKEN",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_CONFIG_SYNC",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_SKILLS_SYNC",
)


def _fresh_config(
    monkeypatch, tmp_path, options: dict | None = None, env: dict | None = None
) -> ModuleType:
    """Import config.py fresh under a controlled environment: a scratch config dir
    (with an optional settings.json carrying install-prompt options), no real env
    vars, no credentials file, and a cwd with no project .claude."""
    cfg_dir = tmp_path / "claude"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    if options is not None:
        # A non-default marketplace name, so the synapse@<any-marketplace> match is exercised.
        (cfg_dir / "settings.json").write_text(
            json.dumps({"pluginConfigs": {"synapse@some-marketplace": {"options": options}}}),
            encoding="utf-8",
        )
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(cfg_dir))
    monkeypatch.setenv("SYNAPSE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.chdir(tmp_path)
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    sys.modules.pop("config", None)
    spec = importlib.util.spec_from_file_location("config", _CONFIG_PY)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules.pop("config", None)
    return mod


def test_settings_json_url_drives_all_endpoints(monkeypatch, tmp_path):
    cfg = _fresh_config(monkeypatch, tmp_path, options={"SYNAPSE_URL": "https://syn.example.net"})
    assert cfg.BASE_URL == "https://syn.example.net"
    assert cfg.INGEST_URL == "https://syn.example.net/ingest"
    assert cfg.RECALL_URL == "https://syn.example.net/recall"
    assert cfg.MCP_URL == "https://syn.example.net/mcp"


def test_env_var_outranks_settings_json(monkeypatch, tmp_path):
    cfg = _fresh_config(
        monkeypatch,
        tmp_path,
        options={"SYNAPSE_URL": "https://syn.example.net"},
        env={"SYNAPSE_URL": "http://envhost:1111"},
    )
    assert cfg.BASE_URL == "http://envhost:1111"
    assert cfg.INGEST_URL == "http://envhost:1111/ingest"


def test_unconfigured_defaults_to_localhost(monkeypatch, tmp_path):
    cfg = _fresh_config(monkeypatch, tmp_path)
    assert cfg.BASE_URL == "http://localhost:8765"
    assert cfg.INGEST_TOKEN == ""


def test_config_sync_defaults_off(monkeypatch, tmp_path):
    """Privacy default: the user's global CLAUDE.md + rules must NOT leave the box
    unless they opt in."""
    cfg = _fresh_config(monkeypatch, tmp_path)
    assert cfg.CONFIG_SYNC is False


def test_config_sync_opt_in_via_env(monkeypatch, tmp_path):
    cfg = _fresh_config(monkeypatch, tmp_path, env={"SYNAPSE_CONFIG_SYNC": "1"})
    assert cfg.CONFIG_SYNC is True


def test_config_sync_opt_in_via_settings(monkeypatch, tmp_path):
    cfg = _fresh_config(monkeypatch, tmp_path, options={"SYNAPSE_CONFIG_SYNC": "1"})
    assert cfg.CONFIG_SYNC is True


def test_skills_sync_defaults_off_opt_in(monkeypatch, tmp_path):
    assert _fresh_config(monkeypatch, tmp_path).SKILLS_SYNC is False
    on = _fresh_config(monkeypatch, tmp_path, env={"SYNAPSE_SKILLS_SYNC": "1"})
    assert on.SKILLS_SYNC is True
    off = _fresh_config(monkeypatch, tmp_path, env={"SYNAPSE_SKILLS_SYNC": "0"})
    assert off.SKILLS_SYNC is False


def test_skills_sync_opt_in_via_settings(monkeypatch, tmp_path):
    cfg = _fresh_config(monkeypatch, tmp_path, options={"SYNAPSE_SKILLS_SYNC": "1"})
    assert cfg.SKILLS_SYNC is True
