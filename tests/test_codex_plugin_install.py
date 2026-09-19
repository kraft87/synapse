"""Credential migration must work without inheriting a shell's environment."""

import importlib.util
import json
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

_PLUGIN = Path(__file__).resolve().parents[1] / "plugin-codex"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def installer(tmp_path):
    module = _load("synapse_installer_test", _PLUGIN / "install.py")
    module.CONFIG_PATH = tmp_path / "config.toml"
    return module


def test_migrate_existing_mcp_preserves_other_settings(installer):
    installer.CONFIG_PATH.write_text(
        'model = "test-model"\n\n[mcp_servers.synapse]\n'
        'url = "https://memory.example/mcp"\n'
        'bearer_token_env_var = "SYNAPSE_INGEST_TOKEN"\n'
        "tool_timeout_sec = 90\nenabled = true\n\n"
        '[mcp_servers.other]\ncommand = "other-tool"\n'
    )
    installer.install_mcp("https://memory.example", False)
    text = installer.CONFIG_PATH.read_text()
    config = tomllib.loads(text)
    server = config["mcp_servers"]["synapse"]
    assert "bearer_token_env_var" not in server
    assert "mcp_headers.py" in server["http_headers_helper"]
    assert server["url"] == "https://memory.example/mcp"
    assert server["tool_timeout_sec"] == 90
    assert config["mcp_servers"]["other"] == {"command": "other-tool"}
    assert config["model"] == "test-model"
    installer.install_mcp("https://memory.example/mcp", False)
    assert installer.CONFIG_PATH.read_text() == text


def test_install_new_and_dry_run(installer):
    installer.install_mcp("https://memory.example", True)
    assert not installer.CONFIG_PATH.exists()
    installer.install_mcp("https://memory.example", False)
    assert tomllib.loads(installer.CONFIG_PATH.read_text())["mcp_servers"]["synapse"]["url"] == (
        "https://memory.example/mcp"
    )


def test_upgrade_hook_matcher_without_touching_unrelated_hooks(installer):
    text = "\n".join(block for _, block in installer._HOOK_BLOCKS)
    text = text.replace("fetch_session|remember)", "fetch_session)")
    unrelated = (
        "\n[[hooks.PreToolUse]]\n"
        'matcher = "mcp__.*__(recall|recall_full_turns|recall_feedback|fetch_session)$"\n'
        '[[hooks.PreToolUse.hooks]]\ncommand = "other-hook"\ntype = "command"\n'
    )
    trust = '\n[hooks.state."existing-hook"]\ntrusted_hash = "existing-hash"\n'
    installer.CONFIG_PATH.write_text(text + unrelated + trust)
    installer.install_hook(True)
    assert installer.CONFIG_PATH.read_text() == text + unrelated + trust
    installer.install_hook(False)
    result = installer.CONFIG_PATH.read_text()
    assert "fetch_session|remember)" in result
    assert unrelated in result
    assert trust in result
    tomllib.loads(result)
    installer.install_hook(False)
    assert installer.CONFIG_PATH.read_text() == result


@pytest.mark.parametrize(
    "custom", ['command = "custom"', 'http_headers = {Authorization = "custom"}']
)
def test_custom_auth_or_transport_not_overwritten(installer, custom):
    original = "[mcp_servers.synapse]\n" + custom + "\n"
    installer.CONFIG_PATH.write_text(original)
    with pytest.raises(ValueError):
        installer.install_mcp("https://memory.example", False)
    assert installer.CONFIG_PATH.read_text() == original


@pytest.mark.parametrize(
    ("token", "url", "expected"),
    [
        ("test-device-token", "https://memory.example/mcp", 0),
        ("", "https://memory.example/mcp", 1),
        ("test-device-token", "https://different.example/mcp", 1),
    ],
)
def test_header_helper_fails_closed(monkeypatch, capsys, token, url, expected):
    common = ModuleType("common")
    common.BASE_URL = "https://memory.example"
    common.TOKEN = token
    monkeypatch.setitem(sys.modules, "common", common)
    helper = _load("synapse_headers_test", _PLUGIN / "scripts/mcp_headers.py")
    monkeypatch.setattr(sys, "argv", ["mcp_headers.py", "--url", url])
    assert helper.main() == expected
    captured = capsys.readouterr()
    if expected == 0:
        assert json.loads(captured.out) == {"Authorization": "Bearer test-device-token"}
    else:
        assert not captured.out
    assert "test-device-token" not in captured.err


def test_saved_credentials_work_without_env_token(monkeypatch, tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "pluginConfigs": {
                    "synapse@test": {
                        "options": {
                            "SYNAPSE_URL": "https://memory.example",
                            "SYNAPSE_INGEST_TOKEN": "saved-device-token",
                        }
                    }
                }
            }
        )
    )
    monkeypatch.delenv("SYNAPSE_INGEST_TOKEN", raising=False)
    monkeypatch.delenv("SYNAPSE_URL", raising=False)
    monkeypatch.delenv("SYNAPSE_INGEST_URL", raising=False)
    monkeypatch.setattr(
        "os.path.expanduser",
        lambda value: str(settings) if value.endswith("settings.json") else value,
    )
    common = _load("synapse_common_test", _PLUGIN / "scripts/common.py")
    assert common.TOKEN == "saved-device-token"
    assert common.BASE_URL == "https://memory.example"
