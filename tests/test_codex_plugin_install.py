"""Credential migration must work without inheriting a shell's environment."""

import importlib.util
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_PLUGIN = Path(__file__).resolve().parents[1] / "plugin-codex"
_SHARED_HELPER = Path(__file__).resolve().parents[1] / "plugin" / "scripts" / "mcp_headers.py"


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
    text = "\n".join(block for _, block in installer.hook_blocks())
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
    ("token", "url", "server", "expected"),
    [
        ("test-device-token", "https://memory.example/mcp", None, 0),
        ("test-device-token", "https://memory.example/mcp", "https://memory.example", 0),
        ("", "https://memory.example/mcp", None, 1),
        ("test-device-token", "https://different.example/mcp", None, 1),
        # Saved URL moved, but the credential was minted by another server.
        ("test-device-token", "https://memory.example/mcp", "https://old.example", 1),
    ],
)
def test_header_helper_fails_closed(monkeypatch, capsys, tmp_path, token, url, server, expected):
    """The shared helper (plugin/scripts/mcp_headers.py) both harnesses run."""
    config = ModuleType("config")
    config.BASE_URL = "https://memory.example"
    config.INGEST_TOKEN = token
    config._FILE_OPTIONS = {}
    config.synapse_home = SimpleNamespace(home=lambda: tmp_path)
    # config decides whether the device token may go there (synapse_client.credentials).
    withheld = bool(server) and server != "https://memory.example"
    config.TOKEN_OVERRIDE = (
        {"withheld": "https://memory.example", "server": server} if withheld else {}
    )
    config.withheld_notice = lambda url, srv: (
        f"{url} is not the server this machine enrolled with ({srv})"
    )
    monkeypatch.setitem(sys.modules, "config", config)
    helper = _load("synapse_headers_test", _SHARED_HELPER)
    assert helper.main(["--url", url]) == expected
    captured = capsys.readouterr()
    if expected == 0:
        assert json.loads(captured.out) == {"Authorization": "Bearer test-device-token"}
    else:
        assert not captured.out
        # Codex discards stderr, so the reason is logged too; never the token.
        assert (tmp_path / "logs" / "mcp_headers.log").read_text().strip()
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


def test_header_helper_reads_only_the_users_own_settings(tmp_path):
    """Codex runs the helper in the session's directory: a repository's
    .claude/settings.json must not choose the server or the token."""
    user = tmp_path / "user-claude"
    user.mkdir()
    options = {"SYNAPSE_URL": "https://mine.example", "SYNAPSE_INGEST_TOKEN": "my-token"}
    (user / "settings.json").write_text(
        json.dumps({"pluginConfigs": {"synapse@synapse": {"options": options}}})
    )
    repo = tmp_path / "repo"
    (repo / ".claude").mkdir(parents=True)
    planted = {"SYNAPSE_URL": "https://mine.example", "SYNAPSE_INGEST_TOKEN": "PLANTED"}
    (repo / ".claude" / "settings.json").write_text(
        json.dumps({"pluginConfigs": {"synapse@synapse": {"options": planted}}})
    )
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path / "home"),
        "CLAUDE_CONFIG_DIR": str(user),
        "SYNAPSE_HOME": str(tmp_path / "synapse-home"),
        "SYNAPSE_DATA_DIR": str(tmp_path / "data"),
    }
    done = subprocess.run(
        [sys.executable, str(_SHARED_HELPER), "--url", "https://mine.example/mcp"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == {"Authorization": "Bearer my-token"}


def test_main_keeps_the_existing_url_of_a_file_old_windows_lines_broke(installer, monkeypatch):
    """0.19 wrote hook commands with raw backslashes on Windows: the whole file doesn't
    parse, but its [mcp_servers.synapse] URL is still the default to keep."""
    installer.CONFIG_PATH.write_text(
        '[mcp_servers.synapse]\nurl = "https://mem.example/mcp"\n\n'
        "[[hooks.Stop]]\n[[hooks.Stop.hooks]]\n"
        'type = "command"\ncommand = "python3 C:\\Users\\x\\plugin-codex/hooks/synapse_stop_hook.py"\n'
    )
    seen = []
    monkeypatch.setattr(installer, "install", lambda url, dry_run=False: seen.append(url) or True)
    monkeypatch.setattr(sys, "argv", ["install.py"])
    assert installer.main() == 0
    assert seen == ["https://mem.example/mcp"]


# The six Synapse hooks of a managed install, and the hash Codex 0.162.1 gives each
# (``codex app-server``, hooks/list ``currentHash``): what it records as ``trusted_hash``
# when the user trusts them in /hooks.
_RUN = '"python3 \\"/home/u/.synapse/bin/synapse_run.py\\" plugin-codex/'
_TRUSTED_BLOCKS = f"""\
[[hooks.Stop]]
[[hooks.Stop.hooks]]
type = "command"
command = {_RUN}hooks/synapse_stop_hook.py"
timeout = 30

[[hooks.SessionStart]]
[[hooks.SessionStart.hooks]]
type = "command"
command = {_RUN}hooks/session_start.py"
timeout = 20

[[hooks.UserPromptSubmit]]
[[hooks.UserPromptSubmit.hooks]]
type = "command"
command = {_RUN}hooks/user_prompt_submit.py"
timeout = 5

[[hooks.PreToolUse]]
matcher = "mcp__.*__(recall|recall_full_turns|recall_feedback|fetch_session|remember)$"
[[hooks.PreToolUse.hooks]]
type = "command"
command = {_RUN}hooks/pre_tool_use.py"
timeout = 5

[[hooks.PostToolUse]]
matcher = "mcp__.*__(recall|recall_full_turns)$"
[[hooks.PostToolUse.hooks]]
type = "command"
command = {_RUN}hooks/post_tool_use.py"
timeout = 5

[[hooks.SessionEnd]]
[[hooks.SessionEnd.hooks]]
type = "command"
command = {_RUN}scripts/private_mode.py --session-end"
timeout = 3
"""
_TRUSTED = {
    "stop": "a0744f0bec5f820430b01e1ce6710d0edd00fdc20cf34576c05ff11ad861cdde",
    "session_start": "67455bc3d0be11d89fa03b397a6e62b9ca12ee6b30c485a4762de48e61e2b44c",
    "user_prompt_submit": "092185c1524e2d7e49844fb8cf29db5da2c2e80e61c5128251ae87abe7cdaa24",
    "pre_tool_use": "6da91340d6ca8d97de7036c44139a9d5d0df310aeb5eb8aa9fcb6db0030a5f8b",
    "post_tool_use": "703ff0ad9d81d04d808e72e304eeb625b6e64e0bf40e2103d27ee6ee5bb0e938",
    "session_end": "7981894e0cb1ca4afcb5527d4ae9d78bab2e9c42a9cc528cd5951380d84a18c3",
}


def _trust_records(path, events=_TRUSTED, group=0) -> str:
    out = ["\n[hooks.state]\n"]
    for event, digest in events.items():
        key = json.dumps(f"{path}:{event}:{group}:0")
        out.append(f'[hooks.state.{key}]\ntrusted_hash = "sha256:{digest}"\n')
    return "\n".join(out)


def test_trust_is_read_as_codex_records_it(installer):
    """Codex keys a hook's trust by file, event and place, and records the hash of what
    the user trusted. The hooks Codex trusted read as trusted; any that changed, moved
    or has no record does not."""
    path = installer.CONFIG_PATH
    path.write_text(_TRUSTED_BLOCKS)
    everything = sorted(marker for marker, *_ in installer._HOOKS)
    assert sorted(installer.untrusted_hooks()) == everything
    path.write_text(_TRUSTED_BLOCKS + _trust_records(path))
    assert installer.untrusted_hooks() == []
    # Recorded for another file (a project's .codex/config.toml): not this one's.
    path.write_text(_TRUSTED_BLOCKS + _trust_records(path.parent / "other" / "config.toml"))
    assert sorted(installer.untrusted_hooks()) == everything
    # A command changed since it was trusted.
    changed = _TRUSTED_BLOCKS.replace("session_start.py", "session_start.py --again")
    path.write_text(changed + _trust_records(path))
    assert installer.untrusted_hooks() == ["session_start.py"]
    # One record missing.
    missing = {k: v for k, v in _TRUSTED.items() if k != "stop"}
    path.write_text(_TRUSTED_BLOCKS + _trust_records(path, missing))
    assert installer.untrusted_hooks() == ["synapse_stop_hook.py"]
    # A hook of the user's own put before ours in Stop: ours moved, and Codex asks again.
    mine = '[[hooks.Stop]]\n[[hooks.Stop.hooks]]\ntype = "command"\ncommand = "mine"\n\n'
    path.write_text(mine + _TRUSTED_BLOCKS + _trust_records(path))
    assert installer.untrusted_hooks() == ["synapse_stop_hook.py"]
    path.write_text("not = [toml")
    assert installer.untrusted_hooks() is None


def test_install_leaves_trust_records_alone_and_needs_none_again(installer):
    path = installer.CONFIG_PATH
    path.write_text(_TRUSTED_BLOCKS + _trust_records(path))
    launcher = installer.Launcher(
        "python3", root=Path("/unused"), runner=Path("/home/u/.synapse/bin/synapse_run.py")
    )
    installer.install("https://memory.example", launcher=launcher)
    assert "[hooks.state." in path.read_text()
    assert installer.untrusted_hooks() == []
