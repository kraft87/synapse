"""Codex plugin hooks must remain executable from PowerShell on Windows."""

import json
from pathlib import Path

_HOOKS_PATH = Path(__file__).resolve().parents[1] / "plugin" / "hooks" / "hooks.json"


def _command_hooks():
    config = json.loads(_HOOKS_PATH.read_text(encoding="utf-8-sig"))
    for event, groups in config["hooks"].items():
        for group in groups:
            for hook in group["hooks"]:
                if hook["type"] == "command":
                    yield event, hook


def test_every_command_hook_has_a_powershell_override():
    hooks = list(_command_hooks())
    assert len(hooks) == 13
    for event, hook in hooks:
        command = hook.get("commandWindows", "")
        assert command, f"{event} hook is missing commandWindows"
        assert "$env:CLAUDE_PLUGIN_ROOT/scripts/" in command
        assert 'python3 "' in command


def test_session_end_timeout_respects_codex_maximum():
    session_end = [hook for event, hook in _command_hooks() if event == "SessionEnd"]
    assert len(session_end) == 1
    assert session_end[0]["timeout"] == 3


def test_codex_hook_outputs_do_not_use_unsupported_suppression():
    root = Path(__file__).resolve().parents[1]
    for path in (root / "plugin-codex" / "hooks").glob("*.py"):
        assert "suppressOutput" not in path.read_text(encoding="utf-8")
