"""The PreToolUse hooks that inject `self_session` and `surface` into synapse tools.

The model cannot know its own session id, and it must not CHOOSE its own surface — a
hallucinated `"surface": "<some-trusted-host>"` would widen what a session can read.
Injecting below the model is what makes host trust meaningful, so these tests pin which
tools get which field, that an already-set value is never overwritten, and that every
failure mode is silent (a broken hook must never block a tool call).

Both the Claude Code hook and its Codex port are covered here; they are separate files
that must not drift.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_CC_HOOK = _ROOT / "plugin" / "scripts" / "self_session_inject.py"
_CODEX_HOOK = _ROOT / "plugin-codex" / "hooks" / "pre_tool_use.py"

_SURFACE = "test-inject-host"
_MCP = "mcp__plugin_synapse_synapse__"


def _run(hook: Path, payload: dict) -> dict | None:
    """Run the hook as the harness does — a subprocess over stdin/stdout. Returns the
    parsed updatedInput block, or None when the hook chose to emit nothing."""
    proc = subprocess.run(
        [sys.executable, str(hook)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env={"SYNAPSE_SURFACE": _SURFACE, "PATH": "/usr/bin:/bin", "HOME": "/nonexistent"},
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr  # fail-open: never a non-zero exit
    out = proc.stdout.strip()
    return json.loads(out)["hookSpecificOutput"]["updatedInput"] if out else None


_HOOKS = pytest.mark.parametrize("hook", [_CC_HOOK, _CODEX_HOOK], ids=["claude-code", "codex"])


@_HOOKS
@pytest.mark.parametrize(
    "tool", ["recall", "recall_full_turns", "fetch", "fetch_session", "remember"]
)
def test_surface_is_injected_into_every_tool_that_accepts_it(hook, tool):
    ti = _run(hook, {"tool_name": _MCP + tool, "session_id": "s-1", "tool_input": {"query": "q"}})
    assert ti["surface"] == _SURFACE


@_HOOKS
def test_recall_feedback_gets_no_surface(hook):
    """It takes no such param — passing an unknown argument would fail the call."""
    ti = _run(
        hook,
        {"tool_name": _MCP + "recall_feedback", "session_id": "s-1", "tool_input": {"query": "q"}},
    )
    assert "surface" not in ti
    assert ti["session_id"] == "s-1"  # its own field, unchanged by this work


@_HOOKS
def test_self_session_still_lands_on_the_recall_family(hook):
    ti = _run(hook, {"tool_name": _MCP + "recall", "session_id": "s-1", "tool_input": {}})
    assert ti["self_session"] == "s-1" and ti["surface"] == _SURFACE


@_HOOKS
def test_fetch_gets_surface_but_no_session_field(hook):
    """fetch has no session dimension at all — injecting one would be an unknown arg."""
    ti = _run(hook, {"tool_name": _MCP + "fetch", "session_id": "s-1", "tool_input": {"ids": []}})
    assert ti["surface"] == _SURFACE
    assert "self_session" not in ti and "session_id" not in ti


@_HOOKS
def test_an_already_set_value_is_never_overwritten(hook):
    """A retry or a nested call must not have its inputs rewritten underneath it."""
    ti = _run(
        hook,
        {
            "tool_name": _MCP + "recall",
            "session_id": "s-1",
            "tool_input": {"surface": "explicit-host", "self_session": "s-orig"},
        },
    )
    assert ti is None or (ti["surface"] == "explicit-host" and ti["self_session"] == "s-orig")


@_HOOKS
@pytest.mark.parametrize(
    "payload",
    [
        {"tool_name": "Bash", "session_id": "s-1", "tool_input": {"command": "ls"}},
        {"tool_name": _MCP + "recall", "session_id": "s-1", "tool_input": "not-a-dict"},
        {},
    ],
    ids=["unrelated-tool", "bad-tool-input", "empty-payload"],
)
def test_hooks_emit_nothing_rather_than_touching_a_call_they_do_not_own(hook, payload):
    assert _run(hook, payload) is None


@_HOOKS
def test_malformed_stdin_is_silent(hook):
    proc = subprocess.run(
        [sys.executable, str(hook)],
        input="{not json",
        capture_output=True,
        text=True,
        env={"SYNAPSE_SURFACE": _SURFACE, "PATH": "/usr/bin:/bin", "HOME": "/nonexistent"},
        timeout=30,
    )
    assert proc.returncode == 0 and proc.stdout.strip() == ""


def test_codex_hook_declares_the_permission_decision():
    """Codex requires permissionDecision alongside updatedInput; without it the rewrite
    is dropped and the surface never arrives."""
    proc = subprocess.run(
        [sys.executable, str(_CODEX_HOOK)],
        input=json.dumps(
            {"tool_name": _MCP + "recall", "session_id": "s-1", "tool_input": {"query": "q"}}
        ),
        capture_output=True,
        text=True,
        env={"SYNAPSE_SURFACE": _SURFACE, "PATH": "/usr/bin:/bin", "HOME": "/nonexistent"},
        timeout=30,
    )
    block = json.loads(proc.stdout)["hookSpecificOutput"]
    assert block["permissionDecision"] == "allow"


def test_hook_matchers_cover_exactly_the_injected_tools():
    """The config matcher and the in-script guard must agree with the tool list, or a
    tool silently stops receiving its surface (fail-open on the client = restricted on
    the server = a board that mysteriously goes empty)."""
    hooks_json = json.loads((_ROOT / "plugin" / "hooks" / "hooks.json").read_text())
    matcher = hooks_json["hooks"]["PreToolUse"][0]["matcher"]
    codex_install = (_ROOT / "plugin-codex" / "install.py").read_text()

    expected = "(recall|recall_full_turns|recall_feedback|fetch|fetch_session|remember)$"
    assert matcher.endswith(expected)
    assert expected in codex_install

    spec = importlib.util.spec_from_file_location("_cc_inject", _CC_HOOK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert set(mod._SURFACE_TOOLS) == {
        "recall",
        "recall_full_turns",
        "fetch",
        "fetch_session",
        "remember",
    }
