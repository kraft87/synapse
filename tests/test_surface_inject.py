"""The PreToolUse hooks that inject `self_session` into synapse tools.

These hooks used to inject a `surface` (this host's name) as well, which is how schema
053 keyed per-host trust. Schema 054 deleted that lane: trust rides on the per-device
TOKEN the client authenticates with, so a hostname the client asserts is evidence of
nothing. The most load-bearing assertion in this file is now a NEGATIVE one — that no
hook re-grows a surface injection — because a client that starts sending one again
would look harmless while quietly reviving a spoofable trust input.

Both the Claude Code hook and its Codex port are covered here; they are separate files
that must not drift. Both also add `self_since` (the time of the session's last
compaction, read from the transcript the harness hands the hook) to recall and
recall_full_turns, and to nothing else.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.helpers.client_core import transcript

_ROOT = Path(__file__).resolve().parent.parent
_CC_HOOK = _ROOT / "plugin" / "scripts" / "self_session_inject.py"
_CODEX_HOOK = _ROOT / "plugin-codex" / "hooks" / "pre_tool_use.py"

_MCP = "mcp__plugin_synapse_synapse__"

#: The tools whose input either hook may rewrite. `fetch` is absent: it has no session
#: dimension, and losing the surface lane left it with nothing to receive.
_INJECTED_TOOLS = ("recall", "recall_full_turns", "recall_feedback", "fetch_session", "remember")


def _run(hook: Path, payload: dict) -> dict | None:
    """Run the hook as the harness does — a subprocess over stdin/stdout. Returns the
    parsed updatedInput block, or None when the hook chose to emit nothing."""
    proc = subprocess.run(
        [sys.executable, str(hook)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        # SYNAPSE_SURFACE is still set on purpose: if a hook ever starts reading it
        # again, the negative assertions below have something to catch it with.
        env={
            "SYNAPSE_SURFACE": "test-inject-host",
            "PATH": "/usr/bin:/bin",
            "HOME": "/nonexistent",
        },
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr  # fail-open: never a non-zero exit
    out = proc.stdout.strip()
    return json.loads(out)["hookSpecificOutput"]["updatedInput"] if out else None


_HOOKS = pytest.mark.parametrize("hook", [_CC_HOOK, _CODEX_HOOK], ids=["claude-code", "codex"])


@_HOOKS
@pytest.mark.parametrize("tool", _INJECTED_TOOLS)
def test_no_hook_injects_a_surface_any_more(hook, tool):
    """The regression pin for schema 054. A client-asserted host name is exactly the
    spoofable input credential-bound trust removed; nothing may start sending it back."""
    ti = _run(hook, {"tool_name": _MCP + tool, "session_id": "s-1", "tool_input": {"query": "q"}})
    assert ti is None or "surface" not in ti


@_HOOKS
@pytest.mark.parametrize("tool", ["recall", "recall_full_turns", "fetch_session"])
def test_self_session_is_injected_into_the_recall_family(hook, tool):
    ti = _run(hook, {"tool_name": _MCP + tool, "session_id": "s-1", "tool_input": {}})
    assert ti["self_session"] == "s-1"


@_HOOKS
@pytest.mark.parametrize("tool", ["recall_feedback", "remember"])
def test_session_id_tools_never_get_self_session(hook, tool):
    """recall_feedback and remember only accept `session_id`; injecting
    `self_session` fails their pydantic validation and kills the whole call
    (regression: remember was unusable under plugin 0.17.0)."""
    ti = _run(
        hook,
        {"tool_name": _MCP + tool, "session_id": "s-1", "tool_input": {"query": "q"}},
    )
    assert ti["session_id"] == "s-1" and "self_session" not in ti


@_HOOKS
def test_fetch_is_no_longer_touched_at_all(hook):
    """fetch has no session dimension, and the surface lane is gone — so the hook has
    nothing to add and must emit nothing rather than an unknown argument."""
    ti = _run(hook, {"tool_name": _MCP + "fetch", "session_id": "s-1", "tool_input": {"ids": []}})
    assert ti is None


@_HOOKS
def test_an_already_set_value_is_never_overwritten(hook):
    """A retry or a nested call must not have its inputs rewritten underneath it."""
    ti = _run(
        hook,
        {
            "tool_name": _MCP + "recall",
            "session_id": "s-1",
            "tool_input": {"self_session": "s-orig"},
        },
    )
    assert ti is None or ti["self_session"] == "s-orig"


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
        env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"},
        timeout=30,
    )
    assert proc.returncode == 0 and proc.stdout.strip() == ""


def test_codex_hook_declares_the_permission_decision():
    """Codex requires permissionDecision alongside updatedInput; without it the rewrite
    is dropped and the session id never arrives."""
    proc = subprocess.run(
        [sys.executable, str(_CODEX_HOOK)],
        input=json.dumps(
            {"tool_name": _MCP + "recall", "session_id": "s-1", "tool_input": {"query": "q"}}
        ),
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"},
        timeout=30,
    )
    block = json.loads(proc.stdout)["hookSpecificOutput"]
    assert block["permissionDecision"] == "allow"


def test_bundled_hook_declares_the_permission_decision():
    proc = subprocess.run(
        [sys.executable, str(_CC_HOOK)],
        input=json.dumps(
            {"tool_name": _MCP + "recall", "session_id": "s-1", "tool_input": {"query": "q"}}
        ),
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"},
        timeout=30,
    )
    block = json.loads(proc.stdout)["hookSpecificOutput"]
    assert block["permissionDecision"] == "allow"


def test_hook_matchers_cover_exactly_the_injected_tools():
    """The Claude Code matcher, the Codex matcher and the in-script guards must agree,
    or a tool silently stops receiving its session id."""
    hooks_json = json.loads((_ROOT / "plugin" / "hooks" / "hooks.json").read_text())
    matcher = hooks_json["hooks"]["PreToolUse"][0]["matcher"]
    codex_install = (_ROOT / "plugin-codex" / "install.py").read_text()

    expected = "(" + "|".join(_INJECTED_TOOLS) + ")$"
    assert matcher.endswith(expected)
    assert expected in codex_install
    for hook in (_CC_HOOK, _CODEX_HOOK):
        assert expected in hook.read_text()


def test_no_client_hook_still_references_a_surface_param():
    """Belt to the negative test's braces: neither hook should even mention the param.
    A leftover helper is how a deleted lane grows back."""
    for hook in (_CC_HOOK, _CODEX_HOOK):
        body = hook.read_text()
        code = "\n".join(
            line for line in body.splitlines() if not line.lstrip().startswith("#")
        ).split('"""')
        # Drop docstrings (odd-indexed chunks); they explain the removal on purpose.
        live = "".join(chunk for i, chunk in enumerate(code) if i % 2 == 0)
        assert '"surface"' not in live and "_SURFACE_TOOLS" not in live


# ---------------------------------------------------------------------------
# self_since: the time of the session's last compaction
# ---------------------------------------------------------------------------

_AT = "2026-11-12T14:10:54.738Z"


def _with_transcript(hook: Path, folder: Path, tool: str, tool_input: dict, at=_AT) -> dict:
    path = transcript(folder, codex=hook == _CODEX_HOOK, at=at)
    return {
        "tool_name": _MCP + tool,
        "session_id": "s-1",
        "tool_input": tool_input,
        "transcript_path": str(path),
    }


@_HOOKS
@pytest.mark.parametrize("tool", ["recall", "recall_full_turns"])
def test_self_since_is_added_after_a_compaction(hook, tool, tmp_path):
    ti = _run(hook, _with_transcript(hook, tmp_path, tool, {"query": "q"}))
    assert ti == {"query": "q", "self_session": "s-1", "self_since": _AT}


@_HOOKS
@pytest.mark.parametrize("tool", ["recall", "recall_full_turns"])
def test_self_since_is_added_when_the_input_already_names_this_session(hook, tool, tmp_path):
    ti = _run(hook, _with_transcript(hook, tmp_path, tool, {"self_session": "s-1"}))
    assert ti == {"self_session": "s-1", "self_since": _AT}


@_HOOKS
@pytest.mark.parametrize("tool", ["fetch_session", "recall_feedback", "remember"])
def test_self_since_never_goes_to_the_other_tools(hook, tool, tmp_path):
    """Their schemas reject an unknown argument: sending it would fail the call."""
    ti = _run(hook, _with_transcript(hook, tmp_path, tool, {"query": "q"}))
    assert ti is not None and "self_since" not in ti


@_HOOKS
def test_self_since_never_goes_with_someone_elses_session(hook, tmp_path):
    ti = _run(hook, _with_transcript(hook, tmp_path, "recall", {"self_session": "s-orig"}))
    assert ti is None


@_HOOKS
def test_a_self_since_already_set_is_never_overwritten(hook, tmp_path):
    kept = {"query": "q", "self_since": "2026-01-01T00:00:00Z"}
    ti = _run(hook, _with_transcript(hook, tmp_path, "recall", dict(kept)))
    assert ti == {**kept, "self_session": "s-1"}
    both = {"self_session": "s-1", "self_since": "2026-01-01T00:00:00Z"}
    assert _run(hook, _with_transcript(hook, tmp_path, "recall", dict(both))) is None


@_HOOKS
@pytest.mark.parametrize("case", ["never-compacted", "missing-file", "no-transcript-path"])
def test_no_compaction_means_no_self_since(hook, case, tmp_path):
    payload = _with_transcript(hook, tmp_path, "recall", {"query": "q"}, at=None)
    if case == "missing-file":
        payload["transcript_path"] = str(tmp_path / "gone.jsonl")
    elif case == "no-transcript-path":
        del payload["transcript_path"]
    assert _run(hook, payload) == {"query": "q", "self_session": "s-1"}


@_HOOKS
def test_a_client_core_that_cannot_load_costs_only_the_cut(hook, tmp_path):
    """The hook copied where no client core sits beside it: the import fails inside the
    hook, and the session id still goes."""
    rel = hook.relative_to(_ROOT)
    copy = tmp_path / "client" / rel
    copy.parent.mkdir(parents=True)
    copy.write_bytes(hook.read_bytes())
    ti = _run(copy, _with_transcript(hook, tmp_path, "recall", {"query": "q"}))
    assert ti == {"query": "q", "self_session": "s-1"}
