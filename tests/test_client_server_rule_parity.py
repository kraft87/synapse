"""Client and server copies of the same transcript rules stay the same.

The Stop hooks run on the user's machine with a bare Python and cannot import the
server's parsers, so each carries its own copy of two rules the server applies when it
groups records into turns:

  * the machinery prefixes: user text that is harness plumbing, not a person typing
    (plugin/scripts/ingest_hook.py vs ingestion/jsonl_client.py, and
    plugin-codex/hooks/synapse_stop_hook.py vs ingestion/codex_client.py);
  * whether a record starts a turn: the hooks chunk, seed and trim at turn boundaries
    (``ingest_hook._is_turn_start``, ``synapse_stop_hook._is_user_boundary``).

For the second, the server's answer is what its parser does with the record: it starts
a turn when it survives the parser's record filter and opens a new group after a real
user record. Each case below asserts the server's answer, then that the hook gives the
same one. A case where the two differ today is marked ``xfail(strict=True)`` with the
reason, so the difference stays visible and the marker has to go once they agree.

The Hermes normaliser in the client core and the server's messages parser strip the
same ``<memory-context>`` block; the last test pins that pattern.

The hooks are loaded by path, as they run, over a scratch user (no real config).
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from ingestion import codex_client, jsonl_client, messages_client
from ingestion.codex_client import CodexRolloutParser
from ingestion.jsonl_client import JSONLParser
from tests.helpers.client_core import REPO, drop_scripts, load_script, scratch_user, use_core

use_core()
from synapse_client.harnesses.hermes import messages as hermes_messages  # noqa: E402

_CODEX_HOOK = REPO / "plugin-codex" / "hooks" / "synapse_stop_hook.py"


@pytest.fixture(scope="module")
def claude_hook(tmp_path_factory) -> ModuleType:
    with pytest.MonkeyPatch.context() as mp:
        scratch_user(mp, tmp_path_factory.mktemp("claude-hook"))
        try:
            return load_script("ingest_hook")
        finally:
            drop_scripts()


@pytest.fixture(scope="module")
def codex_hook(tmp_path_factory) -> ModuleType:
    """The Codex Stop hook puts plugin-codex/scripts first on sys.path and imports
    ``common`` from there; both are put back after the load."""
    tmp: Path = tmp_path_factory.mktemp("codex-hook")
    path_before = list(sys.path)
    saved_common = sys.modules.pop("common", None)
    with pytest.MonkeyPatch.context() as mp:
        scratch_user(mp, tmp)
        mp.setenv("SYNAPSE_CODEX_CURSORS", str(tmp / "codex_cursors.json"))
        mp.setenv("SYNAPSE_CODEX_HOOK_LOG", str(tmp / "codex-hook.log"))
        try:
            spec = importlib.util.spec_from_file_location("synapse_stop_hook", _CODEX_HOOK)
            assert spec and spec.loader
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
        finally:
            drop_scripts()
            sys.modules.pop("common", None)
            if saved_common is not None:
                sys.modules["common"] = saved_common
            sys.path[:] = path_before


# ---------------------------------------------------------------------------
# Machinery prefixes
# ---------------------------------------------------------------------------


def test_claude_hook_machinery_prefixes_match_the_server(claude_hook):
    assert claude_hook._MACHINERY_PREFIXES == jsonl_client._MACHINERY_PREFIXES


def test_codex_hook_machinery_prefixes_match_the_server(codex_hook):
    assert codex_hook._MACHINERY_PREFIXES == codex_client._MACHINERY_PREFIXES


# ---------------------------------------------------------------------------
# Claude Code: does this record start a turn?
# ---------------------------------------------------------------------------


def _user(content: Any, **extra: Any) -> dict[str, Any]:
    return {
        "type": "user",
        "uuid": "u-case",
        "sessionId": "s-parity",
        **extra,
        "message": {"content": content},
    }


def _text(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


_TOOL_RESULT = {"type": "tool_result", "tool_use_id": "toolu_1", "content": "file contents"}

_CLAUDE_CASES = [
    pytest.param(_user("what does the board show?"), True, id="user-string"),
    pytest.param(_user([_text("what does the board show?")]), True, id="user-text-block"),
    *[
        pytest.param(_user(f"{prefix}x"), False, id=f"machinery-{prefix.strip('<>')}")
        for prefix in jsonl_client._MACHINERY_PREFIXES
    ],
    pytest.param(_user("  <system-reminder>x"), False, id="machinery-after-whitespace"),
    pytest.param(
        _user([_text("<command-name>/compact</command-name>")]), False, id="machinery-block"
    ),
    pytest.param(
        _user([_text("<system-reminder>x</system-reminder>"), _text("and a real question")]),
        True,
        id="machinery-block-then-real-text",
    ),
    pytest.param(_user([_TOOL_RESULT]), False, id="tool-result"),
    pytest.param(
        _user([_TOOL_RESULT, _text("[Request interrupted by user]")]),
        True,
        id="tool-result-with-text",
    ),
    pytest.param(
        {
            "type": "assistant",
            "uuid": "a-case",
            "sessionId": "s-parity",
            "message": {"content": [_text("an answer")]},
        },
        False,
        id="assistant",
    ),
    pytest.param(_user(""), False, id="empty-string"),
    pytest.param(_user("   "), False, id="whitespace-only"),
    pytest.param(_user([]), False, id="empty-block-list"),
    pytest.param(
        {"type": "user", "uuid": "u-case", "sessionId": "s-parity", "message": {}},
        False,
        id="message-without-content",
    ),
    pytest.param(
        _user([{"type": "image", "source": {"type": "base64", "data": ""}}]),
        False,
        id="image-only",
    ),
    pytest.param(
        _user("Stop hook feedback:\nplease call recall_feedback"), True, id="stop-hook-feedback"
    ),
    pytest.param(
        _user(
            "This session is being continued from a previous conversation.", isCompactSummary=True
        ),
        True,
        id="compact-summary",
    ),
    pytest.param({"type": "summary", "summary": "a title"}, False, id="summary-record"),
    pytest.param({"type": "system", "content": "a notice"}, False, id="system-record"),
    pytest.param(
        _user("a real question inside a subagent", isSidechain=True),
        False,
        id="sidechain-user",
        marks=pytest.mark.xfail(
            strict=True,
            reason=(
                "ingest_hook._is_turn_start does not look at isSidechain; the server drops "
                "sidechain records before grouping (JSONLParser._filter_records), so a "
                "sidechain user record with real text is a boundary for the hook and "
                "nothing for the server"
            ),
        ),
    ),
    pytest.param(
        {"type": "user", "uuid": "u-case", "sessionId": "s-parity", "message": None},
        False,
        id="message-null",
        marks=pytest.mark.xfail(
            raises=AttributeError,
            strict=True,
            reason=(
                "the hook reads `message` with `or {}` and answers False; the server's "
                "record.get('message', {}) returns None for an explicit null and its "
                "filter raises AttributeError"
            ),
        ),
    ),
]


def _server_claude_starts(rec: dict[str, Any]) -> bool:
    """The server's answer: after a real user record, does ``rec`` open a new group?"""
    opener = _user("the first question")
    opener["uuid"] = "u-opener"
    groups = JSONLParser._group_into_turns(
        JSONLParser._filter_records([opener, copy.deepcopy(rec)])
    )
    assert len(groups) in (1, 2)
    return len(groups) == 2


@pytest.mark.parametrize(("rec", "starts"), _CLAUDE_CASES)
def test_claude_hook_and_server_agree_on_turn_starts(claude_hook, rec, starts):
    assert _server_claude_starts(rec) is starts
    assert claude_hook._is_turn_start(copy.deepcopy(rec)) is starts


# ---------------------------------------------------------------------------
# Codex: does this rollout record start a turn?
# ---------------------------------------------------------------------------


def _item(payload: dict[str, Any] | None, rtype: str = "response_item") -> dict[str, Any]:
    return {"timestamp": "2026-11-12T10:00:00.000Z", "type": rtype, "payload": payload}


def _message(role: str, content: Any) -> dict[str, Any]:
    return _item({"type": "message", "role": role, "content": content})


def _input(text: str) -> dict[str, str]:
    return {"type": "input_text", "text": text}


def _output(text: str) -> dict[str, str]:
    return {"type": "output_text", "text": text}


_CODEX_CASES = [
    pytest.param(
        _message("user", [_input("what does the board show?")]), True, id="user-input-text"
    ),
    pytest.param(_message("user", "what does the board show?"), True, id="user-string"),
    pytest.param(
        _message("user", [{"type": "text", "text": "what does the board show?"}]),
        True,
        id="user-text-part",
    ),
    *[
        pytest.param(
            _message("user", [_input(f"{prefix}x")]),
            False,
            id=f"machinery-{prefix.strip('<>')}",
        )
        for prefix in codex_client._MACHINERY_PREFIXES
    ],
    pytest.param(
        _message("user", [_input("  <environment_context>x")]),
        False,
        id="machinery-after-whitespace",
    ),
    pytest.param(
        _message("user", [_input("<environment_context>x"), _input("and a real question")]),
        False,
        id="machinery-part-then-real-text",
    ),
    pytest.param(
        _message("user", [_input("a real question"), _input("<environment_context>x")]),
        True,
        id="real-text-then-machinery-part",
    ),
    pytest.param(_message("developer", [_input("injected instructions")]), False, id="developer"),
    pytest.param(_message("assistant", [_output("an answer")]), False, id="assistant"),
    pytest.param(
        _item({"type": "function_call", "name": "shell", "arguments": '{"cmd": "ls"}'}),
        False,
        id="function-call",
    ),
    pytest.param(
        _item({"type": "function_call_output", "call_id": "c1", "output": "a b c"}),
        False,
        id="function-call-output",
    ),
    pytest.param(
        _item({"type": "custom_tool_call", "name": "apply_patch", "input": "*** Begin Patch"}),
        False,
        id="custom-tool-call",
    ),
    pytest.param(_item({"type": "reasoning", "encrypted_content": "gAAAA"}), False, id="reasoning"),
    pytest.param(_message("user", []), False, id="user-empty-parts"),
    pytest.param(_message("user", [_input("   ")]), False, id="user-whitespace-only"),
    pytest.param(
        _message("user", [{"type": "input_image", "image_url": "data:image/png;base64,"}]),
        False,
        id="user-image-only",
    ),
    pytest.param(
        _item({"type": "user_message", "message": "what does the board show?"}, "event_msg"),
        False,
        id="event-msg-user-message",
    ),
    pytest.param(_item({"cwd": "/work", "model": "m"}, "turn_context"), False, id="turn-context"),
    pytest.param(_item({"id": "sess", "cwd": "/work"}, "session_meta"), False, id="session-meta"),
    pytest.param(_item(None), False, id="payload-null"),
    pytest.param(
        _message("user", [_output("a question carried as output_text")]),
        True,
        id="user-output-text-only",
        marks=pytest.mark.xfail(
            strict=True,
            reason=(
                "synapse_stop_hook._is_user_boundary reads input_text and text parts only; "
                "the server's codex_client._content_text also reads output_text, so a user "
                "message carried in output_text parts is a turn for the server and not a "
                "boundary for the hook"
            ),
        ),
    ),
]


def _server_codex_starts(rec: dict[str, Any]) -> bool:
    """The server's answer: between a first exchange and a closing assistant message,
    does ``rec`` make the parser emit a second episode? Every payload gets its own id,
    so no group is dropped as a replay."""
    rec = copy.deepcopy(rec)
    if isinstance(rec.get("payload"), dict):
        rec["payload"]["id"] = "item-case"
    records = [
        _message("user", [_input("the first question")]),
        _message("assistant", [_output("the first answer")]),
        rec,
        _message("assistant", [_output("the closing answer")]),
    ]
    for i, r in enumerate((records[0], records[1], records[3])):
        r["payload"]["id"] = f"item-{i}"
    episodes = CodexRolloutParser().parse_records(records, "parity", session_id_hint="s-parity")
    assert len(episodes) in (1, 2)
    return len(episodes) == 2


@pytest.mark.parametrize(("rec", "starts"), _CODEX_CASES)
def test_codex_hook_and_server_agree_on_turn_starts(codex_hook, rec, starts):
    assert _server_codex_starts(rec) is starts
    assert codex_hook._is_user_boundary(copy.deepcopy(rec)) is starts


# ---------------------------------------------------------------------------
# Hermes: the memory-context block both sides strip from user text
# ---------------------------------------------------------------------------


def test_hermes_and_the_server_strip_the_same_memory_context():
    client, server = hermes_messages._MEMORY_CONTEXT, messages_client._MEMORY_CONTEXT
    assert client.pattern == server.pattern
    assert client.flags == server.flags
    sample = "<memory-context>\nrecalled\n</memory-context>\nthe question <memory-context>x</memory-context>"
    assert client.sub("", sample) == server.sub("", sample) == "\nthe question "
