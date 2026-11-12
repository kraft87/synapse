"""The shared transcript line rules (ingestion/render.py), pinned.

Unit tests for each rule at its exact boundary, a golden render per live parser, and a
source guard that keeps the line prefixes out of the parser files.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from ingestion import render
from ingestion.codex_client import CodexRolloutParser
from ingestion.jsonl_client import JSONLParser
from ingestion.messages_client import MessagesParser
from ingestion.render import (
    assistant_line,
    context_line,
    group_turns,
    join_lines,
    repeated_span,
    result_line,
    strip_nulls,
    title_line,
    tool_line,
    user_line,
)


class TestCaps:
    def test_values(self):
        assert render.CONTEXT_CAP == 300
        assert render.USER_CAP == 3000
        assert render.ASSISTANT_CAP == 3000
        assert render.TOOL_CAP == 300
        assert render.RESULT_CAP == 500
        assert render.RESULT_MIN_CHARS == 20
        assert render.TITLE_CAP == 200
        assert render.LINE_SEP == "\n\n"


class TestStripNulls:
    def test_removes_every_nul(self):
        assert strip_nulls("a\x00b\x00\x00c") == "abc"

    def test_text_without_nul_is_unchanged(self):
        assert strip_nulls("plain text") == "plain text"
        assert strip_nulls("") == ""


class TestContextLine:
    def test_shape(self):
        assert context_line("previous reply") == "[context] previous reply"

    def test_strips_before_capping(self):
        text = "   " + "a" * 299 + "b" + "c" + "  \n"
        assert context_line(text) == "[context] " + "a" * 299 + "b"

    def test_cap_boundary(self):
        assert context_line("x" * 300) == "[context] " + "x" * 300
        assert context_line("x" * 301) == "[context] " + "x" * 300


class TestUserLine:
    def test_shape_keeps_whitespace(self):
        assert user_line(" hi ") == "[user]  hi "

    def test_cap_boundary(self):
        assert user_line("u" * 3000) == "[user] " + "u" * 3000
        assert user_line("u" * 3001) == "[user] " + "u" * 3000


class TestAssistantLine:
    def test_shape(self):
        assert assistant_line("Done.") == "[assistant] Done."

    def test_cap_boundary(self):
        assert assistant_line("a" * 3000) == "[assistant] " + "a" * 3000
        assert assistant_line("a" * 3001) == "[assistant] " + "a" * 3000


class TestToolLine:
    def test_shape(self):
        assert tool_line("Bash", "ls -la") == "[tool:Bash] ls -la"

    def test_cap_boundary(self):
        assert tool_line("t", "d" * 300) == "[tool:t] " + "d" * 300
        assert tool_line("t", "d" * 301) == "[tool:t] " + "d" * 300

    def test_name_is_formatted_as_given(self):
        # A transcript's tool name is not validated: null renders as None.
        assert tool_line(None, "x") == "[tool:None] x"

    def test_empty_detail_still_renders(self):
        # Whether an empty detail gets a line is each parser's own rule.
        assert tool_line("t", "") == "[tool:t] "


class TestResultLine:
    def test_twenty_chars_is_no_line(self):
        assert result_line("r" * 20) is None
        assert result_line("") is None

    def test_twenty_one_chars_is_a_line(self):
        assert result_line("r" * 21) == "[result] " + "r" * 21

    def test_cap_boundary(self):
        assert result_line("r" * 500) == "[result] " + "r" * 500
        assert result_line("r" * 501) == "[result] " + "r" * 500


class TestTitleLine:
    def test_shape_and_cap(self):
        assert title_line("My chat") == "[title] My chat"
        assert title_line("t" * 201) == "[title] " + "t" * 200


class TestJoinLines:
    def test_blank_line_between_lines(self):
        assert join_lines(["[user] a", "[assistant] b"]) == "[user] a\n\n[assistant] b"

    def test_strips_nuls(self):
        assert join_lines(["[user] a\x00", "\x00[assistant] b"]) == "[user] a\n\n[assistant] b"

    def test_empty(self):
        assert join_lines([]) == ""


class TestGroupTurns:
    @staticmethod
    def _starts(item: str) -> bool:
        return item.startswith("U")

    def test_each_starter_opens_a_group(self):
        items = ["U1", "a", "b", "U2", "c", "U3"]
        assert group_turns(items, self._starts) == [["U1", "a", "b"], ["U2", "c"], ["U3"]]

    def test_leading_items_form_their_own_group(self):
        assert group_turns(["a", "b", "U1", "c"], self._starts) == [["a", "b"], ["U1", "c"]]

    def test_first_item_starter_opens_no_empty_group(self):
        assert group_turns(["U1", "U2"], self._starts) == [["U1"], ["U2"]]

    def test_no_starters_is_one_group(self):
        assert group_turns(["a", "b"], self._starts) == [["a", "b"]]

    def test_empty(self):
        assert group_turns([], self._starts) == []

    def test_predicate_sees_every_item_once_in_order(self):
        seen: list[str] = []

        def starts(item: str) -> bool:
            seen.append(item)
            return item.startswith("U")

        group_turns(["a", "U1", "b"], starts)
        assert seen == ["a", "U1", "b"]


class TestRepeatedSpan:
    def test_first_sighting_is_recorded(self):
        seen: set[str] = set()
        assert repeated_span(seen, "s1") is False
        assert seen == {"s1"}
        assert repeated_span(seen, "s1") is True
        assert repeated_span(seen, "s2") is False

    def test_no_span_id_is_never_a_repeat(self):
        seen: set[str] = set()
        assert repeated_span(seen, None) is False
        assert repeated_span(seen, None) is False
        assert seen == set()


# ---------------------------------------------------------------------------------------
# Golden renders, one per live parser. The three parsers differ on purpose in where tool
# lines sit, how text blocks combine, what is dropped and when [context] is added; each
# golden pins its parser's full content string so those differences stay visible.


class TestJSONLGolden:
    @staticmethod
    def _rec(rtype: str, content: object, uuid: str) -> dict:
        message: dict = {"role": rtype, "content": content}
        if rtype == "assistant":
            message["model"] = "model-1"
        return {
            "type": rtype,
            "message": message,
            "uuid": uuid,
            "sessionId": "sess-g",
            "cwd": "/home/user/dev/proj",
            "timestamp": "2026-11-12T10:00:00Z",
        }

    def _records(self) -> list[dict]:
        r = self._rec
        return [
            r("user", "list the\x00 files", "u1"),
            # Text BEFORE the tool call in the record; jsonl still renders the tool first.
            r(
                "assistant",
                [
                    {"type": "text", "text": "Let me look."},
                    {"type": "tool_use", "name": "Bash", "input": {"command": "  ls -la  "}},
                ],
                "a1",
            ),
            r(
                "user",
                [
                    {"type": "tool_result", "content": "total 2\n-rw-r--r-- 1 u u 0 notes.txt"},
                    {"type": "tool_result", "content": "ok"},  # 2 chars: no line
                ],
                "r1",
            ),
            # Two text blocks: jsonl keeps only the last.
            r(
                "assistant",
                [{"type": "text", "text": "draft"}, {"type": "text", "text": "One file."}],
                "a2",
            ),
            # Hook feedback opens a turn, but its text is not the user's.
            r("user", "Stop hook feedback:\nCall recall_feedback.", "h1"),
            r("assistant", [{"type": "text", "text": "Feedback sent."}], "a3"),
            # A bounce with no reply leaves only [context]: skipped, sequence 3 used up.
            r("user", "Stop hook feedback:\nAgain.", "h2"),
            r("user", "thanks", "u2"),
            r("assistant", [{"type": "text", "text": "A" * 3001}], "a4"),
            # The same turn again (a compaction re-dump): same span id, emitted once.
            r("user", "thanks", "u2"),
            r("assistant", [{"type": "text", "text": "A" * 3001}], "a4"),
            r("user", "bye", "u3"),
        ]

    def test_content(self):
        eps = JSONLParser().parse_records(self._records(), "golden")
        assert [e.sequence for e in eps] == [1, 2, 4, 6]
        assert [e.span_id for e in eps] == [
            "claude_code:a2",
            "claude_code:a3",
            "claude_code:a4",
            "claude_code:u3",
        ]
        assert eps[0].content == (
            "[user] list the files\n\n"
            "[tool:Bash]   ls -la  \n\n"
            "[assistant] Let me look.\n\n"
            "[result] total 2\n-rw-r--r-- 1 u u 0 notes.txt\n\n"
            "[assistant] One file."
        )
        assert eps[1].content == "[context] One file.\n\n[assistant] Feedback sent."
        assert eps[2].content == (
            "[context] Feedback sent.\n\n[user] thanks\n\n[assistant] " + "A" * 3000
        )
        assert eps[3].content == "[context] " + "A" * 300 + "\n\n[user] bye"

    def test_turns(self):
        eps = JSONLParser().parse_records(self._records(), "golden")
        assert [e.human_turn for e in eps] == ["list the files", None, "thanks", "bye"]
        assert [e.assistant_turn for e in eps] == [
            "One file.",
            "Feedback sent.",
            "A" * 3001,  # the full text, not the capped line
            None,
        ]


class TestCodexGolden:
    SID = "019ffdbe-ff9a-7b41-892a-ddc6221d1a64"

    @staticmethod
    def _rec(rtype: str, payload: dict) -> dict:
        return {"timestamp": "2026-11-12T10:00:00Z", "type": rtype, "payload": payload}

    def _msg(self, role: str, texts: list[str], msg_id: str) -> dict:
        kind = "output_text" if role == "assistant" else "input_text"
        content = [{"type": kind, "text": t} for t in texts]
        return self._rec(
            "response_item", {"type": "message", "id": msg_id, "role": role, "content": content}
        )

    def _records(self) -> list[dict]:
        item = self._rec
        return [
            item("session_meta", {"session_id": self.SID, "cwd": "/home/user/dev/proj"}),
            item("turn_context", {"cwd": "/home/user/dev/proj", "model": "model-1"}),
            self._msg("user", ["<environment_context>x</environment_context>"], "env"),
            # Two text parts: codex joins them with a newline.
            self._msg("user", ["run the", "tests"], "u1"),
            item(
                "response_item",
                {"type": "custom_tool_call", "id": "t1", "name": "exec", "input": "  pytest -q \n"},
            ),
            item(
                "response_item",
                {
                    "type": "custom_tool_call_output",
                    "id": "o1",
                    "output": "47 passed in 3.21s, all green",
                },
            ),
            item("response_item", {"type": "function_call_output", "id": "o2", "output": "ok"}),
            self._msg("assistant", ["All 47 pass.", "Nothing failed."], "a1"),
            # A tool call after the reply renders after it: codex keeps record order.
            item(
                "response_item",
                {"type": "function_call", "id": "f1", "name": "shell", "arguments": "git status"},
            ),
            self._msg("user", ["q" * 3001], "u2"),
            self._msg("assistant", ["A" * 3001], "a2"),
            # A resume re-dump of the same turn: same span id, emitted once.
            self._msg("user", ["q" * 3001], "u2"),
            self._msg("assistant", ["A" * 3001], "a2"),
            self._msg("user", ["bye"], "u3"),
        ]

    def test_content(self):
        eps = CodexRolloutParser().parse_records(self._records(), "golden")
        assert [e.sequence for e in eps] == [1, 2, 4]
        assert [e.span_id for e in eps] == ["codex:f1", "codex:a2", "codex:u3"]
        assert eps[0].content == (
            "[user] run the\ntests\n\n"
            "[tool:exec] pytest -q\n\n"
            "[result] 47 passed in 3.21s, all green\n\n"
            "[assistant] All 47 pass.\nNothing failed.\n\n"
            "[tool:shell] git status"
        )
        assert eps[1].content == (
            "[context] All 47 pass.\nNothing failed.\n\n"
            "[user] " + "q" * 3000 + "\n\n[assistant] " + "A" * 3000
        )
        assert eps[2].content == "[context] " + "A" * 300 + "\n\n[user] bye"

    def test_turns(self):
        eps = CodexRolloutParser().parse_records(self._records(), "golden")
        assert [e.human_turn for e in eps] == ["run the\ntests", "q" * 3000, "bye"]
        assert [e.assistant_turn for e in eps] == [
            "All 47 pass.\nNothing failed.",
            "A" * 3001,
            None,
        ]


class TestMessagesGolden:
    SID = "hermes-golden"

    @staticmethod
    def _records() -> list[dict]:
        return [
            {"role": "system", "content": "be brief"},  # not a kept role
            {"role": "user", "id": "m1", "content": "list the files"},
            # messages renders [assistant] BEFORE the row's tool calls.
            {
                "role": "assistant",
                "id": "m2",
                "content": "Let me look.",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "shell", "arguments": "  ls -la  "}},
                    {"name": "read", "arguments": {"path": "/tmp/x", "lines": 5}},
                ],
            },
            {"role": "tool", "id": "m3", "content": "total 2\n-rw-r--r-- 1 u u 0 notes.txt"},
            {"role": "tool", "id": "m4", "content": "ok"},  # 2 chars: no line
            # Two text parts: joined with a newline.
            {
                "role": "assistant",
                "id": "m5",
                "content": [
                    {"type": "output_text", "text": "One file:"},
                    {"type": "text", "text": "notes.txt."},
                ],
            },
            {"role": "user", "content": "thanks"},  # no ids: the span id is a hash
            {"role": "assistant", "content": "You're welcome."},
            # The same turn again under a different [context]: it hashes the same, because
            # the hash covers the turn's own lines only, so it is emitted once.
            {"role": "user", "content": "thanks"},
            {"role": "assistant", "content": "You're welcome."},
            {"role": "user", "content": "<memory-context>recalled</memory-context>bye"},
        ]

    @staticmethod
    def _hash_span(own_lines: str) -> str:
        digest = hashlib.sha256(own_lines.encode()).hexdigest()[:24]
        return f"hermes:hermes-golden:{digest}"

    def test_content(self):
        eps = MessagesParser().parse_records(
            self._records(), "golden", session_id=self.SID, client="hermes"
        )
        assert [e.sequence for e in eps] == [1, 2, 4]
        assert [e.span_id for e in eps] == [
            "hermes:m5",
            self._hash_span("[user] thanks\n\n[assistant] You're welcome."),
            self._hash_span("[user] bye"),
        ]
        assert eps[0].content == (
            "[user] list the files\n\n"
            "[assistant] Let me look.\n\n"
            "[tool:shell] ls -la\n\n"
            '[tool:read] {"lines": 5, "path": "/tmp/x"}\n\n'
            "[result] total 2\n-rw-r--r-- 1 u u 0 notes.txt\n\n"
            "[assistant] One file:\nnotes.txt."
        )
        assert eps[1].content == (
            "[context] One file:\nnotes.txt.\n\n[user] thanks\n\n[assistant] You're welcome."
        )
        assert eps[2].content == "[context] You're welcome.\n\n[user] bye"

    def test_turns(self):
        eps = MessagesParser().parse_records(
            self._records(), "golden", session_id=self.SID, client="hermes"
        )
        assert [e.human_turn for e in eps] == ["list the files", "thanks", "bye"]
        assert [e.assistant_turn for e in eps] == ["One file:\nnotes.txt.", "You're welcome.", None]


# ---------------------------------------------------------------------------------------
# Drift guard: the line prefixes are spelled only in ingestion/render.py.

_ROOT = Path(__file__).resolve().parents[1]
_PREFIXES = ("[user] ", "[assistant] ", "[context] ", "[result] ", "[tool:")
_CLIENT_FILES = sorted((_ROOT / "ingestion").glob("*_client.py"))

#: Literals a parser keeps on purpose, each with its reason.
_ALLOWED = {
    # The claude.ai export caps [user] and [assistant] at 5000, not render's 3000.
    "claude_ai_client.py": {"[user] ", "[assistant] "},
}


def _docstring_nodes(tree: ast.AST) -> set[int]:
    nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                nodes.add(id(body[0].value))
    return nodes


def _prefixes_in_literals(path: Path) -> set[str]:
    """The prefixes spelled in a string or f-string literal, docstrings excluded."""
    tree = ast.parse(path.read_text())
    docstrings = _docstring_nodes(tree)
    found = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            found.update(p for p in _PREFIXES if p in node.value)
    return found


class TestDriftGuard:
    def test_every_parser_is_guarded(self):
        names = {p.name for p in _CLIENT_FILES}
        assert {
            "jsonl_client.py",
            "codex_client.py",
            "messages_client.py",
            "openai_client.py",
            "cursor_sqlite_client.py",
            "claude_ai_client.py",
        } <= names

    @pytest.mark.parametrize("path", _CLIENT_FILES, ids=lambda p: p.name)
    def test_no_line_prefix_literals(self, path: Path):
        assert _prefixes_in_literals(path) == _ALLOWED.get(path.name, set())

    def test_guard_reads_fstrings_and_skips_docstrings(self, tmp_path: Path):
        src = tmp_path / "x_client.py"
        src.write_text(
            '"""Renders [context] and [result] lines."""\n'
            "def f(name, t):\n"
            '    """An [assistant] line."""\n'
            '    return f"[tool:{name}] x", "[user] " + t\n'
        )
        assert _prefixes_in_literals(src) == {"[tool:", "[user] "}

    @pytest.mark.parametrize("path", _CLIENT_FILES, ids=lambda p: p.name)
    def test_no_private_strip_nulls(self, path: Path):
        tree = ast.parse(path.read_text())
        defs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert "_strip_nulls" not in defs
