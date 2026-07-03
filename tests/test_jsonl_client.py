"""Tests for the JSONL transcript parser (pure unit tests — no I/O beyond tmp files)."""

from __future__ import annotations

import json
from pathlib import Path

from ingestion.jsonl_client import (
    JSONLParser,
    _cwd_to_project,
    _extract_blocks,
    _is_user_turn,
)


def _write_jsonl(tmp_path: Path, records: list[dict]) -> Path:
    p = tmp_path / "session.jsonl"
    with open(p, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p


class TestIsUserTurn:
    def test_string_content_is_turn(self):
        assert _is_user_turn({"type": "user", "message": {"content": "hi there"}})

    def test_text_block_is_turn(self):
        assert _is_user_turn(
            {
                "type": "user",
                "message": {"content": [{"type": "text", "text": "hello"}]},
            }
        )

    def test_tool_result_is_not_turn(self):
        assert not _is_user_turn(
            {
                "type": "user",
                "message": {"content": [{"type": "tool_result", "content": "done"}]},
            }
        )

    def test_empty_string_is_not_turn(self):
        assert not _is_user_turn({"type": "user", "message": {"content": "   "}})

    def test_assistant_is_not_user_turn(self):
        assert not _is_user_turn({"type": "assistant", "message": {"content": "x"}})


class TestExtractBlocks:
    def test_user_string_content(self):
        text, tu, tr, model = _extract_blocks({"content": "hello world"}, "user")
        assert text == "hello world"
        assert tu == []
        assert tr == []
        assert model is None

    def test_assistant_text_and_tool_use(self):
        msg = {
            "model": "claude-sonnet-4-6",
            "content": [
                {"type": "thinking", "thinking": "think..."},
                {"type": "text", "text": "Sure, I'll run a search."},
                {"type": "tool_use", "name": "Grep", "input": {"query": "TODO", "path": "/x"}},
            ],
        }
        text, tu, tr, model = _extract_blocks(msg, "assistant")
        assert text == "Sure, I'll run a search."
        assert tu == ["[tool:Grep] TODO"]
        assert tr == []
        assert model == "claude-sonnet-4-6"

    def test_user_tool_result_long_text(self):
        msg = {
            "content": [
                {
                    "type": "tool_result",
                    "content": [{"type": "text", "text": "x" * 100}],
                }
            ]
        }
        text, _tu, tr, _ = _extract_blocks(msg, "user")
        assert text is None
        assert len(tr) == 1
        assert tr[0].startswith("[result]")

    def test_short_tool_result_filtered(self):
        msg = {"content": [{"type": "tool_result", "content": "ok"}]}
        _text, _tu, tr, _ = _extract_blocks(msg, "user")
        assert tr == []

    def test_tool_use_with_command(self):
        msg = {
            "content": [{"type": "tool_use", "name": "Bash", "input": {"command": "ls -la /tmp"}}]
        }
        _, tu, _, _ = _extract_blocks(msg, "assistant")
        assert tu == ["[tool:Bash] ls -la /tmp"]


class TestCwdToProject:
    def test_basename(self):
        assert _cwd_to_project("/home/user/services/synapse") == "synapse"

    def test_trailing_slash(self):
        assert _cwd_to_project("/home/user/scripts/") == "scripts"

    def test_none(self):
        assert _cwd_to_project(None) is None

    def test_root(self):
        assert _cwd_to_project("/") is None


class TestParseFile:
    def _basic_session(self) -> list[dict]:
        return [
            {"type": "last-prompt", "leafUuid": "x"},  # skip
            {
                "type": "user",
                "message": {"role": "user", "content": "hello, can you list files?"},
                "uuid": "u1",
                "sessionId": "sess-1",
                "cwd": "/home/user/services/synapse",
                "timestamp": "2026-05-02T10:00:00.000Z",
                "gitBranch": "main",
            },
            {
                "type": "assistant",
                "message": {
                    "model": "claude-sonnet-4-6",
                    "content": [
                        {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
                    ],
                },
                "uuid": "a1",
                "sessionId": "sess-1",
                "cwd": "/home/user/services/synapse",
                "timestamp": "2026-05-02T10:00:01.000Z",
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "content": "file1.py\nfile2.py\n" + "x" * 50,
                            "tool_use_id": "t1",
                        }
                    ]
                },
                "uuid": "u2",
                "sessionId": "sess-1",
                "cwd": "/home/user/services/synapse",
                "timestamp": "2026-05-02T10:00:02.000Z",
            },
            {
                "type": "assistant",
                "message": {
                    "model": "claude-sonnet-4-6",
                    "content": [{"type": "text", "text": "Here are the files in the directory."}],
                },
                "uuid": "a2",
                "sessionId": "sess-1",
                "cwd": "/home/user/services/synapse",
                "timestamp": "2026-05-02T10:00:03.000Z",
            },
            {
                "type": "user",
                "message": {"content": "thanks, now read file1.py"},
                "uuid": "u3",
                "sessionId": "sess-1",
                "cwd": "/home/user/services/synapse",
                "timestamp": "2026-05-02T10:00:10.000Z",
            },
            {
                "type": "assistant",
                "message": {
                    "model": "claude-sonnet-4-6",
                    "content": [{"type": "text", "text": "Reading the file now."}],
                },
                "uuid": "a3",
                "sessionId": "sess-1",
                "cwd": "/home/user/services/synapse",
                "timestamp": "2026-05-02T10:00:11.000Z",
            },
        ]

    def test_groups_into_user_turns(self, tmp_path: Path):
        path = _write_jsonl(tmp_path, self._basic_session())
        eps = JSONLParser().parse_file(path)
        assert len(eps) == 2

    def test_first_episode_has_human_and_assistant(self, tmp_path: Path):
        path = _write_jsonl(tmp_path, self._basic_session())
        eps = JSONLParser().parse_file(path)
        ep0 = eps[0]
        assert ep0.session_id == "sess-1"
        assert ep0.sequence == 1
        assert ep0.platform == "claude_code"
        assert ep0.model == "claude-sonnet-4-6"
        assert ep0.human_turn == "hello, can you list files?"
        assert ep0.assistant_turn == "Here are the files in the directory."
        assert "[user] hello" in ep0.content
        assert "[tool:Bash] ls" in ep0.content
        assert "[result]" in ep0.content
        assert "[assistant] Here are the files" in ep0.content
        assert ep0.project == "synapse"
        assert ep0.span_id == "jsonl:a2"
        assert ep0.source == "jsonl"

    def test_second_episode_has_context_prefix(self, tmp_path: Path):
        path = _write_jsonl(tmp_path, self._basic_session())
        eps = JSONLParser().parse_file(path)
        ep1 = eps[1]
        assert ep1.sequence == 2
        assert ep1.content.startswith("[context] Here are the files")
        assert "[user] thanks, now read file1.py" in ep1.content
        assert "[assistant] Reading the file now." in ep1.content

    def test_project_override(self, tmp_path: Path):
        path = _write_jsonl(tmp_path, self._basic_session())
        eps = JSONLParser().parse_file(path, project_override="transcribe-ai")
        assert all(ep.project == "transcribe-ai" for ep in eps)

    def test_skips_sidechain_records(self, tmp_path: Path):
        records = [
            *self._basic_session(),
            {
                "type": "user",
                "isSidechain": True,
                "message": {"content": "spawned subagent prompt"},
                "uuid": "side1",
                "sessionId": "sess-1",
                "cwd": "/home/user/services/synapse",
                "timestamp": "2026-05-02T10:00:20.000Z",
            },
        ]
        path = _write_jsonl(tmp_path, records)
        eps = JSONLParser().parse_file(path)
        assert len(eps) == 2  # sidechain ignored

    def test_empty_file(self, tmp_path: Path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        assert JSONLParser().parse_file(p) == []

    def test_malformed_lines_skipped(self, tmp_path: Path):
        p = tmp_path / "bad.jsonl"
        p.write_text(
            "not json\n"
            + json.dumps(
                {
                    "type": "user",
                    "message": {"content": "hello there"},
                    "uuid": "u1",
                    "sessionId": "sess-2",
                    "cwd": "/x",
                    "timestamp": "2026-05-02T10:00:00Z",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "assistant",
                    "message": {"model": "m", "content": [{"type": "text", "text": "hi back"}]},
                    "uuid": "a1",
                    "sessionId": "sess-2",
                    "timestamp": "2026-05-02T10:00:01Z",
                }
            )
            + "\n"
        )
        eps = JSONLParser().parse_file(p)
        assert len(eps) == 1
        assert eps[0].human_turn == "hello there"
