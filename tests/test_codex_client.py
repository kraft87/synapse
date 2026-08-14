"""Tests for the Codex CLI rollout parser (pure unit tests — no I/O beyond tmp files)."""

from __future__ import annotations

import json
from pathlib import Path

from ingestion.codex_client import (
    CodexRolloutParser,
    _is_user_turn,
    session_id_from_path,
)

SESSION_ID = "019ffdbe-ff9a-7b41-892a-ddc6221d1a64"
ROLLOUT_NAME = f"rollout-2026-08-13T20-49-40-{SESSION_ID}.jsonl"


def _rec(rtype: str, payload: dict, ts: str = "2026-08-14T00:50:07.295Z") -> dict:
    return {"timestamp": ts, "ordinal": 0, "type": rtype, "payload": payload}


def _msg(role: str, text: str, msg_id: str, ts: str = "2026-08-14T00:50:07.295Z") -> dict:
    block_type = "output_text" if role == "assistant" else "input_text"
    return _rec(
        "response_item",
        {
            "type": "message",
            "id": msg_id,
            "role": role,
            "content": [{"type": block_type, "text": text}],
        },
        ts,
    )


def _session_meta(session_id: str = SESSION_ID, cwd: str = "/home/user/dev/myproj") -> dict:
    return _rec(
        "session_meta",
        {
            "session_id": session_id,
            "cwd": cwd,
            "originator": "codex-tui",
            "cli_version": "0.147.0",
        },
    )


def _basic_session() -> list[dict]:
    return [
        _session_meta(),
        _rec("event_msg", {"type": "task_started"}),  # skip
        _msg("developer", "<skills_instructions>...</skills_instructions>", "msg_dev1"),  # skip
        _msg("user", "<environment_context>\n  <cwd>/home/user</cwd>", "msg_env"),  # skip
        _msg("user", "hi", "msg_u1", ts="2026-08-14T00:50:07.295Z"),
        _msg("assistant", "Hello! How can I help?", "msg_a1"),
        _rec("event_msg", {"type": "token_count"}),  # skip
        _rec("turn_context", {"turn_id": "t2", "cwd": "/home/user/dev/myproj"}),
        _msg("user", "run the tests", "msg_u2", ts="2026-08-14T00:51:00.000Z"),
        _rec(
            "response_item",
            {"type": "reasoning", "id": "rs_1", "encrypted_content": "gAAAA..."},
        ),  # skip — ciphertext
        _rec(
            "response_item",
            {
                "type": "custom_tool_call",
                "id": "ctc_1",
                "call_id": "call_1",
                "name": "exec",
                "input": "pytest -q",
            },
        ),
        _rec(
            "response_item",
            {
                "type": "custom_tool_call_output",
                "id": "ctco_1",
                "call_id": "call_1",
                "output": [{"type": "input_text", "text": "47 passed in 3.21s — all green"}],
            },
        ),
        _msg("assistant", "All 47 tests pass.", "msg_a2"),
        _rec("event_msg", {"type": "task_complete"}),  # skip
    ]


def _write_rollout(tmp_path: Path, records: list[dict]) -> Path:
    p = tmp_path / ROLLOUT_NAME
    with open(p, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p


class TestIsUserTurn:
    def test_real_user_message(self):
        assert _is_user_turn(_msg("user", "hello", "m1"))

    def test_environment_context_is_machinery(self):
        assert not _is_user_turn(_msg("user", "<environment_context>x", "m1"))

    def test_assistant_is_not_user_turn(self):
        assert not _is_user_turn(_msg("assistant", "hello", "m1"))

    def test_developer_is_not_user_turn(self):
        assert not _is_user_turn(_msg("developer", "hello", "m1"))

    def test_tool_call_is_not_user_turn(self):
        assert not _is_user_turn(
            _rec("response_item", {"type": "custom_tool_call", "id": "c", "input": "x"})
        )


class TestSessionIdFromPath:
    def test_extracts_uuid(self):
        assert session_id_from_path(f"/x/y/{ROLLOUT_NAME}") == SESSION_ID

    def test_non_rollout_none(self):
        assert session_id_from_path("/x/y/session.jsonl") is None


class TestParseFile:
    def test_two_turns(self, tmp_path):
        path = _write_rollout(tmp_path, _basic_session())
        eps = CodexRolloutParser().parse_file(path)
        assert len(eps) == 2

        first, second = eps
        assert first.session_id == SESSION_ID
        assert first.sequence == 1
        assert first.human_turn == "hi"
        assert first.assistant_turn == "Hello! How can I help?"
        assert first.span_id == "codex:msg_a1"
        assert first.source == "codex"
        assert first.platform == "codex"
        assert first.project == "myproj"
        assert first.created_at is not None
        assert first.created_at.isoformat().startswith("2026-08-14T00:50:07")

        assert second.human_turn == "run the tests"
        assert second.span_id == "codex:msg_a2"
        assert "[tool:exec] pytest -q" in second.content
        assert "[result] 47 passed" in second.content
        assert second.content.startswith("[context] Hello! How can I help?")
        assert second.created_at.isoformat().startswith("2026-08-14T00:51:00")

    def test_developer_and_env_context_filtered(self, tmp_path):
        path = _write_rollout(tmp_path, _basic_session())
        eps = CodexRolloutParser().parse_file(path)
        joined = "\n".join(e.content for e in eps)
        assert "skills_instructions" not in joined
        assert "environment_context" not in joined

    def test_reasoning_ciphertext_excluded(self, tmp_path):
        path = _write_rollout(tmp_path, _basic_session())
        eps = CodexRolloutParser().parse_file(path)
        assert "gAAAA" not in "\n".join(e.content for e in eps)

    def test_session_id_falls_back_to_filename(self, tmp_path):
        records = _basic_session()[1:]  # drop session_meta
        path = _write_rollout(tmp_path, records)
        eps = CodexRolloutParser().parse_file(path)
        assert len(eps) == 2
        assert all(e.session_id == SESSION_ID for e in eps)

    def test_no_session_id_yields_nothing(self, tmp_path):
        records = _basic_session()[1:]
        p = tmp_path / "session.jsonl"  # no uuid in name, no session_meta
        with open(p, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        assert CodexRolloutParser().parse_file(p) == []

    def test_replayed_span_kept_once(self, tmp_path):
        records = _basic_session()
        # resume re-dumps the first turn's records verbatim
        records += [
            _msg("user", "hi", "msg_u1"),
            _msg("assistant", "Hello! How can I help?", "msg_a1"),
        ]
        path = _write_rollout(tmp_path, records)
        eps = CodexRolloutParser().parse_file(path)
        assert [e.span_id for e in eps] == ["codex:msg_a1", "codex:msg_a2"]

    def test_short_tool_output_filtered(self, tmp_path):
        records = [
            _session_meta(),
            _msg("user", "quick check", "msg_u1"),
            _rec(
                "response_item",
                {"type": "custom_tool_call", "id": "ctc_1", "name": "exec", "input": "true"},
            ),
            _rec(
                "response_item",
                {"type": "custom_tool_call_output", "id": "ctco_1", "output": "ok"},
            ),
            _msg("assistant", "Done.", "msg_a1"),
        ]
        path = _write_rollout(tmp_path, records)
        eps = CodexRolloutParser().parse_file(path)
        assert len(eps) == 1
        assert "[result]" not in eps[0].content


class TestParseRecordsSeam:
    """Push and sweep must converge on identical span_ids (the /ingest invariant)."""

    def test_file_and_records_agree(self, tmp_path):
        records = _basic_session()
        path = _write_rollout(tmp_path, records)
        parser = CodexRolloutParser()
        from_file = parser.parse_file(path)
        from_records = parser.parse_records(records, "push", session_id_hint=None)
        assert [e.span_id for e in from_file] == [e.span_id for e in from_records]
        assert [e.session_id for e in from_file] == [e.session_id for e in from_records]

    def test_tail_without_meta_needs_hint(self):
        tail = _basic_session()[8:]  # second turn only, no session_meta
        parser = CodexRolloutParser()
        assert parser.parse_records(tail, "push") == []
        eps = parser.parse_records(tail, "push", session_id_hint=SESSION_ID)
        assert len(eps) == 1
        assert eps[0].session_id == SESSION_ID
        assert eps[0].span_id == "codex:msg_a2"
