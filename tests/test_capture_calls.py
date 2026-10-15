"""ingestion.capture.calls: tool_use / tool_result pairs out of Claude Code and Codex records."""

from __future__ import annotations

from typing import Any

from ingestion.capture import episode_tool_calls, sidechain_tool_calls
from ingestion.codex_client import CodexRolloutParser
from ingestion.jsonl_client import JSONLParser


def _rec(kind: str, uuid: str, content: Any, **kw) -> dict:
    base = {"sessionId": "S1", "cwd": "/work/app", "timestamp": "2026-09-01T12:00:00Z"}
    return {**base, "type": kind, "uuid": uuid, "message": {"role": kind, "content": content}, **kw}


def _use(uuid: str, tid: str, name: str, inp: dict, **kw) -> dict:
    return _rec(
        "assistant", uuid, [{"type": "tool_use", "id": tid, "name": name, "input": inp}], **kw
    )


def _result(uuid: str, tid: str, content: Any, **kw) -> dict:
    return _rec(
        "user", uuid, [{"type": "tool_result", "tool_use_id": tid, "content": content}], **kw
    )


def _turn() -> list[dict]:
    return [
        _rec("user", "u1", "research voyage"),
        _use("a1", "toolu_1", "WebFetch", {"url": "https://example.com"}),
        _result(
            "r1", "toolu_1", [{"type": "text", "text": "page "}, {"type": "text", "text": "text"}]
        ),
        _use("a2", "toolu_2", "WebSearch", {"query": "q"}, timestamp="2026-09-01T12:00:05Z"),
        _result("r2", "toolu_2", "Links: []", **{"toolUseResult": {}}),
        _use("a3", "toolu_unanswered", "WebFetch", {"url": "https://example.com/x"}),
        _rec("assistant", "a4", [{"type": "text", "text": "done"}]),
    ]


def test_a_stored_turn_yields_its_paired_calls_stamped_with_the_turn():
    (ep,) = JSONLParser().parse_records(_turn(), "hook", "proj", with_records=True)
    ep.surface_id = "surf-1"
    calls = episode_tool_calls(ep)
    assert [c.tool_use_id for c in calls] == ["toolu_1", "toolu_2"]  # the unanswered one is not
    first = calls[0]
    assert (first.tool_name, first.tool_input) == ("WebFetch", {"url": "https://example.com"})
    assert first.result_text == "page \ntext"
    assert (first.session_id, first.project, first.span_id, first.surface_id) == (
        "S1",
        "proj",
        ep.span_id,
        "surf-1",
    )
    assert first.timestamp is not None and first.timestamp.year == 2026
    assert calls[1].timestamp.second == 5


def test_a_turn_without_a_project_takes_the_default():
    (ep,) = JSONLParser().parse_records(_turn(), "hook", None, with_records=True)
    assert {c.project for c in episode_tool_calls(ep)} == {"app"}  # from the cwd
    ep = ep.model_copy(update={"project": None})
    assert {c.project for c in episode_tool_calls(ep, "fallback")} == {"fallback"}


def test_the_error_flag_travels():
    recs = [
        _rec("user", "u1", "go"),
        _use("a1", "toolu_1", "WebFetch", {"url": "https://example.com"}),
        _rec(
            "user",
            "r1",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "boom",
                    "is_error": True,
                }
            ],
        ),
    ]
    (ep,) = JSONLParser().parse_records(recs, "hook", "p", with_records=True)
    assert [c.is_error for c in episode_tool_calls(ep)] == [True]


def test_an_episode_from_another_platform_has_no_calls():
    (ep,) = JSONLParser().parse_records(_turn(), "hook", "p", with_records=True)
    assert episode_tool_calls(ep.model_copy(update={"platform": "claude_ai"})) == []


def test_sidechain_records_are_read_though_the_parser_drops_them():
    side = [
        _use("s1", "toolu_s", "WebFetch", {"url": "https://example.com"}, isSidechain=True),
        _result("s2", "toolu_s", "body", isSidechain=True),
    ]
    assert JSONLParser().parse_records(side, "hook", "p") == []
    calls = sidechain_tool_calls(side + _turn(), lambda cwd: f"label-of-{cwd}", "surf-1")
    assert [c.tool_use_id for c in calls] == ["toolu_s"]  # the main thread's calls are not these
    (call,) = calls
    assert (call.session_id, call.project, call.surface_id, call.span_id) == (
        "S1",
        "label-of-/work/app",
        "surf-1",
        None,
    )


def test_codex_function_and_shell_calls_pair_by_call_id():
    def item(ptype: str, payload: dict, ts: str = "2026-09-01T12:00:00Z") -> dict:
        return {"timestamp": ts, "type": "response_item", "payload": {"type": ptype, **payload}}

    records = [
        {
            "timestamp": "2026-09-01T11:59:00Z",
            "type": "session_meta",
            "payload": {"id": "C1", "cwd": "/w/app"},
        },
        item(
            "message", {"role": "user", "content": [{"type": "input_text", "text": "look it up"}]}
        ),
        item(
            "function_call",
            {"name": "web__fetch", "arguments": '{"url": "https://example.com"}', "call_id": "c1"},
        ),
        item("function_call_output", {"call_id": "c1", "output": "the page"}),
        item(
            "local_shell_call",
            {"action": {"command": ["curl", "https://example.com"]}, "call_id": "c2"},
        ),
        item(
            "function_call_output",
            {"call_id": "c2", "output": [{"type": "output_text", "text": "out"}]},
        ),
        item(
            "custom_tool_call", {"name": "apply_patch", "input": "*** Begin Patch", "call_id": "c3"}
        ),
        item(
            "message", {"role": "assistant", "content": [{"type": "output_text", "text": "done"}]}
        ),
    ]
    (ep,) = CodexRolloutParser().parse_records(
        records, "hook", "proj", session_id_hint="C1", with_records=True
    )
    calls = episode_tool_calls(ep)
    assert [(c.tool_use_id, c.tool_name) for c in calls] == [("c1", "web__fetch"), ("c2", "shell")]
    assert calls[0].tool_input == {"url": "https://example.com"}
    assert calls[1].tool_input == {"command": ["curl", "https://example.com"]}
    assert calls[1].result_text == "out"
    assert (calls[0].session_id, calls[0].project) == ("C1", "proj")


def test_a_parse_keeps_no_records_unless_asked():
    """A whole-file sweep must not hold every record of the run through its episodes."""
    (ep,) = JSONLParser().parse_records(_turn(), "hook", "p")
    assert ep.records == [] and episode_tool_calls(ep) == []
