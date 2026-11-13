"""The compaction scans (plugin/synapse_client/compaction.py) on transcripts built here.

``claude_code_cut`` reads a Claude Code transcript, ``codex_cut`` a Codex rollout file;
each returns the time of the last compaction exactly as the file wrote it, or None. The
scans search backward from the end of the file, so the cases that matter are the ones a
backward search can get wrong: a marker inside a tool's output, a later marker that is
not a real one, a head uuid that also appears on other lines, a very long line, a line
still being written.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.helpers.client_core import use_core

use_core()
from synapse_client.compaction import claude_code_cut, codex_cut  # noqa: E402

_HEAD = "0b7c2a4e-1111-4c22-9d33-000000000002"


def _jsonl(path: Path, rows: list[dict[str, Any] | str], tail: str = "\n") -> Path:
    """Write ``rows`` one per line as Claude Code and Codex do (compact JSON); a str row
    is written as it is."""
    lines = [r if isinstance(r, str) else json.dumps(r, separators=(",", ":")) for r in rows]
    path.write_text("\n".join(lines) + tail, encoding="utf-8")
    return path


def _turn(uuid: str, ts: str | None, parent: str | None = None, **extra: Any) -> dict[str, Any]:
    row = {"type": "user", "uuid": uuid, "parentUuid": parent, "message": {"content": "hi"}}
    if ts is not None:
        row["timestamp"] = ts
    return {**row, **extra}


def _boundary(ts: str, head: str | None = None) -> dict[str, Any]:
    meta: dict[str, Any] = {"trigger": "auto", "preTokens": 150000}
    if head is not None:
        meta["preservedSegment"] = {"headUuid": head, "anchorUuid": "a-1", "tailUuid": "t-1"}
    return {
        "type": "system",
        "subtype": "compact_boundary",
        "uuid": f"b-{ts}",
        "timestamp": ts,
        "content": "Conversation compacted",
        "compactMetadata": meta,
    }


# A tool's output that carries the marker: as a nested object it matches the marker
# bytes exactly, which is the case the scan has to reject by the line's own type.
_TOOL_OUTPUT = _turn(
    "u-tool",
    "2026-11-12T15:00:00.000Z",
    toolUseResult={"type": "system", "subtype": "compact_boundary", "timestamp": "nope"},
)
_TOOL_OUTPUT_TEXT = _turn(
    "u-text", "2026-11-12T15:00:01.000Z", toolUseResult={"stdout": '"compact_boundary"'}
)

# ---------------------------------------------------------------------------
# claude_code_cut
# ---------------------------------------------------------------------------


def test_the_cut_is_the_preserved_head_written_as_is(tmp_path):
    """The harness keeps the turns from the head on verbatim, so the cut is the head's
    time. Its uuid also sits in a later line's parentUuid, in a tool output, and in the
    boundary itself: only the line whose own uuid it is counts."""
    path = _jsonl(
        tmp_path / "t.jsonl",
        [
            _turn("u-1", "2026-11-12T14:00:00.000Z"),
            _turn(_HEAD, "2026-11-12T14:10:54.738Z", parent="u-1"),
            _turn("u-3", "2026-11-12T14:20:00.000Z", parent=_HEAD),
            _turn("u-4", "2026-11-12T14:25:00.000Z", toolUseResult={"ref": _HEAD}),
            _boundary("2026-11-12T14:30:00.000Z", head=_HEAD),
            _turn("u-5", "2026-11-12T14:31:00.000Z", parent=_HEAD),
        ],
    )
    assert claude_code_cut(path) == "2026-11-12T14:10:54.738Z"
    assert claude_code_cut(str(path)) == "2026-11-12T14:10:54.738Z"


def test_a_boundary_without_a_preserved_segment_is_its_own_cut(tmp_path):
    path = _jsonl(
        tmp_path / "t.jsonl",
        [_turn("u-1", "2026-11-12T14:00:00Z"), _boundary("2026-11-12T14:30:00.5Z")],
    )
    assert claude_code_cut(path) == "2026-11-12T14:30:00.5Z"


_B = _boundary("2026-11-12T14:30:00Z", head=_HEAD)


@pytest.mark.parametrize(
    "rows",
    [
        [_turn("u-1", "2026-11-12T14:00:00Z"), _B],  # the head uuid is on no line
        [_turn("u-1", "2026-11-12T14:00:00Z", parent=_HEAD), _B],  # only as a parent
        [_turn(_HEAD, None), _B],  # the head has no time
        [_turn("u-1", "2026-11-12T14:00:00Z"), _B, _turn(_HEAD, "2026-11-12T14:32:00Z")],
    ],
    ids=["missing", "parent-only", "no-timestamp", "only-after-the-boundary"],
)
def test_a_head_that_cannot_be_placed_falls_back_to_the_boundary(tmp_path, rows):
    assert claude_code_cut(_jsonl(tmp_path / "t.jsonl", rows)) == "2026-11-12T14:30:00Z"


def test_the_last_boundary_wins(tmp_path):
    first = "0b7c2a4e-1111-4c22-9d33-000000000001"
    path = _jsonl(
        tmp_path / "t.jsonl",
        [
            _turn(first, "2026-11-12T13:00:00Z"),
            _boundary("2026-11-12T13:30:00Z", head=first),
            _turn(_HEAD, "2026-11-12T14:10:00Z", parent=first),
            _boundary("2026-11-12T14:30:00Z", head=_HEAD),
            _turn("u-9", "2026-11-12T14:40:00Z"),
        ],
    )
    assert claude_code_cut(path) == "2026-11-12T14:10:00Z"


def test_a_marker_only_inside_tool_output_is_no_cut(tmp_path):
    path = _jsonl(
        tmp_path / "t.jsonl",
        [_turn("u-1", "2026-11-12T14:00:00Z"), _TOOL_OUTPUT, _TOOL_OUTPUT_TEXT],
    )
    assert claude_code_cut(path) is None


def test_a_later_marker_in_tool_output_does_not_hide_the_real_boundary(tmp_path):
    path = _jsonl(
        tmp_path / "t.jsonl",
        [
            _turn(_HEAD, "2026-11-12T14:10:00Z"),
            _boundary("2026-11-12T14:30:00Z", head=_HEAD),
            _TOOL_OUTPUT,
            _TOOL_OUTPUT_TEXT,
            _turn("u-9", "2026-11-12T15:10:00Z", toolUseResult={"ref": _HEAD}),
        ],
    )
    assert claude_code_cut(path) == "2026-11-12T14:10:00Z"


def test_lines_of_many_megabytes_are_read_fine(tmp_path):
    big = "x" * (6 * 1024 * 1024)
    path = _jsonl(
        tmp_path / "t.jsonl",
        [
            _turn("u-0", "2026-11-12T13:00:00Z", toolUseResult={"stdout": big}),
            _turn(_HEAD, "2026-11-12T14:10:00Z", toolUseResult={"stdout": big}),
            _turn("u-2", "2026-11-12T14:20:00Z", toolUseResult={"stdout": big}),
            _boundary("2026-11-12T14:30:00Z", head=_HEAD),
            _turn("u-3", "2026-11-12T14:40:00Z", toolUseResult={"stdout": big}),
        ],
    )
    assert path.stat().st_size > 24 * 1024 * 1024
    assert claude_code_cut(path) == "2026-11-12T14:10:00Z"


def test_a_line_still_being_written_is_skipped(tmp_path):
    """The harness appends while the hook reads: a half-written last line carrying the
    marker is not a boundary, and the complete one before it still is."""
    torn = json.dumps(_boundary("2026-11-12T16:00:00Z"), separators=(",", ":"))[:-20]
    path = _jsonl(
        tmp_path / "t.jsonl",
        [_turn(_HEAD, "2026-11-12T14:10:00Z"), _boundary("2026-11-12T14:30:00Z", head=_HEAD), torn],
        tail="",
    )
    assert claude_code_cut(path) == "2026-11-12T14:10:00Z"


def test_windows_line_endings_are_read_fine(tmp_path):
    path = tmp_path / "t.jsonl"
    rows = [_turn(_HEAD, "2026-11-12T14:10:00Z"), _boundary("2026-11-12T14:30:00Z", head=_HEAD)]
    path.write_bytes(b"".join(json.dumps(r).encode() + b"\r\n" for r in rows))
    assert claude_code_cut(path) == "2026-11-12T14:10:00Z"


@pytest.mark.parametrize("scan", [claude_code_cut, codex_cut], ids=["claude-code", "codex"])
def test_no_file_an_empty_one_or_garbage_is_no_cut(tmp_path, scan):
    empty = tmp_path / "empty.jsonl"
    empty.write_bytes(b"")
    junk = tmp_path / "junk.jsonl"
    junk.write_bytes(b'\x00\xff "compact_boundary" "compacted" {not json\n')
    assert scan(empty) is None
    assert scan(junk) is None
    assert scan(tmp_path / "missing.jsonl") is None
    assert scan(tmp_path) is None  # a directory
    assert scan("") is None


def test_a_session_that_never_compacted_is_no_cut(tmp_path):
    path = _jsonl(tmp_path / "t.jsonl", [_turn(_HEAD, "2026-11-12T14:10:00Z")])
    assert claude_code_cut(path) is None


# ---------------------------------------------------------------------------
# codex_cut
# ---------------------------------------------------------------------------


def _rollout(ts: str, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"timestamp": ts, "type": kind, "payload": payload}


def _message(ts: str, text: str) -> dict[str, Any]:
    content = [{"type": "output_text", "text": text}]
    return _rollout(
        ts, "response_item", {"type": "message", "role": "assistant", "content": content}
    )


def _compacted(ts: str) -> dict[str, Any]:
    return _rollout(ts, "compacted", {"message": "summary", "replacement_history": []})


def test_the_codex_cut_is_the_last_compacted_line_written_as_is(tmp_path):
    path = _jsonl(
        tmp_path / "rollout.jsonl",
        [
            _rollout("2026-11-12T13:00:00.000Z", "session_meta", {"id": "s-1"}),
            _message("2026-11-12T13:10:00.000Z", "hello"),
            _compacted("2026-11-12T13:30:00.123Z"),
            _message("2026-11-12T14:00:00.000Z", "again"),
            _compacted("2026-11-12T14:30:00.456Z"),
            _rollout("2026-11-12T14:30:00.500Z", "event_msg", {"type": "context_compacted"}),
            _message("2026-11-12T14:40:00.000Z", "after"),
        ],
    )
    assert codex_cut(path) == "2026-11-12T14:30:00.456Z"


def test_a_rollout_that_never_compacted_is_no_cut(tmp_path):
    path = _jsonl(
        tmp_path / "rollout.jsonl",
        [
            _rollout("2026-11-12T13:00:00Z", "session_meta", {"id": "s-1"}),
            _message("2026-11-12T13:10:00Z", "hello"),
        ],
    )
    assert codex_cut(path) is None


def test_a_codex_marker_inside_content_is_skipped(tmp_path):
    """A nested ``"type": "compacted"`` (tool output, a quoted record) matches the marker
    bytes; only a line that is itself of that type is a compaction."""
    nested = _rollout("2026-11-12T15:00:00Z", "response_item", {"type": "compacted"})
    path = _jsonl(
        tmp_path / "rollout.jsonl",
        [
            _message("2026-11-12T13:10:00Z", "hello"),
            _compacted("2026-11-12T14:30:00Z"),
            nested,
            _message("2026-11-12T15:10:00Z", '"compacted"'),
        ],
    )
    assert codex_cut(path) == "2026-11-12T14:30:00Z"
    only_nested = _jsonl(tmp_path / "nested.jsonl", [_message("2026-11-12T13:10:00Z", "x"), nested])
    assert codex_cut(only_nested) is None
