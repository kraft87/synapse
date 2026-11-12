"""When a harness last compacted the conversation, read from its own transcript.

Serving leaves out the calling session's own turns because the model still holds them.
A compaction replaces the turns before it with a summary, so from then on those turns
are memory like any other: the PreToolUse hooks send the compaction time as
``self_since`` and the server hides only the session's turns from that time on. The
transcript stamps each turn with the same clock the server stores it under, so the
time is returned exactly as written.

Transcripts are appended to while the session runs and hold lines of many megabytes, so
each scan maps the file read-only and searches backward from the end for the marker; a
hit that is not the harness's own record (the marker inside a tool's output) is skipped
and the search goes on before it. Every failure (no file, an empty one, no marker,
anything unreadable) is None: no cut, and the whole session stays hidden as before.
"""

from __future__ import annotations

import json
import mmap
import os
from collections.abc import Callable
from typing import Any

_CLAUDE_MARKER = b'"compact_boundary"'
_CODEX_MARKER = b'"compacted"'

Accept = Callable[[dict[str, Any]], bool]


def _last(buf: mmap.mmap, marker: bytes, accept: Accept, end: int) -> tuple[dict[str, Any], int]:
    """The last line ending before ``end`` that holds ``marker`` and parses to a JSON
    object ``accept`` takes, with the offset that line starts at; ``({}, -1)`` if none."""
    while True:
        hit = buf.rfind(marker, 0, end)
        if hit < 0:
            return {}, -1
        start = buf.rfind(b"\n", 0, hit) + 1
        stop = buf.find(b"\n", hit)
        try:
            row = json.loads(buf[start : stop if stop >= 0 else len(buf)])
        except ValueError:
            row = None
        if isinstance(row, dict) and accept(row):
            return row, start
        end = start


def _stamp(row: dict[str, Any]) -> str | None:
    value = row.get("timestamp")
    return value if isinstance(value, str) and value else None


def _is_boundary(row: dict[str, Any]) -> bool:
    return row.get("type") == "system" and row.get("subtype") == "compact_boundary"


def _head_uuid(boundary: dict[str, Any]) -> str | None:
    meta = boundary.get("compactMetadata")
    segment = meta.get("preservedSegment") if isinstance(meta, dict) else None
    head = segment.get("headUuid") if isinstance(segment, dict) else None
    return head if isinstance(head, str) and head else None


def claude_code_cut(transcript_path: str | os.PathLike[str]) -> str | None:
    """The time of the last compaction in a Claude Code transcript, or None.

    The last ``compact_boundary`` system line marks it. When its preserved segment names
    a head line (an earlier line whose ``uuid`` is the boundary's ``headUuid``), the
    harness kept the turns from that head on verbatim, so the cut is the head's
    ``timestamp``; otherwise it is the boundary's own."""
    try:
        with (
            open(transcript_path, "rb") as fh,
            mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as buf,
        ):
            boundary, start = _last(buf, _CLAUDE_MARKER, _is_boundary, len(buf))
            if start < 0:
                return None
            head = _head_uuid(boundary)
            if head:
                needle = json.dumps(head).encode()
                line, at = _last(buf, needle, lambda row: row.get("uuid") == head, start)
                if at >= 0 and (stamp := _stamp(line)):
                    return stamp
            return _stamp(boundary)
    except Exception:  # an unreadable transcript is no cut, never a failed tool call
        return None


def codex_cut(rollout_path: str | os.PathLike[str]) -> str | None:
    """The time of the last compaction in a Codex rollout file, or None: the
    ``timestamp`` of the last line whose type is ``compacted``."""
    try:
        with (
            open(rollout_path, "rb") as fh,
            mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as buf,
        ):
            row, start = _last(buf, _CODEX_MARKER, lambda r: r.get("type") == "compacted", len(buf))
            return _stamp(row) if start >= 0 else None
    except Exception:  # an unreadable rollout is no cut, never a failed tool call
        return None
