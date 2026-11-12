"""The line rules every transcript parser renders an Episode's ``content`` with.

An episode is one user-initiated exchange, rendered as lines joined by a blank line:

    [context] the previous turn's assistant text (orientation anchor)
    [user] the human message
    [tool:name] what a tool was called with
    [result] what a tool returned
    [assistant] the reply

Each parser decides which of its records become which line, in what order, and when a
group is too empty to keep; this module owns only the shapes, their caps and the
grouping loop, so a new parser renders the way the others do.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

CONTEXT_CAP = 300
USER_CAP = 3000
ASSISTANT_CAP = 3000
TOOL_CAP = 300
RESULT_CAP = 500
#: A tool result renders only when its text is longer than this.
RESULT_MIN_CHARS = 20
TITLE_CAP = 200
LINE_SEP = "\n\n"


def strip_nulls(text: str) -> str:
    """Postgres TEXT columns reject NUL bytes — strip them defensively."""
    return text.replace("\x00", "") if "\x00" in text else text


def context_line(prev_assistant: str) -> str:
    """The previous turn's assistant text, stripped and capped, as this turn's anchor."""
    return f"[context] {prev_assistant.strip()[:CONTEXT_CAP]}"


def user_line(text: str) -> str:
    """The human message; callers store ``text[:USER_CAP]`` as the human turn."""
    return f"[user] {text[:USER_CAP]}"


def assistant_line(text: str) -> str:
    """The assistant reply, capped; callers store the full text as the assistant turn."""
    return f"[assistant] {text[:ASSISTANT_CAP]}"


def tool_line(name: object, detail: str) -> str:
    """A tool call. Whether an empty ``detail`` renders at all is the caller's rule."""
    return f"[tool:{name}] {detail[:TOOL_CAP]}"


def result_line(text: str) -> str | None:
    """A tool result, or None when the text is too short to be worth a line."""
    if len(text) <= RESULT_MIN_CHARS:
        return None
    return f"[result] {text[:RESULT_CAP]}"


def title_line(title: str) -> str:
    """A conversation title, for an export's first turn when it has no [context]."""
    return f"[title] {title[:TITLE_CAP]}"


def join_lines(lines: Iterable[str]) -> str:
    """An episode's content: its lines joined by a blank line, NULs stripped."""
    return strip_nulls(LINE_SEP.join(lines))


def group_turns[T](items: Iterable[T], starts_turn: Callable[[T], bool]) -> list[list[T]]:
    """Split ``items`` into exchanges: an item for which ``starts_turn`` is true opens a
    new group, unless nothing precedes it. Leading items before the first such item form
    a group of their own (a batch that begins mid-turn)."""
    groups: list[list[T]] = []
    current: list[T] = []
    for item in items:
        if starts_turn(item) and current:
            groups.append(current)
            current = [item]
        else:
            current.append(item)
    if current:
        groups.append(current)
    return groups


def repeated_span(seen: set[str], span_id: str | None) -> bool:
    """True if ``span_id`` was already emitted; otherwise record it and return False.

    A turn without a span id is never a repeat. Transcripts physically repeat records
    across compaction, resume and fork, and a repeated span id is the same turn."""
    if span_id is None:
        return False
    if span_id in seen:
        return True
    seen.add(span_id)
    return False
