"""Turn assembly shared by every transcript parser.

Claude Code, Codex, Cursor, claude.ai and ChatGPT transcripts differ in format but become the
same Episode: one per user-turn exchange, its ``content`` a run of tagged lines
(``[context]``, ``[user]``, ``[assistant]``, ``[result]`` ...) under one set of truncation
limits. This module owns that layout, the rule for an empty turn, span_id dedup and timestamp
parsing, so a parser only walks its own format and the sources cannot drift apart.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

#: The previous turn's assistant text carried into the next turn as ``[context]``.
CONTEXT_CHARS = 300
#: ``[user]`` / ``[assistant]`` text kept per turn (claude.ai chat turns run longer).
TEXT_CHARS = 3000
#: ``[result]`` (tool output) kept per turn, and the length at or under which it is noise.
RESULT_CHARS = 500
MIN_RESULT_CHARS = 20


def parse_iso(ts: Any) -> datetime | None:
    """An ISO-8601 timestamp (``Z`` suffix allowed) as a datetime; None when absent or malformed."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


class Turn:
    """One exchange being assembled: the tagged lines plus the user and assistant text."""

    def __init__(self, sequence: int, lead: list[str], text_chars: int) -> None:
        self.sequence = sequence
        self.human: str | None = None
        self.assistant: str | None = None
        self._parts = lead
        self._lead = len(lead)
        self._text_chars = text_chars

    def user(self, text: str | None) -> None:
        """The turn's human text: the first one only; later user records are tool output."""
        if text and self.human is None:
            self.human = text[: self._text_chars]
            self._parts.append(f"[user] {self.human}")

    def assistant_text(self, text: str | None) -> None:
        """An assistant reply; the last one in the turn is the turn's ``assistant``."""
        if text:
            self.assistant = text
            self._parts.append(f"[assistant] {text[: self._text_chars]}")

    def result(self, text: str) -> None:
        """Tool output, unless too short to carry information."""
        if len(text) > MIN_RESULT_CHARS:
            self._parts.append(f"[result] {text[:RESULT_CHARS]}")

    def line(self, line: str) -> None:
        """A source-specific line (``[tool:...]``, ``[attachments]``), already formatted."""
        self._parts.append(line)

    @property
    def empty(self) -> bool:
        """True when nothing but the ``[context]`` / ``[title]`` lead was added: a hook
        bounce with no reply, or items that carried no content."""
        return len(self._parts) == self._lead

    @property
    def content(self) -> str:
        return "\n\n".join(self._parts)


class TurnSequence:
    """Numbers one conversation's turns and carries state between them: the previous
    reply (the next turn's ``[context]``) and the span_ids already emitted."""

    def __init__(self, text_chars: int = TEXT_CHARS) -> None:
        self._text_chars = text_chars
        self._seq = 0
        self._prev_assistant: str | None = None
        self._seen_span_ids: set[str] = set()

    def begin(self, title: str | None = None) -> Turn:
        """Start the next turn. The first one leads with ``[title]`` when the conversation
        has a name; later ones with ``[context]`` when the last emitted turn had a reply."""
        self._seq += 1
        lead: list[str] = []
        if self._prev_assistant:
            lead.append(f"[context] {self._prev_assistant.strip()[:CONTEXT_CHARS]}")
        elif self._seq == 1 and title:
            lead.append(f"[title] {title[:200]}")
        return Turn(self._seq, lead, self._text_chars)

    def accept(self, turn: Turn, span_id: str | None) -> bool:
        """Whether ``turn`` becomes an Episode: not empty, and not a repeat of a span_id
        already emitted (transcripts physically repeat records across compaction and
        resume, and a repeated uuid IS the same turn, which would collide on the unique
        span_id index). Accepting records it for the next turn's ``[context]``."""
        if turn.empty:
            return False
        if span_id is not None:
            if span_id in self._seen_span_ids:
                return False
            self._seen_span_ids.add(span_id)
        self._prev_assistant = turn.assistant
        return True


# ---------------------------------------------------------------------------
# Reading a stored turn back
# ---------------------------------------------------------------------------

#: A tagged line starts a block; any other line belongs to the block above it. Only the
#: tags this module and the parsers write count, so a user's own "[x] done" stays text.
_BLOCK_START = re.compile(
    r"^(?=\[(?:context|title|user|assistant|result|attachments|tool:[^\]\n]*)\] )", re.M
)
#: Tool-call lines the per-turn gates read from the end of a long turn.
_TAIL_TOOL_LINES = 40


def _blocks(content: str) -> list[tuple[str, str]]:
    """``(tag, text)`` for each tagged block of a stored turn, in order. Text before the
    first tag (a turn with no tags at all) comes back under the empty tag."""
    out: list[tuple[str, str]] = []
    for part in _BLOCK_START.split(content):
        if part.strip():
            m = re.match(r"\[([^\]\n]*)\] ", part)
            out.append((m.group(1).split(":")[0] if m else "", part.strip()))
    return out


def user_text(content: str) -> str:
    """The turn's ``[user]`` text, without its tag; empty when it has none."""
    return " ".join(text[len("[user] ") :] for tag, text in _blocks(content) if tag == "user")


def compact_turn(content: str, limit: int = TEXT_CHARS * 2) -> str:
    """The turn as a per-turn gate should read it: what was asked and how the turn ended.

    ``content`` is 75-90% tool trace on a long turn, and cutting its head to a fixed size
    kept the trace and lost the ending. This keeps the lead (``[context]``, ``[title]``),
    the ``[user]`` text and the LAST ``[assistant]`` text whole, then as many of the last
    tool-call lines as fit in ``limit`` (at most ``_TAIL_TOOL_LINES``) with no ``[result]``
    bodies; an earlier assistant message or tool line is the part that goes. A turn with no
    tags is cut to ``limit``. ``episodes.content`` itself is not changed: the skills lane
    mines its tool lines.
    """
    blocks = _blocks(content)
    if not any(tag for tag, _ in blocks):
        return content[:limit]
    final = max((i for i, (tag, _) in enumerate(blocks) if tag == "assistant"), default=-1)
    kept = [
        (i, text)
        for i, (tag, text) in enumerate(blocks)
        if tag in ("context", "title", "user", "attachments") or i == final
    ]
    budget = limit - sum(len(text) for _, text in kept)
    tools: list[tuple[int, str]] = []
    for i, (tag, text) in reversed(list(enumerate(blocks))):
        if tag != "tool":
            continue
        if len(tools) == _TAIL_TOOL_LINES or len(text) > budget:
            break
        tools.append((i, text))
        budget -= len(text)
    return "\n\n".join(text for _, text in sorted(kept + tools))
