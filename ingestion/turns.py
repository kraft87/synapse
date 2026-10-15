"""Turn assembly shared by every transcript parser.

Claude Code, Codex, Cursor, claude.ai and ChatGPT transcripts differ in format but become the
same Episode: one per user-turn exchange, its ``content`` a run of tagged lines
(``[context]``, ``[user]``, ``[assistant]``, ``[result]`` ...) under one set of truncation
limits. This module owns that layout, the rule for an empty turn, span_id dedup and timestamp
parsing, so a parser only walks its own format and the sources cannot drift apart.
"""

from __future__ import annotations

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
