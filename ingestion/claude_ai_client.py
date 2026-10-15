"""Parse Claude.ai conversation exports into Episodes.

The export is a single conversations.json with the shape:
    [
      {
        "uuid": "...", "name": "...", "summary": "...",
        "created_at": "...", "updated_at": "...",
        "chat_messages": [
          {
            "uuid": "...", "text": "...",
            "content": [{"type": "text", "text": "...", ...}],
            "sender": "human" | "assistant",
            "created_at": "...", "attachments": [...], "files": [...]
          },
          ...
        ]
      },
      ...
    ]

We pair adjacent human/assistant turns into Episodes that match the schema
produced by the JSONL/Logfire/Cursor paths. No tool-use info exists in the
export — claude.ai didn't capture it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import orjson

from ingestion.models import Episode
from ingestion.turns import TurnSequence, parse_iso

logger = logging.getLogger(__name__)

#: claude.ai turns are long-form chat, so they keep more text than a coding turn.
_TEXT_CHARS = 5000


def _msg_text(msg: dict[str, Any]) -> str:
    """Pull the message text — prefer content[] blocks, fall back to top-level .text."""
    pieces: list[str] = []
    content = msg.get("content")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                t = str(block.get("text") or "").strip()
                if t:
                    pieces.append(t)
    if pieces:
        return "\n\n".join(pieces)
    return str(msg.get("text") or "").strip()


def _attachments_summary(msg: dict[str, Any]) -> str | None:
    """Summarize attachments (claude.ai exports omit content but keep metadata)."""
    parts: list[str] = []
    for att in msg.get("attachments") or []:
        if isinstance(att, dict):
            name = att.get("file_name") or att.get("name")
            if name:
                parts.append(f"file:{name}")
    for f in msg.get("files") or []:
        if isinstance(f, dict):
            name = f.get("file_name") or f.get("name")
            if name:
                parts.append(f"file:{name}")
    return " ".join(parts) if parts else None


class ClaudeAIParser:
    """Convert one claude.ai conversation dict into a list of Episodes."""

    @staticmethod
    def parse_conversation(convo: dict[str, Any]) -> list[Episode]:
        session_id = convo.get("uuid")
        if not session_id:
            return []

        msgs = convo.get("chat_messages") or []
        if not msgs:
            return []

        # Sort defensively by created_at to handle out-of-order exports
        msgs = sorted(msgs, key=lambda m: m.get("created_at") or "")

        # Group messages into turns. A new turn starts at each "human" with text.
        groups: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for m in msgs:
            sender = m.get("sender")
            has_text = bool(_msg_text(m))
            if sender == "human" and has_text and current:
                groups.append(current)
                current = [m]
            else:
                current.append(m)
        if current:
            groups.append(current)

        name = (convo.get("name") or "").strip()
        episodes: list[Episode] = []
        turns = TurnSequence(text_chars=_TEXT_CHARS)

        for group in groups:
            if not group:
                continue
            turn = turns.begin(title=name)
            ts: str | None = None
            last_msg_uuid: str | None = None

            for m in group:
                ts = ts or m.get("created_at")
                last_msg_uuid = m.get("uuid", last_msg_uuid)
                sender = m.get("sender")
                text = _msg_text(m)
                attachments = _attachments_summary(m)

                if sender == "human":
                    turn.user(text)
                    if attachments:
                        turn.line(f"[attachments] {attachments}")
                elif sender == "assistant":
                    turn.assistant_text(text)

            span_id = f"claude_ai:{session_id}:{last_msg_uuid}" if last_msg_uuid else None
            # A group of only attachments (before the first prompt) is not a turn.
            if not (turn.human or turn.assistant) or not turns.accept(turn, span_id):
                continue

            episodes.append(
                Episode(
                    session_id=session_id,
                    sequence=turn.sequence,
                    project=None,
                    platform="claude_ai",
                    model=None,
                    human_turn=turn.human,
                    assistant_turn=turn.assistant,
                    content=turn.content,
                    span_id=span_id,
                    source="claude_ai",
                    metadata={
                        "name": name or None,
                        "ts": ts,
                    },
                    created_at=parse_iso(ts),
                )
            )

        return episodes


def parse_export(path: Path) -> list[Episode]:
    """Load conversations.json and parse all conversations."""
    with open(path, "rb") as f:
        data = orjson.loads(f.read())
    if not isinstance(data, list):
        raise ValueError(f"Expected list at {path}, got {type(data).__name__}")
    episodes: list[Episode] = []
    for convo in data:
        if isinstance(convo, dict):
            episodes.extend(ClaudeAIParser.parse_conversation(convo))
    return episodes
