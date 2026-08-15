"""Parse ChatGPT (OpenAI) data exports into Episodes.

The export zip carries conversations as either a single ``conversations.json``
or, in newer exports, shards named ``conversations-NNN.json`` (listed under
``logical_files`` in ``export_manifest.json``). Each conversation is a
message TREE, not a flat list:

    {
      "conversation_id": "...", "id": "...", "title": "...",
      "create_time": 1712345678.9, "update_time": ...,
      "current_node": "<node-id>", "default_model_slug": "gpt-...",
      "is_do_not_remember": false,
      "mapping": {
        "<node-id>": {
          "id": "...", "parent": "<node-id>|null", "children": [...],
          "message": {                # null on the synthetic root
            "id": "...",
            "author": {"role": "user"|"assistant"|"system"|"tool"},
            "create_time": 1712345679.0,
            "content": {"content_type": "text", "parts": ["..."]},
            "metadata": {"model_slug": "...", ...}
          }
        }, ...
      }
    }

Edits and regenerations create sibling branches; the canonical transcript is
the ``current_node`` walked up its parent chain to the root, reversed. Only
that branch is ingested — abandoned branches are drafts the user replaced.

Content types: "text" and "multimodal_text" carry the conversation
("multimodal" parts that are dicts are image/audio pointers, rendered as
"[image]"); "thoughts" and "reasoning_recap" are reasoning artifacts and are
skipped, matching the Codex parser's encrypted-reasoning exclusion.

Conversations flagged ``is_do_not_remember`` (temporary chats) are skipped
entirely — the user told the vendor not to remember them; we extend the same
courtesy.

Per-message ``create_time`` epochs land as event-time ``created_at``, so
imported history carries real dates, not import day.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson

from ingestion.models import Episode

logger = logging.getLogger(__name__)

_CONTENT_TYPES = ("text", "multimodal_text")


def _strip_nulls(text: str) -> str:
    return text.replace("\x00", "") if "\x00" in text else text


def _epoch_to_dt(ts: Any) -> datetime | None:
    try:
        return datetime.fromtimestamp(float(ts), tz=UTC) if ts else None
    except (ValueError, TypeError, OSError):
        return None


def _msg_text(msg: dict[str, Any]) -> str:
    content = msg.get("content") or {}
    if content.get("content_type") not in _CONTENT_TYPES:
        return ""
    pieces: list[str] = []
    for part in content.get("parts") or []:
        if isinstance(part, str):
            if part.strip():
                pieces.append(part.strip())
        elif isinstance(part, dict):
            pieces.append("[image]")
    return "\n\n".join(pieces)


def _canonical_branch(convo: dict[str, Any]) -> list[dict[str, Any]]:
    """Walk current_node -> root via parents; return message dicts in order.

    Falls back to all message nodes sorted by create_time when the chain is
    missing or broken (defensive against export drift).
    """
    mapping = convo.get("mapping") or {}
    chain: list[dict[str, Any]] = []
    node_id = convo.get("current_node")
    seen: set[str] = set()
    while node_id and node_id in mapping and node_id not in seen:
        seen.add(node_id)
        node = mapping[node_id]
        msg = node.get("message")
        if isinstance(msg, dict):
            chain.append(msg)
        node_id = node.get("parent")
    if chain:
        return list(reversed(chain))
    msgs = [n.get("message") for n in mapping.values() if isinstance(n.get("message"), dict)]
    return sorted(msgs, key=lambda m: m.get("create_time") or 0)


class OpenAIChatParser:
    """Convert one ChatGPT-export conversation dict into a list of Episodes."""

    @staticmethod
    def parse_conversation(convo: dict[str, Any], source_label: str = "") -> list[Episode]:
        session_id = convo.get("conversation_id") or convo.get("id")
        if not session_id:
            return []
        if convo.get("is_do_not_remember"):
            return []

        msgs = []
        for m in _canonical_branch(convo):
            role = (m.get("author") or {}).get("role")
            if role not in ("user", "assistant"):
                continue  # system prompt, tool traffic
            if not _msg_text(m):
                continue  # thoughts / reasoning_recap / empty
            msgs.append(m)
        if not msgs:
            return []

        groups: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for m in msgs:
            if (m.get("author") or {}).get("role") == "user" and current:
                groups.append(current)
                current = [m]
            else:
                current.append(m)
        if current:
            groups.append(current)

        title = (convo.get("title") or "").strip()
        default_model = convo.get("default_model_slug")
        episodes: list[Episode] = []
        prev_assistant: str | None = None
        seq = 0
        seen_span_ids: set[str] = set()

        for group in groups:
            if not group:
                continue
            seq += 1
            content_parts: list[str] = []
            human_turn: str | None = None
            assistant_turn: str | None = None
            model: str | None = None
            ts: Any = None
            last_msg_id: str | None = None

            if prev_assistant:
                content_parts.append(f"[context] {prev_assistant.strip()[:300]}")
            elif seq == 1 and title:
                content_parts.append(f"[title] {title[:200]}")

            for m in group:
                ts = ts or m.get("create_time")
                last_msg_id = m.get("id", last_msg_id)
                model = model or (m.get("metadata") or {}).get("model_slug")
                text = _msg_text(m)
                if (m.get("author") or {}).get("role") == "user":
                    if human_turn is None:
                        human_turn = text[:3000]
                        content_parts.append(f"[user] {human_turn}")
                else:
                    assistant_turn = text
                    content_parts.append(f"[assistant] {text[:3000]}")

            if not content_parts:
                continue

            span_id = f"openai:{last_msg_id}" if last_msg_id else None
            if span_id is not None:
                if span_id in seen_span_ids:
                    continue
                seen_span_ids.add(span_id)

            created_at = _epoch_to_dt(ts) or _epoch_to_dt(convo.get("create_time"))

            episodes.append(
                Episode(
                    session_id=str(session_id),
                    sequence=seq,
                    project=None,
                    platform="chatgpt",
                    model=model or default_model,
                    human_turn=_strip_nulls(human_turn) if human_turn else None,
                    assistant_turn=_strip_nulls(assistant_turn) if assistant_turn else None,
                    content=_strip_nulls("\n\n".join(content_parts)),
                    span_id=span_id,
                    source="openai",
                    metadata={
                        "export_path": source_label,
                        "ts": created_at.isoformat() if created_at else None,
                        "title": title or None,
                    },
                    created_at=created_at,
                )
            )
            prev_assistant = assistant_turn

        return episodes


def _conversation_files(path: Path) -> list[Path]:
    """Resolve an export path to conversation JSON file(s).

    Accepts the extracted export directory (finds conversations.json or the
    conversations-NNN.json shards), or a single JSON file directly.
    """
    if path.is_file():
        return [path]
    single = path / "conversations.json"
    if single.exists():
        return [single]
    return sorted(path.glob("conversations-*.json"))


def parse_export(path: Path) -> tuple[list[Episode], int]:
    """Parse a ChatGPT export into Episodes.

    Returns (episodes, skipped_do_not_remember_count).
    """
    files = _conversation_files(path)
    if not files:
        raise FileNotFoundError(f"no conversations JSON found under {path}")
    episodes: list[Episode] = []
    skipped_dnr = 0
    for f in files:
        data = orjson.loads(f.read_bytes())
        if not isinstance(data, list):
            logger.warning("skipping %s: not a conversation list", f)
            continue
        for convo in data:
            if not isinstance(convo, dict):
                continue
            if convo.get("is_do_not_remember"):
                skipped_dnr += 1
                continue
            episodes.extend(OpenAIChatParser.parse_conversation(convo, str(f)))
    return episodes, skipped_dnr
