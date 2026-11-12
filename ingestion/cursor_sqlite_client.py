"""Parse Cursor IDE chat history from SQLite (state.vscdb / global-state.vscdb) into Episodes.

Cursor stores conversations across two SQLite tables:
- `cursorDiskKV` — primary store keyed by `bubbleId:<composerId>:<bubbleId>` (per-bubble JSON)
- `ItemTable` — fallback for older or workspace-scoped chats (`composer*` keys, etc.)

Each composerId corresponds to one logical conversation. We group bubbles by composerId,
sort by their `createdAt` timestamp, and emit one Episode per user-assistant turn —
matching the schema the Logfire and JSONL paths produce.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import orjson

from ingestion.models import Episode
from ingestion.render import (
    USER_CAP,
    assistant_line,
    context_line,
    group_turns,
    join_lines,
    strip_nulls,
    tool_line,
    user_line,
)

logger = logging.getLogger(__name__)

_BUBBLE_KEY_RE = re.compile(r"^bubbleId:([0-9a-f-]{36}):([0-9a-f-]{36})$")


def _is_user_turn(bubble: dict[str, Any]) -> bool:
    """A user bubble with text opens a new turn."""
    text = (bubble.get("text") or "").strip()
    return bubble.get("type") == 1 and bool(text)


def _parse_iso(ts: Any) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _project_from_uris(uris: Any) -> str | None:
    """Pull a project basename from Cursor's workspaceUris list (URL-encoded file:// URIs)."""
    if not isinstance(uris, list) or not uris:
        return None
    from urllib.parse import unquote, urlparse

    for raw in uris:
        if not isinstance(raw, str):
            continue
        try:
            path = unquote(urlparse(raw).path)
        except (ValueError, TypeError):
            continue
        path = path.rstrip("/")
        if not path:
            continue
        # Strip Windows drive prefix like /c:/...
        path = re.sub(r"^/[a-zA-Z]:", "", path)
        basename = path.rsplit("/", 1)[-1]
        if basename and basename.lower() not in ("projects", "users"):
            return basename
    return None


class CursorSQLiteParser:
    """Read a Cursor SQLite file and yield Episodes per conversation."""

    @staticmethod
    def parse_file(db_path: Path) -> list[Episode]:
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        except sqlite3.Error as e:
            logger.warning("Cannot open %s: %s", db_path, e)
            return []

        try:
            tables = {
                r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if "cursorDiskKV" not in tables:
                return []
            return list(CursorSQLiteParser._extract_bubbles(conn))
        finally:
            conn.close()

    @staticmethod
    def _extract_bubbles(conn: sqlite3.Connection) -> Iterable[Episode]:
        """Walk all bubbleId:* rows, group by composer, build Episodes."""
        by_composer: dict[str, list[dict[str, Any]]] = defaultdict(list)

        rows = conn.execute("SELECT key, value FROM cursorDiskKV WHERE key LIKE 'bubbleId:%'")
        for key, value in rows:
            m = _BUBBLE_KEY_RE.match(key or "")
            if not m:
                continue
            composer_id, bubble_id = m.group(1), m.group(2)
            try:
                bubble = orjson.loads(value)
            except orjson.JSONDecodeError:
                continue
            if not isinstance(bubble, dict):
                continue
            bubble["_composer_id"] = composer_id
            bubble["_bubble_id"] = bubble_id
            by_composer[composer_id].append(bubble)

        for composer_id, bubbles in by_composer.items():
            bubbles.sort(key=lambda b: (b.get("createdAt") or "", b.get("_bubble_id", "")))
            yield from CursorSQLiteParser._composer_to_episodes(composer_id, bubbles)

    @staticmethod
    def _composer_to_episodes(composer_id: str, bubbles: list[dict[str, Any]]) -> Iterable[Episode]:
        groups = group_turns(bubbles, _is_user_turn)
        prev_assistant: str | None = None
        seq = 0

        for group in groups:
            if not group:
                continue
            seq += 1
            content_parts: list[str] = []
            human_turn: str | None = None
            assistant_turn: str | None = None
            model: str | None = None
            ts: str | None = None
            project: str | None = None
            last_bubble_id: str | None = None

            if prev_assistant:
                content_parts.append(context_line(prev_assistant))

            for b in group:
                ts = ts or b.get("createdAt")
                last_bubble_id = b.get("_bubble_id", last_bubble_id)
                btype = b.get("type")
                text = (b.get("text") or "").strip()
                project = project or _project_from_uris(b.get("workspaceUris"))

                model_info = b.get("modelInfo")
                if isinstance(model_info, dict):
                    model = model or model_info.get("model")

                if btype == 1:  # user
                    if text and human_turn is None:
                        human_turn = text[:USER_CAP]
                        content_parts.append(user_line(human_turn))
                elif btype == 2:  # assistant
                    if text:
                        assistant_turn = text
                        content_parts.append(assistant_line(text))
                    # surface tool invocations
                    tool_results = b.get("toolResults") or []
                    for tr in tool_results if isinstance(tool_results, list) else []:
                        if not isinstance(tr, dict):
                            continue
                        name = tr.get("name") or tr.get("tool") or "tool"
                        result = tr.get("result") or tr.get("output") or ""
                        if isinstance(result, dict | list):
                            result = orjson.dumps(result).decode()[:300]
                        result_text = str(result).strip()
                        if len(result_text) > 20:
                            content_parts.append(tool_line(name, result_text))

            if not content_parts or not (human_turn or assistant_turn):
                continue

            try:
                created_at = _parse_iso(ts)
            except (ValueError, TypeError):
                created_at = None

            yield Episode(
                session_id=composer_id,
                sequence=seq,
                project=project,
                platform="cursor",
                model=model,
                human_turn=strip_nulls(human_turn) if human_turn else None,
                assistant_turn=strip_nulls(assistant_turn) if assistant_turn else None,
                content=join_lines(content_parts),
                span_id=f"cursor:{composer_id}:{last_bubble_id}" if last_bubble_id else None,
                source="cursor_sqlite",
                metadata={
                    "composer_id": composer_id,
                    "ts": ts,
                },
                created_at=created_at,
            )
            prev_assistant = assistant_turn
