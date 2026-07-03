"""Parse Cursor AI chat history from state.vscdb / state.sqlite files.

Cursor stores conversations in SQLite databases. This module extracts
human-readable conversation text from them for review.

Expected file location: a local directory of exported Cursor SQLite databases.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

CURSOR_DATA_DIR = os.path.expanduser("~/data/cursor")
LAST_REVIEWED_FILE = os.path.join(CURSOR_DATA_DIR, ".last_reviewed")


def _get_last_reviewed_time() -> float:
    """Get the mtime of the last reviewed DB file, or 0 if never reviewed."""
    try:
        with open(LAST_REVIEWED_FILE) as f:
            return float(f.read().strip())
    except (FileNotFoundError, ValueError):
        return 0.0


def _set_last_reviewed_time(mtime: float) -> None:
    """Record the mtime of the DB we just reviewed."""
    with open(LAST_REVIEWED_FILE, "w") as f:
        f.write(str(mtime))


def _find_db_files() -> list[Path]:
    """Find all state.vscdb and state.sqlite files in the cursor data dir."""
    data_dir = Path(CURSOR_DATA_DIR)
    if not data_dir.exists():
        return []

    files = []
    for pattern in ["state.vscdb", "state.sqlite", "*.vscdb", "*.sqlite"]:
        files.extend(data_dir.glob(pattern))
    return files


def _extract_from_vscdb(db_path: Path) -> list[dict]:
    """Extract conversations from a state.vscdb file (workspace storage format).

    Reads from ItemTable, keys: aiService.prompts, workbench.panel.aichat.view.aichat.chatdata
    """
    conversations = []

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()

        # Check what tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        if "ItemTable" in tables:
            # Try the known chat data keys
            chat_keys = [
                "aiService.prompts",
                "workbench.panel.aichat.view.aichat.chatdata",
            ]
            for key in chat_keys:
                try:
                    cursor.execute("SELECT value FROM ItemTable WHERE [key] = ?", (key,))
                    for (value,) in cursor.fetchall():
                        if value:
                            parsed = _parse_chat_json(value, key)
                            conversations.extend(parsed)
                except sqlite3.Error as e:
                    logger.debug("Error reading key %s: %s", key, e)

            # Also try composer data
            try:
                cursor.execute("SELECT [key], value FROM ItemTable WHERE [key] LIKE 'composer%'")
                for key, value in cursor.fetchall():
                    if value:
                        parsed = _parse_chat_json(value, key)
                        conversations.extend(parsed)
            except sqlite3.Error:
                pass

        # Check for cursorDiskKV table (session DB format)
        if "cursorDiskKV" in tables:
            conversations.extend(_extract_bubbles(cursor))

        conn.close()
    except sqlite3.Error as e:
        logger.warning("Failed to read %s: %s", db_path, e)

    return conversations


def _extract_bubbles(cursor: sqlite3.Cursor) -> list[dict]:
    """Extract bubble-format messages from cursorDiskKV table."""
    conversations = []

    try:
        cursor.execute(
            "SELECT rowid, key, value FROM cursorDiskKV WHERE key LIKE 'bubbleId:%' ORDER BY rowid"
        )
        current_conv = []

        for _rowid, _key, value in cursor.fetchall():
            try:
                bubble = json.loads(value)
                role = "user" if bubble.get("type") == 1 else "assistant"
                text = bubble.get("text", "").strip()
                if text:
                    current_conv.append({"role": role, "content": text})
            except (json.JSONDecodeError, TypeError):
                continue

        if current_conv:
            conversations.append(
                {
                    "source": "cursor_bubbles",
                    "messages": current_conv,
                }
            )
    except sqlite3.Error as e:
        logger.debug("Error reading bubbles: %s", e)

    return conversations


def _parse_chat_json(raw_value: str, source_key: str) -> list[dict]:
    """Parse a JSON chat value from ItemTable into conversation dicts."""
    conversations = []

    try:
        data = json.loads(raw_value)
    except (json.JSONDecodeError, TypeError):
        return []

    # Handle different JSON structures
    if isinstance(data, list):
        # List of messages or prompts
        messages = []
        for item in data:
            if isinstance(item, dict):
                role = item.get("role", "user")
                content = item.get("content") or item.get("text") or item.get("message", "")
                if isinstance(content, str) and content.strip():
                    messages.append({"role": role, "content": content.strip()})
        if messages:
            conversations.append({"source": source_key, "messages": messages})

    elif isinstance(data, dict):
        # Could be chatdata with tabs/conversations
        for key in ["tabs", "allComposers", "conversations", "chats"]:
            items = data.get(key, [])
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    # Look for messages/bubbles in each tab/composer
                    msgs_raw = (
                        item.get("messages")
                        or item.get("bubbles")
                        or item.get("conversation")
                        or []
                    )
                    if not isinstance(msgs_raw, list):
                        continue
                    messages = []
                    for msg in msgs_raw:
                        if not isinstance(msg, dict):
                            continue
                        role = msg.get("role", "")
                        if msg.get("type") == 1:
                            role = "user"
                        elif msg.get("type") == 2:
                            role = "assistant"
                        content = msg.get("content") or msg.get("text") or msg.get("message", "")
                        if isinstance(content, str) and content.strip():
                            messages.append({"role": role, "content": content.strip()})
                    if messages:
                        conversations.append(
                            {
                                "source": f"{source_key}/{key}",
                                "messages": messages,
                            }
                        )

    return conversations


def format_conversations(conversations: list[dict]) -> str:
    """Format parsed conversations into human-readable text for review."""
    parts = []

    for i, conv in enumerate(conversations):
        source = conv.get("source", "unknown")
        messages = conv.get("messages", [])
        if not messages:
            continue

        parts.append(f"=== Cursor Conversation {i + 1} (source: {source}) ===")

        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Cap individual messages but keep them generous
            if len(content) > 3000:
                content = content[:3000] + "..."
            parts.append(f"  [{role}] {content}")

        parts.append("")

    return "\n".join(parts)


def get_new_cursor_conversations() -> str | None:
    """Get new Cursor conversations that haven't been reviewed yet.

    Returns formatted conversation text, or None if nothing new.
    """
    db_files = _find_db_files()
    if not db_files:
        logger.info("No Cursor DB files found in %s", CURSOR_DATA_DIR)
        return None

    last_reviewed = _get_last_reviewed_time()
    newest_mtime = 0.0
    all_conversations = []

    for db_file in db_files:
        mtime = db_file.stat().st_mtime
        if mtime <= last_reviewed:
            logger.info("Skipping %s — already reviewed", db_file.name)
            continue

        newest_mtime = max(newest_mtime, mtime)
        logger.info(
            "Parsing %s (modified %s)",
            db_file.name,
            datetime.fromtimestamp(mtime, tz=UTC).isoformat(),
        )

        conversations = _extract_from_vscdb(db_file)
        all_conversations.extend(conversations)
        logger.info("Found %d conversations in %s", len(conversations), db_file.name)

    if not all_conversations:
        return None

    text = format_conversations(all_conversations)
    if not text.strip():
        return None

    # Record that we've reviewed up to this point
    if newest_mtime > 0:
        _set_last_reviewed_time(newest_mtime)

    logger.info(
        "Extracted %d Cursor conversations, %d chars",
        len(all_conversations),
        len(text),
    )
    return text
