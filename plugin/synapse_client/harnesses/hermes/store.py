"""Hermes's session store, ``<hermes_home>/state.db``, read for uploads.

The live provider and ``synapse import hermes`` read a session's conversation here, with
the same code, so both send the same rows in the same records (hermes/messages.py
``to_records``).

What it reads
-------------
The store is opened read-only (``mode=ro``), by a URI built from the absolute path
(``Path.as_uri``, which also spells a Windows path), and read in one transaction, so a
running Hermes is neither blocked for long nor seen half-way: Hermes writes each flush
of a turn in one transaction (hermes_state_messages.py ``append_messages_batch``). A
``schema_version`` other than SCHEMA_VERSIONS, or a table or column this module reads
that is not there, raises SchemaError before any row is read.

The rows of a session are the live ones and those an in-place compaction archived
(``active=1 OR compacted=1``): what the conversation showed at some point. A row
removed by an undo or rewind is left out. They come in id order, and each
``message_uid`` once, at its first row: a compaction clones the rows it keeps with
their uid (its summary takes the uid of a row it replaces), a branch or a compressed
continuation clones its parent's rows into the new session with theirs. ``until``
stops the read at the first row of one uid: the last row of the run a hook reports,
so a run that started since (its user row is written as it is submitted, its tool rows
as they come) is not read.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple

#: The ``schema_version`` values of Hermes's session store this reader knows.
SCHEMA_VERSIONS = frozenset({33})
#: Seconds to wait for a locked store.
DB_TIMEOUT = 5.0
#: The columns read, per table.
COLUMNS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "sessions": (
            "id",
            "source",
            "user_id",
            "session_key",
            "chat_id",
            "chat_type",
            "model",
            "model_config",
            "parent_session_id",
            "started_at",
            "cwd",
            "title",
        ),
        "messages": (
            "id",
            "session_id",
            "role",
            "content",
            "tool_calls",
            "tool_name",
            "timestamp",
            "_compressed_summary",
            "active",
            "compacted",
            "display_kind",
            "message_uid",
            "absorbed_message_uids",
        ),
        "state_meta": ("key", "value"),
        "session_admissions": ("principal_id", "target_session_id"),
        "schema_version": ("version",),
    }
)

_CONTENT_JSON = "\x00json:"  # hermes_state.py ``_CONTENT_JSON_PREFIX``
#: A gateway principal that is the session's own automation (a background task
#: reporting back): ``automation:<route>``.
AUTOMATION = "automation:"

_ROW_NAMES = (
    "id",
    "role",
    "content",
    "tool_calls",
    "tool_name",
    "timestamp",
    "_compressed_summary",
    "display_kind",
    "message_uid",
    "absorbed_message_uids",
)
_ROWS_SQL = (
    "SELECT id, role, content, tool_calls, tool_name, timestamp, _compressed_summary, "
    "display_kind, message_uid, absorbed_message_uids FROM messages "
    "WHERE session_id=? AND (active=1 OR compacted=1) ORDER BY id"
)


class SchemaError(Exception):
    """The store is not one this reader knows: nothing was read."""


class Read(NamedTuple):
    """A session's conversation as read: the rows as Hermes dicts, and whether the read
    stopped at the uid it was asked to stop at (always True when none was asked)."""

    messages: list[dict[str, Any]]
    reached: bool


def path(home: str | Path) -> Path:
    return Path(home) / "state.db"


def connect(home: str | Path) -> sqlite3.Connection:
    """A read-only connection to ``<home>/state.db``. RAISES FileNotFoundError when there
    is none, sqlite3.Error when it cannot be opened."""
    db = path(home)
    if not db.is_file():
        raise FileNotFoundError(str(db))
    return sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True, timeout=DB_TIMEOUT)


def check_schema(conn: sqlite3.Connection) -> int:
    """The store's schema version. RAISES SchemaError when it is not one of
    SCHEMA_VERSIONS, or a table or column read here is missing."""
    have: dict[str, set[str]] = {}
    for table in COLUMNS:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        have[table] = {str(row[1]) for row in rows}
    missing = [
        f"{table}.{column}"
        for table, columns in COLUMNS.items()
        for column in columns
        if column not in have[table]
    ]
    if not have["schema_version"]:
        raise SchemaError("state.db has no schema_version table: not a Hermes session store")
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    version = row[0] if row else None
    if not isinstance(version, int) or version not in SCHEMA_VERSIONS:
        known = ", ".join(str(v) for v in sorted(SCHEMA_VERSIONS))
        raise SchemaError(
            f"state.db is at schema version {version!r}; this client reads version {known}. "
            "Update the Synapse client (synapse update), or Hermes, and try again."
        )
    if missing:
        raise SchemaError(f"state.db lacks what this client reads: {', '.join(missing)}")
    return version


def decode(content: Any) -> Any:
    """A stored ``content`` as Hermes restores it (hermes_state_messages.py
    ``_decode_content``)."""
    if isinstance(content, str) and content.startswith(_CONTENT_JSON):
        try:
            return json.loads(content[len(_CONTENT_JSON) :])
        except ValueError:
            return content
    return content


def loads(value: Any) -> Any:
    """``value`` parsed as JSON, or None when it is not a JSON string."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return json.loads(value)
    except ValueError:
        return None


def message(row: Mapping[str, Any]) -> dict[str, Any]:
    """A stored row as the dict Hermes hands a memory provider, with the keys
    hermes/messages.py reads (hermes_state_messages.py ``_rows_to_conversation``)."""
    role = row["role"]
    content = decode(row["content"])
    if role in ("user", "assistant") and isinstance(content, str):
        content = content.strip()
    msg: dict[str, Any] = {"role": role, "content": content}
    for key in ("timestamp", "tool_name", "display_kind", "message_uid"):
        if row[key]:
            msg[key] = row[key]
    calls = loads(row["tool_calls"])
    if calls:
        msg["tool_calls"] = calls
    if row["_compressed_summary"]:
        msg["_compressed_summary"] = True
    absorbed = row["absorbed_message_uids"]
    if absorbed:
        parsed = loads(absorbed)
        if parsed is None or parsed:
            msg["_absorbed_message_uids"] = parsed if parsed is not None else absorbed
    return msg


def rows(conn: sqlite3.Connection, session_id: str, until: str | None = None) -> Read:
    """A session's messages in order, each uid once at its first row (see the module
    notes); with ``until``, up to and including the first row of that uid. ``reached``
    is False when ``until`` was asked and no row of the session has it."""
    out: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for values in conn.execute(_ROWS_SQL, (session_id,)):
        row = dict(zip(_ROW_NAMES, values, strict=True))
        uid = row["message_uid"]
        key = ("uid", uid) if uid else ("row", row["role"], row["timestamp"], row["content"])
        if key in seen:
            continue
        seen.add(key)
        out.append(message(row))
        if until is not None and uid == until:
            return Read(out, True)
    return Read(out, until is None)


def uids(conn: sqlite3.Connection, session_ids: Iterable[str]) -> set[str]:
    """Every ``message_uid`` the sessions ``session_ids`` hold, live or archived."""
    found: set[str] = set()
    for sid in session_ids:
        for (uid,) in conn.execute(
            "SELECT message_uid FROM messages WHERE session_id=? AND message_uid IS NOT NULL "
            "AND (active=1 OR compacted=1)",
            (sid,),
        ):
            if isinstance(uid, str) and uid:
                found.add(uid)
    return found


def principals(conn: sqlite3.Connection, session_id: str) -> set[str]:
    """The principals the gateway admitted a turn of ``session_id`` from (empty for a
    session no gateway ran: the stdio TUI and Desktop keep no admissions)."""
    query = "SELECT DISTINCT principal_id FROM session_admissions WHERE target_session_id=?"
    return {str(row[0]) for row in conn.execute(query, (session_id,))}


def admitted_others(
    conn: sqlite3.Connection, session_id: str, owner: str | None, route: Any
) -> bool:
    """Whether the gateway admitted a turn into ``session_id`` from anyone but ``owner``
    and the session's own automation (``automation:<route>``)."""
    allowed = {owner} if owner else set()
    if isinstance(route, str) and route:
        allowed.add(AUTOMATION + route)
    return any(one not in allowed for one in principals(conn, session_id))
