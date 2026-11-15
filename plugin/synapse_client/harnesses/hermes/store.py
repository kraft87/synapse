"""Hermes's session store, ``<hermes_home>/state.db``, read by the client.

``synapse import hermes`` reads past sessions' rows here (``rows``). The live provider
reads only two things that are not text: each row's ``display_kind`` (``kinds``) and who
started each turn the gateway ran (``admitted_others``); the rows it sends come from the
list Hermes hands it (source.py says why).

What it reads
-------------
The store is opened read-only (``mode=ro``), by a URI built from the absolute path
(``uri``, which also spells a Windows drive or UNC path), and read in one transaction,
so a running Hermes is neither blocked for long nor seen half-way. A ``schema_version``
other than SCHEMA_VERSIONS, or a table or column this module reads that is not there,
raises SchemaError before any row is read.

The rows of a session are the live ones and those an in-place compaction archived
(``active=1 OR compacted=1``): what the conversation showed at some point. A row
removed by an undo or rewind is left out, and so is a prompt the TUI wrote as it
accepted it while a run was going (``display_metadata._queued_prompt``): Hermes writes
it again, at its place in the conversation, when the run that answers it starts
(tui_gateway/session_auto_continue.py), and the early row is deactivated then, or
archived with it by a compaction during the run. They come in id order, and each
``message_uid`` once, at its first row: a compaction clones the rows it keeps with
their uid (its summary takes the uid of a row it replaces), a branch or a compressed
continuation clones its parent's rows into the new session with theirs.

What the store is not: the rows are not scrubbed of secrets (Hermes scrubs only what it
hands a provider); a queued prompt's first row sits where it was accepted; a row Hermes
could not write is not there; archived rows stay; a ``display_kind`` can be stamped
after the run; a row can be rewritten after it was first written.
"""

from __future__ import annotations

import json
import sqlite3
import urllib.parse
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

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
            "display_metadata",
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
    "display_metadata",
)
_ROWS_SQL = (
    "SELECT id, role, content, tool_calls, tool_name, timestamp, _compressed_summary, "
    "display_kind, message_uid, absorbed_message_uids, display_metadata FROM messages "
    "WHERE session_id=? AND (active=1 OR compacted=1) ORDER BY id"
)
#: The ``display_metadata`` key the TUI puts on a prompt it wrote as it accepted it
#: during a run (tui_gateway/session_auto_continue.py).
_QUEUED = "_queued_prompt"


class SchemaError(Exception):
    """The store is not one this reader knows: nothing was read."""


def path(home: str | Path) -> Path:
    return Path(home) / "state.db"


def uri(db: Path) -> str:
    """A read-only SQLite URI for the absolute path ``db``. ``Path.as_uri`` spells a UNC
    path ``file://server/share/...``, an authority SQLite refuses; it is spelled with an
    empty authority instead (``file:////server/share/...``), which SQLite hands the
    system as ``//server/share/...``."""
    text = str(db)
    if text.startswith(("\\\\", "//")):
        tail = text.replace("\\", "/").lstrip("/")
        return "file:////" + urllib.parse.quote(tail, safe="/:") + "?mode=ro"
    return db.as_uri() + "?mode=ro"


def connect(home: str | Path) -> sqlite3.Connection:
    """A read-only connection to ``<home>/state.db``. RAISES FileNotFoundError when there
    is none, sqlite3.Error when it cannot be opened."""
    db = path(home)
    if not db.is_file():
        raise FileNotFoundError(str(db))
    return sqlite3.connect(uri(db.resolve()), uri=True, timeout=DB_TIMEOUT)


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


def rows(conn: sqlite3.Connection, session_id: str) -> list[dict[str, Any]]:
    """A session's messages in order, each uid once at its first row, without the rows
    a queued prompt's acceptance wrote (see the module notes)."""
    out: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for values in conn.execute(_ROWS_SQL, (session_id,)):
        row = dict(zip(_ROW_NAMES, values, strict=True))
        meta = loads(row["display_metadata"])
        if isinstance(meta, dict) and meta.get(_QUEUED):
            continue
        uid = row["message_uid"]
        key = ("uid", uid) if uid else ("row", row["role"], row["timestamp"], row["content"])
        if key in seen:
            continue
        seen.add(key)
        out.append(message(row))
    return out


def kinds(conn: sqlite3.Connection, wanted: Iterable[str]) -> dict[str, tuple[str, str]]:
    """``(role, display_kind)`` of the first row the store holds for each uid of
    ``wanted`` (``display_kind`` empty when it has none); a uid it does not hold is not
    in the answer."""
    out: dict[str, tuple[str, str]] = {}
    ids = [one for one in dict.fromkeys(wanted) if isinstance(one, str) and one]
    for start in range(0, len(ids), 500):
        chunk = ids[start : start + 500]
        marks = ",".join("?" for _ in chunk)
        query = (
            "SELECT message_uid, role, display_kind FROM messages WHERE message_uid IN ("
            + marks
            + ") ORDER BY id"
        )
        for uid, role, kind in conn.execute(query, chunk):
            out.setdefault(uid, (str(role or ""), str(kind or "")))
    return out


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
