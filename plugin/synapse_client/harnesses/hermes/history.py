"""Past Hermes sessions, read from ``<hermes_home>/state.db`` for ``synapse import hermes``.

The live provider uploads the turns of a session as they finish. A session from before
the provider was installed, or one it did not reach, sits only in Hermes's session
store. This module reads that store and says, per session, whether its turns may go to
Synapse and which ones; ``plugin/scripts/import_hermes.py`` asks and sends.

What it reads
-------------
The store is opened read-only (``mode=ro``) and read in one transaction, so a running
Hermes (WAL mode) is neither blocked nor seen half-way. Nothing under the Hermes home is
written. A ``schema_version`` other than SCHEMA_VERSIONS, or a table or column this
module reads that is not there, raises SchemaError before anything is read.

- ``sessions``: one row per session. ``source`` (the surface, "local" for the gateway's
  local chats, or "subagent", "api_server", "tool", "cron", a platform), ``user_id``,
  ``chat_id``, ``chat_type``, ``session_key`` (the gateway route), ``parent_session_id``
  and ``model_config`` (lineage.py), ``title``, ``model``, ``cwd``.
- ``state_meta``: the creation receipt of each local chat and the API bindings
  (sessions.py).
- ``session_admissions``: who submitted each turn the gateway ran (``principal_id``).
- ``messages``: the rows of each session in order, with ``message_uid``.

Which sessions
--------------
The live provider's rule, from the data the store keeps (sessions.py has the rule and
the Hermes lines it rests on). A session is imported only when:

- its row has the shape of a local session: a gateway row (``chat_id``, ``user_id``,
  ``session_key``) passes ``sessions.from_keywords`` as the keywords Hermes would have
  built from it (a local dm of this machine's owner); a row with no chat identity is a
  session an in-process surface ran (the stdio TUI, Desktop), except an ACP one and a
  side agent the gateway built (``bg_``, ``btw_``, ``preview_`` ids, as Hermes names
  them in gateway/session_tui_side.py and tui_gateway/methods_prompt.py);
- the store check passes (``sessions._judge``): a source in ROW_SOURCES and not
  ``tool``, no API binding, a creation receipt naming the owner and an owner's surface,
  no Bot Chat or Group Chat title and no chat that follows its profile, on the session,
  its chat or a session it descends from;
- no mark under ``<hermes_home>/synapse/inert/`` names it, its chat or a session it
  descends from (marks.py; read, never written here);
- every turn the gateway admitted into it came from the owner, or from the session's own
  automation (``automation:<route>``: a background task reporting back). A turn another
  person or agent wrote turns the live provider off for good; here the session is
  skipped whole;
- it is not private, and neither is a session whose turns it holds (checked by the
  caller: the marker and the server's row).

Which turns
-----------
The rows become records through the live path's own conversion (hermes/messages.py
``to_records``) from dicts shaped as Hermes restores them (hermes_state_messages.py
``_rows_to_conversation``), and are cut into turns by the core (messages.py). A turn's
span id is the ``message_uid`` of its last row, as when the provider sent it, so a turn
the provider uploaded is the same turn here, and the server keeps one copy.

The rows read are the live ones and those an in-place compaction archived (``active=1
OR compacted=1``): what the conversation showed at some point. A row removed by an
undo or rewind is left out. A compaction clones the rows it keeps, with their uid, so a
uid is read once. A branch or compressed continuation clones its parent's rows into the
new session with their uids: a turn whose opening row is one of the sessions it holds
turns of (``lineage.chain``) is a copy and is not imported under the new id, nor sent as
the context of a later turn. Only complete turns are imported, each with the complete
turn before it as context, in the live path's batches.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple

from ... import messages, project
from . import messages as hm
from . import sessions as hs
from .lineage import chain, lineage
from .marks import marked

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

#: The surface a row's ``source`` stands for, as ``initialize``'s ``platform`` keyword.
_PLATFORM: Mapping[str, str] = MappingProxyType(
    {
        "local": "cli",
        "cli": "cli",
        "oneshot": "cli",
        "tool": "cli",
        "tui": "tui",
        "desktop": "desktop",
        "gui": "desktop",
        "acp": "acp",
    }
)
_SIDE_AGENT = re.compile(r"(bg|btw|preview)_[0-9a-f]+")
_CONTENT_JSON = "\x00json:"  # hermes_state.py ``_CONTENT_JSON_PREFIX``
_AUTOMATION = "automation:"

FOREIGN_TURN = "a turn another person or agent wrote"
SIDE_AGENT = "a side agent the gateway built (/background, /btw)"
NO_TURNS = "no finished turn of its own"

Pair = tuple[messages.Turn | None, messages.Turn]


class SchemaError(Exception):
    """The store is not one this reader knows: nothing was read."""


class Session(NamedTuple):
    """One session of the store. ``reason`` is why it is not imported, empty when it is;
    ``pairs`` are its turns to import, each with its context; ``continues`` the sessions
    whose turns it holds, nearest first; ``copies`` how many of its turns are theirs."""

    id: str
    title: str
    started_at: float
    last_at: float
    model: str | None
    cwd: str | None
    project: str | None
    reason: str
    continues: tuple[str, ...] = ()
    pairs: tuple[Pair, ...] = ()
    copies: int = 0


def connect(home: str | Path) -> sqlite3.Connection:
    """A read-only connection to ``<home>/state.db``. RAISES FileNotFoundError when there
    is none, sqlite3.Error when it cannot be opened."""
    path = Path(home) / "state.db"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=DB_TIMEOUT)


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


def _decode(content: Any) -> Any:
    """A stored ``content`` as Hermes restores it (hermes_state_messages.py
    ``_decode_content``)."""
    if isinstance(content, str) and content.startswith(_CONTENT_JSON):
        try:
            return json.loads(content[len(_CONTENT_JSON) :])
        except ValueError:
            return content
    return content


def _json(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return None
    try:
        return json.loads(value)
    except ValueError:
        return None


def _message(row: Mapping[str, Any]) -> dict[str, Any]:
    """A stored row as the dict Hermes hands a memory provider, with the keys
    hermes/messages.py reads (hermes_state_messages.py ``_rows_to_conversation``)."""
    role = row["role"]
    content = _decode(row["content"])
    if role in ("user", "assistant") and isinstance(content, str):
        content = content.strip()
    msg: dict[str, Any] = {"role": role, "content": content}
    for key in ("timestamp", "tool_name", "display_kind", "message_uid"):
        if row[key]:
            msg[key] = row[key]
    calls = _json(row["tool_calls"])
    if calls:
        msg["tool_calls"] = calls
    if row["_compressed_summary"]:
        msg["_compressed_summary"] = True
    absorbed = row["absorbed_message_uids"]
    if absorbed:
        parsed = _json(absorbed)
        if parsed is None or parsed:
            msg["_absorbed_message_uids"] = parsed if parsed is not None else absorbed
    return msg


def _rows(conn: sqlite3.Connection, session_id: str) -> list[dict[str, Any]]:
    """A session's messages in order: the live rows and those an in-place compaction
    archived, each message once (a compaction clones the rows it keeps)."""
    query = (
        "SELECT role, content, tool_calls, tool_name, timestamp, _compressed_summary, "
        "display_kind, message_uid, absorbed_message_uids FROM messages "
        "WHERE session_id=? AND (active=1 OR compacted=1) ORDER BY id"
    )
    names = (
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
    out: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for values in conn.execute(query, (session_id,)):
        row = dict(zip(names, values, strict=True))
        uid = row["message_uid"]
        key = ("uid", uid) if uid else ("row", row["role"], row["timestamp"], row["content"])
        if key in seen:
            continue
        seen.add(key)
        out.append(_message(row))
    return out


def records(conn: sqlite3.Connection, session_id: str) -> list[messages.Record]:
    """The ``/ingest`` records of a session's stored conversation, by the live path's
    conversion (hermes/messages.py ``to_records``)."""
    return hm.to_records(_rows(conn, session_id))


def _admitted_others(conn: sqlite3.Connection, session_id: str, route: Any) -> bool:
    """Whether the gateway admitted a turn into the session from anyone but the owner
    and the session's own automation."""
    owner = hs.owner_subject()
    allowed = {owner} if owner else set()
    if isinstance(route, str) and route:
        allowed.add(_AUTOMATION + route)
    query = "SELECT DISTINCT principal_id FROM session_admissions WHERE target_session_id=?"
    return any(row[0] not in allowed for row in conn.execute(query, (session_id,)))


def judge(
    conn: sqlite3.Connection, home: Any, row: Mapping[str, Any]
) -> tuple[str, str | None, tuple[str, ...]]:
    """``(reason, cwd, continues)`` for one ``sessions`` row: why it is not imported
    (empty when it may be), the working directory the live provider would have used,
    and the sessions whose turns it holds. RAISES sqlite3.Error, ValueError, and OSError
    when the marks cannot be read."""
    sid = str(row["id"])
    source = str(row["source"] or "").strip().lower()
    chat = str(row["chat_id"] or "")
    identity = {
        "user_id": row["user_id"],
        "chat_id": row["chat_id"],
        "chat_type": row["chat_type"],
        "gateway_session_key": row["session_key"],
    }
    gateway = any(identity.values())
    kwargs: dict[str, Any] = {}
    if gateway:
        kwargs = {key: value for key, value in identity.items() if value}
        if not hs.gateway_session(kwargs):
            return "a session with a user and no chat", None, ()
        keywords = {
            **kwargs,
            "platform": _PLATFORM.get(source, source),
            "agent_context": "primary",
            "session_title": row["title"],
        }
        found = hs.from_keywords(keywords)
        if found.mode != hs.ACTIVE:
            return found.reason, None, ()
    elif _SIDE_AGENT.fullmatch(sid):
        return SIDE_AGENT, None, ()
    elif source == "acp":
        return hs._ACP_ALONE, None, ()
    stored = (row["title"], row["source"], row["chat_id"], row["model_config"])
    parent = row["parent_session_id"]
    parents = lineage(conn, parent, row["model_config"])
    chats = [chat] if chat else []
    found, _ = hs._judge(conn, sid, kwargs, (*stored, parent), chats, parents)
    if found.mode != hs.ACTIVE:
        return found.reason, None, ()
    for one in parents:
        if marked(home, [one.id, one.chat]):
            return hs._MARKED, None, ()
    if marked(home, [sid, *chats]):
        return hs._MARKED, None, ()
    if gateway and _admitted_others(conn, sid, row["session_key"]):
        return FOREIGN_TURN, None, ()
    cwd = _gateway_cwd(conn, chat) if gateway else None
    if not cwd and isinstance(row["cwd"], str) and row["cwd"].strip():
        cwd = row["cwd"]
    return "", cwd, chain(conn, parent, row["model_config"])


def _gateway_cwd(conn: sqlite3.Connection, chat: str) -> str | None:
    """The working directory a gateway session's agent ran in: its policy's ``cwd``,
    which Hermes hands the provider (``session_cwd``), else the client's request."""
    receipt = hs._meta(conn, hs._POLICY_KEY + chat)
    policy = receipt.get("policy") if isinstance(receipt, dict) else None
    if not isinstance(policy, dict):
        return None
    cwd = policy.get("cwd")
    if isinstance(cwd, str) and cwd.strip():
        return cwd
    request = _json(policy.get("request_json"))
    cwd = request.get("cwd") if isinstance(request, dict) else None
    return cwd if isinstance(cwd, str) and cwd.strip() else None


def _openings(conn: sqlite3.Connection, ids: Iterable[str]) -> set[str]:
    """The opening keys of every turn of the sessions ``ids``: a turn of a later session
    opened by one of them is a copy."""
    found: set[str] = set()
    for one in ids:
        found |= {messages.opening_key(turn) for turn in messages.turns(records(conn, one))}
    return found


def select(all_turns: Sequence[messages.Turn], copied: set[str]) -> tuple[tuple[Pair, ...], int]:
    """The complete turns to import, each with its context, and how many complete
    turns were left out as copies. A copy is not imported, nor sent as context."""
    pairs: list[Pair] = []
    copies = 0
    for before, turn in messages.pending(all_turns, ()):
        if messages.opening_key(turn) in copied:
            copies += 1
            continue
        if before is not None and messages.opening_key(before) in copied:
            before = None
        pairs.append((before, turn))
    return tuple(pairs), copies


def read(home: str | Path) -> list[Session]:
    """Every session of ``<home>/state.db``, oldest first, judged and with the turns to
    import. Private mode is not looked at here (``private_reason``). RAISES
    FileNotFoundError, SchemaError, sqlite3.Error, and OSError when the marks cannot be
    read."""
    conn = connect(home)
    try:
        conn.execute("BEGIN")  # one snapshot of a store Hermes may be writing
        check_schema(conn)
        names = COLUMNS["sessions"]
        query = (
            f"SELECT {', '.join(names)}, "
            "(SELECT MAX(timestamp) FROM messages m WHERE m.session_id = s.id) "
            "FROM sessions s ORDER BY started_at, id"
        )
        out: list[Session] = []
        for values in conn.execute(query).fetchall():
            row = dict(zip(names, values[:-1], strict=True))
            out.append(_session(conn, home, row, values[-1]))
        return out
    finally:
        conn.close()


def _session(conn: sqlite3.Connection, home: Any, row: Mapping[str, Any], last: Any) -> Session:
    started = float(row["started_at"] or 0)
    model = row["model"] if isinstance(row["model"], str) and row["model"] else None
    reason, cwd, continues = judge(conn, home, row)
    base = Session(
        id=str(row["id"]),
        title=str(row["title"] or ""),
        started_at=started,
        last_at=float(last) if isinstance(last, int | float) else started,
        model=model,
        cwd=cwd,
        project=project.project_label(cwd) if cwd else None,
        reason=reason,
        continues=continues,
    )
    if reason:
        return base
    all_turns = messages.turns(records(conn, base.id))
    pairs, copies = select(all_turns, _openings(conn, continues))
    return base._replace(pairs=pairs, copies=copies, reason="" if pairs else NO_TURNS)


def private_reason(session: Session, is_private: Callable[[str], bool | None]) -> str:
    """Why ``session`` is not imported for private mode, or empty: ``is_private`` answers
    True, False, or None when that cannot be told, for the session and for each session
    whose turns it holds."""
    for one in (session.id, *session.continues):
        answer = is_private(one)
        if answer is None:
            return "its private mode could not be checked"
        if answer:
            return "private" if one == session.id else "it continues a private session"
    return ""


def payloads(session: Session, *, source: str = "hermes-import") -> list[dict[str, Any]]:
    """The ``/ingest`` bodies for ``session``'s turns, in the live path's batches and
    shape (turn_upload.TurnUploader)."""
    out: list[dict[str, Any]] = []
    for batch, _ in messages.batches(session.pairs):
        out.append(
            {
                "format": "messages",
                "client": "hermes",
                "session_id": session.id,
                "project": session.project,
                "cwd": session.cwd,
                "model": session.model,
                "source": source,
                "records": batch,
            }
        )
    return out
