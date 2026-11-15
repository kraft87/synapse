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

import re
import sqlite3
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple

from ... import messages, project
from ...shipped import Ledger, Shipped, nothing
from . import messages as hm
from . import sessions as hs
from . import store
from .lineage import chain, lineage
from .marks import marked

SCHEMA_VERSIONS = store.SCHEMA_VERSIONS
DB_TIMEOUT = store.DB_TIMEOUT
COLUMNS = store.COLUMNS

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

FOREIGN_TURN = "a turn another person or agent wrote"
SIDE_AGENT = "a side agent the gateway built (/background, /btw)"
NO_TURNS = "no finished turn of its own"
ALREADY = "every finished turn already sent"

Pair = tuple[messages.Turn | None, messages.Turn]


SchemaError = store.SchemaError


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


connect = store.connect
check_schema = store.check_schema
_message = store.message


def _rows(conn: sqlite3.Connection, session_id: str) -> list[dict[str, Any]]:
    """A session's messages in order, each uid once (store.rows)."""
    return store.rows(conn, session_id).messages


def records(conn: sqlite3.Connection, session_id: str) -> list[messages.Record]:
    """The ``/ingest`` records of a session's stored conversation, by the live path's
    conversion (hermes/messages.py ``to_records``)."""
    return hm.to_records(_rows(conn, session_id))


def _admitted_others(conn: sqlite3.Connection, session_id: str, route: Any) -> bool:
    """Whether the gateway admitted a turn into the session from anyone but the owner
    and the session's own automation (store.admitted_others)."""
    return store.admitted_others(conn, session_id, hs.owner_subject(), route)


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
    request = store.loads(policy.get("request_json"))
    cwd = request.get("cwd") if isinstance(request, dict) else None
    return cwd if isinstance(cwd, str) and cwd.strip() else None


def _openings(conn: sqlite3.Connection, ids: Iterable[str]) -> set[str]:
    """The opening keys of every turn of the sessions ``ids``: a turn of a later session
    opened by one of them is a copy."""
    found: set[str] = set()
    for one in ids:
        found |= {messages.opening_key(turn) for turn in messages.turns(records(conn, one))}
    return found


def select(
    all_turns: Sequence[messages.Turn], copied: set[str], shipped: Shipped | None = None
) -> tuple[tuple[Pair, ...], int]:
    """The complete turns to import, each with its context, and how many complete
    turns were left out as copies. A copy is not imported, nor sent as context. Rows in
    ``shipped`` (the ledger the live provider shares) are not imported again: of a turn
    shipped in part, the rest goes, as the live path sends it."""
    pairs: list[Pair] = []
    copies = 0
    shipped = shipped or nothing()
    for before, turn in messages.pending(all_turns, shipped.keys, shipped=shipped.uids):
        if messages.opening_key(turn) in copied:
            copies += 1
            continue
        if before is not None and messages.opening_key(before) in copied:
            before = None
        pairs.append((before, turn))
    return tuple(pairs), copies


def read(home: str | Path, *, ledger: Ledger | None = None) -> list[Session]:
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
            out.append(_session(conn, home, row, values[-1], ledger))
        return out
    finally:
        conn.close()


def _session(
    conn: sqlite3.Connection,
    home: Any,
    row: Mapping[str, Any],
    last: Any,
    ledger: Ledger | None = None,
) -> Session:
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
    shipped = ledger.load(base.id) if ledger is not None else nothing()
    pairs, copies = select(all_turns, _openings(conn, continues), shipped)
    reason = "" if pairs else (ALREADY if shipped.uids or shipped.keys else NO_TURNS)
    return base._replace(pairs=pairs, copies=copies, reason=reason)


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


def batches(
    session: Session, *, source: str = "hermes-import"
) -> list[tuple[dict[str, Any], list[messages.Turn]]]:
    """The ``/ingest`` bodies for ``session``'s turns, in the live path's batches and
    shape (turn_upload.TurnUploader), each with the turns it ships."""
    out: list[tuple[dict[str, Any], list[messages.Turn]]] = []
    for batch, new_turns in messages.batches(session.pairs):
        payload = {
            "format": "messages",
            "client": "hermes",
            "session_id": session.id,
            "project": session.project,
            "cwd": session.cwd,
            "model": session.model,
            "source": source,
            "records": batch,
        }
        out.append((payload, new_turns))
    return out


def payloads(session: Session, *, source: str = "hermes-import") -> list[dict[str, Any]]:
    """The ``/ingest`` bodies for ``session``'s turns (``batches``)."""
    return [payload for payload, _ in batches(session, source=source)]


def record(ledger: Ledger, session_id: str, new_turns: Sequence[messages.Turn]) -> None:
    """Write the turns of a delivered body to the ledger the live provider shares, so it
    does not send them again. RAISES OSError when it cannot be written."""
    uids = [uid for turn in new_turns for uid in messages.ids(turn)]
    keys = [messages.turn_key(turn) for turn in new_turns if not messages.ids(turn)]
    ledger.add(session_id, uids, keys)
