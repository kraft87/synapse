"""Where a Hermes upload's rows come from.

Hermes writes a turn's rows to its session store before it calls ``sync_turn``
(agent/turn_finalizer.py persists, then mirrors the turn to memory providers), so an
upload reads them there (``read``): the session's rows up to the last row of the list
Hermes handed over (``end_uid``), and no further, since the next run's rows are written
as it goes (its user row as it is submitted, its tool rows as they come). The store
keeps what the list does not always hold: each row's ``display_kind`` after Hermes
rebuilt the agent, the marker and the message after it as two rows, and the rows a
compaction archived.

The list stands in when the store cannot serve: there is no home or no ``state.db``, it
cannot be opened or read, its schema is not one store.py knows (a newer Hermes), or the
row the hook names is not in it (Hermes failed to write the turn, or the list holds no
stored row). ``fell_back`` keeps the reason in ``<hermes_home>/synapse/store_fallback.json``
for ``synapse status``, and clears it when the store serves again.

The store is the better witness of who started a turn, too: for a session the gateway
runs, a turn admitted from anyone but the owner and the session's own automation
(``foreign``) means nothing is uploaded and the provider turns INERT. And the rows a
session holds of the sessions it goes on from (a branch, a compressed continuation) are
known by uid (``copied``), so they are never uploaded under the new id.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

from ...messages import Record
from . import messages as hm
from . import store

#: Where the reason the last upload read the live list, not the store, is kept.
FALLBACK_FILE = "store_fallback.json"


class Got(NamedTuple):
    """What ``read`` found: the records (None when the list must stand in, with
    ``reason``), the row uids the session holds of earlier sessions, and whether a
    turn of the session was admitted from someone else."""

    records: list[Record] | None
    reason: str = ""
    copied: frozenset[str] = frozenset()
    foreign: bool = False


def end_uid(messages: Any) -> str | None:
    """The ``message_uid`` of the last row of ``messages`` that has one: the last row
    Hermes wrote of the run a hook reports."""
    for message in reversed(messages or ()):
        uid = message.get("message_uid") if isinstance(message, Mapping) else None
        if isinstance(uid, str) and uid:
            return uid
    return None


def read(
    home: Any,
    session_id: str,
    end: str | None,
    continues: Iterable[str] = (),
    *,
    gateway: bool = False,
    owner: str | None = None,
    route: str | None = None,
) -> Got:
    """The records of ``session_id`` read from ``<home>/state.db``, up to the row
    ``end``. Never raises."""
    if not home:
        return Got(None, "no Hermes home was named")
    if end is None:
        return Got(None, "Hermes named no stored row for the turn")
    try:
        conn = store.connect(home)
    except FileNotFoundError:
        return Got(None, "there is no state.db")
    except (sqlite3.Error, OSError, ValueError) as err:
        return Got(None, f"state.db could not be opened ({type(err).__name__})")
    try:
        conn.execute("BEGIN")  # one snapshot of a store Hermes may be writing
        store.check_schema(conn)
        if gateway and store.admitted_others(conn, session_id, owner, route):
            return Got(None, "", foreign=True)
        got = store.rows(conn, session_id, until=end)
        copied = store.uids(conn, [one for one in continues if one and one != session_id])
    except store.SchemaError as err:
        return Got(None, str(err))
    except (sqlite3.Error, OSError, ValueError) as err:
        return Got(None, f"state.db could not be read ({type(err).__name__})")
    finally:
        conn.close()
    if not got.reached:
        return Got(None, "the turn's last row is not in state.db")
    return Got(hm.to_records(got.messages), copied=frozenset(copied))


def last_uid(home: Any, session_id: str) -> str | None:
    """The ``message_uid`` of the newest row of ``session_id`` in the store, or None
    when there is none or the store cannot be read. Never raises."""
    if not home or not session_id:
        return None
    try:
        conn = store.connect(home)
    except (FileNotFoundError, sqlite3.Error, OSError, ValueError):
        return None
    try:
        row = conn.execute(
            "SELECT message_uid FROM messages WHERE session_id=? AND message_uid IS NOT NULL "
            "AND (active=1 OR compacted=1) ORDER BY id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    return row[0] if row and isinstance(row[0], str) and row[0] else None


def fell_back(home: Any, reason: str | None) -> None:
    """Keep ``reason`` (the list stood in) for ``synapse status``; with None, the store
    served and the reason is cleared. Never raises."""
    if not home:
        return
    file = Path(str(home)) / "synapse" / FALLBACK_FILE
    try:
        if reason is None:
            if file.exists():
                os.remove(file)
            return
        file.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now(UTC).isoformat()
        file.write_text(json.dumps({"reason": reason, "at": now}), encoding="utf-8")
    except OSError:
        pass


def fallback_reason(home: Any) -> dict[str, Any] | None:
    """Why the last upload of ``home`` read the live list, or None. Never raises."""
    try:
        data = json.loads((Path(str(home)) / "synapse" / FALLBACK_FILE).read_text("utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None
