"""What a Hermes upload takes from the session store besides the list Hermes hands over.

The rows an upload sends come from the list Hermes hands a memory provider: it is the
run as Hermes ran it (a prompt queued during a run sits where it was answered, not where
it was accepted), it holds the rows Hermes could not write, and Hermes scrubbed every
string in it of secrets before the hand-off (agent/memory_manager.py
``_redact_for_provider``), which the store's rows are not. The store gives two things
the list does not always hold, and neither is text:

- each row's ``display_kind``, by ``message_uid``: the list loses it after Hermes
  rebuilds the agent (a model switch) and when it folds a marker into the next message;
- who started each turn the gateway ran (``session_admissions``): for a gateway session,
  a turn admitted from anyone but the owner and the session's own automation means
  nothing more is uploaded (``foreign``).

It is also a check: a list row whose uid the store holds under another role is named
(``mismatched``), and the caller logs it once; the row goes as the list has it, as in
the released client. When the store cannot be read (no home, no ``state.db``, an error,
a schema store.py does not know), or does not hold a row (Hermes could not write it),
the list goes up as it is, with the kinds it holds, as in the released client; the
reason is logged once by the caller.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from typing import Any, NamedTuple

from . import store


class Attributes(NamedTuple):
    """What ``read`` found: the stored ``display_kind`` by uid, the uids whose stored
    role is not the list's (``mismatched``), whether a turn of the session was admitted
    from someone else, and why the store was not read (empty when it was)."""

    kinds: dict[str, str]
    foreign: bool = False
    reason: str = ""
    mismatched: tuple[str, ...] = ()


def uids(messages: Iterable[Any]) -> list[str]:
    """The ``message_uid`` of every row of ``messages`` that has one."""
    out = []
    for message in messages or ():
        uid = message.get("message_uid") if isinstance(message, Mapping) else None
        if isinstance(uid, str) and uid:
            out.append(uid)
    return out


def read(
    home: Any,
    session_id: str,
    wanted: Iterable[Any],
    *,
    gateway: bool = False,
    owner: str | None = None,
    route: str | None = None,
) -> Attributes:
    """The stored kinds of the rows ``wanted`` (the list's rows, or their uids), the
    rows whose stored role differs from the list's, and the admissions check. A row the
    store does not hold (Hermes could not write it) is not a mismatch: it goes as the
    list has it. Never raises."""
    given = list(wanted)
    rows = [w for w in given if isinstance(w, Mapping)]
    wanted = uids(rows) if rows else [w for w in given if isinstance(w, str)]
    if not home:
        return Attributes({}, reason="no Hermes home was named")
    try:
        conn = store.connect(home)
    except FileNotFoundError:
        return Attributes({}, reason="there is no state.db")
    except (sqlite3.Error, OSError, ValueError) as err:
        return Attributes({}, reason=f"state.db could not be opened ({type(err).__name__})")
    try:
        conn.execute("BEGIN")
        store.check_schema(conn)
        if gateway and store.admitted_others(conn, session_id, owner, route):
            return Attributes({}, foreign=True)
        stored = store.kinds(conn, wanted)
    except store.SchemaError as err:
        return Attributes({}, reason=str(err))
    except (sqlite3.Error, OSError, ValueError) as err:
        return Attributes({}, reason=f"state.db could not be read ({type(err).__name__})")
    finally:
        conn.close()
    kinds = {uid: kind for uid, (_, kind) in stored.items() if kind}
    # A compaction's summary takes the uid of a row it replaces, whatever that row's role.
    mismatched = tuple(
        uid
        for row in rows
        if not row.get("_compressed_summary")
        and (uid := row.get("message_uid")) in stored
        and stored[uid][0] != row.get("role")
    )
    return Attributes(kinds, mismatched=mismatched)
