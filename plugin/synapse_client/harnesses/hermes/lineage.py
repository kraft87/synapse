"""The sessions a session descends from.

A branch is a new chat holding a copy of its parent's turns
(hermes_state_mutation_branch.py ``branch_in_transaction``); a reset or compressed
continuation is a new row of the same chat (hermes_state_mutation_reset.py,
hermes_state_mutation_compress.py). The gateway's writers name the session a row came
from twice: in ``parent_session_id``, which Hermes clears when that session is deleted
or pruned (hermes_state_sessions.py, hermes_state_mutation_retirement.py,
hermes_state_maintenance.py), and in its ``model_config`` (``_branched_from``,
``_reset_from``, ``_compressed_from``), which stays. Both are read, so such a row is
still known as a copy after its parent's row is gone: the id it names is kept as a last
parent, with nothing but its mark (marks.py) to go by. An in-process compression
(agent/conversation_compression.py) writes the link alone, so its child is known only
while the parent's row is there, or by the mark sessions.py copied down to it.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, NamedTuple

#: How many parents up a session's lineage is read.
LINEAGE_LIMIT = 64
_ORIGINS = ("_branched_from", "_reset_from", "_compressed_from")


class Parent(NamedTuple):
    """A session another descends from. ``title`` and ``model_config`` are None, and
    ``chat`` empty, when its row is gone."""

    id: str
    title: Any
    model_config: Any
    chat: str


def origin(parent: Any, model_config: Any) -> str | None:
    """The session a row came from: its parent link, else the one its ``model_config``
    still names."""
    if isinstance(parent, str) and parent:
        return parent
    try:
        config = json.loads(model_config) if isinstance(model_config, str) else None
    except ValueError:
        return None  # Hermes reads an unparseable model_config as empty too
    if isinstance(config, dict):
        for key in _ORIGINS:
            named = config.get(key)
            if isinstance(named, str) and named:
                return named
    return None


def lineage(conn: sqlite3.Connection, parent: Any, model_config: Any) -> list[Parent]:
    """A session's parents, nearest first, from its row's ``parent_session_id`` and
    ``model_config``. RAISES sqlite3.Error."""
    found: list[Parent] = []
    seen: set[str] = set()
    query = "SELECT title, model_config, chat_id, parent_session_id FROM sessions WHERE id=?"
    named = origin(parent, model_config)
    while named and named not in seen and len(found) < LINEAGE_LIMIT:
        seen.add(named)
        row = conn.execute(query, (named,)).fetchone()
        if row is None:
            found.append(Parent(named, None, None, ""))
            break
        found.append(Parent(named, row[0], row[1], str(row[2] or "")))
        named = origin(row[3], row[1])
    return found
