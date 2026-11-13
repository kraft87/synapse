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

``continues`` names the one session whose turns a row holds: the session it was branched
from, or the one it goes on from after a compression. A reset is a new conversation and
a subagent's is its own, so neither holds its parent's turns. ``chain`` follows that up
the store: a branch of a branch holds the turns of both.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, NamedTuple

#: How many parents up a session's lineage is read.
LINEAGE_LIMIT = 64
_ORIGINS = ("_branched_from", "_reset_from", "_compressed_from")
#: The ``model_config`` keys of a child that holds none of its parent's turns: a reset
#: and a subagent (tools/delegate_tool.py).
_OWN_CONVERSATION = ("_reset_from", "_delegate_from")


class Parent(NamedTuple):
    """A session another descends from. ``title`` and ``model_config`` are None, and
    ``chat`` empty, when its row is gone."""

    id: str
    title: Any
    model_config: Any
    chat: str


def _config(model_config: Any) -> dict[str, Any]:
    try:
        config = json.loads(model_config) if isinstance(model_config, str) else None
    except ValueError:
        return {}  # Hermes reads an unparseable model_config as empty too
    return config if isinstance(config, dict) else {}


def origin(parent: Any, model_config: Any) -> str | None:
    """The session a row came from: its parent link, else the one its ``model_config``
    still names."""
    if isinstance(parent, str) and parent:
        return parent
    config = _config(model_config)
    for key in _ORIGINS:
        named = config.get(key)
        if isinstance(named, str) and named:
            return named
    return None


def continues(parent: Any, model_config: Any) -> str | None:
    """The session whose turns a row holds, from its ``parent_session_id`` and
    ``model_config``: the one it was branched from (a copy of its turns) or goes on from
    after a compression. None for a row that holds no other session's turns: a first
    session, a reset, a subagent.

    How a key is read is Hermes's (hermes_state_common.py
    ``_non_continuation_child_sql``): a key in ``model_config`` speaks of the parent only
    when it names it, since a continuation inherits its parent's ``model_config`` whole,
    the keys of a reset or a branch further up included. The rule is not Hermes's. That
    one asks whether a child is the same session going on after a compression, so it
    turns a branch away, and a child whose ``source`` is ``tool``. This one asks whose
    turns a row holds: a branch holds a copy of its parent's, and ``source`` is not read,
    so such a child counts as holding them, which errs on the side of private.

    With the parent link gone (Hermes clears it when that session is deleted) the copy is
    still there, so the session ``model_config`` names as branched or compressed from is
    the answer. A row from before Hermes wrote ``_reset_from`` reads as a continuation."""
    config = _config(model_config)
    if isinstance(parent, str) and parent:
        if any(config.get(key) == parent for key in _OWN_CONVERSATION):
            return None
        return parent
    for key in ("_compressed_from", "_branched_from"):
        named = config.get(key)
        if isinstance(named, str) and named:
            return named
    return None


def chain(conn: sqlite3.Connection, parent: Any, model_config: Any) -> tuple[str, ...]:
    """The sessions whose turns a row holds, nearest first: the one it ``continues``, the
    one that one continues, and so on while the store has their rows. RAISES
    sqlite3.Error."""
    found: list[str] = []
    query = "SELECT parent_session_id, model_config FROM sessions WHERE id=?"
    named = continues(parent, model_config)
    while named and named not in found and len(found) < LINEAGE_LIMIT:
        found.append(named)
        row = conn.execute(query, (named,)).fetchone()
        named = continues(row[0], row[1]) if row is not None else None
    return tuple(found)


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
