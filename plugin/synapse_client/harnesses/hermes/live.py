"""Which session the agent is in, for a hook that does not say.

Hermes names a session id in ``initialize``, ``on_session_switch``, ``prefetch`` and
``sync_turn``. ``on_session_end`` and ``on_pre_compress`` carry none, and the id the
provider was built for or last switched to need not be the agent's: Hermes points an
agent at another session without a call when another holder's compression started one
(agent/turn_facade_lease.py, agent/session_persistence.py). A flush under the old id
would upload the new session's turns as the old one's, past the new one's private mode.
So the provider notes each id a hook names, and a flush goes by the last, when the
provider can tell the turns are that session's (provider.py, ``_upload``).

Two kinds of hook name an id. ``initialize``, ``on_session_switch`` and ``prefetch`` run
as the agent is in that session (``moved``). ``sync_turn`` runs on Hermes's background
worker, after its turn, and so possibly after a switch made since (``named``): an id the
agent was moved away from is not gone back to on its word: the id ``moved`` found it
in, and the parent a switch names, which the agent was in though no hook need have
said so.

Still unseen: a move Hermes did not announce, until a hook names the new id. A flush
that runs before then (the turn's prompt gave no prefetch, and its ``sync_turn`` is still
queued) goes under the id from before.
"""

from __future__ import annotations

import threading


class Live:
    """The id of the session the agent is in, as far as Hermes has said. Safe across
    threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._id = ""
        self._left: set[str] = set()

    @property
    def id(self) -> str:
        with self._lock:
            return self._id

    def moved(self, session_id: str, left: tuple[str, ...] = ()) -> None:
        """The agent is in ``session_id`` now, and no longer in the one before, nor in
        one of ``left`` (the ids a switch says it leaves)."""
        with self._lock:
            if not session_id:
                return
            self._left.update(one for one in left if one and one != session_id)
            if session_id != self._id:
                if self._id:
                    self._left.add(self._id)
                self._left.discard(session_id)
                self._id = session_id

    def named(self, session_id: str) -> str:
        """A hook that may run late names ``session_id``: the agent is in it, unless it
        was moved away from it since. Returns the id the hook is for: the one it names,
        or with none the one the agent is in."""
        with self._lock:
            if session_id and session_id not in self._left:
                self._id = session_id
            return session_id or self._id
