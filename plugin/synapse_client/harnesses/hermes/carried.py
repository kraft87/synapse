"""What is kept about a session that holds another's turns.

A session that holds another's turns (a branch, a compression that moved to a new id) is
off the record when that one is, and Hermes builds a new agent for it when it likes: the
TUI and Desktop for every branch, the gateway daemon after a restart. So the provider
settles the question when it is bound to such a session (upload.Uploader.carry_private),
and once only: a branch made before its parent went private stays on the record, and a
session put back on the record (``synapse private off --forget``) is not taken off again
because of the session it came from. This is where "once" is kept: one owner-only, empty
file per settled session under ``<hermes_home>/synapse/settled/``, named by the SHA-256
of its id. A record that could not be written does not undo it within one agent: the
uploader makes a session private this way once too (``carry_private``).

The record says the session was asked about, not how that ended. Private mode covers
whole sessions, and whether one is private is for its marker and the server's row to
say: a session put back on the record is back whole, with the turns it took over.

A settled session's own answer stands for the sessions above it: it took them into
account when it was settled. So a session is asked about up its chain only as far as the
first settled one (``to_ask``).

Clients 0.24.10 to 0.24.15 also guessed that a session holds another's turns when
nothing names the session they came from (the ACP session manager in the adapter's own
process copies a fork's turns into such a row), and kept what they found in a file of
its own under ``<hermes_home>/synapse/copied/``. Nothing finds such a session any more:
an ACP session with no gateway identity is INERT (sessions.py), and a branch names its
parent. The records already written are still read, for one more release: an agent
built for a recorded session looks back at none of its turns, and nor does a session
that goes on from it, which is noted the same way (``copies``). A record settles
nothing: a row that names a session is still asked about, since that one may be
private.

Nothing here raises. A record that cannot be written or read leaves the session to be
asked about again: it is private for as long as the session it came from is.
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...turn_upload import TurnUploader
    from ...upload import Uploader

_SETTLED = "settled"
_COPIED = "copied"


def _file(home: Any, session_id: str, kind: str) -> Path:
    digest = hashlib.sha256(session_id.encode("utf-8", "replace")).hexdigest()
    return Path(str(home)) / "synapse" / kind / digest


def _noted(home: Any, session_id: str, kind: str) -> bool:
    """Whether the session has a record of ``kind``. False when there is none, when it
    cannot be read, and without a home or an id."""
    if not home or not session_id:
        return False
    try:
        os.lstat(_file(home, session_id, kind))
    except (OSError, ValueError):
        return False
    return True


def _note(home: Any, session_id: str, kind: str) -> None:
    """Write the session's record of ``kind``, in ``<home>/synapse/<kind>``, owner-only.
    Without a home or an id, and when the record cannot be written, nothing is kept."""
    if not home or not session_id:
        return
    try:
        folder = Path(str(home))
        for part in ("synapse", kind):
            folder = folder / part
            if not folder.is_dir():
                folder.mkdir(mode=0o700, exist_ok=True)
        os.close(os.open(_file(home, session_id, kind), os.O_WRONLY | os.O_CREAT, 0o600))
    except (OSError, ValueError):
        return


def settled(home: Any, session_id: str) -> bool:
    """Whether ``session_id`` was settled under ``home``. False when that cannot be
    read, and without a home or an id."""
    return _noted(home, session_id, _SETTLED)


def settle(home: Any, session_id: str) -> None:
    """Record that ``session_id`` is settled: it is not asked about again."""
    _note(home, session_id, _SETTLED)


def to_ask(home: Any, chain: Iterable[str]) -> list[str]:
    """The sessions of ``chain`` (the ones a session holds turns of, nearest first) to
    go by: as far up as the first settled one, which answers for those above it."""
    ask: list[str] = []
    for session_id in chain:
        ask.append(session_id)
        if settled(home, session_id):
            break
    return ask


def copied(home: Any, session_id: str) -> bool:
    """Whether ``session_id`` is recorded as holding turns copied from a session nothing
    names (by a client before 0.24.16, or ``copies``). False when that cannot be read,
    and without a home or an id."""
    return _noted(home, session_id, _COPIED)


def note_copied(home: Any, session_id: str) -> None:
    """Record that ``session_id`` holds turns copied from a session nothing names."""
    _note(home, session_id, _COPIED)


def copies(home: Any, session_id: Any, earlier: Iterable[Any] = ()) -> bool:
    """Whether ``session_id`` holds copies, so that none of its turns from before is
    looked back at: it is recorded so (``copied``), or a session of ``earlier``, whose
    turns it holds, is. It is noted then as one that does, for an agent built later
    and for the sessions that go on from it."""
    sid = str(session_id or "")
    if copied(home, sid):
        return True
    if not sid or not any(copied(home, str(one)) for one in earlier if one):
        return False
    note_copied(home, sid)
    return True


def carry(
    uploader: Uploader, home: Any, earlier: Iterable[Any], session_id: Any, timeout: float | None
) -> bool:
    """Keep ``session_id`` off the record when a session of ``earlier``, whose turns it
    holds, is (Uploader.carry_private: it gets the marker and the server's row). With
    no marker to say one of them is private, the server is asked at the session's first
    upload. Once per session of ``home``: a settled one is left as it is, and so is one
    with no session in ``earlier``. With ``timeout`` None no request is made here, and
    telling the server is left to the caller or that upload. True when a marker made
    ``session_id`` private here: in this call, or for this uploader before it."""
    sid = str(session_id or "")
    ids = [str(one) for one in earlier if one]
    if not sid or not ids or settled(home, sid):
        return False
    return uploader.carry_private(ids, sid, timeout=timeout, settled=lambda _: settle(home, sid))


class Followed:
    """The sessions one agent uploaded for whose row was read for the ones it goes on
    from (``follow``), once each. Safe across threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ids: set[str] = set()

    def only(self, *session_ids: str) -> None:
        """Forget every id but ``session_ids``, which are not read: ``initialize`` has
        carried to the one it was given."""
        with self._lock:
            self._ids = set(session_ids)

    def follow(
        self,
        uploader: Uploader,
        turns: TurnUploader,
        home: Any,
        session_id: str,
        continues: tuple[str, ...],
        lookback: int,
    ) -> None:
        """Before an upload for ``session_id``: keep it off the record when a session
        its row says it holds turns of (``continues``) is, once per id and agent.
        ``initialize`` does this for the id it is given. This is for the others: a
        session resumed in place, one Hermes moved the agent to without a call, and a
        row written after ``initialize``. Its next upload looks back ``lookback`` turns,
        and none when it holds copies. Noted only once done: a carry that raises, and a
        row that names no session yet, leave the id to be read at the next upload."""
        with self._lock:
            if session_id in self._ids:
                return
        if not continues:
            if copied(home, session_id):
                turns.keep_out(session_id)  # recorded as holding copies: no look-back
            return
        carry(uploader, home, to_ask(home, continues), session_id, None)
        holds = copies(home, session_id, continues)  # no look-back
        turns.start(session_id, 0 if holds else lookback, continues=continues)
        with self._lock:
            self._ids.add(session_id)
