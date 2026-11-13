"""The sessions whose private lineage has been settled.

A session that holds another's turns (a branch, a compression that moved to a new id) is
off the record when that one is, and Hermes builds a new agent for it when it likes: the
TUI and Desktop for every branch, the gateway daemon after a restart. So the provider
settles the question when it is bound to such a session (upload.Uploader.carry_private),
and once only: a branch made before its parent went private stays on the record, and a
session put back on the record (``synapse private off --forget``) is not taken off again
because of the session it came from. This is where "once" is kept: one owner-only file
per settled session under ``<hermes_home>/synapse/settled/``, named by the SHA-256 of
its id. A record that could not be written does not undo it within one agent: the
uploader makes a session private this way once too (``carry_private``).

The record also says how the question ended. A session that was made private holds
turns spoken off the record: the ones it took over, and its own until it is put back.
Its file is not empty, and that outlives the marker and the server's row: an agent
built for the session later, also after ``off --forget``, does not look back at them
(``held``). A session that holds the turns of such a one holds those turns too, and its
record says the same, whether or not it was made private.

A settled session's own answer stands for the sessions above it: it took them into
account when it was settled. So a session is asked about up its chain only as far as the
first settled one (``to_ask``).

A session whose row names no session has nothing to be asked about, and can hold
another's turns all the same: Hermes copies them into an ACP fork and records no
origin. The uploader finds that out from the fork's first turn (turn_upload: a session
said to be new that holds turns from before), in memory. ``note_copied`` keeps it, in a
file of its own under ``<hermes_home>/synapse/copied/``: an agent built for the session
later looks back at none of its turns either, and a session that goes on from it holds
turns from off the record as far as anyone can tell (``carry``). It settles nothing: a
row that names a session later is still asked about, since that one may be private. And
it stands whatever is settled: the copies stay out also when the answer is "no".

Nothing here raises. A record that cannot be written or read leaves the session to be
asked about again: it is private for as long as the session it came from is, and an
agent built once that one is back on the record looks back at its turns, as does one
built for a session whose copies could not be recorded. One that holds turns from off
the record is written whole or not at all (a file beside it, moved into place).
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...upload import Uploader

_SETTLED = "settled"
_COPIED = "copied"


def _file(home: Any, session_id: str, kind: str = _SETTLED) -> Path:
    digest = hashlib.sha256(session_id.encode("utf-8", "replace")).hexdigest()
    return Path(str(home)) / "synapse" / kind / digest


def _folder(home: Any, kind: str) -> None:
    """Make ``<home>/synapse/<kind>``, owner-only. RAISES what the file system does."""
    folder = Path(str(home))
    for part in ("synapse", kind):
        folder = folder / part
        if not folder.is_dir():
            folder.mkdir(mode=0o700, exist_ok=True)


def _size(home: Any, session_id: str, kind: str = _SETTLED) -> int:
    """The size of the session's record of ``kind``; -1 when there is none, when it
    cannot be read, and without a home or an id."""
    if not home or not session_id:
        return -1
    try:
        return os.lstat(_file(home, session_id, kind)).st_size
    except (OSError, ValueError):
        return -1


def settled(home: Any, session_id: str) -> bool:
    """Whether ``session_id`` was settled under ``home``. False when that cannot be
    read, and without a home or an id."""
    return _size(home, session_id) >= 0


def to_ask(home: Any, chain: Iterable[str]) -> list[str]:
    """The sessions of ``chain`` (the ones a session holds turns of, nearest first) to
    go by: as far up as the first settled one, which answers for those above it."""
    ask: list[str] = []
    for session_id in chain:
        ask.append(session_id)
        if settled(home, session_id):
            break
    return ask


def held(home: Any, session_id: str) -> bool:
    """Whether ``session_id`` was settled as one made private because a session it
    holds turns of was: it holds turns spoken off the record, whatever it is now.
    False when that cannot be read, and without a home or an id."""
    return _size(home, session_id) > 0


def copied(home: Any, session_id: str) -> bool:
    """Whether ``session_id`` was found to hold turns copied from a session nothing
    names (``note_copied``). False when that cannot be read, and without a home or an
    id."""
    return _size(home, session_id, _COPIED) >= 0


def note_copied(home: Any, session_id: str) -> None:
    """Record that ``session_id`` holds turns copied from a session nothing names.
    Without a home or an id, and when the record cannot be written, nothing is kept."""
    if not home or not session_id:
        return
    try:
        _folder(home, _COPIED)
        os.close(os.open(_file(home, session_id, _COPIED), os.O_WRONLY | os.O_CREAT, 0o600))
    except (OSError, ValueError):
        return


def settle(home: Any, session_id: str, made: bool = False) -> None:
    """Record that ``session_id`` is settled, and with ``made`` that it was made
    private (``held``); a record that says so keeps saying it. Without a home or an
    id, and when the record cannot be written, nothing is kept."""
    if not home or not session_id:
        return
    try:
        _folder(home, _SETTLED)
        path = _file(home, session_id)
        if not made:
            os.close(os.open(path, os.O_WRONLY | os.O_CREAT, 0o600))
            return
        beside = path.with_name(path.name + ".new")
        fd = os.open(beside, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, b"held\n")
        finally:
            os.close(fd)
        os.replace(beside, path)
    except (OSError, ValueError):
        return


def carry(
    uploader: Uploader, home: Any, earlier: Iterable[Any], session_id: Any, timeout: float | None
) -> bool:
    """Whether ``session_id`` holds turns spoken off the record because of a session of
    ``earlier``, whose turns it holds: that one is private (Uploader.carry_private:
    ``session_id`` is made private too), or it holds such turns itself, also when it is
    back on the record (``held``, ``copied``: ``session_id`` is left on the record), or
    ``session_id`` was found to hold copies itself. With no marker to say a session of
    ``earlier`` is private the server is asked at the session's first upload, and that
    part of the answer is False for now. Once per session of ``home``: a settled one is
    left as it is, and the answer is what its records say. With no session in
    ``earlier`` nothing is asked or settled, and the answer is whether ``session_id``
    was found to hold copies. With ``timeout`` None no request is made here, and telling
    the server is left to the caller or that upload."""
    sid = str(session_id or "")
    own = copied(home, sid)
    if settled(home, sid):
        return own or held(home, sid)
    ids = [str(one) for one in earlier if one]
    if not sid or not ids:
        return own
    passed = own or any(held(home, one) or copied(home, one) for one in ids)
    made = uploader.carry_private(
        ids, sid, timeout=timeout, settled=lambda made: settle(home, sid, made or passed)
    )
    return made or passed
