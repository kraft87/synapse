"""The sessions whose private lineage has been settled.

A session that holds another's turns (a branch, a compression that moved to a new id) is
off the record when that one is, and Hermes builds a new agent for it when it likes: the
TUI and Desktop for every branch, the gateway daemon after a restart. So the provider
settles the question when it is bound to such a session (upload.Uploader.carry_private),
and once only: a branch made before its parent went private stays on the record, and a
session put back on the record (``synapse private off --forget``) is not taken off again
because of the session it came from. This is where "once" is kept: one empty owner-only
file per settled session under ``<hermes_home>/synapse/settled/``, named by the SHA-256
of its id. A record that could not be written does not undo it within one agent: the
uploader makes a session private this way once too (``carry_private``).

A settled session's own answer stands for the sessions above it: it took them into
account when it was settled. So a session is asked about up its chain only as far as the
first settled one (``to_ask``).

Nothing here raises. A record that cannot be written or read leaves the session to be
asked about again, which errs on the side of private.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...upload import Uploader

_DIR = ("synapse", "settled")


def _file(home: Any, session_id: str) -> Path:
    digest = hashlib.sha256(session_id.encode("utf-8", "replace")).hexdigest()
    return Path(str(home)).joinpath(*_DIR, digest)


def settled(home: Any, session_id: str) -> bool:
    """Whether ``session_id`` was settled under ``home``. False when that cannot be
    read, and without a home or an id."""
    if not home or not session_id:
        return False
    try:
        os.lstat(_file(home, session_id))
    except (OSError, ValueError):
        return False
    return True


def to_ask(home: Any, chain: Iterable[str]) -> list[str]:
    """The sessions of ``chain`` (the ones a session holds turns of, nearest first) to
    go by: as far up as the first settled one, which answers for those above it."""
    ask: list[str] = []
    for session_id in chain:
        ask.append(session_id)
        if settled(home, session_id):
            break
    return ask


def settle(home: Any, session_id: str) -> None:
    """Record that ``session_id`` is settled. Without a home or an id, and when the
    record cannot be written, nothing is kept."""
    if not home or not session_id:
        return
    try:
        folder = Path(str(home))
        for part in _DIR:
            folder = folder / part
            if not folder.is_dir():
                folder.mkdir(mode=0o700, exist_ok=True)
        os.close(os.open(_file(home, session_id), os.O_WRONLY | os.O_CREAT, 0o600))
    except (OSError, ValueError):
        return


def carry(
    uploader: Uploader, home: Any, earlier: Iterable[Any], session_id: Any, timeout: float | None
) -> bool:
    """Whether ``session_id`` is off the record because a session of ``earlier``, whose
    turns it holds, is (Uploader.carry_private). With no marker to say so the server is
    asked at the session's first upload, and the answer is False for now. Once per
    session of ``home``: a settled one is left as it is. With ``timeout`` None no request
    is made here, and telling the server is left to the caller or that upload."""
    sid = str(session_id or "")
    ids = [str(one) for one in earlier if one]
    if not sid or not ids or settled(home, sid):
        return False
    return uploader.carry_private(ids, sid, timeout=timeout, settled=lambda: settle(home, sid))
