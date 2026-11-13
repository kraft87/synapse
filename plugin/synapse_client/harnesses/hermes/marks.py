"""The chats others have written into: marks that outlast a title and an agent.

A Bot Chat that is archived or renamed leaves rows that read as an owner's chat, with
other agents' turns still in them, and Hermes builds a new agent for a session when it
likes. So the first time a session is judged a Bot Chat, a Group Chat member or a chat
that follows its profile (sessions.py), or is handed a turn someone else wrote (the
provider's ``_authored``), it is marked, along with the session or chat that said so:
one owner-only file each under ``<hermes_home>/synapse/inert/``, named by the SHA-256 of
the id and holding the id and the reason. sessions.py reads the marks of a session, of
its chats and of the sessions it descends from, and copies a parent's mark down to the
session that inherits it, since Hermes forgets the parent when its row is deleted.

Marks are only added. A session comes back on only when every file that names it, its
chat and the sessions it descends from is deleted, and the rows no longer say what was
marked. A chat that lost its title before this provider ever judged it has no mark. A
mark that cannot be written is logged and lost; marks that cannot be read stop the
provider (``marked`` raises).
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_MARK_DIR = ("synapse", "inert")


def _mark_file(home: Any, key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8", "replace")).hexdigest()
    return Path(str(home)).joinpath(*_MARK_DIR, digest)


def mark(home: Any, keys: Iterable[Any], reason: str) -> None:
    """Record for good that others write into the sessions and chats named by ``keys``:
    one owner-only file per key under ``<home>/synapse/inert/``, named by the key's
    SHA-256 and holding the key and ``reason``. Never raises; without a home nothing is
    kept, and a mark that cannot be written is logged."""
    if not home:
        return
    try:
        folder = Path(str(home))
        for part in _MARK_DIR:
            folder = folder / part
            if not folder.is_dir():
                folder.mkdir(mode=0o700, exist_ok=True)
        for key in {str(key) for key in keys if key}:
            try:
                fd = os.open(_mark_file(home, key), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                continue
            with os.fdopen(fd, "w", encoding="utf-8", errors="replace") as handle:
                handle.write(f"{key}\n{reason}\n")
    except OSError as err:
        logging.getLogger(__name__).warning(
            "synapse: could not note a chat others write into (%s); it is off for now only",
            type(err).__name__,
        )


def marked(home: Any, keys: Iterable[Any]) -> bool:
    """Whether any session or chat of ``keys`` carries a mark under ``home``. RAISES
    OSError when a mark cannot be looked up for any reason but its file not being
    there: the caller must not take that for "not marked"."""
    if not home:
        return False
    for key in keys:
        if not key:
            continue
        try:
            os.stat(_mark_file(home, str(key)))
        except FileNotFoundError:
            continue
        return True
    return False
