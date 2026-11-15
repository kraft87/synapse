"""The upload half of the Hermes provider: whose turns a hook's upload is for, where its
rows come from, and the send.

``Uploads`` is a mixin of provider.SynapseMemoryProvider and reads its state (the bound
session, the turn uploader, the home, the gateway identity). An upload reads the
session's rows from Hermes's store up to the row the hook named (source.py) and hands
them to the core (turn_upload.TurnUploader); the list Hermes handed over stands in when
the store cannot serve, and the reason is logged once.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from ...messages import Record
from ...turn_upload import TurnUploader
from ...upload import Uploader
from . import carried, source
from . import sessions as hs
from .live import Live

#: How many complete turns a session sends at most on its first upload.
LOOKBACK = 10


def _log() -> logging.Logger:
    return logging.getLogger(__name__)


def _warn(line: str) -> None:
    _log().warning("%s", line)


class Uploads:
    """Uploads for one provider instance (see the module notes)."""

    _lock: threading.Lock
    _said: set[str]
    _home: str | None
    _session_id: str
    _gateway: bool
    _owner_id: str | None
    _route: str | None
    _project: str | None
    _cwd: str | None
    _turns: TurnUploader | None
    _uploader: Uploader | None
    _followed: carried.Followed
    _live: Live

    def _verdict(self, session_id: str = "") -> hs.Verdict | None:  # provider.py
        raise NotImplementedError

    def _authored(self) -> None:  # provider.py
        raise NotImplementedError

    def _whose(self, session_id: str) -> str | None:
        """The session a hook's turns are for, settled as the hook runs: a turn upload
        is sent later, and Hermes may have moved the agent by then. The id is the one
        Hermes names, which need not be one this instance was moved to. With none named
        (a flush) it is the last one Hermes named (live.py), when that is the bound id
        or its row says it goes on from it. Otherwise nothing says whose turns they are
        (after ``/new`` Hermes flushes the session left once the next turn has named
        the new one): None, and nothing is sent."""
        sid = self._live.named(str(session_id or ""))
        if session_id:
            return sid
        verdict = self._verdict(sid) if sid else None
        with self._lock:
            bound = self._session_id
        if verdict is None or (sid != bound and bound not in verdict.continues):
            return None
        return sid

    def _send(
        self,
        sid: str,
        end: str | None,
        listed: list[Record],
        timeout: float,
        model: str,
        finished: bool = False,
    ) -> bool:
        """Send the complete turns of session ``sid`` not shipped yet, up to the row
        ``end``: read from the store (``_stored``), or ``listed`` when it cannot be
        (turn_upload.TurnUploader). True when at least one batch reached the server.
        ``finished``: a turn has just ended. The session's row is read as this runs: is
        it still the owner's, and what does it continue (carried.Followed)."""
        turns, uploader = self._turns, self._uploader
        verdict = self._verdict(sid) if turns is not None and uploader is not None else None
        if turns is None or uploader is None or verdict is None:
            return False
        self._followed.follow(uploader, turns, self._home, sid, verdict.continues, LOOKBACK)
        records = self._stored(sid, end, verdict.continues, bool(listed))
        if records is None:
            return False
        if isinstance(records, str):
            records = listed
        about = {"project": self._project, "cwd": self._cwd, "model": model or None}
        return turns.upload(records, sid, timeout=timeout, finished=finished, **about)

    def _stored(
        self, sid: str, end: str | None, continues: tuple[str, ...], listed: bool = True
    ) -> Any:
        """The records of ``sid`` from the store up to the row ``end`` (source.read): a
        list; a string saying why the store could not serve them (said and kept for
        ``synapse status`` when a ``listed`` list stands in); None for a turn someone
        else started, which turns this instance INERT."""
        got = source.read(
            self._home,
            sid,
            end,
            continues,
            gateway=self._gateway,
            owner=self._owner_id,
            route=self._route,
        )
        if got.foreign:
            self._authored()
            return None
        if got.records is None and not listed:
            return got.reason
        if got.records is None:
            line = f"synapse: uploads read Hermes's live list, not its store: {got.reason}"
            with self._lock:
                first = line not in self._said
                self._said.add(line)
            if first:
                _warn(line)
            source.fell_back(self._home, got.reason)
            return got.reason
        if self._turns is not None:
            self._turns.note_copies(sid, got.copied)
        source.fell_back(self._home, None)
        return got.records

    def _drain(self, timeout: float) -> None:
        uploader = self._uploader
        if uploader is None:
            return
        try:
            uploader.drain(timeout)
        except Exception:
            _log().warning("synapse: outbox drain failed", exc_info=True)
