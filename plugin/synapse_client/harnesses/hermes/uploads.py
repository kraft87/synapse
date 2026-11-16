"""The upload half of the Hermes provider: whose turns a hook's upload is for, and the
send.

``Uploads`` is a mixin of provider.SynapseMemoryProvider and reads its state (the bound
session, the turn uploader, the home, the gateway identity). An upload converts the
list Hermes handed over (a copy taken as the hook ran), with each row's
``display_kind`` from the session store where the list lost it, and hands the records
to the core (turn_upload.TurnUploader), which sends each row once (shipped.Ledger).
The store is also read for who started each gateway turn (source.py). A store that
cannot be read costs only the kinds the list lost; the reason is logged once.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from ...turn_upload import TurnUploader
from ...upload import Uploader
from . import carried, source
from . import messages as hm
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
        listed: list[dict[str, Any]],
        timeout: float,
        model: str,
        finished: bool = False,
    ) -> bool:
        """Send the complete turns of session ``sid`` in ``listed`` (the rows Hermes
        handed over) not shipped yet (turn_upload.TurnUploader). True when at least one
        batch reached the server. ``finished``: a turn has just ended. The session's row
        is read as this runs: is it still the owner's, and what does it continue
        (carried.Followed)."""
        turns, uploader = self._turns, self._uploader
        verdict = self._verdict(sid) if turns is not None and uploader is not None else None
        if turns is None or uploader is None or verdict is None:
            return False
        self._followed.follow(uploader, turns, self._home, sid, verdict.continues, LOOKBACK)
        got = source.read(
            self._home,
            sid,
            listed,
            gateway=self._gateway,
            owner=self._owner_id,
            route=self._route,
        )
        if got.foreign:
            self._authored()
            return False
        if got.reason and listed and self._gateway:
            line = (
                f"synapse: the admission check could not run for gateway session {sid} "
                f"({got.reason}); uploads go on with the turn_author check alone, as in the "
                "released client"
            )
            with self._lock:
                first = line not in self._said
                self._said.add(line)
            if first:
                _warn(line)
        if got.reason and listed:
            line = f"synapse: display kinds were not read from Hermes's store: {got.reason}"
            with self._lock:
                first = line not in self._said
                self._said.add(line)
            if first:
                _warn(line)
        if got.mismatched:
            line = (
                f"synapse: Hermes's list and its store disagree on {len(got.mismatched)} row(s) "
                f"of session {sid}; they are sent as the list has them"
            )
            with self._lock:
                first = line not in self._said
                self._said.add(line)
            if first:
                _warn(line)
        records = hm.to_records(listed, got.kinds)
        about = {"project": self._project, "cwd": self._cwd, "model": model or None}
        return turns.upload(records, sid, timeout=timeout, finished=finished, **about)

    def _drain(self, timeout: float) -> None:
        uploader = self._uploader
        if uploader is None:
            return
        try:
            uploader.drain(timeout)
        except Exception:
            _log().warning("synapse: outbox drain failed", exc_info=True)
