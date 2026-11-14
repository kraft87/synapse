"""Uploads: how a turn leaves this machine.

``Uploader`` delivers one ingest body, or leaves it in the outbox. It refuses a body of
a session in private mode before anything is written or sent (private.check_payload,
which ``Client.ingest`` runs again at the request). It writes the body to the outbox
before its request, so a process that exits mid-request leaves it queued; removes it
once delivered; keeps it, as waiting, when the send failed. ``drain`` sends what is
waiting, and drops what belongs to a session taken off the record since it was queued.
``carry_private`` keeps a conversation private when it goes on under a new session id:
from the marker of the id it had, or, with no marker to go by, from what the server
answers when asked (``settle_private``; ``settle`` also says where the session stands).

An adapter hands its records to a ``turn_upload.TurnUploader``, which sends through an
``Uploader``, or its bodies to an ``Uploader``, and never posts to the ingest route
itself: that is what keeps a private session's turns on the machine without each adapter
remembering to check.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from . import private
from .api import Client
from .outbox import Outbox, Skip, short_reason

#: What ``Uploader.send`` and ``Uploader.queue`` answer.
SENT = "sent"  # delivered
QUEUED = "queued"  # not delivered: it waits in the outbox
LOST = "lost"  # not delivered, and it could not be queued
PRIVATE = "private"  # refused, nothing sent or queued: its session is in private mode

#: What ``Uploader.settle`` answers besides PRIVATE.
OPEN = "open"  # the session is on the record: it may be uploaded
WAITING = "waiting"  # the server has not said whether a session it continues is private

Log = Callable[[str], None]


class Uploader:
    """Uploads through ``client``, with ``outbox`` behind it (None: a failed upload is
    lost). ``log`` receives a line for each upload that failed or was refused; ``warn``
    (default: ``log``) the line for one that is lost, which nothing will send again."""

    def __init__(
        self,
        client: Client,
        outbox: Outbox | None = None,
        *,
        log: Log | None = None,
        warn: Log | None = None,
    ) -> None:
        self.client = client
        self.outbox = outbox
        self._log = log
        self._warn = warn or log
        # Sessions this uploader refuses itself (carry_private): private ones with no
        # marker to read, private ones the server is still to be told about, and ones
        # that wait for the server to say whether a session they continue is private
        # (with the ids to ask about). ``_settled``: whom to call for a session once
        # nothing is left open for it. ``_made``: every session made private here.
        self._unmarked: set[str] = set()
        self._owed: set[str] = set()
        self._asking: dict[str, tuple[str, ...]] = {}
        self._settled: dict[str, list[Callable[[bool], None]]] = {}
        self._made: set[str] = set()
        self._holding = threading.Lock()  # guards the five, which two threads change

    def _say(self, line: str, to: Log | None = None) -> None:
        log = to or self._log
        if log is not None:
            try:
                log(line)
            except Exception:
                pass

    def warn(self, line: str) -> None:
        """Say ``line`` where a loss is said (``warn``)."""
        self._say(line, self._warn)

    def _held(self, session_id: str) -> bool:
        """Whether this uploader holds ``session_id`` itself."""
        with self._holding:
            return (
                session_id in self._unmarked
                or session_id in self._owed
                or session_id in self._asking
            )

    def _holds(self, sessions: Iterable[str]) -> bool:
        return any(self._held(sid) for sid in sessions)

    def is_private(self, session_id: str) -> bool:
        """Whether nothing may be uploaded for ``session_id``: it is in private mode
        (private.is_private), or held here by ``carry_private``."""
        return self._held(session_id) or private.is_private(
            self.client.settings, session_id, self._log
        )

    def asking(self, session_id: str) -> bool:
        """Whether ``session_id`` is held until the server has said if a session it
        continues is private (``carry_private``)."""
        with self._holding:
            return session_id in self._asking

    def _private_here(self, session_id: str, lines: list[str]) -> bool:
        """Whether ``session_id`` is known on this machine to be private: held here as
        such, or by its marker (one that cannot be read counts). Called with
        ``_holding`` held, so what the marker's reader has to say goes to ``lines``:
        the caller says them once the lock is released, since a log callback may ask
        this uploader something."""
        return (
            session_id in self._unmarked
            or session_id in self._owed
            or private.is_private(self.client.settings, session_id, lines.append)
        )

    def _say_all(self, lines: Iterable[str]) -> None:
        for line in lines:
            self._say(line)

    def carry_private(
        self,
        earlier: Iterable[str],
        session_id: str,
        *,
        timeout: float | None = 5,
        settled: Callable[[bool], None] | None = None,
    ) -> bool:
        """Keep a conversation off the record when it goes on under a new session id (a
        compression that starts a new session, a branch): private mode is kept per id,
        and the new one has neither a marker nor a server row. ``earlier`` are the ids
        the conversation had.

        One of them is private here (a marker, one that cannot be read, or a session
        this uploader holds as private): ``session_id`` gets the marker now, and the
        answer is True. The server is told now, within ``timeout`` seconds, or by the
        next ``settle_private`` when ``timeout`` is None or the request failed. Until it
        has been told this uploader holds the session itself, and for as long as it
        lives when the marker could not be written; both failures are said to ``warn``.

        None of them is private here: that is not yet an answer, since a marker lasts
        twelve hours and ``off`` removes it, while the server's row stays and keeps
        dropping that session's turns. So the server is asked, by the next
        ``settle_private`` and not here, and until it has answered nothing is uploaded
        for ``session_id``. A session already waiting keeps what it was asking and asks
        about ``earlier`` as well, and an earlier id that is itself waiting passes its
        question on. The answer now is False.

        A session is made private this way once. Carried again, it is left as it is,
        and the answer is whether it is still private here: one put back on the record
        since (``synapse private off --forget``) stays there.

        ``settled`` is called once the question is closed for ``session_id``: the server
        said no earlier id is private (it is called with False), or the session was
        made private here and the server has its own row (True). It is the caller's
        note that the session need not be asked about again: put back on the record
        after that, it stays there, whole. A session carried again while its question
        is open (the server has not answered, or is still to be told) has each call's
        ``settled`` called. Not called when there was nothing to ask, nor for a session
        made private here whose question closed before the call. Never raises."""
        ids = tuple(dict.fromkeys(sid for sid in earlier if sid))
        failed = ""
        lines: list[str] = []
        try:
            with self._holding:
                if session_id in self._made:
                    if settled is not None and session_id in self._owed:
                        self._settled.setdefault(session_id, []).append(settled)
                    return self._private_here(session_id, lines)
                # One look at the earlier ids. Between two, a settle on another thread
                # could make one private and drop its question, and neither would be seen.
                source = next((sid for sid in ids if self._private_here(sid, lines)), "")
                if not session_id or session_id == source:
                    return bool(source)
                if source:
                    failed = self._hold(session_id)
                else:
                    waiting = [asked for sid in ids for asked in self._asking.get(sid, ())]
                    asked = [*self._asking.get(session_id, ()), *ids, *waiting]
                    ask = tuple(sid for sid in dict.fromkeys(asked) if sid != session_id)
                    if not ask:
                        return False
                    self._asking[session_id] = ask
                if settled is not None:
                    self._settled.setdefault(session_id, []).append(settled)
        finally:
            self._say_all(lines)
        if not source:
            return False
        said = self._carried(source, session_id, failed)
        if timeout is None:
            self._say(f"{said}: nothing is uploaded for it, and the server is still to be told")
        else:
            self._finish(session_id, timeout, said)
        return True

    def _hold(self, session_id: str) -> str:
        """Make ``session_id`` private here: its marker, and a hold until the server has
        been told. A question it was waiting on is dropped. Returns the name of what
        kept the marker from being written, if anything did: the uploader then holds
        the session for as long as it lives. Called with ``_holding`` held, so that no
        upload sees the session between the question and the hold."""
        failed = ""
        try:
            private.mark(self.client.settings, session_id)
        except (OSError, ValueError) as err:
            failed = type(err).__name__
            self._unmarked.add(session_id)
        self._owed.add(session_id)
        self._made.add(session_id)
        self._asking.pop(session_id, None)
        return failed

    def _carried(self, source: str, session_id: str, failed: str) -> str:
        """How the log names a session made private as ``source`` is; a marker that
        could not be written (``failed``) is said to ``warn``."""
        said = f"synapse: session {session_id} continues {source}, which is in private mode"
        if failed:
            self._say(
                f"{said}; its marker could not be written ({failed}), so "
                "nothing is uploaded for it until this program ends",
                self._warn,
            )
        return said

    def _report(self, said: str, told: bool) -> None:
        if told:
            self._say(f"{said}: it is off the record too")
        else:
            self._say(
                f"{said}; the server could not be told. Nothing is uploaded for it here, "
                "and the server is told again at the next turn",
                self._warn,
            )

    def _close(self, session_id: str) -> None:
        """Nothing is left open for ``session_id``: tell whoever asked to know."""
        with self._holding:
            asked = self._settled.pop(session_id, ())
            made = session_id in self._made
        for settled in asked:
            try:
                settled(made)
            except Exception:
                pass

    def _row(self, session_id: str, timeout: float) -> bool:
        """Whether the server holds ``session_id`` off the record. RAISES when it did
        not say: no answer, an error status, a reply that is not the route's."""
        answer = self.client.private_get(session_id, timeout=timeout).get("private")
        if not isinstance(answer, bool):
            raise ValueError("the reply does not say whether the session is private")
        return answer

    def _source(self, ids: Iterable[str], timeout: float) -> str:
        """The first of ``ids`` that is private: known here, without a request, or by
        the server's row. RAISES when the server did not say (``_row``)."""
        for sid in ids:
            lines: list[str] = []
            with self._holding:
                here = self._private_here(sid, lines)
            self._say_all(lines)
            if here or self._row(sid, timeout):
                return sid
        return ""

    def _tell(self, session_id: str, timeout: float) -> bool:
        """Tell the server ``session_id`` is private, if it is still to be told."""
        with self._holding:
            if session_id not in self._owed:
                return True
        try:
            self.client.private_put(session_id, timeout=timeout)
        except Exception:
            return False
        with self._holding:
            self._owed.discard(session_id)
        return True

    def settle_private(self, session_id: str, *, timeout: float = 5) -> bool:
        """Close what ``carry_private`` left open for ``session_id``: ask the server
        whether a session it continues is private (a row for one, or one found private
        here meanwhile, makes ``session_id`` private, marker and row; an answer that
        there is none lets it go), and tell the server about a private session it has
        not been told about. True when nothing is left open for it; False when the
        server did not answer, and the session stays held: a session the server could
        not be told about is said to ``warn`` each time. Once the server has answered,
        every other session it is still to be told about is told too (``settle_owed``).
        Never raises."""
        said = ""
        while not said:
            with self._holding:
                ids = self._asking.get(session_id)
            if ids is None:
                break
            try:
                source = self._source(ids, timeout)
            except Exception:
                return False
            failed = ""
            with self._holding:
                # Asked about more ids meanwhile (carry_private): those are next.
                left = tuple(sid for sid in self._asking.get(session_id, ()) if sid not in ids)
                if source:
                    failed = self._hold(session_id)
                elif left:
                    self._asking[session_id] = left
                else:
                    self._asking.pop(session_id, None)
            if source:
                said = self._carried(source, session_id, failed)
        return self._finish(session_id, timeout, said)

    def settle(self, session_id: str, *, timeout: float = 5) -> str:
        """Where ``session_id`` stands for an upload, once what is open for it has been
        tried (``settle_private``): OPEN, it may be uploaded; WAITING, the server has not
        said whether a session it continues is private; PRIVATE, it is private (its own
        marker, held here, or made so by the server's answer just now). A session with
        nothing open costs no request. Never raises."""
        if not self.is_private(session_id):
            return OPEN
        self.settle_private(session_id, timeout=timeout)
        if self.asking(session_id):
            return WAITING
        return PRIVATE if self.is_private(session_id) else OPEN

    def _finish(self, session_id: str, timeout: float, said: str = "") -> bool:
        """Tell the server about ``session_id`` if it is still to be told, say how that
        went (``said``: the session was made private just now), and close what was
        open for it. False when the server could not be told."""
        told = self._tell(session_id, timeout)
        if said or not told:
            self._report(said or f"synapse: session {session_id} is in private mode", told)
        if not told:
            return False
        self._close(session_id)
        self.settle_owed(timeout=timeout)
        return True

    def settle_owed(self, *, timeout: float = 5) -> None:
        """Tell the server about every private session it is still to be told about,
        stopping at the first request that fails, which is said to ``warn``: a session
        left behind by a later switch is otherwise never the one an upload is for.
        Never raises."""
        with self._holding:
            owed = sorted(self._owed)
        for session_id in owed:
            if not self._tell(session_id, timeout):
                self._report(f"synapse: session {session_id} is in private mode", False)
                return
            self._close(session_id)

    def _refused(self, payload: Mapping[str, Any]) -> bool:
        if self._holds(private.sessions_of(payload)):
            return True
        try:
            private.check_payload(self.client.settings, payload, (), self._log)
        except private.PrivateSession:
            return True
        return False

    def send(self, payload: Mapping[str, Any], *, timeout: float) -> tuple[str, str]:
        """Deliver ``payload`` (an ingest body) now. Returns the outcome and, when the
        send failed, its short reason: SENT; QUEUED (it waits in the outbox); LOST (it
        could not be queued either); PRIVATE (its session is in private mode: nothing was
        written or sent). ``timeout`` bounds each socket wait, not the request: no drain
        takes the body for three times that. Never raises over a failed send."""
        if self._refused(payload):
            return PRIVATE, ""
        ahead = self._ahead(payload, timeout)
        try:
            self.client.ingest(payload, timeout=timeout)
        except private.PrivateSession:  # taken off the record since the check above
            if self.outbox is not None and ahead is not None:
                self.outbox.remove(ahead)
            return PRIVATE, ""
        except Exception as err:
            reason = short_reason(err)
            self._say(f"synapse: upload failed ({reason}); queued for retry")
            if self.outbox is not None and ahead is not None:
                self.outbox.keep(ahead, reason)
                return QUEUED, reason
            return self.queue(payload, reason), reason
        if self.outbox is not None and ahead is not None:
            self.outbox.remove(ahead)
        self.settle_owed()  # the server is there: tell it what it is still to be told
        return SENT, ""

    def _ahead(self, payload: Mapping[str, Any], timeout: float) -> Path | None:
        """``payload`` queued for this send, or None when it could not be (it is then
        queued only if its send fails)."""
        if self.outbox is None:
            return None
        try:
            return self.outbox.put(payload, sending=3 * timeout)
        except (OSError, TypeError, ValueError):
            return None

    def queue(self, payload: Mapping[str, Any], reason: str) -> str:
        """Leave ``payload`` in the outbox without a request (the one before it just
        failed with ``reason``). QUEUED, LOST or PRIVATE. Never raises."""
        if self._refused(payload):
            return PRIVATE
        if self.outbox is None:
            return LOST
        try:
            self.outbox.put(payload, reason=reason)
        except (OSError, TypeError, ValueError) as err:
            self._say(
                f"synapse: could not queue a failed upload ({type(err).__name__}: "
                f"{str(err)[:160]}); it is lost",
                self._warn,
            )
            return LOST
        return QUEUED

    def drain(self, timeout: float, max_items: int = 20) -> tuple[int, int]:
        """Send what waits in the outbox (Outbox.drain): ``(sent, kept)``. A body whose
        session is now in private mode is dropped; one whose marker cannot be read stays
        waiting; neither stops the drain. Never raises over a failed send."""
        if self.outbox is None:
            return 0, 0

        def send(payload: Any) -> None:
            if not isinstance(payload, Mapping):
                raise Skip("not an ingest body", drop=True)
            if self._holds(private.sessions_of(payload)):
                raise Skip("its session is held off the record here", drop=False)
            try:
                self.client.ingest(payload, timeout=timeout, log=self._log)
            except private.PrivateSession as refused:
                if refused.sure:
                    raise Skip("its session is in private mode", drop=True) from None
                raise Skip("its session's private marker cannot be read", drop=False) from None

        return self.outbox.drain(send, max_items)
