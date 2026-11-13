"""Uploads: how a turn leaves this machine.

``Uploader`` delivers one ingest body, or leaves it in the outbox. It refuses a body of
a session in private mode before anything is written or sent (private.check_payload,
which ``Client.ingest`` runs again at the request). It writes the body to the outbox
before its request, so a process that exits mid-request leaves it queued; removes it
once delivered; keeps it, as waiting, when the send failed. ``drain`` sends what is
waiting, and drops what belongs to a session taken off the record since it was queued.
``carry_private`` keeps a conversation private when it goes on under a new session id:
from the marker of the id it had, or, with no marker to go by, from what the server
answers when asked (``settle_private``).

``TurnUploader`` uploads the finished turns of one conversation in the ``messages``
format (messages.py), once each, with the turn before as context.

An adapter hands its records to a ``TurnUploader``, or its bodies to an ``Uploader``,
and never posts to the ingest route itself: that is what keeps a private session's
turns on the machine without each adapter remembering to check.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import messages, private
from .api import Client
from .outbox import Outbox, Skip, short_reason

#: What ``Uploader.send`` and ``Uploader.queue`` answer.
SENT = "sent"  # delivered
QUEUED = "queued"  # not delivered: it waits in the outbox
LOST = "lost"  # not delivered, and it could not be queued
PRIVATE = "private"  # refused, nothing sent or queued: its session is in private mode

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
        # (with the ids to ask about). ``_settled``: what to call for a session once
        # nothing is left open for it.
        self._unmarked: set[str] = set()
        self._owed: set[str] = set()
        self._asking: dict[str, tuple[str, ...]] = {}
        self._settled: dict[str, Callable[[], None]] = {}
        self._holding = threading.Lock()  # guards the four, which two threads change

    def _say(self, line: str, to: Log | None = None) -> None:
        log = to or self._log
        if log is not None:
            try:
                log(line)
            except Exception:
                pass

    def _held(self, session_id: str, *, asking: bool = True) -> bool:
        """Whether this uploader holds ``session_id`` itself; with ``asking`` False,
        only as a session known to be private."""
        with self._holding:
            return (
                session_id in self._unmarked
                or session_id in self._owed
                or (asking and session_id in self._asking)
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

    def carry_private(
        self,
        earlier: Iterable[str],
        session_id: str,
        *,
        timeout: float | None = 5,
        settled: Callable[[], None] | None = None,
    ) -> bool:
        """Keep a conversation off the record when it goes on under a new session id (a
        compression that starts a new session, a branch): private mode is kept per id,
        and the new one has neither a marker nor a server row. ``earlier`` are the ids
        the conversation had.

        One of them is private here (a marker, one that cannot be read, or a session
        this uploader holds as private): ``session_id`` gets the marker now, and the
        answer is True. The server is told now, within ``timeout`` seconds, or at the
        session's next upload when ``timeout`` is None or the request failed
        (``settle_private``). Until it has been told this uploader holds the session
        itself, and for as long as it lives when the marker could not be written; both
        failures are said to ``warn``.

        None of them is private here: that is not yet an answer, since a marker lasts
        twelve hours and ``off`` removes it, while the server's row stays and keeps
        dropping that session's turns. So the server is asked, at the session's next
        upload and not here (``settle_private``), and until it has answered nothing is
        uploaded for ``session_id``. The answer now is False.

        ``settled`` is called once nothing is left open for ``session_id``: the server
        said no earlier id is private, or it has the session's own row. It is the
        caller's note that the session need not be asked about again. Never raises."""
        ids = tuple(dict.fromkeys(sid for sid in earlier if sid))
        source = next(
            (
                sid
                for sid in ids
                if self._held(sid, asking=False)
                or private.is_private(self.client.settings, sid, self._log)
            ),
            "",
        )
        if not session_id or session_id == source:
            return bool(source)
        with self._holding:
            if settled is not None:
                self._settled[session_id] = settled
            # A session that is itself waiting for an answer passes its question on.
            waiting = [asked for sid in ids for asked in self._asking.get(sid, ())]
        if source:
            said = self._carry(source, session_id)
            if timeout is None:
                self._say(
                    f"{said}: nothing is uploaded for it, and the server is told at its next upload"
                )
            else:
                self._report(said, self.settle_private(session_id, timeout=timeout))
            return True
        ask = tuple(sid for sid in dict.fromkeys([*ids, *waiting]) if sid != session_id)
        if not ask:
            self._close(session_id)
            return False
        with self._holding:
            self._asking[session_id] = ask
        return False

    def _carry(self, source: str, session_id: str) -> str:
        """Make ``session_id`` private here as ``source`` is: its marker, and a hold
        until the server has been told. Returns how the log names the two."""
        said = f"synapse: session {session_id} continues {source}, which is in private mode"
        unmarked = False
        try:
            private.mark(self.client.settings, session_id)
        except (OSError, ValueError) as err:
            unmarked = True
            self._say(
                f"{said}; its marker could not be written ({type(err).__name__}), so "
                "nothing is uploaded for it until this program ends",
                self._warn,
            )
        with self._holding:
            if unmarked:
                self._unmarked.add(session_id)
            self._owed.add(session_id)
        return said

    def _report(self, said: str, told: bool) -> None:
        if told:
            self._say(f"{said}: it is off the record too")
        else:
            self._say(
                f"{said}; the server could not be told. Nothing is uploaded for it here, "
                "and the server is told again at the session's next turn",
                self._warn,
            )

    def _close(self, session_id: str) -> None:
        """Nothing is left open for ``session_id``: tell whoever asked to know."""
        with self._holding:
            settled = self._settled.pop(session_id, None)
        if settled is not None:
            try:
                settled()
            except Exception:
                pass

    def _row(self, session_id: str, timeout: float) -> bool:
        """Whether the server holds ``session_id`` off the record. RAISES when it did
        not say: no answer, an error status, a reply that is not the route's."""
        answer = self.client.private_get(session_id, timeout=timeout).get("private")
        if not isinstance(answer, bool):
            raise ValueError("the reply does not say whether the session is private")
        return answer

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
        whether a session it continues is private (a row for one makes ``session_id``
        private, marker and row; an answer that there is none lets it go), and tell the
        server about a private session it has not been told about. True when nothing is
        left open for it; False when the server did not answer, and the session stays
        held. Once the server has answered, every other session it is still to be told
        about is told too (``settle_owed``). Never raises."""
        with self._holding:
            ids = self._asking.get(session_id)
        said = ""
        if ids is not None:
            try:
                source = next((sid for sid in ids if self._row(sid, timeout)), "")
            except Exception:
                return False
            if source:
                # Before the question is dropped: the session is never left unheld.
                said = self._carry(source, session_id)
            with self._holding:
                self._asking.pop(session_id, None)
        told = self._tell(session_id, timeout)
        if said:
            self._report(said, told)
        if not told:
            return False
        self._close(session_id)
        self.settle_owed(timeout=timeout)
        return True

    def settle_owed(self, *, timeout: float = 5) -> None:
        """Tell the server about every private session it is still to be told about,
        stopping at the first request that fails: a session left behind by a later
        switch is otherwise never the one an upload is for. Never raises."""
        with self._holding:
            owed = sorted(self._owed)
        for session_id in owed:
            if not self._tell(session_id, timeout):
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


class TurnUploader:
    """One conversation's finished turns, uploaded once each in the ``messages`` format.

    ``client`` and ``source`` go in every body: the harness's name, and which of its
    hooks sends. Turns are remembered by ``messages.turn_key`` in memory only: the server
    dedups on span id, so a restart that re-sends a turn costs a request, not a
    duplicate. With ``lookback``, the first upload considers only the last ``lookback``
    complete turns and counts the ones before as sent: a resumed conversation's older
    turns were sent, or not, before this process. Safe across threads."""

    def __init__(
        self, uploader: Uploader, *, client: str, source: str, lookback: int | None = None
    ) -> None:
        self.uploader = uploader
        self._client = client
        self._source = source
        self._lock = threading.Lock()
        self._sent: set[str] = set()
        self._lookback = lookback
        # How many complete turns there were at the last upload held back for the
        # server's answer (``_wait``).
        self._waited: int | None = None

    def forget(self, lookback: int | None = None) -> None:
        """Forget which turns were sent (a new session id, a reset). The next upload
        then looks back at most ``lookback`` complete turns."""
        with self._lock:
            self._sent.clear()
            self._lookback = lookback
            self._waited = None

    def _wait(self, records: Sequence[messages.Record]) -> None:
        """An upload was held back because the server has not said whether the session
        is private: widen the first upload's window by the turns finished since the last
        one held back, so that they are sent once it says the session is on the record."""
        done = sum(1 for turn in messages.turns(records) if messages.complete(turn))
        with self._lock:
            if self._lookback is not None and self._waited is not None:
                self._lookback += max(0, done - self._waited)
            self._waited = done

    def upload(
        self,
        records: Sequence[messages.Record],
        session_id: str,
        *,
        timeout: float,
        project: str | None = None,
        cwd: str | None = None,
        model: str | None = None,
    ) -> bool:
        """Send the complete turns of ``records`` not sent yet, oldest first, in batches.
        True when at least one batch reached the server.

        Nothing is sent, and nothing remembered, for a session in private mode or with no
        id. One the uploader holds is settled first (Uploader.settle_private): the server
        is asked or told, and the upload goes on when the session turns out to be on the
        record. When it turns out to be private, no turn from before is looked back at
        later (``forget(0)``), as after a switch that knew; while the server does not
        answer, nothing is sent or queued, and the turns finished meanwhile are sent
        once it does (``_wait``). A batch whose send fails waits in the outbox and
        counts as handled; the batches after it are queued without a request, since
        theirs would fail the same way. A batch that could not be queued either is tried
        again by the next upload. Never raises over a failed send."""
        if not session_id:
            return False
        if self.uploader.is_private(session_id):
            asked = self.uploader.asking(session_id)
            self.uploader.settle_private(session_id)
            if self.uploader.is_private(session_id):
                if asked and self.uploader.asking(session_id):
                    self._wait(records)
                elif asked:
                    self.forget(0)  # the server has just said it is private
                return False
        all_turns = messages.turns(records)
        with self._lock:
            if self._lookback is not None:
                # The turns before the window were this session's business before this
                # process: count them as sent, now and for every later upload. The
                # window takes in the turns finished while uploads were held back.
                done = [turn for turn in all_turns if messages.complete(turn)]
                held = 0 if self._waited is None else max(0, len(done) - self._waited)
                older = done[: max(0, len(done) - self._lookback - held)]
                self._sent |= {messages.turn_key(turn) for turn in older}
                self._lookback = None
            self._waited = None
            selected = messages.pending(all_turns, self._sent)
            # Claimed before the request: a concurrent upload must not send them again.
            self._sent |= {messages.turn_key(turn) for _, turn in selected}
        sent_any = failed = False
        reason = ""
        for batch, new in messages.batches(selected):
            payload = {
                "format": "messages",
                "client": self._client,
                "session_id": session_id,
                "project": project,
                "cwd": cwd,
                "model": model,
                "source": self._source,
                "records": batch,
            }
            if failed:
                outcome = self.uploader.queue(payload, reason)
            else:
                outcome, reason = self.uploader.send(payload, timeout=timeout)
                failed = outcome in (QUEUED, LOST)
            if outcome == SENT:
                sent_any = True
            elif outcome in (LOST, PRIVATE):
                with self._lock:
                    self._sent -= {messages.turn_key(turn) for turn in new}
        return sent_any
