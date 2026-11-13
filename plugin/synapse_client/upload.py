"""Uploads: how a turn leaves this machine.

``Uploader`` delivers one ingest body, or leaves it in the outbox. It refuses a body of
a session in private mode before anything is written or sent (private.check_payload,
which ``Client.ingest`` runs again at the request). It writes the body to the outbox
before its request, so a process that exits mid-request leaves it queued; removes it
once delivered; keeps it, as waiting, when the send failed. ``drain`` sends what is
waiting, and drops what belongs to a session taken off the record since it was queued.

``TurnUploader`` uploads the finished turns of one conversation in the ``messages``
format (messages.py), once each, with the turn before as context.

An adapter hands its records to a ``TurnUploader``, or its bodies to an ``Uploader``,
and never posts to the ingest route itself: that is what keeps a private session's
turns on the machine without each adapter remembering to check.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
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
    lost). ``log`` receives a line for each upload that failed or was refused."""

    def __init__(
        self, client: Client, outbox: Outbox | None = None, *, log: Log | None = None
    ) -> None:
        self.client = client
        self.outbox = outbox
        self._log = log

    def _say(self, line: str) -> None:
        if self._log is not None:
            try:
                self._log(line)
            except Exception:
                pass

    def is_private(self, session_id: str) -> bool:
        """Whether nothing may be uploaded for ``session_id`` (private.is_private)."""
        return private.is_private(self.client.settings, session_id, self._log)

    def _refused(self, payload: Mapping[str, Any]) -> bool:
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
            self._say(f"synapse: could not queue a failed upload ({type(err).__name__})")
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
            try:
                self.client.ingest(payload, timeout=timeout)
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

    def forget(self, lookback: int | None = None) -> None:
        """Forget which turns were sent (a new session id, a reset). The next upload
        then looks back at most ``lookback`` complete turns."""
        with self._lock:
            self._sent.clear()
            self._lookback = lookback

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
        id. A batch whose send fails waits in the outbox and counts as handled; the
        batches after it are queued without a request, since theirs would fail the same
        way. A batch that could not be queued either is tried again by the next upload.
        Never raises over a failed send."""
        if not session_id or self.uploader.is_private(session_id):
            return False
        all_turns = messages.turns(records)
        with self._lock:
            if self._lookback is not None:
                # The turns before the window were this session's business before this
                # process: count them as sent, now and for every later upload.
                done = [turn for turn in all_turns if messages.complete(turn)]
                older = done[: max(0, len(done) - self._lookback)]
                self._sent |= {messages.turn_key(turn) for turn in older}
                self._lookback = None
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
