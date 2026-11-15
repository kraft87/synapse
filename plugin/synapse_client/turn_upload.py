"""Finished turns, uploaded once each.

``TurnUploader`` takes a harness's conversation as ``messages`` records (messages.py)
and uploads its finished turns, once each, with the turn before as context. It sends
through an ``upload.Uploader``, which refuses a private session and queues what fails.

Private mode covers whole sessions. Nothing is sent for a session while it is private,
and nothing is remembered here of which turns were spoken then: one put back on the
record (``synapse private off --forget``) is back whole, and its next upload sends every
finished turn not sent yet, as far back as its look-back reaches.

What it remembers of turns is kept per session id: which were sent, which are copies,
and the ones it holds back while the server has not said whether the session is
private. A harness names the session at every upload and may name several over its life
(a switch, a late upload for the id it left), so no turn remembered for one id is ever
sent under another.

With a ``ledger`` (shipped.Ledger), what was shipped is also kept on disk, per session,
by row id: a turn goes once however often a harness hands it over, across restarts
and across the uploaders of one session, and only the rows of a turn not shipped yet
go (messages.pending). A batch is recorded once it is delivered or in the outbox; one
refused or lost is not. A session the ledger knows is not looked back at: every
complete turn it has not shipped goes. One it does not know is looked back at as
below, and the turns before the window are recorded as passed over: not looked at
again here, still sent by an import.

A session's records can hold turns that are not its own to upload: copied from another
session by a harness that names no session they came from. A harness that knows of such
a session says so (``keep_out``, or ``start`` with a look-back of 0): nothing its
records hold before the first turn that ends here is uploaded, as a turn or as context,
by the row that opens each (messages.opening_key): however such a turn goes on or ends
later, it is the same turn. The uploader does not find such a session itself.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence

from . import messages
from .shipped import Ledger
from .upload import LOST, PRIVATE, QUEUED, SENT, WAITING, Uploader

#: How many turns a TurnUploader keeps in memory for a session it may not upload yet.
KEPT_TURNS = 200

_Pair = tuple[messages.Turn | None, messages.Turn]  # a turn's context, and the turn


class _Session:
    """What a TurnUploader remembers of one session id."""

    def __init__(self, lookback: int | None) -> None:
        self.sent: set[str] = set()
        # Row ids shipped (with a ledger: as recorded on disk, read again at each upload).
        self.shipped: set[str] = set()
        # Turns not handed over yet that the conversation may no longer hold when they
        # can be (a compression in place): by key, oldest first, each with its context.
        self.kept: dict[str, _Pair] = {}
        # Copies, not this session's to upload, by the row that opens each: no context
        # either.
        self.copies: set[str] = set()
        self.lookback = lookback
        # Whether it holds copies: it was started so, or the harness said so.
        self.closed = lookback == 0

    def close(self) -> None:
        """It holds turns that are not its own: none of them is looked back at."""
        self.lookback, self.closed = 0, True


class TurnUploader:
    """Finished turns, uploaded once each in the ``messages`` format.

    ``client`` and ``source`` go in every body: the harness's name, and which of its
    hooks sends. Turns are remembered by ``messages.turn_key`` and their rows' ids, per
    session id, in memory and, with a ``ledger``, on disk; without one, a restart that
    re-sends a turn costs a request, not a duplicate (the server dedups on span id). With a look-back, a session's first upload considers only
    its last so many complete turns and counts the ones before as sent: a resumed
    conversation's older turns were sent, or not, before this process. ``lookback`` is
    the look-back of a session first met at an upload; ``start`` gives
    one session its own. No look-back is for a conversation new here, or one whose turns
    from before may all go up. A look-back of 0 is for a session that holds copies of
    another's turns: nothing is uploaded from it until an upload that ends a turn
    (``finished``), which holds the whole conversation. Of what the records hold then,
    only the last turn is uploaded, the one that has just ended; the others are noted
    as copies, as are the turns an upload before it was shown, and do not go as the
    context of a later turn either. Safe across threads."""

    def __init__(
        self,
        uploader: Uploader,
        *,
        client: str,
        source: str,
        lookback: int | None = None,
        ledger: Ledger | None = None,
    ) -> None:
        self.uploader = uploader
        self._client = client
        self._source = source
        self._lock = threading.Lock()
        self._lookback = lookback
        self._sessions: dict[str, _Session] = {}
        self.ledger = ledger

    def start(
        self, session_id: str, lookback: int | None = None, *, continues: Iterable[str] = ()
    ) -> None:
        """Forget what is remembered of ``session_id``, the turns sent and the ones
        kept: it is new here, or reset. Its next upload looks back at most ``lookback``
        complete turns. That is 0 whatever is asked when the session, or one of
        ``continues`` (the sessions it goes on from, whose turns it may hold), holds
        copies as far as is known here: no later look-back may reach one."""
        with self._lock:
            ids = dict.fromkeys((session_id, *continues))
            if any(sid in self._sessions and self._sessions[sid].closed for sid in ids):
                lookback = 0
            self._sessions[session_id] = _Session(lookback)

    def keep_out(self, session_id: str) -> None:
        """``session_id`` holds turns from before that are not its own to upload, and
        the harness knows. It is treated as one started with a look-back of 0, from its
        next upload. What is remembered of it stays, and one treated so already is left
        as it is."""
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                self._sessions[session_id] = _Session(0)
            elif not state.closed:
                state.close()

    def _record(
        self, session_id: str, turns: Sequence[messages.Turn], *, passed: bool = False
    ) -> None:
        """Write ``turns`` to the ledger as shipped (``passed``: as passed over by a first
        look-back). A record that cannot be written is said: those turns may go again
        after a restart, which the server's span dedup absorbs."""
        if self.ledger is None or not turns:
            return
        uids = [uid for turn in turns for uid in messages.ids(turn)]
        # A turn that went whole is recorded by its key too: a later turn may then lead
        # with it as context (messages.pending).
        keys = [messages.turn_key(turn) for turn in turns if messages.whole(turn)]
        try:
            if passed:
                self.ledger.add(session_id, passed=uids)
            else:
                self.ledger.add(session_id, uids, keys)
        except OSError as err:
            self.uploader.warn(
                f"synapse: what was shipped for session {session_id} could not be recorded "
                f"({type(err).__name__}); it may be sent again"
            )

    def upload(
        self,
        records: Sequence[messages.Record],
        session_id: str,
        *,
        timeout: float,
        project: str | None = None,
        cwd: str | None = None,
        model: str | None = None,
        finished: bool = False,
    ) -> bool:
        """Send the complete turns of ``records`` not sent yet, oldest first, in batches.
        True when at least one batch reached the server. ``finished`` says a turn has
        just ended, in this process; without it the last turn may still be running.

        Nothing is sent for a session in private mode or with no id, and nothing is
        noted of the turns it shows then: put back on the record, the session is back
        whole. One the uploader holds is settled first (Uploader.settle): the server is
        asked or told, and the upload goes on when the session turns out to be on the
        record. While the server does not answer, nothing is sent or queued: the turns
        this upload would send are kept in memory (KEPT_TURNS at most; the oldest are
        given up, with a warning) and go up once the session is on the record, also
        those the conversation no longer holds by then: when the server says so, or,
        when its answer made the session private, once that is put back.

        A session that holds copies (``start`` with 0, ``keep_out``): every turn of
        ``records`` is noted as a copy but the last one. With ``finished`` that one is
        the turn that has just ended: it is uploaded when it is complete and holds one
        exchange (messages.joined: no user row besides the one that opens it, nothing
        after its reply), and noted as a copy when not. Without ``finished`` the records
        may be part of the conversation (a flush before a compression) and the last
        turn may still be running: it is left for the next upload to settle, and the
        noting is done again at each upload until one that ends a turn. A turn noted
        as a copy stays one.

        A batch whose send fails waits in the outbox and counts as handled; the batches
        after it are queued without a request, since theirs would fail the same way. A
        batch that could not be queued either is kept too, and tried again by the next
        upload, as is one refused because the session waits for the server's answer
        again. Never raises over a failed send."""
        if not session_id:
            return False
        standing = self.uploader.settle(session_id)
        waiting = standing == WAITING
        all_turns = messages.turns(records)
        key, opening = messages.turn_key, messages.opening_key
        given_up = 0
        held = None  # the opening of a last turn left for the next upload to settle
        on_disk = self.ledger.load(session_id) if self.ledger is not None else None
        windowed: list[messages.Turn] = []  # turns before a first look-back window
        with self._lock:
            if standing == PRIVATE:
                return False
            state = self._sessions.get(session_id)
            if state is None:
                state = self._sessions[session_id] = _Session(self._lookback)
            if on_disk is not None:
                state.shipped |= on_disk.uids | on_disk.passed
                state.sent |= on_disk.keys
                if on_disk.known and state.lookback is not None and state.lookback > 0:
                    state.lookback = None  # shipped before: every turn not shipped goes
            # A turn sent or kept here is this session's own: it is no copy.
            ours = state.sent | state.kept.keys()
            if state.lookback == 0:
                # A flush may hold part of the conversation only (a compression hands
                # over the rows it is about to summarise): what it shows is noted, and
                # the look-back stays until a turn ends, which shows the whole of it.
                rest, state.lookback = all_turns, None if finished else 0
                last = rest[-1] if rest else None
                if last and not messages.joined(last):
                    if not finished:
                        rest, held = rest[:-1], opening(last)  # it may still be running
                    elif messages.complete(last):
                        rest = rest[:-1]  # the turn that has just ended
                state.copies |= {opening(turn) for turn in rest if key(turn) not in ours}
            elif state.lookback is not None:
                # The turns before the window were this session's business before this
                # process: count them as sent, now and for every later upload.
                done = [turn for turn in all_turns if messages.complete(turn)]
                older = done[: max(0, len(done) - state.lookback)]
                state.sent |= {key(turn) for turn in older}
                state.shipped |= {uid for turn in older for uid in messages.ids(turn)}
                windowed = older
                state.lookback = None
            selected: list[_Pair] = []
            for before, turn in messages.pending(all_turns, state.sent, shipped=state.shipped):
                if opening(turn) in state.copies or opening(turn) == held:
                    continue
                if before is not None and opening(before) in state.copies:
                    before = None
                selected.append((before, turn))
            if waiting:
                for pair in selected:
                    state.kept.setdefault(key(pair[1]), pair)
                while len(state.kept) > KEPT_TURNS:
                    oldest = next(iter(state.kept))
                    del state.kept[oldest]
                    state.sent.add(oldest)  # given up for good, not kept again
                    given_up += 1
            else:
                # Kept turns the conversation no longer holds lead: they are the older.
                here = {key(turn) for _, turn in selected}
                selected = [
                    *(pair for kept, pair in state.kept.items() if kept not in here),
                    *selected,
                ]
                state.kept.clear()
                # Claimed before the request: a concurrent upload must not send them again.
                state.sent |= {key(turn) for _, turn in selected if messages.whole(turn)}
                state.shipped |= {uid for _, turn in selected for uid in messages.ids(turn)}
        self._record(session_id, windowed, passed=True)
        if waiting:
            if given_up:
                self.uploader.warn(
                    f"synapse: more than {KEPT_TURNS} turns of session {session_id} wait for "
                    f"the server's answer; {given_up} of the oldest will not be uploaded"
                )
            return False
        contexts = {key(turn): context for context, turn in selected}
        sent_any = failed = False
        reason = ""
        for batch, new_turns in messages.batches(selected):
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
            if outcome in (SENT, QUEUED):
                self._record(session_id, new_turns)  # delivered, or waiting in the outbox
            elif outcome in (LOST, PRIVATE):
                # Refused: marked private since, or it waits for the server's answer again.
                again = outcome == LOST or self.uploader.asking(session_id)
                with self._lock:
                    state.sent -= {key(turn) for turn in new_turns}
                    state.shipped -= {uid for turn in new_turns for uid in messages.ids(turn)}
                    if again:
                        for turn in new_turns:
                            state.kept.setdefault(key(turn), (contexts.get(key(turn)), turn))
        return sent_any
