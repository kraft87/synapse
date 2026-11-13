"""Finished turns, uploaded once each.

``TurnUploader`` takes a harness's conversation as ``messages`` records (messages.py)
and uploads its finished turns, once each, with the turn before as context. It sends
through an ``upload.Uploader``, which refuses a private session and queues what fails.

What it remembers is kept per session id: which turns were sent, which are off the
record, and the turns it holds back while the server has not said whether the session is
private. A harness names the session at every upload and may name several over its life
(a switch, a late upload for the id it left), so nothing remembered for one id is ever
sent or counted under another.

A session's records can hold turns that are not its own to upload: spoken while it was
off the record, or copied from another session before this process met it. Their shape
does not tell them from a new turn, so the harness says when each turn starts
(``begin``): of the turns such a session holds when it is first found on the record,
only a last one seen to begin there is uploaded. The others stay out, as turns and as
context, by the row that opens each (messages.opening_key): however such a turn goes on
or ends later, it is the same turn.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence

from . import messages
from .upload import LOST, PRIVATE, QUEUED, SENT, TAKEN, WAITING, Uploader

#: How many turns a TurnUploader keeps in memory for a session it may not upload yet.
KEPT_TURNS = 200

_Pair = tuple[messages.Turn | None, messages.Turn]  # a turn's context, and the turn


class _Session:
    """What a TurnUploader remembers of one session id."""

    def __init__(self, lookback: int | None, checked: bool = False) -> None:
        self.sent: set[str] = set()
        # Turns not handed over yet that the conversation may no longer hold when they
        # can be (a compression in place): by key, oldest first, each with its context.
        self.kept: dict[str, _Pair] = {}
        # Turns off the record, by the row that opens each: no context either.
        self.off: set[str] = set()
        self.lookback = lookback
        self.was_private = lookback == 0
        # Said to hold no turn from before (no look-back, no session it goes on from),
        # by a harness that says when turns start. Until its first upload, the turn
        # starts (``begun`` counts them) show whether that is so.
        self.new = checked and lookback is None
        self.begun = 0
        self.began = False  # the last turn to start began on the record, its end not seen
        self.unseen = 0  # turns begun off the record whose end has not been seen

    def close(self) -> None:
        """It holds turns from before after all: none of them is looked back at."""
        self.lookback, self.was_private, self.new = 0, True, False


class TurnUploader:
    """Finished turns, uploaded once each in the ``messages`` format.

    ``client`` and ``source`` go in every body: the harness's name, and which of its
    hooks sends. Turns are remembered by ``messages.turn_key``, per session id and in
    memory only: the server dedups on span id, so a restart that re-sends a turn costs a
    request, not a duplicate. With a look-back, a session's first upload considers only
    its last so many complete turns and counts the ones before as sent: a resumed
    conversation's older turns were sent, or not, before this process. ``lookback`` is
    the look-back of a session first met at an upload or a turn start; ``start`` gives
    one session its own. A look-back of 0 is for a conversation that was off the record
    until now: of what its records hold when it is first found on the record, only a
    last turn seen to begin there (``begin``) is uploaded, and the others do not go as
    the context of a later turn either. No look-back is for a conversation new here,
    or one whose turns from before may all go up. With ``begins`` (the harness calls
    ``begin`` at every turn start), one started with no look-back and no session it
    goes on from is checked: when its first turn start or its first upload shows it
    holds turns from before (copies, written after the harness said it was new, from a
    session nothing names), it is treated as one that was off the record. Safe across
    threads."""

    def __init__(
        self,
        uploader: Uploader,
        *,
        client: str,
        source: str,
        lookback: int | None = None,
        begins: bool = False,
    ) -> None:
        self.uploader = uploader
        self._client = client
        self._source = source
        self._lock = threading.Lock()
        self._lookback = lookback
        self._begins = begins
        self._sessions: dict[str, _Session] = {}

    def start(
        self, session_id: str, lookback: int | None = None, *, continues: Iterable[str] = ()
    ) -> None:
        """Forget what is remembered of ``session_id``, the turns sent and the ones
        kept: it is new here, or reset. Its next upload looks back at most ``lookback``
        complete turns. That is 0 whatever is asked when the session, or one of
        ``continues`` (the sessions it goes on from, whose turns it may hold), was off
        the record here: no later look-back may reach a turn spoken then. What is known
        of the turn now running stays, also from the sessions it goes on from (a turn
        that began under one of them ends under this id), as does a doubt not settled
        yet: whether one said to be new holds turns from before. A session that names
        the ones it goes on from is not said to be new: what they are says whether
        their turns may go up."""
        with self._lock:
            ids = dict.fromkeys((session_id, *continues))
            past = [self._sessions[sid] for sid in ids if sid in self._sessions]
            if any(one.was_private for one in past):
                lookback = 0
            state = self._sessions[session_id] = _Session(lookback, self._begins and len(ids) == 1)
            state.new = state.new or (lookback != 0 and any(one.new for one in past))
            state.begun = sum(one.begun for one in past if one.new)
            state.began = any(one.began for one in past)
            state.unseen = sum(one.unseen for one in past)

    def begin(self, session_id: str, *, earlier: bool = False) -> None:
        """A turn starts in ``session_id``, now. A harness calls this from its
        turn-start hook: of the turns a session holds when it is first found on the
        record (it was off it, or they are copies), only one seen here to begin on the
        record is uploaded. One begun while the session is private is counted until the
        upload that ends it (``finished``) has been seen, so an upload that runs late is
        not taken for the end of a later turn. One begun while the server has not said
        yet counts as begun on the record until it says "private" (``upload``).

        ``earlier`` says the conversation held a turn before this one. When the first
        turn start of a session said to be new (``start``) says so, those turns are
        copies this process never saw begin (the harness built its agent before it
        wrote them): they are not looked back at, as for a session that was off the
        record. No request: whether the session is private is read on this machine."""
        if not session_id:
            return
        asking = self.uploader.asking(session_id)
        private = not asking and self.uploader.is_private(session_id)
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = self._sessions[session_id] = _Session(self._lookback, self._begins)
            for other in self._sessions.values():
                other.began = False  # one turn runs at a time, and it is this one
            state.began = not private
            state.unseen += private
            if state.new and not state.begun and earlier:
                state.close()
            state.begun += 1

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

        Nothing is sent for a session in private mode or with no id. One the uploader
        holds is settled first (Uploader.settle): the server is asked or told, and the
        upload goes on when the session turns out to be on the record. When the server's
        answer makes it private, no turn from before is looked back at later (``start``
        with 0), as after a switch that knew. While the server does not answer, nothing
        is sent or queued: the turns this upload would send are kept in memory
        (KEPT_TURNS at most; the oldest are given up, with a warning) and go up once it
        says the session is on the record, also those the conversation no longer holds
        by then.

        A session that holds turns from before (``start`` with 0, ``begin``): when it is
        first found on the record, every turn of ``records`` is noted as off it but the
        last one, and that one only when it holds one exchange (messages.joined: no
        user row besides the one that opens it, nothing after its reply) and the last
        turn start was on the record with no turn begun off it still to end. Such a
        last turn is uploaded when ``finished`` says a turn has ended and it is
        complete; without ``finished`` it is left for the next upload to settle, as is
        any last turn still running then, since it may be about to begin. A turn noted
        as off the record before stays there.
        The same noting is done when the turn that has just ended began off the record,
        and at every upload refused as private. Nothing else is remembered of a private
        session.

        A session said to be new (``start``) uploads nothing at a flush before its
        first turn start, and is taken to hold turns from before when its first upload
        shows more turns than were begun: also a turn that ends with no start seen.

        A batch whose send fails waits in the outbox and counts as handled; the batches
        after it are queued without a request, since theirs would fail the same way. A
        batch that could not be queued either is kept too, and tried again by the next
        upload, as is one refused because the session waits for the server's answer
        again. Never raises over a failed send."""
        if not session_id:
            return False
        standing = self.uploader.settle(session_id)
        if standing == TAKEN:
            self.start(session_id, 0)
        refused = standing in (TAKEN, PRIVATE)
        waiting = standing == WAITING
        all_turns = messages.turns(records)
        key, opening = messages.turn_key, messages.opening_key
        given_up = 0
        held = None  # the opening of a last turn left for the next upload to settle
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                if refused:
                    return False
                state = self._sessions[session_id] = _Session(self._lookback, self._begins)
            if standing == TAKEN:
                # A turn begun before the answer began off the record, as it turns out.
                state.unseen, state.began = state.unseen + state.began, False
            # Whether the turn now running, or the one just ended, began on the record.
            on_record = state.began and not state.unseen
            began_off = finished and state.unseen > 0
            if finished:
                state.unseen = max(0, state.unseen - 1)
                state.began = False
            # A turn sent or kept here was spoken on the record: it stays there.
            ours = state.sent | state.kept.keys()
            if refused:
                if state.was_private:
                    state.off |= {opening(turn) for turn in all_turns if key(turn) not in ours}
                return False
            if waiting and not self.uploader.asking(session_id):
                # Answered on another thread since: what it found decides (it starts the
                # session again when the answer was "private"), and nothing is kept here.
                return False
            if state.new:
                if not state.begun and not finished:
                    return False  # what it holds is not known before its first turn
                if len(all_turns) > state.begun + (not finished):
                    state.close()
                state.new = False
            if state.lookback == 0 or (began_off and state.was_private):
                rest, state.lookback = all_turns, None
                last = rest[-1] if rest else None
                if last and not messages.joined(last):
                    if finished:
                        if on_record and messages.complete(last):
                            rest = rest[:-1]  # the turn seen to begin on the record
                    elif on_record or not messages.complete(last):
                        rest, held, state.lookback = rest[:-1], opening(last), 0
                state.off |= {opening(turn) for turn in rest if key(turn) not in ours}
            elif state.lookback is not None:
                # The turns before the window were this session's business before this
                # process: count them as sent, now and for every later upload.
                done = [turn for turn in all_turns if messages.complete(turn)]
                older = done[: max(0, len(done) - state.lookback)]
                state.sent |= {key(turn) for turn in older}
                state.lookback = None
            selected: list[_Pair] = []
            for before, turn in messages.pending(all_turns, state.sent):
                if opening(turn) in state.off or opening(turn) == held:
                    continue
                if before is not None and opening(before) in state.off:
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
                state.sent |= {key(turn) for _, turn in selected}
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
            elif outcome in (LOST, PRIVATE):
                # Refused: marked private since, or it waits for the server's answer again.
                again = outcome == LOST or self.uploader.asking(session_id)
                with self._lock:
                    state.sent -= {key(turn) for turn in new_turns}
                    if again:
                        for turn in new_turns:
                            state.kept.setdefault(key(turn), (contexts.get(key(turn)), turn))
        return sent_any
