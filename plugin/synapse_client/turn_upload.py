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

A session's records can hold turns that are not its own to upload: copied from another
session before this process met it, by a harness that names no session they came from.
Their shape does not tell them from a new turn, so the harness says when each turn
starts (``begin``): of the turns such a session holds at its first upload, only a last
one seen to begin here is uploaded. The others stay out, as turns and as context, by
the row that opens each (messages.opening_key): however such a turn goes on or ends
later, it is the same turn. An upload notes the turns it is shown, and one that ends no
turn may be shown part of the conversation (a flush before a compression): for a
harness that says when turns start, the session stays as it was first found until an
upload that ends a turn has shown all of it.

Turn starts are counted for the uploader as a whole, not per session id: it serves one
agent, whose turns run one at a time, and a turn can begin under one id and end under
another (a switch inside it, an upload that names the id left). The count is all that
ties a turn start to its end, so it rests on two things a harness must keep to:
uploads that end a turn come in the order the turns began, and each shows the turn it
ends last.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Sequence

from . import messages
from .upload import LOST, PRIVATE, QUEUED, SENT, WAITING, Uploader

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
        # Copies, not this session's to upload, by the row that opens each: no context
        # either.
        self.copies: set[str] = set()
        self.lookback = lookback
        # Whether it holds copies: it was started so, or found to.
        self.closed = lookback == 0
        # Said to hold no turn from before (no look-back, no session it goes on from),
        # by a harness that says when turns start. Until its first upload, the turn
        # starts (``begun`` counts them) show whether that is so.
        self.new = checked and lookback is None
        # Each start is kept under a number of its own, not as a sum: a session that
        # goes on from several others (``start``) holds what they hold once, whichever
        # of them had it from which.
        self.begun: set[int] = set()
        # The sessions it went on from, and the ones those went on from (``_close``):
        # itself among them, once it was started again.
        self.took: frozenset[str] = frozenset()

    def close(self) -> None:
        """It holds turns from before after all: none of them is looked back at."""
        self.lookback, self.closed, self.new = 0, True, False


class TurnUploader:
    """Finished turns, uploaded once each in the ``messages`` format.

    ``client`` and ``source`` go in every body: the harness's name, and which of its
    hooks sends. Turns are remembered by ``messages.turn_key``, per session id and in
    memory only: the server dedups on span id, so a restart that re-sends a turn costs a
    request, not a duplicate. With a look-back, a session's first upload considers only
    its last so many complete turns and counts the ones before as sent: a resumed
    conversation's older turns were sent, or not, before this process. ``lookback`` is
    the look-back of a session first met at an upload or a turn start; ``start`` gives
    one session its own. No look-back is for a conversation new here, or one whose turns
    from before may all go up. A look-back of 0 is for a session that holds copies of
    another's turns: of what its records hold at its first upload, only a last turn
    seen to begin here (``begin``) is uploaded, and the others do not go as the context
    of a later turn either. With ``begins`` (the harness calls ``begin`` at every turn
    start) that look-back stays until an upload that ends a turn (``finished``), which
    holds the whole conversation: one before it notes the turns it is shown. And one
    started with no look-back and no session it goes on from is checked: when its first
    turn start or its first upload shows it holds turns from before (copies, written
    after the harness said it was new, from a session nothing names), it is treated as
    one started with 0, and ``copied`` is called with its id: the harness's chance to
    keep that for a process built later, which this one's memory does not reach. Safe
    across threads."""

    def __init__(
        self,
        uploader: Uploader,
        *,
        client: str,
        source: str,
        lookback: int | None = None,
        begins: bool = False,
        copied: Callable[[str], None] | None = None,
    ) -> None:
        self.uploader = uploader
        self._client = client
        self._source = source
        self._lock = threading.Lock()
        self._lookback = lookback
        self._begins = begins
        self._copied = copied
        self._sessions: dict[str, _Session] = {}
        self._counted = 0
        # Turn starts whose end (an upload with ``finished``) has not been seen, for
        # the uploader as a whole (``begin``).
        self._open = 0

    def _one(self) -> int:
        """A number for one more thing counted. Called with the lock held."""
        self._counted += 1
        return self._counted

    def start(
        self, session_id: str, lookback: int | None = None, *, continues: Iterable[str] = ()
    ) -> None:
        """Forget what is remembered of ``session_id``, the turns sent and the ones
        kept: it is new here, or reset. Its next upload looks back at most ``lookback``
        complete turns. That is 0 whatever is asked when the session, or one of
        ``continues`` (the sessions it goes on from, whose turns it may hold), holds
        copies as far as is known here: no later look-back may reach one. What is known
        of the turns now running is not kept per session (``begin``), and stays. So does
        a doubt not settled yet, also from the sessions it goes on from: whether one
        said to be new holds turns from before, with the turn starts seen since, each
        once, however often this id is started and whichever of them went on from
        which. A session that names the ones it goes on from is not said to be new:
        what they are says whether their turns may go up."""
        with self._lock:
            ids = dict.fromkeys((session_id, *continues))
            past = {sid: self._sessions[sid] for sid in ids if sid in self._sessions}
            if any(one.closed for one in past.values()):
                lookback = 0
            state = self._sessions[session_id] = _Session(lookback, self._begins and len(ids) == 1)
            state.took = frozenset(past).union(*(one.took for one in past.values()))
            state.new = state.new or (lookback != 0 and any(one.new for one in past.values()))
            for one in past.values():
                state.begun |= one.begun if one.new else set()

    def keep_out(self, session_id: str) -> None:
        """``session_id`` holds turns from before that are not its own to upload, and
        the harness knows: by what ``copied`` told a process before this one. It is
        treated as one started with a look-back of 0, from its next upload. What is
        remembered of it stays, and one treated so already is left as it is."""
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                self._sessions[session_id] = _Session(0, self._begins)
            elif not state.closed:
                state.close()

    def _close(self, session_id: str, state: _Session) -> list[str]:
        """``state``, said to be new, holds turns from before after all. So does
        every session that doubt came from, through however many ids, which is closed
        with it. The ids, for ``_found``. Called with the lock held."""
        found = [session_id]
        state.close()
        for sid in sorted(state.took):
            other = self._sessions.get(sid)
            if other is not None and other.new:
                other.close()
                found.append(sid)
        return found

    def _found(self, ids: Iterable[str]) -> None:
        """Tell the harness which sessions were found to hold copies (``copied``).
        Called without the lock; what the harness raises stops nothing here."""
        for session_id in ids if self._copied is not None else ():
            try:
                self._copied(session_id)
            except Exception:
                pass

    def begin(self, session_id: str, *, earlier: bool = False) -> None:
        """A turn starts in ``session_id``, now. A harness calls this from its
        turn-start hook: of the turns a session that holds copies shows at its first
        upload, only one seen here to begin is uploaded.

        A turn start is counted until an upload that ends a turn (``finished``) has
        been seen for it, whatever session ids the two name, and whether or not that
        upload was refused for a private session. Which turn an upload ends is not
        known here, only that uploads come in the order their turns began: each takes
        the oldest start still counted, and one may run a turn late, after the next
        turn has begun. A turn whose upload never comes (it was stopped) stays counted
        from then on.

        ``earlier`` says the conversation held a turn before this one. When the first
        turn start of a session said to be new (``start``) says so, those turns are
        copies this process never saw begin (the harness built its agent before it
        wrote them): they are not looked back at. No request."""
        if not session_id:
            return
        found: list[str] = []
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = self._sessions[session_id] = _Session(self._lookback, self._begins)
            # The upload that ends the turn before may still come, after this start:
            # that turn stays counted, or its upload would pass for the end of this one.
            self._open += 1
            if state.new and not state.begun and earlier:
                found = self._close(session_id, state)
            state.begun.add(self._one())
        self._found(found)

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

        A session that holds copies (``start`` with 0, ``keep_out``, ``begin``): at its
        first upload every turn of ``records`` is noted as a copy but the last one, and
        that one only when it holds one exchange (messages.joined: no user row besides
        the one that opens it, nothing after its reply) and a turn start is still
        counted (``begin``). Such a last turn is uploaded when ``finished`` says a turn
        has ended and it is complete: the turn an upload ends is taken to be the last
        its records show. Without ``finished`` it is left for the next upload to
        settle, as is any last turn still running then, since it may be about to begin.
        A turn noted as a copy before stays one. Without ``finished`` the records may
        be part of the conversation (a flush before a compression): for a harness that
        says when turns start, the noting is done again at each upload until one that
        ends a turn.

        A session said to be new (``start``) uploads nothing at a flush before its
        first turn start, and is taken to hold copies when its first upload shows more
        turns than were begun: also a turn that ends with no start seen.

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
        found: list[str] = []
        with self._lock:
            # Whether the turn now running, or the one just ended, was seen to begin.
            seen = bool(self._open)
            if finished and self._open:
                self._open -= 1  # one turn has ended: the oldest still counted
            if standing == PRIVATE:
                return False
            state = self._sessions.get(session_id)
            if state is None:
                state = self._sessions[session_id] = _Session(self._lookback, self._begins)
            # A turn sent or kept here is this session's own: it is no copy.
            ours = state.sent | state.kept.keys()
            if state.new:
                if not state.begun and not finished:
                    return False  # what it holds is not known before its first turn
                if len(all_turns) > len(state.begun) + (not finished):
                    found = self._close(session_id, state)
                state.new = False
            if state.lookback == 0:
                # A flush may hold part of the conversation only (a compression hands
                # over the rows it is about to summarise): what it shows is noted, and
                # the look-back stays until a turn ends, which shows the whole of it.
                rest, state.lookback = all_turns, None if finished or not self._begins else 0
                last = rest[-1] if rest else None
                if last and not messages.joined(last):
                    if finished:
                        if seen and messages.complete(last):
                            rest = rest[:-1]  # the turn seen to begin here
                    elif seen or not messages.complete(last):
                        rest, held, state.lookback = rest[:-1], opening(last), 0
                state.copies |= {opening(turn) for turn in rest if key(turn) not in ours}
            elif state.lookback is not None:
                # The turns before the window were this session's business before this
                # process: count them as sent, now and for every later upload.
                done = [turn for turn in all_turns if messages.complete(turn)]
                older = done[: max(0, len(done) - state.lookback)]
                state.sent |= {key(turn) for turn in older}
                state.lookback = None
            selected: list[_Pair] = []
            for before, turn in messages.pending(all_turns, state.sent):
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
                state.sent |= {key(turn) for _, turn in selected}
        self._found(found)
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
