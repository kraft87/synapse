"""Finished turns, uploaded once each.

``TurnUploader`` takes a harness's conversation as ``messages`` records (messages.py)
and uploads its finished turns, once each, with the turn before as context. It sends
through an ``upload.Uploader``, which refuses a private session and queues what fails.

What it remembers of turns is kept per session id: which were sent, which are off the
record, and the ones it holds back while the server has not said whether the session is
private. A harness names the session at every upload and may name several over its life
(a switch, a late upload for the id it left), so no turn remembered for one id is ever
sent under another.

A session's records can hold turns that are not its own to upload: spoken while it was
off the record, or copied from another session before this process met it. Their shape
does not tell them from a new turn, so the harness says when each turn starts
(``begin``): of the turns such a session holds when it is first found on the record,
only a last one seen to begin there is uploaded. The others stay out, as turns and as
context, by the row that opens each (messages.opening_key): however such a turn goes on
or ends later, it is the same turn. An upload notes the turns it is shown, and one that
ends no turn may be shown part of the conversation (a flush before a compression): for
a harness that says when turns start, the session stays as it was first found until an
upload that ends a turn has shown all of it. Once a turn begun off the record has
ended, the session's next upload that ends a turn notes again what it holds. Until
then, and while a turn begun off the record is still to end, an upload that ends none
sends no new turn for such a session: the rows of a turn from off the record may be
whole.

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
        # How many turns begun off the record had ended, under whatever id, when an
        # upload that ends a turn last noted what this session holds.
        self.noted = 0
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
    the context of a later turn either. With ``begins`` that look-back stays until an
    upload that ends a turn (``finished``), which holds the whole conversation: one
    before it notes the turns it is shown. No look-back is for a conversation new here,
    or one whose turns from before may all go up. With ``begins`` (the harness calls
    ``begin`` at every turn start), one started with no look-back and no session it
    goes on from is checked: when its first turn start or its first upload shows it
    holds turns from before (copies, written after the harness said it was new, from a
    session nothing names), it is treated as one that was off the record, and
    ``copied`` is called with its id: the harness's chance to keep that for a process
    built later, which this one's memory does not reach. Safe across threads."""

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
        # the uploader as a whole (``begin``). ``_open``: begun on the record, this
        # turn and at most the one before. ``_unseen``: begun off the record, each
        # with the turn before it when that one was still open; all older than the
        # ones in ``_open``. ``_ended``: how many of those have ended since.
        self._open = 0
        self._unseen = 0
        self._ended = 0

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
        ``continues`` (the sessions it goes on from, whose turns it may hold), was off
        the record here: no later look-back may reach a turn spoken then. What is known
        of the turns now running is not kept per session (``begin``), and stays. So does
        a doubt not settled yet, also from the sessions it goes on from: whether one
        said to be new holds turns from before, with the turn starts seen since, each
        once, however often this id is started and whichever of them went on from
        which. A session that names the ones it goes on from is not said to be new:
        what they are says whether their turns may go up."""
        with self._lock:
            ids = dict.fromkeys((session_id, *continues))
            past = {sid: self._sessions[sid] for sid in ids if sid in self._sessions}
            if any(one.was_private for one in past.values()):
                lookback = 0
            state = self._sessions[session_id] = _Session(lookback, self._begins and len(ids) == 1)
            state.took = frozenset(past).union(*(one.took for one in past.values()))
            state.new = state.new or (lookback != 0 and any(one.new for one in past.values()))
            for one in past.values():
                state.begun |= one.begun if one.new else set()

    def keep_out(self, session_id: str) -> None:
        """``session_id`` holds turns from before that are not its own to upload, and
        the harness knows: by what ``copied`` told a process before this one. It is
        treated as a session that was off the record, from its next upload. What is
        remembered of it stays, and one treated so already is left as it is."""
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                self._sessions[session_id] = _Session(0, self._begins)
            elif not state.was_private:
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
        turn-start hook: of the turns a session holds when it is first found on the
        record (it was off it, or they are copies), only one seen here to begin on the
        record is uploaded.

        A turn start is counted until an upload that ends a turn (``finished``) has
        been seen for it, whatever session ids the two name. Which turn an upload
        ends is not known here, only that uploads come in the order their turns
        began: each takes the oldest start still counted, and the count errs to
        private. On the record two starts are kept at most, this turn's and the one
        before's, since an upload may run one turn late. A turn begun while its
        session is private here (its marker, or made private by
        ``Uploader.carry_private``: the server's open question is not waited for) is
        counted with the turn before it when that one is still counted: the late
        upload of a turn begun off the record is not taken for the end of a later turn,
        and the late upload of the turn before is not taken for the end of the one
        begun off the record (neither goes up then). One turn before is counted, not
        two: with the uploads of two turns still to come when a turn begins off the
        record, the second takes that turn's count, and its own upload, or a flush, can
        then send it. A turn begun on the record whose upload never comes (it was
        stopped) stays counted, as the turn before, from then on: each later stretch
        off the record is counted with it, and costs a session that holds turns from
        before the first turn back on the record. One begun off the record whose upload
        never comes stays counted until the upload of a later turn takes its place,
        which costs such a session that turn and leaves that turn's own start counted,
        as one stopped on the record is. One begun while the server has not said yet
        counts as begun on the record until it says "private" (``upload``).

        ``earlier`` says the conversation held a turn before this one. When the first
        turn start of a session said to be new (``start``) says so, those turns are
        copies this process never saw begin (the harness built its agent before it
        wrote them): they are not looked back at, as for a session that was off the
        record. No request: whether the session is private is read on this machine."""
        if not session_id:
            return
        private = self.uploader.private_here(session_id)
        found: list[str] = []
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                state = self._sessions[session_id] = _Session(self._lookback, self._begins)
            # The upload that ends the turn before may still come, after this start:
            # that turn stays counted, or its upload would pass for the end of this one.
            before = min(self._open, 1)
            if private:
                self._open, self._unseen = 0, self._unseen + 1 + before
            else:
                self._open = before + 1
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
        user row besides the one that opens it, nothing after its reply) and the oldest
        turn start still counted (``begin``) was on the record. Such a last turn is
        uploaded when ``finished`` says a turn has ended and it is complete: the turn
        an upload ends is taken to be the last its records show. Without ``finished``
        it is left for the next upload to settle, as is any last turn still running
        then, since it may be about to begin. A turn noted as off the record before
        stays there. Without ``finished`` the records may be part of the conversation
        (a flush before a compression): for a harness that says when turns start, the
        noting is done again at each upload until one that ends a turn. It is done
        again by the session's first upload that ends a turn after one begun off the
        record has ended, whichever session id that turn's own upload named: its rows
        are in the records by then. An upload refused as private notes every turn it
        is shown, and nothing else is remembered of a private session. While a turn
        begun off the record is still to end, or has ended with nothing noted for this
        session since, an upload without ``finished`` picks no turn to send: that turn
        may be whole in the records (its reply cut short, its upload late), and only
        an upload that ends a turn notes it. The turns kept for the server's answer
        still go.

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
        found: list[str] = []
        with self._lock:
            if standing == TAKEN:
                # Turns begun before the answer began off the record, as it turns out.
                self._open, self._unseen = 0, self._unseen + self._open
            # Whether the turn now running, or the one just ended, began on the record.
            on_record = bool(self._open) and not self._unseen
            if finished and self._unseen:
                self._unseen -= 1  # one turn has ended: the oldest still counted
                self._ended += 1
            elif finished and self._open:
                self._open -= 1
            state = self._sessions.get(session_id)
            if state is None:
                if refused:
                    return False
                state = self._sessions[session_id] = _Session(self._lookback, self._begins)
            # A turn sent or kept here was spoken on the record: it stays there.
            ours = state.sent | state.kept.keys()
            if refused:
                if state.was_private:
                    state.off |= {opening(turn) for turn in all_turns if key(turn) not in ours}
                    state.noted = self._ended if finished else state.noted
                return False
            if waiting and not self.uploader.asking(session_id):
                # Answered on another thread since: what it found decides (it starts the
                # session again when the answer was "private"), and nothing is kept here.
                return False
            if state.new:
                if not state.begun and not finished:
                    return False  # what it holds is not known before its first turn
                if len(all_turns) > len(state.begun) + (not finished):
                    found = self._close(session_id, state)
                state.new = False
            # A turn begun off the record has ended since this session last noted what
            # it holds, under this id or another: its rows are in the records by now.
            behind = state.was_private and state.noted < self._ended
            if state.lookback == 0 or (finished and behind):
                # A flush may hold part of the conversation only (a compression hands
                # over the rows it is about to summarise): what it shows is noted, and
                # the look-back stays until a turn ends, which shows the whole of it.
                rest, state.lookback = all_turns, None if finished or not self._begins else 0
                last = rest[-1] if rest else None
                if last and not messages.joined(last):
                    if finished:
                        if on_record and messages.complete(last):
                            rest = rest[:-1]  # the turn seen to begin on the record
                    elif on_record or not messages.complete(last):
                        rest, held, state.lookback = rest[:-1], opening(last), 0
                state.off |= {opening(turn) for turn in rest if key(turn) not in ours}
                if finished:
                    state.noted = self._ended
            elif state.lookback is not None:
                # The turns before the window were this session's business before this
                # process: count them as sent, now and for every later upload.
                done = [turn for turn in all_turns if messages.complete(turn)]
                older = done[: max(0, len(done) - state.lookback)]
                state.sent |= {key(turn) for turn in older}
                state.lookback = None
            # A turn begun off the record is still to end, and its rows may be whole
            # (a reply cut short, an upload that runs late). The upload that ends a
            # turn notes it: until then one that ends none picks no turn to send.
            quiet = not finished and state.was_private and (bool(self._unseen) or behind)
            selected: list[_Pair] = []
            for before, turn in () if quiet else messages.pending(all_turns, state.sent):
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
