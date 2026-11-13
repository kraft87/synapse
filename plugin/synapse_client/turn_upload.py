"""Finished turns, uploaded once each.

``TurnUploader`` takes a harness's conversation as ``messages`` records (messages.py)
and uploads its finished turns, once each, with the turn before as context. It sends
through an ``upload.Uploader``, which refuses a private session and queues what fails.

What it remembers is kept per session id: which turns were sent, which were spoken off
the record, and the turns it holds back while the server has not said whether the
session is private. A harness names the session at every upload and may name several
over its life (a switch, a late upload for the id it left), so nothing remembered for
one id is ever sent or counted under another.
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

    def __init__(self, lookback: int | None) -> None:
        self.sent: set[str] = set()
        # Turns not handed over yet that the conversation may no longer hold when they
        # can be (a compression in place): by key, oldest first, each with its context.
        self.kept: dict[str, _Pair] = {}
        self.off: set[str] = set()  # turns spoken off the record: no context either
        self.lookback = lookback
        self.was_private = lookback == 0


class TurnUploader:
    """Finished turns, uploaded once each in the ``messages`` format.

    ``client`` and ``source`` go in every body: the harness's name, and which of its
    hooks sends. Turns are remembered by ``messages.turn_key``, per session id and in
    memory only: the server dedups on span id, so a restart that re-sends a turn costs a
    request, not a duplicate. With a look-back, a session's first upload considers only
    its last so many complete turns and counts the ones before as sent: a resumed
    conversation's older turns were sent, or not, before this process. ``lookback`` is
    the look-back of a session first met at an upload; ``start`` gives one session its
    own. A look-back of 0 is for a conversation that was off the record until now: its
    turns from before do not go as the context of a later turn either, nor does a turn
    that only goes on from one of them (messages.stems). Safe across threads."""

    def __init__(
        self, uploader: Uploader, *, client: str, source: str, lookback: int | None = None
    ) -> None:
        self.uploader = uploader
        self._client = client
        self._source = source
        self._lock = threading.Lock()
        self._lookback = lookback
        self._sessions: dict[str, _Session] = {}

    def start(
        self, session_id: str, lookback: int | None = None, *, continues: Iterable[str] = ()
    ) -> None:
        """Forget what is remembered of ``session_id``, the turns sent and the ones
        kept: it is new here, or reset. Its next upload looks back at most ``lookback``
        complete turns. That is 0 whatever is asked when the session, or one of
        ``continues`` (the sessions it goes on from, whose turns it may hold), was off
        the record here: no later look-back may reach a turn spoken then."""
        with self._lock:
            past = (self._sessions.get(sid) for sid in (session_id, *continues))
            if any(one is not None and one.was_private for one in past):
                lookback = 0
            self._sessions[session_id] = _Session(lookback)

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
        just ended, in this process: when the last turn of ``records`` is complete and
        does not merely go on from a finished one (messages.stems), it is that one, and
        never one of the turns from before that the look-back leaves out.

        Nothing is sent, and nothing remembered, for a session in private mode or with no
        id. One the uploader holds is settled first (Uploader.settle): the server is
        asked or told, and the upload goes on when the session turns out to be on the
        record. When the server's answer makes it private, no turn from before is looked
        back at later (``start`` with 0), as after a switch that knew. While the server
        does not answer, nothing is sent or queued: the turns this upload would send are
        kept in memory (KEPT_TURNS at most; the oldest are given up, with a warning) and
        go up once it says the session is on the record, also those the conversation no
        longer holds by then. A batch whose send fails waits in the outbox and counts as
        handled; the batches after it are queued without a request, since theirs would
        fail the same way. A batch that could not be queued either is kept too, and
        tried again by the next upload. Never raises over a failed send."""
        if not session_id:
            return False
        standing = self.uploader.settle(session_id)
        if standing == TAKEN:
            self.start(session_id, 0)
        if standing in (TAKEN, PRIVATE):
            return False
        waiting = standing == WAITING
        all_turns = messages.turns(records)
        key = messages.turn_key
        given_up = 0
        with self._lock:
            if waiting and not self.uploader.asking(session_id):
                # Answered on another thread since: what it found decides (it starts the
                # session again when the answer was "private"), and nothing is kept here.
                return False
            state = self._sessions.get(session_id)
            if state is None:
                state = self._sessions[session_id] = _Session(self._lookback)
            if state.lookback is not None:
                # The turns before the window were this session's business before this
                # process: count them as sent, now and for every later upload.
                done = [turn for turn in all_turns if messages.complete(turn)]
                new = finished and bool(done) and done[-1] is all_turns[-1]
                new = new and not messages.stems(done[-1])
                older = done[: max(0, len(done) - max(state.lookback, int(new)))]
                state.sent |= {key(turn) for turn in older}
                if state.lookback == 0:
                    state.off |= {key(turn) for turn in older}
                state.lookback = None
            selected: list[_Pair] = []
            for before, turn in messages.pending(all_turns, state.sent):
                if state.off and not state.off.isdisjoint(messages.stems(turn)):
                    # It goes on from a turn spoken off the record, and holds it.
                    state.sent.add(key(turn))
                    state.off.add(key(turn))
                    continue
                selected.append((None if before and key(before) in state.off else before, turn))
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
                with self._lock:
                    state.sent -= {key(turn) for turn in new_turns}
                    if outcome == LOST:
                        for turn in new_turns:
                            state.kept.setdefault(key(turn), (contexts.get(key(turn)), turn))
        return sent_any
