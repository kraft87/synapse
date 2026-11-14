"""What a ``TurnUploader`` remembers (plugin/synapse_client/turn_upload.py).

A harness names the session at every upload and may name several over its life: the one
it was moved to, a late upload for the one it left. Turns are remembered per session
id, and turn starts are counted for the uploader. Pinned here:
  * turns kept for one session (held back for the server's answer, or lost with no
    outbox) go up under that session's id or not at all;
  * each session has its own look-back, and a session that holds copies of another's
    turns is never looked back at, under any id it goes on with;
  * of the turns such a session holds at its first upload, one goes up only when it
    was seen to begin here (``begin``): not one that holds more than its own exchange,
    or was there before any turn start was seen;
  * an upload that ends no turn notes the turns it is shown and settles nothing more;
  * turn starts are counted for the uploader, whichever ids a start and its end name,
    and an upload that runs one turn late is not taken for the end of the turn begun
    since;
  * private mode covers whole sessions: nothing is noted of the turns a session shows
    while it is private, and put back on the record it is back whole;
  * a session said to be new is not believed when its turn starts show otherwise, and
    the harness is told (``copied``), so that it can say so later (``keep_out``);
  * a session's own marker is not taken for the server's answer about the session it
    continues, and turns another upload sent meanwhile are not sent again.

The helpers are those of tests/test_client_core_upload.py: a recorder in place of the
client's transport, temp directories only.
"""

from __future__ import annotations

import threading
import urllib.error
from typing import Any

import pytest

from tests import test_client_core_upload as cu
from tests.helpers.client_core import use_core

use_core()
from synapse_client import upload  # noqa: E402

_mark, _turn, _turns, _get = cu._mark, cu._turn, cu._turns, cu._get
_DOWN = urllib.error.URLError("down")


def _sent(wire: Any) -> list[tuple[str, list[str]]]:
    """Each upload: its session id, and the ids of its user rows."""
    return [
        (
            post["body"]["session_id"],
            [row["id"] for row in post["body"]["records"] if row["role"] == "user"],
        )
        for post in wire.posts
    ]


def _records(wire: Any) -> list[list[dict[str, Any]]]:
    return [post["body"]["records"] for post in wire.posts]


def _forget(wire: Any, tmp_path: Any, session_id: str) -> None:
    """`synapse private off --forget`: the marker and the server's row go."""
    (tmp_path / "private" / session_id).unlink()
    wire.rows.discard(session_id)


# ---------------------------------------------------------------------------
# Kept turns stay with their session
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("private", [True, False], ids=["private", "on the record"])
def test_turns_kept_for_one_session_do_not_go_up_under_another(tmp_path, private):
    """s-a waits for the server's answer and its turns are kept. An upload for s-b,
    which nothing holds, sends s-b's turns alone. What becomes of s-a's is the server's
    answer for s-a."""
    turns, wire = _turns(tmp_path)
    if private:
        wire.rows.add("s-old")  # private on the server, its marker gone
    assert turns.uploader.carry_private(["s-old"], "s-a") is False
    wire.fail_private = _DOWN
    held = _turn(1) + _turn(2)
    assert turns.upload(held, "s-a", timeout=5, finished=True) is False and wire.tries == 0
    wire.fail_private = None
    assert turns.upload(_turn(7), "s-b", timeout=5, finished=True) is True
    assert _sent(wire) == [("s-b", ["u7"])]
    # The conversation dropped turn 1 meanwhile: it was kept, and it is s-a's.
    assert turns.upload(_turn(2) + _turn(3), "s-a", timeout=5, finished=True) is not private
    assert (tmp_path / "private" / "s-a").is_file() is private
    assert _sent(wire)[1:] == ([] if private else [("s-a", ["u1", "u2", "u3"])])


def test_a_turn_that_could_not_be_queued_stays_with_its_session(tmp_path):
    turns, wire = _turns(tmp_path)  # no outbox: a failed send is lost, and kept
    wire.fail = _DOWN
    assert turns.upload(_turn(1), "s-a", timeout=5) is False
    wire.fail = None
    marker = _mark(tmp_path, "s-a")  # `synapse private on`
    assert turns.upload(_turn(9), "s-b", timeout=5) is True
    assert _sent(wire) == [("s-b", ["u9"])]
    marker.unlink()
    assert turns.upload(_turn(2), "s-a", timeout=5) is True
    assert _sent(wire)[1:] == [("s-a", ["u1", "u2"])]


def test_a_turn_refused_at_the_send_because_the_session_is_private_is_not_kept(
    tmp_path, monkeypatch
):
    """`synapse private on` between the look at the session and the request: the send
    is refused. The turn is not sent, and it is not kept either: kept, it would go up
    after `off` although the conversation no longer holds it. One the conversation
    still holds goes up then, as every turn of a session marked by hand does."""
    turns, wire = _turns(tmp_path)
    marker = _mark(tmp_path, "s-1")
    with monkeypatch.context() as patch:
        patch.setattr(turns.uploader, "settle", lambda session_id: upload.OPEN)
        assert turns.upload(_turn(1) + _turn(2), "s-1", timeout=5, finished=True) is False
    assert wire.tries == 0
    marker.unlink()
    assert turns.upload(_turn(2) + _turn(3), "s-1", timeout=5, finished=True) is True
    assert _sent(wire) == [("s-1", ["u2", "u3"])]


def test_a_turn_refused_at_the_send_because_the_session_waits_is_kept(tmp_path, monkeypatch):
    """A switch on another thread, between the look at the session and the request,
    has the session wait for the server's answer: the send is refused. The turn is
    kept, as one the upload itself held back would be, and goes up once the server says
    the session is on the record, though the conversation dropped it meanwhile."""
    turns, wire = _turns(tmp_path)

    def carried(session_id: str) -> str:
        assert turns.uploader.carry_private(["s-old"], session_id) is False
        return upload.OPEN

    with monkeypatch.context() as patch:
        patch.setattr(turns.uploader, "settle", carried)
        assert turns.upload(_turn(1), "s-1", timeout=5, finished=True) is False
    assert wire.tries == 0
    assert turns.upload(_turn(2), "s-1", timeout=5, finished=True) is True
    assert _sent(wire) == [("s-1", ["u1", "u2"])]


# ---------------------------------------------------------------------------
# Each session has its own look-back
# ---------------------------------------------------------------------------


def test_a_late_upload_for_one_session_does_not_use_up_the_look_back_of_another(tmp_path):
    turns, wire = _turns(tmp_path, lookback=1)  # for a session first met at an upload
    turns.start("s-new", 2)
    assert turns.upload(_turn(1) + _turn(2) + _turn(3), "s-old", timeout=5) is True
    new = [row for n in range(11, 16) for row in _turn(n)]
    assert turns.upload(new, "s-new", timeout=5) is True
    # Each with the turn before its first as context.
    assert _sent(wire) == [("s-old", ["u2", "u3"]), ("s-new", ["u13", "u14", "u15"])]


@pytest.mark.parametrize("held", [2, 3, 5])
def test_a_look_back_longer_than_the_conversation_takes_all_of_it(tmp_path, held):
    turns, wire = _turns(tmp_path, lookback=3)
    said = [row for n in range(1, held + 1) for row in _turn(n)]
    assert turns.upload(said, "s-1", timeout=5) is True
    # The last three, and the one before them as context.
    assert _sent(wire) == [("s-1", [f"u{n}" for n in range(max(1, held - 3), held + 1)])]


def test_only_the_first_upload_of_a_session_looks_back(tmp_path):
    """The window is for what was said before this process. A turn that ends later goes
    up however many have followed it by the time an upload runs."""
    turns, wire = _turns(tmp_path, lookback=1)
    assert turns.upload(_turn(1) + _turn(2), "s-1", timeout=5, finished=True) is True
    said = [row for n in range(1, 5) for row in _turn(n)]  # turn 3's own upload never ran
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    assert _sent(wire) == [("s-1", ["u1", "u2"]), ("s-1", ["u2", "u3", "u4"])]


def test_a_look_back_counts_finished_turns_only(tmp_path):
    turns, wire = _turns(tmp_path, lookback=2)
    said = _turn(1) + _turn(2) + _turn(3) + _turn(4, done=False)
    assert turns.upload(said, "s-1", timeout=5) is True
    assert _records(wire) == [_turn(1) + _turn(2) + _turn(3)]  # 1 is the context of 2


def test_a_session_that_holds_copies_is_never_looked_back_at(tmp_path):
    """Whatever look-back is asked for later, for the session itself or for one that
    goes on from it: the copies are in the conversation those sessions hold."""
    turns, wire = _turns(tmp_path)
    turns.start("s-1", 0)
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    turns.start("s-2", 10, continues=["s-0", "s-1"])  # the conversation gets a new id
    said = said + _turn(3)
    assert turns.upload(said, "s-2", timeout=5, finished=True) is True
    turns.start("s-2", 10)  # and that id is started again
    said = said + _turn(4)
    assert turns.upload(said, "s-2", timeout=5, finished=True) is True
    turns.start("s-3", 10, continues=["s-2"])  # a third id, which goes on from the second
    assert turns.upload(said + _turn(5), "s-3", timeout=5, finished=True) is True
    turns.start("s-other", 10)  # another conversation looks back as asked
    assert turns.upload(said, "s-other", timeout=5) is True
    assert _sent(wire) == [
        ("s-1", ["u2"]),
        ("s-2", ["u3"]),
        ("s-2", ["u4"]),
        ("s-3", ["u5"]),
        ("s-other", ["u1", "u2", "u3", "u4"]),
    ]


# ---------------------------------------------------------------------------
# In a session that holds copies, a turn goes up when it was seen to begin
# ---------------------------------------------------------------------------


def test_a_flush_leaves_the_turn_now_running_for_the_upload_that_ends_it(tmp_path):
    """A harness may flush before it says the turn starts (Hermes compresses first),
    and again while the turn runs. The turn is not written off for that: it goes up
    when it ends, since it was then seen to begin."""
    turns, wire = _turns(tmp_path, lookback=0)
    running = _turn(1) + _turn(2) + _turn(3, done=False)
    assert turns.upload(running, "s-1", timeout=5) is False
    assert turns.upload(running, "s-1", timeout=5) is False and wire.tries == 0
    said = _turn(1) + _turn(2) + _turn(3)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3)]


@pytest.mark.parametrize("own", [True, False], ids=["its own turn", "a turn from before"])
def test_a_flush_does_not_send_the_last_turn_whole_while_a_turn_runs(tmp_path, own):
    """A turn began and the last turn of the records is whole. It is the one that began
    when it has just ended and its upload is still to run. It is one from before when
    the harness began the turn itself, with a row in the user's place that opens no
    turn. The upload that ends the turn says which."""
    turns, wire = _turns(tmp_path, lookback=0)
    said = _turn(1) + _turn(2)
    if not own:
        said = said + _goes_on(1, shown=True)[:1]
    assert turns.upload(said, "s-1", timeout=5) is False and wire.tries == 0
    if not own:
        said = said + _goes_on(1, shown=True)[1:]
    assert turns.upload(said, "s-1", timeout=5, finished=True) is own
    assert _records(wire) == ([_turn(2)] if own else [])


def test_a_turn_sent_stays_sent_when_the_session_goes_private_and_comes_back(tmp_path):
    """An upload refused for a private session notes nothing of the turns it shows: the
    one sent before is still the context of the next, not a copy."""
    turns, wire = _turns(tmp_path, lookback=0)
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    marker = _mark(tmp_path, "s-1")  # `private on`
    assert turns.upload(said, "s-1", timeout=5) is False
    marker.unlink()
    assert turns.upload(said + _turn(3), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(2), _turn(2) + _turn(3)]  # 2 is the context of 3


def test_a_turn_kept_is_still_owed_when_the_session_goes_private_and_comes_back(tmp_path):
    turns, wire = _turns(tmp_path, lookback=0)  # no outbox: a failed send is kept
    said = _turn(1) + _turn(2)
    wire.fail = _DOWN
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False
    wire.fail = None
    marker = _mark(tmp_path, "s-1")
    assert turns.upload(said, "s-1", timeout=5) is False
    marker.unlink()
    assert turns.upload(said + _turn(3), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(2) + _turn(3)]


def test_a_turn_that_began_under_the_id_left_ends_under_the_new_one(tmp_path):
    """A compression gives the conversation a new id while the turn runs."""
    turns, wire = _turns(tmp_path)
    turns.start("s-1", 0)
    turns.start("s-2", 10, continues=["s-1"])
    assert turns.upload(_turn(1) + _turn(2), "s-2", timeout=5, finished=True) is True
    assert _sent(wire) == [("s-2", ["u2"])]


def test_a_turn_begun_under_one_id_ends_under_whichever_id_its_upload_names(tmp_path):
    """Turn starts are counted for the uploader, not per id: it serves one agent, whose
    turns run one at a time. Turn 2 begins under s-a. The conversation moves on to s-b
    and turn 3 begins there before turn 2's upload has run, under the id left. That
    upload ends turn 2, and the one after it ends turn 3: both go up."""
    turns, wire = _turns(tmp_path)
    turns.start("s-a", 0)
    turns.start("s-b", 0, continues=["s-a"])
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-a", timeout=5, finished=True) is True
    assert turns.upload(said + _turn(3), "s-b", timeout=5, finished=True) is True
    assert _sent(wire) == [("s-a", ["u2"]), ("s-b", ["u3"])]


def test_a_session_put_back_on_the_record_sends_the_turn_that_began_under_the_marker(tmp_path):
    """Private mode covers whole sessions: once the marker is gone every turn the
    session holds goes up (`synapse private off --forget` says so), also the one that
    began under it."""
    turns, wire = _turns(tmp_path)
    marker = _mark(tmp_path, "s-1")
    marker.unlink()
    assert turns.upload(_turn(1) + _turn(2), "s-1", timeout=5, finished=True) is True
    assert _sent(wire) == [("s-1", ["u1", "u2"])]


def test_a_turn_begun_while_the_server_does_not_answer_goes_up_when_it_says_on_the_record(
    tmp_path,
):
    """Also for a session that holds copies: the turn is kept as one seen to begin,
    not written off."""
    turns, wire = _turns(tmp_path)
    turns.start("s-new", 0)
    assert turns.uploader.carry_private(["s-old"], "s-new") is False
    wire.fail_private = _DOWN
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-new", timeout=5, finished=True) is False and wire.tries == 0
    wire.fail_private = None
    assert turns.upload(said + _turn(3), "s-new", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(2) + _turn(3)]


# ---------------------------------------------------------------------------
# A turn that goes on from a copy
# ---------------------------------------------------------------------------


def _goes_on(n: int, *, shown: bool) -> list[dict[str, Any]]:
    """What a harness adds when it begins a turn itself: a row in the user's place (or
    none in the records, with ``shown`` False: the adapter left it out) and the reply."""
    row = {"role": "user", "content": f"[process {n} ended]", "synthetic": True, "id": f"n{n}"}
    return [
        *([row] if shown else []),
        {"role": "assistant", "content": f"noted {n}", "id": f"r{n}"},
    ]


@pytest.mark.parametrize("shown", [True, False], ids=["a synthetic row", "no user row"])
def test_a_reply_the_user_did_not_ask_for_is_not_a_new_turn_in_a_session_with_copies(
    tmp_path, shown
):
    """No real user row opens what the harness began, so its reply lands on the turn
    before, a copy: the last turn of the records is complete, a turn began, and it is
    not that turn. Nor is it when the harness goes on once more. The next turn the
    user begins goes up, alone."""
    turns, wire = _turns(tmp_path, lookback=0)
    said = _turn(1) + _turn(2) + _goes_on(1, shown=shown)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False
    said = said + _goes_on(2, shown=shown)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False and wire.tries == 0
    assert turns.upload(said + _turn(3), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3)]


def test_a_reply_that_lands_on_a_turn_cut_short_is_not_a_new_turn_either(tmp_path):
    """The copied turn had no reply. The reply to what the harness began makes it
    complete: the user's message would go up with an answer to something else."""
    turns, wire = _turns(tmp_path, lookback=0)
    said = _turn(1) + _turn(2, done=False) + _goes_on(1, shown=True)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False and wire.tries == 0
    assert turns.upload(said + _turn(3), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3)]


def test_a_turn_a_flush_left_open_is_judged_when_it_ends(tmp_path):
    """A flush finds the last turn with no reply and leaves it for the next upload: it
    may be about to begin. This one is a copy of a turn cut short, and the reply to
    what the harness began closes it afterwards. The flush has not used up the
    look-back: the turn is judged when it ends, and stays out."""
    turns, wire = _turns(tmp_path, lookback=0)
    cut = _turn(1) + _turn(2, done=False)
    assert turns.upload(cut, "s-1", timeout=5) is False
    said = cut + _goes_on(1, shown=True)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False and wire.tries == 0
    assert turns.upload(said + _turn(3), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3)]


def test_a_turn_still_running_when_another_ends_is_not_the_one_that_ended(tmp_path):
    """A turn has ended, and the last turn of the records has no reply: it is not that
    turn. (It is a copy of a turn cut short, and what began here left no row.) It
    stays out when it gets a reply later."""
    turns, wire = _turns(tmp_path, lookback=0)
    cut = _turn(1) + _turn(2, done=False)
    assert turns.upload(cut, "s-1", timeout=5, finished=True) is False
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False and wire.tries == 0


def test_a_copied_turn_stays_out_when_the_harness_goes_on_from_it(tmp_path):
    """A flush used the look-back up before any turn ended: the turn that then goes on
    is still one from before."""
    turns, wire = _turns(tmp_path, lookback=0)
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-1", timeout=5) is False
    assert turns.upload(said + _goes_on(1, shown=True), "s-1", timeout=5, finished=True) is False
    assert wire.tries == 0


def test_a_row_the_harness_adds_after_a_reply_may_have_begun_a_turn(tmp_path):
    """A row in the user's place after the last reply, and nothing after it. It may be
    a task-list snapshot after the turn that has just ended. It may also be how the
    harness began a turn that left no other row (it failed), after a copied turn. The
    records do not say which, so the turn does not go up."""
    turns, wire = _turns(tmp_path, lookback=0)
    snapshot = {"role": "user", "content": "[tasks]", "synthetic": True, "id": "t1"}
    said = [*_turn(1), *_turn(2), snapshot]
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False and wire.tries == 0
    assert turns.upload(said + _turn(3), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3)]


def test_a_reply_the_user_did_not_ask_for_goes_up_with_a_turn_of_the_sessions_own(tmp_path):
    turns, wire = _turns(tmp_path)
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    said = said + _goes_on(1, shown=True)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    assert _records(wire)[1] == said  # turn 2 as it is now, after turn 1 as its context


# ---------------------------------------------------------------------------
# A flush that shows part of the conversation
# ---------------------------------------------------------------------------


def _back(tmp_path):
    """A session that holds copies, as a harness that says when turns start
    (``begins``) starts it."""
    turns, wire = _turns(tmp_path, lookback=10)
    turns.start("s-1", 0)
    return turns, wire


@pytest.mark.parametrize("shown", [0, 2, 4], ids=["no turn", "the first two", "all four"])
def test_a_flush_notes_the_turns_it_is_shown_and_uses_up_no_look_back(tmp_path, shown):
    """A compression hands a flush the rows it is about to summarise, which need not be
    all of them (Hermes: `/compress here N`). The turns it was not shown are copies as
    well: they stay out when the next turn ends, which shows them all."""
    turns, wire = _back(tmp_path)
    said = [row for n in range(1, 5) for row in _turn(n)]
    assert turns.upload(said[: 2 * shown], "s-1", timeout=5) is False
    assert turns.upload(said + _turn(5), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(5)]


def test_a_flush_that_shows_the_rest_sends_none_of_it(tmp_path):
    """Two flushes and no turn in between (the Hermes gateway: one before the
    compression, with the rows to be summarised, and one as the agent is released,
    with all of them)."""
    turns, wire = _back(tmp_path)
    said = [row for n in range(1, 5) for row in _turn(n)]
    assert turns.upload(said[:4], "s-1", timeout=5) is False
    assert turns.upload(said, "s-1", timeout=5) is False and wire.tries == 0


def test_a_turn_whose_upload_never_came_goes_up_with_the_next(tmp_path):
    """No upload ended turn 4 (Hermes sends none for a turn cut short). Both go up when
    turn 5 ends: once its first turn has ended, a session that holds copies sends
    every turn after them."""
    turns, wire = _back(tmp_path)
    said = _turn(1) + _turn(2) + _turn(3)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    said = said + _turn(4) + _turn(5)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3), _turn(3) + _turn(4) + _turn(5)]


def test_a_flush_sends_a_turn_whose_own_upload_has_not_run(tmp_path):
    """The session's first turn has ended, so what it holds is noted. Turn 4 has ended
    since, and the upload that ends it has not run when the harness flushes (it
    exits): the flush sends it."""
    turns, wire = _back(tmp_path)
    said = _turn(1) + _turn(2) + _turn(3)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    assert turns.upload(said + _turn(4), "s-1", timeout=5) is True
    assert _records(wire) == [_turn(3), _turn(3) + _turn(4)]


def test_a_flush_of_a_session_put_back_on_the_record_sends_what_it_holds(tmp_path):
    """Private mode covers whole sessions. One marked private and put back holds no
    turn to keep out: its flush sends the turns not sent yet, the one begun while it
    was private too."""
    turns, wire = _turns(tmp_path, lookback=10)
    assert turns.upload(_turn(1), "s-1", timeout=5, finished=True) is True
    marker = _mark(tmp_path, "s-1")
    marker.unlink()
    assert turns.upload(_turn(1) + _turn(2), "s-1", timeout=5) is True
    assert _records(wire) == [_turn(1), _turn(1) + _turn(2)]


# ---------------------------------------------------------------------------
# Turn starts are counted for the uploader, whichever ids a start and its end name
# ---------------------------------------------------------------------------


def test_an_upload_that_ends_no_turn_uses_up_no_turn_start(tmp_path):
    """A flush while the turn runs: the turn's own upload still finds its start."""
    turns, wire = _turns(tmp_path)
    turns.start("s-1", 0)
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-other", timeout=5) is True  # a flush, for another session
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    assert _sent(wire) == [("s-other", ["u1", "u2"]), ("s-1", ["u2"])]


# ---------------------------------------------------------------------------
# A session said to be new
# ---------------------------------------------------------------------------


def test_a_new_session_that_holds_the_turns_begun_in_it_uploads_them_all(tmp_path):
    turns, wire = _turns(tmp_path)
    # A flush before the harness says turn 3 starts.
    said = _turn(1) + _turn(2) + _turn(3, done=False)
    assert turns.upload(said, "s-1", timeout=5) is True
    assert turns.upload(_turn(1) + _turn(2) + _turn(3), "s-1", timeout=5, finished=True) is True
    assert _sent(wire) == [("s-1", ["u1", "u2"]), ("s-1", ["u2", "u3"])]


def test_a_new_session_is_checked_at_its_first_upload_only(tmp_path):
    """Its first upload showed no more turns than were begun: the doubt is settled. A
    turn that appears later is new by where it stands, also one whose start was not
    seen."""
    turns, wire = _turns(tmp_path)
    assert turns.upload(_turn(1), "s-1", timeout=5, finished=True) is True
    said = _turn(1) + _turn(2) + _turn(3)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    assert _sent(wire) == [("s-1", ["u1"]), ("s-1", ["u1", "u2", "u3"])]


def test_the_turns_begun_under_the_id_left_count_for_the_new_one(tmp_path):
    turns, wire = _turns(tmp_path)
    turns.start("s-1")
    turns.start("s-2", 10, continues=["s-1"])
    said = _turn(1, done=False) + _turn(2)
    assert turns.upload(said, "s-2", timeout=5, finished=True) is True
    turns.start("s-3", 10, continues=["s-2"])  # the doubt is settled: s-3 looks back
    assert turns.upload(said, "s-3", timeout=5) is True
    assert _sent(wire) == [("s-2", ["u2"]), ("s-3", ["u2"])]


def test_a_session_that_holds_copies_is_not_looked_back_at_under_a_later_id(tmp_path):
    """What is known of a session stays with the conversation in this process: the id it
    goes on under asks for a look-back in vain."""
    turns, wire = _turns(tmp_path)
    turns.start("s-1", 0)
    copies = _turn(1) + _turn(2)
    assert turns.upload(copies + _turn(3), "s-1", timeout=5, finished=True) is True
    turns.start("s-2", 10, continues=["s-1"])
    said = copies + _turn(3) + _turn(4)
    assert turns.upload(said, "s-2", timeout=5, finished=True) is True
    assert _sent(wire) == [("s-1", ["u3"]), ("s-2", ["u4"])]


def test_a_session_the_harness_knows_to_hold_copies_is_not_looked_back_at(tmp_path):
    """A process built later was not there when the copies were found. The harness
    says so from what it kept (``keep_out``): for an id the uploader has not met, and
    for one it was told to look back at (a session resumed in place)."""
    turns, wire = _turns(tmp_path, lookback=10)
    turns.keep_out("s-1")
    turns.start("s-2", 10)
    turns.keep_out("s-2")
    copies = _turn(1) + _turn(2)
    for sid in ("s-1", "s-2"):
        assert turns.upload(copies, sid, timeout=5) is False  # a flush before the turn
        assert turns.upload(copies + _turn(3), sid, timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3), _turn(3)] and wire.tries == 2


def test_keeping_a_session_out_forgets_nothing_it_sent(tmp_path):
    """The harness may say so in the middle of a session's life: the turns sent stay
    sent, as the context of the next one too."""
    turns, wire = _turns(tmp_path)
    assert turns.upload(_turn(1), "s-1", timeout=5, finished=True) is True
    turns.keep_out("s-1")
    assert turns.upload(_turn(1) + _turn(2), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(1), _turn(1) + _turn(2)]


def test_keeping_a_session_out_forgets_nothing_it_keeps_either(tmp_path):
    """A turn whose send failed is kept, and is the session's own: it is no copy when
    the harness then says the session holds some, and it is the context of the next."""
    turns, wire = _turns(tmp_path)  # no outbox: a failed send is kept
    wire.fail = _DOWN
    assert turns.upload(_turn(1), "s-1", timeout=5, finished=True) is False
    wire.fail = None
    turns.keep_out("s-1")
    assert turns.upload(_turn(1), "s-1", timeout=5) is True  # a flush: it notes what it is shown
    assert turns.upload(_turn(1) + _turn(2), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(1), _turn(1) + _turn(2)]


def test_a_session_kept_out_already_is_left_as_it_is(tmp_path):
    """The harness reads its note at every upload and says so each time. The turns
    from before were noted once, at the first upload that showed them all: a turn of
    the session's own that no upload came for still goes up with the next one."""
    turns, wire = _turns(tmp_path)
    turns.keep_out("s-1")
    assert turns.upload(_turn(1) + _turn(2), "s-1", timeout=5, finished=True) is True
    turns.keep_out("s-1")
    said = _turn(1) + _turn(2) + _turn(3) + _turn(4)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    assert _sent(wire) == [("s-1", ["u2"]), ("s-1", ["u2", "u3", "u4"])]


def test_a_session_that_names_the_one_it_goes_on_from_is_not_said_to_be_new(tmp_path):
    """A branch holds copies and says whose: what that session is decides whether they
    may go up (``Uploader.carry_private``), and here it is on the record."""
    turns, wire = _turns(tmp_path)
    turns.start("s-1", continues=["s-0"])
    assert turns.upload(_turn(1) + _turn(2), "s-1", timeout=5) is True
    assert _sent(wire) == [("s-1", ["u1", "u2"])]


def test_a_harness_that_does_not_say_when_turns_start_is_taken_at_its_word(tmp_path):
    turns, wire = _turns(tmp_path)
    assert turns.upload(_turn(1) + _turn(2), "s-1", timeout=5) is True
    assert _sent(wire) == [("s-1", ["u1", "u2"])]


# ---------------------------------------------------------------------------
# The server's answer, and who got it
# ---------------------------------------------------------------------------


def test_a_marker_of_its_own_is_not_taken_for_the_servers_answer(tmp_path):
    """`synapse private on` for a session that waits for the server's answer. The answer
    is that the session it continues is on the record, and the marker is the session's
    own: after `off --forget` every turn it holds is sent, as for any session marked by
    hand."""
    turns, wire = _turns(tmp_path)
    assert turns.uploader.carry_private(["s-old"], "s-new") is False
    marker = _mark(tmp_path, "s-new")
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-new", timeout=5, finished=True) is False
    assert wire.requests == [_get("s-old")] and wire.tries == 0
    marker.unlink()
    assert turns.upload(said + _turn(3), "s-new", timeout=5, finished=True) is True
    assert _sent(wire) == [("s-new", ["u1", "u2", "u3"])]


def _answered_meanwhile(turns: Any, wire: Any, monkeypatch: Any, said: list[Any]) -> bool:
    """One upload of s-new finds the server down. Before it takes its turns up, another
    upload of the session, on another thread, gets the server's answer. Returns what
    the first upload answers."""
    up = turns.uploader
    real = up.settle

    def settle(session_id: str) -> str:
        wire.fail_private = _DOWN
        standing = real(session_id)
        assert standing == upload.WAITING
        wire.fail_private = None
        monkeypatch.setattr(up, "settle", real)
        other = threading.Thread(
            target=turns.upload, args=(said, session_id), kwargs={"timeout": 5}
        )
        other.start()
        other.join(5)
        return standing

    monkeypatch.setattr(up, "settle", settle)
    return bool(turns.upload(said, "s-new", timeout=5, finished=True))


def test_turns_another_upload_sent_meanwhile_are_not_kept_or_sent_again(tmp_path, monkeypatch):
    turns, wire = _turns(tmp_path)
    assert turns.uploader.carry_private(["s-old"], "s-new") is False
    said = _turn(1) + _turn(2) + _turn(3)
    assert _answered_meanwhile(turns, wire, monkeypatch, said) is False
    assert _records(wire) == [said]
    assert turns.upload(said + _turn(4), "s-new", timeout=5, finished=True) is True
    assert _records(wire)[1:] == [_turn(3) + _turn(4)]  # 3 as the context of 4
