"""What a ``TurnUploader`` remembers (plugin/synapse_client/turn_upload.py).

A harness names the session at every upload and may name several over its life: the one
it was moved to, a late upload for the one it left. Turns are remembered per session
id, and turn starts are counted for the uploader. Pinned here:
  * turns kept for one session (held back for the server's answer, or lost with no
    outbox) go up under that session's id or not at all;
  * each session has its own look-back, and a session that was off the record is never
    looked back at, under any id it goes on with;
  * of the turns such a session holds when it is first found on the record, one goes
    up only when it was seen to begin there (``begin``): not one that began off it,
    holds more than its own exchange, or was there before any turn start was seen;
  * an upload that ends no turn notes the turns it is shown and settles nothing more,
    and one that runs late is not taken for the end of another turn;
  * turn starts are counted for the uploader, whichever ids a start and its end name:
    a turn begun off the record is counted once however often sessions are started and
    through however many ids it moves, an upload that runs one turn late is not taken
    for the end of the turn begun since, and until the end of a turn begun off the
    record was seen and what the session holds noted again no flush sends a new turn;
  * a session said to be new is not believed when its turn starts show otherwise, and
    the harness is told (``copied``), so that it can say so later (``keep_out``);
  * an upload acts on the server's answer only when it got the answer itself
    (``Uploader.settle``).

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


def test_a_session_that_was_off_the_record_is_never_looked_back_at(tmp_path):
    """Whatever look-back is asked for later, for the session itself or for one that
    goes on from it: its turns are in the conversation those sessions hold."""
    turns, wire = _turns(tmp_path)
    turns.start("s-1", 0)
    said = _turn(1) + _turn(2)
    turns.begin("s-1")
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    turns.start("s-2", 10, continues=["s-0", "s-1"])  # the conversation gets a new id
    said = said + _turn(3)
    turns.begin("s-2")
    assert turns.upload(said, "s-2", timeout=5, finished=True) is True
    turns.start("s-2", 10)  # and that id is started again
    said = said + _turn(4)
    turns.begin("s-2")
    assert turns.upload(said, "s-2", timeout=5, finished=True) is True
    turns.start("s-3", 10, continues=["s-2"])  # a third id, which goes on from the second
    turns.begin("s-3")
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
# After time off the record, a turn goes up when it was seen to begin on it
# ---------------------------------------------------------------------------


def test_a_turn_no_start_was_seen_for_is_one_from_before(tmp_path):
    """The shape of the records does not tell the turn that has just ended from one
    spoken off the record: without a turn start, nothing they hold is uploaded. A turn
    that comes after is new by where it stands."""
    turns, wire = _turns(tmp_path, lookback=0)
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False and wire.tries == 0
    assert turns.upload(said + _turn(3), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3)]  # alone: 2 is no context


def test_a_turn_that_began_off_the_record_does_not_go_up_when_it_ends_on_it(tmp_path):
    """`off --forget` while the turn runs."""
    turns, wire = _turns(tmp_path, lookback=0)
    marker = _mark(tmp_path, "s-1")
    turns.begin("s-1")
    marker.unlink()
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False and wire.tries == 0
    turns.begin("s-1")
    assert turns.upload(said + _turn(3), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3)]


def test_an_upload_that_runs_late_is_not_the_end_of_the_turn_begun_since(tmp_path):
    """A harness uploads on a worker: the upload that ends a turn spoken off the record
    may run after the next turn has begun, on it. The turn it ends is still the one
    that began off the record."""
    turns, wire = _turns(tmp_path, lookback=0)
    marker = _mark(tmp_path, "s-1")
    turns.begin("s-1")  # turn 2, off the record
    marker.unlink()
    turns.begin("s-1")  # turn 3, on it: turn 2's upload has not run yet
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False and wire.tries == 0
    assert turns.upload(said + _turn(3), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3)]


def test_a_turn_still_running_when_it_was_refused_stays_off_the_record_when_it_ends(tmp_path):
    """A flush while the session is private sees the turn half done. However it ends,
    it is the same turn: its opening row says so, where its key has changed."""
    turns, wire = _turns(tmp_path, lookback=0)
    marker = _mark(tmp_path, "s-1")
    turns.begin("s-1")
    assert turns.upload(_turn(1) + _turn(2, done=False), "s-1", timeout=5) is False
    marker.unlink()
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-1", timeout=5) is False  # a flush on the record
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False and wire.tries == 0
    turns.begin("s-1")
    assert turns.upload(said + _turn(3), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3)]


def test_a_flush_leaves_the_turn_now_running_for_the_upload_that_ends_it(tmp_path):
    """A harness may flush before it says the turn starts (Hermes compresses first),
    and again while the turn runs. The turn is not written off for that: it goes up
    when it ends, since it was then seen to begin on the record."""
    turns, wire = _turns(tmp_path, lookback=0)
    running = _turn(1) + _turn(2) + _turn(3, done=False)
    assert turns.upload(running, "s-1", timeout=5) is False
    turns.begin("s-1")
    assert turns.upload(running, "s-1", timeout=5) is False and wire.tries == 0
    said = _turn(1) + _turn(2) + _turn(3)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3)]


@pytest.mark.parametrize("own", [True, False], ids=["its own turn", "a turn from before"])
def test_a_flush_does_not_send_the_last_turn_whole_while_a_turn_runs(tmp_path, own):
    """A turn began on the record and the last turn of the records is whole. It is the
    one that began when it has just ended and its upload is still to run. It is one
    from before when the harness began the turn itself, with a row in the user's place
    that opens no turn. The upload that ends the turn says which."""
    turns, wire = _turns(tmp_path, lookback=0)
    turns.begin("s-1")
    said = _turn(1) + _turn(2)
    if not own:
        said = said + _goes_on(1, shown=True)[:1]
    assert turns.upload(said, "s-1", timeout=5) is False and wire.tries == 0
    if not own:
        said = said + _goes_on(1, shown=True)[1:]
    assert turns.upload(said, "s-1", timeout=5, finished=True) is own
    assert _records(wire) == ([_turn(2)] if own else [])


def test_a_turn_sent_on_the_record_stays_on_it_when_the_session_goes_private_again(tmp_path):
    turns, wire = _turns(tmp_path, lookback=0)
    turns.begin("s-1")
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    marker = _mark(tmp_path, "s-1")  # `private on` once more
    assert turns.upload(said, "s-1", timeout=5) is False
    marker.unlink()
    turns.begin("s-1")
    assert turns.upload(said + _turn(3), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(2), _turn(2) + _turn(3)]  # 2 is the context of 3


def test_a_turn_kept_on_the_record_stays_on_it_when_the_session_goes_private_again(tmp_path):
    turns, wire = _turns(tmp_path, lookback=0)  # no outbox: a failed send is kept
    turns.begin("s-1")
    said = _turn(1) + _turn(2)
    wire.fail = _DOWN
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False
    wire.fail = None
    marker = _mark(tmp_path, "s-1")
    assert turns.upload(said, "s-1", timeout=5) is False
    marker.unlink()
    turns.begin("s-1")
    assert turns.upload(said + _turn(3), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(2) + _turn(3)]


def test_a_turn_spoken_while_the_session_is_private_again_stays_off_the_record(tmp_path):
    """Back on the record and uploading, then `private on` for one whole turn. Its
    upload is refused, and that is where the turn is noted: once the marker is gone it
    would read as a turn not sent yet."""
    turns, wire = _turns(tmp_path, lookback=0)
    turns.begin("s-1")
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    marker = _mark(tmp_path, "s-1")
    turns.begin("s-1")
    said = said + _turn(3)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False
    marker.unlink()
    turns.begin("s-1")
    assert turns.upload(said + _turn(4), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(2), _turn(4)]


def test_a_turn_sent_on_the_record_stays_on_it_when_a_turn_begun_off_it_ends(tmp_path):
    """`private on`, a turn begins, `off --forget`, and the turn is cut short. The
    upload that ends it notes the turns off the record once more. The one sent before
    is not one of them: it is the context of the turn after."""
    turns, wire = _turns(tmp_path, lookback=0)
    turns.begin("s-1")
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    marker = _mark(tmp_path, "s-1")
    turns.begin("s-1")  # turn 3, off the record
    marker.unlink()
    said = said + _turn(3, done=False)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False
    turns.begin("s-1")
    assert turns.upload(said + _turn(4), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(2), _turn(2) + _turn(4)]


def test_a_turn_that_began_under_the_id_left_ends_under_the_new_one(tmp_path):
    """A compression gives the conversation a new id while the turn runs."""
    turns, wire = _turns(tmp_path)
    turns.start("s-1", 0)
    turns.begin("s-1")
    turns.start("s-2", 10, continues=["s-1"])
    assert turns.upload(_turn(1) + _turn(2), "s-2", timeout=5, finished=True) is True
    assert _sent(wire) == [("s-2", ["u2"])]


def test_a_turn_that_began_off_the_record_under_the_id_left_is_counted_under_the_new_one(
    tmp_path,
):
    """Its upload runs late, after a turn has begun on the record under the new id."""
    turns, wire = _turns(tmp_path)
    turns.start("s-1", 0)
    _mark(tmp_path, "s-1")
    turns.begin("s-1")  # turn 2, off the record
    turns.start("s-2", 0, continues=["s-1"])
    turns.begin("s-2")  # turn 3, on it
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-2", timeout=5, finished=True) is False and wire.tries == 0
    assert turns.upload(said + _turn(3), "s-2", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3)]


def test_a_turn_begun_before_the_server_said_private_began_off_the_record(tmp_path):
    """A session that waits for the server's answer counts as on the record until it
    comes. The answer is "private", and it comes at a flush, with the turn still
    running. The conversation then goes on under a new id, where nothing is written off
    yet: the turn is not one seen to begin on the record there either."""
    turns, wire = _turns(tmp_path)
    wire.rows.add("s-old")
    assert turns.uploader.carry_private(["s-old"], "s-new") is False
    turns.begin("s-new")
    assert turns.upload(_turn(1)[:1], "s-new", timeout=5) is False  # a flush, mid-turn
    assert (tmp_path / "private" / "s-new").is_file()
    turns.start("s-next", 0, continues=["s-new"])
    _forget(wire, tmp_path, "s-new")
    assert turns.upload(_turn(1), "s-next", timeout=5, finished=True) is False
    assert wire.tries == 0
    turns.begin("s-next")
    assert turns.upload(_turn(1) + _turn(2), "s-next", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(2)]


def test_a_turn_begun_under_a_marker_is_counted_once_when_the_server_says_private_too(
    tmp_path,
):
    """The turn began under a marker, which `synapse private off` then took away. The
    id the conversation goes on with asks the server, which still says private. One
    turn began off the record, not two: once it has ended, the next turn begun on the
    record goes up."""
    turns, wire = _turns(tmp_path)
    marker = _mark(tmp_path, "s-1")
    turns.begin("s-1")
    marker.unlink()
    wire.rows.add("s-1")
    assert turns.uploader.carry_private(["s-1"], "s-2") is False
    turns.start("s-2", 10, continues=["s-1"])
    assert turns.upload(_turn(1), "s-2", timeout=5, finished=True) is False
    assert (tmp_path / "private" / "s-2").is_file() and wire.tries == 0
    _forget(wire, tmp_path, "s-2")
    turns.begin("s-2")
    assert turns.upload(_turn(1) + _turn(2), "s-2", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(2)]


def test_a_turn_begun_before_the_server_said_private_is_counted_until_its_own_end(tmp_path):
    """The answer comes at a flush that shows no row of the turn still running, and
    `off --forget` follows. The next turn begins on the record before the upload of the
    first has run. That upload ends the turn begun off the record, as it turned out,
    and not the one begun since: it stays out."""
    turns, wire = _turns(tmp_path)
    wire.rows.add("s-old")
    assert turns.uploader.carry_private(["s-old"], "s-new") is False
    turns.begin("s-new")  # turn 1, while the server has not said
    assert turns.upload([], "s-new", timeout=5) is False  # a flush: "private"
    assert (tmp_path / "private" / "s-new").is_file()
    _forget(wire, tmp_path, "s-new")
    turns.begin("s-new")  # turn 2, on the record
    assert turns.upload(_turn(1), "s-new", timeout=5, finished=True) is False
    assert wire.tries == 0
    assert turns.upload(_turn(1) + _turn(2), "s-new", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(2)]


def test_a_turn_that_ended_before_the_server_said_private_is_not_one_still_running(tmp_path):
    """The turn ends while the server does not answer, and a flush gets the answer:
    private. No turn was running then, so none is counted as begun off the record, and
    the first turn begun back on the record goes up."""
    turns, wire = _turns(tmp_path)
    wire.rows.add("s-old")
    assert turns.uploader.carry_private(["s-old"], "s-new") is False
    turns.begin("s-new")
    wire.fail_private = _DOWN
    assert turns.upload(_turn(1), "s-new", timeout=5, finished=True) is False
    wire.fail_private = None
    assert turns.upload(_turn(1), "s-new", timeout=5) is False
    assert (tmp_path / "private" / "s-new").is_file() and wire.tries == 0
    _forget(wire, tmp_path, "s-new")
    turns.begin("s-new")
    assert turns.upload(_turn(1) + _turn(2), "s-new", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(2)]


def test_a_turn_begun_under_one_id_ends_under_whichever_id_its_upload_names(tmp_path):
    """Turn starts are counted for the uploader, not per id: it serves one agent, whose
    turns run one at a time. Turn 2 begins on the record under s-a. The conversation
    moves on to s-b and turn 3 begins there before turn 2's upload has run, under the
    id left. That upload ends turn 2, and the one after it ends turn 3: both go up."""
    turns, wire = _turns(tmp_path)
    turns.start("s-a", 0)
    turns.begin("s-a")  # turn 2
    turns.start("s-b", 0, continues=["s-a"])
    turns.begin("s-b")  # turn 3: turn 2's upload has not run yet
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-a", timeout=5, finished=True) is True
    assert turns.upload(said + _turn(3), "s-b", timeout=5, finished=True) is True
    assert _sent(wire) == [("s-a", ["u2"]), ("s-b", ["u3"])]


def test_a_turn_stopped_on_the_record_stays_counted_as_the_turn_before(tmp_path):
    """Nothing tells the uploader that a turn was stopped (Hermes sends no upload for
    one): its start stays counted, as the turn before, from then on. On the record that
    costs nothing. A turn begun off the record is counted with it, and its own upload
    takes one of the two: the first turn back on the record is taken for the other, and
    is not uploaded. The one after it is. That upload was the first turn back's own, so
    one start stays counted still, and the next stretch off the record costs the same."""
    turns, wire = _turns(tmp_path)
    turns.start("s-1", 0)
    turns.begin("s-1")  # turn 2, on the record, stopped: no upload ends it
    turns.begin("s-1")  # turn 3
    said = _turn(1) + _turn(2, done=False) + _turn(3)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    for off in (4, 7):
        marker = _mark(tmp_path, "s-1")
        turns.begin("s-1")  # off the record
        said = said + _turn(off)
        assert turns.upload(said, "s-1", timeout=5, finished=True) is False
        marker.unlink()
        turns.begin("s-1")  # back on the record
        said = said + _turn(off + 1)
        assert turns.upload(said, "s-1", timeout=5, finished=True) is False  # the cost
        turns.begin("s-1")
        said = said + _turn(off + 2)
        assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3), _turn(6), _turn(9)]


def test_a_turn_begun_off_the_record_a_second_time_does_not_go_up_either(tmp_path):
    """The session was back on the record and uploading when it was marked private
    again. A turn that begins then, and ends once the marker is gone, began off the
    record: no upload saw it while the session was private, and it stays out."""
    turns, wire = _turns(tmp_path)
    turns.start("s-1", 0)
    turns.begin("s-1")
    assert turns.upload(_turn(1) + _turn(2), "s-1", timeout=5, finished=True) is True
    marker = _mark(tmp_path, "s-1")
    turns.begin("s-1")
    marker.unlink()
    said = _turn(1) + _turn(2) + _turn(3)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False
    turns.begin("s-1")
    assert turns.upload(said + _turn(4), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(2), _turn(4)]


def test_a_session_marked_by_hand_sends_the_turn_that_began_under_the_marker(tmp_path):
    """An ordinary session, never off the record before this process met it: once the
    marker is gone every turn it holds goes up (`synapse private off` says so), also
    the one that began under it."""
    turns, wire = _turns(tmp_path)
    marker = _mark(tmp_path, "s-1")
    turns.begin("s-1")
    marker.unlink()
    assert turns.upload(_turn(1) + _turn(2), "s-1", timeout=5, finished=True) is True
    assert _sent(wire) == [("s-1", ["u1", "u2"])]


def test_a_turn_begun_while_the_server_does_not_answer_goes_up_when_it_says_on_the_record(
    tmp_path,
):
    """Also for a session that was off the record: the turn is kept as one seen to
    begin on it, not written off."""
    turns, wire = _turns(tmp_path)
    turns.start("s-new", 0)
    assert turns.uploader.carry_private(["s-old"], "s-new") is False
    wire.fail_private = _DOWN
    turns.begin("s-new")
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-new", timeout=5, finished=True) is False and wire.tries == 0
    wire.fail_private = None
    turns.begin("s-new")
    assert turns.upload(said + _turn(3), "s-new", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(2) + _turn(3)]


def test_a_turn_begun_under_the_sessions_own_marker_began_off_the_record_whatever_is_asked(
    tmp_path,
):
    """The uploader still waits for the server to say whether the session this one goes
    on from is private. That open question is not "private" (the test above). The
    session's own marker is: a turn begun under it began off the record, and stays out
    when `off --forget` comes before its upload."""
    turns, wire = _turns(tmp_path)
    turns.start("s-new", 0)
    assert turns.uploader.carry_private(["s-old"], "s-new") is False  # the server is asked
    marker = _mark(tmp_path, "s-new")  # `synapse private on`, for the session itself
    turns.begin("s-new")
    marker.unlink()
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-new", timeout=5, finished=True) is False and wire.tries == 0
    turns.begin("s-new")
    assert turns.upload(said + _turn(3), "s-new", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3)]


# ---------------------------------------------------------------------------
# A turn that goes on from one spoken off the record
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
def test_a_reply_the_user_did_not_ask_for_is_not_a_new_turn_after_time_off_the_record(
    tmp_path, shown
):
    """No real user row opens what the harness began, so its reply lands on the turn
    before, spoken off the record: the last turn of the records is complete, a turn
    began on the record, and it is not that turn. Nor is it when the harness goes on
    once more. The next turn the user begins goes up, alone."""
    turns, wire = _turns(tmp_path, lookback=0)
    said = _turn(1) + _turn(2) + _goes_on(1, shown=shown)
    turns.begin("s-1")
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False
    said = said + _goes_on(2, shown=shown)
    turns.begin("s-1")
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False and wire.tries == 0
    turns.begin("s-1")
    assert turns.upload(said + _turn(3), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3)]


def test_a_reply_that_lands_on_a_turn_cut_short_is_not_a_new_turn_either(tmp_path):
    """The turn spoken off the record got no reply. The reply to what the harness began
    makes it complete: the user's message would go up with an answer to something else."""
    turns, wire = _turns(tmp_path, lookback=0)
    said = _turn(1) + _turn(2, done=False) + _goes_on(1, shown=True)
    turns.begin("s-1")
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False and wire.tries == 0
    turns.begin("s-1")
    assert turns.upload(said + _turn(3), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3)]


def test_a_turn_a_flush_left_open_is_judged_when_it_ends(tmp_path):
    """A flush finds the last turn with no reply and leaves it for the next upload: it
    may be about to begin. This one was cut short off the record, and the reply to what
    the harness began closes it afterwards. The flush has not used up the look-back:
    the turn is judged when it ends, and stays off the record."""
    turns, wire = _turns(tmp_path, lookback=0)
    cut = _turn(1) + _turn(2, done=False)
    assert turns.upload(cut, "s-1", timeout=5) is False
    turns.begin("s-1")
    said = cut + _goes_on(1, shown=True)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False and wire.tries == 0
    turns.begin("s-1")
    assert turns.upload(said + _turn(3), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3)]


def test_a_turn_still_running_when_another_ends_is_not_the_one_that_ended(tmp_path):
    """A turn has ended on the record, and the last turn of the records has no reply:
    it is not that turn. (It was cut short off the record, and what began on the record
    left no row.) It stays off the record when it gets a reply later."""
    turns, wire = _turns(tmp_path, lookback=0)
    turns.begin("s-1")
    cut = _turn(1) + _turn(2, done=False)
    assert turns.upload(cut, "s-1", timeout=5, finished=True) is False
    turns.begin("s-1")
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False and wire.tries == 0


def test_a_turn_spoken_off_the_record_stays_there_when_the_harness_goes_on_from_it(tmp_path):
    """A flush used the look-back up before any turn ended: the turn that then goes on
    is still one from before."""
    turns, wire = _turns(tmp_path, lookback=0)
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-1", timeout=5) is False
    turns.begin("s-1")
    assert turns.upload(said + _goes_on(1, shown=True), "s-1", timeout=5, finished=True) is False
    assert wire.tries == 0


def test_a_row_the_harness_adds_after_a_reply_may_have_begun_a_turn(tmp_path):
    """A row in the user's place after the last reply, and nothing after it. It may be
    a task-list snapshot after the turn that has just ended. It may also be how the
    harness began a turn that left no other row (it failed), after a turn spoken off
    the record. The records do not say which, so the turn does not go up."""
    turns, wire = _turns(tmp_path, lookback=0)
    snapshot = {"role": "user", "content": "[tasks]", "synthetic": True, "id": "t1"}
    said = [*_turn(1), *_turn(2), snapshot]
    turns.begin("s-1")
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False and wire.tries == 0
    turns.begin("s-1")
    assert turns.upload(said + _turn(3), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3)]


def test_a_reply_the_user_did_not_ask_for_goes_up_with_a_turn_on_the_record(tmp_path):
    turns, wire = _turns(tmp_path)
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    said = said + _goes_on(1, shown=True)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    assert _records(wire)[1] == said  # turn 2 as it is now, after turn 1 as its context


# ---------------------------------------------------------------------------
# A flush that shows part of the conversation, and uploads that run late
# ---------------------------------------------------------------------------


def _back(tmp_path):
    """A session that was off the record, as a harness that says when turns start
    (``begins``) starts it. Nothing marks it now: it is back on the record."""
    turns, wire = _turns(tmp_path, begins=True, lookback=10)
    turns.start("s-1", 0)
    return turns, wire


@pytest.mark.parametrize("shown", [0, 2, 4], ids=["no turn", "the first two", "all four"])
def test_a_flush_notes_the_turns_it_is_shown_and_uses_up_no_look_back(tmp_path, shown):
    """A compression hands a flush the rows it is about to summarise, which need not be
    all of them (Hermes: `/compress here N`). The turns it was not shown were spoken off
    the record as well: they stay out when the next turn ends, which shows them all."""
    turns, wire = _back(tmp_path)
    said = [row for n in range(1, 5) for row in _turn(n)]
    assert turns.upload(said[: 2 * shown], "s-1", timeout=5) is False
    turns.begin("s-1", earlier=True)
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


def test_an_upload_that_runs_late_is_not_the_end_of_a_turn_begun_off_the_record_since(tmp_path):
    """The other way round from the test above. Turn 4 began on the record and its
    upload runs late: after `private on`, the start of turn 5 and `off --forget`. The
    upload that ends turn 4 is not taken for the end of turn 5, which would leave turn
    5's own upload with nothing to say it began off the record. Which of the two an
    upload ends is not known here, so neither goes up."""
    turns, wire = _back(tmp_path)
    said = _turn(1) + _turn(2) + _turn(3)
    turns.begin("s-1")
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    turns.begin("s-1")  # turn 4, on the record
    marker = _mark(tmp_path, "s-1")
    turns.begin("s-1")  # turn 5, off it: turn 4's upload has not run yet
    marker.unlink()
    said = said + _turn(4)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False
    assert turns.upload(said + _turn(5), "s-1", timeout=5, finished=True) is False
    turns.begin("s-1")
    assert turns.upload(said + _turn(5) + _turn(6), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3), _turn(6)]


def test_a_turn_begun_off_the_record_is_counted_once_when_the_turn_before_has_ended(tmp_path):
    """The turn before it was seen to end, so one upload is still to come, not two:
    the turn after it goes up."""
    turns, wire = _back(tmp_path)
    said = _turn(1) + _turn(2) + _turn(3)
    turns.begin("s-1")
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    marker = _mark(tmp_path, "s-1")
    turns.begin("s-1")  # turn 4, off the record
    marker.unlink()
    said = said + _turn(4)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False
    turns.begin("s-1")
    assert turns.upload(said + _turn(5), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3), _turn(5)]


def test_a_turn_begun_on_the_record_is_not_counted_for_the_turn_before_it(tmp_path):
    """No upload ended turn 4 (Hermes sends none for a turn cut short). Turn 5 begins on
    the record as turn 4 did: nothing began off it, and both go up when turn 5 ends."""
    turns, wire = _back(tmp_path)
    said = _turn(1) + _turn(2) + _turn(3)
    turns.begin("s-1")
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    turns.begin("s-1")  # turn 4, on the record: its end is not seen
    turns.begin("s-1")  # turn 5, on the record too
    said = said + _turn(4) + _turn(5)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3), _turn(3) + _turn(4) + _turn(5)]


def test_each_turn_begun_off_the_record_is_counted_until_its_own_end(tmp_path):
    """Two turns begin off the record, and both uploads run late, after `off --forget`.
    The first does not take the count of both."""
    turns, wire = _back(tmp_path)
    marker = _mark(tmp_path, "s-1")
    turns.begin("s-1")  # turn 2
    turns.begin("s-1")  # turn 3: turn 2's upload has not run yet
    marker.unlink()
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False
    assert turns.upload(said + _turn(3), "s-1", timeout=5, finished=True) is False
    assert wire.tries == 0


def test_turns_begun_off_the_record_under_two_ids_left_are_all_counted(tmp_path):
    """Each of two ids had a turn begun off the record whose end was not seen. Both
    are counted for the uploader: the uploads under a third id take one each."""
    turns, wire = _turns(tmp_path, begins=True, lookback=10)
    for left in ("s-a", "s-b"):
        turns.start(left, 0)
        marker = _mark(tmp_path, left)
        turns.begin(left)
        marker.unlink()
    turns.start("s-1", 0, continues=["s-a", "s-b"])
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False
    assert turns.upload(said + _turn(3), "s-1", timeout=5, finished=True) is False
    assert wire.tries == 0


def test_a_session_started_again_counts_the_sessions_it_goes_on_from_once(tmp_path):
    """A harness may start an id twice before its first upload, naming the id left both
    times (Hermes: at the switch, and again when the session's row is read). The turn
    begun off the record under the id left is one turn: the upload that ends it takes
    the count, and the turn after it goes up."""
    turns, wire = _turns(tmp_path, begins=True, lookback=10)
    turns.start("s-0", 0)
    marker = _mark(tmp_path, "s-0")
    turns.begin("s-0")  # turn 2, off the record
    turns.start("s-1", 0, continues=["s-0"])
    turns.start("s-1", 0, continues=["s-0"])
    marker.unlink()
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False
    turns.begin("s-1")
    assert turns.upload(said + _turn(3), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3)]


def test_a_session_started_again_counts_one_it_did_not_go_on_from_before(tmp_path):
    """Two ids each had a turn begun off the record. Whatever the starts of a third id
    name (one of the two, both, nothing), both turns stay counted, once each: a start
    changes no count, and two uploads that end a turn go by before one is sent."""
    turns, wire = _turns(tmp_path, begins=True, lookback=10)
    for left in ("s-a", "s-b"):
        turns.start(left, 0)
        marker = _mark(tmp_path, left)
        turns.begin(left)
        marker.unlink()
    turns.start("s-1", 0, continues=["s-a"])
    turns.start("s-1", 0, continues=["s-a", "s-b"])
    turns.start("s-1", 0)
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False
    turns.begin("s-1")
    assert turns.upload(said + _turn(3), "s-1", timeout=5, finished=True) is False
    turns.begin("s-1")
    assert turns.upload(said + _turn(3) + _turn(4), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(4)]


def test_a_third_start_does_not_count_again_what_the_first_took(tmp_path):
    """Nor is the turn begun under the id left counted twice when that id is named
    again at a third start, after one that named nothing."""
    turns, wire = _turns(tmp_path, begins=True, lookback=10)
    turns.start("s-0", 0)
    marker = _mark(tmp_path, "s-0")
    turns.begin("s-0")
    marker.unlink()
    turns.start("s-1", 0, continues=["s-0"])
    turns.start("s-1", 0)
    turns.start("s-1", 0, continues=["s-0"])
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False
    turns.begin("s-1")
    assert turns.upload(said + _turn(3), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3)]


@pytest.mark.parametrize("switch", [True, False], ids=["a switch first", "its row alone"])
@pytest.mark.parametrize("ends_under", ["s-1", "s-2"])
def test_a_turn_begun_off_the_record_is_counted_once_across_two_moves(tmp_path, ends_under, switch):
    """A session moves twice. The turn begun off the record under the first id ends
    under the second, or is moved again and ends under the third. The third id is
    started with the whole chain (Hermes: its row names every id before it), after a
    start that named the second alone (the switch) or with none. It is one turn,
    counted once whichever ids are started: its end takes the count, under the second
    id or the third, and the turn after it goes up."""
    turns, wire = _turns(tmp_path, begins=True, lookback=10)
    turns.start("s-0", 0)
    marker = _mark(tmp_path, "s-0")
    turns.begin("s-0")  # turn 2, off the record
    marker.unlink()
    said = _turn(1) + _turn(2)
    turns.start("s-1", 0, continues=["s-0"])
    if ends_under == "s-1":
        assert turns.upload(said, "s-1", timeout=5, finished=True) is False
    if switch:
        turns.start("s-2", 0, continues=["s-1"])
    turns.start("s-2", 0, continues=["s-1", "s-0"])
    if ends_under == "s-2":
        assert turns.upload(said, "s-2", timeout=5, finished=True) is False
    turns.begin("s-2")
    assert turns.upload(said + _turn(3), "s-2", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3)]


@pytest.mark.parametrize("named", [["s-0", "s-1"], ["s-1", "s-0"]], ids=["back first", "on first"])
def test_two_sessions_that_went_on_from_each_other_keep_the_turn_begun_off_the_record(
    tmp_path, named
):
    """The id left is gone back to (Hermes: an undo that names it, after a compression
    whose turn raised). A turn begins off the record under it, and a third id goes on
    from both. Each of the two went on from the other, and the turn is still counted,
    once: its upload runs late, after the next turn has begun, and is not taken for
    the end of that one."""
    turns, wire = _turns(tmp_path, begins=True, lookback=10)
    turns.start("s-0", 0)
    turns.start("s-1", 0, continues=["s-0"])
    turns.start("s-0", 0, continues=["s-1"])
    marker = _mark(tmp_path, "s-0")
    turns.begin("s-0")  # turn 2, off the record
    marker.unlink()
    turns.start("s-2", 0, continues=named)
    turns.begin("s-2")  # turn 3, on the record: turn 2's upload has not run yet
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-2", timeout=5, finished=True) is False
    assert turns.upload(said + _turn(3), "s-2", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3)]


@pytest.mark.parametrize("ended", [False, True], ids=["both to end", "the first has ended"])
def test_a_turn_begun_under_an_id_after_it_was_gone_on_from_is_counted_too(tmp_path, ended):
    """A turn start goes under the id the harness named last, and that may be one
    another id has gone on from already. The turn begun there off the record is
    counted like any other: beside the one begun before it, or alone when that one's
    end was seen under the second id. A third id that goes on from both waits for as
    many uploads, and the turn begun in it goes up."""
    turns, wire = _turns(tmp_path, begins=True, lookback=10)
    turns.start("s-0", 0)
    marker = _mark(tmp_path, "s-0")
    turns.begin("s-0")  # turn 2, off the record
    turns.start("s-1", 0, continues=["s-0"])
    said = _turn(1) + _turn(2)
    if ended:
        assert turns.upload(said, "s-1", timeout=5, finished=True) is False
    turns.begin("s-0")  # turn 3, off the record too, under the id left
    marker.unlink()
    turns.start("s-2", 0, continues=["s-1", "s-0"])
    if not ended:
        assert turns.upload(said, "s-2", timeout=5, finished=True) is False
    said = said + _turn(3)
    assert turns.upload(said, "s-2", timeout=5, finished=True) is False
    turns.begin("s-2")
    assert turns.upload(said + _turn(4), "s-2", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(4)]


@pytest.mark.parametrize("again", [False, True], ids=["as it was", "started again since"])
def test_a_turn_whose_end_was_seen_is_not_waited_for_under_an_id_that_goes_on_later(
    tmp_path, again
):
    """The end of the turn begun off the record was seen under the first id, after a
    second id had gone on from it. The end counts whichever id saw it: a third that
    goes on from both waits for no upload, and the turn begun in it goes up. Also when
    the first id was started again since it saw that end."""
    turns, wire = _turns(tmp_path, begins=True, lookback=10)
    turns.start("s-0", 0)
    marker = _mark(tmp_path, "s-0")
    turns.begin("s-0")  # turn 2, off the record
    marker.unlink()
    turns.start("s-1", 0, continues=["s-0"])
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-0", timeout=5, finished=True) is False
    if again:
        turns.start("s-0", 0)
    turns.start("s-2", 0, continues=["s-1", "s-0"])
    turns.begin("s-2")
    assert turns.upload(said + _turn(3), "s-2", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3)]


def test_a_flush_sends_no_new_turn_while_a_turn_begun_off_the_record_is_to_end(tmp_path):
    """Turn 4 began off the record and its rows are complete: its reply was cut short
    while it came in (Hermes keeps the part that came and sends no upload), or its
    upload runs late. Back on the record, a flush is not the upload that ends it and
    cannot tell it from a turn spoken there: it sends nothing new. The upload that ends
    a turn notes it, and it stays out."""
    turns, wire = _back(tmp_path)
    said = _turn(1) + _turn(2) + _turn(3)
    turns.begin("s-1")
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    marker = _mark(tmp_path, "s-1")
    turns.begin("s-1")  # turn 4, off the record
    marker.unlink()
    said = said + _turn(4)
    assert turns.upload(said, "s-1", timeout=5) is False
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False
    assert turns.upload(said, "s-1", timeout=5) is False
    turns.begin("s-1")
    assert turns.upload(said + _turn(5), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3), _turn(5)]


def test_a_flush_is_not_the_end_of_a_turn_begun_off_the_record(tmp_path):
    """Turn 4 ended on the record and its upload waits. A flush while turn 5, begun off
    the record, runs takes no count: the two uploads still to come are both taken for
    turns begun off the record, as with no flush, and neither turn goes up. That costs
    turn 4, which the flush does not send either."""
    turns, wire = _back(tmp_path)
    said = _turn(1) + _turn(2) + _turn(3)
    turns.begin("s-1")
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    turns.begin("s-1")  # turn 4, on the record
    marker = _mark(tmp_path, "s-1")
    turns.begin("s-1")  # turn 5, off it
    marker.unlink()
    said = said + _turn(4)
    assert turns.upload(said + _turn(5, done=False), "s-1", timeout=5) is False
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False
    assert turns.upload(said + _turn(5), "s-1", timeout=5, finished=True) is False
    assert _records(wire) == [_turn(3)]


def test_a_flush_sends_a_turn_begun_on_the_record_with_none_begun_off_it_to_end(tmp_path):
    """The rule holds only while a turn begun off the record is still to end. Turn 4
    began on the record in a session back on it, and the upload that ends it has not
    run when the harness flushes (it exits): the flush sends it."""
    turns, wire = _back(tmp_path)
    said = _turn(1) + _turn(2) + _turn(3)
    turns.begin("s-1")
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    turns.begin("s-1")  # turn 4, on the record
    assert turns.upload(said + _turn(4), "s-1", timeout=5) is True
    assert _records(wire) == [_turn(3), _turn(3) + _turn(4)]


def test_a_flush_of_a_session_never_off_the_record_sends_what_it_holds(tmp_path):
    """The rule above is for a session that held turns from off the record. One marked
    private by hand and put back holds no such turn to keep out: its flush sends the
    turns not sent yet, the one begun while it was private too."""
    turns, wire = _turns(tmp_path, begins=True, lookback=10)
    turns.begin("s-1")
    assert turns.upload(_turn(1), "s-1", timeout=5, finished=True) is True
    marker = _mark(tmp_path, "s-1")
    turns.begin("s-1")  # turn 2, while private by hand
    marker.unlink()
    assert turns.upload(_turn(1) + _turn(2), "s-1", timeout=5) is True
    assert _records(wire) == [_turn(1), _turn(1) + _turn(2)]


def test_turns_kept_for_the_servers_answer_go_up_at_a_flush_all_the_same(tmp_path):
    """Turn 4 was spoken on the record and kept while the server had not answered. The
    flush that sends no new turn still hands over what was kept: it was owed."""
    turns, wire = _back(tmp_path)
    said = _turn(1) + _turn(2) + _turn(3)
    turns.begin("s-1")
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    wire.fail_private = _DOWN
    turns.uploader.carry_private(["s-0"], "s-1", timeout=None)
    turns.begin("s-1")  # turn 4: the server has not said, so it counts as on the record
    said = said + _turn(4)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False  # kept
    wire.fail_private = None
    assert turns.uploader.settle("s-1") == upload.OPEN
    marker = _mark(tmp_path, "s-1")
    turns.begin("s-1")  # turn 5, off the record
    marker.unlink()
    assert turns.upload(said + _turn(5), "s-1", timeout=5) is True
    assert _records(wire) == [_turn(3), _turn(3) + _turn(4)]


# ---------------------------------------------------------------------------
# Turn starts are counted for the uploader, whichever ids a start and its end name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("first_off", [False, True], ids=["begun on the record", "begun off it"])
def test_an_upload_one_turn_late_twice_running_is_not_the_end_of_the_turn_begun_since(
    tmp_path, first_off
):
    """Every upload runs one turn late (the user types ahead). Turn 2's runs after turn
    3 began on the record, and turn 3's after turn 4 began off it. Turn 3 was still
    counted when turn 4 began, whatever the upload in between ended: its own late
    upload is not taken for the end of turn 4, and turn 4's does not send that turn."""
    turns, wire = _back(tmp_path)
    marker = _mark(tmp_path, "s-1") if first_off else None
    turns.begin("s-1", earlier=True)  # turn 2
    if marker:
        marker.unlink()  # `off --forget`
    turns.begin("s-1", earlier=True)  # turn 3, on the record: turn 2's upload waits
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is not first_off
    marker = _mark(tmp_path, "s-1")
    turns.begin("s-1", earlier=True)  # turn 4, off the record: turn 3's upload waits
    marker.unlink()
    said = said + _turn(3)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False  # turn 3's
    said = said + _turn(4)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False  # turn 4's own
    turns.begin("s-1", earlier=True)
    assert turns.upload(said + _turn(5), "s-1", timeout=5, finished=True) is True
    assert _sent(wire) == ([] if first_off else [("s-1", ["u2"])]) + [("s-1", ["u5"])]


def test_a_turn_begun_off_the_record_is_counted_with_one_turn_before_it_not_two(tmp_path):
    """Two turns begun on the record are still counted when one begins off it: the
    first was stopped, the second's upload runs late. An upload runs one turn late at
    most, so one turn before is counted, and the turn after `off --forget` goes up."""
    turns, wire = _back(tmp_path)
    turns.begin("s-1")  # turn 2, stopped: no upload ends it
    turns.begin("s-1")  # turn 3
    marker = _mark(tmp_path, "s-1")
    turns.begin("s-1")  # turn 4, off the record: turn 3's upload has not run yet
    marker.unlink()
    said = _turn(1) + _turn(2, done=False) + _turn(3)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False  # turn 3's
    said = said + _turn(4)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False  # turn 4's own
    turns.begin("s-1")
    assert turns.upload(said + _turn(5), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(5)]


def test_the_turn_before_counted_with_one_begun_off_the_record_is_counted_there_alone(tmp_path):
    """Turn 2's upload waits when turn 3 begins off the record: the two are counted
    together, and the two uploads take both. Nothing of turn 2 stays counted on the
    record besides: after the next stretch off it, the first turn back goes up."""
    turns, wire = _back(tmp_path)
    turns.begin("s-1")  # turn 2: its upload waits
    marker = _mark(tmp_path, "s-1")
    turns.begin("s-1")  # turn 3, off the record
    marker.unlink()
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False  # turn 2's, late
    said = said + _turn(3)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False  # turn 3's own
    turns.begin("s-1")
    said = said + _turn(4)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    marker = _mark(tmp_path, "s-1")
    turns.begin("s-1")  # turn 5, off the record
    said = said + _turn(5)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False
    marker.unlink()
    turns.begin("s-1")
    assert turns.upload(said + _turn(6), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(4), _turn(6)]


def test_a_turn_begun_before_the_servers_private_answer_is_counted_there_alone(tmp_path):
    """The same for a turn begun while the server had not said: its answer "private"
    counts the turn as begun off the record, and no longer as one begun on it. After
    the next stretch off the record, the first turn back goes up."""
    turns, wire = _turns(tmp_path)
    wire.rows.add("s-old")
    assert turns.uploader.carry_private(["s-old"], "s-new") is False
    turns.begin("s-new")  # turn 1
    said = _turn(1)
    assert turns.upload(said, "s-new", timeout=5, finished=True) is False  # "private"
    _forget(wire, tmp_path, "s-new")
    turns.begin("s-new")
    said = said + _turn(2)
    assert turns.upload(said, "s-new", timeout=5, finished=True) is True
    marker = _mark(tmp_path, "s-new")
    turns.begin("s-new")  # turn 3, off the record
    said = said + _turn(3)
    assert turns.upload(said, "s-new", timeout=5, finished=True) is False
    marker.unlink()
    turns.begin("s-new")
    assert turns.upload(said + _turn(4), "s-new", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(2), _turn(4)]


@pytest.mark.parametrize("answered", ["at a flush", "at the first upload"])
def test_the_servers_private_answer_counts_both_turns_begun_before_it(tmp_path, answered):
    """Two turns begin while the server has not said, the second before the first's
    upload has run. The answer is "private", at a flush or at that first upload: both
    began off the record, as it turns out. After `off --forget` neither upload sends
    its turn, also when the second runs late, after the next turn has begun: it is
    not taken for the end of that one."""
    turns, wire = _turns(tmp_path)
    wire.rows.add("s-old")
    assert turns.uploader.carry_private(["s-old"], "s-new") is False
    turns.begin("s-new")  # turn 1
    turns.begin("s-new")  # turn 2: turn 1's upload has not run yet
    said = _turn(1)
    if answered == "at a flush":
        assert turns.upload([], "s-new", timeout=5) is False  # "private"
        assert (tmp_path / "private" / "s-new").is_file()
        _forget(wire, tmp_path, "s-new")
    assert turns.upload(said, "s-new", timeout=5, finished=True) is False
    if answered == "at the first upload":
        assert (tmp_path / "private" / "s-new").is_file()
        _forget(wire, tmp_path, "s-new")
    turns.begin("s-new")  # turn 3, on the record: turn 2's upload has not run yet
    said = said + _turn(2)
    assert turns.upload(said, "s-new", timeout=5, finished=True) is False
    assert wire.tries == 0
    assert turns.upload(said + _turn(3), "s-new", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3)]


def test_two_turn_starts_are_kept_on_the_record_at_most(tmp_path):
    """Three turns begin while the server has not said, and the first two were stopped.
    An upload runs one turn late at most, so the start of this turn and of the one
    before are kept. The answer "private" counts those two as begun off the record: the
    third turn's upload takes one, the next turn's the other, and the one after that
    goes up."""
    turns, wire = _turns(tmp_path)
    wire.rows.add("s-old")
    assert turns.uploader.carry_private(["s-old"], "s-new") is False
    for _ in range(3):
        turns.begin("s-new")
    assert turns.upload([], "s-new", timeout=5) is False  # a flush: "private"
    _forget(wire, tmp_path, "s-new")
    said = _turn(1)
    assert turns.upload(said, "s-new", timeout=5, finished=True) is False
    for n in (2, 3):
        turns.begin("s-new")
        said = said + _turn(n)
        assert turns.upload(said, "s-new", timeout=5, finished=True) is (n == 3)
    assert _records(wire) == [_turn(3)]


def test_turns_whose_ends_were_seen_under_two_ids_are_not_waited_for_under_a_third(tmp_path):
    """Turn 2 begins off the record under s-0, and its upload runs late, under that id,
    after the move to s-1. Turn 3 begins and ends off the record under s-1. Both ends
    were seen, each under another id: s-2, which goes on from both, waits for neither,
    and the first turn back on the record goes up."""
    turns, wire = _turns(tmp_path, begins=True, lookback=10)
    turns.start("s-0", 0)
    _mark(tmp_path, "s-0")
    turns.begin("s-0")  # turn 2, off the record
    turns.start("s-1", 0, continues=["s-0"])
    marker = _mark(tmp_path, "s-1")
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-0", timeout=5, finished=True) is False  # late, the id left
    turns.begin("s-1")  # turn 3, off the record
    said = said + _turn(3)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False
    marker.unlink()  # `off --forget`
    turns.start("s-2", 0, continues=["s-1", "s-0"])
    turns.begin("s-2")  # turn 4, on the record
    assert turns.upload(said + _turn(4), "s-2", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(4)]


def test_a_turn_begun_off_the_record_under_an_id_already_gone_on_from_is_counted(tmp_path):
    """A turn start goes under the id the harness named last, and that may be one the
    id of its upload went on from before the turn began (Hermes: a rewind that names
    the id left). No start of the newer id comes after it to take anything over: the
    turn is counted all the same, for the uploader, and its late upload under the
    newer id is not taken for the end of the turn begun there since."""
    turns, wire = _turns(tmp_path, begins=True, lookback=10)
    turns.start("s-0", 0)
    turns.start("s-1", 0, continues=["s-0"])
    marker = _mark(tmp_path, "s-0")
    turns.begin("s-0")  # turn 2, off the record, under the id left
    marker.unlink()
    turns.begin("s-1")  # turn 3, on the record: turn 2's upload has not run yet
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False and wire.tries == 0
    assert turns.upload(said + _turn(3), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3)]


@pytest.mark.parametrize("flush", [True, False], ids=["a flush first", "no flush"])
def test_a_turn_begun_off_the_record_whose_end_another_id_saw_stays_out_under_this_one(
    tmp_path, flush
):
    """s-1 is back on the record and uploading. The conversation moves on to s-2, where
    a turn begins off the record and ends: s-2 notes it. The harness then names s-1
    again (Hermes: a rewind that names the id left, a flush for it), and its records
    hold that turn too. s-1 has not noted it: a flush sends no new turn for it, and
    the upload that ends its next turn notes what the session holds before it sends."""
    turns, wire = _back(tmp_path)
    turns.begin("s-1")
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    turns.start("s-2", 0, continues=["s-1"])
    marker = _mark(tmp_path, "s-2")
    turns.begin("s-2")  # turn 3, off the record
    marker.unlink()  # `off --forget`
    said = said + _turn(3)
    assert turns.upload(said, "s-2", timeout=5, finished=True) is False
    if flush:
        assert turns.upload(said, "s-1", timeout=5) is False
    turns.begin("s-1")  # turn 4, on the record
    assert turns.upload(said + _turn(4), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(2), _turn(4)]


def test_a_flush_that_shows_part_of_the_conversation_does_not_note_for_all_of_it(tmp_path):
    """As above, with a harness whose flush uses up the look-back. The flush for s-1
    shows the first turn only: what s-1 holds of the turn begun off the record is noted
    by the upload that ends its next turn, which shows all of it."""
    turns, wire = _turns(tmp_path)
    turns.start("s-1", 0)
    turns.start("s-2", 0, continues=["s-1"])
    marker = _mark(tmp_path, "s-2")
    turns.begin("s-2")  # turn 2, off the record
    marker.unlink()
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-2", timeout=5, finished=True) is False
    assert turns.upload(_turn(1), "s-1", timeout=5) is False
    turns.begin("s-1")  # turn 3, on the record
    assert turns.upload(said + _turn(3), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3)]


def test_a_flush_refused_as_private_that_shows_part_does_not_note_for_all_of_it(tmp_path):
    """The same when that flush is refused as private: s-1 is marked again as it
    runs. It notes the turn it is shown, and not that the session has noted what it
    holds: after `off --forget`, the upload that ends s-1's next turn still notes the
    turn begun off the record, which the flush was not shown."""
    turns, wire = _back(tmp_path)
    turns.begin("s-1")
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    turns.start("s-2", 0, continues=["s-1"])
    marker = _mark(tmp_path, "s-2")
    turns.begin("s-2")  # turn 3, off the record
    marker.unlink()
    said = said + _turn(3)
    assert turns.upload(said, "s-2", timeout=5, finished=True) is False
    marker = _mark(tmp_path, "s-1")
    assert turns.upload(_turn(1), "s-1", timeout=5) is False  # refused: the head alone
    marker.unlink()
    turns.begin("s-1")  # turn 4, on the record
    assert turns.upload(said + _turn(4), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(2), _turn(4)]


def test_a_flush_notes_nothing_for_a_session_that_has_still_to_note(tmp_path):
    """s-1 has still to note the turn begun off the record whose end s-2 saw. Turn 4
    ends on the record under s-1 with its upload waiting, turn 5 begins, and a flush
    shows both. The flush sends nothing and notes nothing: turn 4's own upload, which
    shows the conversation as far as turn 4, notes what the session holds and sends
    turn 4, seen to begin on the record."""
    turns, wire = _back(tmp_path)
    turns.begin("s-1")
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    turns.start("s-2", 0, continues=["s-1"])
    marker = _mark(tmp_path, "s-2")
    turns.begin("s-2")  # turn 3, off the record
    marker.unlink()
    said = said + _turn(3)
    assert turns.upload(said, "s-2", timeout=5, finished=True) is False
    turns.begin("s-1")  # turn 4: its upload waits
    turns.begin("s-1")  # turn 5
    said = said + _turn(4)
    assert turns.upload(said + _turn(5, done=False), "s-1", timeout=5) is False  # a flush
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True  # turn 4's own
    assert _records(wire) == [_turn(2), _turn(4)]


@pytest.mark.parametrize("refused", [False, True], ids=["put back first", "refused as private"])
def test_a_flush_sends_a_turn_begun_on_the_record_once_the_one_begun_off_it_was_noted(
    tmp_path, refused
):
    """A flush is quiet until the upload that ends the turn begun off the record has
    noted it, and no longer: also when that upload was refused as private. Turn 5 began
    on the record, and the flush as the harness exits sends it."""
    turns, wire = _back(tmp_path)
    said = _turn(1) + _turn(2) + _turn(3)
    turns.begin("s-1")
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    marker = _mark(tmp_path, "s-1")
    turns.begin("s-1")  # turn 4, off the record
    said = said + _turn(4)
    if not refused:
        marker.unlink()
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False
    marker.unlink(missing_ok=True)
    turns.begin("s-1")  # turn 5, on the record
    assert turns.upload(said + _turn(5), "s-1", timeout=5) is True
    assert _records(wire) == [_turn(3), _turn(5)]


# ---------------------------------------------------------------------------
# A session said to be new
# ---------------------------------------------------------------------------


def test_a_session_said_to_be_new_uploads_nothing_before_its_first_turn_starts(tmp_path):
    """With ``begins`` the harness says when turns start. A flush finds turns in a
    session it said was new before any has started: copies, written after it said so.
    The first turn start says there were turns before it, and only the turn that
    starts there goes up."""
    turns, wire = _turns(tmp_path, begins=True)
    copies = _turn(1) + _turn(2)
    assert turns.upload(copies, "s-1", timeout=5) is False and wire.tries == 0
    turns.begin("s-1", earlier=True)
    running = copies + _turn(3, done=False)
    assert turns.upload(running, "s-1", timeout=5) is False and wire.tries == 0
    assert turns.upload(copies + _turn(3), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3)]
    turns.begin("s-1", earlier=True)
    assert turns.upload(copies + _turn(3) + _turn(4), "s-1", timeout=5, finished=True) is True
    assert _records(wire)[1:] == [_turn(3) + _turn(4)]


@pytest.mark.parametrize("finished", [True, False], ids=["a turn's end", "a flush"])
def test_more_turns_than_were_begun_are_turns_from_before(tmp_path, finished):
    """The harness could not say whether a turn came before the first one (Hermes gives
    the last user message with text, and there may be none). The first upload counts:
    a turn's end shows no more turns than were begun, a flush at most one more (it may
    run before the harness says the turn starts)."""
    turns, wire = _turns(tmp_path, begins=True)
    turns.begin("s-1")
    said = _turn(1) + _turn(2) + _turn(3, done=finished)
    if not finished:
        assert turns.upload(said, "s-1", timeout=5) is False and wire.tries == 0
        said = _turn(1) + _turn(2) + _turn(3)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3)]


def test_a_new_session_that_holds_the_turns_begun_in_it_uploads_them_all(tmp_path):
    turns, wire = _turns(tmp_path, begins=True)
    turns.begin("s-1")  # turn 1, cut short: no upload came for it
    turns.begin("s-1")
    # A flush before the harness says turn 3 starts.
    said = _turn(1) + _turn(2) + _turn(3, done=False)
    assert turns.upload(said, "s-1", timeout=5) is True
    turns.begin("s-1")
    assert turns.upload(_turn(1) + _turn(2) + _turn(3), "s-1", timeout=5, finished=True) is True
    assert _sent(wire) == [("s-1", ["u1", "u2"]), ("s-1", ["u2", "u3"])]


def test_a_turn_that_ends_with_no_start_seen_closes_a_new_session(tmp_path):
    """A turn ends in a session said to be new, and the harness announced no start:
    more turns than were begun, so nothing says the ones it holds are its own. They
    stay out; a turn that comes after is new whoever began it."""
    turns, wire = _turns(tmp_path, begins=True)
    assert turns.upload(_turn(1), "s-1", timeout=5, finished=True) is False
    assert wire.tries == 0
    assert turns.upload(_turn(1) + _turn(2), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(2)]


def test_a_new_session_is_checked_at_its_first_upload_only(tmp_path):
    """Its first upload showed no more turns than were begun: the doubt is settled. A
    turn that appears later is new by where it stands, also one whose start was not
    seen."""
    turns, wire = _turns(tmp_path, begins=True)
    turns.begin("s-1")
    assert turns.upload(_turn(1), "s-1", timeout=5, finished=True) is True
    said = _turn(1) + _turn(2) + _turn(3)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    assert _sent(wire) == [("s-1", ["u1"]), ("s-1", ["u1", "u2", "u3"])]


def test_the_doubt_goes_on_to_the_id_a_new_session_is_given(tmp_path):
    """A compression at the start of the first turn gives the conversation a new id
    before any turn start was seen under the old one."""
    turns, wire = _turns(tmp_path, begins=True)
    turns.start("s-1")
    turns.start("s-2", 10, continues=["s-1"])
    copies = _turn(1) + _turn(2)
    assert turns.upload(copies, "s-2", timeout=5) is False and wire.tries == 0
    turns.begin("s-2", earlier=True)
    assert turns.upload(copies + _turn(3), "s-2", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3)]


def test_the_turns_begun_under_the_id_left_count_for_the_new_one(tmp_path):
    turns, wire = _turns(tmp_path, begins=True)
    turns.start("s-1")
    turns.begin("s-1")  # turn 1, cut short
    turns.start("s-2", 10, continues=["s-1"])
    turns.begin("s-2", earlier=True)  # turn 2: the one before it was seen to begin
    said = _turn(1, done=False) + _turn(2)
    assert turns.upload(said, "s-2", timeout=5, finished=True) is True
    turns.start("s-3", 10, continues=["s-2"])  # the doubt is settled: s-3 looks back
    assert turns.upload(said, "s-3", timeout=5) is True
    assert _sent(wire) == [("s-2", ["u2"]), ("s-3", ["u2"])]


def test_a_session_found_to_hold_copies_at_a_turn_start_is_named_to_the_harness(tmp_path):
    """What the uploader finds is in memory only. ``copied`` hands the id to the
    harness, once, at the turn start that showed it and before any upload."""
    found: list[str] = []
    turns, wire = _turns(tmp_path, begins=True, copied=found.append)
    turns.start("s-1")
    turns.begin("s-1", earlier=True)
    assert found == ["s-1"] and wire.tries == 0
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    turns.begin("s-1", earlier=True)
    assert turns.upload(said + _turn(3), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(2), _turn(2) + _turn(3)]
    assert found == ["s-1"]


@pytest.mark.parametrize("finished", [True, False], ids=["a turn's end", "a flush"])
def test_a_session_found_to_hold_copies_at_its_first_upload_is_named_too(tmp_path, finished):
    found: list[str] = []
    turns, _ = _turns(tmp_path, begins=True, copied=found.append)
    turns.begin("s-1")
    said = _turn(1) + _turn(2) + _turn(3, done=finished)
    turns.upload(said, "s-1", timeout=5, finished=finished)
    assert found == ["s-1"]


def test_a_session_that_holds_its_own_turns_is_not_named(tmp_path):
    """New and found to be so, resumed with a look-back, or one that names the session
    it goes on from: none of them was found to hold copies."""
    found: list[str] = []
    turns, _ = _turns(tmp_path, begins=True, lookback=10, copied=found.append)
    turns.start("s-new")
    turns.begin("s-new")
    assert turns.upload(_turn(1), "s-new", timeout=5, finished=True) is True
    turns.begin("s-resumed", earlier=True)
    assert turns.upload(_turn(1) + _turn(2), "s-resumed", timeout=5, finished=True) is True
    turns.start("s-branch", continues=["s-0"])
    turns.begin("s-branch", earlier=True)
    assert turns.upload(_turn(1) + _turn(2), "s-branch", timeout=5, finished=True) is True
    assert found == []


def test_the_id_left_is_named_with_the_one_the_doubt_went_on_to(tmp_path):
    """The doubt was settled under the new id. The id left holds the same copies and
    is still said to be new here: it is named as well, and closed, so an upload under
    it looks back at nothing either, once the turn that began has ended."""
    found: list[str] = []
    turns, wire = _turns(tmp_path, begins=True, copied=found.append)
    turns.start("s-1")
    turns.start("s-2", 10, continues=["s-1"])
    turns.begin("s-2", earlier=True)
    assert found == ["s-2", "s-1"]
    copies = _turn(1) + _turn(2)
    assert turns.upload(copies + _turn(3), "s-2", timeout=5, finished=True) is True
    assert turns.upload(copies, "s-1", timeout=5, finished=True) is False
    assert _sent(wire) == [("s-2", ["u3"])]


def test_every_id_the_doubt_came_through_is_named(tmp_path):
    """Two moves before the first turn start, each new id started with the one left
    (no order of Hermes's calls is known that does it; nothing here counts on that).
    The copies are found under the third id. The first holds them too and is still
    said to be new here: it is named with the second, and closed."""
    found: list[str] = []
    turns, wire = _turns(tmp_path, begins=True, copied=found.append)
    turns.start("s-1")
    turns.start("s-2", 10, continues=["s-1"])
    turns.start("s-3", 10, continues=["s-2"])
    turns.begin("s-3", earlier=True)
    assert found == ["s-3", "s-1", "s-2"]
    copies = _turn(1) + _turn(2)
    assert turns.upload(copies + _turn(3), "s-3", timeout=5, finished=True) is True
    assert turns.upload(copies, "s-1", timeout=5, finished=True) is False
    assert _sent(wire) == [("s-3", ["u3"])]


def test_the_turns_begun_under_the_id_left_are_counted_once_too(tmp_path):
    """The harness starts the new id at the switch and again when it reads the id's
    row, before the first upload. One turn was begun and the upload shows two: the
    other is a copy, whichever start is asked."""
    found: list[str] = []
    turns, wire = _turns(tmp_path, begins=True, copied=found.append)
    turns.start("s-1")
    turns.begin("s-1")  # nothing said a turn came before this one
    turns.start("s-2", 10, continues=["s-1"])
    turns.start("s-2", 10, continues=["s-1"])
    assert turns.upload(_turn(1) + _turn(2), "s-2", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(2)] and found == ["s-2", "s-1"]


@pytest.mark.parametrize("under", ["s-1", "s-2"], ids=["the id gone back to", "the id left again"])
@pytest.mark.parametrize("copies", [0, 1], ids=["its own turns", "a copy among them"])
def test_turns_begun_are_counted_once_each_whichever_session_went_on_from_which(
    tmp_path, under, copies
):
    """A session said to be new moves to a second id and back, and a third id goes on
    from both. Two turns were begun, the first before the moves and the second under
    either id: each is counted once and neither is left out. An upload that shows two
    turns finds nothing, and they go up. One that shows three finds a copy."""
    found: list[str] = []
    turns, wire = _turns(tmp_path, begins=True, copied=found.append)
    turns.start("s-1")
    turns.begin("s-1")  # nothing said a turn came before this one
    turns.start("s-2", 10, continues=["s-1"])
    turns.start("s-1", 10, continues=["s-2"])
    turns.begin(under)
    turns.start("s-3", 10, continues=["s-1", "s-2"])
    said = [row for n in range(1, 3 + copies) for row in _turn(n)]
    assert turns.upload(said, "s-3", timeout=5, finished=True) is True
    assert _records(wire) == ([_turn(3)] if copies else [said])
    assert found == (["s-3", "s-1", "s-2"] if copies else [])


def test_a_session_named_before_it_was_met_is_not_one_the_doubt_came_through(tmp_path):
    """A start may name a session nothing is remembered of (the harness read it off
    the new id's row). The doubt did not come through it: when it is met later and
    said to be new, the copies found under the id that named it say nothing of it."""
    found: list[str] = []
    turns, _ = _turns(tmp_path, begins=True, copied=found.append)
    turns.start("s-1")
    turns.start("s-2", 10, continues=["s-1", "s-0"])
    turns.start("s-0")
    turns.begin("s-2", earlier=True)
    assert found == ["s-2", "s-1"]


def test_the_turns_begun_in_a_session_not_in_doubt_do_not_count_for_one_that_is(tmp_path):
    """The new id goes on from a session said to be new and from one that was never in
    doubt, in which a turn was begun. That turn start says nothing about the copies:
    the first turn start under the new id still shows them."""
    found: list[str] = []
    turns, _ = _turns(tmp_path, begins=True, copied=found.append)
    turns.start("s-0", 10)
    turns.begin("s-0")
    turns.start("s-1")
    turns.start("s-2", 10, continues=["s-1", "s-0"])
    turns.begin("s-2", earlier=True)
    assert found == ["s-2", "s-1"]


def test_an_id_left_that_was_not_in_doubt_is_not_named(tmp_path):
    """The id left had a look-back: it was never said to be new, and the copies found
    under the id the conversation went on to say nothing about it."""
    found: list[str] = []
    turns, _ = _turns(tmp_path, begins=True, copied=found.append)
    turns.start("s-0", 10)
    turns.start("s-1")
    turns.start("s-2", 10, continues=["s-1", "s-0"])
    turns.begin("s-2", earlier=True)
    assert found == ["s-2", "s-1"]


def test_a_session_found_to_hold_copies_is_not_looked_back_at_under_a_later_id(tmp_path):
    """What was found stays with the conversation in this process, whatever the harness
    kept of it: the id it goes on under asks for a look-back in vain, as after time off
    the record."""
    turns, wire = _turns(tmp_path, begins=True)
    turns.start("s-1")
    copies = _turn(1) + _turn(2)
    turns.begin("s-1", earlier=True)
    assert turns.upload(copies + _turn(3), "s-1", timeout=5, finished=True) is True
    turns.start("s-2", 10, continues=["s-1"])
    turns.begin("s-2", earlier=True)
    said = copies + _turn(3) + _turn(4)
    assert turns.upload(said, "s-2", timeout=5, finished=True) is True
    assert _sent(wire) == [("s-1", ["u3"]), ("s-2", ["u4"])]


def test_a_harness_whose_note_fails_stops_no_upload(tmp_path):
    def fails(session_id: str) -> None:
        raise OSError("read-only")

    turns, wire = _turns(tmp_path, begins=True, copied=fails)
    turns.begin("s-1", earlier=True)  # found at the turn start
    assert turns.upload(_turn(1) + _turn(2), "s-1", timeout=5, finished=True) is True
    turns.begin("s-2")  # found at the upload
    assert turns.upload(_turn(1) + _turn(2), "s-2", timeout=5, finished=True) is True
    assert _sent(wire) == [("s-1", ["u2"]), ("s-2", ["u2"])]


def test_a_session_the_harness_knows_to_hold_copies_is_not_looked_back_at(tmp_path):
    """A process built later was not there when the copies were found. The harness
    says so from what it kept (``keep_out``): for an id the uploader has not met, and
    for one it was told to look back at (a session resumed in place)."""
    turns, wire = _turns(tmp_path, begins=True, lookback=10)
    turns.keep_out("s-1")
    turns.start("s-2", 10)
    turns.keep_out("s-2")
    copies = _turn(1) + _turn(2)
    for sid in ("s-1", "s-2"):
        assert turns.upload(copies, sid, timeout=5) is False  # a flush before the turn
        turns.begin(sid, earlier=True)
        assert turns.upload(copies + _turn(3), sid, timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3), _turn(3)] and wire.tries == 2


def test_keeping_a_session_out_forgets_nothing_it_sent(tmp_path):
    """The harness may say so in the middle of a session's life: the turns sent stay
    sent, and on the record, as the context of the next one too."""
    turns, wire = _turns(tmp_path, begins=True)
    turns.begin("s-1")
    assert turns.upload(_turn(1), "s-1", timeout=5, finished=True) is True
    turns.keep_out("s-1")
    turns.begin("s-1", earlier=True)
    assert turns.upload(_turn(1) + _turn(2), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(1), _turn(1) + _turn(2)]


def test_a_session_kept_out_already_is_left_as_it_is(tmp_path):
    """The harness reads its note at every upload and says so each time. The turns
    from before were noted once, at the first upload that showed them all: a turn of
    the session's own that no upload came for still goes up with the next one."""
    turns, wire = _turns(tmp_path, begins=True)
    turns.keep_out("s-1")
    turns.begin("s-1", earlier=True)
    assert turns.upload(_turn(1) + _turn(2), "s-1", timeout=5, finished=True) is True
    turns.begin("s-1", earlier=True)  # turn 3, cut short while the reply streamed
    turns.keep_out("s-1")
    turns.begin("s-1", earlier=True)
    said = _turn(1) + _turn(2) + _turn(3) + _turn(4)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    assert _sent(wire) == [("s-1", ["u2"]), ("s-1", ["u2", "u3", "u4"])]


def test_a_session_that_names_the_one_it_goes_on_from_is_not_said_to_be_new(tmp_path):
    """A branch holds copies and says whose: what that session is decides whether they
    may go up (``Uploader.carry_private``), and here it is on the record."""
    turns, wire = _turns(tmp_path, begins=True)
    turns.start("s-1", continues=["s-0"])
    assert turns.upload(_turn(1) + _turn(2), "s-1", timeout=5) is True
    assert _sent(wire) == [("s-1", ["u1", "u2"])]


def test_a_harness_that_does_not_say_when_turns_start_is_taken_at_its_word(tmp_path):
    turns, wire = _turns(tmp_path)
    turns.begin("s-1", earlier=True)
    assert turns.upload(_turn(1) + _turn(2), "s-1", timeout=5) is True
    assert _sent(wire) == [("s-1", ["u1", "u2"])]


# ---------------------------------------------------------------------------
# The server's answer, and who got it
# ---------------------------------------------------------------------------


def test_a_marker_of_its_own_is_not_taken_for_the_servers_answer(tmp_path):
    """`synapse private on` for a session that waits for the server's answer. The answer
    is that the session it continues is on the record, so nothing was spoken off the
    record before the marker: after `off --forget` every turn is sent, as for any
    session marked by hand."""
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


def test_a_turn_is_not_kept_once_another_upload_was_told_the_session_is_private(
    tmp_path, monkeypatch
):
    """Kept after the other upload forgot the session, the turn would go up once the
    session is back on the record, as a turn and as the context of the next. The
    session is left as the answer made it: a turn spoken off the record since, which
    no upload saw, is not looked back at either."""
    turns, wire = _turns(tmp_path)
    wire.rows.add("s-old")
    assert turns.uploader.carry_private(["s-old"], "s-new") is False
    said = _turn(1) + _turn(2) + _turn(3)
    assert _answered_meanwhile(turns, wire, monkeypatch, said) is False
    assert (tmp_path / "private" / "s-new").is_file() and wire.tries == 0
    _forget(wire, tmp_path, "s-new")
    turns.begin("s-new")
    said = said + _turn(4) + _turn(5)  # 4 was spoken off the record
    assert turns.upload(said, "s-new", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(5)]


def test_turns_another_upload_sent_meanwhile_are_not_kept_or_sent_again(tmp_path, monkeypatch):
    turns, wire = _turns(tmp_path)
    assert turns.uploader.carry_private(["s-old"], "s-new") is False
    said = _turn(1) + _turn(2) + _turn(3)
    assert _answered_meanwhile(turns, wire, monkeypatch, said) is False
    assert _records(wire) == [said]
    assert turns.upload(said + _turn(4), "s-new", timeout=5, finished=True) is True
    assert _records(wire)[1:] == [_turn(3) + _turn(4)]  # 3 as the context of 4
