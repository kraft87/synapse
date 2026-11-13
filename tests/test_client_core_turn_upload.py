"""What a ``TurnUploader`` remembers, kept per session id (plugin/synapse_client/turn_upload.py).

A harness names the session at every upload and may name several over its life: the one
it was moved to, a late upload for the one it left. Pinned here:
  * turns kept for one session (held back for the server's answer, or lost with no
    outbox) go up under that session's id or not at all;
  * each session has its own look-back, and a session that was off the record is never
    looked back at, under any id it goes on with;
  * of the turns such a session holds when it is first found on the record, one goes
    up only when it was seen to begin there (``begin``): not one that began off it,
    holds more than its own exchange, or was there before any turn start was seen;
  * a session said to be new is not believed when its turn starts show otherwise;
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


def test_a_turn_start_in_one_session_ends_the_turn_running_in_another(tmp_path):
    """One turn runs at a time. A turn begun on the record in s-a and cut short is not
    the one a later upload for s-a ends, once a turn has begun elsewhere: nothing says
    the turn that upload holds last is not one from before."""
    turns, wire = _turns(tmp_path)
    turns.start("s-a", 0)
    turns.begin("s-a")  # cut short: no upload ends it
    turns.begin("s-b")
    assert turns.upload(_turn(1) + _turn(2), "s-a", timeout=5, finished=True) is False
    assert wire.tries == 0


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
