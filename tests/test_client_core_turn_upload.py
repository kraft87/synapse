"""What a ``TurnUploader`` remembers, kept per session id (plugin/synapse_client/turn_upload.py).

A harness names the session at every upload and may name several over its life: the one
it was moved to, a late upload for the one it left. Pinned here:
  * turns kept for one session (held back for the server's answer, or lost with no
    outbox) go up under that session's id or not at all;
  * each session has its own look-back, and a session that was off the record is never
    looked back at, under any id it goes on with;
  * a turn that only goes on from one spoken off the record (the harness began it, or
    left its user row out) is not a new turn;
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


def test_a_session_that_was_off_the_record_is_never_looked_back_at(tmp_path):
    """Whatever look-back is asked for later, for the session itself or for one that
    goes on from it: its turns are in the conversation those sessions hold."""
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
    before, spoken off the record: the last turn of the records is complete, and it is
    not the one that has just ended. Nor is it when the harness goes on once more. The
    next turn the user begins goes up, alone."""
    turns, wire = _turns(tmp_path, lookback=0)
    said = _turn(1) + _turn(2) + _goes_on(1, shown=shown)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False
    said = said + _goes_on(2, shown=shown)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is False and wire.tries == 0
    assert turns.upload(said + _turn(3), "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(3)]


def test_a_turn_spoken_off_the_record_stays_there_when_the_harness_goes_on_from_it(tmp_path):
    """A flush used the look-back up before any turn ended: the turn that then goes on
    is still one from before."""
    turns, wire = _turns(tmp_path, lookback=0)
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-1", timeout=5) is False
    assert turns.upload(said + _goes_on(1, shown=True), "s-1", timeout=5, finished=True) is False
    assert wire.tries == 0


def test_a_row_the_harness_adds_after_a_reply_does_not_make_the_turn_one_from_before(tmp_path):
    """A task-list snapshot after the reply (a synthetic row, and no reply to it): the
    turn has gone on from nothing, and it is the one that has just ended."""
    turns, wire = _turns(tmp_path, lookback=0)
    snapshot = {"role": "user", "content": "[tasks]", "synthetic": True, "id": "t1"}
    said = [*_turn(1), *_turn(2), snapshot]
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    assert _records(wire) == [[*_turn(2), snapshot]]


def test_a_reply_the_user_did_not_ask_for_goes_up_with_a_turn_on_the_record(tmp_path):
    turns, wire = _turns(tmp_path)
    said = _turn(1) + _turn(2)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    said = said + _goes_on(1, shown=True)
    assert turns.upload(said, "s-1", timeout=5, finished=True) is True
    assert _records(wire)[1] == said  # turn 2 as it is now, after turn 1 as its context


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
    session is back on the record, as a turn and as the context of the next."""
    turns, wire = _turns(tmp_path)
    wire.rows.add("s-old")
    assert turns.uploader.carry_private(["s-old"], "s-new") is False
    said = _turn(1) + _turn(2) + _turn(3)
    assert _answered_meanwhile(turns, wire, monkeypatch, said) is False
    assert (tmp_path / "private" / "s-new").is_file() and wire.tries == 0
    _forget(wire, tmp_path, "s-new")
    assert turns.upload(said + _turn(4), "s-new", timeout=5, finished=True) is True
    assert _records(wire) == [_turn(4)]


def test_turns_another_upload_sent_meanwhile_are_not_kept_or_sent_again(tmp_path, monkeypatch):
    turns, wire = _turns(tmp_path)
    assert turns.uploader.carry_private(["s-old"], "s-new") is False
    said = _turn(1) + _turn(2) + _turn(3)
    assert _answered_meanwhile(turns, wire, monkeypatch, said) is False
    assert _records(wire) == [said]
    assert turns.upload(said + _turn(4), "s-new", timeout=5, finished=True) is True
    assert _records(wire)[1:] == [_turn(3) + _turn(4)]  # 3 as the context of 4
