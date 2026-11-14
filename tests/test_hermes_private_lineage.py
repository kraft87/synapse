"""A session that holds a private session's turns is private too, whichever agent runs it.

Private mode is kept per session id, and Hermes gives a conversation a new one when it
is branched, or compressed with rotation on. The provider learns of the new id in two
ways: the agent is moved to it (``on_session_switch``, tests/test_hermes_provider.py),
or Hermes builds a new agent for it, as the TUI and Desktop do for every branch
(tui_gateway/methods_session_branch.py ``_branch_live``: the child's row is written,
then the agent is built) and the gateway daemon does after a restart. These are the
tests of the second way: what the session store says a session holds turns of
(lineage.continues), what ``initialize`` does with it, and what is recorded of a session
once that is settled (carried.py).

The fixtures are the provider tests' own: the plugin as Hermes loads it, a fake server
that keeps the rows of the private-session routes, a ``state.db`` with the columns the
rule reads.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

from tests import test_hermes_gateway_sessions as gs
from tests import test_hermes_privacy_audiences as pa
from tests.helpers.client_core import use_core

use_core()
from synapse_client.harnesses.hermes import carried  # noqa: E402
from synapse_client.harnesses.hermes import lineage as hl  # noqa: E402
from synapse_client.outbox import Outbox  # noqa: E402

hs = gs.hs
hermes_stub = gs.hermes_stub
user = gs.user
plugin = gs.plugin
server = gs.server
_process_env = pa._process_env
_new = gs._new
_conversation = gs._conversation
_say = gs._say
_users = gs._users
_store = gs._store

pytestmark = gs.pytestmark

_GRAND = "20261111_090000_0a0a0a"
_PARENT = "20261112_101500_ab12cd"
_CHILD = "20261112_120000_ffffff"
_BRANCHED = {"_branched_from": _PARENT}


# ---------------------------------------------------------------------------
# What the store says
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("parent", "model_config", "expected"),
    [
        (None, None, None),  # a first session
        ("", {"max_iterations": 90}, None),
        ("p", None, "p"),  # an in-process compression writes the link alone
        ("p", {"_compressed_from": "p"}, "p"),
        ("p", {"_branched_from": "p"}, "p"),  # a copy of p's turns
        ("p", {"_reset_from": "p"}, None),  # /new: another conversation
        ("p", {"_delegate_from": "p"}, None),  # a subagent
        # A continuation inherits its parent's model_config whole: these keys speak of
        # a session further up, and the row goes on from p.
        ("p", {"_reset_from": "g"}, "p"),
        ("p", {"_branched_from": "g"}, "p"),
        ("p", {"_delegate_from": "g", "_compressed_from": "p"}, "p"),
        ("p", "not json", "p"),
        ("p", "[1, 2]", "p"),
        # The parent was deleted, and Hermes cleared the link: the copy is still there.
        (None, {"_branched_from": "p"}, "p"),
        (None, {"_compressed_from": "p"}, "p"),
        (None, {"_compressed_from": "c", "_branched_from": "b"}, "c"),
        (None, {"_reset_from": "p"}, None),
        (None, {"_branched_from": 7}, None),
        (None, "not json", None),
        (7, {"_branched_from": "p"}, "p"),
    ],
)
def test_the_session_whose_turns_a_row_holds(parent, model_config, expected):
    config = json.dumps(model_config) if isinstance(model_config, dict) else model_config
    assert hl.continues(parent, config) == expected


def _child(
    home: Path,
    model_config: dict[str, Any] | None = _BRANCHED,
    *,
    parent: str = _PARENT,
    child: str = _CHILD,
) -> None:
    """Two rows of the in-process TUI: ``parent``, and ``child`` linked to it."""
    for sid in (parent, child):
        _store(home, sid, source="tui", receipt=False, chat_id="")
    _link(home, child, parent, model_config)


def _link(home: Path, child: str, parent: str, model_config: dict[str, Any] | None) -> None:
    conn = sqlite3.connect(home / "state.db")
    try:
        conn.execute(
            "UPDATE sessions SET parent_session_id=?, model_config=? WHERE id=?",
            (parent, json.dumps(model_config) if model_config is not None else None, child),
        )
        conn.commit()
    finally:
        conn.close()


def test_the_verdict_names_the_session_a_branch_holds_turns_of(tmp_path):
    _child(tmp_path)
    assert hs.check_session(tmp_path, _CHILD, gs._in_process()) == hs.Verdict(
        hs.ACTIVE, continues=(_PARENT,)
    )
    assert hs.check_session(tmp_path, _PARENT, gs._in_process()) == hs.Verdict(hs.ACTIVE)
    assert hs.recheck(tmp_path, _CHILD).continues == (_PARENT,)
    _child(tmp_path, {"_reset_from": _PARENT})
    assert hs.check_session(tmp_path, _CHILD, gs._in_process()) == hs.Verdict(hs.ACTIVE)


def _three(home: Path, model_config: dict[str, Any] | None = None) -> None:
    """``_CHILD`` branched from ``_PARENT``, which came from ``_GRAND`` (a branch of it,
    unless ``model_config`` says otherwise)."""
    _child(home)
    _child(home, model_config or {"_branched_from": _GRAND}, parent=_GRAND, child=_PARENT)


def test_the_verdict_names_every_session_up_the_chain_of_copies(tmp_path):
    """A branch of a branch holds the turns of both."""
    _three(tmp_path)
    verdict = hs.check_session(tmp_path, _CHILD, gs._in_process())
    assert verdict.continues == (_PARENT, _GRAND)


def test_the_chain_ends_at_a_session_that_holds_none_of_its_parents_turns(tmp_path):
    _three(tmp_path, {"_reset_from": _GRAND})
    assert hs.check_session(tmp_path, _CHILD, gs._in_process()).continues == (_PARENT,)


def test_rows_that_name_each_other_are_read_once(tmp_path):
    _child(tmp_path)
    _link(tmp_path, _PARENT, _CHILD, {"_branched_from": _CHILD})
    conn = sqlite3.connect(tmp_path / "state.db")
    try:
        assert hl.chain(conn, _PARENT, json.dumps(_BRANCHED)) == (_PARENT, _CHILD)
    finally:
        conn.close()


def test_the_verdict_of_a_gateway_branch_names_its_parent_and_keeps_its_cwd(tmp_path):
    _store(tmp_path)
    pa._branch(tmp_path)
    verdict = hs.check_session(tmp_path, pa._BRANCH, gs._gateway(pa._BRANCH))
    assert (verdict.mode, verdict.continues, verdict.cwd) == (hs.ACTIVE, (gs._LOCAL,), gs._WORK)


def test_a_session_that_is_off_names_none(tmp_path):
    _child(tmp_path)
    gs._retitle(tmp_path, _CHILD, "Bot Chat")
    verdict = hs.check_session(tmp_path, _CHILD, gs._in_process())
    assert verdict.mode == hs.INERT and verdict.continues == ()


# ---------------------------------------------------------------------------
# The record of what was settled
# ---------------------------------------------------------------------------


def test_a_session_is_settled_once_it_is_recorded(tmp_path):
    assert not carried.settled(tmp_path, _CHILD)
    carried.settle(tmp_path, _CHILD)
    carried.settle(tmp_path, _CHILD)
    assert carried.settled(tmp_path, _CHILD) and not carried.settled(tmp_path, _PARENT)
    (record,) = (tmp_path / "synapse" / "settled").iterdir()
    assert len(record.name) == 64 and record.read_bytes() == b""
    assert record.stat().st_mode & 0o777 == 0o600
    # The names are hashes of session ids: the folders made for them are the owner's.
    for folder in (record.parent, record.parent.parent):
        assert folder.stat().st_mode & 0o777 == 0o700


class _Carrier:
    """An uploader's ``carry_private``, with a set answer."""

    def __init__(self, made: bool) -> None:
        self.made = made
        self.calls: list[tuple[list[str], str, float | None]] = []
        self.settled: Any = None

    def carry_private(
        self, earlier: list[str], session_id: str, *, timeout: float | None, settled: Any
    ) -> bool:
        self.calls.append((list(earlier), session_id, timeout))
        self.settled = settled
        return self.made


@pytest.mark.parametrize("made", [False, True], ids=["on the record", "made private"])
def test_a_carry_asks_once_and_records_that_it_did(tmp_path, made):
    """The answer is the uploader's: whether the session was made private. The record
    is written once the question is closed, not before, and says only that the session
    was asked about: whether it is private is for its marker and the server's row to
    say, and put back on the record it is back whole."""
    uploader = _Carrier(made)
    assert carried.carry(uploader, tmp_path, [_PARENT, None, _GRAND], _CHILD, 5) is made
    assert uploader.calls == [([_PARENT, _GRAND], _CHILD, 5)]
    assert not carried.settled(tmp_path, _CHILD)
    uploader.settled(made)
    assert carried.settled(tmp_path, _CHILD)
    (record,) = (tmp_path / "synapse" / "settled").iterdir()
    assert record.read_bytes() == b""


def test_a_settled_session_is_not_carried_again(tmp_path):
    uploader = _Carrier(True)
    carried.settle(tmp_path, _CHILD)
    assert carried.carry(uploader, tmp_path, [_PARENT], _CHILD, None) is False
    # With no session to go by, or no id, there is nothing to ask or to settle.
    assert carried.carry(uploader, tmp_path, [None, ""], _GRAND, None) is False
    assert carried.carry(uploader, tmp_path, [_PARENT], "", None) is False
    assert uploader.calls == [] and not carried.settled(tmp_path, _GRAND)


def test_a_chain_is_asked_about_as_far_as_its_first_settled_session(tmp_path):
    chain = (_PARENT, _GRAND, "20261110_080000_0b0b0b")
    assert carried.to_ask(tmp_path, chain) == list(chain)
    carried.settle(tmp_path, _GRAND)
    assert carried.to_ask(tmp_path, chain) == [_PARENT, _GRAND]
    carried.settle(tmp_path, _PARENT)
    assert carried.to_ask(tmp_path, chain) == [_PARENT]
    assert carried.to_ask(tmp_path, ()) == [] and carried.to_ask(None, chain) == list(chain)


@pytest.mark.parametrize("home", [None, ""])
def test_without_a_home_nothing_is_settled(home):
    carried.settle(home, _CHILD)
    assert not carried.settled(home, _CHILD)


def test_a_record_that_cannot_be_kept_leaves_the_session_unsettled(tmp_path):
    (tmp_path / "synapse").write_text("a file where the folder goes", encoding="utf-8")
    carried.settle(tmp_path, _CHILD)
    assert not carried.settled(tmp_path, _CHILD)
    carried.settle(tmp_path, "")
    assert not carried.settled(tmp_path, "")


def test_a_session_found_to_hold_copies_has_a_record_of_its_own(tmp_path):
    """Kept apart from what was settled: it says what the session holds, and closes no
    question about a session its row may name."""
    assert not carried.copied(tmp_path, _CHILD)
    carried.note_copied(tmp_path, _CHILD)
    carried.note_copied(tmp_path, _CHILD)
    assert carried.copied(tmp_path, _CHILD) and not carried.copied(tmp_path, _PARENT)
    assert not carried.settled(tmp_path, _CHILD)
    (record,) = (tmp_path / "synapse" / "copied").iterdir()
    assert len(record.name) == 64 and record.read_bytes() == b""
    assert record.stat().st_mode & 0o777 == 0o600
    for folder in (record.parent, record.parent.parent):
        assert folder.stat().st_mode & 0o777 == 0o700
    assert not (tmp_path / "synapse" / "settled").exists()
    carried.settle(tmp_path, _PARENT)  # neither record is taken for the other
    assert not carried.copied(tmp_path, _PARENT)


def test_copies_that_cannot_be_recorded_are_not_kept(tmp_path, monkeypatch):
    here, home = tmp_path / "cwd", tmp_path / "home"
    here.mkdir()
    home.mkdir()
    monkeypatch.chdir(here)
    for none in (None, ""):
        carried.note_copied(none, _CHILD)
        assert not carried.copied(none, _CHILD)
    carried.note_copied(home, "")
    assert not carried.copied(home, "")
    assert list(here.iterdir()) == [] and list(home.iterdir()) == []
    (home / "synapse").write_text("a file where the folder goes", encoding="utf-8")
    carried.note_copied(home, _CHILD)
    assert not carried.copied(home, _CHILD)


def test_a_session_holds_copies_by_its_own_record_or_that_of_one_it_goes_on_from(tmp_path):
    """Whether a session's turns from before are looked back at. A fork's copies are in
    what goes on from it too: that session is noted as holding copies itself, for an
    agent built later and for whatever goes on from it in turn."""
    assert not carried.copies(tmp_path, _CHILD)
    assert not carried.copies(tmp_path, _CHILD, [_PARENT, None, ""])
    assert not (tmp_path / "synapse").exists()  # nothing was found: nothing is noted
    carried.note_copied(tmp_path, _GRAND)
    assert carried.copies(tmp_path, _GRAND) and carried.copies(tmp_path, _GRAND, [_PARENT])
    assert not carried.copies(tmp_path, _CHILD, [_PARENT])
    assert not carried.copied(tmp_path, _CHILD)
    assert carried.copies(tmp_path, _CHILD, [_PARENT, None, _GRAND])
    assert carried.copied(tmp_path, _CHILD)  # its own record answers from now on
    assert carried.copies(tmp_path, _CHILD) and not carried.copied(tmp_path, _PARENT)
    # Nothing is settled by it, and a session with no id holds nothing.
    assert not (tmp_path / "synapse" / "settled").exists()
    assert not carried.copies(tmp_path, "", [_GRAND]) and not carried.copies(None, _CHILD)


def test_copies_stay_out_whatever_is_settled(tmp_path):
    """Settled as on the record, and found to hold copies: in either order, since the
    question can close after the copies are found (a fork moved to a new id before its
    first turn). The copies stay out, and the session is not asked about again."""
    uploader = _Carrier(True)
    carried.settle(tmp_path, _CHILD)
    assert not carried.copies(tmp_path, _CHILD, [_PARENT])
    carried.note_copied(tmp_path, _CHILD)
    assert carried.copies(tmp_path, _CHILD, [_PARENT])
    assert carried.carry(uploader, tmp_path, [_PARENT], _CHILD, None) is False
    assert uploader.calls == []


@pytest.mark.parametrize("made", [False, True], ids=["on the record", "made private"])
@pytest.mark.parametrize("whose", ["its own", "those of a session it goes on from"])
def test_a_session_with_copies_is_asked_about_all_the_same_and_keeps_them_out(
    tmp_path, made, whose
):
    """A copies record closes no question: the session a row names may be private
    (Desktop writes a branch's row at its first prompt, after its agent was built), so
    it is asked about. Whatever the answer, the copies are not the session's to
    upload, nor those of one that goes on from it."""
    carried.note_copied(tmp_path, _CHILD if whose == "its own" else _GRAND)
    uploader = _Carrier(made)
    assert carried.carry(uploader, tmp_path, [_PARENT, _GRAND], _CHILD, 5) is made
    assert uploader.calls == [([_PARENT, _GRAND], _CHILD, 5)]
    uploader.settled(made)
    assert carried.settled(tmp_path, _CHILD)
    assert carried.copies(tmp_path, _CHILD, [_PARENT, _GRAND])
    assert carried.copied(tmp_path, _CHILD)


# ---------------------------------------------------------------------------
# A new agent for a session that holds another's turns
# ---------------------------------------------------------------------------


def _open(plugin: Any, home: Path, sid: str = _CHILD, **kwargs: Any) -> Any:
    """A new agent for ``sid``, as the stdio TUI builds one."""
    provider = _new(plugin)
    provider.initialize(sid, hermes_home=str(home), **gs._in_process(**kwargs))
    return provider


def _marker(tmp_path: Path, session_id: str) -> Path:
    marker = tmp_path / "private" / session_id
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("", encoding="utf-8")
    return marker


def test_a_branch_of_a_private_session_is_private_from_its_first_turn(plugin, server, tmp_path):
    """The TUI builds a new agent for a branch, with the parent's turns copied into it:
    no ``on_session_switch`` says where it came from, only its row."""
    home = tmp_path / "hermes-a"
    _child(home)
    _marker(tmp_path, _PARENT)
    provider = _open(plugin, home, resumed=True, message_count=6)
    assert (tmp_path / "private" / _CHILD).is_file()
    # The server is told as the session is bound: `private off --forget` on the branch
    # before its first turn is then not undone by a row written later.
    assert server.puts() == [f"/private-sessions/{_CHILD}"] and server.asked() == []
    assert carried.settled(home, _CHILD)
    _say(provider, _conversation(4), _CHILD)
    provider.on_session_end(_conversation(4))
    assert server.posts() == [] and server.puts() == [f"/private-sessions/{_CHILD}"]


def test_a_branch_put_back_on_the_record_is_back_whole(plugin, server, tmp_path):
    """`private off --forget` on the branch while its agent lives. Private mode covers
    whole sessions, so the branch is back with every turn it holds: the ones it was
    made with too, as far as a resumed session is looked back at. The parent stays
    private under its own id."""
    home = tmp_path / "hermes-a"
    _child(home)
    _marker(tmp_path, _PARENT)
    provider = _open(plugin, home, resumed=True)
    _say(provider, _conversation(3), _CHILD)
    assert server.posts() == [] and server.puts() == [f"/private-sessions/{_CHILD}"]

    (tmp_path / "private" / _CHILD).unlink()
    server.private_rows.discard(_CHILD)
    _say(provider, _conversation(4), _CHILD)
    provider.on_session_end(_conversation(4))
    whole = ["question 1", "question 2", "question 3", "question 4"]
    assert [_users(body["records"]) for body in server.posts()] == [whole]
    assert {body["session_id"] for body in server.posts()} == {_CHILD}
    _say(provider, _conversation(5), _CHILD)
    assert _users(server.posts()[-1]["records"]) == ["question 4", "question 5"]
    assert len(server.posts()) == 2 and len(server.puts()) == 1  # and no row again


@pytest.mark.parametrize("resumed", [True, False])
def test_a_branch_of_a_session_the_server_holds_private_is_private(
    plugin, server, tmp_path, resumed
):
    """`private off` removed the parent's marker, or its twelve hours passed. The row on
    the server still drops the parent's turns, and the branch holds copies of them."""
    home = tmp_path / "hermes-a"
    _child(home)
    server.private_rows.add(_PARENT)
    provider = _open(plugin, home, resumed=resumed)
    assert not (tmp_path / "private" / _CHILD).exists() and server.asked() == []
    _say(provider, _conversation(3), _CHILD)
    assert server.asked() == [_PARENT] and server.puts() == [f"/private-sessions/{_CHILD}"]
    assert server.posts() == [] and (tmp_path / "private" / _CHILD).is_file()
    _say(provider, _conversation(4), _CHILD)
    assert server.posts() == [] and server.asked() == [_PARENT]

    # `private off --forget` on the branch holds: the next agent does not ask again.
    (tmp_path / "private" / _CHILD).unlink()
    server.private_rows.discard(_CHILD)
    again = _open(plugin, home, resumed=True)
    _say(again, _conversation(5), _CHILD)
    assert server.asked() == [_PARENT] and len(server.puts()) == 1
    assert {body["session_id"] for body in server.posts()} == {_CHILD}


def test_a_branch_of_a_branch_is_private_with_the_session_at_the_top(plugin, server, tmp_path):
    """The session in between was carried while the server could not be reached and
    never got its row, and its marker has gone: its own answer is not enough to go by."""
    home = tmp_path / "hermes-a"
    _three(home)
    server.private_rows.add(_GRAND)
    provider = _open(plugin, home, resumed=True)
    _say(provider, _conversation(3), _CHILD)
    assert server.asked() == [_PARENT, _GRAND]
    assert server.posts() == [] and server.puts() == [f"/private-sessions/{_CHILD}"]


def test_a_settled_session_answers_for_the_ones_above_it(plugin, server, tmp_path):
    """The branch in between was made before the first session went private, and was
    asked about then: its branch is on the record as it is."""
    home = tmp_path / "hermes-a"
    _three(home)
    carried.settle(home, _PARENT)
    server.private_rows.add(_GRAND)
    provider = _open(plugin, home)
    _say(provider, _conversation(2), _CHILD)
    assert server.asked() == [_PARENT] and server.puts() == []
    assert {body["session_id"] for body in server.posts()} == {_CHILD}


def test_a_branch_made_before_its_parent_went_private_stays_on_the_record(plugin, server, tmp_path):
    home = tmp_path / "hermes-a"
    _child(home)
    provider = _open(plugin, home)
    _say(provider, _conversation(2), _CHILD)
    (first,) = server.posts()
    assert first["session_id"] == _CHILD and server.asked() == [_PARENT]
    assert _users(first["records"]) == ["question 1", "question 2"]
    assert carried.settled(home, _CHILD) and server.puts() == []

    _marker(tmp_path, _PARENT)
    server.private_rows.add(_PARENT)
    later = _open(plugin, home, resumed=True)
    assert not (tmp_path / "private" / _CHILD).exists()
    _say(later, _conversation(3), _CHILD)
    assert len(server.posts()) == 2 and server.asked() == [_PARENT] and server.puts() == []


@pytest.mark.parametrize(
    "model_config",
    [{"_reset_from": _PARENT}, {"_delegate_from": _PARENT}],
)
def test_another_conversation_is_not_private_because_the_one_before_was(
    plugin, server, tmp_path, model_config
):
    home = tmp_path / "hermes-a"
    _child(home, model_config)
    _marker(tmp_path, _PARENT)
    server.private_rows.add(_PARENT)
    provider = _open(plugin, home)
    _say(provider, _conversation(1), _CHILD)
    (body,) = server.posts()
    assert body["session_id"] == _CHILD and server.asked() == [] and server.puts() == []
    assert not (tmp_path / "private" / _CHILD).exists()


@pytest.mark.parametrize("model_config", [None, {"_compressed_from": _PARENT}])
def test_a_compressed_continuation_is_private_with_the_session_it_goes_on_from(
    plugin, server, tmp_path, model_config
):
    """The daemon restarted, or dropped the agent, after a compression moved the
    conversation to a new id: the next agent is built for that id."""
    home = tmp_path / "hermes-a"
    _child(home, model_config)
    server.private_rows.add(_PARENT)
    provider = _open(plugin, home, resumed=True)
    _say(provider, _conversation(2), _CHILD)
    assert server.posts() == [] and (tmp_path / "private" / _CHILD).is_file()


def test_while_the_server_does_not_answer_nothing_leaves_and_the_next_agent_asks_again(
    plugin, server, tmp_path
):
    home = tmp_path / "hermes-a"
    _child(home)
    server.private_rows.add(_PARENT)
    server.private_status = 503
    provider = _open(plugin, home, resumed=True)
    _say(provider, _conversation(1), _CHILD)
    provider.on_session_end(_conversation(1))
    assert server.posts() == [] and Outbox(home / "synapse" / "outbox").pending() == 0
    assert not carried.settled(home, _CHILD) and len(server.asked()) >= 2

    server.private_status = 200
    asked = len(server.asked())
    again = _open(plugin, home, resumed=True)
    _say(again, _conversation(2), _CHILD)
    assert len(server.asked()) == asked + 1 and server.posts() == []
    assert (tmp_path / "private" / _CHILD).is_file() and carried.settled(home, _CHILD)


def test_a_session_whose_marker_could_not_be_written_is_asked_about_by_the_next_agent(
    plugin, server, tmp_path, monkeypatch
):
    """No marker says the session is private: the agent that made it so holds it
    itself, and keeps no record that it was asked about. Finding one, the next agent
    would ask nothing and send what it looks back at (to a server whose row drops it)."""
    home = tmp_path / "hermes-a"
    _child(home)
    _marker(tmp_path, _PARENT)
    first = _new(plugin)
    core = type(first).__module__.rsplit(".harnesses.", 1)[0]  # the provider's own copy

    def refuse(settings: Any, session_id: str) -> Path:
        raise PermissionError(13, "read-only", session_id)

    with monkeypatch.context() as patched:
        patched.setattr(sys.modules[f"{core}.private"], "mark", refuse)
        first.initialize(_CHILD, hermes_home=str(home), **gs._in_process(resumed=True))
        _say(first, _conversation(4), _CHILD)
    assert server.puts() == [f"/private-sessions/{_CHILD}"]  # the server has its row
    assert server.posts() == [] and not (tmp_path / "private" / _CHILD).exists()
    assert not carried.settled(home, _CHILD)

    again = _open(plugin, home, resumed=True)  # a restart, and the folder can be written
    _say(again, _conversation(5), _CHILD)
    assert server.posts() == [] and (tmp_path / "private" / _CHILD).is_file()
    assert carried.settled(home, _CHILD)


def test_turns_finished_while_the_server_did_not_answer_go_up_when_it_does(
    plugin, server, tmp_path
):
    home = tmp_path / "hermes-a"
    _child(home)
    server.private_status = 503
    provider = _open(plugin, home, resumed=True)
    for count in (12, 13, 14):
        _say(provider, _conversation(count), _CHILD)
    assert server.posts() == []
    server.private_status = 200
    _say(provider, _conversation(15), _CHILD)
    sent = {user for body in server.posts() for user in _users(body["records"])}
    # Ten back from the first turn held (12), and every turn since; 2 is 3's context.
    assert sent == {f"question {n}" for n in range(2, 16)}


def test_a_gateway_branch_of_a_private_session_is_private(plugin, server, tmp_path):
    home = tmp_path / "hermes-a"
    _store(home)
    pa._branch(home)
    _marker(tmp_path, gs._LOCAL)
    provider = gs._start_gateway(_new(plugin), home, pa._BRANCH, resumed=True)
    assert (tmp_path / "private" / pa._BRANCH).is_file()
    _say(provider, _conversation(2), pa._BRANCH)
    assert server.posts() == [] and server.puts() == [f"/private-sessions/{pa._BRANCH}"]


def test_a_session_that_is_off_is_not_carried(plugin, server, tmp_path):
    home = tmp_path / "hermes-a"
    _child(home)
    _marker(tmp_path, _PARENT)
    _open(plugin, home, agent_context="subagent")
    assert not (tmp_path / "private" / _CHILD).exists() and server.puts() == []


@pytest.mark.parametrize("flush", ["on_pre_compress", "on_session_end"])
def test_a_flush_after_forget_sends_the_turns_the_session_holds(plugin, server, tmp_path, flush):
    """A compression is due, or the session ends, with no turn ended since `private off
    --forget`: the session is back whole, and the flush sends its finished turns."""
    home = tmp_path / "hermes-a"
    _child(home)
    _marker(tmp_path, _PARENT)
    provider = _open(plugin, home, resumed=True)
    _say(provider, _conversation(3), _CHILD)
    (tmp_path / "private" / _CHILD).unlink()
    server.private_rows.discard(_CHILD)
    getattr(provider, flush)(_conversation(3))
    whole = ["question 1", "question 2", "question 3"]
    assert [_users(body["records"]) for body in server.posts()] == [whole]
    _say(provider, _conversation(4), _CHILD)
    assert _users(server.posts()[-1]["records"]) == ["question 3", "question 4"]
    assert len(server.posts()) == 2


def test_a_branch_put_back_on_the_record_before_its_first_turn_stays_there(
    plugin, server, tmp_path
):
    """The server was told as the branch was bound, so `private off --forget` run
    before the first turn deletes a row that is there, and no later request writes it
    again."""
    home = tmp_path / "hermes-a"
    _child(home)
    _marker(tmp_path, _PARENT)
    provider = _open(plugin, home, resumed=True)
    (tmp_path / "private" / _CHILD).unlink()
    server.private_rows.discard(_CHILD)
    _say(provider, _conversation(4), _CHILD)
    assert server.puts() == [f"/private-sessions/{_CHILD}"] and server.private_rows == set()
    whole = ["question 1", "question 2", "question 3", "question 4"]
    assert [_users(body["records"]) for body in server.posts()] == [whole]


def test_a_server_that_cannot_be_told_as_a_branch_is_bound_is_a_warning(
    plugin, server, tmp_path, caplog
):
    """Until the server has the row, `private off --forget` is undone when it gets it:
    the log says so at once, and again at each turn the server still cannot be told."""
    home = tmp_path / "hermes-a"
    _child(home)
    _marker(tmp_path, _PARENT)
    server.private_status = 503
    with caplog.at_level(logging.WARNING):
        provider = _open(plugin, home, resumed=True)
        assert (tmp_path / "private" / _CHILD).is_file() and not carried.settled(home, _CHILD)
        _say(provider, _conversation(4), _CHILD)
    untold = [r.getMessage() for r in caplog.records if "could not be told" in r.getMessage()]
    assert len(untold) == 2 and all(_CHILD in line for line in untold)
    assert all(r.levelno == logging.WARNING for r in caplog.records if r.getMessage() in untold)
    server.private_status = 200
    _say(provider, _conversation(5), _CHILD)
    assert server.posts() == [] and _CHILD in server.private_rows
    assert carried.settled(home, _CHILD)


# ---------------------------------------------------------------------------
# A session the agent reaches without being built for it
# ---------------------------------------------------------------------------

_OTHER = "20261113_090000_0c0c0c"


def _private_parent(server: Any, tmp_path: Path, known: str) -> list[str]:
    """Make ``_PARENT`` private by its marker, or by the server's row alone (`private
    off`, or the marker's twelve hours). Returns what the server is then asked."""
    if known == "marker":
        _marker(tmp_path, _PARENT)
        return []
    server.private_rows.add(_PARENT)
    return [_PARENT]


@pytest.mark.parametrize("known", ["marker", "row"])
def test_a_session_resumed_in_place_answers_for_the_session_it_continues(
    plugin, server, tmp_path, known
):
    """`/resume` on a live agent opens another conversation, so nothing is carried to
    it from the one left. What its own row says it holds turns of still counts, as it
    does for a new agent built for the same session."""
    home = tmp_path / "hermes-a"
    _child(home)
    _store(home, _OTHER, source="tui", receipt=False, chat_id="")
    asked = _private_parent(server, tmp_path, known)
    provider = _open(plugin, home, _OTHER)
    provider.on_session_switch(_CHILD, parent_session_id=_OTHER, reason="resume")
    assert server.asked() == [] and server.puts() == []
    _say(provider, _conversation(4), _CHILD)
    provider.on_session_end(_conversation(4))
    assert server.posts() == [] and (tmp_path / "private" / _CHILD).is_file()
    assert server.asked() == asked and server.puts() == [f"/private-sessions/{_CHILD}"]
    assert carried.settled(home, _CHILD)


@pytest.mark.parametrize("known", ["marker", "row"])
def test_an_upload_for_an_id_the_agent_was_not_moved_to_goes_by_its_row(
    plugin, server, tmp_path, known
):
    """Hermes can point its agent at a compression's new id without a call to the
    provider (agent/turn_facade_lease.py): the next ``sync_turn`` names an id this
    instance was never given. The id's row says whose turns it holds."""
    home = tmp_path / "hermes-a"
    _child(home, None)  # a rotation child: the link alone
    asked = _private_parent(server, tmp_path, known)
    provider = _open(plugin, home, _PARENT, resumed=True)
    _say(provider, _conversation(4), _CHILD)
    assert server.posts() == [] and (tmp_path / "private" / _CHILD).is_file()
    assert server.asked() == asked and server.puts() == [f"/private-sessions/{_CHILD}"]
    # Read once: the session is not carried again at each turn, and `private off
    # --forget` puts it back whole.
    (tmp_path / "private" / _CHILD).unlink()
    server.private_rows.discard(_CHILD)
    _say(provider, _conversation(5), _CHILD)
    whole = [f"question {n}" for n in range(1, 6)]
    assert [_users(body["records"]) for body in server.posts()] == [whole]
    assert server.asked() == asked and len(server.puts()) == 1


def test_a_row_written_after_the_agent_was_built_is_read_at_the_first_upload(
    plugin, server, tmp_path
):
    """Desktop builds the agent for a branch whose row it could not write, and writes
    row and copied turns at the first prompt (tui_gateway/methods_prompt.py): at
    ``initialize`` the session has no row, and so nothing it continues."""
    home = tmp_path / "hermes-a"
    _store(home, _PARENT, source="tui", receipt=False, chat_id="")
    server.private_rows.add(_PARENT)
    provider = _open(plugin, home, message_count=0)
    assert server.asked() == [] and not carried.settled(home, _CHILD)
    _store(home, _CHILD, source="tui", receipt=False, chat_id="")
    _link(home, _CHILD, _PARENT, _BRANCHED)
    _say(provider, _conversation(4), _CHILD)
    assert server.posts() == [] and (tmp_path / "private" / _CHILD).is_file()
    assert server.asked() == [_PARENT] and server.puts() == [f"/private-sessions/{_CHILD}"]


def test_a_row_that_names_no_session_is_read_again_before_the_next_upload(plugin, server, tmp_path):
    """A row that names no session settles nothing: it is read before every upload
    until it names one. No order of Hermes's calls is known that puts an upload before
    the write above (test_hermes_checkout_names.py pins the order of a branch). An
    upload that found no row sends what it was shown, as for any session nothing names
    a parent of; the session is read again, and it is off the record from the upload
    after its row names a private one."""
    home = tmp_path / "hermes-a"
    _store(home, _PARENT, source="tui", receipt=False, chat_id="")
    server.private_rows.add(_PARENT)
    provider = _open(plugin, home, message_count=0)
    provider.on_pre_compress(_conversation(3))
    assert server.asked() == [] and len(server.posts()) == 1
    _store(home, _CHILD, source="tui", receipt=False, chat_id="")
    _link(home, _CHILD, _PARENT, _BRANCHED)
    _say(provider, _conversation(4), _CHILD)
    assert len(server.posts()) == 1 and (tmp_path / "private" / _CHILD).is_file()
    assert server.asked() == [_PARENT] and server.puts() == [f"/private-sessions/{_CHILD}"]


def test_an_id_is_read_once_also_when_its_answer_could_not_be_recorded(
    plugin, server, tmp_path, monkeypatch
):
    """The record under ``settled/`` is what the next agent goes by. This one asks once
    per id whatever became of the record: asked again at every turn, a session put back
    on the record would be taken off it again."""
    home = tmp_path / "hermes-a"
    _child(home, None)
    provider = _open(plugin, home, _PARENT, resumed=True)
    module = sys.modules[type(provider).__module__]  # the provider's own copy of the core
    monkeypatch.setattr(module.carried, "settle", lambda home, session_id, made=False: None)
    _say(provider, _conversation(2), _CHILD)
    _say(provider, _conversation(3), _CHILD)
    assert server.asked() == [_PARENT] and len(server.posts()) == 2
    assert not carried.settled(home, _CHILD)


def _carry_raises(monkeypatch: Any, provider: Any) -> None:
    def broken(*args: Any, **kwargs: Any) -> bool:
        raise RuntimeError("the carry broke")

    module = sys.modules[type(provider).__module__]  # the provider's own copy of the core
    monkeypatch.setattr(module.carried, "settled", broken)


def test_a_switch_whose_carry_raises_turns_the_provider_off(plugin, server, tmp_path, monkeypatch):
    """Whether the new id is private was not found out. Uploading under it anyway would
    be a guess, so the instance is off for good, as after an ``initialize`` that
    raised."""
    home = tmp_path / "hermes-a"
    _child(home, None)
    _marker(tmp_path, _PARENT)
    provider = _open(plugin, home, _PARENT, resumed=True)
    with monkeypatch.context() as patch:
        _carry_raises(patch, provider)
        provider.on_session_switch(_CHILD, parent_session_id=_PARENT, reason="compression")
    _say(provider, _conversation(3), _CHILD)
    provider.on_session_end(_conversation(3))
    assert server.posts() == [] and Outbox(home / "synapse" / "outbox").pending() == 0
    assert provider._mode == "inert"


def test_an_upload_whose_carry_raises_sends_nothing_and_reads_the_row_again(
    plugin, server, tmp_path, monkeypatch
):
    home = tmp_path / "hermes-a"
    _child(home, None)
    server.private_rows.add(_PARENT)
    provider = _open(plugin, home, _PARENT, resumed=True)
    with monkeypatch.context() as patch:
        _carry_raises(patch, provider)
        _say(provider, _conversation(3), _CHILD)
    assert server.posts() == [] and server.asked() == []
    _say(provider, _conversation(4), _CHILD)
    assert server.posts() == [] and server.asked() == [_PARENT]
    assert (tmp_path / "private" / _CHILD).is_file()


def test_turns_compressed_away_while_the_server_does_not_answer_go_up_when_it_does(
    plugin, server, tmp_path
):
    """A compression in place (Hermes's default) keeps the session id and replaces the
    older turns by a summary. The ones that finished while the session was held were
    kept by the uploader, and are sent once the server says it is on the record."""
    home = tmp_path / "hermes-a"
    _child(home)
    server.private_status = 503
    provider = _open(plugin, home, resumed=True)
    whole = _conversation(8)
    _say(provider, whole, _CHILD)
    provider.on_pre_compress(whole)
    provider.on_session_end(whole)
    assert server.posts() == [] and Outbox(home / "synapse" / "outbox").pending() == 0
    provider.on_session_switch(_CHILD, parent_session_id=_CHILD, reason="compression")
    nine = [whole[0], *whole[-4:], *_conversation(9)[-2:]]  # turns 7 and 8, then 9
    server.private_status = 200
    _say(provider, nine, _CHILD)
    sent = [text for body in server.posts() for text in _users(body["records"])]
    assert sent == [f"question {n}" for n in range(1, 10)]  # each once, oldest first
    assert carried.settled(home, _CHILD)
