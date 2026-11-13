"""A conversation off the record stays off it, whatever Hermes does with its session next.

tests/test_hermes_private_lineage.py covers how the provider learns that a session holds
a private session's turns. These are the ways such turns could still leave afterwards:

  * a flush hook (``on_session_end``, ``on_pre_compress``) names no session, and the
    agent may be in another than the one the provider was built for or switched to
    (harnesses/hermes/live.py);
  * after ``synapse private off --forget`` on a carried session, the turns from before
    are still in the conversation: a turn Hermes begins itself lands on the last of
    them, and a compression that starts a new session copies them under a new id;
  * a late upload for the session the agent left, while turns are kept for it.

The fixtures and helpers are those of the lineage tests.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from tests import test_hermes_gateway_sessions as gs
from tests import test_hermes_private_lineage as pl
from tests.helpers.client_core import use_core

use_core()
from synapse_client.harnesses.hermes import carried  # noqa: E402
from synapse_client.harnesses.hermes.live import Live  # noqa: E402

hermes_stub = gs.hermes_stub
user = gs.user
plugin = gs.plugin
server = gs.server
_conversation = gs._conversation
_users = gs._users
_store = gs._store
_open, _marker, _child, _link = pl._open, pl._marker, pl._child, pl._link

pytestmark = gs.pytestmark

_GRAND, _PARENT, _CHILD, _OTHER = pl._GRAND, pl._PARENT, pl._CHILD, pl._OTHER
_THIRD = "20261113_100000_dddddd"


def _posted(server: Any) -> list[tuple[str, list[str]]]:
    """Each upload: its session id, and the texts of its user rows."""
    return [(body["session_id"], _users(body["records"])) for body in server.posts()]


def _questions(first: int, last: int, tag: str = "question") -> list[str]:
    return [f"{tag} {n}" for n in range(first, last + 1)]


def _other_conversation(count: int) -> list[dict[str, Any]]:
    """Another conversation: other texts, other message ids."""
    rows: list[dict[str, Any]] = [{"role": "system", "content": "You are Hermes."}]
    for n in range(1, count + 1):
        rows += [
            {"role": "user", "content": f"other question {n}", "message_uid": f"ou{n}"},
            {"role": "assistant", "content": f"other answer {n}", "message_uid": f"oa{n}"},
        ]
    return rows


def _notice(n: int) -> list[dict[str, Any]]:
    """A turn Hermes begins itself: a background-process notice in the user's place
    (tools/process_registry_notifications.py), and the model's reply."""
    return [
        {
            "role": "user",
            "content": f"[IMPORTANT: Background process proc_{n} completed (exit code 0).]",
            "message_uid": f"n{n}",
        },
        {"role": "assistant", "content": f"noted {n}", "message_uid": f"an{n}"},
    ]


def _forget(server: Any, tmp_path: Path, session_id: str) -> None:
    """`synapse private off --forget`: the marker and the server's row go."""
    (tmp_path / "private" / session_id).unlink()
    server.private_rows.discard(session_id)


# ---------------------------------------------------------------------------
# A hook that names no session
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flush", ["on_pre_compress", "on_session_end"])
def test_a_flush_goes_under_the_session_hermes_last_named(plugin, server, tmp_path, flush):
    """The agent was built for _PARENT. Hermes points it at _CHILD, the session another
    holder's compression started, with no call to the provider
    (agent/turn_facade_lease.py): ``sync_turn`` names _CHILD from then on, the flush
    hooks name nothing. Under _PARENT, which has no row, the flush would upload what
    was said in _CHILD past _CHILD's private mode."""
    home = tmp_path / "hermes-a"
    _child(home, None)  # a rotation child: the link alone
    provider = _open(plugin, home, _PARENT, resumed=True)
    provider.sync_turn("q", "a", session_id=_CHILD, messages=_conversation(2))
    getattr(provider, flush)(_conversation(3))  # a turn no sync_turn came for
    assert _posted(server) == [(_CHILD, _questions(1, 2)), (_CHILD, _questions(2, 3))]
    _marker(tmp_path, _CHILD)  # `synapse private on`, for the id Hermes shows
    server.private_rows.add(_CHILD)
    provider.sync_turn("q", "a", session_id=_CHILD, messages=_conversation(4))
    getattr(provider, flush)(_conversation(4))
    assert len(server.posts()) == 2


def test_the_start_of_a_turn_names_the_session_before_any_upload(plugin, server, tmp_path):
    """Hermes names the session as each turn starts (``prefetch``, inline), so a flush
    in that turn goes by it before the turn's own ``sync_turn`` has run. A ``sync_turn``
    still queued for the session left is for that session, and does not bring the agent
    back to it."""
    home = tmp_path / "hermes-a"
    _child(home, None)
    provider = _open(plugin, home, _PARENT, resumed=True)
    _marker(tmp_path, _CHILD)
    server.private_rows.add(_CHILD)
    provider.prefetch("question 3", session_id=_CHILD)
    provider.on_pre_compress(_conversation(3))
    assert server.posts() == []
    provider.sync_turn("q", "a", session_id=_PARENT, messages=_conversation(2))  # late
    provider.on_session_end(_conversation(3))
    assert _posted(server) == [(_PARENT, _questions(1, 2))]


def test_a_hook_that_may_run_late_does_not_take_the_agent_back_to_a_session_it_left():
    """live.py on its own."""
    live = Live()
    assert live.named("") == "" and live.id == ""
    live.moved("s-a")
    live.moved("")  # a hook with no id to give says nothing
    assert live.named("") == "s-a"
    assert live.named("s-b") == "s-b" and live.id == "s-b"  # a move Hermes did not announce
    live.moved("s-c")
    assert live.named("s-b") == "s-b" and live.id == "s-c"  # late: for s-b, the agent stays
    assert live.named("") == "s-c"
    live.moved("s-b")  # back, and said so: s-b is not a session left any more
    assert live.named("s-d") == "s-d" and live.named("s-b") == "s-b" and live.id == "s-b"
    assert live.named("s-c") == "s-c" and live.id == "s-b"  # s-c still is


def test_a_flush_after_a_switch_goes_under_the_new_id(plugin, server, tmp_path):
    home = tmp_path / "hermes-a"
    _child(home, None)
    provider = _open(plugin, home, _PARENT)
    provider.on_session_switch(_CHILD, parent_session_id=_PARENT, reason="compression")
    provider.sync_turn("q", "a", session_id=_PARENT, messages=_conversation(1))  # late
    provider.on_session_end(_conversation(2))
    assert _posted(server) == [(_PARENT, _questions(1, 1)), (_CHILD, _questions(1, 2))]


# ---------------------------------------------------------------------------
# After `synapse private off --forget` on a carried session
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spoken_private", [0, 1], ids=["a copied turn", "a turn of its own"])
def test_a_turn_hermes_begins_after_forget_takes_no_turn_from_before_along(
    plugin, server, tmp_path, spoken_private
):
    """A branch of a private session, put back on the record. The first turn that ends
    is one Hermes began with a row in the user's place. That row opens no turn, so the
    reply lands on the turn before it, spoken off the record: neither goes. The next
    turn the user begins goes up, alone."""
    home = tmp_path / "hermes-a"
    _child(home)
    _marker(tmp_path, _PARENT)
    provider = _open(plugin, home, resumed=True)
    said = _conversation(3)  # the parent's turns, copied
    if spoken_private:
        said = _conversation(4)
        provider.sync_turn("q", "a", session_id=_CHILD, messages=said)
    _forget(server, tmp_path, _CHILD)
    said = said + _notice(1)
    provider.sync_turn("n", "noted 1", session_id=_CHILD, messages=said)
    assert server.posts() == []
    said = said + _conversation(5)[-2:]
    provider.sync_turn("q", "a", session_id=_CHILD, messages=said)
    assert _posted(server) == [(_CHILD, ["question 5"])]


@pytest.mark.parametrize(
    ("switch", "linked"),
    [(True, True), (True, False), (False, True)],
    ids=["a switch and a row", "a switch, no link in the row", "a row, no switch"],
)
def test_a_new_session_id_after_forget_is_not_looked_back_at_either(
    plugin, server, tmp_path, switch, linked
):
    """A compression that starts a new session copies the tail of the conversation
    under a new id. The id it leaves is on the record by now, so the new one is too,
    but the tail still holds turns the private parent spoke. However the provider
    learns which session the new id goes on from: the switch, or the row alone (the
    compression was another holder's, and Hermes moved this agent without a call)."""
    home = tmp_path / "hermes-a"
    _child(home)
    _marker(tmp_path, _PARENT)
    server.private_rows.add(_PARENT)
    provider = _open(plugin, home, resumed=True)
    _forget(server, tmp_path, _CHILD)
    provider.sync_turn("q", "a", session_id=_CHILD, messages=_conversation(4))
    assert _posted(server) == [(_CHILD, ["question 4"])]
    _store(home, _THIRD, source="tui", receipt=False, chat_id="")
    if linked:
        _link(home, _THIRD, _CHILD, None)
    if switch:
        provider.on_session_switch(_THIRD, parent_session_id=_CHILD, reason="compression")
    tail = [_conversation(4)[0], *_conversation(4)[3:], *_conversation(5)[-2:]]  # turns 2 to 5
    provider.sync_turn("q", "a", session_id=_THIRD, messages=tail)
    assert _posted(server)[1:] == [(_THIRD, ["question 5"])]
    assert not (tmp_path / "private" / _THIRD).exists() and _THIRD not in server.private_rows


def test_forget_after_a_switch_stands_when_the_note_of_the_carry_could_not_be_kept(
    plugin, server, tmp_path, monkeypatch
):
    """The record under ``settled/`` could not be written (a full disk), so the first
    upload under the new id reads its row and carries once more. A session made private
    by a carry is made so once: its marker and row are not written again."""
    home = tmp_path / "hermes-a"
    _child(home, None)
    _marker(tmp_path, _PARENT)
    provider = _open(plugin, home, _PARENT, resumed=True)
    module = sys.modules[type(provider).__module__]  # the provider's own copy of the core
    monkeypatch.setattr(module.carried, "settle", lambda home, session_id: None)
    provider.on_session_switch(_CHILD, parent_session_id=_PARENT, reason="compression")
    assert server.puts() == [f"/private-sessions/{_CHILD}"]
    _forget(server, tmp_path, _CHILD)
    provider.sync_turn("q", "a", session_id=_CHILD, messages=_conversation(4))
    assert _posted(server) == [(_CHILD, ["question 4"])]
    assert not (tmp_path / "private" / _CHILD).exists() and len(server.puts()) == 1
    assert not carried.settled(home, _CHILD)


# ---------------------------------------------------------------------------
# An upload for the session the agent left
# ---------------------------------------------------------------------------


def test_turns_kept_for_a_session_stay_with_it_when_another_is_resumed(plugin, server, tmp_path):
    """_CHILD waits for the server's answer about its parent. The user resumes _OTHER in
    place, and the upload of _CHILD's last turn, still on Hermes's worker, runs after
    the switch: its turns are kept for _CHILD, and do not go up as _OTHER's."""
    home = tmp_path / "hermes-a"
    _child(home)
    _store(home, _OTHER, source="tui", receipt=False, chat_id="")
    server.private_rows.add(_PARENT)
    server.private_status = 503
    provider = _open(plugin, home, resumed=True)
    provider.on_session_switch(_OTHER, parent_session_id=_CHILD, reason="resume")
    provider.sync_turn("q", "a", session_id=_CHILD, messages=_conversation(3))  # late
    assert server.posts() == []
    server.private_status = 200
    provider.sync_turn("q", "a", session_id=_OTHER, messages=_other_conversation(2))
    assert _posted(server) == [(_OTHER, _questions(1, 2, "other question"))]
    provider.sync_turn("q", "a", session_id=_CHILD, messages=_conversation(3))
    assert (tmp_path / "private" / _CHILD).is_file() and len(server.posts()) == 1


def test_a_late_upload_for_the_session_left_leaves_the_new_one_its_look_back(
    plugin, server, tmp_path
):
    home = tmp_path / "hermes-a"
    for sid in (_CHILD, _OTHER):
        _store(home, sid, source="tui", receipt=False, chat_id="")
    provider = _open(plugin, home, _CHILD, resumed=True)
    provider.on_session_switch(_OTHER, parent_session_id=_CHILD, reason="resume")
    provider.sync_turn("q", "a", session_id=_CHILD, messages=_conversation(2))  # late
    provider.sync_turn("q", "a", session_id=_OTHER, messages=_other_conversation(30))
    assert _posted(server) == [
        (_CHILD, _questions(1, 2)),
        (_OTHER, _questions(20, 30, "other question")),  # ten turns, and the one before
    ]


def test_a_session_first_met_at_an_upload_is_looked_back_at_as_a_resumed_one(
    plugin, server, tmp_path
):
    """No switch and no row said the agent is in _OTHER: its older turns were sent, or
    not, before this upload, and only the last ones are considered."""
    home = tmp_path / "hermes-a"
    _child(home, None)
    provider = _open(plugin, home, _PARENT)
    provider.sync_turn("q", "a", session_id=_OTHER, messages=_other_conversation(14))
    sent = [text for _, users in _posted(server) for text in users]
    assert sent == _questions(4, 14, "other question")  # ten turns, and the one before


# ---------------------------------------------------------------------------
# Which sessions a new id is asked about
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("named", ["", _GRAND])
def test_a_switch_goes_by_the_session_it_leaves_whatever_parent_it_names(
    plugin, server, tmp_path, named
):
    home = tmp_path / "hermes-a"
    _child(home, None)
    _marker(tmp_path, _PARENT)
    provider = _open(plugin, home, _PARENT, resumed=True)
    provider.on_session_switch(_CHILD, parent_session_id=named, reason="compression")
    assert (tmp_path / "private" / _CHILD).is_file()
    assert server.puts() == [f"/private-sessions/{_CHILD}"]


def test_a_switch_goes_by_the_session_hermes_last_named_too(plugin, server, tmp_path):
    """After a move Hermes did not announce, the id the provider is bound to is not the
    one the agent leaves."""
    home = tmp_path / "hermes-a"
    _child(home, None)
    provider = _open(plugin, home, _PARENT, resumed=True)
    provider.sync_turn("q", "a", session_id=_CHILD, messages=_conversation(2))
    _marker(tmp_path, _CHILD)
    _store(home, _THIRD, source="tui", receipt=False, chat_id="")
    provider.on_session_switch(_THIRD, reason="compression")
    assert (tmp_path / "private" / _THIRD).is_file()
    assert server.puts() == [f"/private-sessions/{_THIRD}"]


def test_a_switch_goes_by_the_session_the_provider_was_bound_to_as_well(plugin, server, tmp_path):
    """The agent was moved on from a private session without a call, and no upload has
    read the row of the session it is in now: the next id is private by the first."""
    home = tmp_path / "hermes-a"
    _child(home, None)
    for sid in (_OTHER, _THIRD):
        _store(home, sid, source="tui", receipt=False, chat_id="")
    _marker(tmp_path, _PARENT)
    provider = _open(plugin, home, _PARENT, resumed=True)
    provider.prefetch("question 3", session_id=_OTHER)
    provider.on_session_switch(_THIRD, parent_session_id=_OTHER, reason="compression")
    assert (tmp_path / "private" / _THIRD).is_file()
    assert server.puts() == [f"/private-sessions/{_THIRD}"]


def test_an_upload_asks_only_as_far_up_as_the_first_settled_session(plugin, server, tmp_path):
    """As ``initialize`` does: _PARENT was branched before _GRAND went private and was
    settled then, so its branch, met at an upload, is on the record as it is."""
    home = tmp_path / "hermes-a"
    pl._three(home)
    _store(home, _OTHER, source="tui", receipt=False, chat_id="")
    carried.settle(home, _PARENT)
    _marker(tmp_path, _GRAND)
    provider = _open(plugin, home, _OTHER)
    provider.sync_turn("q", "a", session_id=_CHILD, messages=_conversation(2))
    assert _posted(server) == [(_CHILD, _questions(1, 2))]
    assert server.asked() == [_PARENT] and server.puts() == []


def test_a_new_session_that_holds_anothers_turns_is_read_once(plugin, server, tmp_path):
    """``initialize`` read the row of the session it was given: its first upload does
    not read it again, which would start the session over with a resume's look-back."""
    home = tmp_path / "hermes-a"
    _child(home)
    provider = _open(plugin, home)
    provider.sync_turn("q", "a", session_id=_CHILD, messages=_conversation(12))
    sent = [text for _, users in _posted(server) for text in users]
    assert sent == _questions(1, 12) and server.asked() == [_PARENT]
