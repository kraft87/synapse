"""A conversation off the record stays off it, whatever Hermes does with its session next.

tests/test_hermes_private_lineage.py covers how the provider learns that a session holds
a private session's turns. These are the ways such turns could still leave afterwards:

  * a flush hook (``on_session_end``, ``on_pre_compress``) names no session, and the
    agent may be in another than the one the provider was built for or switched to
    (harnesses/hermes/live.py);
  * a late upload for the session the agent left, while turns are kept for it;
  * an agent Hermes builds later for such a session: it finds the session settled
    (harnesses/hermes/carried.py) and goes by its marker and the server's row.

Private mode covers whole sessions. ``synapse private off --forget`` on a carried
session puts all of it back on the record, with the turns it holds from the session it
came from, and what goes on from it afterwards is on the record too.

The fixtures and helpers are those of the lineage tests.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from tests import test_hermes_gateway_sessions as gs
from tests import test_hermes_private_lineage as pl
from tests import test_hermes_provider as base
from tests.helpers.client_core import use_core

use_core()
from synapse_client.harnesses.hermes import carried  # noqa: E402
from synapse_client.harnesses.hermes.live import Live  # noqa: E402

hermes_stub = gs.hermes_stub
user = gs.user
plugin = gs.plugin
server = gs.server
_conversation = gs._conversation
_say = gs._say
_users = gs._users
_store = gs._store
_open, _marker, _child, _link = pl._open, pl._marker, pl._child, pl._link

pytestmark = gs.pytestmark

_GRAND, _PARENT, _CHILD, _OTHER = pl._GRAND, pl._PARENT, pl._CHILD, pl._OTHER
_THIRD = "20261113_100000_dddddd"
_NEW = "20261113_110000_eeeeee"


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
    _say(provider, _conversation(2), _CHILD)
    getattr(provider, flush)(_conversation(3))  # a turn no sync_turn came for
    assert _posted(server) == [(_CHILD, _questions(1, 2)), (_CHILD, _questions(2, 3))]
    _marker(tmp_path, _CHILD)  # `synapse private on`, for the id Hermes shows
    server.private_rows.add(_CHILD)
    _say(provider, _conversation(4), _CHILD)
    getattr(provider, flush)(_conversation(4))
    assert len(server.posts()) == 2


@pytest.mark.parametrize("flush", ["on_pre_compress", "on_session_end"])
def test_a_flush_under_an_id_that_goes_on_from_some_other_session_sends_nothing(
    plugin, server, tmp_path, flush
):
    """The user resumes _THIRD, a session that goes on from _OTHER, and the next turn
    names it before the flush of the session left has run. That _THIRD goes on from
    some session does not make the flush's turns its own: only going on from the
    bound one does."""
    home = tmp_path / "hermes-a"
    for sid in (_PARENT, _OTHER, _THIRD):
        _store(home, sid, source="tui", receipt=False, chat_id="")
    _link(home, _THIRD, _OTHER, None)
    provider = _open(plugin, home, _PARENT, resumed=True)
    provider.prefetch("other question 1", session_id=_THIRD)
    getattr(provider, flush)(_conversation(3))
    assert server.posts() == []


def test_the_start_of_a_turn_names_the_session_before_any_upload(plugin, server, tmp_path):
    """Hermes names the session as each turn starts (``prefetch``, when it runs
    one), so a flush in that turn goes by it before the turn's own ``sync_turn`` has run. A ``sync_turn``
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
    # A switch says which ids the agent leaves, also one no hook had named.
    live.moved("s-e", ("s-b", "s-x", "", "s-e"))
    assert live.named("s-x") == "s-x" and live.named("s-b") == "s-b" and live.id == "s-e"
    live.moved("s-e", ("s-y",))  # the id it is in already: what it leaves still counts
    assert live.named("s-y") == "s-y" and live.id == "s-e"
    live.moved("", ("s-e",))  # no id to move to: nothing is left
    assert live.named("s-z") == "s-z" and live.named("s-e") == "s-e" and live.id == "s-e"
    live.moved("s-e", ("s-e",))  # a compression in place: the id it is in is not left
    assert live.named("s-f") == "s-f" and live.named("s-e") == "s-e" and live.id == "s-e"


def test_a_flush_after_a_switch_goes_under_the_new_id(plugin, server, tmp_path):
    home = tmp_path / "hermes-a"
    _child(home, None)
    provider = _open(plugin, home, _PARENT)
    provider.on_turn_start(1, "question 1", previous_message="")
    provider.on_session_switch(_CHILD, parent_session_id=_PARENT, reason="compression")
    provider.on_turn_start(2, "question 2", previous_message="question 1")
    provider.sync_turn("q", "a", session_id=_PARENT, messages=_conversation(1))  # late
    provider.on_session_end(_conversation(2))
    assert _posted(server) == [(_PARENT, _questions(1, 1)), (_CHILD, _questions(1, 2))]


@pytest.mark.parametrize("private", [True, False], ids=["private", "on the record"])
def test_the_flush_of_the_session_left_at_new_does_not_go_under_the_new_id(
    plugin, server, tmp_path, private
):
    """CLI ``/new``: the agent's id is the new one at once, while the last flush of the
    session left (``on_session_end``) and the switch wait on Hermes's worker
    (``commit_session_boundary_async``). A message typed meanwhile starts its turn
    first and names the new session (``prefetch``). The flush that follows holds the
    old conversation: as the new session's it would go up under the wrong id, and past
    the old one's private mode."""
    home = tmp_path / "hermes-a"
    for sid in (_PARENT, _NEW):
        _store(home, sid, source="cli", receipt=False, chat_id="")
    _link(home, _NEW, _PARENT, {"_reset_from": _PARENT})
    provider = _open(plugin, home, _PARENT, platform="cli")
    _say(provider, _conversation(1), _PARENT)
    if private:
        _marker(tmp_path, _PARENT)
    provider.on_turn_start(2, "question 2", previous_message="question 1")
    # Question 2 has ended, its upload still queued. /new, and the next message at once:
    provider.on_turn_start(1, "other question 1", previous_message="")
    provider.prefetch("other question 1", session_id=_NEW)
    provider.sync_turn("question 2", "a", session_id=_PARENT, messages=_conversation(2))  # late
    provider.on_session_end(_conversation(2))
    provider.on_session_switch(_NEW, parent_session_id=_PARENT, reset=True, reason="new_session")
    provider.sync_turn("other question 1", "a", session_id=_NEW, messages=_other_conversation(1))
    assert _posted(server) == [
        (_PARENT, _questions(1, 1)),
        *([] if private else [(_PARENT, _questions(1, 2))]),
        (_NEW, ["other question 1"]),
    ]


# ---------------------------------------------------------------------------
# Private mode covers whole sessions
# ---------------------------------------------------------------------------


def test_a_turn_goes_by_what_its_session_is_when_its_upload_runs(plugin, server, tmp_path):
    """An upload goes by the session as it is then, not by when the turn began.
    `synapse private on` while the model works on a turn keeps that turn in. `off
    --forget` while it works on the next puts the session back whole, with both."""
    home = tmp_path / "hermes-a"
    _store(home, _PARENT, source="tui", receipt=False, chat_id="")
    provider = _open(plugin, home, _PARENT)
    _say(provider, _conversation(1), _PARENT)
    provider.on_turn_start(2, "question 2", previous_message="question 1")
    _marker(tmp_path, _PARENT)  # `synapse private on`
    server.private_rows.add(_PARENT)
    provider.sync_turn("question 2", "a", session_id=_PARENT, messages=_conversation(2))
    provider.on_session_end(_conversation(2))
    assert _posted(server) == [(_PARENT, ["question 1"])]
    provider.on_turn_start(3, "question 3", previous_message="question 2")
    _forget(server, tmp_path, _PARENT)
    provider.sync_turn("question 3", "a", session_id=_PARENT, messages=_conversation(3))
    assert _posted(server)[1:] == [(_PARENT, _questions(1, 3))]  # 1 as the context of 2


# ---------------------------------------------------------------------------
# After `synapse private off --forget` on a carried session
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("switch", "linked"),
    [(True, True), (True, False), (False, True)],
    ids=["a switch and a row", "a switch, no link in the row", "a row, no switch"],
)
def test_a_new_session_id_after_forget_is_on_the_record_too(
    plugin, server, tmp_path, switch, linked
):
    """A compression that starts a new session copies the tail of the conversation
    under a new id. The id it leaves is back on the record, whole, so the new one is on
    it too and is looked back at as after any compression, though the parent is still
    private. However the provider learns which session the new id goes on from: the
    switch, or the row alone (the compression was another holder's, and Hermes moved
    this agent without a call)."""
    home = tmp_path / "hermes-a"
    _child(home)
    _marker(tmp_path, _PARENT)
    server.private_rows.add(_PARENT)
    provider = _open(plugin, home, resumed=True)
    _forget(server, tmp_path, _CHILD)
    _say(provider, _conversation(4), _CHILD)
    assert _posted(server) == [(_CHILD, _questions(1, 4))]
    _store(home, _THIRD, source="tui", receipt=False, chat_id="")
    if linked:
        _link(home, _THIRD, _CHILD, None)
    if switch:
        provider.on_session_switch(_THIRD, parent_session_id=_CHILD, reason="compression")
    tail = [_conversation(4)[0], *_conversation(4)[3:], *_conversation(5)[-2:]]  # turns 2 to 5
    _say(provider, tail, _THIRD)
    assert _posted(server)[1:] == [(_THIRD, _questions(2, 5))]
    assert not (tmp_path / "private" / _THIRD).exists() and _THIRD not in server.private_rows
    assert (tmp_path / "private" / _PARENT).is_file()


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
    _say(provider, _conversation(4), _CHILD)
    assert _posted(server) == [(_CHILD, _questions(1, 4))]
    assert not (tmp_path / "private" / _CHILD).exists() and len(server.puts()) == 1
    assert not carried.settled(home, _CHILD)


def test_a_conversation_stays_private_through_a_rewind_and_the_next_compression(
    plugin, server, tmp_path
):
    """A compression moves a private session to a second id, and the turn raises: the
    TUI keeps the first id as its key (tui_gateway/prompt_turn.py re-anchors it only
    with a result). `/undo` then names the first id (tui_gateway/methods_tools.py; that
    Hermes takes the old key there was read off its code, not run): the provider goes
    back to it, and each of the two ids has gone on from the other. A compression in
    the middle of question 4 moves the agent to a third id, which goes on from both and
    is private as they are. `off --forget` on the third puts that session back, whole."""
    home = tmp_path / "hermes-a"
    _store(home, _PARENT, source="tui", receipt=False, chat_id="")
    _marker(tmp_path, _PARENT)
    server.private_rows.add(_PARENT)
    agent = _open(plugin, home, _PARENT, resumed=True, message_count=6)
    asked = [*_conversation(3), {"role": "user", "content": "question 4", "message_uid": "u4"}]
    _store(home, _THIRD, source="tui", receipt=False, chat_id="")
    _link(home, _THIRD, _PARENT, None)
    agent.on_pre_compress(asked)
    agent.on_session_switch(_THIRD, parent_session_id=_PARENT, reason="compression")
    assert (tmp_path / "private" / _THIRD).is_file()
    agent.on_session_switch(_PARENT, parent_session_id="", reset=False, rewound=True)
    agent.on_turn_start(4, "question 4", previous_message="question 3")
    agent.prefetch("question 4", session_id=_THIRD)
    _store(home, _NEW, source="tui", receipt=False, chat_id="")
    _link(home, _NEW, _THIRD, None)
    agent.on_pre_compress(asked)
    agent.on_session_switch(_NEW, parent_session_id=_THIRD, reason="compression")
    assert (tmp_path / "private" / _NEW).is_file()
    agent.sync_turn("question 4", "a", session_id=_NEW, messages=_conversation(4))
    assert server.posts() == []
    _forget(server, tmp_path, _NEW)
    _say(agent, _conversation(5), _NEW)
    assert _posted(server) == [(_NEW, _questions(1, 5))]
    assert (tmp_path / "private" / _PARENT).is_file() and (tmp_path / "private" / _THIRD).is_file()


# ---------------------------------------------------------------------------
# An agent built later for a session that was carried
# ---------------------------------------------------------------------------


def _carried(plugin: Any, server: Any, home: Path, tmp_path: Path) -> Any:
    """A TUI branch of a private session, off the record from its agent's start, with
    one turn of its own spoken there."""
    _child(home)
    _marker(tmp_path, _PARENT)
    server.private_rows.add(_PARENT)
    provider = _open(plugin, home, resumed=True, message_count=6)
    _say(provider, _conversation(4), _CHILD)
    assert server.posts() == [] and carried.settled(home, _CHILD)
    return provider


@pytest.mark.parametrize("built", ["while it is private", "after forget"])
def test_an_agent_built_later_for_a_carried_session_does_not_carry_it_again(
    plugin, server, tmp_path, built
):
    """Hermes builds a session a new agent when it likes: the gateway after an idle
    hour, ``/model``, ``/undo``, ``/retry`` and a restart. The new agent finds the
    session settled and does not carry it again, though the parent is still private.
    While the session is private nothing leaves. After ``off --forget`` it is back
    whole, looked back at as any resumed session, whether the agent was built before
    or after."""
    home = tmp_path / "hermes-a"
    first = _carried(plugin, server, home, tmp_path)
    first.on_session_end(_conversation(4))
    count = 5
    if built == "after forget":
        _forget(server, tmp_path, _CHILD)
    later = _open(plugin, home, resumed=True, message_count=8)
    if built == "while it is private":
        _say(later, _conversation(5), _CHILD)
        _forget(server, tmp_path, _CHILD)
        count = 6
    assert server.posts() == []
    _say(later, _conversation(count), _CHILD)
    assert _posted(server) == [(_CHILD, _questions(1, count))]
    assert not (tmp_path / "private" / _CHILD).exists()
    assert server.puts() == [f"/private-sessions/{_CHILD}"] and server.asked() == []


@pytest.mark.parametrize("how", ["a branch", "a compression"])
def test_what_goes_on_from_a_session_put_back_is_on_the_record(plugin, server, tmp_path, how):
    """The carried session is back on the record. A branch of it made now, with an
    agent of its own, and the session a compression moves it to both hold its turns
    under a new id. That id is on the record, as the session it comes from is by now:
    it is asked about that one, once, and looked back at as any session that goes on
    from another, by this agent and by one built for it later."""
    home = tmp_path / "hermes-a"
    first = _carried(plugin, server, home, tmp_path)
    _forget(server, tmp_path, _CHILD)
    _store(home, _THIRD, source="tui", receipt=False, chat_id="")
    said = _conversation(5)
    if how == "a branch":
        _link(home, _THIRD, _CHILD, {"_branched_from": _CHILD})
        branch = _open(plugin, home, _THIRD, resumed=True, message_count=8)
        _say(branch, said, _THIRD)
    else:
        _link(home, _THIRD, _CHILD, None)
        first.on_session_switch(_THIRD, parent_session_id=_CHILD, reason="compression")
        said = [said[0], *said[5:]]  # turns 3 to 5
        _say(first, said, _THIRD)
    held = _questions(1 if how == "a branch" else 3, 5)
    assert _posted(server) == [(_THIRD, held)]
    assert carried.settled(home, _THIRD) and not carried.copied(home, _THIRD)
    assert not (tmp_path / "private" / _THIRD).exists()
    later = _open(plugin, home, _THIRD, resumed=True, message_count=10)
    _say(later, [*said, *_conversation(6)[-2:]], _THIRD)
    # A new agent sends what its look-back reaches again; the server keeps one of each.
    assert _posted(server)[1:] == [(_THIRD, [*held, "question 6"])]
    assert server.puts() == [f"/private-sessions/{_CHILD}"] and server.asked() == [_CHILD]


def test_a_branch_settled_as_on_the_record_is_looked_back_at_as_any_resumed_session(
    plugin, server, tmp_path
):
    """The other way round: the branch was made before its parent went private, and
    its record says no session it holds turns of was private. A later agent for it
    looks back as for any resumed session."""
    home = tmp_path / "hermes-a"
    _child(home)
    first = _open(plugin, home, resumed=True, message_count=6)
    first.on_session_end([])  # the server is asked, and says no
    assert carried.settled(home, _CHILD)
    _marker(tmp_path, _PARENT)
    later = _open(plugin, home, resumed=True, message_count=6)
    _say(later, _conversation(4), _CHILD)
    assert _posted(server) == [(_CHILD, _questions(1, 4))]


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
    _say(provider, _other_conversation(2), _OTHER)
    assert _posted(server) == [(_OTHER, _questions(1, 2, "other question"))]
    _say(provider, _conversation(3), _CHILD)
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
    _say(provider, _other_conversation(30), _OTHER)
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
    _say(provider, _other_conversation(14), _OTHER)
    sent = [text for _, users in _posted(server) for text in users]
    assert sent == _questions(4, 14, "other question")  # ten turns, and the one before


def test_a_session_found_at_an_upload_to_go_on_from_another_is_looked_back_at_too(
    plugin, server, tmp_path
):
    """Hermes moved the agent to _CHILD without a call, and its row says it goes on
    from _PARENT, which is on the record. The session is started here for that row,
    with a resume's look-back, not as one new here."""
    home = tmp_path / "hermes-a"
    _child(home, None)
    provider = _open(plugin, home, _PARENT, resumed=True)
    _say(provider, _conversation(14), _CHILD)
    sent = [text for _, users in _posted(server) for text in users]
    assert sent == _questions(4, 14)  # ten turns, and the one before


def test_a_late_upload_for_the_parent_a_switch_names_does_not_take_the_agent_back(
    plugin, server, tmp_path
):
    """Hermes pointed the agent at _CHILD without a call, the turn's prompt gave no
    ``prefetch``, and its ``sync_turn`` waits on the worker: no hook has named _CHILD
    when the next turn's compression moves the agent on to _THIRD. The switch names
    _CHILD as the session left. The late upload is _CHILD's, and a flush after it goes
    by _THIRD: its turns go up under that id, and no more once it is off the record."""
    home = tmp_path / "hermes-a"
    _child(home, None)
    _store(home, _THIRD, source="tui", receipt=False, chat_id="")
    _link(home, _THIRD, _CHILD, None)
    provider = _open(plugin, home, _PARENT, resumed=True)
    provider.on_session_switch(_THIRD, parent_session_id=_CHILD, reason="compression")
    provider.sync_turn("q", "a", session_id=_CHILD, messages=_conversation(2))  # late
    assert _posted(server) == [(_CHILD, _questions(1, 2))]
    provider.on_pre_compress(_conversation(3))
    assert _posted(server)[1:] == [(_THIRD, _questions(1, 3))]
    _marker(tmp_path, _THIRD)  # `synapse private on`, for the id Hermes shows
    server.private_rows.add(_THIRD)
    provider.on_session_end(_conversation(4))
    assert len(server.posts()) == 2


def test_a_switch_back_to_the_bound_id_does_not_send_its_turns_again(plugin, server, tmp_path):
    """Hermes pointed the agent at _CHILD without a call, and the user resumes _PARENT:
    the id the provider is still bound to. It is no new id here, so what was sent
    under it stays sent."""
    home = tmp_path / "hermes-a"
    _child(home, None)
    provider = _open(plugin, home, _PARENT, resumed=True)
    _say(provider, _conversation(2), _PARENT)
    provider.prefetch("question 3", session_id=_CHILD)
    provider.on_session_switch(_PARENT, reason="resume")
    _say(provider, _conversation(3), _PARENT)
    assert _posted(server) == [(_PARENT, _questions(1, 2)), (_PARENT, _questions(2, 3))]


def test_a_compression_of_the_id_the_agent_is_in_is_one_in_place(plugin, server, tmp_path):
    """Hermes pointed the agent at _CHILD without a call and then compresses _CHILD in
    place. The switch names the id the agent is in, not the one the provider is bound
    to. It is no move: what was sent under _CHILD stays sent, and the cut for recall is
    taken."""
    home = tmp_path / "hermes-a"
    _child(home, None)
    provider = _open(plugin, home, _PARENT, resumed=True)
    provider.on_turn_start(2, "question 2", previous_message="question 1")
    provider.prefetch("question 2", session_id=_CHILD)
    provider.sync_turn("question 2", "a", session_id=_CHILD, messages=_conversation(2))
    provider.on_session_switch(_CHILD, parent_session_id=_CHILD, reason="compression")
    _say(provider, _conversation(3), _CHILD)
    assert _posted(server) == [(_CHILD, _questions(1, 2)), (_CHILD, _questions(2, 3))]
    sent = base._sent(provider, server, "recall", {"query": "x"})
    assert sent["self_session"] == _CHILD and "self_since" in sent


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
    _say(provider, _conversation(2), _CHILD)
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


@pytest.mark.parametrize("undo", [False, True], ids=["as it is", "after /undo"])
def test_a_marker_on_the_id_the_tui_still_shows_covers_the_id_the_agent_is_in(
    plugin, server, tmp_path, undo
):
    """A turn compresses the session to a new id and then raises: the TUI keeps the id
    left as its key until a turn ends with a result (tui_gateway/prompt_turn.py,
    session_compression.py), and `/undo` names that key (tui_gateway/methods_tools.py;
    read off Hermes's code, not run). The agent's turns are stored, and uploaded, under
    the new id. `synapse private on` for the id the TUI shows then marks the session
    the new id goes on from, before any upload has settled the new id: it is private
    with it."""
    home = tmp_path / "hermes-a"
    _store(home, _CHILD, source="tui", receipt=False, chat_id="")
    provider = _open(plugin, home, _CHILD, resumed=True, message_count=6)
    provider.on_turn_start(4, "question 4", previous_message="question 3")
    provider.prefetch("question 4", session_id=_CHILD)
    _store(home, _THIRD, source="tui", receipt=False, chat_id="")
    _link(home, _THIRD, _CHILD, None)
    provider.on_session_switch(_THIRD, parent_session_id=_CHILD, reason="compression")
    if undo:  # the turn raised, so no sync_turn came for it
        provider.on_session_switch(_CHILD, parent_session_id="", reset=False, rewound=True)
    _marker(tmp_path, _CHILD)  # `synapse private on`, for the id the TUI shows
    server.private_rows.add(_CHILD)
    for n in (4, 5):
        provider.on_turn_start(n, f"question {n}", previous_message=f"question {n - 1}")
        provider.prefetch(f"question {n}", session_id=_THIRD)
        provider.sync_turn(f"question {n}", "a", session_id=_THIRD, messages=_conversation(n))
    assert server.posts() == []
    assert (tmp_path / "private" / _THIRD).is_file()


def test_an_upload_asks_only_as_far_up_as_the_first_settled_session(plugin, server, tmp_path):
    """As ``initialize`` does: _PARENT was branched before _GRAND went private and was
    settled then, so its branch, met at an upload, is on the record as it is."""
    home = tmp_path / "hermes-a"
    pl._three(home)
    _store(home, _OTHER, source="tui", receipt=False, chat_id="")
    carried.settle(home, _PARENT)
    _marker(tmp_path, _GRAND)
    provider = _open(plugin, home, _OTHER)
    _say(provider, _conversation(2), _CHILD)
    assert _posted(server) == [(_CHILD, _questions(1, 2))]
    assert server.asked() == [_PARENT] and server.puts() == []


def test_a_new_session_that_holds_anothers_turns_looks_back_like_any_other(
    plugin, server, tmp_path
):
    """A branch of a session on the record: its first upload looks back ten turns, the
    one before them as context, whatever Hermes says of how new it is. Its parent is
    asked about once, and the next turn goes up alone."""
    home = tmp_path / "hermes-a"
    _child(home)
    provider = _open(plugin, home)
    _say(provider, _conversation(12), _CHILD)
    assert _posted(server) == [(_CHILD, _questions(2, 12))]
    _say(provider, _conversation(13), _CHILD)
    assert _posted(server)[1:] == [(_CHILD, _questions(12, 13))]
    assert server.asked() == [_PARENT]
