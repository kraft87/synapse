"""A session that holds copies of another's turns uploads only the ones it sees begin.

Hermes copies a session's turns into an ACP fork and records no origin: the fork is
said to be new, and nothing in a turn's rows tells a copy from a new turn. So the
provider goes by what Hermes says as each turn starts: ``on_turn_start``, after the
turn-start compaction and before the turn's ``prefetch`` (agent/turn_context.py), with
the user message before this one (``previous_message``). The rule itself is the core's
(tests/test_client_core_turn_upload.py). These are Hermes's own sequences of hooks:

  * an ACP fork: its agent is built before its row and its copied history are written,
    and the row names no origin (acp_adapter/session.py ``fork_session``);
  * the agent built for the fork later (``_restore``), or for an id it went on to, which
    goes by what the first one kept of it (carried.py ``note_copied``), and three
    orders in which nothing was kept and a copy still goes up (``test_still_open_...``);
  * what such an agent is shown: a flush as the turn starts or in the middle of it, a
    turn Hermes began itself, a message with no text;
  * a compression of the head alone (``/compress here N``), whose flush hooks are handed
    the rows to be summarised and not the turns kept;
  * a message Hermes folded into one that got no reply
    (agent/agent_runtime_helpers.py ``_merge_consecutive_users``).

Private mode itself covers whole sessions and needs no turn start: an upload goes by
the session it names (tests/test_hermes_private_lineage.py, test_hermes_private_moves.py).

The fixtures and helpers are those of tests/test_hermes_private_moves.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from tests import test_hermes_gateway_sessions as gs
from tests import test_hermes_private_lineage as pl
from tests import test_hermes_private_moves as pm

hermes_stub = gs.hermes_stub
user = gs.user
plugin = gs.plugin
server = gs.server
_conversation = gs._conversation
_say = gs._say
_store = gs._store
_open, _marker, _child, _link = pl._open, pl._marker, pl._child, pl._link
_posted, _questions = pm._posted, pm._questions

pytestmark = gs.pytestmark

_PARENT, _CHILD = pl._PARENT, pl._CHILD
_ACP = "7d1f0c3e-58b1-4b64-8a55-0b9e7a1c2f00"  # an ACP session id is a uuid4
_FORK = "0f6b2a54-2f0e-4c53-9d2b-6f4d2d7c9a11"
_NEXT = "5b0c9d7e-1a2f-4e3b-8c4d-9e8f7a6b5c4d"
_LATER = "c2a4e6f8-0b1d-4f3a-9c5e-7d9b1f3a5c7e"


#: A message that is a picture alone: its row has no text, and opens no turn.
_PICTURE = {
    "role": "user",
    "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}],
    "message_uid": "p1",
}


def _tools(n: int) -> list[dict[str, Any]]:
    """A tool call and its result: a turn still running."""
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c9",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
            "message_uid": f"nc{n}",
        },
        {
            "role": "tool",
            "content": "ok",
            "tool_call_id": "c9",
            "name": "terminal",
            "tool_name": "terminal",
            "message_uid": f"nt{n}",
        },
    ]


# ---------------------------------------------------------------------------
# A session Hermes said was new
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "how", ["said", "counted", "counted at a flush", "flushed first", "rotated first"]
)
def test_an_acp_fork_uploads_none_of_the_turns_copied_into_it(plugin, server, tmp_path, how):
    """ACP ``session/fork``: a new id, the agent built for it first (no row yet, so not
    resumed), then the row, with no parent, and the original's history copied in. The
    original is off the record, and nothing says the fork holds its turns. What does:
    the first turn start names a message before it ("said"); or it names none (a
    picture, a scaffolded prompt) and the first upload holds more turns than were begun
    ("counted"). Before that turn start a flush sends nothing, and a rotation leaves the
    question open for the new id. Once the fork is known to hold copies, they stay out
    under any id it goes on with."""
    home = tmp_path / "hermes-a"
    _store(home, _ACP, source="acp", receipt=False, chat_id="")
    _marker(tmp_path, _ACP)  # `synapse private on`, in the original
    server.private_rows.add(_ACP)
    fork = _open(plugin, home, _FORK, platform="acp", message_count=0)
    _store(home, _FORK, source="acp", receipt=False, chat_id="")
    sid = _FORK
    asked = _conversation(4)[:-1]  # three copied turns, and the fork's first message
    if how in ("flushed first", "rotated first"):
        fork.on_pre_compress(asked)  # the turn-start compaction: before on_turn_start
    if how == "rotated first":
        sid = _NEXT
        _store(home, sid, source="acp", receipt=False, chat_id="")
        _link(home, sid, _FORK, None)
        fork.on_session_switch(sid, parent_session_id=_FORK, reason="compression")
    previous = "" if how.startswith("counted") else "question 3"
    fork.on_turn_start(4, "question 4", previous_message=previous)
    fork.prefetch("question 4", session_id=sid)
    if how == "counted at a flush":
        fork.on_session_end(asked)
    assert server.posts() == []
    fork.sync_turn("question 4", "a", session_id=sid, messages=_conversation(4))
    assert _posted(server) == [(sid, ["question 4"])]
    _say(fork, _conversation(5), sid)
    assert _posted(server)[1:] == [(sid, _questions(4, 5))]
    # A later compression moves the fork to a new id, with the tail of the conversation.
    _store(home, _LATER, source="acp", receipt=False, chat_id="")
    _link(home, _LATER, sid, None)
    fork.on_session_switch(_LATER, parent_session_id=sid, reason="compression")
    tail = [_conversation(6)[0], *_conversation(6)[5:]]  # turns 3 to 6
    _say(fork, tail, _LATER)
    assert _posted(server)[2:] == [(_LATER, ["question 6"])]
    assert [one.name for one in (tmp_path / "private").iterdir()] == [_ACP]
    assert server.puts() == []


def test_a_fork_keeps_its_copies_out_under_an_id_only_the_switch_says_it_goes_on_to(
    plugin, server, tmp_path
):
    """A compression moves the fork to a new id, and the new id's row does not say yet
    what it goes on from: the switch does. The fork was never made private, so no
    marker or server row says anything of it: the agent that found the copies takes
    that along to the new id (for an agent built later, see the record below)."""
    home = tmp_path / "hermes-a"
    _store(home, _ACP, source="acp", receipt=False, chat_id="")
    _marker(tmp_path, _ACP)
    server.private_rows.add(_ACP)
    fork = _open(plugin, home, _FORK, platform="acp", message_count=0)
    _store(home, _FORK, source="acp", receipt=False, chat_id="")
    fork.on_turn_start(4, "question 4", previous_message="question 3")
    fork.prefetch("question 4", session_id=_FORK)
    fork.sync_turn("question 4", "a", session_id=_FORK, messages=_conversation(4))
    assert _posted(server) == [(_FORK, ["question 4"])]
    _store(home, _LATER, source="acp", receipt=False, chat_id="")  # no parent named
    fork.on_session_switch(_LATER, parent_session_id=_FORK, reason="compression")
    _say(fork, _conversation(5), _LATER)
    assert _posted(server)[1:] == [(_LATER, ["question 5"])]


def test_an_id_found_at_an_upload_to_go_on_from_a_fork_keeps_the_copies_out_with_no_record(
    plugin, server, tmp_path
):
    """The agent found the fork's copies and could not record them. Hermes then points
    it at a new id without a call (a compression made while another holder of the
    session had it): the id's row says it goes on from the fork, and what this agent
    knows of the fork goes along."""
    home = tmp_path / "hermes-a"
    _store(home, _ACP, source="acp", receipt=False, chat_id="")
    (home / "synapse").mkdir(parents=True, exist_ok=True)
    (home / "synapse" / "copied").write_text("")  # no record can be written there
    fork = _open(plugin, home, _FORK, platform="acp", message_count=0)
    _store(home, _FORK, source="acp", receipt=False, chat_id="")
    _say(fork, _conversation(4), _FORK)
    assert _posted(server) == [(_FORK, ["question 4"])]
    assert not pl.carried.copied(home, _FORK)
    _store(home, _NEXT, source="acp", receipt=False, chat_id="")
    _link(home, _NEXT, _FORK, None)
    _say(fork, _conversation(5), _NEXT)
    assert _posted(server)[1:] == [(_NEXT, ["question 5"])]


def test_a_fork_marked_private_and_put_back_uploads_from_its_next_turn(plugin, server, tmp_path):
    """A session put back on the record (`synapse private off --forget`) is back whole.
    A fork is not: nothing was noted of the turns that ended in it while it was
    private, and nothing tells them from the copies it began with. Put back, it uploads
    the turns that begin from then on."""
    home = tmp_path / "hermes-a"
    _store(home, _ACP, source="acp", receipt=False, chat_id="")
    fork = _open(plugin, home, _FORK, platform="acp", message_count=0)
    _store(home, _FORK, source="acp", receipt=False, chat_id="")
    _marker(tmp_path, _FORK)  # `synapse private on`, in the fork
    server.private_rows.add(_FORK)
    _say(fork, _conversation(4), _FORK)
    _say(fork, _conversation(5), _FORK)
    assert server.posts() == []
    pm._forget(server, tmp_path, _FORK)
    _say(fork, _conversation(6), _FORK)
    _say(fork, _conversation(7), _FORK)
    assert _posted(server) == [(_FORK, ["question 6"]), (_FORK, _questions(6, 7))]


def test_an_acp_forks_first_message_with_no_text_takes_no_copied_turn_along(
    plugin, server, tmp_path
):
    """The fork's first message is a picture alone. No row opens a turn for it, so the
    reply lands on the one turn copied in, and the fork holds as many turns as were
    begun: the count says nothing. The message Hermes names before this one does."""
    home = tmp_path / "hermes-a"
    _store(home, _ACP, source="acp", receipt=False, chat_id="")
    _marker(tmp_path, _ACP)
    server.private_rows.add(_ACP)
    fork = _open(plugin, home, _FORK, platform="acp", message_count=0)
    _store(home, _FORK, source="acp", receipt=False, chat_id="")
    said = [*_conversation(1), _PICTURE, {"role": "assistant", "content": "A cat."}]
    fork.on_turn_start(2, "", previous_message="question 1")
    fork.sync_turn("[1 image]", "A cat.", session_id=_FORK, messages=said)
    assert server.posts() == []
    _say(fork, said + _conversation(2)[-2:], _FORK)
    assert _posted(server) == [(_FORK, ["question 2"])]


def test_a_new_session_moved_without_a_call_keeps_its_turn_starts(plugin, server, tmp_path):
    """Hermes says a session is new, then points the agent at another id without a call
    (the turn's prefetch names it), and the first turn is stopped: no upload comes for
    it. The second turn's upload names the new id, whose row says it goes on from the
    first. Both turns began in this agent, and their starts were counted for the
    session the provider is bound to, which the new id goes on from: neither turn is a
    copy."""
    home = tmp_path / "hermes-a"
    provider = _open(plugin, home, _FORK, platform="acp", message_count=0)
    _store(home, _FORK, source="acp", receipt=False, chat_id="")
    provider.on_turn_start(1, "question 1", previous_message="")
    _store(home, _NEXT, source="acp", receipt=False, chat_id="")
    _link(home, _NEXT, _FORK, None)
    provider.prefetch("question 1", session_id=_NEXT)  # and the turn is stopped
    provider.on_turn_start(2, "question 2", previous_message="question 1")
    provider.prefetch("question 2", session_id=_NEXT)
    provider.sync_turn("question 2", "a", session_id=_NEXT, messages=_conversation(2))
    assert _posted(server) == [(_NEXT, _questions(1, 2))]
    assert not pl.carried.copied(home, _NEXT) and not pl.carried.copied(home, _FORK)


def test_a_turn_that_ends_in_a_new_session_with_no_start_seen_is_not_uploaded(
    plugin, server, tmp_path
):
    """Hermes leaves out ``on_turn_start`` and the turn's upload together (a vetoed
    prompt, a subagent, a caller of another profile: agent/memory_manager.py). Should
    an upload ever run without the turn start, in a session Hermes said was new,
    nothing says the turns it holds began there. None goes up: in an ACP fork they are
    the copies of a private original."""
    home = tmp_path / "hermes-a"
    _store(home, _ACP, source="acp", receipt=False, chat_id="")
    _marker(tmp_path, _ACP)
    server.private_rows.add(_ACP)
    fork = _open(plugin, home, _FORK, platform="acp", message_count=0)
    _store(home, _FORK, source="acp", receipt=False, chat_id="")
    fork.sync_turn("question 4", "a", session_id=_FORK, messages=_conversation(4))
    assert server.posts() == []
    _say(fork, _conversation(5), _FORK)
    assert _posted(server) == [(_FORK, ["question 5"])]
    assert server.puts() == []


def _fork(plugin: Any, server: Any, tmp_path: Path) -> tuple[Any, Path]:
    """An ACP fork of a private session with three turns: its agent, then its row."""
    home = tmp_path / "hermes-a"
    _store(home, _ACP, source="acp", receipt=False, chat_id="")
    _marker(tmp_path, _ACP)
    server.private_rows.add(_ACP)
    fork = _open(plugin, home, _FORK, platform="acp", message_count=0)
    _store(home, _FORK, source="acp", receipt=False, chat_id="")
    return fork, home


@pytest.mark.parametrize("found", ["at a turn start", "at an upload"])
def test_an_agent_built_later_for_an_acp_fork_keeps_the_copies_out(plugin, server, tmp_path, found):
    """The ACP adapter is restarted, and restores the fork from the store with a new
    agent, which is told it resumes a session (acp_adapter/session.py ``_restore``).
    The fork's row names no session, and what the first agent found was in its memory.
    It kept that on disk (carried.py ``note_copied``): the new agent looks back at
    none of the fork's turns, at a flush or at the end of a turn."""
    fork, home = _fork(plugin, server, tmp_path)
    if found == "at a turn start":
        _say(fork, _conversation(4), _FORK)
    else:
        fork.on_turn_start(4, "question 4", previous_message="")
        fork.sync_turn("question 4", "a", session_id=_FORK, messages=_conversation(4))
    assert _posted(server) == [(_FORK, ["question 4"])]
    later = _open(plugin, home, _FORK, platform="acp", resumed=True, message_count=8)
    later.on_session_end(_conversation(4))
    assert len(server.posts()) == 1
    _say(later, _conversation(5), _FORK)
    assert _posted(server)[1:] == [(_FORK, ["question 5"])]
    assert server.puts() == [] and server.asked() == []


@pytest.mark.parametrize(
    "linked", [True, False], ids=["its row names the fork", "the switch alone says"]
)
@pytest.mark.parametrize("moved", ["after its first turn", "before it"])
def test_an_agent_built_later_for_the_id_a_fork_went_on_to_keeps_them_out_too(
    plugin, server, tmp_path, moved, linked
):
    """A compression moves the fork to a new id: after its first turn, or before it
    (the turn-start compaction), when the copies are found under the new id and the
    question whether the fork is private closes after that, with "no". A new agent is
    built for the new id, whose row names the fork or, written without it, nothing.
    Whichever way, the copies stay out."""
    fork, home = _fork(plugin, server, tmp_path)
    n = 4
    if moved == "after its first turn":
        _say(fork, _conversation(4), _FORK)
        n = 5
    else:
        fork.on_pre_compress(_conversation(4)[:-1])
    _store(home, _LATER, source="acp", receipt=False, chat_id="")
    if linked:
        _link(home, _LATER, _FORK, None)
    fork.on_session_switch(_LATER, parent_session_id=_FORK, reason="compression")
    _say(fork, _conversation(n), _LATER)
    sent = len(server.posts())
    assert _posted(server)[-1] == (_LATER, [f"question {n}"])
    later = _open(plugin, home, _LATER, platform="acp", resumed=True, message_count=2 * n)
    _say(later, _conversation(n + 1), _LATER)
    assert _posted(server)[sent:] == [(_LATER, [f"question {n + 1}"])]
    assert server.puts() == []


def test_an_agent_built_for_an_id_no_agent_was_moved_to_reads_the_forks_record(
    plugin, server, tmp_path
):
    """The id's row says it goes on from the fork, and no agent was moved to it (a
    compression made while another holder had the session): the agent built for it
    reads the fork's record, keeps the copies out, and notes the id as one that holds
    them, for the sessions that go on from it."""
    fork, home = _fork(plugin, server, tmp_path)
    _say(fork, _conversation(4), _FORK)
    sent = len(server.posts())
    _store(home, _LATER, source="acp", receipt=False, chat_id="")
    _link(home, _LATER, _FORK, None)
    later = _open(plugin, home, _LATER, platform="acp", resumed=True, message_count=8)
    _say(later, _conversation(5), _LATER)
    assert _posted(server)[sent:] == [(_LATER, ["question 5"])]
    assert pl.carried.copied(home, _LATER)


@pytest.mark.parametrize("noted", [True, False], ids=["noted", "no note could be kept"])
def test_a_switch_from_a_fork_this_agent_knows_nothing_of_goes_by_the_forks_record(
    plugin, server, tmp_path, monkeypatch, noted
):
    """The switch names the session the new id goes on from, and this agent has seen
    nothing of it: Hermes pointed it at the fork without a call, and the compaction
    at the start of the next turn moves it on before a hook has named the fork. What
    another agent recorded of the fork is all that says the new id holds copies. The
    new id is noted as one that does; where that note cannot be kept, what the switch
    read still holds for this agent."""
    fork, home = _fork(plugin, server, tmp_path)
    _say(fork, _conversation(4), _FORK)
    sent = len(server.posts())
    _store(home, _NEXT, source="tui", receipt=False, chat_id="")
    # Resumed: a session said to be new would leave the doubt to its first turn start.
    other = _open(plugin, home, _NEXT, resumed=True, message_count=4)
    if not noted:
        module = sys.modules[type(other).__module__]  # the provider's own copy
        monkeypatch.setattr(module.carried, "note_copied", lambda home, session_id: None)
    _store(home, _LATER, source="acp", receipt=False, chat_id="")  # no parent named
    other.on_session_switch(_LATER, parent_session_id=_FORK, reason="compression")
    _say(other, _conversation(5), _LATER)
    assert _posted(server)[sent:] == [(_LATER, ["question 5"])]
    assert pl.carried.copied(home, _LATER) is noted


def test_a_fork_resumed_in_place_by_a_later_agent_keeps_the_copies_out(plugin, server, tmp_path):
    """``/resume`` opens the fork in an agent that was built for another session: no
    ``initialize`` names the fork, and the switch says a session is resumed, which is
    looked back at. Its first upload reads what was kept of it."""
    fork, home = _fork(plugin, server, tmp_path)
    _say(fork, _conversation(4), _FORK)
    _store(home, _NEXT, source="tui", receipt=False, chat_id="")
    other = _open(plugin, home, _NEXT)
    other.on_session_switch(_FORK, parent_session_id=_NEXT, reason="resume")
    _say(other, _conversation(5), _FORK)
    assert _posted(server) == [(_FORK, ["question 4"]), (_FORK, ["question 5"])]


def test_an_id_a_fork_went_on_to_resumed_in_place_keeps_the_copies_out(plugin, server, tmp_path):
    """The same for an id whose row says it goes on from the fork, which no agent was
    moved to: its first upload reads the fork's record, and notes the id."""
    fork, home = _fork(plugin, server, tmp_path)
    _say(fork, _conversation(4), _FORK)
    _store(home, _LATER, source="acp", receipt=False, chat_id="")
    _link(home, _LATER, _FORK, None)
    _store(home, _NEXT, source="tui", receipt=False, chat_id="")
    other = _open(plugin, home, _NEXT)
    other.on_session_switch(_LATER, parent_session_id=_NEXT, reason="resume")
    _say(other, _conversation(5), _LATER)
    assert _posted(server) == [(_FORK, ["question 4"]), (_LATER, ["question 5"])]
    assert pl.carried.copied(home, _LATER)


def test_a_session_served_by_an_agent_per_turn_uploads_each_turn_as_before(
    plugin, server, tmp_path
):
    """A gateway with ``gateway.managed_workers`` on builds an agent for every turn and
    tells each the session is new. From its second turn such a session is taken for one
    that holds copies, and that is kept on disk now. The agents after it go by the
    record where they went by the turn start: each turn still goes up as it ends,
    alone."""
    home = tmp_path / "hermes-a"
    _store(home, _PARENT, source="tui", receipt=False, chat_id="")
    for n in range(1, 5):
        worker = _open(plugin, home, _PARENT, platform="tui", message_count=0)
        _say(worker, _conversation(n), _PARENT)
        assert pl.carried.copied(home, _PARENT) is (n > 1)
    assert _posted(server) == [(_PARENT, [f"question {n}"]) for n in range(1, 5)]
    assert server.puts() == []


def _unanswered(plugin: Any, home: Path) -> tuple[Any, list[dict[str, Any]]]:
    """A chat whose first message got no reply, opened again. The TUI stores a message
    as it is sent (tui_gateway/session_workdir.py), so one whose turn never ran (the
    agent could not be built, the app was quit) stays in the store. The agent built
    for the next message is told the session is new: rows, and no reply among them
    (agent/agent_init_memory_provider.py ``_stored_transcript_kwargs``). Its first
    turn start names the message before, and Hermes folds the two into one row."""
    _store(home, _PARENT, source="tui", receipt=False, chat_id="")
    provider = _open(plugin, home, _PARENT, resumed=False, message_count=2)
    said = [_conversation(0)[0], *_folded(1)]
    provider.on_turn_start(1, "question 2", previous_message="question 1")
    provider.sync_turn("question 2", "answer 2", session_id=_PARENT, messages=said)
    return provider, said


def test_a_chat_whose_first_message_got_no_reply_is_not_recorded_as_holding_copies(
    plugin, server, tmp_path
):
    """Its first turn start looks like a fork's, and this agent takes it for one: the
    folded first turn stays out. But Hermes held rows for the session when it built
    the agent, and a fork's are written after its agent: nothing is kept of it on
    disk. The agent built next is told it resumes a session and looks back, at the
    first turn and at one whose upload had not run."""
    home = tmp_path / "hermes-a"
    first, said = _unanswered(plugin, home)
    assert server.posts() == [] and not (home / "synapse" / "copied").exists()
    said = [*said, *_conversation(3)[-2:]]
    _say(first, said, _PARENT)
    assert _posted(server) == [(_PARENT, ["question 3"])]
    said = [*said, *_conversation(4)[-2:]]
    first.on_turn_start(3, "question 4", previous_message="question 3")  # no upload runs
    later = _open(plugin, home, _PARENT, resumed=True, message_count=7)
    _say(later, [*said, *_conversation(5)[-2:]], _PARENT)
    both = ["question 1\n\nquestion 2", ""]  # the folded row, and the row that marks it
    assert _posted(server)[1:] == [(_PARENT, [*both, *_questions(3, 5)])]
    assert server.puts() == [] and not (home / "synapse" / "copied").exists()


def test_a_session_that_goes_on_from_such_a_chat_is_looked_back_at(plugin, server, tmp_path):
    """A compression moves the chat to a new id. Nothing was recorded of the chat, so
    the new id is not recorded as one that holds copies, and the agent built for it
    later looks back at its turns."""
    home = tmp_path / "hermes-a"
    provider, said = _unanswered(plugin, home)
    said = [*said, *_conversation(3)[-2:]]
    _say(provider, said, _PARENT)
    third = pm._THIRD
    _store(home, third, source="tui", receipt=False, chat_id="")
    _link(home, third, _PARENT, None)
    provider.on_pre_compress(said)
    provider.on_session_switch(third, parent_session_id=_PARENT, reason="compression")
    said = [*said, *_conversation(4)[-2:]]
    _say(provider, said, third)
    assert pl.carried.settled(home, third) and not pl.carried.copied(home, third)
    sent = len(server.posts())
    provider.on_turn_start(4, "question 5", previous_message="question 4")  # no upload runs
    later = _open(plugin, home, third, resumed=True, message_count=9)
    _say(later, [*said, *_conversation(5)[-2:], *_conversation(6)[-2:]], third)
    assert _posted(server)[sent:] == [(third, ["question 1\n\nquestion 2", "", *_questions(3, 6)])]
    assert server.puts() == [] and not (tmp_path / "private").exists()


def test_an_agent_told_no_message_count_keeps_what_it_finds(plugin, server, tmp_path):
    """Hermes tells every agent it builds how many messages its store holds. Should
    an agent be built without that keyword, nothing says the session's messages were
    there before it: what its uploader finds is kept, as for a count of 0."""
    home = tmp_path / "hermes-a"
    _store(home, _ACP, source="acp", receipt=False, chat_id="")
    _marker(tmp_path, _ACP)
    server.private_rows.add(_ACP)
    keywords = gs._in_process(platform="acp")
    del keywords["message_count"]
    fork = pl._new(plugin)
    fork.initialize(_FORK, hermes_home=str(home), **keywords)
    _store(home, _FORK, source="acp", receipt=False, chat_id="")
    _say(fork, _conversation(4), _FORK)
    assert _posted(server) == [(_FORK, ["question 4"])] and pl.carried.copied(home, _FORK)


# ---------------------------------------------------------------------------
# Still open (HANDOFF.md): copies no agent saw arrive
#
# The three tests below state what happens today, not what is wanted: a copy of a
# private session's turn goes up. Each is listed under "Still open", and each changes
# when its gap is closed.
# ---------------------------------------------------------------------------


def test_still_open_an_agent_built_again_for_a_fork_before_its_first_turn_sends_the_copies(
    plugin, server, tmp_path
):
    """The agent that saw the fork arrive (built for an empty session, the copies
    written after) never runs a turn: the user picks another model, and the ACP adapter
    builds a new agent for the same session (acp_adapter/server.py), or the adapter is
    restarted. The new agent is told it resumes a session with replies in it, as for
    any chat opened again, and looks back at the copied turns."""
    _, home = _fork(plugin, server, tmp_path)
    second = _open(plugin, home, _FORK, platform="acp", resumed=True, message_count=6)
    _say(second, _conversation(4), _FORK)
    assert _posted(server) == [(_FORK, _questions(1, 4))]  # 1 to 3 were private
    assert not pl.carried.copied(home, _FORK)


def _unanswered_copy(server: Any, tmp_path: Path, origin: str, source: str) -> Path:
    """A private session whose one message, "question 1", got no reply."""
    home = tmp_path / "hermes-a"
    _store(home, origin, source=source, receipt=False, chat_id="")
    _marker(tmp_path, origin)
    server.private_rows.add(origin)
    return home


@pytest.mark.parametrize("copy", ["an ACP fork built twice", "a session seeded with it"])
def test_still_open_a_copy_with_no_reply_is_sent_by_the_agent_after_the_one_that_found_it(
    plugin, server, tmp_path, copy
):
    """The copy is one message that got no reply. An agent built once the copy is in
    the store is told the session holds a message and is new: what an ordinary chat
    whose first message got no reply looks like (``_unanswered``). That is the second
    agent of an ACP fork (a model switch, or a restart, before the fork's first
    prompt), and the first agent of a session a client seeded with messages
    (tui_gateway/methods_session.py). It takes the session for one that holds a copy,
    and keeps the folded turn out, but records nothing (carried.py ``noter``): the
    agent built next looks back, and the private message goes up."""
    if copy == "an ACP fork built twice":
        sid, platform = _FORK, "acp"
        home = _unanswered_copy(server, tmp_path, _ACP, "acp")
        _open(plugin, home, sid, platform=platform, message_count=0)
        _store(home, sid, source="acp", receipt=False, chat_id="")
    else:
        sid, platform = _CHILD, "tui"
        home = _unanswered_copy(server, tmp_path, _PARENT, "tui")
        _store(home, sid, source="tui", receipt=False, chat_id="")  # names no parent
    found = _open(plugin, home, sid, platform=platform, resumed=False, message_count=1)
    said = [_conversation(0)[0], *_folded(1)]
    found.on_turn_start(1, "question 2", previous_message="question 1")
    found.sync_turn("question 2", "answer 2", session_id=sid, messages=said)
    assert server.posts() == [] and not pl.carried.copied(home, sid)
    later = _open(plugin, home, sid, platform=platform, resumed=True, message_count=2)
    _say(later, [*said, *_conversation(3)[-2:]], sid)
    assert _posted(server) == [(sid, ["question 1\n\nquestion 2", "", "question 3"])]
    assert server.puts() == []


# ---------------------------------------------------------------------------
# A session known to hold copies, in an agent built for it later
# ---------------------------------------------------------------------------


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


def _restored(plugin: Any, server: Any, tmp_path: Path) -> Any:
    """An ACP fork of a private session, in the agent built for it after a restart
    (acp_adapter/session.py ``_restore``). It holds three turns, copies of the
    original's, and the record of the agent that found them says so."""
    home = tmp_path / "hermes-a"
    for sid in (_ACP, _FORK):
        _store(home, sid, source="acp", receipt=False, chat_id="")
    _marker(tmp_path, _ACP)
    server.private_rows.add(_ACP)
    pl.carried.note_copied(home, _FORK)
    return _open(plugin, home, _FORK, platform="acp", resumed=True, message_count=6)


@pytest.mark.parametrize("flush", ["on_pre_compress", "on_session_end"])
def test_the_first_turn_goes_up_though_a_flush_saw_it_before_it_began(
    plugin, server, tmp_path, flush
):
    """Hermes compresses at the start of a turn, before ``on_turn_start``: the flush
    finds the new message with no turn said to have begun. It is neither sent nor
    written off, there or at a flush in the middle of the turn: the turn's own upload
    settles it."""
    provider = _restored(plugin, server, tmp_path)
    asked = _conversation(4)[:-1]
    getattr(provider, flush)(asked)
    provider.on_turn_start(4, "question 4", previous_message="question 3")
    getattr(provider, flush)(asked + _tools(4))
    assert server.posts() == []
    provider.sync_turn("question 4", "a", session_id=_FORK, messages=_conversation(4))
    assert _posted(server) == [(_FORK, ["question 4"])]


@pytest.mark.parametrize("flush", [None, "on_pre_compress", "on_session_end"])
def test_a_turn_hermes_begins_takes_no_copied_turn_along(plugin, server, tmp_path, flush):
    """Hermes begins a turn itself: a background-process notice in the user's place. It
    starts as any turn does. The notice opens no turn, so the tool calls and the reply
    land on the last turn the conversation holds, a copy of the private original's:
    with or without a flush in the middle, it does not go, as a turn or as the next
    one's context."""
    provider = _restored(plugin, server, tmp_path)
    notice, reply = _notice(1)
    mid = [*_conversation(3), notice, *_tools(1)]
    provider.on_turn_start(4, notice["content"], previous_message="question 3")
    if flush:
        getattr(provider, flush)(mid)
    provider.sync_turn(
        notice["content"], reply["content"], session_id=_FORK, messages=[*mid, reply]
    )
    assert server.posts() == []
    _say(provider, [*mid, reply, *_conversation(4)[-2:]], _FORK)
    assert _posted(server) == [(_FORK, ["question 4"])]


def test_a_turn_hermes_begins_that_fails_takes_no_copied_turn_along(plugin, server, tmp_path):
    """The turn Hermes began fails before the model says anything. Hermes closes it
    with a failed-turn boundary, which is never sent, and its finalizer mirrors a turn
    whether or not it failed, with the error as the reply (agent/turn_finalizer.py
    ``_mirror_turn_to_memory``). The records then end with the last copied turn, whole,
    and the notice: a turn began and one has ended, but not that one."""
    provider = _restored(plugin, server, tmp_path)
    notice, _ = _notice(1)
    boundary = {
        "role": "assistant",
        "content": "Your request was not processed. Send it again if you still want me to "
        "carry it out.",
        "display_kind": "failed_turn",
    }
    said = [*_conversation(3), notice, boundary]
    provider.on_turn_start(4, notice["content"], previous_message="question 3")
    provider.sync_turn(notice["content"], "HTTP 500", session_id=_FORK, messages=said)
    assert server.posts() == []
    _say(provider, said + _conversation(4)[-2:], _FORK)
    assert _posted(server) == [(_FORK, ["question 4"])]


@pytest.mark.parametrize("flush", [None, "on_pre_compress", "on_session_end"])
def test_a_copied_turn_cut_short_is_not_finished_by_a_reply_hermes_began(
    plugin, server, tmp_path, flush
):
    """Question 3 was asked in the private original and its reply interrupted: the turn
    was copied open. The reply to a notice closes it. That reply began later than the
    turn did, and takes none of it along: also when a flush saw the open turn first
    and left it for the upload that ends it."""
    provider = _restored(plugin, server, tmp_path)
    cut = _conversation(2) + _conversation(3)[-2:-1] + _tools(3)
    if flush:
        getattr(provider, flush)(cut)
    notice, reply = _notice(1)
    provider.on_turn_start(4, notice["content"], previous_message="question 3")
    provider.sync_turn(
        notice["content"], reply["content"], session_id=_FORK, messages=[*cut, notice, reply]
    )
    assert server.posts() == []


@pytest.mark.parametrize("answered", [True, False], ids=["answered", "cut short"])
def test_a_message_with_no_text_takes_no_copied_turn_along(plugin, server, tmp_path, answered):
    """The first message this agent sees is a picture alone. Its row has no text and
    opens no turn, so the reply lands on the last copied turn: also on one that was cut
    short, whose own reply the picture's would otherwise read as. The row stays in the
    records to show it (harnesses/hermes/messages.py). Hermes still mirrors the turn,
    with "[1 image]" for its text (run_agent.py ``_sync_external_memory_for_turn``)."""
    provider = _restored(plugin, server, tmp_path)
    copied = _conversation(3) if answered else _conversation(3)[:-1]
    said = [*copied, _PICTURE, {"role": "assistant", "content": "A cat."}]
    provider.on_turn_start(4, "", previous_message="question 3")
    provider.sync_turn("[1 image]", "A cat.", session_id=_FORK, messages=said)
    assert server.posts() == []
    provider.on_turn_start(5, "question 4", previous_message="")
    provider.sync_turn("question 4", "a", session_id=_FORK, messages=said + _conversation(4)[-2:])
    assert _posted(server) == [(_FORK, ["question 4"])]


# ---------------------------------------------------------------------------
# A flush that shows part of the conversation
# ---------------------------------------------------------------------------


#: The row a compression leaves where the turns it summarised were.
_SUMMARY = {
    "role": "assistant",
    "content": "[CONTEXT COMPACTION] Earlier turns were summarised.",
    "_compressed_summary": True,
    "message_uid": "sum1",
}


@pytest.mark.parametrize("where", ["the agent stays", "the agent is let go"])
def test_a_compression_of_the_head_alone_lets_none_of_the_kept_turns_out(
    plugin, server, tmp_path, where
):
    """``/compress here 2`` before any turn has ended in this agent. Hermes hands the
    hooks the rows it is about to summarise and no others
    (agent/conversation_compression_manual.py). In process both flush hooks get that
    head, and the session keeps its id and its agent. Where the agent is let go after
    the compression (the gateway does that), ``on_pre_compress`` gets the head and the
    flush as the agent goes gets its whole list: the only turn that agent saw begin was
    interrupted. The two exchanges kept as they were are still copies."""
    home = tmp_path / "hermes-a"
    provider = _restored(plugin, server, tmp_path)
    whole = _conversation(3)
    if where == "the agent is let go":
        provider.on_turn_start(4, "question 4", previous_message="question 3")  # interrupted
    provider.on_pre_compress(whole[:-4])
    provider.on_session_end(whole[:-4] if where == "the agent stays" else whole)
    assert server.posts() == []
    if where == "the agent stays":
        provider.on_session_switch(
            _FORK, parent_session_id=_FORK, reset=False, reason="compression"
        )
    else:
        provider = _open(plugin, home, _FORK, platform="acp", resumed=True, message_count=6)
    _say(provider, [whole[0], _SUMMARY, *whole[-4:], *_conversation(4)[-2:]], _FORK)
    assert _posted(server) == [(_FORK, ["question 4"])]


# ---------------------------------------------------------------------------
# A message Hermes folded into the one before it
# ---------------------------------------------------------------------------


def _folded(n: int) -> list[dict[str, Any]]:
    """Message ``n`` got no reply, and Hermes folded the next one into its row, in
    place (agent/agent_runtime_helpers.py ``_merge_consecutive_users``): one user row
    with both texts, which keeps the first one's id, and the reply."""
    return [
        {
            "role": "user",
            "content": f"question {n}\n\nquestion {n + 1}",
            "message_uid": f"u{n}",
            "_merged_turn_prefix": f"question {n}",
            "_absorbed_message_uids": [f"u{n + 1}"],
        },
        {"role": "assistant", "content": f"answer {n + 1}", "message_uid": f"a{n + 1}"},
    ]


def test_a_message_folded_into_a_copied_one_does_not_take_it_along(plugin, server, tmp_path):
    """The private original's last message got no reply: its turn failed, or Hermes was
    killed. The fork copied the row. Hermes folds the next message into it, and the
    turn begins with one user row that holds both texts. Nothing in the row says where
    one ends, so the turn stays out, whole; the next one goes up without it."""
    provider = _restored(plugin, server, tmp_path)
    said = [*_conversation(3), *_folded(4)]
    provider.on_turn_start(5, "question 5", previous_message="question 4")
    provider.sync_turn("question 5", "answer 5", session_id=_FORK, messages=said)
    assert server.posts() == []
    _say(provider, [*said, *_conversation(6)[-2:]], _FORK)
    assert _posted(server) == [(_FORK, ["question 6"])]


def test_in_a_new_session_whose_turn_starts_were_seen_a_folded_message_goes_up_as_it_is(
    plugin, server, tmp_path
):
    """A new session, and each of its turn starts was seen: both texts are its own, and
    the reply answers both. The row is sent whole, with the row that marks it
    (harnesses/hermes/messages.py). A session taken for a fork does not upload such a
    turn, though it need hold no turn from before (a gateway's managed workers: each
    agent is told the session is new): HANDOFF.md, "Still open"."""
    home = tmp_path / "hermes-a"
    _store(home, _PARENT, source="tui", receipt=False, chat_id="")
    provider = _open(plugin, home, _PARENT)
    provider.on_turn_start(1, "question 1", previous_message="")  # cut short: no reply
    provider.on_turn_start(2, "question 2", previous_message="question 1")
    said = [_conversation(0)[0], *_folded(1)]
    provider.sync_turn("question 2", "answer 2", session_id=_PARENT, messages=said)
    assert _posted(server) == [(_PARENT, ["question 1\n\nquestion 2", ""])]
