"""A session that holds turns from before uploads only the ones it sees begin on the record.

Two kinds of session hold turns that are not theirs to upload: one put back on the record
(``synapse private off --forget``) after it was carried off it, and one Hermes said was
new although turns were copied into it. Nothing in a turn's rows tells it from a new
one, so the provider goes by what Hermes says as each turn starts: ``on_turn_start``,
after the turn-start compaction and before the turn's ``prefetch``
(agent/turn_context.py), with the user message before this one (``previous_message``).
The rule itself is the core's (tests/test_client_core_turn_upload.py). These are
Hermes's own sequences of hooks:

  * an ACP fork: its agent is built before its row and its copied history are written,
    and the row names no origin (acp_adapter/session.py ``fork_session``);
  * the agent built for the fork later (``_restore``), or for an id it went on to, which
    goes by what the first one kept of it (carried.py ``note_copied``), and three
    orders in which nothing was kept and a copy still goes up (``test_still_open_...``);
  * the first turns after ``off --forget``: a flush as the turn starts or in the middle
    of it, an upload that runs late, a turn Hermes began itself, a message with no text;
  * a compression of the head alone (``/compress here N``), whose flush hooks are handed
    the rows to be summarised and not the turns kept;
  * a message Hermes folded into one that got no reply
    (agent/agent_runtime_helpers.py ``_merge_consecutive_users``).

The fixtures and helpers are those of tests/test_hermes_private_moves.py.
"""

from __future__ import annotations

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
_posted, _questions, _notice, _forget = pm._posted, pm._questions, pm._notice, pm._forget

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
    the new id is not recorded as one that holds turns from off the record, and the
    agent built for it later looks back at its turns."""
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
    assert pl.carried.settled(home, third) and not pl.carried.held(home, third)
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
# The first turns after `synapse private off --forget`
# ---------------------------------------------------------------------------


def _carried(plugin: Any, server: Any, tmp_path: Path) -> Any:
    """A TUI branch of a private session, off the record from its agent's start. It
    holds three turns, copies of its parent's."""
    home = tmp_path / "hermes-a"
    _child(home)
    _marker(tmp_path, _PARENT)
    server.private_rows.add(_PARENT)
    provider = _open(plugin, home, resumed=True, message_count=6)
    assert (tmp_path / "private" / _CHILD).is_file()
    return provider


@pytest.mark.parametrize("flush", ["on_pre_compress", "on_session_end"])
def test_the_first_turn_back_goes_up_though_a_flush_saw_it_before_it_began(
    plugin, server, tmp_path, flush
):
    """Hermes compresses at the start of a turn, before ``on_turn_start``: the flush
    finds the new message with no turn said to have begun. It is neither sent nor
    written off, there or at a flush in the middle of the turn: the turn's own upload
    settles it."""
    provider = _carried(plugin, server, tmp_path)
    _say(provider, _conversation(4), _CHILD)  # a turn of its own, off the record
    _forget(server, tmp_path, _CHILD)
    asked = _conversation(5)[:-1]
    getattr(provider, flush)(asked)
    provider.on_turn_start(5, "question 5", previous_message="question 4")
    getattr(provider, flush)(asked + _tools(5))
    assert server.posts() == []
    provider.sync_turn("question 5", "a", session_id=_CHILD, messages=_conversation(5))
    assert _posted(server) == [(_CHILD, ["question 5"])]


@pytest.mark.parametrize("late", [False, True], ids=["in time", "late"])
def test_a_turn_begun_off_the_record_stays_off_it_when_it_ends_after_forget(
    plugin, server, tmp_path, late
):
    """``off --forget`` is typed while the model works on question 3. Its upload runs
    after, on the record, and may run late: after question 4 has begun. It is not taken
    for the end of a turn begun on the record."""
    provider = _carried(plugin, server, tmp_path)
    _say(provider, _conversation(2), _CHILD)
    provider.on_turn_start(3, "question 3", previous_message="question 2")
    _forget(server, tmp_path, _CHILD)
    if late:
        provider.on_turn_start(4, "question 4", previous_message="question 3")
    provider.sync_turn("question 3", "a", session_id=_CHILD, messages=_conversation(3))
    assert server.posts() == []
    if not late:
        provider.on_turn_start(4, "question 4", previous_message="question 3")
    provider.sync_turn("question 4", "a", session_id=_CHILD, messages=_conversation(4))
    assert _posted(server) == [(_CHILD, ["question 4"])]


@pytest.mark.parametrize(
    "linked", [True, False], ids=["its row names the id left", "the switch alone says"]
)
def test_a_turn_begun_off_the_record_is_counted_once_when_a_compression_moves_it(
    plugin, server, tmp_path, linked
):
    """Question 4 begins off the record, and a compression in the middle of it moves
    the agent to a new id, where the turn ends. The switch takes the count of turns
    still to end along, and the first upload under the new id reads its row, which
    names the id left: the same turn, counted once. Its upload is its end, and the
    first turn after ``off --forget`` goes up."""
    provider = _carried(plugin, server, tmp_path)
    home = tmp_path / "hermes-a"
    third = pm._THIRD
    provider.on_turn_start(4, "question 4", previous_message="question 3")
    _store(home, third, source="tui", receipt=False, chat_id="")
    if linked:
        _link(home, third, _CHILD, None)
    provider.on_pre_compress(_conversation(4)[:-1])
    provider.on_session_switch(third, parent_session_id=_CHILD, reason="compression")
    provider.sync_turn("question 4", "a", session_id=third, messages=_conversation(4))
    assert server.posts() == [] and (tmp_path / "private" / third).is_file()
    _forget(server, tmp_path, third)
    _say(provider, _conversation(5), third)
    assert _posted(server) == [(third, ["question 5"])]


@pytest.mark.parametrize("back", [False, True], ids=["still off the record", "back on it"])
def test_a_turn_a_compression_moved_is_not_counted_again_at_the_next_compression(
    plugin, server, tmp_path, back
):
    """The turn above ended under the id the compression moved it to, and the id it
    began under keeps its count. Hermes compresses again: the switch names the id
    left, and the new id's row names every id before it, the first too. That one's
    count is the same turn, which the id in between took. The first turn on the
    record under the new id goes up, whether the session was put back on the record
    before the second compression or after it."""
    provider = _carried(plugin, server, tmp_path)
    home = tmp_path / "hermes-a"
    third, new = pm._THIRD, pm._NEW
    provider.on_turn_start(4, "question 4", previous_message="question 3")
    _store(home, third, source="tui", receipt=False, chat_id="")
    _link(home, third, _CHILD, None)
    provider.on_pre_compress(_conversation(4)[:-1])
    provider.on_session_switch(third, parent_session_id=_CHILD, reason="compression")
    provider.sync_turn("question 4", "a", session_id=third, messages=_conversation(4))
    n = 4
    if back:
        _forget(server, tmp_path, third)
        _say(provider, _conversation(5), third)
        n = 5
    _store(home, new, source="tui", receipt=False, chat_id="")
    _link(home, new, third, None)
    provider.on_pre_compress(_conversation(n))
    provider.on_session_switch(new, parent_session_id=third, reason="compression")
    if not back:
        _forget(server, tmp_path, new)
    _say(provider, _conversation(n + 1), new)
    sent = [(third, ["question 5"])] if back else []
    assert _posted(server) == [*sent, (new, [f"question {n + 1}"])]


@pytest.mark.parametrize("flush", ["on_pre_compress", "on_session_end"])
def test_a_turn_stopped_off_the_record_does_not_go_up_at_a_flush_after_forget(
    plugin, server, tmp_path, flush
):
    """The carried session is back on the record and has sent a turn there, so its
    look-back is used up. ``synapse private on`` again, and question 5 is stopped
    while the reply streams: Hermes keeps the part of the reply as a row and mirrors
    no interrupted turn (agent/turn_api_call.py, run_agent.py). The turn is whole in
    the rows, and its end was never seen. After ``off --forget`` a flush shows it
    (the user quits, or a compression starts): it sends no turn. The turn that ends
    next is taken for the one still owed, and stays out too."""
    provider = _carried(plugin, server, tmp_path)
    _forget(server, tmp_path, _CHILD)
    _say(provider, _conversation(4), _CHILD)
    _marker(tmp_path, _CHILD)
    provider.on_turn_start(5, "question 5", previous_message="question 4")
    (tmp_path / "private" / _CHILD).unlink()  # `off --forget`: the server has no row
    getattr(provider, flush)(_conversation(5))
    assert _posted(server) == [(_CHILD, ["question 4"])]
    _say(provider, _conversation(6), _CHILD)
    _say(provider, _conversation(7), _CHILD)
    assert _posted(server)[1:] == [(_CHILD, ["question 7"])]


@pytest.mark.parametrize("flush", [None, "on_pre_compress", "on_session_end"])
def test_a_turn_hermes_begins_takes_no_copied_turn_along(plugin, server, tmp_path, flush):
    """After ``off --forget`` Hermes begins a turn itself: a background-process notice
    in the user's place. It starts as any turn does, on the record. The notice opens no
    turn, so the tool calls and the reply land on the last turn the conversation holds,
    a copy of the private parent's: with or without a flush in the middle, it does not
    go, as a turn or as the next one's context."""
    provider = _carried(plugin, server, tmp_path)
    _forget(server, tmp_path, _CHILD)
    notice, reply = _notice(1)
    mid = [*_conversation(3), notice, *_tools(1)]
    provider.on_turn_start(4, notice["content"], previous_message="question 3")
    if flush:
        getattr(provider, flush)(mid)
    provider.sync_turn(
        notice["content"], reply["content"], session_id=_CHILD, messages=[*mid, reply]
    )
    assert server.posts() == []
    _say(provider, [*mid, reply, *_conversation(4)[-2:]], _CHILD)
    assert _posted(server) == [(_CHILD, ["question 4"])]


def test_a_turn_hermes_begins_that_fails_takes_no_copied_turn_along(plugin, server, tmp_path):
    """The turn Hermes began fails before the model says anything. Hermes closes it
    with a failed-turn boundary, which is never sent, and its finalizer mirrors a turn
    whether or not it failed, with the error as the reply (agent/turn_finalizer.py
    ``_mirror_turn_to_memory``). The records then end with the last copied turn, whole,
    and the notice: a turn began on the record and one has ended, but not that one."""
    provider = _carried(plugin, server, tmp_path)
    _forget(server, tmp_path, _CHILD)
    notice, _ = _notice(1)
    boundary = {
        "role": "assistant",
        "content": "Your request was not processed. Send it again if you still want me to "
        "carry it out.",
        "display_kind": "failed_turn",
    }
    said = [*_conversation(3), notice, boundary]
    provider.on_turn_start(4, notice["content"], previous_message="question 3")
    provider.sync_turn(notice["content"], "HTTP 500", session_id=_CHILD, messages=said)
    assert server.posts() == []
    _say(provider, said + _conversation(4)[-2:], _CHILD)
    assert _posted(server) == [(_CHILD, ["question 4"])]


@pytest.mark.parametrize("flush", [None, "on_pre_compress", "on_session_end"])
@pytest.mark.parametrize("spoken", ["here", "in the parent"])
def test_a_turn_cut_short_off_the_record_is_not_finished_by_a_reply_hermes_began(
    plugin, server, tmp_path, spoken, flush
):
    """Question 3 was asked off the record and its reply interrupted: Hermes mirrors no
    interrupted turn, and the turn stays open in the conversation. After ``off
    --forget`` the reply to a notice closes it. That reply began later than the turn
    did, and takes none of it along: also when a flush saw the open turn first and
    left it for the upload that ends it."""
    provider = _carried(plugin, server, tmp_path)
    cut = _conversation(2) + _conversation(3)[-2:-1] + _tools(3)
    if spoken == "here":
        _say(provider, _conversation(2), _CHILD)
        provider.on_turn_start(3, "question 3", previous_message="question 2")
    _forget(server, tmp_path, _CHILD)
    if flush:
        getattr(provider, flush)(cut)
    notice, reply = _notice(1)
    provider.on_turn_start(4, notice["content"], previous_message="question 3")
    provider.sync_turn(
        notice["content"], reply["content"], session_id=_CHILD, messages=[*cut, notice, reply]
    )
    assert server.posts() == []


@pytest.mark.parametrize("answered", [True, False], ids=["answered", "cut short"])
def test_a_message_with_no_text_takes_no_copied_turn_along(plugin, server, tmp_path, answered):
    """The first message after ``off --forget`` is a picture alone. Its row has no text
    and opens no turn, so the reply lands on the last copied turn: also on one that
    was cut short, whose own reply the picture's would otherwise read as. The row
    stays in the records to show it (harnesses/hermes/messages.py). Hermes still
    mirrors the turn, with "[1 image]" for its text (run_agent.py
    ``_sync_external_memory_for_turn``)."""
    provider = _carried(plugin, server, tmp_path)
    _forget(server, tmp_path, _CHILD)
    copied = _conversation(3) if answered else _conversation(3)[:-1]
    said = [*copied, _PICTURE, {"role": "assistant", "content": "A cat."}]
    provider.on_turn_start(4, "", previous_message="question 3")
    provider.sync_turn("[1 image]", "A cat.", session_id=_CHILD, messages=said)
    assert server.posts() == []
    provider.on_turn_start(5, "question 4", previous_message="")
    provider.sync_turn("question 4", "a", session_id=_CHILD, messages=said + _conversation(4)[-2:])
    assert _posted(server) == [(_CHILD, ["question 4"])]


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


@pytest.mark.parametrize("where", ["in process", "gateway"])
def test_a_compression_of_the_head_alone_lets_none_of_the_kept_turns_out(
    plugin, server, tmp_path, where
):
    """``/compress here 2`` after ``off --forget``, before any turn has ended on the
    record. Hermes hands the hooks the rows it is about to summarise and no others
    (agent/conversation_compression_manual.py). In process both flush hooks get that
    head, and the session keeps its id. The gateway gives ``on_pre_compress`` the head,
    then flushes the agent's whole list as it lets the agent go: the only turn that
    agent saw begin was interrupted, so no refused upload noted anything. The two
    exchanges kept as they were are still the private parent's."""
    home = tmp_path / "hermes-a"
    provider = _carried(plugin, server, tmp_path)
    whole = _conversation(3)
    if where == "gateway":
        provider.on_turn_start(4, "question 4", previous_message="question 3")  # interrupted
    _forget(server, tmp_path, _CHILD)
    provider.on_pre_compress(whole[:-4])
    provider.on_session_end(whole[:-4] if where == "in process" else whole)
    assert server.posts() == []
    if where == "in process":
        provider.on_session_switch(
            _CHILD, parent_session_id=_CHILD, reset=False, reason="compression"
        )
    else:
        provider = _open(plugin, home, resumed=True, message_count=6)  # the next agent
    _say(provider, [whole[0], _SUMMARY, *whole[-4:], *_conversation(4)[-2:]], _CHILD)
    assert _posted(server) == [(_CHILD, ["question 4"])]


# ---------------------------------------------------------------------------
# An upload that runs late, past a turn begun off the record
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("forgot", ["before the late upload", "after it"])
def test_a_late_upload_is_not_the_end_of_a_turn_begun_off_the_record_since(
    plugin, server, tmp_path, forgot
):
    """Back on the record, question 5 begins. Hermes has one background worker for the
    memory hooks and nothing waits for it between turns (agent/memory_manager.py), so
    the upload of question 5 may run after question 6 has begun: here the session was
    marked by hand in between, and question 6 began off the record. ``off --forget``
    while the model works on it. Two ends are then owed, and nothing tells which
    upload is whose: neither turn goes up, whether the late upload was refused or
    not. Taken for the end of question 6, it would let question 6 out at its own."""
    provider = _carried(plugin, server, tmp_path)
    _forget(server, tmp_path, _CHILD)
    _say(provider, _conversation(4), _CHILD)
    provider.on_turn_start(5, "question 5", previous_message="question 4")
    _marker(tmp_path, _CHILD)  # `synapse private on`
    provider.on_turn_start(6, "question 6", previous_message="question 5")
    if forgot == "before the late upload":
        _forget(server, tmp_path, _CHILD)
    provider.sync_turn("question 5", "a", session_id=_CHILD, messages=_conversation(5))
    if forgot == "after it":
        _forget(server, tmp_path, _CHILD)
    provider.sync_turn("question 6", "a", session_id=_CHILD, messages=_conversation(6))
    assert _posted(server) == [(_CHILD, ["question 4"])]
    _say(provider, _conversation(7), _CHILD)
    assert _posted(server)[1:] == [(_CHILD, ["question 7"])]


def test_a_turn_that_ended_before_one_begun_off_the_record_is_not_counted_with_it(
    plugin, server, tmp_path
):
    """The same, in order: question 5 ends before the session is marked. One end is
    owed, question 6's, and the turn after it goes up."""
    provider = _carried(plugin, server, tmp_path)
    _forget(server, tmp_path, _CHILD)
    _say(provider, _conversation(4), _CHILD)
    _say(provider, _conversation(5), _CHILD)
    _marker(tmp_path, _CHILD)
    provider.on_turn_start(6, "question 6", previous_message="question 5")
    _forget(server, tmp_path, _CHILD)
    provider.sync_turn("question 6", "a", session_id=_CHILD, messages=_conversation(6))
    _say(provider, _conversation(7), _CHILD)
    assert _posted(server) == [
        (_CHILD, ["question 4"]),
        (_CHILD, _questions(4, 5)),
        (_CHILD, ["question 7"]),
    ]


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


def test_a_message_folded_into_one_from_off_the_record_does_not_take_it_along(
    plugin, server, tmp_path
):
    """The private parent's last message got no reply: its turn failed, or Hermes was
    killed. The branch copied the row. After ``off --forget`` Hermes folds the first
    message into it, and the turn begins on the record with one user row that holds
    both texts. Nothing in the row says where one ends, so the turn stays out, whole;
    the next one goes up without it."""
    provider = _carried(plugin, server, tmp_path)
    _forget(server, tmp_path, _CHILD)
    said = [*_conversation(3), *_folded(4)]
    provider.on_turn_start(5, "question 5", previous_message="question 4")
    provider.sync_turn("question 5", "answer 5", session_id=_CHILD, messages=said)
    assert server.posts() == []
    _say(provider, [*said, *_conversation(6)[-2:]], _CHILD)
    assert _posted(server) == [(_CHILD, ["question 6"])]


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
