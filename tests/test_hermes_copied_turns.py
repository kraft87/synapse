"""Turns a session holds that are not its own: the provider does not guess.

A branch, and the fork ``hermes acp`` makes (a gateway branch), name the session they
come from: private mode follows that name (tests/test_hermes_private_lineage.py,
test_hermes_private_moves.py). Until 0.24.15 the provider also guessed that a session
"said to be new" held copies from a session nothing names, by counting turn starts. That
was built for the ACP session manager in the adapter's own process, which no shipped
entry point of Hermes runs, and it cost ordinary sessions their turns. Two rules stand
in its place:

  * an ACP session with no gateway identity is INERT (sessions.from_keywords): that is
    the adapter's own session manager, whose forks name no session;
  * a session's first upload looks back ten turns whatever ``resumed`` says: a gateway
    with managed workers builds an agent for every turn and tells each one that the
    session is new.

These are Hermes's own sequences of hooks, as recorded at bca51750 (the probe traces
``s11_acp_fork``, ``s12_seeded``, ``s14_gw_branch`` and ``s18_acp_init``), and what is
left of the old rule: a session an earlier client recorded as holding copies
(``<hermes_home>/synapse/copied/``) still uploads none of them, for one more release.

The fixtures and helpers are those of tests/test_hermes_private_moves.py.
"""

from __future__ import annotations

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
_ACP = "7d1f0c3e-58b1-4b64-8a55-0b9e7a1c2f00"  # an in-process ACP session id is a uuid4
_FORK = "0f6b2a54-2f0e-4c53-9d2b-6f4d2d7c9a11"
_LOOKBACK = 10


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


# ---------------------------------------------------------------------------
# Rule 1: an ACP session with no gateway identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "built",
    [{"resumed": False, "message_count": 0}, {"resumed": True, "message_count": 6}],
    ids=["the fork's first agent", "an agent built for it again"],
)
def test_an_acp_session_with_no_gateway_identity_uploads_nothing(plugin, server, tmp_path, built):
    """The ACP session manager in the adapter's own process (``s11_acp_fork``): a fork's
    agent is built first, then its row, with no parent, and the original's history is
    copied in. The original is off the record and nothing names it. Such a session is
    INERT: neither the copies nor its own turns go up, by a turn's upload or a flush,
    and it gets no memory."""
    home = tmp_path / "hermes-a"
    _store(home, _ACP, source="acp", receipt=False, chat_id="")
    _marker(tmp_path, _ACP)  # `synapse private on`, in the original
    server.private_rows.add(_ACP)
    fork = _open(plugin, home, _FORK, platform="acp", **built)
    _store(home, _FORK, source="acp", receipt=False, chat_id="")
    assert fork.system_prompt_block() == "" and fork.prefetch("x") == ""
    assert "local sessions only" in fork.handle_tool_call("recall", {"query": "x"})
    fork.on_pre_compress(_conversation(3))
    _say(fork, _conversation(4), _FORK)
    _say(fork, _conversation(5), _FORK)
    fork.on_session_end(_conversation(5))
    fork.shutdown()
    assert server.seen == []
    assert [one.name for one in (tmp_path / "private").iterdir()] == [_ACP]
    assert not (home / "synapse" / "copied").exists()


def test_an_acp_session_the_gateway_runs_is_on_the_record(plugin, server, tmp_path):
    """``hermes acp`` (``s18_acp_init``): the session runs in the gateway, which names
    the platform "acp" and the chat. It is the owner's local session like any other."""
    home = tmp_path / "hermes-a"
    _store(home)
    editor = gs._start_gateway(pl._new(plugin), home, platform="acp")
    assert "BOARD-TEXT" in editor.system_prompt_block()
    _say(editor, _conversation(1), gs._LOCAL)
    assert _posted(server) == [(gs._LOCAL, ["question 1"])]


@pytest.mark.parametrize("private", [True, False], ids=["a private original", "an open one"])
def test_the_fork_hermes_acp_makes_names_its_original(plugin, server, tmp_path, private):
    """``session/fork`` through ``hermes acp`` is a gateway branch (``s18_acp_init``,
    ``s14_gw_branch``): the fork's row names the original, its rows are the original's
    with their ids, and its agent is told it resumes a session. So the fork of a
    private original is private, with nothing guessed. The fork of an open one is on
    the record and looks back like any resumed session (the server skips a turn whose
    id it holds under any session)."""
    home = tmp_path / "hermes-a"
    _store(home)
    _store(home, gs._OTHER)
    _link(home, gs._OTHER, gs._LOCAL, {"_branched_from": gs._LOCAL})
    if private:
        _marker(tmp_path, gs._LOCAL)
        server.private_rows.add(gs._LOCAL)
    fork = gs._start_gateway(
        pl._new(plugin), home, gs._OTHER, platform="acp", resumed=True, message_count=4
    )
    _say(fork, _conversation(3), gs._OTHER)
    if private:
        assert server.posts() == [] and (tmp_path / "private" / gs._OTHER).is_file()
        assert server.puts() == [f"/private-sessions/{gs._OTHER}"]
    else:
        assert _posted(server) == [(gs._OTHER, _questions(1, 3))] and server.puts() == []


# ---------------------------------------------------------------------------
# Rule 2: the same look-back, whatever Hermes says of the session
# ---------------------------------------------------------------------------


def test_a_session_served_by_an_agent_per_turn_uploads_each_turn_once_it_ends(
    plugin, server, tmp_path
):
    """A gateway with ``gateway.managed_workers`` on builds an agent for every turn and
    tells each the session is new. Every turn goes up as it ends, with the turn before
    it as context. No agent sends the whole conversation again: each looks back ten
    turns. Nothing is recorded of the session."""
    home = tmp_path / "hermes-a"
    _store(home, _PARENT, source="tui", receipt=False, chat_id="")
    for n in range(1, 15):
        worker = _open(plugin, home, _PARENT, platform="tui", message_count=2 * (n - 1))
        before = len(server.posts())
        _say(worker, _conversation(n), _PARENT)
        users = [text for _, texts in _posted(server)[before:] for text in texts]
        # The turn that ended, the nine before it, and the one before those as context.
        assert users == _questions(max(1, n - _LOOKBACK), n)
    assert server.puts() == [] and not (home / "synapse" / "copied").exists()


def test_a_new_session_longer_than_the_look_back_sends_its_last_ten_turns(plugin, server, tmp_path):
    """What the rule costs: an agent told the session is new, whose first upload shows
    more than ten finished turns (none of its uploads ran before), sends the last ten.
    Before 0.24.16 it sent them all."""
    home = tmp_path / "hermes-a"
    _store(home, _PARENT, source="tui", receipt=False, chat_id="")
    provider = _open(plugin, home, _PARENT, platform="tui", message_count=0)
    _say(provider, _conversation(14), _PARENT)
    users = [text for _, texts in _posted(server) for text in texts]
    assert users == _questions(4, 14)  # 4 is the context of 5


def test_a_chat_whose_first_message_got_no_reply_uploads_it_with_the_next(plugin, server, tmp_path):
    """The TUI stores a message as it is sent, so one whose turn never ran stays in the
    store. The agent built for the next message is told the session is new and holds
    rows, its first turn start names a message before, and Hermes folds the two into
    one row. That looked like a fork, and the exchange was not uploaded. It is the
    session's own."""
    home = tmp_path / "hermes-a"
    _store(home, _PARENT, source="tui", receipt=False, chat_id="")
    provider = _open(plugin, home, _PARENT, resumed=False, message_count=2)
    said = [_conversation(0)[0], *_folded(1)]
    provider.on_turn_start(1, "question 2", previous_message="question 1")
    provider.sync_turn("question 2", "answer 2", session_id=_PARENT, messages=said)
    both = ["question 1\n\nquestion 2", ""]  # the folded row, and the row that marks it
    assert _posted(server) == [(_PARENT, both)]
    _say(provider, [*said, *_conversation(3)[-2:]], _PARENT)
    assert _posted(server)[1:] == [(_PARENT, [*both, "question 3"])]
    assert server.puts() == [] and not (home / "synapse" / "copied").exists()


def test_a_first_turn_whose_upload_never_ran_goes_up_with_the_next_one(plugin, server, tmp_path):
    """Hermes skips ``sync_turn`` for a turn it stopped, and its stall gate can drop
    one. The turn is stored all the same, and the next turn's upload shows two turns
    where one began since: that was taken for a copy. Both are the session's own."""
    home = tmp_path / "hermes-a"
    _store(home, _PARENT, source="tui", receipt=False, chat_id="")
    provider = _open(plugin, home, _PARENT, platform="tui", message_count=0)
    provider.on_turn_start(1, "question 1", previous_message="")  # no sync_turn follows
    _say(provider, _conversation(2), _PARENT)
    assert _posted(server) == [(_PARENT, _questions(1, 2))]


@pytest.mark.parametrize("flush", ["on_pre_compress", "on_session_end"])
def test_a_flush_before_a_sessions_first_turn_ends_uploads_what_it_holds(
    plugin, server, tmp_path, flush
):
    """A session Hermes says is new can hold finished turns at a flush (an agent built
    for each turn, a session whose last agent uploaded nothing). They are its own."""
    home = tmp_path / "hermes-a"
    _store(home, _PARENT, source="tui", receipt=False, chat_id="")
    provider = _open(plugin, home, _PARENT, platform="tui", message_count=4)
    getattr(provider, flush)(_conversation(2))
    assert _posted(server) == [(_PARENT, _questions(1, 2))]


def test_still_open_a_seeded_session_uploads_the_turns_it_was_seeded_with(plugin, server, tmp_path):
    """Still open, as before: ``session.create`` with ``messages`` (``s12_seeded``)
    writes the seed as rows of a new session, with new ids and no session named. Its
    agent is told it resumes a session. Nothing tells the seed from the session's own
    turns: up to ten of them are uploaded, whatever the session they came from was."""
    home = tmp_path / "hermes-a"
    _store(home, _PARENT, source="tui", receipt=False, chat_id="")
    _marker(tmp_path, _PARENT)  # the session the seed was taken from is off the record
    server.private_rows.add(_PARENT)
    _store(home, _CHILD, source="tui", receipt=False, chat_id="")
    seeded = _open(plugin, home, _CHILD, resumed=True, message_count=5)
    _say(seeded, _conversation(3), _CHILD)
    assert _posted(server) == [(_CHILD, _questions(1, 3))]


# ---------------------------------------------------------------------------
# What is left of the old rule: a session recorded as holding copies
# ---------------------------------------------------------------------------


def test_a_session_an_earlier_client_recorded_keeps_its_copies_out(plugin, server, tmp_path):
    """Clients 0.24.10 to 0.24.15 recorded a session they took for a fork
    (``copied/``). An agent built for it now still looks back at none of its turns:
    nothing goes up before a turn ends in it, then that turn alone, with no copy as its
    context."""
    home = tmp_path / "hermes-a"
    _store(home, _PARENT, source="tui", receipt=False, chat_id="")
    pl.carried.note_copied(home, _PARENT)
    provider = _open(plugin, home, _PARENT, resumed=True, message_count=6)
    provider.on_pre_compress(_conversation(3))
    provider.on_session_end(_conversation(3))
    assert server.posts() == []
    _say(provider, _conversation(4), _PARENT)
    assert _posted(server) == [(_PARENT, ["question 4"])]
    _say(provider, _conversation(5), _PARENT)
    assert _posted(server)[1:] == [(_PARENT, _questions(4, 5))]


def test_a_turn_that_fails_in_a_recorded_session_lets_no_copy_out(plugin, server, tmp_path):
    """The first turn in it ends with no reply: the upload that ends it shows a last
    turn cut short, and the copies before it. None of them is the turn that ended."""
    home = tmp_path / "hermes-a"
    _store(home, _PARENT, source="tui", receipt=False, chat_id="")
    pl.carried.note_copied(home, _PARENT)
    provider = _open(plugin, home, _PARENT, resumed=True, message_count=6)
    _say(provider, _conversation(4)[:-1], _PARENT)  # question 4 got no reply
    assert server.posts() == []
    _say(provider, [*_conversation(4)[:-1], *_conversation(5)[-2:]], _PARENT)
    assert _posted(server) == [(_PARENT, ["question 5"])]


def test_a_session_that_goes_on_from_a_recorded_one_is_kept_out_and_recorded(
    plugin, server, tmp_path
):
    """The record passes to the id the session goes on under (a compression that
    rotates), by its row: the agent built for that id looks back at nothing either."""
    home = tmp_path / "hermes-a"
    _child(home, None)
    pl.carried.note_copied(home, _PARENT)
    provider = _open(plugin, home, _CHILD, resumed=True, message_count=6)
    _say(provider, _conversation(4), _CHILD)
    assert _posted(server) == [(_CHILD, ["question 4"])]
    assert pl.carried.copied(home, _CHILD)


def test_an_agent_moved_to_a_recorded_session_keeps_its_copies_out(plugin, server, tmp_path):
    """The agent was built for another session, and Hermes moved it with no call (a
    resume in place): the record is read before the first upload under that id."""
    home = tmp_path / "hermes-a"
    _store(home, _PARENT, source="tui", receipt=False, chat_id="")
    _store(home, _CHILD, source="tui", receipt=False, chat_id="")
    pl.carried.note_copied(home, _CHILD)
    provider = _open(plugin, home, _PARENT)
    provider.on_pre_compress(_conversation(3))
    _say(provider, _conversation(4), _CHILD)
    assert _posted(server) == [(_PARENT, _questions(1, 3)), (_CHILD, ["question 4"])]


def test_no_session_is_recorded_as_holding_copies_any_more(plugin, server, tmp_path):
    """The shape the old rule took for a fork (said to be new, a turn start that names
    a message before, more turns than began here) leaves no record."""
    home = tmp_path / "hermes-a"
    _store(home, _PARENT, source="tui", receipt=False, chat_id="")
    provider = _open(plugin, home, _PARENT, resumed=False, message_count=0)
    _say(provider, _conversation(4), _PARENT)
    assert _posted(server) == [(_PARENT, _questions(1, 4))]
    assert not (home / "synapse" / "copied").exists()
