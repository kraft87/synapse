"""A turn's upload runs after ``sync_turn`` has returned.

Hermes's guide says ``sync_turn`` must not block (memory-provider-plugin.md, "Threading
Contract"). It calls the hook on one worker per agent, which also runs the agent's
prefetches and the hooks of a session boundary, and it takes a call past 30 seconds for
a stuck one (agent/memory_manager.py). The provider made its requests inside the call:
with a server that took six seconds a request and six batches waiting in the outbox,
one call ran 42 seconds.

``sync_turn`` now settles whose turns it was given, turns them into records and hands
the upload to a thread of the provider's own (background.py): one upload at a time, in
the order the turns ended. What it must not lose by that: a turn, the order, and the
session a turn was for when Hermes moves the agent before the upload runs.

The fakes are the ones tests/test_hermes_provider.py builds. Its ``_new`` gives a
provider whose ``sync_turn`` waits for the upload, which every other test file uses;
the provider here is built without that. A request "in flight" is one the fake server
holds (``hang``) until it is released.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import pytest

from tests import test_hermes_provider as base

Outbox = base.Outbox

hermes_stub = base.hermes_stub
user = base.user
plugin = base.plugin
server = base.server

_SID = base._SID
_NEXT = "20261113_080000_bbbbbb"
_THREAD = "synapse-sync"


def _own(plugin: types.ModuleType, home: Path, session_id: str = _SID, **kwargs: Any) -> Any:
    """A started provider whose ``sync_turn`` is its own, and no thread of the start
    left running."""
    provider = base._start(base._new(plugin, waits=False), home, session_id, **kwargs)
    for thread in threading.enumerate():
        if thread.name.startswith("synapse"):
            thread.join(10)
    return provider


def _timed(call: Any, *args: Any, **kwargs: Any) -> float:
    began = time.monotonic()
    call(*args, **kwargs)
    return time.monotonic() - began


def _wait_for(predicate: Any, seconds: float = 10.0) -> None:
    deadline = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def _threads() -> int:
    return sum(thread.name == _THREAD for thread in threading.enumerate())


def _new_turns(posts: list[dict[str, Any]]) -> list[str]:
    """The turn each upload ends with: the ones before it in a batch are its context,
    or turns sent with it."""
    return [base._users(body["records"])[-1] for body in posts]


def _asked(posts: list[dict[str, Any]]) -> set[str]:
    return {text for body in posts for text in base._users(body["records"])}


# ---------------------------------------------------------------------------
# The hook returns; the upload runs
# ---------------------------------------------------------------------------


def test_sync_turn_is_back_before_its_request_is_answered(plugin, server, tmp_path, hermes_stub):
    home = tmp_path / "hermes-a"
    provider = _own(plugin, home)
    server.hang.set()
    assert _timed(base._say, provider, base._conversation(1), _SID) < 2.0
    _wait_for(lambda: len(server.posts()) == 1)  # made on the provider's thread
    assert not base._uploaded(provider, 0.2) and _threads() == 1
    assert hermes_stub.SPAWNED.count(_THREAD) == 1  # started through Hermes
    server.release.set()
    assert base._uploaded(provider)
    assert Outbox(home / "synapse" / "outbox").pending() == 0
    _wait_for(lambda: _threads() == 0)  # an idle session holds no thread


def test_uploads_run_one_at_a_time_in_the_order_their_turns_ended(
    plugin, server, tmp_path, hermes_stub
):
    provider = _own(plugin, tmp_path / "hermes-a")
    server.hang.set()
    for n in range(1, 5):
        assert _timed(base._say, provider, base._conversation(n), _SID) < 2.0
    _wait_for(lambda: len(server.posts()) == 1)
    time.sleep(0.2)
    assert len(server.posts()) == 1  # the others wait for it
    server.release.set()
    assert base._uploaded(provider)
    assert _new_turns(server.posts()) == [f"question {n}" for n in range(1, 5)]
    assert hermes_stub.SPAWNED.count(_THREAD) == 1  # one thread for the four


def test_a_backlog_in_the_outbox_does_not_keep_the_hook(plugin, server, tmp_path):
    """The case that ran 42 seconds: batches wait in the outbox and the server, back,
    is slow. The drain is the upload thread's work too."""
    home = tmp_path / "hermes-a"
    provider = _own(plugin, home)
    outbox = Outbox(home / "synapse" / "outbox")
    server.ingest_status = 503
    for n in range(1, 5):
        base._say(provider, base._conversation(n), _SID)
        assert base._uploaded(provider)
    assert outbox.pending() == 4
    before = len(server.posts())

    server.ingest_status = 200
    server.hang.set()
    assert _timed(base._say, provider, base._conversation(5), _SID) < 2.0
    server.release.set()
    assert base._uploaded(provider)
    assert outbox.pending() == 0 and outbox.last_error() == ""
    assert _asked(server.posts()[before:]) == {f"question {n}" for n in range(1, 6)}


def test_an_upload_that_raises_does_not_stop_the_ones_after_it(plugin, server, tmp_path, caplog):
    provider = _own(plugin, tmp_path / "hermes-a")
    send, calls = provider._send, []

    def fails_once(*args: Any, **kwargs: Any) -> bool:
        calls.append(args)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return send(*args, **kwargs)

    provider._send = fails_once
    base._say(provider, base._conversation(1), _SID)
    base._say(provider, base._conversation(2), _SID)
    assert base._uploaded(provider)
    (body,) = server.posts()  # the second upload sends the turn the first did not
    assert base._users(body["records"]) == ["question 1", "question 2"]
    assert "a background job failed" in caplog.text


def test_a_session_that_is_off_starts_no_upload_thread(plugin, server, tmp_path, hermes_stub):
    provider = _own(plugin, tmp_path / "hermes-a", platform="telegram")
    base._say(provider, base._conversation(1), _SID)
    assert base._uploaded(provider, 1.0)
    assert _THREAD not in hermes_stub.SPAWNED and server.posts() == []


# ---------------------------------------------------------------------------
# The session a turn is for is settled as the hook runs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("private", [False, True])
def test_a_turn_given_with_no_id_is_not_sent_as_the_session_gone_to_since(
    plugin, server, tmp_path, private
):
    """Hermes's worker ran ``sync_turn`` to its end before the hooks of a session
    boundary (``on_session_end``, then ``on_session_switch`` with ``reset``). The upload
    now runs after them, when the id the agent is in is the new session's: a turn handed
    over with no id is the old session's all the same, on the record or off it."""
    provider = _own(plugin, tmp_path / "hermes-a")
    server.hang.set()
    base._say(provider, base._conversation(1), _SID)
    _wait_for(lambda: len(server.posts()) == 1)  # in flight: the next upload waits
    if private:
        base._private(tmp_path, _SID)
    base._say(provider, base._conversation(2))  # no id: the session the agent is in
    provider.on_session_switch(_NEXT, reset=True, reason="new_session")
    server.release.set()
    assert base._uploaded(provider)

    posts = server.posts()
    assert {body["session_id"] for body in posts} == {_SID}
    assert _new_turns(posts) == (["question 1"] if private else ["question 1", "question 2"])
    assert not (tmp_path / "private" / _NEXT).exists() and server.puts() == []

    # The new session uploads its own turns, and none of the one before.
    base._say(provider, base._turn(7), _NEXT)
    assert base._uploaded(provider)
    (new,) = [body for body in server.posts() if body["session_id"] == _NEXT]
    assert base._users(new["records"]) == ["question 7"]


def test_a_turn_handed_over_is_uploaded_with_the_model_it_ran_on(plugin, server, tmp_path):
    provider = _own(plugin, tmp_path / "hermes-a")
    server.hang.set()
    provider.on_turn_start(1, "question 1", model="model-a")
    provider.sync_turn("question 1", "a", session_id=_SID, messages=base._conversation(1))
    provider.on_turn_start(2, "question 2", model="model-b", previous_message="question 1")
    provider.sync_turn("question 2", "a", session_id=_SID, messages=base._conversation(2))
    provider.on_turn_start(3, "question 3", model="model-c", previous_message="question 2")
    server.release.set()
    assert base._uploaded(provider)
    assert [body["model"] for body in server.posts()] == ["model-a", "model-b"]


# ---------------------------------------------------------------------------
# Closing, and a server that answers nothing
# ---------------------------------------------------------------------------


def test_shutdown_waits_for_an_upload_and_not_for_long(plugin, server, tmp_path, monkeypatch):
    home = tmp_path / "hermes-a"
    provider = _own(plugin, home)
    base._say(provider, base._conversation(1), _SID)
    provider.shutdown()
    assert len(server.posts()) == 1  # sent before the session is closed

    server.hang.set()
    monkeypatch.setattr(base._module(plugin), "LEAVE_WAIT", 0.3)
    base._say(provider, base._conversation(2), _SID)
    _wait_for(lambda: len(server.posts()) == 2)
    assert 0.25 < _timed(provider.shutdown) < 3.0
    # Still in flight: its batch is in the outbox, where a later drain finds it.
    (name,) = [n.name for n in (home / "synapse" / "outbox").iterdir() if n.suffix == ".sending"]
    assert name
    server.release.set()
    assert base._uploaded(provider)
    assert Outbox(home / "synapse" / "outbox").pending() == 0


def test_past_the_limit_a_turn_goes_up_with_a_later_upload(plugin, server, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    provider = _own(plugin, tmp_path / "hermes-a")
    provider._later._limit = 2
    server.hang.set()
    base._say(provider, base._conversation(1), _SID)
    _wait_for(lambda: len(server.posts()) == 1)  # running; two may wait
    for n in range(2, 6):
        assert _timed(base._say, provider, base._conversation(n), _SID) < 2.0
    server.release.set()
    assert base._uploaded(provider)
    assert _new_turns(server.posts()) == ["question 1", "question 2", "question 3"]
    assert caplog.text.count("uploads are behind") == 1  # said once

    base._say(provider, base._conversation(6), _SID)
    assert base._uploaded(provider)
    late = server.posts()[3:]
    assert _asked(late) >= {"question 4", "question 5", "question 6"}
    assert _asked(server.posts()) == {f"question {n}" for n in range(1, 7)}


# ---------------------------------------------------------------------------
# background.Background on its own
# ---------------------------------------------------------------------------


def _background(plugin: types.ModuleType, spawn: Any = None, **kwargs: Any) -> Any:
    module = sys.modules[plugin.__name__ + ".synapse_client.harnesses.hermes.background"]

    def thread(target: Any, *, name: str) -> threading.Thread:
        return threading.Thread(target=target, name=name, daemon=True)

    return module.Background(spawn or thread, "test-jobs", **kwargs)


def test_jobs_run_in_the_order_given(plugin):
    jobs, ran = _background(plugin, limit=1000), []
    for n in range(300):
        assert jobs.put(ran.append, n)
    assert jobs.wait(10) and ran == list(range(300))


def test_a_job_given_as_the_thread_ends_is_run(plugin):
    """Each job here is given as the thread of the one before may be ending: none is
    left with no thread to run it."""
    jobs, ran = _background(plugin), []
    for n in range(300):
        assert jobs.put(ran.append, n)
        assert jobs.wait(10)
    assert ran == list(range(300))


def test_wait_says_whether_the_jobs_are_done(plugin):
    jobs, go = _background(plugin), threading.Event()
    assert jobs.wait(0)  # nothing given yet
    jobs.put(go.wait, 10)
    assert not jobs.wait(0.1)
    go.set()
    assert jobs.wait(10)


def test_no_more_than_the_limit_wait(plugin):
    jobs, go, ran = _background(plugin, limit=2), threading.Event(), []
    began = threading.Event()
    jobs.put(lambda: (began.set(), go.wait(10)))
    assert began.wait(10)  # running: it does not count as waiting
    assert [jobs.put(ran.append, n) for n in range(4)] == [True, True, False, False]
    go.set()
    assert jobs.wait(10) and ran == [0, 1]
    assert jobs.put(ran.append, 9) and jobs.wait(10) and ran == [0, 1, 9]


def test_a_job_that_raises_is_logged_and_the_next_one_runs(plugin, caplog):
    jobs, ran = _background(plugin), []

    def boom() -> None:
        raise ValueError("no")

    jobs.put(boom)
    jobs.put(ran.append, 1)
    assert jobs.wait(10) and ran == [1]
    assert "a background job failed" in caplog.text and "ValueError" in caplog.text


def test_a_thread_that_cannot_start_keeps_the_job_for_the_next_one(plugin):
    refuse = [True]

    def spawn(target: Any, *, name: str) -> threading.Thread:
        if refuse:
            refuse.clear()
            raise RuntimeError("can't start new thread")
        return threading.Thread(target=target, name=name, daemon=True)

    jobs, ran = _background(plugin, spawn), []
    with pytest.raises(RuntimeError):
        jobs.put(ran.append, 1)
    assert jobs.wait(0) and ran == []  # no thread: nothing to wait for
    assert jobs.put(ran.append, 2)
    assert jobs.wait(10) and ran == [1, 2]
