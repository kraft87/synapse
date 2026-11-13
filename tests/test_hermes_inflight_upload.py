"""An upload in flight when its process exits is not lost.

Hermes calls ``sync_turn`` on a background worker that an exiting process does not wait
for, and the provider marks a turn sent before its request so that a concurrent hook
cannot send it twice; a request cut off by the exit used to take the turn with it. Each
batch is now written to the outbox before its request and removed once delivered, so
the batch of a request that never finished waits there for a later drain.

The outbox is shared by every session of a Hermes home, in one process (the gateway
daemon runs them all) or several. No drain takes a batch in flight until its request
has had three times its timeout; past that its sender died or hung, and a drain sends
it, once more at most, which the server's dedup on span id absorbs. Nothing is queued
for good. The fakes are the ones tests/test_hermes_provider.py builds; "later" is the
outbox's clock moved on past the deadline.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from tests import test_hermes_provider as base
from tests.helpers.client_core import PLUGIN

hermes_stub = base.hermes_stub
user = base.user
plugin = base.plugin
server = base.server

_ROOT = Path(__file__).resolve().parents[1]
_A = "20261112_120000_aa0001"
_B = "20261112_120100_bb0002"


def _outbox(home: Path) -> Path:
    return home / "synapse" / "outbox"


def _files(home: Path) -> list[str]:
    """Every file in the outbox, temp files included."""
    try:
        return sorted(os.listdir(_outbox(home)))
    except FileNotFoundError:
        return []


def _queued(home: Path, suffix: str = ".json") -> list[dict[str, Any]]:
    """The payloads waiting (``.json``) or being sent (``.sending``)."""
    folder = _outbox(home)
    names = [n for n in _files(home) if n.endswith(suffix) and not n.startswith(".")]
    return [json.loads((folder / n).read_text(encoding="utf-8")) for n in names]


class _Clock:
    """The outbox's ``time``, ``shift`` seconds ahead."""

    def __init__(self, shift: float) -> None:
        self.shift = shift

    def time(self) -> float:
        return time.time() + self.shift

    def time_ns(self) -> int:
        return time.time_ns() + int(self.shift * 1e9)


def _later(monkeypatch, provider: Any, seconds: float = 3600) -> None:
    module = sys.modules[type(provider._outbox).__module__]
    monkeypatch.setattr(module, "time", _Clock(seconds))


def _join(prefix: str = "synapse") -> None:
    for thread in threading.enumerate():
        if thread is not threading.current_thread() and thread.name.startswith(prefix):
            thread.join(10)


def _wait_for(predicate, seconds: float = 10.0) -> None:
    deadline = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def _sessions(posts: list[Any]) -> list[str]:
    return [p["session_id"] for p in posts]


class _Held:
    """A client whose ``ingest`` waits until released: a request in flight."""

    def __init__(self, client: Any, *, fail: bool = False) -> None:
        self._client = client
        self.fail = fail
        self.entered = threading.Event()
        self.go = threading.Event()

    def ingest(self, payload: Any, timeout: float) -> Any:
        self.entered.set()
        assert self.go.wait(10)
        if self.fail:
            raise ConnectionResetError("reset")
        return self._client.ingest(payload, timeout=timeout)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


# ---------------------------------------------------------------------------
# One session
# ---------------------------------------------------------------------------


def test_a_delivered_batch_leaves_nothing_behind(plugin, server, tmp_path):
    home = tmp_path / "hermes-a"
    provider = base._start(base._new(plugin), home, _A)
    _join()
    provider.sync_turn("q", "a", messages=base._conversation(1))
    provider.sync_turn("q", "a", messages=base._conversation(2))
    assert len(server.posts()) == 2  # its own drain does not send it again
    assert _files(home) == []


def test_a_failed_send_keeps_its_one_copy(plugin, server, tmp_path):
    home = tmp_path / "hermes-a"
    provider = base._start(base._new(plugin), home, _A)
    _join()
    server.ingest_status = 503
    provider.sync_turn("q", "a", messages=base._conversation(1))
    (failed,) = server.posts()
    assert _queued(home) == [failed] and _queued(home, ".sending") == []
    assert [n for n in _files(home) if n.startswith(".")] == [".last_error"]
    server.ingest_status = 200
    provider.sync_turn("q", "a", messages=base._conversation(2))
    _, fresh, drained = server.posts()
    assert drained == failed
    assert base._users(fresh["records"]) == ["question 1", "question 2"]
    assert _files(home) == []


def test_without_a_copy_a_failed_send_is_still_queued(plugin, server, tmp_path):
    home = tmp_path / "hermes-a"
    provider = base._start(base._new(plugin), home, _A)
    _join()
    box = type(provider._outbox)
    real = box.put

    def no_copy(self, payload, reason="", *, sending=0):
        if sending:
            raise OSError("disk full")
        return real(self, payload, reason)

    box.put = no_copy
    try:
        provider.sync_turn("q", "a", messages=base._conversation(1))
        assert _files(home) == []
        server.ingest_status = 503
        provider.sync_turn("q", "a", messages=base._conversation(2))
    finally:
        box.put = real
    assert _queued(home) == [server.posts()[1]]


# ---------------------------------------------------------------------------
# A process that exits mid-request
# ---------------------------------------------------------------------------

_CHILD = textwrap.dedent(
    """
    import sys, threading, types
    from pathlib import Path

    sys.path.insert(0, sys.argv[1])
    from tests import test_hermes_provider as base

    package = types.ModuleType("agent")
    package.__path__ = []
    module = types.ModuleType("agent.memory_provider")
    exec(compile(base._STUB, "agent/memory_provider.py", "exec"), module.__dict__)
    package.memory_provider = module
    sys.modules["agent"] = package
    sys.modules["agent.memory_provider"] = module

    home = Path(sys.argv[2])
    plugin = base._load(home / "plugins" / "synapse")
    provider = base._start(base._new(plugin), home, sys.argv[3])
    for thread in threading.enumerate():
        if thread is not threading.current_thread():
            thread.join(10)
    print("ready", flush=True)
    sys.stdin.readline()
    provider.sync_turn("q", "a", messages=base._conversation(1))
    print("returned", flush=True)
    """
)


def test_a_batch_in_flight_when_the_process_dies_is_drained_once(
    plugin, server, tmp_path, monkeypatch
):
    home = tmp_path / "hermes-a"
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD, str(_ROOT), str(home), _A],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None and child.stdin is not None
        assert child.stdout.readline().strip() == "ready"
        server.hang.set()  # the server takes the request and never answers
        child.stdin.write("go\n")
        child.stdin.flush()
        _wait_for(lambda: len(server.posts()) == 1)
        child.kill()
        child.wait(10)
    finally:
        if child.poll() is None:
            child.kill()
    assert "returned" not in (child.stdout.read() if child.stdout else "")
    (in_flight,) = server.posts()
    assert _queued(home, ".sending") == [in_flight]

    server.hang.clear()
    provider = base._start(base._new(plugin), home, _B)
    _join()  # a session start's drain: too soon to tell a dead sender from a slow one
    assert server.posts() == [in_flight]
    _later(monkeypatch, provider)
    provider.sync_turn("q", "a", messages=base._conversation(1))  # its own, then a drain
    assert _sessions(server.posts()) == [_A, _B, _A]
    assert server.posts()[2] == in_flight
    assert _files(home) == []
    provider.sync_turn("q", "a", messages=base._conversation(2))
    assert _sessions(server.posts()) == [_A, _B, _A, _B]


# ---------------------------------------------------------------------------
# Sessions sharing one outbox
# ---------------------------------------------------------------------------

_DRAIN = textwrap.dedent(
    """
    import json, sys, urllib.request

    sys.path.insert(0, sys.argv[1])
    from synapse_client.outbox import Outbox

    def send(payload):
        request = urllib.request.Request(
            sys.argv[3] + "/ingest",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(request, timeout=10).read()

    print(Outbox(sys.argv[2]).drain(send))
    """
)


def _drain_in_another_process(home: Path, url: str) -> str:
    done = subprocess.run(
        [sys.executable, "-c", _DRAIN, str(PLUGIN), str(_outbox(home)), url],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return done.stdout.strip()


def _hold(provider: Any, *, fail: bool = False) -> tuple[_Held, threading.Thread]:
    held = provider._client = _Held(provider._client, fail=fail)
    worker = threading.Thread(
        target=provider.sync_turn, args=("q", "a"), kwargs={"messages": base._conversation(1)}
    )
    worker.start()
    assert held.entered.wait(10)
    return held, worker


def test_no_drain_anywhere_takes_a_batch_in_flight(plugin, server, tmp_path):
    home = tmp_path / "hermes-a"
    a = base._start(base._new(plugin), home, _A)
    b = base._start(base._new(plugin), home, _B)
    _join()
    held, worker = _hold(a)
    (copy,) = _queued(home, ".sending")
    assert copy["session_id"] == _A
    b.sync_turn("q", "a", messages=base._conversation(1))  # sends its own, then drains
    base._start(base._new(plugin), home, "20261112_120200_dd0003")
    _join("synapse-outbox")  # a session start's drain
    assert _drain_in_another_process(home, server.url) == "(0, 1)"
    assert _sessions(server.posts()) == [_B]
    held.go.set()
    worker.join(10)
    assert _sessions(server.posts()) == [_B, _A]
    assert _files(home) == []


@pytest.mark.parametrize("fails", [False, True])
def test_a_sender_past_its_deadline_costs_one_request_more_at_most(
    plugin, server, tmp_path, monkeypatch, fails
):
    home = tmp_path / "hermes-a"
    a = base._start(base._new(plugin), home, _A)
    b = base._start(base._new(plugin), home, _B)
    _join()
    held, worker = _hold(a, fail=fails)
    with monkeypatch.context() as later:
        _later(later, b)
        b.sync_turn("q", "a", messages=base._conversation(1))  # its own, then a drain
    assert _sessions(server.posts()) == [_B, _A]
    held.go.set()
    worker.join(10)
    assert _sessions(server.posts()) == ([_B, _A] if fails else [_B, _A, _A])
    assert _files(home) == []  # neither a second copy nor a stale reason


def test_concurrent_sessions_each_deliver_their_own_turns_once(plugin, server, tmp_path):
    home = tmp_path / "hermes-a"
    ids = [f"20261112_1300{n:02d}_cc{n:04d}" for n in range(6)]
    providers = [base._start(base._new(plugin), home, sid) for sid in ids]
    _join()
    gate = threading.Barrier(len(providers))

    def run(provider: Any) -> None:
        gate.wait(10)
        provider.sync_turn("q", "a", messages=base._conversation(2))

    workers = [threading.Thread(target=run, args=(p,)) for p in providers]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(20)
    posts = server.posts()
    assert sorted(_sessions(posts)) == ids
    for payload in posts:
        assert base._users(payload["records"]) == ["question 1", "question 2"]
    assert _files(home) == []
