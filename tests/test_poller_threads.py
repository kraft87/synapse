"""The poller's two threads: extraction and maintenance each run on their own schedule.

No database: the queue and the embedding backlog are in-memory fakes, and the loops run on
real threads, so these show what the DB-backed tests in test_poller.py cannot, that neither
thread waits for the other and that a stop joins both.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from collections.abc import Callable
from types import SimpleNamespace

import pytest

from ingestion import poller as poller_mod
from ingestion.llm_client import UsageLimitError
from ingestion.poller import Maintenance, Poller

_WAIT_S = 10.0


class _Queue:
    """The slice of Database the extraction side uses, over rows 1..n held in memory."""

    def __init__(self, n: int) -> None:
        self._lock = threading.Lock()
        self.pending = list(range(1, n + 1))
        self.done: list[int] = []
        self.released: list[int] = []
        self.claimed: list[int] = []

    def release_stale_claims(self) -> int:
        return 0

    def claim_pending_extractions(self, limit: int) -> list[dict]:
        with self._lock:
            take, self.pending = self.pending[:limit], self.pending[limit:]
            self.claimed += take
        return [{"id": i, "content_type": "chunk", "project": "p"} for i in take]

    def release_claims(self, ids: list[int]) -> int:
        with self._lock:
            self.pending += ids
            self.released += ids
        return len(ids)

    def mark_extraction_done(self, queue_id: int) -> None:
        with self._lock:
            self.done.append(queue_id)

    def mark_extraction_failed(self, queue_id: int, error: str) -> None:
        raise AssertionError(f"queue_id={queue_id} failed: {error}")


class _Stack:
    """The maintenance side's Database and pipeline in one: an embedding backlog that
    ``add`` grows and a lock that ``leader`` grants or refuses."""

    def __init__(self, leader: bool = True, backlog: int = 0) -> None:
        self.leader = leader
        self.backlog = backlog
        self.embedded = 0
        self.stale_sweeps = 0
        self._lock = threading.Lock()
        self._embedder = SimpleNamespace(embed=self._embed)
        self._kg = SimpleNamespace(delete_orphan_entities=lambda hours: 0)

    def _embed(self, texts: list[str], task: str) -> list[list[float]]:
        return [[0.5] for _ in texts]

    def add(self, n: int) -> None:
        with self._lock:
            self.backlog += n

    def try_maintenance_lock(self) -> bool:
        return self.leader

    def sessions_with_pending_chunks(self) -> list[str]:
        return []

    def get_unembedded_episodes(self, limit: int) -> list[dict]:
        with self._lock:
            return [{"id": i, "content": "turn"} for i in range(min(limit, self.backlog))]

    def get_unembedded_chunks(self, limit: int) -> list[dict]:
        return []

    def get_unembedded_web_chunks(self, limit: int) -> list[dict]:
        return []

    def artifacts_without_chunks(self, limit: int) -> list[dict]:
        return []

    def artifacts_to_contextualize(self, limit: int) -> list[dict]:
        return []

    def enqueue_web_extraction(self) -> int:
        return 0

    def set_embeddings(self, table: str, pairs: list) -> None:
        with self._lock:
            self.backlog -= len(pairs)
            self.embedded += len(pairs)

    def release_stale_claims(self) -> int:
        self.stale_sweeps += 1
        return 0

    def blank_finished_queue_text(self, days: int) -> int:
        return 0


def _poller(queue: _Queue, process: Callable[[dict], None], stack: _Stack | None) -> Poller:
    worker = SimpleNamespace(process_item=process)
    return Poller(
        db=queue,  # type: ignore[arg-type]
        worker_factory=lambda: (queue, worker),  # type: ignore[arg-type,return-value]
        maintenance=Maintenance(db=stack, pipeline=stack) if stack else None,  # type: ignore[arg-type]
    )


def _until(condition: Callable[[], object], what: str) -> None:
    deadline = time.monotonic() + _WAIT_S
    while not condition():
        if time.monotonic() > deadline:
            pytest.fail(f"never saw: {what}")
        time.sleep(0.01)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Heartbeats in tmp, and the test runner's own signal handlers left alone."""
    monkeypatch.setattr(poller_mod, "_EXTRACTION_HEARTBEAT", tmp_path / "poller.alive")
    monkeypatch.setattr(poller_mod, "_MAINTENANCE_HEARTBEAT", tmp_path / "poller.maintenance.alive")
    monkeypatch.setattr(signal, "signal", lambda *_: None)
    monkeypatch.delenv("SYNAPSE_DRAIN_ONLY", raising=False)
    monkeypatch.setenv("SYNAPSE_DRAIN_CONCURRENCY", "2")


class _Running:
    """``run_loop`` on a thread of its own; leaving the block stops and joins it."""

    def __init__(self, p: Poller, interval_seconds: float = 3600) -> None:
        self.p = p
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, args=(interval_seconds,))

    def _run(self, interval_seconds: float) -> None:
        try:
            self.p.run_loop(interval_seconds=interval_seconds)  # type: ignore[arg-type]
        except BaseException as e:
            self.error = e

    def stop(self) -> None:
        self.p.request_stop()
        self._thread.join(_WAIT_S)
        assert not self._thread.is_alive(), "run_loop did not return after the stop"

    def __enter__(self) -> _Running:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def test_embedding_runs_while_an_extraction_batch_is_in_flight():
    queue, stack = _Queue(1), _Stack(backlog=3)
    in_flight, finish = threading.Event(), threading.Event()

    def process(item: dict) -> None:
        in_flight.set()
        assert finish.wait(_WAIT_S)

    with _Running(_poller(queue, process, stack)):
        assert in_flight.wait(_WAIT_S)
        _until(lambda: stack.embedded == 3, "the backlog embedded behind a running item")
        assert queue.done == []  # the item is still in flight
        finish.set()
        _until(lambda: queue.done == [1], "the item finishing")


def test_a_quota_backoff_pauses_extraction_and_not_embedding():
    queue, stack = _Queue(1), _Stack()

    def process(item: dict) -> None:
        raise UsageLimitError("weekly cap hit")

    with _Running(_poller(queue, process, stack), interval_seconds=0.01):
        _until(lambda: queue.released == [1], "the quota-hit claim released")
        stack.add(5)  # arrives while extraction sits out its backoff
        _until(lambda: stack.embedded == 5, "embedding during the quota backoff")
        assert queue.claimed == [1]  # extraction is still backing off, claiming nothing


def test_only_the_replica_holding_the_lock_runs_maintenance():
    holder, waiting = _Stack(leader=True, backlog=2), _Stack(leader=False, backlog=2)
    with _Running(_poller(_Queue(0), lambda item: None, holder), interval_seconds=0.01):
        with _Running(_poller(_Queue(0), lambda item: None, waiting), interval_seconds=0.01):
            _until(lambda: holder.embedded == 2 and holder.stale_sweeps >= 2, "the holder cycling")
            assert (waiting.embedded, waiting.stale_sweeps) == (0, 0)
            holder.leader, waiting.leader = False, True  # the holder's connection ends
            _until(lambda: waiting.embedded == 2, "the waiting replica taking over")


def test_a_replica_that_lost_the_lock_keeps_extracting():
    queue = _Queue(3)
    with _Running(_poller(queue, lambda item: None, _Stack(leader=False))):
        _until(lambda: sorted(queue.done) == [1, 2, 3], "extraction without the maintenance lock")


def test_a_stop_joins_both_threads_and_lets_the_in_flight_items_finish():
    queue, stack = _Queue(6), _Stack()
    started, finish = [], threading.Event()

    def process(item: dict) -> None:
        started.append(item["id"])
        assert finish.wait(_WAIT_S)

    p = _poller(queue, process, stack)
    with _Running(p) as run:
        _until(lambda: len(started) == 2, "two items in flight")
        p.request_stop()  # SIGTERM
        finish.set()
        run.stop()
        assert run.error is None
    names = {t.name for t in threading.enumerate()}
    assert not names & {"extraction", "maintenance"}
    assert sorted(queue.done) == [1, 2]  # the in-flight pair finished
    assert queue.claimed == [1, 2]  # and nothing else was claimed
    assert queue.pending == [3, 4, 5, 6]


def test_drain_only_runs_no_maintenance(caplog, monkeypatch):
    monkeypatch.setenv("SYNAPSE_DRAIN_ONLY", "1")
    queue, stack = _Queue(2), _Stack(backlog=3)
    with caplog.at_level(logging.WARNING, logger="ingestion.poller"):
        with _Running(_poller(queue, lambda item: None, stack)):
            _until(lambda: sorted(queue.done) == [1, 2], "extraction on a drain-only replica")
    assert (stack.embedded, stack.stale_sweeps) == (0, 0)
    assert "SYNAPSE_DRAIN_ONLY is deprecated" in caplog.text
    assert not poller_mod._MAINTENANCE_HEARTBEAT.exists()


def test_each_thread_touches_its_own_heartbeat():
    with _Running(_poller(_Queue(0), lambda item: None, _Stack())):
        _until(
            lambda: (
                poller_mod._EXTRACTION_HEARTBEAT.exists()
                and poller_mod._MAINTENANCE_HEARTBEAT.exists()
            ),
            "both heartbeats",
        )


def test_a_thread_that_dies_stops_the_other_and_raises(tmp_path, monkeypatch):
    # An unwritable heartbeat directory kills the maintenance thread at its first iteration.
    monkeypatch.setattr(poller_mod, "_MAINTENANCE_HEARTBEAT", tmp_path / "gone" / "beat")
    with _Running(_poller(_Queue(0), lambda item: None, _Stack())) as run:
        _until(lambda: run.error, "run_loop raising")
    assert isinstance(run.error, RuntimeError) and "maintenance" in str(run.error)
    assert run.p._stop_requested.is_set()  # the extraction thread was told to stop too


def test_a_failing_stage_does_not_stop_the_others():
    stack = _Stack(backlog=2)

    def broken() -> list[str]:
        raise RuntimeError("chunk rebuild failed")

    stack.sessions_with_pending_chunks = broken  # type: ignore[method-assign]
    Maintenance(db=stack, pipeline=stack).cycle(daily=True)  # type: ignore[arg-type]
    assert stack.embedded == 2 and stack.stale_sweeps == 1


@pytest.mark.parametrize("threads,limit", [("3", 6), ("1", 2)])
def test_extraction_drains_two_waves_of_the_pool_per_call(monkeypatch, threads, limit):
    monkeypatch.setenv("SYNAPSE_DRAIN_CONCURRENCY", threads)
    p = _poller(_Queue(0), lambda item: None, None)
    limits = []

    def drain(batch_limit: int) -> int:
        limits.append(batch_limit)
        p.request_stop()
        return 0

    p.drain_extraction_queue = drain  # type: ignore[method-assign]
    p._extract()
    assert limits == [limit]
