"""DB-backed tests for the drain worker's concurrent batch path (issue #13).

The claimed batch runs through a thread pool when a worker_factory is provided and
SYNAPSE_DRAIN_CONCURRENCY > 1. These cover the care points from the issue: genuine
in-batch concurrency, per-item failure isolation (one bad item never releases the
rest), the UsageLimitError contract (quota items released to pending + re-raise so
run_loop backs off), and the serial kill switch.
"""

from __future__ import annotations

import threading

import pytest

from ingestion.db import Database
from ingestion.llm_client import UsageLimitError
from ingestion.poller import Poller


def _seed(conn, n: int) -> list[int]:
    conn.execute("TRUNCATE extraction_queue RESTART IDENTITY")
    ids = []
    for i in range(n):
        row = conn.execute(
            "INSERT INTO extraction_queue (session_id, content, content_type, project) "
            "VALUES ('sess-drain', %s, 'chunk', 'test') RETURNING id",
            (f"chunk content {i}",),
        ).fetchone()
        ids.append(row[0])
    return ids


def _statuses(conn) -> dict[int, str]:
    return dict(conn.execute("SELECT id, status FROM extraction_queue ORDER BY id").fetchall())


class FakePipeline:
    """process_item stand-in; `behavior(item)` decides success/raise per item."""

    def __init__(self, behavior):
        self._behavior = behavior

    def process_item(self, item):
        self._behavior(item)


def _poller(db_url: str, behavior, factory_calls: list[int] | None = None) -> Poller:
    def factory():
        if factory_calls is not None:
            factory_calls.append(1)
        return Database(db_url), FakePipeline(behavior)

    return Poller(
        db=Database(db_url),
        extraction_pipeline=FakePipeline(behavior),
        worker_factory=factory,
    )


def test_batch_runs_concurrently_and_marks_done(conn, db_url, monkeypatch):
    monkeypatch.setenv("SYNAPSE_DRAIN_CONCURRENCY", "4")
    _seed(conn, 4)
    # All 4 items must be in flight at once to pass the barrier; a serial worker
    # would time out waiting -> BrokenBarrierError -> items marked failed.
    barrier = threading.Barrier(4, timeout=15)

    def behavior(item):
        barrier.wait()

    p = _poller(db_url, behavior)
    assert p.drain_extraction_queue(batch_limit=8) == 4
    assert set(_statuses(conn).values()) == {"done"}


def test_per_item_failure_is_isolated(conn, db_url, monkeypatch):
    monkeypatch.setenv("SYNAPSE_DRAIN_CONCURRENCY", "4")
    ids = _seed(conn, 3)
    bad = ids[1]

    def behavior(item):
        if int(item["id"]) == bad:
            raise ValueError("boom")

    p = _poller(db_url, behavior)
    assert p.drain_extraction_queue(batch_limit=8) == 2
    st = _statuses(conn)
    assert st[bad] == "failed"
    assert [st[i] for i in ids if i != bad] == ["done", "done"]
    err = conn.execute("SELECT error FROM extraction_queue WHERE id = %s", (bad,)).fetchone()[0]
    assert "boom" in err


def test_usage_limit_releases_quota_items_and_raises(conn, db_url, monkeypatch):
    monkeypatch.setenv("SYNAPSE_DRAIN_CONCURRENCY", "2")
    ids = _seed(conn, 4)
    quota_id = ids[0]

    def behavior(item):
        if int(item["id"]) == quota_id:
            raise UsageLimitError("weekly cap hit")

    p = _poller(db_url, behavior)
    with pytest.raises(UsageLimitError):
        p.drain_extraction_queue(batch_limit=8)
    st = _statuses(conn)
    assert st[quota_id] == "pending"  # released for retry, NOT failed
    # nothing in the batch may be lost: every row is done or back to pending
    assert set(st.values()) <= {"done", "pending"}


def test_concurrency_1_stays_serial(conn, db_url, monkeypatch):
    monkeypatch.setenv("SYNAPSE_DRAIN_CONCURRENCY", "1")
    _seed(conn, 3)
    factory_calls: list[int] = []
    p = _poller(db_url, lambda item: None, factory_calls)
    assert p.drain_extraction_queue(batch_limit=8) == 3
    assert factory_calls == []  # serial path never builds per-thread workers
    assert set(_statuses(conn).values()) == {"done"}
