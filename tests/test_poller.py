"""DB-backed tests for the extraction drain (issue #13) and the maintenance stages.

Items run through a thread pool of SYNAPSE_DRAIN_CONCURRENCY threads, a row claimed as each
thread frees up. These cover the care points from the issue: genuine concurrency, per-item
failure isolation (one bad item never releases the rest), the UsageLimitError contract
(quota items released to pending + re-raise so the extraction loop backs off), and the
serial kill switch. The two threads together are in tests/test_poller_threads.py.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import psycopg
import pytest

from ingestion.db import Database
from ingestion.llm_client import UsageLimitError
from ingestion.models import Episode
from ingestion.poller import Maintenance, Poller


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


def _poller(db_url: str, behavior) -> Poller:
    def factory():
        return Database(db_url), FakePipeline(behavior)

    return Poller(db=Database(db_url), worker_factory=factory)


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


def test_concurrency_1_runs_one_item_at_a_time(conn, db_url, monkeypatch):
    monkeypatch.setenv("SYNAPSE_DRAIN_CONCURRENCY", "1")
    _seed(conn, 3)
    lock = threading.Lock()
    running = peak = 0

    def behavior(item):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.05)
        with lock:
            running -= 1

    p = _poller(db_url, behavior)
    assert p.drain_extraction_queue(batch_limit=8) == 3
    assert peak == 1
    assert set(_statuses(conn).values()) == {"done"}


def test_stop_request_at_concurrency_1_claims_nothing_more(conn, db_url, monkeypatch):
    monkeypatch.setenv("SYNAPSE_DRAIN_CONCURRENCY", "1")
    ids = _seed(conn, 3)
    stop_after = ids[0]
    p_holder: list[Poller] = []

    def behavior(item):
        # First item requests the stop mid-flight; it must still complete,
        # and the remaining claimed items must go back to pending.
        if int(item["id"]) == stop_after:
            p_holder[0].request_stop()

    p = _poller(db_url, behavior)
    p_holder.append(p)
    assert p.drain_extraction_queue(batch_limit=8) == 1
    st = _statuses(conn)
    assert st[stop_after] == "done"
    assert [st[i] for i in ids[1:]] == ["pending", "pending"]


def test_stop_request_claims_nothing_more_concurrently(conn, db_url, monkeypatch):
    monkeypatch.setenv("SYNAPSE_DRAIN_CONCURRENCY", "2")
    ids = _seed(conn, 6)
    p_holder: list[Poller] = []
    both_in_flight = threading.Barrier(2, timeout=15)

    def behavior(item):
        both_in_flight.wait()
        p_holder[0].request_stop()  # the stop arrives while two items are in flight

    p = _poller(db_url, behavior)
    p_holder.append(p)
    assert p.drain_extraction_queue(batch_limit=8) == 2
    st = _statuses(conn)
    assert [st[i] for i in ids[:2]] == ["done", "done"]  # in-flight items finish
    assert {st[i] for i in ids[2:]} == {"pending"}  # nothing else was claimed


def test_one_slow_item_does_not_idle_the_other_threads(conn, db_url, monkeypatch):
    """The drain claims a row as each thread frees up. Claiming a whole batch and waiting
    on it would leave the other thread idle behind the slow item (and time it out here)."""
    monkeypatch.setenv("SYNAPSE_DRAIN_CONCURRENCY", "2")
    ids = _seed(conn, 4)
    slow, quick = ids[0], ids[1:]
    quick_done = threading.Semaphore(0)
    peaks: list[int] = []

    def behavior(item):
        peaks.append(
            conn.execute(
                "SELECT count(*) FROM extraction_queue WHERE status = 'processing'"
            ).fetchone()[0]
        )
        if int(item["id"]) == slow:
            for _ in quick:
                assert quick_done.acquire(timeout=15), "the other thread was idle"
        else:
            quick_done.release()

    p = _poller(db_url, behavior)
    assert p.drain_extraction_queue(batch_limit=8) == 4
    assert set(_statuses(conn).values()) == {"done"}
    assert max(peaks) <= 2  # never more claimed than there are threads


def test_batch_limit_bounds_one_concurrent_drain(conn, db_url, monkeypatch):
    monkeypatch.setenv("SYNAPSE_DRAIN_CONCURRENCY", "2")
    ids = _seed(conn, 6)
    p = _poller(db_url, lambda item: None)
    assert p.drain_extraction_queue(batch_limit=4) == 4
    assert [_statuses(conn)[i] for i in ids] == ["done"] * 4 + ["pending"] * 2


def test_a_failed_item_is_retried_once_it_cools_off(conn, db_url, monkeypatch):
    """The end of the silent-loss path: a transport failure marks the row failed WITH
    the reason, and the next drain after the retry window picks it back up. Before this,
    `failed` was terminal — claim only ever looked at `pending` — so an item that failed
    against a wrong model id or a dead token stayed graph-less no matter what was fixed.
    """
    from ingestion.llm_client import LLMUnavailableError

    monkeypatch.setenv("SYNAPSE_DRAIN_CONCURRENCY", "1")
    (qid,) = _seed(conn, 1)
    attempts: list[int] = []

    def behavior(item):
        attempts.append(int(item["id"]))
        if len(attempts) == 1:
            raise LLMUnavailableError("model returned no output on any of 3 attempts")

    p = _poller(db_url, behavior)
    assert p.drain_extraction_queue(batch_limit=8) == 0
    row = conn.execute(
        "SELECT status, attempts, error FROM extraction_queue WHERE id = %s", (qid,)
    ).fetchone()
    assert row[0] == "failed" and row[1] == 1 and "no output" in row[2]

    # Still inside the cool-off window: nothing to claim.
    assert p.drain_extraction_queue(batch_limit=8) == 0
    assert attempts == [qid]

    conn.execute(
        "UPDATE extraction_queue SET processed_at = now() - make_interval(mins => %s) WHERE id = %s",
        (Database.FAILED_RETRY_MINUTES + 1, qid),
    )
    assert p.drain_extraction_queue(batch_limit=8) == 1
    assert attempts == [qid, qid]
    assert _statuses(conn)[qid] == "done"


def test_one_replica_holds_the_maintenance_lock(db_url):
    """The lock is the session's: a second replica gets it once the first's connection ends.
    (A distinct application_name keeps both off the test pool, whose connections go back to
    it with their session locks still held.)"""
    sep = "&" if "?" in db_url else "?"
    first, second = (Database(f"{db_url}{sep}application_name=maint_lock_{n}") for n in "ab")
    try:
        assert first.try_maintenance_lock()
        assert not second.try_maintenance_lock()
        assert first.try_maintenance_lock()  # the holder asks again every cycle
        first.close()
        for _ in range(100):  # the server drops the closed session's locks a beat later
            if second.try_maintenance_lock():
                break
            time.sleep(0.05)
        else:
            pytest.fail("the lock was never handed over")
    finally:
        first.close()
        second.close()


def test_the_maintenance_lock_does_not_share_a_keyspace_with_session_locks(db_url):
    """The episode writer takes pg_advisory_xact_lock(hashtext(session_id)); the maintenance
    lock lives as long as the leader's connection, so a session id hashing to its key must
    not block that session's writes."""
    sep = "&" if "?" in db_url else "?"
    leader = Database(f"{db_url}{sep}application_name=maint_lock_keyspace")
    try:
        assert leader.try_maintenance_lock()
        with psycopg.connect(db_url) as conn:
            # 'poller.maintenance' is the session id the old single-key lock collided with
            row = conn.execute(
                "SELECT pg_try_advisory_xact_lock(hashtext('poller.maintenance'))"
            ).fetchone()
        assert row is not None and row[0]
    finally:
        leader.close()


def test_orphan_sweep_goes_through_the_pipeline_kg_client():
    calls: list[int] = []

    class _KG:
        def delete_orphan_entities(self, min_age_hours):
            calls.append(min_age_hours)
            return 3

    class _Pipe:
        _kg = _KG()

    assert Maintenance(db=None, pipeline=_Pipe()).sweep_orphan_entities() == 3  # type: ignore[arg-type]
    assert calls == [24]


# --- embed_pending ---------------------------------------------------------------------


class _Backlog:
    """Stands in for Database: the unembedded rows, handed out newest first like the real
    query, and gone once ``set_embeddings`` stores them."""

    def __init__(self, episodes: int, chunks: int, web_chunks: int = 0) -> None:
        self.rows = {
            "episodes": [{"id": i, "content": f"ep {i}"} for i in range(episodes, 0, -1)],
            "chunks": [{"id": i, "content": f"chunk {i}"} for i in range(chunks, 0, -1)],
            "web_chunks": [{"id": i, "content": f"web {i}"} for i in range(web_chunks, 0, -1)],
        }
        self.stored: list[tuple[str, list[int]]] = []

    def get_unembedded_episodes(self, limit):
        return list(self.rows["episodes"][:limit])

    def get_unembedded_chunks(self, limit):
        return list(self.rows["chunks"][:limit])

    def get_unembedded_web_chunks(self, limit):
        return list(self.rows["web_chunks"][:limit])

    def set_embeddings(self, table, pairs):
        done = {i for i, _ in pairs}
        self.rows[table] = [r for r in self.rows[table] if r["id"] not in done]
        self.stored.append((table, [i for i, _ in pairs]))


def _embedding_maintenance(backlog: _Backlog) -> Maintenance:
    embedder = SimpleNamespace(embed=lambda texts, task: [[0.5] for _ in texts])
    return Maintenance(
        db=backlog,  # type: ignore[arg-type]
        pipeline=SimpleNamespace(_embedder=embedder),  # type: ignore[arg-type]
    )


def test_embed_pending_works_through_the_whole_backlog_in_one_cycle():
    backlog = _Backlog(episodes=7, chunks=3, web_chunks=4)
    assert _embedding_maintenance(backlog).embed_pending(batch_size=3) == 14
    # newest first, one statement per batch, episodes, then chunks, then web chunks
    assert backlog.stored == [
        ("episodes", [7, 6, 5]),
        ("episodes", [4, 3, 2]),
        ("episodes", [1]),
        ("chunks", [3, 2, 1]),
        ("web_chunks", [4, 3, 2]),
        ("web_chunks", [1]),
    ]


def test_embed_pending_stops_at_its_time_budget():
    backlog = _Backlog(episodes=7, chunks=0)
    assert _embedding_maintenance(backlog).embed_pending(batch_size=3, budget_s=0) == 0
    assert backlog.stored == []


def test_embed_pending_with_nothing_pending_does_nothing():
    backlog = _Backlog(episodes=0, chunks=0, web_chunks=0)
    assert _embedding_maintenance(backlog).embed_pending() == 0
    assert backlog.stored == []


def test_embeddings_are_stored_in_one_statement_and_newest_turns_come_first(conn, db_url):
    conn.execute("TRUNCATE episodes RESTART IDENTITY CASCADE")
    db = Database(db_url)
    for seq in range(1, 4):
        db.upsert_episode(Episode(session_id="emb", sequence=seq, content=f"turn {seq}"))
    assert [r["id"] for r in db.get_unembedded_episodes(limit=2)] == [3, 2]
    vec = [0.25] * 2048
    db.set_embeddings("episodes", [(3, vec), (2, vec)])
    assert [r["id"] for r in db.get_unembedded_episodes(limit=10)] == [1]
    stored = conn.execute(
        "SELECT id, is_embedded, vector_dims(embedding) FROM episodes ORDER BY id"
    ).fetchall()
    assert stored == [(1, False, None), (2, True, 2048), (3, True, 2048)]
    db.set_embeddings("episodes", [])  # nothing to store is not an error


def test_the_daily_queue_sweep_empties_text_finished_a_week_ago():
    asked: list[int] = []

    class _Db:
        def blank_finished_queue_text(self, days):
            asked.append(days)
            return 5

    assert Maintenance(db=_Db(), pipeline=None).sweep_finished_queue_text() == 5  # type: ignore[arg-type]
    assert asked == [7]


def test_rebuild_enqueues_each_chunk_by_id_and_a_flushed_tail_behind_live_ingest(conn, db_url):
    conn.execute("TRUNCATE episodes, extraction_queue RESTART IDENTITY CASCADE")
    conn.execute("DELETE FROM chunks WHERE session_id = 'rb-1'")
    db = Database(db_url)
    last = datetime.now(UTC) - timedelta(hours=3)  # the session went quiet
    for seq in range(1, 6):
        db.upsert_episode(
            Episode(
                session_id="rb-1",
                sequence=seq,
                content=f"turn {seq}",
                created_at=last - timedelta(minutes=5 - seq),
            )
        )
    assert Maintenance(db=db, pipeline=None).rebuild_pending_chunks(sessions={"rb-1"}) == 2  # type: ignore[arg-type]

    rows = conn.execute(
        "SELECT c.start_sequence, c.end_sequence, q.priority, q.content "
        "FROM extraction_queue q JOIN chunks c ON c.id = q.chunk_id ORDER BY c.start_sequence"
    ).fetchall()
    assert rows == [(1, 4, 0, ""), (4, 5, 10, "")]  # the full window is live; the tail waits
    conn.execute("DELETE FROM chunks WHERE session_id = 'rb-1'")
