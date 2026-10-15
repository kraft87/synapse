"""DB-backed tests for the recall_metrics write path (mcp_server/recall_metrics.py::_do_record).

Covers issue #10: served_ids JSONB round-trips per bucket.

The timeline and preferences legs left recall() (2026-08-07, 2026-07-27); their
n_timeline / ms_timeline / n_prefs / ms_prefs columns (schema 039) and n_entities (021) are
no longer written, and old rows keep their values until a later migration drops the
columns.
"""

from __future__ import annotations

import uuid

from mcp_server.recall import Recall


def test_served_ids_round_trip(conn, db_url):
    conn.execute("TRUNCATE recall_metrics RESTART IDENTITY")
    r = Recall(db_url, "")
    served = {"episodes": ["e:12", "e:7"], "facts": ["f:41", "f:42"]}
    r._do_record(
        {
            "kind": "recall",
            "source": "test",
            "query": "q",
            "group_id": "technical",
            "served_ids": served,
        }
    )
    row = conn.execute(
        "SELECT served_ids, n_timeline, ms_timeline, n_prefs, ms_prefs "
        "FROM recall_metrics WHERE source = 'test'"
    ).fetchone()
    assert row is not None, "metrics row was not inserted (writer swallowed an error)"
    assert row[0] == served
    assert row[1:] == (None, None, None, None)


def test_missing_served_ids_inserts_null(conn, db_url):
    conn.execute("TRUNCATE recall_metrics RESTART IDENTITY")
    r = Recall(db_url, "")
    r._do_record({"kind": "episodes", "source": "test", "query": "q"})
    row = conn.execute("SELECT served_ids FROM recall_metrics WHERE source = 'test'").fetchone()
    assert row is not None
    assert row[0] is None


def test_served_ids_carries_echo_suppression_count(conn, db_url):
    # Query-echo suppression records n_echo_suppressed inside the served_ids envelope (no DDL).
    conn.execute("TRUNCATE recall_metrics RESTART IDENTITY")
    r = Recall(db_url, "")
    served = {"episodes": ["e:5"], "facts": [], "n_echo_suppressed": 3}
    r._do_record({"kind": "recall", "source": "test", "query": "q", "served_ids": served})
    row = conn.execute("SELECT served_ids FROM recall_metrics WHERE source = 'test'").fetchone()
    assert row is not None
    assert row[0] == served
    assert row[0]["n_echo_suppressed"] == 3


class _DeferredExecutor:
    """Collects submitted work instead of running it: what a call did not wait for."""

    def __init__(self) -> None:
        self.jobs: list = []

    def submit(self, fn, *args) -> None:
        self.jobs.append((fn, args))

    def run(self) -> None:
        for fn, args in self.jobs:
            fn(*args)


def _episode(conn, content: str) -> int:
    return conn.execute(
        "INSERT INTO episodes (session_id, sequence, content, project) "
        "VALUES (%s, 1, %s, 'bump-test') RETURNING id",
        (f"bump-{uuid.uuid4().hex[:12]}", content),
    ).fetchone()[0]


def _count(conn, episode_id: int) -> int:
    return conn.execute(
        "SELECT retrieval_count FROM episodes WHERE id = %s", (episode_id,)
    ).fetchone()[0]


def test_an_episode_bump_does_not_hold_up_the_call_that_served_it(clean_tables, conn, db_url):
    """fetch() and the full-turn tools used to wait for an UPDATE of the served rows
    (an index write on every index of episodes) before answering."""
    from ingestion.surfaces import FULL_TRUST

    a, b = _episode(conn, "bump one"), _episode(conn, "bump two")
    r = Recall(db_url, "")
    deferred = _DeferredExecutor()
    r._async_executor = deferred  # type: ignore[assignment]
    r._record_metrics = lambda m: None  # type: ignore[method-assign]

    out = r.fetch([f"e:{a}", f"e:{b}"], trust=FULL_TRUST)

    assert [e["id"] for e in out["episodes"]] == [f"e:{a}", f"e:{b}"]
    assert (_count(conn, a), _count(conn, b)) == (0, 0)  # not waited for
    deferred.run()
    assert (_count(conn, a), _count(conn, b)) == (1, 1)  # one bump per served episode


def test_an_empty_bump_queues_nothing(db_url):
    r = Recall(db_url, "")
    deferred = _DeferredExecutor()
    r._async_executor = deferred  # type: ignore[assignment]
    r._increment_retrieval_counts([])
    assert deferred.jobs == []


def test_fetch_stays_inside_the_per_call_budget(clean_tables, conn, db_url):
    """Twenty ids can name twenty 100K-character turns. The first is whole (fetch is the
    only way to read it), the rest share what the budget has left and come back as
    openings with full_chars."""
    from ingestion.surfaces import FULL_TRUST

    ids = [_episode(conn, "x" * 30_000 + f" turn {i}") for i in range(3)]
    r = Recall(db_url, "")
    r._async_executor = _DeferredExecutor()  # type: ignore[assignment]
    sent: list[dict] = []
    r._record_metrics = sent.append  # type: ignore[method-assign]

    out = r.fetch([f"e:{i}" for i in ids], trust=FULL_TRUST)

    first, second, third = out["episodes"]
    assert len(first["content"]) == 30_000 + len(" turn 0") and "full_chars" not in first
    assert len(second["content"]) <= 1_000 and second["full_chars"] == 30_000 + len(" turn 1")
    assert len(third["content"]) <= 1_000 and third["full_chars"] == 30_000 + len(" turn 2")
    assert sent[0]["served_ids"]["n_excerpted"] == 2
