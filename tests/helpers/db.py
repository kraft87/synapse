"""The scratch test database's DSN, in one place.

Test modules that talk to the database directly (instead of through the ``conn`` /
``db_url`` fixtures) import ``DB_URL`` from here and mark themselves
``pytestmark = pytest.mark.db``, so they run in the serial DB passes. An unreachable
database is a failure, never a skip: a skipped DB suite looks like a green one.

``decide_behind_a_writer`` is the one two-connection helper: it makes two decisions on one
row land in the same instant."""

import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import psycopg
import pytest

DB_URL = os.environ.get(
    "SYNAPSE_TEST_URL", "postgresql://synapse:synapse@127.0.0.1:5432/synapse_test"
)


def decide_behind_a_writer(db_url: str, write_sql: str, params: tuple, decide: Callable[[], Any]):
    """Run ``decide()`` on a worker thread while a second connection holds an uncommitted
    ``write_sql`` on the row it decides, commit that write once the worker is parked on the
    row lock, and return what ``decide`` returned: a decision that lost the race."""
    with (
        psycopg.connect(db_url) as writer,
        psycopg.connect(db_url, autocommit=True) as watcher,  # its own: a transaction caches stats
        ThreadPoolExecutor(1) as pool,
    ):
        writer.execute(write_sql, params)
        fut = pool.submit(decide)
        for _ in range(100):
            parked = watcher.execute(
                "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() "
                "AND wait_event_type = 'Lock'"
            ).fetchone()[0]
            if parked:
                break
            time.sleep(0.05)
        else:
            writer.rollback()  # release the worker so the pool can shut down
            pytest.fail("the decision never waited on the row")
        writer.commit()
        return fut.result(timeout=10)
