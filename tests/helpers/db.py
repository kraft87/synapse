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


def research_archive_exists(conn: psycopg.Connection[Any]) -> bool:
    """Whether the cold archive's table is here. ``schema/076_research_archive.sql`` is an
    optional migration: an image can land before it is applied by hand."""
    return conn.execute("SELECT to_regclass('research_archive')").fetchone()[0] is not None


def research_archive_ready(conn: psycopg.Connection[Any]) -> bool:
    """Whether the archive is usable as the code under test sees it: the table with the
    provenance column ``ARCHIVE_PROBE`` checks. A table a cron made on the fly, before 076
    added that column, is not."""
    from ingestion.web_artifacts import ARCHIVE_PROBE

    return ARCHIVE_PROBE.ready(conn, f"tests:{conn.info.dsn}")


def truncate_research_archive(conn: psycopg.Connection[Any]) -> None:
    """Empty the cold archive, on a database that has one (the same guard conftest puts
    around the optional ``deleted_turns``). Nothing references it, so it needs no CASCADE."""
    if research_archive_exists(conn):
        conn.execute("TRUNCATE research_archive RESTART IDENTITY")


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
