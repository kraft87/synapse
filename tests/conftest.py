import os
from unittest.mock import patch
from urllib.parse import urlsplit

import psycopg
import pytest

from tests.helpers.db import DB_URL


def _refuse_production(url: str) -> None:
    """Abort the run if the test DSN names anything but a scratch database.

    The fixtures below TRUNCATE episodes, kg_entities, notes and friends. pytest.ini
    loads .env, which carries the production SYNAPSE_DB_URL beside SYNAPSE_TEST_URL, so
    one typo (or a copy-pasted export) is the whole distance between a test run and
    wiping live memory. Allowlist the shapes that are disposable by construction --
    `*_test` locally, `ci_<run>_<attempt>` in CI -- and refuse everything else."""
    name = urlsplit(url).path.lstrip("/")
    if not name.endswith("_test") and not name.startswith("ci_"):
        raise RuntimeError(
            f"SYNAPSE_TEST_URL points at database {name!r}, which is not a scratch "
            "database (expected a name ending in '_test' or starting with 'ci_'). "
            "The DB fixtures TRUNCATE real tables -- refusing to run."
        )


_refuse_production(DB_URL)

# Code under test that reads SYNAPSE_DB_URL directly
# would otherwise pick up the production DSN that .env just loaded into this process.
# Pin it to the test database so a missing monkeypatch cannot reach prod.
os.environ["SYNAPSE_DB_URL"] = DB_URL

# A test that reaches the shared Postgres test DB runs in the serial `db` passes, where
# TRUNCATE/INSERT cannot race across xdist workers; every other test fans out freely.
# Tests get the marker from the DB fixtures they use. A module that talks to the database
# directly says so with ``pytestmark = pytest.mark.db`` (DSN: tests/helpers/db.py).
_DB_FIXTURES = frozenset({"conn", "db_url", "clean_tables", "full_surface"})


def pytest_collection_modifyitems(config, items):
    for item in items:
        if _DB_FIXTURES.intersection(item.fixturenames):
            item.add_marker(pytest.mark.db)


@pytest.fixture()
def no_retry_sleep():
    """Short-circuit tenacity's backoff sleep so retry tests don't stall.

    Opt-in (NOT autouse): request it directly, or via a local autouse shim in
    files that want it applied to every test."""
    with patch("tenacity.nap.time.sleep"):
        yield


@pytest.fixture(scope="session", autouse=True)
def _session_pool():
    """SYNAPSE_TEST_POOL=1 runs the suite the way the server runs: one pool open on the
    test DB for the whole session, so every call that goes through ingestion/pg_pool
    borrows from it. CI runs the DB suite both ways. Unset, nothing is opened."""
    if os.environ.get("SYNAPSE_TEST_POOL") != "1":
        yield
        return
    from ingestion import pg_pool

    pg_pool.open_pool(DB_URL)
    yield
    pg_pool.close_pool(DB_URL)


@pytest.fixture(scope="session")
def db_url():
    return DB_URL


@pytest.fixture(scope="session")
def conn(db_url):
    with psycopg.connect(db_url, autocommit=True) as c:
        yield c


@pytest.fixture()
def clean_tables(conn):
    """Truncate all data tables before each test. Opt-in via parameter, not autouse."""
    conn.execute("""
        TRUNCATE episodes, ingestion_state, extraction_queue RESTART IDENTITY CASCADE
    """)
    # Delete tombstones (schema 123, optional) hang off no table, so CASCADE misses them, and
    # a stale one would drop a later test's ingest of the same words as a deleted turn.
    if conn.execute("SELECT to_regclass('deleted_turns')").fetchone()[0]:
        conn.execute("TRUNCATE deleted_turns")
    yield


@pytest.fixture()
def full_surface(conn):
    """A registered trust='full' surface id (schema 053), for tests that are about
    something other than audience scoping.

    Serving is fail-closed: no surface means restricted means empty. Tests that exercise
    ordinary full-corpus serving must therefore name a trusted host explicitly — pass
    this fixture's value as ``surface=``. Registrations are torn down after each test so
    one test's grant can never widen the next one's."""
    from tests.helpers.surfaces import clear_surfaces, register_full

    clear_surfaces(conn)
    sid = register_full(conn)
    yield sid
    clear_surfaces(conn)


@pytest.fixture(autouse=True)
def _reset_dedup_index_cache():
    """The dedup module caches hydrated group indexes process-wide (one shared
    copy per group instead of one per worker thread). Tests build indexes from
    per-test fake KG clients under the same group ids, so the cache must be
    dropped between tests or one test's entities leak into the next."""
    from ingestion import dedup

    dedup._index_cache_reset()
    yield
    dedup._index_cache_reset()


@pytest.fixture(autouse=True)
def _reset_allowlist_variant_cache():
    """ingestion.surfaces caches each allowlist's stored casings for a minute; a test
    that relabels or seeds episodes must not be served the previous test's variants."""
    from ingestion import surfaces

    surfaces._variant_cache.clear()
    yield
    surfaces._variant_cache.clear()


@pytest.fixture(autouse=True)
def _scratch_synapse_home(tmp_path_factory, monkeypatch):
    """Point SYNAPSE_HOME (~/.synapse: the client's config.json, lib/ and bin/) at an
    empty scratch dir for every test. Many client tests isolate CLAUDE_CONFIG_DIR and
    SYNAPSE_DATA_DIR but not HOME, so without this a developer's real config.json would
    leak into their resolution, and a login or enroll test would write to it.
    Subprocesses inherit it through the environment. A test that needs its own home
    sets SYNAPSE_HOME again."""
    monkeypatch.setenv("SYNAPSE_HOME", str(tmp_path_factory.mktemp("synapse-home")))
