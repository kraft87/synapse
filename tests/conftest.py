import os
from unittest.mock import patch
from urllib.parse import urlsplit

import psycopg
import pytest

DB_URL = os.environ.get(
    "SYNAPSE_TEST_URL", "postgresql://synapse:synapse@127.0.0.1:5432/synapse_test"
)


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

# Code under test that reads SYNAPSE_DB_URL directly (ingestion.kg_pg_read/kg_pg_write)
# would otherwise pick up the production DSN that .env just loaded into this process.
# Pin it to the test database so a missing monkeypatch cannot reach prod.
os.environ["SYNAPSE_DB_URL"] = DB_URL

# Files whose tests touch the shared Postgres test DB. The collection hook below
# auto-tags every test in these files with `xdist_group="db"`, which pins them
# onto a single xdist worker so TRUNCATE/INSERT operations don't race across
# parallel workers. Pure-Python test files are left untagged and fan out freely.
_DB_FILES = {
    "test_db.py",
    "test_poller.py",
    "test_schema.py",
    "test_mcp_server.py",
    "test_web_artifacts.py",
    "test_web_enqueue.py",
    "test_contradiction.py",
    "test_extractor.py",
    "test_kg_pg_write.py",
    "test_kg_pg_read.py",
    "test_skills_provider.py",
    "test_skills_lane_v2.py",
    "test_supersede_leg.py",
    "test_config_lane.py",
    "test_config_proposer_db.py",
    "test_config_review_routes.py",
    "test_timeline_routes.py",
    "test_preferences_routes.py",
    "test_recall_metrics.py",
    "test_dedup_gate.py",
    "test_notes_store.py",
    "test_notes_curation.py",
    "test_remember_notes.py",
    "test_board.py",
    "test_dashboard_routes.py",
    "test_telemetry_kinds.py",
    "test_import_notes.py",
    "test_tool_surface.py",
    "test_restamp_inherited_dates.py",
    "test_recall_feedback.py",
    "test_private_session_routes.py",
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        file_name = os.path.basename(str(item.fspath))
        if file_name in _DB_FILES:
            item.add_marker(pytest.mark.xdist_group(name="db"))
            item.add_marker(pytest.mark.db)


@pytest.fixture()
def no_retry_sleep():
    """Short-circuit tenacity's backoff sleep so retry tests don't stall.

    Opt-in (NOT autouse): request it directly, or via a local autouse shim in
    files that want it applied to every test."""
    with patch("tenacity.nap.time.sleep"):
        yield


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
        TRUNCATE episodes, search_cache,
                 ingestion_state, extraction_queue RESTART IDENTITY CASCADE
    """)
    yield


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
