import os

import psycopg
import pytest

DB_URL = os.environ.get(
    "SYNAPSE_TEST_URL", "postgresql://synapse:synapse@127.0.0.1:5432/synapse_test"
)

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
    "test_supersede_leg.py",
    "test_config_lane.py",
    "test_config_proposer_db.py",
    "test_config_review_routes.py",
    "test_timeline_routes.py",
    "test_preferences_routes.py",
    "test_recall_metrics.py",
    "test_dedup_gate.py",
    "test_notes_store.py",
    "test_remember_notes.py",
    "test_board.py",
    "test_telemetry_kinds.py",
    "test_import_notes.py",
    "test_tool_surface.py",
    "test_restamp_inherited_dates.py",
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        file_name = os.path.basename(str(item.fspath))
        if file_name in _DB_FILES:
            item.add_marker(pytest.mark.xdist_group(name="db"))
            item.add_marker(pytest.mark.db)


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
