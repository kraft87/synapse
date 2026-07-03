"""
Schema smoke tests. Run against a live synapse_test database.
All tests should pass once scripts/apply_schema.sh has been applied.
"""

import psycopg
import pytest

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean(clean_tables):
    """Auto-clean tables before every test in this module."""


def test_pgvector_extension(conn):
    row = conn.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'").fetchone()
    assert row is not None, "pgvector extension not installed"


def test_pg_search_extension(conn):
    row = conn.execute("SELECT extname FROM pg_extension WHERE extname = 'pg_search'").fetchone()
    assert row is not None, "pg_search extension not installed (ParadeDB image required)"


# ---------------------------------------------------------------------------
# Table existence
# ---------------------------------------------------------------------------

EXPECTED_TABLES = [
    "episodes",
    "search_cache",
    "ingestion_state",
    "extraction_queue",
]


@pytest.mark.parametrize("table", EXPECTED_TABLES)
def test_table_exists(conn, table):
    row = conn.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname='public' AND tablename=%s", (table,)
    ).fetchone()
    assert row is not None, f"Table '{table}' missing"


# ---------------------------------------------------------------------------
# episodes
# ---------------------------------------------------------------------------


def test_episodes_insert_and_bm25(conn):
    conn.execute("""
        INSERT INTO episodes (session_id, sequence, content, platform)
        VALUES ('sess-1', 1, 'Kyle is building a knowledge graph memory system', 'claude_code')
    """)
    # BM25 via ParadeDB pg_search
    row = conn.execute("""
        SELECT id FROM episodes
        WHERE id @@@ paradedb.match('content', 'knowledge graph')
    """).fetchone()
    assert row is not None


def test_episodes_unique_session_sequence(conn):
    conn.execute(
        "INSERT INTO episodes (session_id, sequence, content) VALUES ('sess-2', 1, 'first')"
    )
    with pytest.raises(psycopg.errors.UniqueViolation):
        conn.execute(
            "INSERT INTO episodes (session_id, sequence, content) VALUES ('sess-2', 1, 'duplicate')"
        )


def test_episodes_vector_column(conn):
    # Insert a fake 2048-dim embedding (all zeros) — schema/003 migrated to VECTOR(2048)
    embedding = "[" + ",".join(["0.0"] * 2048) + "]"
    conn.execute(
        "INSERT INTO episodes (session_id, sequence, content, embedding, is_embedded) "
        "VALUES ('sess-3', 1, 'vector test', %s::vector, TRUE)",
        (embedding,),
    )
    row = conn.execute("SELECT is_embedded FROM episodes WHERE session_id='sess-3'").fetchone()
    assert row[0] is True


def test_episodes_metadata_jsonb(conn):
    conn.execute("""
        INSERT INTO episodes (session_id, sequence, content, metadata)
        VALUES ('sess-4', 1, 'meta test', '{"tool": "bash", "exit_code": 0}')
    """)
    row = conn.execute(
        "SELECT metadata->>'tool' FROM episodes WHERE session_id='sess-4'"
    ).fetchone()
    assert row[0] == "bash"


# ---------------------------------------------------------------------------
# session_summaries — retired (schema/007 drops it; summaries retired 2026-06)
# ---------------------------------------------------------------------------


def test_session_summaries_dropped(conn):
    """schema/007_drop_summaries.sql must have removed the table. Tests only
    tolerated its presence historically because CI's old provision list
    skipped 007; the shipped schema (scripts/apply_schema.sh) applies it."""
    row = conn.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname='public' AND tablename='session_summaries'"
    ).fetchone()
    assert row is None, "session_summaries should be dropped by schema/007"


# ---------------------------------------------------------------------------
# search_cache
# ---------------------------------------------------------------------------


def test_search_cache_unique_query_url(conn):
    conn.execute("""
        INSERT INTO search_cache (query, source_url, content)
        VALUES ('FalkorDB performance', 'https://example.com/falkordb', 'FalkorDB is fast')
    """)
    with pytest.raises(psycopg.errors.UniqueViolation):
        conn.execute("""
            INSERT INTO search_cache (query, source_url, content)
            VALUES ('FalkorDB performance', 'https://example.com/falkordb', 'duplicate')
        """)


def test_search_cache_bm25(conn):
    conn.execute("""
        INSERT INTO search_cache (query, source_url, title, content)
        VALUES ('graph databases', 'https://example.com/graph',
                'Graph DB overview', 'Neo4j and FalkorDB comparison')
    """)
    row = conn.execute("""
        SELECT id FROM search_cache
        WHERE id @@@ paradedb.match('content', 'FalkorDB comparison')
    """).fetchone()
    assert row is not None


# ---------------------------------------------------------------------------
# ingestion_state
# ---------------------------------------------------------------------------


def test_ingestion_state_upsert(conn):
    conn.execute("""
        INSERT INTO ingestion_state (source, last_ingested_at)
        VALUES ('logfire', NOW())
        ON CONFLICT (source) DO UPDATE SET last_ingested_at = EXCLUDED.last_ingested_at
    """)
    row = conn.execute("SELECT source FROM ingestion_state WHERE source='logfire'").fetchone()
    assert row[0] == "logfire"


# ---------------------------------------------------------------------------
# extraction_queue
# ---------------------------------------------------------------------------


def test_extraction_queue_lifecycle(conn):
    conn.execute("""
        INSERT INTO episodes (session_id, sequence, content) VALUES ('sess-q', 1, 'queued ep')
    """)
    ep_id = conn.execute("SELECT id FROM episodes WHERE session_id='sess-q'").fetchone()[0]

    conn.execute(
        """
        INSERT INTO extraction_queue (episode_id, content, content_type, project)
        VALUES (%s, 'queued ep', 'episode', 'synapse')
    """,
        (ep_id,),
    )

    # Simulate processing
    conn.execute(
        """
        UPDATE extraction_queue SET status='processing', attempts=1
        WHERE episode_id=%s
    """,
        (ep_id,),
    )
    conn.execute(
        """
        UPDATE extraction_queue SET status='done', processed_at=NOW()
        WHERE episode_id=%s
    """,
        (ep_id,),
    )

    row = conn.execute(
        "SELECT status, attempts FROM extraction_queue WHERE episode_id=%s", (ep_id,)
    ).fetchone()
    assert row[0] == "done"
    assert row[1] == 1


def test_extraction_queue_cascade_delete(conn):
    conn.execute(
        "INSERT INTO episodes (session_id, sequence, content) VALUES ('sess-cascade', 1, 'will delete')"
    )
    ep_id = conn.execute("SELECT id FROM episodes WHERE session_id='sess-cascade'").fetchone()[0]
    conn.execute(
        "INSERT INTO extraction_queue (episode_id, content, content_type) VALUES (%s, 'x', 'episode')",
        (ep_id,),
    )
    conn.execute("DELETE FROM episodes WHERE id=%s", (ep_id,))
    row = conn.execute("SELECT id FROM extraction_queue WHERE episode_id=%s", (ep_id,)).fetchone()
    assert row is None, "cascade delete should remove queue entry"
