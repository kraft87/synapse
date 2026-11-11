"""
Schema smoke tests. Run against a live synapse_test database.
All tests should pass once scripts/apply_schema.sh has been applied.
"""

import shutil
import subprocess
from pathlib import Path

import psycopg
import pytest

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean(clean_tables):
    """Auto-clean tables before every test in this module."""


@pytest.fixture(scope="module", autouse=True)
def _database_at_067(conn, db_url):
    """The tests below read what 067 leaves (no insert trigger, the group defaults, the owner-only
    indexes). conftest applies no schema, and a 066 re-run, here or from another checkout's suite
    on the same database, puts the 066 trigger back; apply 067 first when it is not in place (a
    re-run of a finished file changes nothing)."""
    stamp = conn.execute("SELECT value FROM synapse_meta WHERE key = 'schema_version'").fetchone()
    trigger = conn.execute(
        "SELECT 1 FROM pg_trigger WHERE tgname = 'kg_rel_personal_audience'"
    ).fetchone()
    if stamp is None or stamp[0] < "067" or trigger:
        _apply_067(db_url)


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


def test_schema_version_stamp_matches_code(conn):
    """apply_schema.sh must stamp synapse_meta with the newest migration's
    number — the boot-time guard (ingestion/schema_check.py) compares this
    stamp against the shipped schema/ directory."""
    from ingestion.schema_check import expected_schema_version

    row = conn.execute("SELECT value FROM synapse_meta WHERE key = 'schema_version'").fetchone()
    assert row is not None, "schema_version stamp missing — apply_schema.sh didn't stamp"
    assert row[0] == expected_schema_version()


# ---------------------------------------------------------------------------
# episodes
# ---------------------------------------------------------------------------


def test_episodes_insert_and_bm25(conn):
    conn.execute("""
        INSERT INTO episodes (session_id, sequence, content, platform)
        VALUES ('sess-1', 1, 'The user is building a knowledge graph memory system', 'claude_code')
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
# dedup_gate_shadow — retired (040 creates it, 059 drops it; gate deleted 2026-10)
# ---------------------------------------------------------------------------


def test_dedup_gate_shadow_dropped(conn):
    """A fresh database runs 040 then 059 and must end without the table,
    the same state an upgraded database converges to."""
    row = conn.execute("SELECT to_regclass('public.dedup_gate_shadow')").fetchone()
    assert row[0] is None, "dedup_gate_shadow should be dropped by schema/059"


def test_schema_059_does_not_block_boot():
    """Nothing reads the dropped table, so the boot guard must start this build on
    a database still at 058 (the image lands before the migration is applied)."""
    from ingestion.schema_check import is_optional_migration

    path = Path(__file__).resolve().parent.parent / "schema" / "059_drop_dedup_gate_shadow.sql"
    assert is_optional_migration(path)


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


# ---------------------------------------------------------------------------
# 053 — audience scoping
# ---------------------------------------------------------------------------


def test_notes_audience_defaults_personal_and_is_constrained(conn):
    """Fail-closed in the SCHEMA, not just in the application: a note inserted by any
    path that forgets the column lands on the tier that never leaves a trusted host."""
    conn.execute(
        "INSERT INTO notes (owner_id, group_id, type, hook, body) "
        "VALUES ('t', 'technical', 'user', 'h', 'b')"
    )
    row = conn.execute(
        "SELECT audience FROM notes WHERE owner_id = 't' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row[0] == "personal"

    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute(
            "INSERT INTO notes (owner_id, group_id, type, hook, body, audience) "
            "VALUES ('t', 'technical', 'user', 'h', 'b', 'public')"
        )
    conn.execute("DELETE FROM notes WHERE owner_id = 't'")


def test_surfaces_defaults_to_restricted_with_an_empty_allowlist(conn):
    """Promotion to 'full' must always be an explicit act — an INSERT that omits the
    column cannot grant trust by accident."""
    conn.execute("INSERT INTO surfaces (surface_id) VALUES ('schema-test-host')")
    row = conn.execute(
        "SELECT trust, allowed_projects FROM surfaces WHERE surface_id = 'schema-test-host'"
    ).fetchone()
    assert row[0] == "restricted" and list(row[1]) == []

    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute(
            "INSERT INTO surfaces (surface_id, trust) VALUES ('bad-trust-host', 'partial')"
        )
    conn.execute("DELETE FROM surfaces WHERE surface_id LIKE '%-host'")


# ---------------------------------------------------------------------------
# 066 — fact audience
# ---------------------------------------------------------------------------

_SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schema"
_SCHEMA_066 = _SCHEMA_DIR / "066_kg_fact_audience.sql"
_SCHEMA_067 = _SCHEMA_DIR / "067_kg_one_memory.sql"
_CUTOVER_SQL = "SELECT value FROM synapse_meta WHERE key = 'fact_audience_legacy_max_id'"


def _wipe_kg(conn) -> None:
    conn.execute("TRUNCATE kg_relationships, kg_entities RESTART IDENTITY CASCADE")
    # a 067 run on a re-applied 066 leaves its ledger behind (tests/test_pile_merge_067_db.py owns that)
    for (t,) in conn.execute(
        "SELECT tablename FROM pg_tables WHERE tablename LIKE 'kg_pile_merge_067_%'"
    ).fetchall():
        conn.execute(f'DROP TABLE IF EXISTS "{t}"')


@pytest.fixture()
def kg_rows(conn):
    """A clean KG slate for the test, and again afterwards."""
    _wipe_kg(conn)
    yield conn
    _wipe_kg(conn)


def _fact(conn, uid: str, group: str, audience: str | None = None):
    return conn.execute(
        "INSERT INTO kg_relationships (uuid, group_id, src_uuid, tgt_uuid, fact, audience) "
        "VALUES (%s, %s, 'a', 'b', 'f', %s) RETURNING audience, label_version",
        (uid, group, audience),
    ).fetchone()


def _entity(conn, uid: str, group: str, *, owner: str = "default", flag: bool = False) -> None:
    conn.execute(
        "INSERT INTO kg_entities (uuid, owner_id, group_id, name, is_self) "
        "VALUES (%s, %s, %s, 'n', %s)",
        (uid, owner, group, flag),
    )


def test_fact_audience_columns_exist_with_the_documented_types(conn):
    cols = dict(
        conn.execute(
            "SELECT table_name || '.' || column_name, data_type || ' ' || is_nullable "
            "FROM information_schema.columns "
            "WHERE (table_name = 'kg_relationships' "
            "       AND column_name IN ('audience', 'label_version', 'label_model')) "
            "   OR (table_name = 'kg_entities' AND column_name = 'is_self')"
        ).fetchall()
    )
    assert cols == {
        "kg_relationships.audience": "text YES",
        "kg_relationships.label_version": "smallint YES",
        "kg_relationships.label_model": "text YES",
        "kg_entities.is_self": "boolean NO",
    }


def test_the_audience_vocabulary_is_the_notes_one(kg_rows):
    _fact(kg_rows, "w", "technical", "work-safe")
    _fact(kg_rows, "p", "technical", "personal")
    assert _fact(kg_rows, "n", "technical") == (None, None)
    with pytest.raises(psycopg.errors.CheckViolation):
        _fact(kg_rows, "bad", "technical", "private")


def test_the_cutover_is_stamped_as_a_row_id(conn):
    row = conn.execute(_CUTOVER_SQL).fetchone()
    assert row is not None and row[0].isdigit()


def test_067_dropped_the_personal_insert_trigger_so_no_group_is_labelled_on_insert(kg_rows):
    """066 labelled every insert into the personal group; 067 merged that pile, dropped the
    trigger and the function, and keeps no group special."""
    assert (
        kg_rows.execute(
            "SELECT count(*) FROM pg_trigger WHERE tgname = 'kg_rel_personal_audience'"
        ).fetchone()[0]
        == 0
    )
    assert (
        kg_rows.execute(
            "SELECT count(*) FROM pg_proc WHERE proname = 'kg_relationships_personal_audience_trg'"
        ).fetchone()[0]
        == 0
    )
    assert _fact(kg_rows, "p1", "personal") == (None, None)
    assert _fact(kg_rows, "t1", "technical") == (None, None)
    assert _fact(kg_rows, "p2", "personal", "work-safe") == ("work-safe", None)


@pytest.mark.parametrize("table", ["kg_entities", "kg_relationships", "notes", "preferences"])
def test_067_makes_technical_the_default_group_of_every_row(conn, table):
    default = conn.execute(
        "SELECT column_default FROM information_schema.columns "
        "WHERE table_name = %s AND column_name = 'group_id'",
        (table,),
    ).fetchone()
    assert default == ("'technical'::text",)


def test_067_built_the_owner_only_indexes_valid(conn):
    valid = dict(
        conn.execute(
            "SELECT c.relname, i.indisvalid FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
            "WHERE c.relname IN ('kg_rel_pair_any', 'kg_rel_tgt_any', 'kg_entities_norm_any')"
        ).fetchall()
    )
    assert valid == {"kg_rel_pair_any": True, "kg_rel_tgt_any": True, "kg_entities_norm_any": True}


def test_the_owner_flag_defaults_off_and_is_unique_per_owner_and_group(kg_rows):
    _entity(kg_rows, "e0", "technical")
    assert kg_rows.execute("SELECT is_self FROM kg_entities WHERE uuid = 'e0'").fetchone() == (
        False,
    )
    _entity(kg_rows, "e1", "technical", flag=True)
    _entity(kg_rows, "e2", "elsewhere", flag=True)  # one per group (067 leaves the index as it was)
    _entity(kg_rows, "e3", "technical", owner="someone-else", flag=True)  # one per owner
    _entity(kg_rows, "e4", "technical")  # unflagged rows are not limited
    with pytest.raises(psycopg.errors.UniqueViolation):
        _entity(kg_rows, "e5", "technical", flag=True)


def test_066_is_optional_and_067_required_and_both_registered():
    from ingestion.schema_check import is_optional_migration, required_schema_version

    assert is_optional_migration(_SCHEMA_066)
    assert not is_optional_migration(_SCHEMA_067)
    assert required_schema_version() == "067"
    script = (_SCHEMA_DIR.parent / "scripts" / "apply_schema.sh").read_text()
    assert script.index(_SCHEMA_066.name) < script.index(_SCHEMA_067.name)


def _apply(db_url: str, path: Path) -> None:
    if shutil.which("psql") is None:  # pragma: no cover - CI installs the client
        pytest.skip("psql not installed")
    subprocess.run(
        ["psql", db_url, "-q", "-v", "ON_ERROR_STOP=1", "-f", str(path)],
        check=True,
        capture_output=True,
    )


def _apply_066(db_url: str) -> None:
    _apply(db_url, _SCHEMA_066)


def _apply_067(db_url: str) -> None:
    """Applied after every 066 re-run: that file puts the insert trigger back, and 067 is
    what takes it out again (and folds any personal row the 066 seed just labelled)."""
    _apply(db_url, _SCHEMA_067)


def test_a_rerun_of_066_never_moves_the_cutover_and_reseeds_a_straggling_personal_row(
    kg_rows, db_url
):
    """066 on a database already at 067 seeds the personal rows old code wrote since, and puts
    its insert trigger back; 067 then regroups them (keeping the label) and drops the trigger."""
    stamp = kg_rows.execute(_CUTOVER_SQL).fetchone()
    _fact(kg_rows, "t1", "technical")
    _fact(kg_rows, "p1", "personal")
    kg_rows.execute("UPDATE kg_relationships SET audience = NULL, label_version = NULL")
    try:
        _apply_066(db_url)
        _apply_066(db_url)
        assert kg_rows.execute(_CUTOVER_SQL).fetchone() == stamp
        rows = kg_rows.execute(
            "SELECT uuid, audience, label_version FROM kg_relationships"
        ).fetchall()
        assert dict((u, (a, v)) for u, a, v in rows) == {"t1": (None, None), "p1": ("personal", 0)}
    finally:
        _apply_067(db_url)  # 066 put its insert trigger back, whatever the assertions said
    rows = kg_rows.execute(
        "SELECT uuid, group_id, audience, label_version FROM kg_relationships"
    ).fetchall()
    assert {r[0]: r[1:] for r in rows} == {
        "t1": ("technical", None, None),
        "p1": ("technical", "personal", 0),
    }
    assert (
        kg_rows.execute(
            "SELECT count(*) FROM pg_trigger WHERE tgname = 'kg_rel_personal_audience'"
        ).fetchone()[0]
        == 0
    )


def test_066_leaves_statistics_on_the_label_column(kg_rows, db_url):
    """The seed touches too few rows for autovacuum to analyze the column, so the file does."""
    _fact(kg_rows, "p1", "personal")
    _fact(kg_rows, "t1", "technical")
    kg_rows.execute("UPDATE kg_relationships SET audience = NULL, label_version = NULL")
    kg_rows.execute("ANALYZE kg_relationships")
    kg_rows.execute("UPDATE kg_relationships SET audience = 'work-safe' WHERE uuid = 't1'")
    try:
        _apply_066(db_url)
        stat = kg_rows.execute(
            "SELECT n_distinct, null_frac FROM pg_stats "
            "WHERE tablename = 'kg_relationships' AND attname = 'audience'"
        ).fetchone()
    finally:
        _apply_067(db_url)  # 066 put its insert trigger back
    # t1 is work-safe and p1 was re-seeded personal: no NULLs left, and the stats say so
    assert stat is not None and stat[1] == 0
