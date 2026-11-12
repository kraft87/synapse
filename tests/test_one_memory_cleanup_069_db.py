"""schema/069_one_memory_cleanup.sql against a real database.

069 drops ``group_id`` from the four one-memory tables, the nine indexes that lead with it,
and the 067 ledger. What only a database shows:

  * past 069 the column is gone from the four tables and stays on recall_metrics; the
    one-self rule and the two live-set indexes stand on owner_id alone
  * on a database at 068, with a merge done and its ledger, 069 removes the column, the
    indexes and the ledger and keeps every row as it was
  * every preflight refusal raises and changes nothing
  * a second run changes nothing and waits for no lock; the stamp never moves down
  * every schema file applied again after 069 succeeds and brings back no old index, ledger
    or column (017, 035, 041, 066 and 067 build what names group_id only while it exists)

The suite's database is past 069. Tests that need 068 rebuild it in place
(tests/helpers/one_memory_states.py) and put the database back afterwards.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import psycopg
import pytest

from ingestion import schema_check
from tests.helpers import one_memory_states as states

_ROOT = Path(__file__).resolve().parent.parent
OWNER = "default"

#: The nine indexes that lead with group_id; 069 drops them with the column.
NINE = (
    "kg_entities_norm",
    "kg_entities_owner",
    "kg_entities_one_self",
    "kg_rel_owner",
    "kg_rel_pair",
    "kg_rel_src",
    "kg_rel_tgt",
    "notes_live_idx",
    "preferences_live_idx",
)
#: The three 069 builds again on owner_id alone, as (unique, definition after USING).
REBUILT = {
    "kg_entities_one_self": (True, "btree (owner_id) WHERE is_self"),
    "notes_live_idx": (False, "btree (owner_id) WHERE (superseded_by IS NULL)"),
    "preferences_live_idx": (False, "btree (owner_id) WHERE (t_invalid IS NULL)"),
}
GONE = tuple(ix for ix in NINE if ix not in REBUILT)
LEDGER = tuple(
    f"kg_pile_merge_067_{t}"
    for t in ("runs", "entities", "relationships", "notes", "preferences", "entity_map", "exclude")
)

pytestmark = pytest.mark.skipif(shutil.which("psql") is None, reason="psql not installed")


def catalog(conn) -> dict[str, Any]:
    """What 069 changes, read from the catalog: the tables that carry group_id, every index
    on the four tables, the ledger tables, and the stamp."""
    return {
        "columns": [
            r[0]
            for r in conn.execute(
                "SELECT table_name FROM information_schema.columns "
                "WHERE table_schema = 'public' AND column_name = 'group_id' "
                "  AND table_name = ANY(%s) ORDER BY 1",
                (list(states.TABLES),),
            ).fetchall()
        ],
        "indexes": {
            name: (unique, definition)
            for name, unique, definition in conn.execute(
                "SELECT c.relname, i.indisunique, "
                "       regexp_replace(pg_get_indexdef(i.indexrelid), '^.* USING ', '') "
                "  FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
                "  JOIN pg_class t ON t.oid = i.indrelid "
                " WHERE t.relnamespace = 'public'::regnamespace AND t.relname = ANY(%s)",
                (list(states.TABLES),),
            ).fetchall()
        },
        "ledger": [
            r[0]
            for r in conn.execute(
                "SELECT tablename FROM pg_tables "
                "WHERE schemaname = 'public' AND tablename LIKE 'kg_pile_merge_067_%' ORDER BY 1"
            ).fetchall()
        ],
        "stamp": states.stamp(conn),
    }


def contents(conn) -> dict[str, tuple[int, str]]:
    """Each table's row count and a digest of every row but its group_id."""
    out = {}
    for t in states.TABLES:
        out[t] = conn.execute(
            "SELECT count(*), md5(COALESCE(string_agg((to_jsonb(x) - 'group_id')::text, '|' "
            f"ORDER BY id), '')) FROM {t} x"
        ).fetchone()
    return out


@pytest.fixture()
def at_068(conn, db_url):
    """A database at 068 with a merge done: both groups written by old code, folded by 067,
    and the ledger it leaves (the exclusion list included). Afterwards: past 069 again."""
    before = states.stamp(conn)
    try:
        states.to_068(conn, db_url)
        conn.execute(
            "INSERT INTO kg_entities (uuid, owner_id, group_id, name, normalized_name, is_self) "
            "VALUES ('t-alpha', %(o)s, 'technical', 'Alpha', 'alpha', true), "
            "       ('p-alpha', %(o)s, 'personal', 'Alpha', 'alpha', false), "
            "       ('p-beta', %(o)s, 'personal', 'Beta', 'beta', false), "
            "       ('o2-alpha', 'o2', 'technical', 'Alpha', 'alpha', true)",
            {"o": OWNER},
        )
        conn.execute(
            "INSERT INTO kg_relationships (uuid, owner_id, group_id, src_uuid, tgt_uuid, name, fact) "
            "VALUES ('f-p', %(o)s, 'personal', 'p-alpha', 'p-beta', 'RELATES', 'a fact'), "
            "       ('f-t', %(o)s, 'technical', 't-alpha', 'p-beta', 'RELATES', 'another')",
            {"o": OWNER},
        )
        conn.execute(
            "INSERT INTO notes (owner_id, group_id, type, hook, body) "
            "VALUES (%(o)s, 'personal', 'user', 'pn', 'body'), "
            "       (%(o)s, 'technical', 'user', 'tn', 'body')",
            {"o": OWNER},
        )
        conn.execute(
            "INSERT INTO preferences (owner_id, group_id, pref, polarity) "
            "VALUES (%(o)s, 'personal', 'pp', 'like'), (%(o)s, 'technical', 'tp', 'like')",
            {"o": OWNER},
        )
        states.psql(db_url, states.SQL_067)
        conn.execute(
            "CREATE TABLE kg_pile_merge_067_exclude "
            "(owner_id text NOT NULL, uuid text NOT NULL, PRIMARY KEY (owner_id, uuid))"
        )
        yield conn
    finally:
        states.to_069(conn, db_url, before)


def _run_069(db_url: str) -> tuple[int, str]:
    """Apply 069 once: its exit code and what it printed to stderr."""
    ran = states.psql(db_url, states.SQL_069, check=False)
    return ran.returncode, ran.stderr


# ---------------------------------------------------------------------------
# What the suite's database (past 069) looks like
# ---------------------------------------------------------------------------


def test_the_four_tables_have_no_group_column_and_recall_metrics_keeps_it(conn):
    assert catalog(conn)["columns"] == []
    assert conn.execute(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = 'recall_metrics' AND column_name = 'group_id'"
    ).fetchone() == (1,)


def test_the_old_indexes_and_the_ledger_are_gone_and_three_stand_on_owner_alone(conn):
    seen = catalog(conn)
    assert not set(GONE) & set(seen["indexes"])
    assert {ix: seen["indexes"].get(ix) for ix in REBUILT} == REBUILT
    assert seen["ledger"] == []


def test_one_self_allows_one_flagged_entity_per_owner(conn):
    conn.execute("TRUNCATE kg_relationships, kg_entities RESTART IDENTITY CASCADE")
    try:
        conn.execute(
            "INSERT INTO kg_entities (uuid, owner_id, name, is_self) "
            "VALUES ('me', 'a', 'Me', true), ('me-b', 'b', 'Me', true), ('x', 'a', 'X', false)"
        )
        with pytest.raises(psycopg.errors.UniqueViolation):
            conn.execute(
                "INSERT INTO kg_entities (uuid, owner_id, name, is_self) VALUES ('me2', 'a', 'Me', true)"
            )
    finally:
        conn.execute("TRUNCATE kg_relationships, kg_entities RESTART IDENTITY CASCADE")


def test_069_is_optional_and_068_stays_the_version_the_code_needs():
    assert schema_check.is_optional_migration(states.SQL_069)
    assert schema_check.required_schema_version() == "068"
    assert schema_check.expected_schema_version() == "069"
    script = (_ROOT / "scripts" / "apply_schema.sh").read_text()
    assert script.index("068_span_id_prefix_is_platform.sql") < script.index(states.SQL_069.name)


def test_the_stamp_never_moves_down(conn, db_url):
    before = states.stamp(conn)
    try:
        states.set_stamp(conn, "071")
        code, err = _run_069(db_url)
        assert code == 0, err
        assert states.stamp(conn) == "071"
    finally:
        states.set_stamp(conn, before)


# ---------------------------------------------------------------------------
# 069 on a database at 068
# ---------------------------------------------------------------------------


def test_on_068_it_drops_the_column_the_nine_indexes_and_the_ledger_and_keeps_every_row(
    at_068, db_url
):
    before = catalog(at_068)
    assert before["columns"] == sorted(states.TABLES)
    assert set(NINE) <= set(before["indexes"])
    assert before["ledger"] == sorted(LEDGER)
    assert before["stamp"] == "068"
    rows = contents(at_068)
    assert all(n > 0 for n, _ in rows.values())

    code, err = _run_069(db_url)
    assert code == 0, err

    after = catalog(at_068)
    assert after["columns"] == []
    assert not set(GONE) & set(after["indexes"])
    assert {ix: after["indexes"].get(ix) for ix in REBUILT} == REBUILT
    others = {k: v for k, v in before["indexes"].items() if k not in NINE}
    assert {k: v for k, v in after["indexes"].items() if k not in REBUILT} == others
    assert after["ledger"] == []
    assert after["stamp"] == "069"
    assert contents(at_068) == rows


def test_a_second_run_changes_nothing_and_waits_for_no_lock(at_068, db_url):
    code, err = _run_069(db_url)
    assert code == 0, err
    seen, rows = catalog(at_068), contents(at_068)
    with psycopg.connect(db_url) as holder:
        holder.execute(
            "LOCK TABLE kg_entities, kg_relationships, notes, preferences IN ACCESS EXCLUSIVE MODE"
        )
        code, err = _run_069(db_url)
        holder.rollback()
    assert code == 0, err
    assert (catalog(at_068), contents(at_068)) == (seen, rows)


REFUSALS = {
    "stamp below 067": (
        "UPDATE synapse_meta SET value = '066' WHERE key = 'schema_version'",
        "UPDATE synapse_meta SET value = '068' WHERE key = 'schema_version'",
        "069 needs 067 applied first (schema_version is 066)",
    ),
    "a row that is not technical": (
        "INSERT INTO notes (owner_id, group_id, type, hook, body) "
        "VALUES ('default', 'elsewhere', 'user', 'stray', 'body')",
        "DELETE FROM notes WHERE hook = 'stray'",
        "notes has 1 rows whose group_id is not technical",
    ),
    "an owner-only index missing": (
        "DROP INDEX kg_rel_tgt_any",
        "CREATE INDEX kg_rel_tgt_any ON kg_relationships (owner_id, tgt_uuid)",
        "index kg_rel_tgt_any is missing or INVALID",
    ),
    "an index the file does not name": (
        "CREATE INDEX kg_rel_by_group ON kg_relationships (group_id)",
        "DROP INDEX kg_rel_by_group",
        "index kg_rel_by_group depends on group_id",
    ),
    "a view": (
        "CREATE VIEW notes_by_group AS SELECT id, group_id FROM notes",
        "DROP VIEW notes_by_group",
        "view notes_by_group depends on group_id",
    ),
    "a function": (
        "CREATE FUNCTION notes_in_group(g text) RETURNS bigint LANGUAGE sql "
        "AS $f$ SELECT count(*) FROM notes WHERE group_id = g $f$",
        "DROP FUNCTION notes_in_group(text)",
        "function notes_in_group(text) names group_id",
    ),
}


@pytest.mark.parametrize("case", list(REFUSALS))
def test_each_preflight_refusal_raises_and_changes_nothing(at_068, db_url, case):
    breaking, mending, message = REFUSALS[case]
    at_068.execute(breaking)
    try:
        seen, rows = catalog(at_068), contents(at_068)
        code, err = _run_069(db_url)
        assert code != 0
        assert message in err
        assert (catalog(at_068), contents(at_068)) == (seen, rows)
    finally:
        at_068.execute(mending)


# ---------------------------------------------------------------------------
# Every schema file applied again
# ---------------------------------------------------------------------------

#: Files that fail when applied again to a migrated database, for reasons of their own that
#: predate 069 and name no group_id: 001 indexes a column a later file dropped, 004 alters
#: the table 007 dropped, 023 adds a constraint with no guard.
NOT_REAPPLIABLE = {
    "001_initial.sql",
    "004_session_embeddings.sql",
    "023_skill_ledger_hardening.sql",
}


def _old_state(conn) -> dict[str, Any]:
    seen = catalog(conn)
    return {
        "columns": seen["columns"],
        "old indexes": sorted(set(GONE) & set(seen["indexes"])),
        "rebuilt": {ix: seen["indexes"].get(ix) for ix in REBUILT},
        "ledger": seen["ledger"],
    }


def test_every_schema_file_applied_again_after_069_brings_nothing_back(conn, db_url):
    """017, 035, 041, 066 and 067 name group_id; each of those statements runs only while
    the column exists, so every file applied again past 069, in apply_schema.sh's order,
    builds none of what 069 dropped, and the ones that fail fail for their own reasons."""
    before = catalog(conn)
    clean = {"columns": [], "old indexes": [], "rebuilt": REBUILT, "ledger": []}
    assert _old_state(conn) == clean
    files = re.findall(
        r"^  (\d{3}_\S+\.sql)$", (_ROOT / "scripts" / "apply_schema.sh").read_text(), re.M
    )
    assert files[-1] == states.SQL_069.name
    failed = {}
    for name in files:
        ran = states.psql(db_url, states.SCHEMA / name, check=False)
        if ran.returncode:
            failed[name] = ran.stderr[-500:]
        assert _old_state(conn) == clean, name
    assert set(failed) == NOT_REAPPLIABLE, failed
    assert not [name for name, err in failed.items() if "group_id" in err], failed
    assert catalog(conn) == before
