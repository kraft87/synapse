"""scripts/apply_schema.sh: which files it applies, and when it refuses.

Runs the real script against a scratch database with a few fake migrations
(SCHEMA_DIR), so the version gating is tested without paying for the real schema.
The real schema's re-run safety is CI's job (it runs the script a second time with
--all after provisioning)."""

import os
import subprocess
from pathlib import Path

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict, make_conninfo

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "apply_schema.sh"

# 001 stands in for the early migrations (it creates the tables the script probes); each
# file logs its own number, so a test can see exactly which ones ran.
FAKE_MIGRATIONS = {
    "001_base.sql": """
        CREATE TABLE IF NOT EXISTS episodes (id BIGSERIAL PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS synapse_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS applied (n TEXT);
        INSERT INTO applied VALUES ('001');
    """,
    "002_second.sql": "INSERT INTO applied VALUES ('002');",
    "003_third.sql": "INSERT INTO applied VALUES ('003');",
}


@pytest.fixture()
def scratch(db_url, tmp_path):
    """(dsn, schema_dir) of an empty scratch database plus a directory of fake migrations."""
    name = f"{conninfo_to_dict(db_url)['dbname']}_apply"
    admin = make_conninfo(db_url, dbname="postgres")
    with psycopg.connect(admin, autocommit=True) as c:
        c.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        c.execute(f'CREATE DATABASE "{name}"')
    schema_dir = tmp_path / "schema"
    schema_dir.mkdir()
    for fname, sql in FAKE_MIGRATIONS.items():
        (schema_dir / fname).write_text(sql)
    yield make_conninfo(db_url, dbname=name), schema_dir
    with psycopg.connect(admin, autocommit=True) as c:
        c.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def _run(dsn, schema_dir, *args):
    return subprocess.run(
        ["bash", str(SCRIPT), *args, dsn],
        env={**os.environ, "SCHEMA_DIR": str(schema_dir), "SYNAPSE_EMBED_DIMS": "2048"},
        capture_output=True,
        text=True,
    )


def _sql(dsn, sql):
    with psycopg.connect(dsn, autocommit=True) as c:
        cur = c.execute(sql)
        return cur.fetchall() if cur.description else []


def _applied(dsn):
    return [r[0] for r in _sql(dsn, "SELECT n FROM applied")]


def _stamp(dsn):
    return _sql(dsn, "SELECT value FROM synapse_meta WHERE key = 'schema_version'")[0][0]


def test_fresh_database_gets_every_file_and_the_stamp(scratch):
    dsn, schema_dir = scratch
    r = _run(dsn, schema_dir)
    assert r.returncode == 0, r.stderr
    assert _applied(dsn) == ["001", "002", "003"]
    assert _stamp(dsn) == "003"


def test_rerun_on_a_current_database_applies_nothing(scratch):
    dsn, schema_dir = scratch
    assert _run(dsn, schema_dir).returncode == 0
    r = _run(dsn, schema_dir)
    assert r.returncode == 0, r.stderr
    assert "nothing to apply" in r.stdout
    assert _applied(dsn) == ["001", "002", "003"]


def test_upgrade_applies_only_files_above_the_stamp(scratch):
    dsn, schema_dir = scratch
    assert _run(dsn, schema_dir).returncode == 0
    _sql(dsn, "UPDATE synapse_meta SET value = '001' WHERE key = 'schema_version'")
    r = _run(dsn, schema_dir)
    assert r.returncode == 0, r.stderr
    assert _applied(dsn) == ["001", "002", "003", "002", "003"]
    assert _stamp(dsn) == "003"


def test_all_reapplies_everything(scratch):
    dsn, schema_dir = scratch
    assert _run(dsn, schema_dir).returncode == 0
    r = _run(dsn, schema_dir, "--all")
    assert r.returncode == 0, r.stderr
    assert _applied(dsn) == ["001", "002", "003"] * 2


def test_database_without_a_stamp_needs_an_explicit_start(scratch):
    dsn, schema_dir = scratch
    assert _run(dsn, schema_dir).returncode == 0
    _sql(dsn, "DELETE FROM synapse_meta")
    r = _run(dsn, schema_dir)
    assert r.returncode != 0
    assert "--from" in r.stderr and "--all" in r.stderr
    assert _applied(dsn) == ["001", "002", "003"], "a refused run must apply nothing"
    r = _run(dsn, schema_dir, "--from", "3")
    assert r.returncode == 0, r.stderr
    assert _applied(dsn) == ["001", "002", "003", "003"]
    assert _stamp(dsn) == "003"


def test_all_and_from_are_exclusive(scratch):
    dsn, schema_dir = scratch
    r = _run(dsn, schema_dir, "--all", "--from", "2")
    assert r.returncode != 0
    assert "mutually exclusive" in r.stderr


def test_a_misnamed_migration_is_an_error_not_a_silent_skip(scratch):
    dsn, schema_dir = scratch
    (schema_dir / "4_late.sql").write_text("INSERT INTO applied VALUES ('4');")
    r = _run(dsn, schema_dir)
    assert r.returncode != 0
    assert "4_late.sql" in r.stderr
