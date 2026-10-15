"""scripts/backfill_normalized_name.py against a real database.

The exact-name lookup reads only the indexed ``normalized_name``, so an entity row that
has NULL there is invisible to dedup until the script fills it. What is pinned: it fills
exactly the NULL rows, with the value the write path would have stored; --dry-run
writes nothing; a second run finds nothing.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from ingestion.kg_client import KGClient
from tests.helpers.embed import GROUP

_SPEC = importlib.util.spec_from_file_location(
    "backfill_normalized_name",
    Path(__file__).resolve().parent.parent / "scripts" / "backfill_normalized_name.py",
)
assert _SPEC and _SPEC.loader
script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(script)


@pytest.fixture()
def entities(conn):
    conn.execute("TRUNCATE kg_entities RESTART IDENTITY CASCADE")
    conn.execute(
        "INSERT INTO kg_entities (uuid, owner_id, group_id, name, normalized_name) VALUES "
        "('e-old', 'default', %(g)s, '  Voyage   AI. ', NULL), "
        "('e-punct', 'default', %(g)s, '...', NULL), "
        "('e-set', 'default', %(g)s, 'Synapse', 'kept as stored')",
        {"g": GROUP},
    )


def _stored(conn) -> dict[str, str | None]:
    return dict(conn.execute("SELECT uuid, normalized_name FROM kg_entities").fetchall())


def test_dry_run_counts_and_writes_nothing(entities, conn):
    assert script.backfill(conn, dry_run=True) == (2, 0)
    assert _stored(conn)["e-old"] is None


def test_fills_only_the_null_rows_with_the_write_side_value(entities, conn):
    assert script.backfill(conn, dry_run=False) == (2, 2)
    stored = _stored(conn)
    assert stored["e-old"] == "voyage ai"  # lowercased, whitespace collapsed, ends stripped
    assert stored["e-punct"] == ""  # normalizes to nothing: filled, never matches
    assert stored["e-set"] == "kept as stored"


def test_backfilled_row_is_found_by_the_exact_lookup(entities, conn, db_url):
    script.backfill(conn, dry_run=False)
    kg = KGClient(db_url)
    try:
        assert kg.entity_uuid_by_normalized_name("voyage ai", GROUP) == "e-old"
    finally:
        kg._reset()


def test_second_run_finds_nothing(entities, conn):
    script.backfill(conn, dry_run=False)
    assert script.backfill(conn, dry_run=False) == (0, 0)
