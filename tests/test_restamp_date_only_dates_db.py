"""scripts/restamp_date_only_dates.py against a real database.

Facts dated with no time of day were stored at UTC midnight, the previous evening for a
viewer west of UTC; the script moves them to noon UTC. What is pinned: only midnight values
move, a retirement moves with the date it retires at, a valid-from never passes its own
retirement, --dry-run writes nothing, and a second run finds nothing.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tests.helpers.embed import GROUP

_SPEC = importlib.util.spec_from_file_location(
    "restamp_date_only_dates",
    Path(__file__).resolve().parent.parent / "scripts" / "restamp_date_only_dates.py",
)
assert _SPEC and _SPEC.loader
script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(script)

_ROWS = {
    # uuid: (t_valid, t_invalid)
    "date-only": ("2026-01-31 00:00:00+00", None),
    "instant": ("2026-01-31 20:30:00+00", None),
    "local-midnight": ("2026-01-31 05:00:00+00", None),  # midnight in UTC-5: a real instant
    "retired-at-date": ("2026-01-01 00:00:00+00", "2026-01-31 00:00:00+00"),
    "retired-same-day-instant": ("2026-02-01 00:00:00+00", "2026-02-01 03:00:00+00"),
    "no-date": (None, None),
}


@pytest.fixture()
def facts(conn):
    conn.execute("TRUNCATE kg_relationships RESTART IDENTITY CASCADE")
    for uuid, (valid, invalid) in _ROWS.items():
        conn.execute(
            "INSERT INTO kg_relationships (uuid, owner_id, group_id, src_uuid, tgt_uuid, "
            "t_valid, t_invalid) VALUES (%s, 'default', %s, 's', 't', %s, %s)",
            (uuid, GROUP, valid, invalid),
        )


def _stored(conn) -> dict[str, tuple[str | None, str | None]]:
    rows = conn.execute(
        "SELECT uuid, to_char(t_valid AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI'), "
        "       to_char(t_invalid AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI') "
        "FROM kg_relationships"
    ).fetchall()
    return {r[0]: (r[1], r[2]) for r in rows}


def test_dry_run_counts_and_writes_nothing(facts, conn):
    before = _stored(conn)
    assert script.restamp(conn, dry_run=True) == (3, 1)
    assert _stored(conn) == before


def test_only_midnight_utc_values_move_to_noon(facts, conn):
    assert script.restamp(conn, dry_run=False) == (3, 1)
    stored = _stored(conn)
    assert stored["date-only"] == ("2026-01-31 12:00", None)
    assert stored["instant"] == ("2026-01-31 20:30", None)
    assert stored["local-midnight"] == ("2026-01-31 05:00", None)
    assert stored["no-date"] == (None, None)


def test_a_retirement_moves_with_its_date_and_never_ends_before_it_began(facts, conn):
    script.restamp(conn, dry_run=False)
    stored = _stored(conn)
    assert stored["retired-at-date"] == ("2026-01-01 12:00", "2026-01-31 12:00")
    # Retired at 03:00 on the day it became valid at midnight: noon would outlive it.
    assert stored["retired-same-day-instant"] == ("2026-02-01 03:00", "2026-02-01 03:00")


def test_second_run_finds_nothing(facts, conn):
    script.restamp(conn, dry_run=False)
    assert script.restamp(conn, dry_run=False) == (0, 0)
