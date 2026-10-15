#!/usr/bin/env python3
"""Restamp fact dates that sit at UTC midnight to noon UTC.

The edge-date prompt used to say "if only a date is mentioned, assume 00:00:00", so a fact
dated "January 31" was stored as 2026-01-31T00:00Z, which is the evening of the 30th for a
viewer west of UTC. Dates the extractor reads are now stored at noon UTC (the timeline gate
always did), which lands on the same calendar day from UTC-11 to UTC+11, and the readers
serve a fact's day in the user's own zone. This brings the rows written before then in line:
``kg_relationships.t_valid`` and ``t_invalid`` values that are exactly 00:00:00 UTC move to
12:00:00 UTC. (A real instant at exactly UTC midnight is rare, and moving one costs half a day.)

A retirement date moves with the date it retires at, and a valid-from never moves past its
own retirement. Idempotent: after a run nothing is left at midnight.

Usage:
    scripts/restamp_date_only_dates.py [--dry-run] [--db-url DSN]

The DSN comes from --db-url, else $SYNAPSE_DB_URL. --dry-run only counts. The output is
counts only: fact text stays out of terminals and logs.
"""

from __future__ import annotations

import argparse
import os
import sys

import psycopg

_AT_MIDNIGHT = "(({col} AT TIME ZONE 'UTC')::time = time '00:00')"


def restamp(conn: psycopg.Connection, *, dry_run: bool) -> tuple[int, int]:
    """Returns (t_valid values at UTC midnight, t_invalid values at UTC midnight) found
    before the run; with ``dry_run`` False, those are the values moved to noon."""
    valid_at_midnight = _AT_MIDNIGHT.format(col="t_valid")
    invalid_at_midnight = _AT_MIDNIGHT.format(col="t_invalid")
    row = conn.execute(
        f"SELECT count(*) FILTER (WHERE {valid_at_midnight}), "  # nosec B608 — fixed fragments, no input
        f"       count(*) FILTER (WHERE {invalid_at_midnight}) "
        "FROM kg_relationships"
    ).fetchone()
    assert row is not None
    found = (int(row[0]), int(row[1]))
    if dry_run:
        return found
    with conn.transaction():
        # The retirement first, so the valid-from clamp below sees where it ended up.
        conn.execute(
            "UPDATE kg_relationships SET t_invalid = t_invalid + interval '12 hours' "  # nosec B608
            f"WHERE {invalid_at_midnight}"
        )
        conn.execute(
            "UPDATE kg_relationships "  # nosec B608
            "SET t_valid = LEAST(t_valid + interval '12 hours', "
            "                    COALESCE(t_invalid, t_valid + interval '12 hours')) "
            f"WHERE {valid_at_midnight}"
        )
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="count the rows, write nothing")
    parser.add_argument("--db-url", default=os.environ.get("SYNAPSE_DB_URL"))
    args = parser.parse_args(argv)
    if not args.db_url:
        parser.error("no database: pass --db-url or set SYNAPSE_DB_URL")
    with psycopg.connect(args.db_url, autocommit=True) as conn:
        valid, invalid = restamp(conn, dry_run=args.dry_run)
    verb = "would move" if args.dry_run else "moved"
    print(
        f"{verb} {valid} t_valid and {invalid} t_invalid values from UTC midnight to noon"
        + (" (dry run, nothing written)" if args.dry_run else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
