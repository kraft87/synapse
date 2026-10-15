#!/usr/bin/env python3
"""Fill ``kg_entities.normalized_name`` where it is NULL.

Every entity write stores it (``KGClient.insert_node``), but an entity from before the
column existed has NULL, and the exact-name lookup that decides whether a name is new
reads only that indexed column. It used to fall back to an unindexed
``lower(trim(name))`` scan of the whole group on every miss, which is every genuinely
new name, twice per entity per chunk; that fallback is gone. A NULL row is therefore
invisible to dedup, and a mention of its name would mint a duplicate entity. Run this
once, before or right after deploying the image that dropped the fallback; it is
idempotent and touches only NULL rows.

The value is ``ingestion.dedup._normalize_name(name)``: the same function that writes the
column, so a backfilled row is found exactly like a freshly written one. (A name that
normalizes to the empty string is stored as '' and never matches, as it never did.)

Usage:
    scripts/backfill_normalized_name.py [--dry-run] [--db-url DSN]

The DSN comes from --db-url, else $SYNAPSE_DB_URL. --dry-run only counts. The output is
counts only: entity names stay out of terminals and logs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.dedup import _normalize_name

_BATCH = 1000


def backfill(conn: psycopg.Connection, *, dry_run: bool) -> tuple[int, int]:
    """Returns (rows that had no normalized_name, rows written)."""
    rows = conn.execute("SELECT id, name FROM kg_entities WHERE normalized_name IS NULL").fetchall()
    if dry_run:
        return len(rows), 0
    written = 0
    for start in range(0, len(rows), _BATCH):
        batch = [
            (_normalize_name(name or ""), row_id) for row_id, name in rows[start : start + _BATCH]
        ]
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE kg_entities SET normalized_name = %s WHERE id = %s AND normalized_name IS NULL",
                batch,
            )
            written += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
    return len(rows), written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="count the NULL rows, write nothing")
    parser.add_argument("--db-url", default=os.environ.get("SYNAPSE_DB_URL"))
    args = parser.parse_args(argv)
    if not args.db_url:
        parser.error("no database: pass --db-url or set SYNAPSE_DB_URL")
    with psycopg.connect(args.db_url, autocommit=True) as conn:
        missing, written = backfill(conn, dry_run=args.dry_run)
    if args.dry_run:
        print(f"{missing} entities have no normalized_name (dry run, nothing written)")
    else:
        print(f"{missing} entities had no normalized_name; {written} filled")
    return 0


if __name__ == "__main__":
    sys.exit(main())
