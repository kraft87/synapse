#!/usr/bin/env python3
"""One-time repair: re-stamp episodes.created_at to EVENT time from metadata.ts.

Before the upsert fix (issue #43), the transcript's own timestamp was parsed
into Episode.created_at but dropped at INSERT, so every imported episode got
created_at = import wall-clock. That poisoned served episode dates, recency
ranking, and KG fact t_valid (get_episodes_valid_at). The true timestamp
survived in metadata->>'ts' — this copies it back.

Usage:
    python scripts/backfill_episode_event_time.py [DSN]           # dry-run report
    python scripts/backfill_episode_event_time.py [DSN] --apply   # execute

DSN defaults to $SYNAPSE_DB_URL. Idempotent: rows already matching their
metadata ts are untouched. Only ISO-formatted ts values are trusted.

NOTE: facts/timeline events extracted BEFORE this repair still carry t_valid
inherited from the wrong dates; this script fixes the substrate only.
"""

from __future__ import annotations

import os
import sys

import psycopg

_WHERE = """
    metadata ? 'ts'
    AND (metadata->>'ts') ~ '^20[0-9]{2}-[01][0-9]-[0-3][0-9]T'
    AND abs(extract(epoch FROM created_at - (metadata->>'ts')::timestamptz)) > 60
"""


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--apply"]
    apply = "--apply" in sys.argv
    dsn = args[0] if args else os.environ.get("SYNAPSE_DB_URL", "")
    if not dsn:
        sys.exit("usage: backfill_episode_event_time.py <DSN> [--apply]  (or set SYNAPSE_DB_URL)")

    with psycopg.connect(dsn, autocommit=True) as conn:
        n, lo, hi = conn.execute(  # type: ignore[misc]
            "SELECT count(*), min((metadata->>'ts')::timestamptz), "
            f"max((metadata->>'ts')::timestamptz) FROM episodes WHERE {_WHERE}"
        ).fetchone()
        print(
            f"{n} episode(s) with created_at drifted >60s from metadata.ts (ts span {lo} .. {hi})"
        )
        if not n:
            return
        if not apply:
            print("dry run — pass --apply to re-stamp created_at from metadata.ts")
            return
        updated = conn.execute(
            f"UPDATE episodes SET created_at = (metadata->>'ts')::timestamptz WHERE {_WHERE}"
        ).rowcount
        print(f"re-stamped {updated} episode(s)")


if __name__ == "__main__":
    main()
