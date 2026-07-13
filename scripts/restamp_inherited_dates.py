#!/usr/bin/env python3
"""One-time repair (issue #44): restamp fact/timeline t_valid inherited from
misdated episodes.

Issue #43 repaired episodes.created_at (import wall-clock -> true event time),
but facts and chat timeline events extracted BEFORE that repair inherited
t_valid from the wrong episode dates. This restamps them from the corrected
episodes. Evidence-based dates are left alone, identified by their stamp shape:

- KG facts: an EdgeDateExtractor in-text date parses to date precision
  (midnight UTC); the inherited default (segment max episode created_at)
  carries wall-clock time-of-day. Only NON-midnight edges whose t_valid
  disagrees >1 day with the recomputed max(created_at) of their episode
  backlinks are restamped (t_valid AND valid_at — the insert path sets both).
  Web-artifact edges and midnight (evidence-dated) edges are untouched.
  Multi-mention edges use the same max — later reinforcement episodes can
  bias the recomputed stamp late; still strictly better than an import day.

- Timeline chat events: the gate stamps LLM-resolved dates at NOON UTC and
  same-day events with the full turn timestamp (timeline_gate.py). Only
  non-noon events disagreeing >1 day with their ep:N episode's corrected
  created_at are restamped; noon (LLM-resolved) events are counted and left.

Usage:
    python scripts/restamp_inherited_dates.py [DSN]                    # dry-run, both lanes
    python scripts/restamp_inherited_dates.py [DSN] --facts --apply    # one lane, execute

DSN defaults to $SYNAPSE_DB_URL. Idempotent: restamped rows agree with their
recomputed value on re-run and drop out of the WHERE. Relative-date anchors
baked into fact TEXT by the old reference_time are out of scope (issue #44).
"""

from __future__ import annotations

import os
import sys

import psycopg

# Facts candidate set: recompute the default (max corrected created_at over the
# edge's episode backlinks) and keep rows whose current t_valid is non-midnight
# (not evidence-dated) and >1 day off the recomputed value.
_FACTS_CANDIDATES = """
    SELECT r.id, max(e.created_at) AS expected,
           bool_and(r.mention_count = 1) AS single_mention
    FROM kg_relationships r
    CROSS JOIN LATERAL jsonb_array_elements_text(r.episodes) AS ep(eid)
    JOIN episodes e ON e.id = ep.eid::bigint
    WHERE r.t_valid IS NOT NULL
      AND (r.t_valid AT TIME ZONE 'UTC')
          <> date_trunc('day', r.t_valid AT TIME ZONE 'UTC')
      AND r.web_artifact_id IS NULL
      AND r.episodes IS NOT NULL
    GROUP BY r.id, r.t_valid
    HAVING abs(extract(epoch FROM r.t_valid - max(e.created_at))) > 86400
"""

_TIMELINE_WHERE = """
    te.source = 'chat'
    AND te.source_ref = 'ep:' || e.id
    AND (te.t_valid AT TIME ZONE 'UTC')::time <> '12:00:00'
    AND abs(extract(epoch FROM te.t_valid - e.created_at)) > 86400
"""


def repair_facts(conn: psycopg.Connection, apply: bool) -> None:
    row = conn.execute(
        f"""SELECT count(*), count(*) FILTER (WHERE NOT single_mention),
                   min(expected), max(expected)
            FROM ({_FACTS_CANDIDATES}) c"""
    ).fetchone()
    n, multi, lo, hi = row  # type: ignore[misc]
    print(f"[facts] {n} edge(s) with inherited t_valid >1 day off the corrected")
    print(f"[facts] episode dates ({multi} multi-mention; corrected span {lo} .. {hi})")
    if not n or not apply:
        return
    updated = conn.execute(
        f"""UPDATE kg_relationships r
            SET t_valid = c.expected, valid_at = c.expected
            FROM ({_FACTS_CANDIDATES}) c
            WHERE r.id = c.id"""
    ).rowcount
    print(f"[facts] restamped {updated} edge(s)")


def repair_timeline(conn: psycopg.Connection, apply: bool) -> None:
    n = conn.execute(
        f"SELECT count(*) FROM timeline_events te JOIN episodes e ON true WHERE {_TIMELINE_WHERE}"
    ).fetchone()[0]  # type: ignore[index]
    noon = conn.execute(
        """SELECT count(*) FROM timeline_events te
           JOIN episodes e ON te.source_ref = 'ep:' || e.id
           WHERE te.source = 'chat'
             AND (te.t_valid AT TIME ZONE 'UTC')::time = '12:00:00'
             AND abs(extract(epoch FROM te.t_valid - e.created_at)) > 86400"""
    ).fetchone()[0]  # type: ignore[index]
    print(f"[timeline] {n} chat event(s) with inherited t_valid >1 day off their episode")
    print(f"[timeline] ({noon} noon-stamped LLM-resolved event(s) left alone)")
    if not n or not apply:
        return
    updated = conn.execute(
        f"UPDATE timeline_events te SET t_valid = e.created_at FROM episodes e WHERE {_TIMELINE_WHERE}"
    ).rowcount
    print(f"[timeline] restamped {updated} event(s)")


def main() -> None:
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    unknown = flags - {"--facts", "--timeline", "--apply"}
    if unknown:
        sys.exit(f"unknown flag(s): {', '.join(sorted(unknown))}")
    apply = "--apply" in flags
    # No lane flag = both lanes.
    facts = "--facts" in flags or "--timeline" not in flags
    timeline = "--timeline" in flags or "--facts" not in flags
    dsn = args[0] if args else os.environ.get("SYNAPSE_DB_URL", "")
    if not dsn:
        sys.exit(
            "usage: restamp_inherited_dates.py <DSN> [--facts] [--timeline] [--apply]"
            "  (or set SYNAPSE_DB_URL)"
        )

    with psycopg.connect(dsn, autocommit=True) as conn:
        if facts:
            repair_facts(conn, apply)
        if timeline:
            repair_timeline(conn, apply)
    if not apply:
        print("dry run — pass --apply to restamp")


if __name__ == "__main__":
    main()
