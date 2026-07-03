"""DB-backed test for the dream->config ledger upsert (dream/config/nightly._upsert_correction).

Guards the observe->proposed accumulation against the real config_proposals schema — including the
scope=general default (config_proposals.scope is blast radius local|general, NOT the registry's
global|project axis; conflating them threw a CHECK violation on the first live run).
"""

from __future__ import annotations

from dream.config.nightly import _upsert_correction


def test_upsert_accumulates_and_proposes_with_valid_scope(conn, db_url):
    conn.execute("TRUNCATE config_lane.config_proposals RESTART IDENTITY")
    cur = conn.cursor()

    id1, st1, n1 = _upsert_correction(
        cur, {"rule": "Do not use em-dashes in prose", "session_id": "s1", "quote": "no dashes"}
    )
    assert st1 == "observe" and n1 == 1  # first sighting -> observe

    # a re-phrasing of the same rule in a NEW session -> same row, proposed (2 distinct sessions)
    id2, st2, n2 = _upsert_correction(
        cur, {"rule": "Avoid em-dashes in prose", "session_id": "s2", "quote": "still no dashes"}
    )
    assert id2 == id1 and st2 == "proposed" and n2 == 2

    # same session again -> no double-count, stays proposed at 2
    _, _, n3 = _upsert_correction(
        cur, {"rule": "Avoid em-dashes in prose", "session_id": "s2", "quote": "dup"}
    )
    assert n3 == 2

    row = conn.execute(
        "SELECT scope, kind, file_key, status FROM config_lane.config_proposals WHERE id=%s", (id1,)
    ).fetchone()
    assert row[0] == "general"  # the bug that hit the first live run
    assert row[1] == "add" and row[2] == "rules/learned.md" and row[3] == "proposed"

    # a distinct rule opens its own observe row
    idx, stx, _ = _upsert_correction(
        cur,
        {"rule": "Filter out contract job postings", "session_id": "s3", "quote": "no contract"},
    )
    assert idx != id1 and stx == "observe"
