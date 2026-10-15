"""DB-backed tests for the dream->config ledger (dream/config/nightly).

Guards the observe->proposed accumulation against the real config_proposals schema — including the
scope=general default (config_proposals.scope is blast radius local|general, NOT the registry's
global|project axis; conflating them threw a CHECK violation on the first live run) — and what a
sighting does to a row the user already decided on, an observe row nobody saw again, and a rule the
user already has in a config file or a feedback note.
"""

from __future__ import annotations

import pytest

from dream.config.nightly import (
    LEARNED_FILE,
    OBSERVE_TTL_DAYS,
    REJECT_COOLING_DAYS,
    _expire_observed,
    _known_rules,
    _rule,
    _same_rule,
    _upsert_correction,
)
from dream.evidence import EVIDENCE_CAP


@pytest.fixture()
def cur(conn):
    conn.execute("TRUNCATE config_lane.config_proposals RESTART IDENTITY")
    return conn.cursor()


def _c(rule, session, quote="q"):
    return {"rule": rule, "session_id": session, "quote": quote}


def _row(conn, pid):
    return conn.execute(
        "SELECT status, jsonb_array_length(evidence), updated_at FROM config_lane.config_proposals "
        "WHERE id=%s",
        (pid,),
    ).fetchone()


def _seed(conn, rule, status, *, days_old=0, sessions=("s1",)):
    """A config_proposals row in `status` whose last change was `days_old` days ago."""
    ev = ",".join(f'{{"session_id": "{s}", "quote": "q"}}' for s in sessions)
    return conn.execute(
        "INSERT INTO config_lane.config_proposals "
        "  (kind, file_key, scope, diff, summary, evidence, status, updated_at) "
        "VALUES ('add', %s, 'general', %s, %s, %s::jsonb, %s, now() - make_interval(days => %s)) "
        "RETURNING id",
        (LEARNED_FILE, rule, rule, f"[{ev}]", status, days_old),
    ).fetchone()[0]


def test_upsert_accumulates_and_proposes_with_valid_scope(conn, cur):
    first = _upsert_correction(cur, _c("Do not use em-dashes in prose", "s1", "no dashes"))
    assert first.status == "observe" and first.sessions == 1  # first sighting -> observe

    # a re-phrasing of the same rule in a NEW session -> same row, proposed (2 distinct sessions)
    second = _upsert_correction(cur, _c("Avoid em-dashes in prose", "s2", "still no dashes"))
    assert second.id == first.id and second.status == "proposed" and second.sessions == 2

    # same session again -> no double-count, stays proposed at 2
    assert _upsert_correction(cur, _c("Avoid em-dashes in prose", "s2", "dup")).sessions == 2

    row = conn.execute(
        "SELECT scope, kind, file_key, status FROM config_lane.config_proposals WHERE id=%s",
        (first.id,),
    ).fetchone()
    assert row[0] == "general"  # the bug that hit the first live run
    assert row[1] == "add" and row[2] == "rules/learned.md" and row[3] == "proposed"

    # a distinct rule opens its own observe row
    other = _upsert_correction(cur, _c("Filter out contract job postings", "s3", "no contract"))
    assert other.id != first.id and other.status == "observe"


def test_only_the_call_that_graduates_a_rule_raises_it(cur):
    # The count behind dream_runs' config_proposals: a rule waiting for review that is seen
    # again must not be counted as raised again.
    assert not _upsert_correction(cur, _c("Do not use em-dashes in prose", "s1")).raised
    assert _upsert_correction(
        cur, _c("Avoid em-dashes in prose", "s2")
    ).raised  # observe -> proposed
    assert not _upsert_correction(cur, _c("Avoid em-dashes in prose", "s3")).raised  # waiting
    assert not _upsert_correction(cur, _c("Avoid em-dashes in prose", "s3")).raised  # same session


def test_a_new_rule_is_raised_at_once_when_the_gate_is_one(cur, monkeypatch):
    monkeypatch.setattr("dream.config.nightly.PROPOSE_SESSIONS", 1)
    r = _upsert_correction(cur, _c("Keep answers short", "s1"))
    assert r.status == "proposed" and r.raised


def test_a_rejected_rule_stays_suppressed_inside_its_cooling_window(conn, cur):
    pid = _seed(conn, "Do not use em-dashes in prose", "rejected", days_old=REJECT_COOLING_DAYS - 5)
    stamp = _row(conn, pid)[2]

    r = _upsert_correction(cur, _c("Avoid em-dashes in prose", "s9"))

    assert (r.id, r.status, r.raised, r.recurred) == (pid, "rejected", False, False)
    status, n_ev, after = _row(conn, pid)
    assert status == "rejected" and n_ev == 2  # the sighting is kept as evidence ...
    assert after == stamp  # ... and does not restart the window it is counted from
    assert conn.execute("SELECT count(*) FROM config_lane.config_proposals").fetchone()[0] == 1


def test_a_rejected_rule_starts_over_after_its_window(conn, cur):
    old = _seed(conn, "Do not use em-dashes in prose", "rejected", days_old=REJECT_COOLING_DAYS + 5)

    r = _upsert_correction(cur, _c("Avoid em-dashes in prose", "s9"))

    # a fresh observe row with only this sighting; the rejection stays on record
    assert r.id != old and r.status == "observe" and r.sessions == 1
    assert _row(conn, old)[0] == "rejected"


@pytest.mark.parametrize("status", ["accepted", "applied"])
def test_a_decided_rule_collects_evidence_and_stays_decided(conn, cur, status):
    pid = _seed(conn, "Do not use em-dashes in prose", status, days_old=3)

    r = _upsert_correction(cur, _c("Avoid em-dashes in prose", "s9"))

    assert (r.id, r.status, r.raised) == (pid, status, False)
    assert r.recurred is (status == "applied")  # only an applied rule can be "not followed"
    assert _row(conn, pid)[:2] == (status, 2)


def test_an_applied_rule_counts_one_recurrence_per_new_session(conn, cur):
    _seed(conn, "Do not use em-dashes in prose", "applied")
    assert _upsert_correction(cur, _c("Avoid em-dashes in prose", "s9")).recurred
    assert not _upsert_correction(cur, _c("Avoid em-dashes in prose", "s9")).recurred  # a re-judge
    assert not _upsert_correction(cur, _c("Avoid em-dashes in prose", "s1")).recurred  # known


def test_an_applied_rule_at_the_evidence_cap_still_counts_a_new_session(conn, cur):
    """The row's evidence is capped like the skills ledger's; a full row drops its oldest
    entry for the new one, so "a session was added" cannot be read off the list length."""
    pid = _seed(
        conn,
        "Do not use em-dashes in prose",
        "applied",
        sessions=[f"s{i}" for i in range(EVIDENCE_CAP)],
    )
    assert _upsert_correction(cur, _c("Avoid em-dashes in prose", "s-new")).recurred
    assert _row(conn, pid)[1] == EVIDENCE_CAP
    assert not _upsert_correction(cur, _c("Avoid em-dashes in prose", "s-new")).recurred


def test_expiry_drops_only_stale_observe_rows(conn, cur):
    stale = _seed(conn, "rule one", "observe", days_old=OBSERVE_TTL_DAYS + 1)
    fresh = _seed(conn, "rule two", "observe", days_old=OBSERVE_TTL_DAYS - 1)
    waiting = _seed(conn, "rule three", "proposed", days_old=OBSERVE_TTL_DAYS + 30)

    assert _expire_observed(cur) == 1

    left = {r[0] for r in conn.execute("SELECT id FROM config_lane.config_proposals").fetchall()}
    assert left == {fresh, waiting} and stale not in left


def test_known_rules_cover_mirrored_lines_and_live_feedback_notes(conn, cur):
    def clean():
        conn.execute("DELETE FROM config_lane.config_registry WHERE surface_id = 'knownt'")
        conn.execute("DELETE FROM notes WHERE owner_id = 'knownt'")

    clean()
    for key, body in (
        (
            "CLAUDE.md",
            "# Style\n- Never use em-dashes in prose\n\nRun the tests before committing\n"
            "## Commit messages\nCode style:\n- Commit messages\n- Code style",
        ),
        (LEARNED_FILE, "- Prefer tabs over spaces"),  # the lane's own file: recurrence, not known
    ):
        conn.execute(
            "INSERT INTO config_lane.config_registry "
            "  (surface_id, scope, file_key, abs_path, content, content_hash) "
            "VALUES ('knownt', 'global', %s, '', %s, %s)",
            (key, body, key),
        )
    for hook in ("Answer in one paragraph unless asked", "Old advice nobody follows"):
        nid = conn.execute(
            "INSERT INTO notes (owner_id, group_id, type, hook, body) "
            "VALUES ('knownt', 'knownt', 'feedback', %s, '') RETURNING id",
            (hook,),
        ).fetchone()[0]
    conn.execute("UPDATE notes SET superseded_by = id WHERE id = %s", (nid,))  # the second: retired
    try:
        known = _known_rules(cur)
    finally:
        clean()

    def is_known(rule):
        return any(_same_rule(_rule(rule), k) for k in known)

    assert is_known("Never use em-dashes in prose")
    assert is_known("Answer in one paragraph unless asked")
    assert not is_known("Prefer tabs over spaces")  # learned.md itself
    assert not is_known("Old advice nobody follows")  # a retired note
    # headings, labels and two-word lines are topics, not rules: they restate nothing
    assert not is_known("Keep commit messages short")
    assert not is_known("Match the code style of the file")
    # a user overriding a line of their own config is the rule the lane is here to surface
    assert is_known("Run the tests before committing")
    assert not is_known("Do not run the tests before committing")
