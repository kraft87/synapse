"""DB-backed tests for the dream->config ledger (dream/config/nightly: _upsert_correction,
ensure_embeddings, regate, run_lane).

Guards the observe->proposed accumulation against the real config_proposals schema (065),
including the scope=general default (config_proposals.scope is blast radius local|general, NOT
the registry's global|project axis; conflating them threw a CHECK violation on the first live
run). Identity is by embedding: the tests build unit vectors with a chosen cosine to an anchor,
so no embedding service is called, and stub the judge.
"""

from __future__ import annotations

import json
import math

import pytest

from dream.config import nightly as N

DIMS = 2048


def vec(i: int, cos: float = 1.0, k: int = 1500) -> list[float]:
    """A unit vector whose cosine to basis vector e_i is `cos` (the rest on e_k)."""
    v = [0.0] * DIMS
    v[i] = cos
    if cos < 1.0:
        v[k] = math.sqrt(1.0 - cos * cos)
    return v


def corr(rule, session, v, quote="q"):
    return {"rule": rule, "session_id": session, "quote": quote, "vec": v}


@pytest.fixture()
def lane(conn):
    conn.execute("TRUNCATE config_lane.config_proposals RESTART IDENTITY")
    conn.execute("TRUNCATE config_lane.config_registry")
    return conn


def row(conn, rid):
    return conn.execute(
        "SELECT status, reject_reason, evidence, proposed_at, coverage, scope, kind, file_key "
        "FROM config_lane.config_proposals WHERE id=%s",
        (rid,),
    ).fetchone()


def judge(covered=False, calls=None):
    def _call(prompt):
        if calls is not None:
            calls.append(prompt)
        return json.dumps({"covered": covered, "by": "an existing rule"})

    return _call


def test_rephrasings_across_sessions_are_one_row_and_get_proposed(lane, monkeypatch):
    """The reason the lane never produced a proposal: two sessions' wordings of one rule share
    almost no tokens, so token overlap never matched. By embedding they are one row."""
    cur = lane.cursor()
    id1, st1, n1 = N._upsert_correction(
        cur, corr("Do not use em-dashes in prose", "s1", vec(0), "no dashes")
    )
    assert st1 == "observe" and n1 == 1
    id2, _, n2 = N._upsert_correction(
        cur, corr("Punctuate with commas instead", "s2", vec(0, 0.9), "still no dashes")
    )
    assert id2 == id1 and n2 == 2  # one row, two distinct sessions
    # same session again: no double count
    _, _, n3 = N._upsert_correction(cur, corr("Punctuate with commas", "s2", vec(0, 0.9), "dup"))
    assert n3 == 2
    # the gate runs in regate, after the coverage check
    assert row(lane, id1)[0] == "observe"
    monkeypatch.setattr(N, "_judge_call", judge(covered=False))
    out = N.regate(lane)
    assert [p["id"] for p in out["proposed"]] == [id1]
    status, _, ev, proposed_at, coverage, scope, kind, file_key = row(lane, id1)
    assert status == "proposed" and proposed_at is not None
    assert scope == "general" and kind == "add" and file_key == "rules/learned.md"
    assert [e["session_id"] for e in ev] == ["s1", "s2"]
    assert coverage["covered"] is False

    # a distinct rule opens its own observe row
    idx, stx, _ = N._upsert_correction(cur, corr("Filter out contract postings", "s3", vec(7)))
    assert idx != id1 and stx == "observe"


def test_same_session_paraphrase_updates_the_row(lane):
    """A long session is re-scanned each night and worded differently each time: the row that
    already counts the session absorbs it (lower bar) instead of a near-duplicate row."""
    cur = lane.cursor()
    id1, _, _ = N._upsert_correction(cur, corr("Rule as first worded", "s1", vec(0)))
    id2, st, n = N._upsert_correction(cur, corr("Rule, reworded", "s1", vec(0, 0.78)))
    assert id2 == id1 and st == "observe" and n == 1
    assert lane.execute("SELECT count(*) FROM config_lane.config_proposals").fetchone()[0] == 1
    # the same cosine from ANOTHER session is a different rule
    id3, _, _ = N._upsert_correction(cur, corr("Rule, reworded", "s2", vec(0, 0.78)))
    assert id3 != id1


def test_anchor_does_not_drift(lane):
    """Rows compare to their founding embedding, so a chain of near-paraphrases (each close to
    the last, the ends far apart) does not merge into one rule."""
    cur = lane.cursor()
    # anchor e0; b is 0.9 from it; c is 0.90 from b but only 0.70 from the anchor
    a, _, _ = N._upsert_correction(cur, corr("A", "s1", vec(0)))
    b, _, _ = N._upsert_correction(cur, corr("B", "s2", vec(0, 0.9, k=1500)))
    assert b == a
    c_vec = [0.0] * DIMS
    c_vec[0], c_vec[1500] = 0.70, 0.62
    c_vec[1600] = math.sqrt(1 - 0.70**2 - 0.62**2)
    c, _, _ = N._upsert_correction(cur, corr("C", "s3", c_vec))
    assert c != a


def test_a_rule_held_open_keeps_one_row_per_rule(lane):
    cur = lane.cursor()
    id1, _, _ = N._upsert_correction(cur, corr("Rule", "s1", vec(0)))
    N._upsert_correction(cur, corr("Rule again", "s2", vec(0, 0.95)))
    N._upsert_correction(cur, corr("Rule thrice", "s3", vec(0, 0.95, k=1501)))
    n = lane.execute("SELECT count(*) FROM config_lane.config_proposals").fetchone()[0]
    assert n == 1 and len(row(lane, id1)[2]) == 3


def test_rejected_rule_absorbs_sightings_during_cooldown_only(lane):
    cur = lane.cursor()
    rid, _, _ = N._upsert_correction(cur, corr("Rule", "s1", vec(0)))
    lane.execute(
        "UPDATE config_lane.config_proposals SET status='rejected', reject_reason='no thanks', "
        "updated_at = now() - interval '5 days' WHERE id=%s",
        (rid,),
    )
    again, status, n = N._upsert_correction(cur, corr("Rule", "s2", vec(0, 0.95)))
    assert again == rid and status == "rejected" and n == 2  # absorbed, still rejected
    st, _, ev, *_ = row(lane, rid)
    assert st == "rejected" and len(ev) == 2
    stamp = lane.execute(
        "SELECT updated_at < now() - interval '4 days' FROM config_lane.config_proposals WHERE id=%s",
        (rid,),
    ).fetchone()[0]
    assert stamp  # absorbing did not restart the cooldown clock
    # past the cooldown a sighting starts a fresh row
    lane.execute(
        "UPDATE config_lane.config_proposals SET updated_at = now() - interval '40 days' WHERE id=%s",
        (rid,),
    )
    fresh, status, _ = N._upsert_correction(cur, corr("Rule", "s3", vec(0, 0.95, k=1501)))
    assert fresh != rid and status == "observe"


def test_applied_rule_absorbs_sightings(lane):
    cur = lane.cursor()
    rid, _, _ = N._upsert_correction(cur, corr("Rule", "s1", vec(0)))
    lane.execute("UPDATE config_lane.config_proposals SET status='applied' WHERE id=%s", (rid,))
    again, status, _ = N._upsert_correction(cur, corr("Rule", "s2", vec(0, 0.95)))
    assert again == rid and status == "applied"
    assert lane.execute("SELECT count(*) FROM config_lane.config_proposals").fetchone()[0] == 1


def test_expired_proposal_reopens_on_a_new_session_only(lane):
    cur = lane.cursor()
    rid, _, _ = N._upsert_correction(cur, corr("Rule", "s1", vec(0)))
    N._upsert_correction(cur, corr("Rule", "s2", vec(0, 0.95)))
    lane.execute(
        "UPDATE config_lane.config_proposals SET status='rejected', reject_reason='expired', "
        "recommendation='{\"verdict\":\"accept\"}'::jsonb, updated_at = now() - interval '3 days' "
        "WHERE id=%s",
        (rid,),
    )
    # a repeat from a session the row already counts changes nothing
    _, status, _ = N._upsert_correction(cur, corr("Rule", "s2", vec(0, 0.9)))
    assert status == "rejected"
    # a new session reopens it (to observe: regate re-checks it against the rule files)
    again, status, n = N._upsert_correction(cur, corr("Rule", "s3", vec(0, 0.9, k=1501)))
    assert again == rid and status == "observe" and n == 3
    st, reason, _, _, _, *_ = row(lane, rid)
    assert st == "observe" and reason is None
    rec = lane.execute(
        "SELECT recommendation FROM config_lane.config_proposals WHERE id=%s", (rid,)
    ).fetchone()[0]
    assert rec is None  # the old recommendation was for other evidence


def test_ensure_embeddings_fills_rows_without_an_anchor(lane, monkeypatch):
    lane.execute(
        "INSERT INTO config_lane.config_proposals (kind, file_key, diff, summary, evidence, status) "
        "VALUES ('add','rules/learned.md','r1','r1','[{\"session_id\":\"s1\"}]','observe'),"
        "       ('add','rules/learned.md','r2','r2','[{\"session_id\":\"s2\"}]','observe')"
    )
    seen = []

    def fake_embed(texts):
        seen.append(list(texts))
        return [vec(0), vec(5)]

    monkeypatch.setattr(N, "embed", fake_embed)
    assert N.ensure_embeddings(lane) == 2 and seen == [["r1", "r2"]]
    assert (
        lane.execute(
            "SELECT count(*) FROM config_lane.config_proposals WHERE embedding IS NULL"
        ).fetchone()[0]
        == 0
    )
    assert N.ensure_embeddings(lane) == 0  # nothing left: no second call
    assert len(seen) == 1


# ---------------------------------------------------------------- coverage gate
def _two_session_row(lane, rule="Rule"):
    cur = lane.cursor()
    rid, _, _ = N._upsert_correction(cur, corr(rule, "s1", vec(0)))
    N._upsert_correction(cur, corr(rule, "s2", vec(0, 0.95)))
    return rid


def _registry(conn, rows):
    for surface, scope, key, content in rows:
        conn.execute(
            "INSERT INTO config_lane.config_registry "
            "(surface_id, scope, file_key, abs_path, content, content_hash) "
            "VALUES (%s,%s,%s,'/x',%s,md5(%s))",
            (surface, scope, key, content, content),
        )


def test_registry_digest_is_the_distinct_global_files(lane):
    _registry(
        lane,
        [
            ("m1", "global", "rules/a.md", "Be brief."),
            (
                "m2",
                "global",
                "rules/a.md",
                "Be brief.",
            ),  # the same file mirrored by a second machine
            ("m1", "project:x", ".claude/rules/a.md", "Be brief."),  # a project copy: not global
            ("m1", "project:x", "CLAUDE.md", "Project only rule"),
        ],
    )
    text, h = N.registry_digest(lane.cursor())
    assert text.count("Be brief.") == 1 and "Project only rule" not in text and h
    lane.execute("TRUNCATE config_lane.config_registry")
    assert N.registry_digest(lane.cursor()) == ("", "")


def test_a_rule_the_user_already_has_is_dropped_not_proposed(lane, monkeypatch):
    rid = _two_session_row(lane)
    _registry(lane, [("m1", "global", "rules/a.md", "Be brief.")])
    calls: list[str] = []
    monkeypatch.setattr(N, "_judge_call", judge(covered=True, calls=calls))
    out = N.regate(lane)
    assert out["proposed"] == [] and [c["id"] for c in out["covered"]] == [rid]
    assert out["covered"][0]["covered_by"] == "an existing rule"
    status, reason, _, proposed_at, coverage, *_ = row(lane, rid)
    assert (status, reason) == ("rejected", "auto:covered") and proposed_at is None
    assert coverage["covered"] is True and coverage["registry_hash"]
    assert "Be brief." in calls[0] and "PROPOSED RULE: Rule" in calls[0]  # sees the user's rules


def test_no_rule_files_means_no_judge_call(lane, monkeypatch):
    rid = _two_session_row(lane)
    calls: list[str] = []
    monkeypatch.setattr(N, "_judge_call", judge(calls=calls))
    out = N.regate(lane)
    assert [p["id"] for p in out["proposed"]] == [rid] and calls == []


def test_a_failed_coverage_call_leaves_the_row_at_observe(lane, monkeypatch):
    rid = _two_session_row(lane)
    _registry(lane, [("m1", "global", "rules/a.md", "Be brief.")])

    def boom(prompt):
        raise RuntimeError("claude CLI not logged in")

    monkeypatch.setattr(N, "_judge_call", boom)
    out = N.regate(lane)
    assert out["failed"] == 1 and out["proposed"] == []
    assert row(lane, rid)[0] == "observe"
    # an unparseable answer is a failure too
    monkeypatch.setattr(N, "_judge_call", lambda p: "sure, probably")
    assert N.regate(lane)["failed"] == 1 and row(lane, rid)[0] == "observe"
    # tomorrow it works
    monkeypatch.setattr(N, "_judge_call", judge(covered=False))
    assert [p["id"] for p in N.regate(lane)["proposed"]] == [rid]


def test_proposed_rows_are_rechecked_only_when_the_rule_files_change(lane, monkeypatch):
    rid = _two_session_row(lane)
    _registry(lane, [("m1", "global", "rules/a.md", "Be brief.")])
    calls: list[str] = []
    monkeypatch.setattr(N, "_judge_call", judge(covered=False, calls=calls))
    N.regate(lane)
    assert len(calls) == 1 and row(lane, rid)[0] == "proposed"
    N.regate(lane)
    assert len(calls) == 1  # same files: memoized, no call
    # the user writes the rule by hand while the proposal waits
    _registry(lane, [("m1", "global", "rules/b.md", "Rule, exactly.")])
    monkeypatch.setattr(N, "_judge_call", judge(covered=True, calls=calls))
    out = N.regate(lane)
    assert len(calls) == 2 and [c["id"] for c in out["covered"]] == [rid]
    assert row(lane, rid)[:2] == ("rejected", "auto:covered")


def test_coverage_calls_are_capped_per_night(lane, monkeypatch):
    ids = []
    cur = lane.cursor()
    for i in range(3):
        rid, _, _ = N._upsert_correction(cur, corr(f"Rule {i}", "s1", vec(i * 10)))
        N._upsert_correction(cur, corr(f"Rule {i}", "s2", vec(i * 10, 0.95)))
        ids.append(rid)
    _registry(lane, [("m1", "global", "rules/a.md", "Be brief.")])
    monkeypatch.setattr(N, "_judge_call", judge(covered=False))
    out = N.regate(lane, cap=2)
    assert len(out["proposed"]) == 2 and out["deferred"] == 1
    assert [row(lane, i)[0] for i in ids].count("observe") == 1


# ---------------------------------------------------------------- the orchestrator
def _views():
    return [{"session": "sess-a", "user_msgs": ["m"]}, {"session": "sess-b", "user_msgs": ["m"]}]


def test_run_lane_end_to_end(lane, monkeypatch):
    lane.execute("UPDATE config_lane.config_scan_cursor SET last_scan_at = NULL")
    lane.execute('UPDATE skills_lane.skill_scan_cursor SET config = \'{"last_report": "keep me"}\'')
    found = {
        "sess-a": [{"rule": "Wording one", "session_id": "sess-a", "quote": "qa"}],
        "sess-b": [{"rule": "Wording two", "session_id": "sess-b", "quote": "qb"}],
    }
    vecs = {"Wording one": vec(0), "Wording two": vec(0, 0.92)}
    monkeypatch.setattr(N.DB, "sessions_since", lambda last, max_sessions: _views())
    monkeypatch.setattr(N, "scan_corrections", lambda s: [dict(c) for c in found[s["session"]]])
    monkeypatch.setattr(N, "embed", lambda texts: [vecs[t] for t in texts])
    monkeypatch.setattr(
        N,
        "_judge_call",
        lambda p: json.dumps(
            {"verdict": "accept", "reason": "a standing preference"}
            if "PROPOSED RULE #" in p
            else {"covered": False, "by": ""}
        ),
    )
    res = N.run_lane(limit=5, no_discord=True)
    assert res["sessions"] == 2 and res["found"] == 2 and res["proposed"] == 1
    assert res["recommended"] == 1 and res["covered"] == 0 and res["expired"] == 0
    r = lane.execute(
        "SELECT status, jsonb_array_length(evidence), recommendation->>'verdict' "
        "FROM config_lane.config_proposals"
    ).fetchall()
    assert r == [("proposed", 2, "accept")]
    # the watermark moved, the report is stored next to the skills lane's own key
    assert (
        lane.execute("SELECT last_scan_at FROM config_lane.config_scan_cursor").fetchone()[0]
        is not None
    )
    cfg = lane.execute("SELECT config FROM skills_lane.skill_scan_cursor WHERE id=1").fetchone()[0]
    assert cfg["last_report"] == "keep me"  # the skills lane's key is untouched
    assert "1 new" in cfg["config_last_report"]["text"] and cfg["config_report_sent_at"]
    # tomorrow: nothing new, nothing re-proposed
    again = N.run_lane(limit=5, no_discord=True)
    assert again["proposed"] == 0
    assert lane.execute("SELECT count(*) FROM config_lane.config_proposals").fetchone()[0] == 1


def test_run_lane_idles_until_schema_065(lane, monkeypatch):
    import dream.config as config_pkg
    from ingestion.schema_check import ColumnProbe

    lane.execute("UPDATE config_lane.config_scan_cursor SET last_scan_at = NULL")
    monkeypatch.setattr(
        config_pkg,
        "IDENTITY_PROBE",
        ColumnProbe("SELECT 1 WHERE false", "schema 065 (test: absent)"),
    )

    def must_not_run(*a, **k):
        raise AssertionError("the lane must not scan before schema 065")

    monkeypatch.setattr(N.DB, "sessions_since", must_not_run)
    assert N.run_lane(limit=5, no_discord=True) == {"skipped": "schema 065 not applied"}
    assert (
        lane.execute("SELECT last_scan_at FROM config_lane.config_scan_cursor").fetchone()[0]
        is None
    )  # nothing is skipped when it starts


def test_an_embedding_failure_aborts_the_run_and_keeps_the_watermark(lane, monkeypatch):
    lane.execute("UPDATE config_lane.config_scan_cursor SET last_scan_at = NULL")
    monkeypatch.setattr(N.DB, "sessions_since", lambda last, max_sessions: _views()[:1])
    monkeypatch.setattr(
        N,
        "scan_corrections",
        lambda s: [{"rule": "A rule", "session_id": "sess-a", "quote": "q"}],
    )

    def down(texts):
        raise RuntimeError("embedding service unavailable")

    monkeypatch.setattr(N, "embed", down)
    with pytest.raises(RuntimeError, match="unavailable"):
        N.run_lane(limit=5, no_discord=True)
    assert lane.execute("SELECT count(*) FROM config_lane.config_proposals").fetchone()[0] == 0
    assert (
        lane.execute("SELECT last_scan_at FROM config_lane.config_scan_cursor").fetchone()[0]
        is None
    )
