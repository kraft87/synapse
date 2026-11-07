"""The skills autopilot against a real database — schema 064 + autoreview + the routes' writes.

Runs against the shared Postgres test DB (tagged into the `db` xdist group via conftest). The
LLM seams (reviewer, verifier) are stubbed; everything else is the production code path.
Every skill, session and candidate is synthetic (ap- prefix) and cleaned up around each test.

Covers: auto apply of a new skill and of a description-only change (one apply_candidate, the
audit row, lane_body_hash, no grounded entry, skill_history keeps the old body), the table
constraint, the memo (a row is judged once per evidence+draft; shadow -> live reuses the
judgement), the caps, what stays a person's, auto-reject, revert, automatic rollback and the
circuit breaker, the sync race in both orders, and the report (auto lines first, persisted).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import psycopg
import pytest
from psycopg.errors import CheckViolation
from psycopg.types.json import Json

import dream.skills.autoreview as A
import dream.skills.review_queue as RQ
import dream.skills.skill_ledger as L
from dream.skills.skill_doc import body_hash, replace_description
from mcp_server.skill_sync_routes import (
    _autopilot,
    _proposal_act,
    _publish_skill,
    _record_overwrite,
)

OLD_TIME = "2026-01-01T00:00:00+00:00"
FUTURE = "2099-01-01T00:00:00+00:00"


def _skill(name, desc, step="Do the thing."):
    return f"---\nname: {name}\ndescription: {desc}\n---\n# {name}\n\n1. {step}\n"


def _wide(body, desc):
    return replace_description(body, desc)


@pytest.fixture()
def db(conn, db_url):
    conn.dsn_url = db_url  # the routes open their own connections; info.dsn drops the password

    def _clean():
        conn.execute("DELETE FROM skills_lane.skill_usage WHERE skill LIKE 'ap-%%'")
        conn.execute("DELETE FROM skills_lane.skill_gap_candidates WHERE name LIKE 'ap-%%'")
        conn.execute("DELETE FROM skills_lane.skill_auto_reviews WHERE skill LIKE 'ap-%%'")
        conn.execute("DELETE FROM skills_lane.skill_registry WHERE name LIKE 'ap-%%'")
        conn.execute("DELETE FROM skills_lane.skill_history WHERE name LIKE 'ap-%%'")
        conn.execute(
            "UPDATE skills_lane.skill_scan_cursor SET config = config - 'last_report' "
            "- 'autopilot' - 'report_sent_at' WHERE id=1"
        )

    _clean()
    conn.execute(
        "UPDATE skills_lane.skill_gap_candidates SET status='observe' WHERE status='proposed'"
    )
    yield conn
    _clean()


@pytest.fixture()
def ap(db_url, db):
    """The lane's own kind of connection: plain, non-autocommit."""
    with psycopg.connect(db_url) as c:
        yield c


class LLM:
    """The reviewer and verifier, stubbed. verifier: 'clean' | 'flag' | 'split' (clean in the
    first order, flagged in the second)."""

    def __init__(self, mp, rev="accept", ver="clean"):
        self.rev, self.ver = rev, ver
        self.n_rev = self.n_ver = 0
        mp.setattr(A, "_reviewer_call", self.reviewer)
        mp.setattr(A, "_verifier_call", self.verifier)
        mp.setattr(A, "verifier_model", lambda: "other/model")

    def reviewer(self, _prompt):
        self.n_rev += 1
        return json.dumps({"verdict": self.rev, "reason": "grounded and reusable"})

    def verifier(self, _prompt):
        self.n_ver += 1
        bad = self.ver == "flag" or (self.ver == "split" and self.n_ver % 2 == 0)
        return json.dumps(
            {
                "traces_to_evidence": not bad,
                "invents": False,
                "steals_from": "",
                "negative_list": False,
                "note": "n",
            }
        )


@pytest.fixture()
def llm(monkeypatch):
    return LLM(monkeypatch)


def _registry(conn, name, body, *, scope="global", pinned=False, lane=False):
    conn.execute(
        "INSERT INTO skills_lane.skill_registry (name, scope, status, body, description, pinned, "
        "lane_body_hash, content_modified_at) VALUES (%s,%s,'active',%s,%s,%s,%s,%s)",
        (
            name,
            scope,
            body,
            body.split("description: ")[1].split("\n")[0],
            pinned,
            body_hash(body) if lane else None,
            OLD_TIME,
        ),
    )


def _two_days(prefix="ap-s"):
    return [
        {"session_id": f"{prefix}1", "class": "judge", "signal": "gap", "date": "2026-11-01"},
        {"session_id": f"{prefix}2", "class": "judge", "signal": "gap", "date": "2026-11-03"},
    ]


def _under(sid, missing="restore it"):
    return {
        "session_id": sid,
        "class": "judge",
        "signal": "under_trigger",
        "missing_phrasing": missing,
    }


def _cand(conn, *, name, body, kind="retune", direction="widen", base=None, evidence=None):
    ev = evidence if evidence is not None else [_under("ap-s1")]
    target = [name] if kind == "retune" else []
    return conn.execute(
        "INSERT INTO skills_lane.skill_gap_candidates (kind, name, direction, target_skills, "
        "status, summary, evidence, proposal_body, base_body_hash, judge_sessions, judge_weight, "
        "proposed_at) VALUES (%s,%s,%s,%s,'proposed','synthetic',%s,%s,%s,1,1.0, now()) "
        "RETURNING id",
        (kind, name, direction if kind == "retune" else None, target, Json(ev), body, base),
    ).fetchone()[0]


def _desc_case(conn, name):
    """A registry skill and a widen draft that changes only its description."""
    old = _skill(name, "Back up the demo database.")
    _registry(conn, name, old)
    return _cand(
        conn,
        name=name,
        body=_wide(old, "Back up or restore the demo database."),
        base=body_hash(old),
    ), old


def _new_case(conn, name):
    body = _skill(name, "Rotate the demo logs.")
    return _cand(conn, name=name, body=body, kind="derive", evidence=_two_days()), body


def _cand_row(conn, cid, cols="status, applied_by, auto_review_id, reject_reason, evidence"):
    return conn.execute(
        f"SELECT {cols} FROM skills_lane.skill_gap_candidates WHERE id=%s", (cid,)
    ).fetchone()


def _reg(conn, name):
    return conn.execute(
        "SELECT body, status, lane_body_hash FROM skills_lane.skill_registry WHERE name=%s",
        (name,),
    ).fetchone()


def _reviews(conn, cid=None):
    where = "WHERE candidate_id=%s" if cid else ""
    return conn.execute(
        "SELECT id, verdict, tier, mode, reviewer, verifier, previous_body_hash, "
        f"applied_body_hash, history_id, rolled_back_at, rollback_reason FROM "
        f"skills_lane.skill_auto_reviews {where} ORDER BY id",
        (cid,) if cid else (),
    ).fetchall()


def _grounded(evidence):
    return [e for e in evidence if e.get("class") == "grounded"]


# ------------------------------------------------------------------ apply
def test_a_new_skill_that_passed_the_gate_is_applied_without_a_grounded_entry(db, ap, llm):
    cid, body = _new_case(db, "ap-logrotate")

    res = A.run(ap, mode="live")

    assert [a["id"] for a in res["applied"]] == [cid]
    status, by, rid, _reason, ev = _cand_row(db, cid)
    assert (status, by) == ("promoted", "auto") and rid is not None
    assert _grounded(ev) == []  # the auto path never writes a grounded entry
    got, reg_status, lane = _reg(db, "ap-logrotate")
    assert (got, reg_status, lane) == (body, "active", body_hash(body))
    (r,) = _reviews(db, cid)
    assert (r[1], r[2], r[3]) == ("applied", "new_skill", "live")
    assert r[4]["verdict"] == "accept" and r[5]["result"] == "pass"  # reviewer + both orders
    assert r[6] is None and r[7] == body_hash(body) and r[8] is None  # a new skill replaced nothing


def test_a_description_only_change_is_applied_and_the_old_body_stays_in_history(db, ap, llm):
    cid, old = _desc_case(db, "ap-backup")

    res = A.run(ap, mode="live")

    assert [a["tier"] for a in res["applied"]] == ["description"]
    new_body = _reg(db, "ap-backup")[0]
    assert "or restore" in new_body and new_body != old
    (r,) = _reviews(db, cid)
    assert (r[6], r[7]) == (body_hash(old), body_hash(new_body))
    hist = db.execute(
        "SELECT body, op FROM skills_lane.skill_history WHERE id=%s", (r[8],)
    ).fetchone()
    assert hist == (old, "superseded")  # what an undo restores from
    assert _cand_row(db, cid, "applied_by")[0] == "auto"


def test_the_constraint_wants_a_grounded_weight_or_a_recorded_auto_review(db):
    cid = _cand(db, name="ap-con", body=_skill("ap-con", "Do it."), kind="derive")
    bare = "UPDATE skills_lane.skill_gap_candidates SET status='promoted' WHERE id=%s"
    with pytest.raises(CheckViolation):
        db.execute(bare, (cid,))
    with pytest.raises(CheckViolation):  # claiming auto without a review row is not enough
        db.execute(
            "UPDATE skills_lane.skill_gap_candidates SET status='promoted', applied_by='auto' "
            "WHERE id=%s",
            (cid,),
        )
    rid = db.execute(
        "INSERT INTO skills_lane.skill_auto_reviews (candidate_id, night, mode, evidence_hash, "
        "tier, verdict) VALUES (%s, current_date, 'live', 'h', 'new_skill', 'applied') RETURNING id",
        (cid,),
    ).fetchone()[0]
    db.execute(
        "UPDATE skills_lane.skill_gap_candidates SET status='promoted', applied_by='auto', "
        "auto_review_id=%s WHERE id=%s",
        (rid, cid),
    )
    other = _cand(db, name="ap-con2", body=_skill("ap-con2", "Do it."), kind="derive")
    db.execute(  # a person's grounded weight still satisfies it
        "UPDATE skills_lane.skill_gap_candidates SET status='promoted', grounded_weight=3, "
        "applied_by='human' WHERE id=%s",
        (other,),
    )


def test_a_human_accept_records_who_and_the_lane_body(db):
    cid, _old = _desc_case(db, "ap-human")

    res = _proposal_act(db.dsn_url, cid, "accept", None, lambda *a: "")

    assert res["status"] == "promoted"
    status, by, rid, _r, ev = _cand_row(db, cid)
    assert (status, by, rid) == ("promoted", "human", None)
    assert len(_grounded(ev)) == 1
    assert _reg(db, "ap-human")[2] == body_hash(_reg(db, "ap-human")[0])


def test_the_auto_path_refuses_a_row_a_person_already_decided(db, ap, llm):
    cid, _ = _new_case(db, "ap-theirs")
    ev = [*_two_days(), {"session_id": None, "class": "grounded", "signal": "revive"}]
    db.execute(
        "UPDATE skills_lane.skill_gap_candidates SET evidence=%s WHERE id=%s", (Json(ev), cid)
    )
    res = A.run(ap, mode="live")
    assert res["applied"] == [] and llm.n_rev == 0  # never even judged


# ------------------------------------------------------------------ the policy
def test_a_body_edit_to_a_hand_written_skill_is_left_for_a_person_with_a_recommendation(
    db, ap, llm
):
    old = _skill("ap-edit", "Back up the demo database.")
    _registry(db, "ap-edit", old)  # no lane_body_hash: hand-written
    edited = old.replace("1. Do the thing.", "1. Do the thing.\n2. Verify it.")
    cid = _cand(db, name="ap-edit", body=edited, direction="extend", base=body_hash(old))

    res = A.run(ap, mode="live")

    assert res["applied"] == [] and res["asked"] == [cid]
    assert _cand_row(db, cid, "status")[0] == "proposed"
    assert _reg(db, "ap-edit")[0] == old
    rec = db.execute(
        "SELECT recommendation FROM skills_lane.skill_gap_candidates WHERE id=%s", (cid,)
    ).fetchone()[0]
    assert rec["verdict"] == "accept" and rec["source"] == "autopilot"
    assert "hand-written" in rec["reason"]
    assert _reviews(db, cid)[0][1] == "ask"


def test_a_pinned_skill_asks(db, ap, llm):
    cid, _old = _desc_case(db, "ap-pin")
    db.execute("UPDATE skills_lane.skill_registry SET pinned=true WHERE name='ap-pin'")
    res = A.run(ap, mode="live")
    assert res["applied"] == [] and _cand_row(db, cid, "status")[0] == "proposed"
    assert _reviews(db, cid)[0][2] == "pinned"


def test_a_merge_and_a_derive_without_recurrence_ask(db, ap, llm):
    body = _skill("ap-once", "Do once.")
    same_day = [
        {"session_id": "ap-s1", "class": "judge", "signal": "gap", "date": "2026-11-01"},
        {"session_id": "ap-s2", "class": "judge", "signal": "gap", "date": "2026-11-01"},
    ]
    cid = _cand(db, name="ap-once", body=body, kind="derive", evidence=same_day)
    res = A.run(ap, mode="live")
    assert res["applied"] == [] and _cand_row(db, cid, "status")[0] == "proposed"
    assert _reviews(db, cid)[0][2] == "derive_no_recurrence"


@pytest.mark.parametrize("ver", ["flag", "split"])
def test_reviewer_and_verifier_disagreeing_asks(db, ap, monkeypatch, ver):
    LLM(monkeypatch, rev="accept", ver=ver)
    cid, _ = _desc_case(db, "ap-disagree")
    res = A.run(ap, mode="live")
    assert res["applied"] == [] and res["rejected"] == []
    assert _cand_row(db, cid, "status")[0] == "proposed"
    assert (
        "verifier"
        in db.execute(
            "SELECT recommendation->>'reason' FROM skills_lane.skill_gap_candidates WHERE id=%s",
            (cid,),
        ).fetchone()[0]
    )


def test_reviewer_reject_and_verifier_agreeing_auto_rejects_without_a_grounded_minus(
    db, ap, monkeypatch
):
    LLM(monkeypatch, rev="reject", ver="flag")
    cid, old = _desc_case(db, "ap-nope")

    res = A.run(ap, mode="live")

    assert [r["id"] for r in res["rejected"]] == [cid]
    status, _by, _rid, reason, ev = _cand_row(db, cid)
    assert (status, reason) == ("rejected", "auto:review")
    assert _grounded(ev) == []  # as #229's auto-reject: no -3, the usual 30-day cooldown
    until = db.execute(
        "SELECT rejected_until - now() FROM skills_lane.skill_gap_candidates WHERE id=%s", (cid,)
    ).fetchone()[0]
    assert 29 <= until.days <= 30
    assert _reg(db, "ap-nope")[0] == old


def test_off_does_nothing_and_a_missing_schema_skips(db, ap, llm, monkeypatch):
    cid, _ = _desc_case(db, "ap-off")
    assert A.run(ap, mode="off")["ran"] is False and llm.n_rev == 0
    monkeypatch.setattr(L, "autopilot_ready", lambda conn: False)
    assert A.run(ap, mode="live")["skipped"] == "schema 064"
    assert _cand_row(db, cid, "status")[0] == "proposed"


# ------------------------------------------------------------------ shadow, memo
def test_shadow_records_what_it_would_do_and_changes_nothing(db, ap, llm):
    cid, old = _desc_case(db, "ap-shadow")
    res = A.run(ap, mode="shadow")
    assert res["would"] == {"would_apply": 1} and res["applied"] == []
    assert _cand_row(db, cid, "status")[0] == "proposed" and _reg(db, "ap-shadow")[0] == old
    assert [r[1] for r in _reviews(db, cid)] == ["would_apply"]


def test_a_row_is_judged_once_and_a_shadow_judgement_is_reused_when_live(db, ap, llm):
    cid, _ = _desc_case(db, "ap-memo")
    A.run(ap, mode="shadow")
    A.run(ap, mode="shadow")  # same evidence + draft: nothing new
    assert (llm.n_rev, llm.n_ver) == (1, 2) and len(_reviews(db, cid)) == 1

    res = A.run(ap, mode="live")  # goes live: acts on the stored judgement, no new calls

    assert [a["id"] for a in res["applied"]] == [cid]
    assert (llm.n_rev, llm.n_ver) == (1, 2)
    assert [r[1] for r in _reviews(db, cid)] == ["would_apply", "applied"]


def test_a_rescan_is_not_new_evidence_but_a_new_session_is(db, ap, llm):
    old = _skill("ap-ev", "Back up the demo database.")
    _registry(db, "ap-ev", old)
    edited = old.replace("1. Do the thing.", "1. Do the thing.\n2. Verify it.")
    cid = _cand(db, name="ap-ev", body=edited, direction="extend", base=body_hash(old))
    A.run(ap, mode="live")
    assert llm.n_rev == 1

    ev = [_under("ap-s1")]
    ev.append({**ev[0], "scan_night": "2026-11-09", "quote": "again"})  # the same session
    db.execute(
        "UPDATE skills_lane.skill_gap_candidates SET evidence=%s WHERE id=%s", (Json(ev), cid)
    )
    A.run(ap, mode="live")
    assert llm.n_rev == 1

    ev.append(_under("ap-s2"))  # a second session
    db.execute(
        "UPDATE skills_lane.skill_gap_candidates SET evidence=%s WHERE id=%s", (Json(ev), cid)
    )
    A.run(ap, mode="live")
    assert llm.n_rev == 2


def test_a_failed_reviewer_is_retried_not_remembered(db, ap, monkeypatch):
    llm = LLM(monkeypatch)
    llm.reviewer = lambda _p: ""  # CLI not logged in
    monkeypatch.setattr(A, "_reviewer_call", llm.reviewer)
    cid, _ = _desc_case(db, "ap-down")
    res = A.run(ap, mode="live")
    assert res["applied"] == [] and [r[1] for r in _reviews(db, cid)] == ["failed"]
    assert llm.n_ver == 0  # nothing to verify
    monkeypatch.setattr(A, "_reviewer_call", LLM(monkeypatch).reviewer)
    assert [a["id"] for a in A.run(ap, mode="live")["applied"]] == [cid]


# ------------------------------------------------------------------ caps
def test_at_most_three_auto_applies_a_night(db, ap, llm):
    ids = [_desc_case(db, f"ap-cap{i}")[0] for i in range(4)]
    res = A.run(ap, mode="live")
    assert len(res["applied"]) == 3 and len(res["capped"]) == 1
    capped = res["capped"][0]["id"]
    assert _cand_row(db, capped, "status")[0] == "proposed"
    assert sorted(i for i in ids if i != capped) == sorted(a["id"] for a in res["applied"])
    assert [r[1] for r in _reviews(db, capped)] == ["capped"]
    A.run(ap, mode="live")  # the cap still holds the same night, and says nothing new
    assert [r[1] for r in _reviews(db, capped)] == ["capped"]


def test_new_skills_one_a_night_and_three_a_week(db, ap, llm):
    _a, _ = _new_case(db, "ap-new-a")
    _b, _ = _new_case(db, "ap-new-b")
    res = A.run(ap, mode="live")
    assert len(res["applied"]) == 1 and len(res["capped"]) == 1
    # earlier nights: two more new skills this week -> the weekly cap of three is reached
    db.execute(
        "UPDATE skills_lane.skill_auto_reviews SET night = night - 1, "
        "applied_at = now() - interval '1 day' WHERE verdict='applied'"
    )
    for i in range(2):
        cid, _ = _new_case(db, f"ap-new-old{i}")
        db.execute(
            "INSERT INTO skills_lane.skill_auto_reviews (candidate_id, night, mode, evidence_hash,"
            " tier, verdict, skill, applied_at, rolled_back_at, rollback_reason) VALUES "
            "(%s, current_date - 2, 'live', 'h', 'new_skill', 'applied', %s, "
            "now() - interval '2 days', now(), 'local_edit')",  # undone, but still an apply
            (cid, f"ap-new-old{i}"),
        )
        db.execute(
            "UPDATE skills_lane.skill_gap_candidates SET status='observe' WHERE id=%s", (cid,)
        )
    _new_case(db, "ap-new-c")  # a fresh one tonight: still held back, now by the weekly cap
    res = A.run(ap, mode="live")
    assert res["applied"] == [] and "this week" in res["capped"][0]["why"]


def _next_night(conn):
    """Tonight's autopilot rows become last night's (the caps count by night and week)."""
    conn.execute(
        "UPDATE skills_lane.skill_auto_reviews SET night = night - 1, "
        "applied_at = applied_at - interval '1 day' WHERE skill LIKE 'ap-%%'"
    )


def test_a_stored_apply_is_judged_again_once_a_sibling_was_applied(db, ap, monkeypatch):
    """Two derives for the same procedure, both judged 'apply' while neither existed. Once one
    is applied, the other's stored judgement is stale: it is judged again against a catalog
    that has its sibling, not applied from the memo as a duplicate skill."""
    llm = LLM(monkeypatch)
    prompts: list[str] = []

    def reviewer(prompt):
        prompts.append(prompt)
        llm.n_rev += 1
        dup = "- ap-sib-a:" in prompt or "- ap-sib-b:" in prompt
        verdict = "reject" if dup else "accept"
        return json.dumps({"verdict": verdict, "reason": "covered" if dup else "reusable"})

    monkeypatch.setattr(A, "_reviewer_call", reviewer)
    a, _ = _new_case(db, "ap-sib-a")
    b, _ = _new_case(db, "ap-sib-b")
    A.run(ap, mode="shadow")
    assert llm.n_rev == 2 and [r[1] for r in _reviews(db, b)] == ["would_apply"]

    res = A.run(ap, mode="live")

    assert [x["id"] for x in res["applied"]] == [a]
    assert llm.n_rev == 3 and "- ap-sib-a:" in prompts[-1]  # b judged again, with a installed
    assert [r[1] for r in _reviews(db, b)] == ["would_apply", "ask"]
    _next_night(db)
    res = A.run(ap, mode="live")  # a later night: still not applied from the old memo
    assert res["applied"] == [] and _reg(db, "ap-sib-b") is None
    assert _cand_row(db, b, "status")[0] == "proposed"


def test_a_stored_backlog_acts_at_most_the_review_cap_a_night_and_each_is_reported(
    db, ap, monkeypatch
):
    """A shadow period leaves stored judgements; going live acts on at most REVIEW_CAP of
    them a night (not all at once), and the report lists every one with its revive."""
    llm = LLM(monkeypatch, rev="reject", ver="flag")
    ids = [_desc_case(db, f"ap-back{i}")[0] for i in range(A.REVIEW_CAP + 2)]
    A.run(ap, mode="shadow")
    A.run(ap, mode="shadow")
    assert llm.n_rev == len(ids)  # all judged, over two nights

    res = A.run(ap, mode="live")

    assert len(res["rejected"]) == A.REVIEW_CAP and llm.n_rev == len(ids)
    when = datetime(2026, 11, 7, 8, 0, tzinfo=UTC)
    text = RQ.report(ap, when, {"autopilot": res}, deliver=False)["text"]
    assert all(f"revive {r['id']}'" in text for r in res["rejected"])
    rest = A.run(ap, mode="live")["rejected"]
    assert sorted(r["id"] for r in res["rejected"] + rest) == sorted(ids)


def test_a_row_reopened_on_the_evidence_it_was_rejected_on_is_rejected_again(db, ap, monkeypatch):
    llm = LLM(monkeypatch, rev="reject", ver="flag")
    cid, _ = _desc_case(db, "ap-again")
    assert [r["id"] for r in A.run(ap, mode="live")["rejected"]] == [cid]
    # past its cooldown a re-sighting of the same session reopens it (skill_ledger)
    db.execute(
        "UPDATE skills_lane.skill_gap_candidates SET status='proposed', reject_reason=NULL, "
        "rejected_until=NULL WHERE id=%s",
        (cid,),
    )

    res = A.run(ap, mode="live")

    assert [r["id"] for r in res["rejected"]] == [cid] and llm.n_rev == 1  # memo, no new call
    assert _cand_row(db, cid, "status, reject_reason") == ("rejected", "auto:review")
    assert [r[1] for r in _reviews(db, cid)] == ["rejected", "rejected"]


def test_an_auto_reject_is_a_snooze_that_new_evidence_ends(db, ap, monkeypatch):
    """End to end through the ledger: no -3, the cooldown holds, a new session after it
    reopens the row and the autopilot judges it again (the evidence hash changed)."""
    LLM(monkeypatch, rev="reject", ver="flag")
    cid, _ = _desc_case(db, "ap-snooze")
    before = db.execute(
        "SELECT score, grounded_weight FROM skills_lane.skill_gap_candidates WHERE id=%s", (cid,)
    ).fetchone()
    A.run(ap, mode="live")
    after = db.execute(
        "SELECT score, grounded_weight, status FROM skills_lane.skill_gap_candidates WHERE id=%s",
        (cid,),
    ).fetchone()
    assert after == (*before, "rejected")  # no grounded -3, no score change

    def sight(sid):
        new = {**_under(sid), "quote": "can it restore the demo database too"}
        return L.merge_candidate(
            ap,
            "retune",
            "ap-snooze",
            [new],
            direction="widen",
            target_skills=["ap-snooze"],
            do_embed=False,
        )

    assert sight("ap-s3")["status"] == "rejected"  # inside the 30 days: the snooze holds
    db.execute(
        "UPDATE skills_lane.skill_gap_candidates SET rejected_until = now() - interval '1 day' "
        "WHERE id=%s",
        (cid,),
    )
    assert sight("ap-s4")["status"] == "proposed"  # after it: new evidence reopens the row
    llm = LLM(monkeypatch)  # the evidence changed, so it is judged again

    res = A.run(ap, mode="live")

    assert llm.n_rev == 1 and [x["id"] for x in res["applied"]] == [cid]
    assert [r[1] for r in _reviews(db, cid)] == ["rejected", "applied"]


def test_revive_takes_an_auto_rejected_row_out_of_the_lanes_hands(db, ap, monkeypatch):
    llm = LLM(monkeypatch, rev="reject", ver="flag")
    cid, _ = _desc_case(db, "ap-revive")
    A.run(ap, mode="live")
    assert _proposal_act(db.dsn_url, cid, "revive", None, lambda *a: "")["status"] == "proposed"
    A.run(ap, mode="live")
    assert _cand_row(db, cid, "status")[0] == "proposed" and llm.n_rev == 1  # a person's now


# ------------------------------------------------------------------ revert
def test_revert_retires_an_auto_created_skill_and_cools_the_candidate(db, ap, llm):
    cid, _body = _new_case(db, "ap-undo-new")
    A.run(ap, mode="live")

    res = _proposal_act(db.dsn_url, cid, "revert", "not wanted", lambda *a: "")

    assert (res["status"], res["action"]) == ("reverted", "retired")
    assert _reg(db, "ap-undo-new")[1] == "retired"
    status, _by, _rid, reason, ev = _cand_row(db, cid)
    assert (status, reason) == ("rejected", "reverted") and len(_grounded(ev)) == 1  # a person did
    assert _reviews(db, cid)[0][9] is not None and _reviews(db, cid)[0][10] == "not wanted"
    again = _proposal_act(db.dsn_url, cid, "revert", None, lambda *a: "")
    assert again["status"] == "refused" and again["reason"] == "already_reverted"


def test_revert_restores_the_previous_description_body(db, ap, llm):
    cid, old = _desc_case(db, "ap-undo-desc")
    A.run(ap, mode="live")
    assert _reg(db, "ap-undo-desc")[0] != old
    res = _proposal_act(db.dsn_url, cid, "revert", None, lambda *a: "")
    assert res["action"] == "restored" and _reg(db, "ap-undo-desc")[0] == old


def test_revert_keeps_a_later_hand_edit_and_only_puts_the_description_back(db, ap, llm):
    cid, _old = _desc_case(db, "ap-undo-keep")
    A.run(ap, mode="live")
    cur = _reg(db, "ap-undo-keep")[0]
    mine = cur.replace("1. Do the thing.", "1. Do the thing.\n2. My own step.")
    _publish_skill(
        db.dsn_url,
        {"name": "ap-undo-keep", "scope": "global", "body": mine, "content_modified_at": FUTURE},
    )
    res = _proposal_act(db.dsn_url, cid, "revert", None, lambda *a: "")
    assert res["action"] == "description_restored"
    got = _reg(db, "ap-undo-keep")[0]
    assert "My own step." in got and "or restore" not in got


def test_revert_of_an_edited_new_skill_needs_force(db, ap, llm):
    cid, body = _new_case(db, "ap-undo-edited")
    A.run(ap, mode="live")
    _publish_skill(
        db.dsn_url,
        {
            "name": "ap-undo-edited",
            "scope": "global",
            "body": body + "\n2. Mine.\n",
            "content_modified_at": FUTURE,
        },
    )
    res = _proposal_act(db.dsn_url, cid, "revert", None, lambda *a: "")
    assert res["status"] == "refused" and res["reason"] == "edited_since"
    assert _reg(db, "ap-undo-edited")[1] == "active"
    forced = _proposal_act(db.dsn_url, cid, "revert", None, lambda *a: "", force=True)
    assert forced["action"] == "retired"


def test_a_human_applied_row_cannot_be_reverted_this_way(db):
    cid, _ = _desc_case(db, "ap-hum2")
    _proposal_act(db.dsn_url, cid, "accept", None, lambda *a: "")
    res = _proposal_act(db.dsn_url, cid, "revert", None, lambda *a: "")
    assert res["status"] == "refused" and res["reason"] == "not_auto_applied"


# ------------------------------------------------------------------ rollback + breaker
def _fire(conn, skill, *, dismissed):
    conn.execute(
        "INSERT INTO skills_lane.skill_usage (skill, fired_at, session_id, via, dismissed) "
        "VALUES (%s, now() + interval '1 hour', 'ap-fire', 'tool', %s)",
        (skill, dismissed),
    )


def test_a_dismissed_fire_after_the_change_rolls_it_back(db, ap, llm):
    cid, old = _desc_case(db, "ap-dis")
    A.run(ap, mode="live")
    _fire(db, "ap-dis", dismissed=False)
    assert A.reconcile_once(ap)["rolled_back"] == []  # a clean fire is fine
    _fire(db, "ap-dis", dismissed=True)

    rec = A.reconcile_once(ap)

    assert [(r["id"], r["reason"]) for r in rec["rolled_back"]] == [(cid, "dismissed")]
    assert _reg(db, "ap-dis")[0] == old
    status, _by, _rid, reason, ev = _cand_row(db, cid)
    assert (status, reason) == ("rejected", "auto_rollback:dismissed")
    assert _grounded(ev) == []  # the lane's own call: no grounded entry
    assert _reviews(db, cid)[0][10] == "dismissed"


def test_a_post_fire_deviation_on_a_later_day_rolls_it_back(db, ap, llm):
    _cid, old = _desc_case(db, "ap-dev")
    A.run(ap, mode="live")
    soon = {
        "class": "judge",
        "signal": "post_fire_deviation",
        "session_id": "ap-x",
        "date": "2020-01-01",
    }
    later = {**soon, "date": "2099-01-01"}
    db.execute(
        "INSERT INTO skills_lane.skill_gap_candidates (kind, name, direction, target_skills, "
        "status, summary, evidence, source_detector) VALUES ('retune','ap-dev','fix',"
        "ARRAY['ap-dev'],'observe','s',%s,'post_fire')",
        (Json([soon]),),
    )
    assert A.reconcile_once(ap)["rolled_back"] == []  # a fire from before the change says nothing
    db.execute(
        "UPDATE skills_lane.skill_gap_candidates SET evidence=%s WHERE name='ap-dev' "
        "AND source_detector='post_fire'",
        (Json([soon, later]),),
    )
    rec = A.reconcile_once(ap)
    assert [r["reason"] for r in rec["rolled_back"]] == ["post_fire_deviation"]
    assert _reg(db, "ap-dev")[0] == old


def test_the_owner_editing_the_description_away_rolls_it_back_and_their_edit_stands(db, ap, llm):
    cid, old = _desc_case(db, "ap-owner")
    A.run(ap, mode="live")
    mine = old.replace("1. Do the thing.", "1. Do the thing.\n2. My own step.")
    _publish_skill(
        db.dsn_url,
        {"name": "ap-owner", "scope": "global", "body": mine, "content_modified_at": FUTURE},
    )

    rec = A.reconcile_once(ap)

    assert [r["reason"] for r in rec["rolled_back"]] == ["owner_edit"]
    assert _reg(db, "ap-owner")[0] == mine  # nothing is written over the owner's version
    assert _cand_row(db, cid, "reject_reason")[0] == "auto_rollback:owner_edit"


def test_an_owner_edit_that_keeps_the_new_description_is_not_a_rollback(db, ap, llm):
    cid, _old = _desc_case(db, "ap-owner-keep")
    A.run(ap, mode="live")
    mine = _reg(db, "ap-owner-keep")[0].replace("1. Do the thing.", "1. Do it my way.")
    _publish_skill(
        db.dsn_url,
        {"name": "ap-owner-keep", "scope": "global", "body": mine, "content_modified_at": FUTURE},
    )
    assert A.reconcile_once(ap)["rolled_back"] == []
    assert _cand_row(db, cid, "status")[0] == "promoted"


def test_the_owner_retiring_a_new_skill_is_a_rollback(db, ap, llm):
    cid, _ = _new_case(db, "ap-gone")
    A.run(ap, mode="live")
    db.execute("UPDATE skills_lane.skill_registry SET status='retired' WHERE name='ap-gone'")
    rec = A.reconcile_once(ap)
    assert [r["reason"] for r in rec["rolled_back"]] == ["owner_retired"]
    assert _cand_row(db, cid, "status")[0] == "rejected"


def test_two_rollbacks_trip_the_breaker_until_a_person_resumes(db, ap, llm):
    for i in range(2):
        _desc_case(db, f"ap-brk{i}")
    A.run(ap, mode="live")
    for i in range(2):
        _fire(db, f"ap-brk{i}", dismissed=True)
    more, _ = _desc_case(db, "ap-brk-next")

    res = A.run(ap, mode="live")  # reconciles first, then runs as shadow

    assert len(res["rolled_back"]) == 2 and res["breaker"]["rollbacks"] == 2
    assert res["applied"] == [] and _cand_row(db, more, "status")[0] == "proposed"
    assert _reviews(db, more)[0][1] == "would_apply"
    assert A.status(ap_dict(ap))["breaker"]["tripped"] is True

    out = _autopilot(db.dsn_url, "resume")
    assert out["status"] == "resumed" and out["tripped"] is False
    assert [a["id"] for a in A.run(ap, mode="live")["applied"]] == [more]


def ap_dict(conn):
    from psycopg.rows import dict_row

    conn.row_factory = dict_row
    return conn


# ------------------------------------------------------------------ the sync race
def _race_case(db, ap, name):
    cid, old = _desc_case(db, name)
    A.run(ap, mode="live")
    applied = _reg(db, name)[0]
    assert applied != old
    return cid, old, applied


def test_a_local_edit_pulled_over_keeps_both_when_it_kept_the_old_description(db, ap, llm):
    # the machine edited the body but not the description, and its sync pulled the auto change
    # over it (the client records the clobbered body first, as 'disk_overwrite')
    cid, old, _applied = _race_case(db, ap, "ap-race-a")
    local = old.replace("1. Do the thing.", "1. Do the thing.\n2. My local step.")
    _record_overwrite(
        db.dsn_url,
        {"name": "ap-race-a", "scope": "global", "body": local, "content_modified_at": OLD_TIME},
    )

    rec = A.reconcile_once(ap)

    assert [c["what"] for c in rec["conflicts"]] == ["merged"] and rec["rolled_back"] == []
    got = _reg(db, "ap-race-a")[0]
    assert "My local step." in got and "or restore the demo database" in got
    assert _cand_row(db, cid, "status")[0] == "promoted"
    again = A.reconcile_once(ap)  # handled once, not again
    assert again["conflicts"] == [] and _reg(db, "ap-race-a")[0] == got


def test_a_local_edit_that_changed_the_description_too_is_restored_and_goes_to_a_person(
    db, ap, llm
):
    cid, old, _applied = _race_case(db, ap, "ap-race-b")
    local = old.replace("Back up the demo database.", "Snapshot the demo database.")
    _record_overwrite(
        db.dsn_url,
        {"name": "ap-race-b", "scope": "global", "body": local, "content_modified_at": OLD_TIME},
    )

    rec = A.reconcile_once(ap)

    assert [c["what"] for c in rec["conflicts"]] == ["restored"]
    assert _reg(db, "ap-race-b")[0] == local  # the local edit is the newest version again
    assert _cand_row(db, cid, "status")[0] == "proposed"  # back in the queue
    (r,) = _reviews(db, cid)
    assert r[10] == "local_edit"  # neutral: not a strike against the breaker
    assert A.breaker_state(ap_dict(ap))["tripped"] is False
    res = A.run(ap, mode="live")  # the same evidence + draft is never auto-applied again
    assert res["applied"] == [] and _cand_row(db, cid, "status")[0] == "proposed"


def test_a_newer_local_edit_that_was_pushed_over_the_change_is_the_owners_to_keep(db, ap, llm):
    _cid, old, applied = _race_case(db, ap, "ap-race-c")
    local = old.replace("1. Do the thing.", "1. Do the thing.\n2. My pushed step.")
    _publish_skill(
        db.dsn_url,
        {"name": "ap-race-c", "scope": "global", "body": local, "content_modified_at": FUTURE},
    )
    rec = A.reconcile_once(ap)
    assert [r["reason"] for r in rec["rolled_back"]] == ["owner_edit"]
    assert _reg(db, "ap-race-c")[0] == local
    # the auto change is not lost either: it is the body that edit superseded
    assert (
        db.execute(
            "SELECT count(*) FROM skills_lane.skill_history WHERE name='ap-race-c' AND body=%s",
            (applied,),
        ).fetchone()[0]
        == 1
    )


def test_a_pull_of_a_body_the_registry_already_had_is_not_a_local_edit(db, ap, llm):
    # a machine that simply had not synced yet still holds the OLD body; the sync records it
    _cid, old, applied = _race_case(db, ap, "ap-race-d")
    _record_overwrite(
        db.dsn_url,
        {"name": "ap-race-d", "scope": "global", "body": old, "content_modified_at": OLD_TIME},
    )
    rec = A.reconcile_once(ap)
    assert rec["conflicts"] == [] and rec["rolled_back"] == []
    assert _reg(db, "ap-race-d")[0] == applied


# ------------------------------------------------------------------ report
def test_the_report_leads_with_auto_applied_items_and_is_stored(db, ap, llm):
    _cid, _ = _new_case(db, "ap-rep")
    _desc_case(db, "ap-rep2")
    res = A.run(ap, mode="live")
    when = datetime(2026, 11, 7, 8, 0, tzinfo=UTC)

    rep = RQ.report(ap, when, {"autopilot": res}, deliver=False)

    lines = rep["text"].split("\n")
    assert lines[0].startswith("Skills lane 11-07: 2 auto-applied")
    assert lines[1].startswith("auto-applied #") and "(undo: revert " in lines[1]
    assert {f"(undo: revert {a['id']})" for a in res["applied"]} <= {
        "(undo: " + ln.split("(undo: ")[1].split(")")[0] + ")" for ln in lines if "(undo: " in ln
    }
    assert len(rep["text"]) <= RQ.REPORT_CHARS
    stored = L.get_cursor(ap)["config"]["last_report"]
    assert stored == {"at": when.isoformat(), "text": rep["text"]}  # built == stored
    assert "report_sent_at" not in L.get_cursor(ap)["config"]  # delivery is the caller's


def test_the_report_says_when_the_breaker_paused_the_autopilot(db, ap, llm):
    text = RQ.build_report(
        "11-07",
        [],
        [],
        [],
        {
            "breaker": {"rollbacks": 2, "skills": ["a", "b"]},
            "rolled_back": [{"id": 3, "what": "a description", "reason": "dismissed"}],
        },
    )
    assert "autopilot paused" in text and "autopilot resume" in text and "rolled back #3" in text


def test_nothing_new_and_no_reminder_due_builds_nothing(db, ap, llm):
    when = datetime(2026, 11, 7, 8, 0, tzinfo=UTC)
    RQ.mark_sent(ap, when)
    rep = RQ.report(ap, when, {"autopilot": {"mode": "live"}}, deliver=False)
    assert "text" not in rep
    assert "last_report" not in L.get_cursor(ap)["config"]


def test_the_status_route_reads_the_run(db, ap, llm):
    cid, _ = _desc_case(db, "ap-status")
    A.run(ap, mode="live")
    st = _autopilot(db.dsn_url, "status")
    assert st["last_mode"] == "live" and st["counts_14d"] == {"applied": 1}
    assert st["recent"][0]["candidate_id"] == cid


# ------------------------------------------------------------------ wiring
def test_a_human_accept_still_works_before_schema_064(db, monkeypatch):
    import mcp_server.skill_sync_routes as R

    monkeypatch.setattr(R, "_autopilot_ready", lambda c: False)
    cid, _old = _desc_case(db, "ap-pre064")
    res = _proposal_act(db.dsn_url, cid, "accept", None, lambda *a: "")
    assert res["status"] == "promoted"
    status, by, rid, _r, _ev = _cand_row(db, cid)
    assert (status, by, rid) == ("promoted", None, None)  # nothing recorded that is not there
    assert _reg(db, "ap-pre064")[2] is None
    refused = R.apply_candidate(db, cid, None, None, False, actor="auto", review={})
    assert refused["status"] == "error"  # and the auto path will not run without it


def _quiet_lane(monkeypatch):
    """run_lane with every stage that needs sessions or a model stood down: what is left is
    the autopilot's wiring, the review queue and the report."""
    import dream.skills.nightly as N

    monkeypatch.setenv("SYNAPSE_SKILLS_DETECTORS", "")
    monkeypatch.setattr(N.SM, "load_skills", lambda: {})
    monkeypatch.setattr(N.SM, "load_scopes", lambda conn: {})
    monkeypatch.setattr(N.DB, "sessions_since", lambda *a, **k: [])
    for name in ("consolidate_pass", "sync_usage", "run_derive", "run_retune", "capture_grounded"):
        monkeypatch.setattr(N, name, lambda *a, **k: {})
    monkeypatch.setattr(N.L, "regate_observe", lambda conn: {})
    monkeypatch.setattr(N.skill_draft, "draft_pending", lambda conn: {})
    monkeypatch.setattr(N.review_queue, "recommend", lambda conn: {})
    return N


def test_the_nightly_lane_reconciles_but_only_reviews_when_this_container_can(
    db, monkeypatch, capsys
):
    N = _quiet_lane(monkeypatch)
    llm = LLM(monkeypatch)
    monkeypatch.delenv("SKILLS_AUTOREVIEW_REVIEW", raising=False)
    monkeypatch.delenv("SKILLS_AUTOREVIEW", raising=False)  # the default mode
    cid, _old = _desc_case(db, "ap-night")

    counts = N.run_lane(no_discord=True)

    assert llm.n_rev == 0 and _cand_row(db, cid, "status")[0] == "proposed"  # no reviewer here
    assert counts["autoreview_judged"] == 0 and counts["autoreview_applied"] == 0

    monkeypatch.setenv("SKILLS_AUTOREVIEW_REVIEW", "1")
    counts = N.run_lane(no_discord=True)

    assert counts["autoreview_applied"] == 1 and llm.n_rev == 1
    assert _cand_row(db, cid, "applied_by")[0] == "auto"
    out = capsys.readouterr().out
    assert "auto-applied #" in out and "(undo: revert " in out  # the report, delivered to stdout
    last = L.get_cursor(db)["config"]["last_report"]
    assert "auto-applied #" in last["text"] and last["at"]


def test_the_default_mode_is_live_and_off_is_respected_by_the_nightly_lane(db, monkeypatch):
    N = _quiet_lane(monkeypatch)
    llm = LLM(monkeypatch)
    monkeypatch.setenv("SKILLS_AUTOREVIEW_REVIEW", "1")
    monkeypatch.setenv("SKILLS_AUTOREVIEW", "off")
    cid, _old = _desc_case(db, "ap-night-off")
    counts = N.run_lane(no_discord=True)
    assert counts["autoreview_judged"] == 0 and llm.n_rev == 0
    assert _cand_row(db, cid, "status")[0] == "proposed"
    assert A.DEFAULT_MODE == "live"
