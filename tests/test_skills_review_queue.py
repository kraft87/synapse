"""The dream->skills review queue looks after itself — DB-backed (schema 062 + review_queue).

Runs against the shared Postgres test DB (tagged into the `db` xdist group via conftest).
Every skill, session and candidate is synthetic (rq- prefix) and cleaned up around each test.

Covers:
  - auto_reject: scope mismatch (the #76 shape), retired / missing target, renamed, widen
    no-op and already-covered; the -3 a human reject writes is NOT written, out-of-scope
    evidence and its draft are dropped, a re-sighting past the cooldown reopens the row
  - expire after 21 days unanswered, and a re-sighting reopening an expired row
  - recommend: one judge call per draft, re-asked when the draft changes
  - report: verdict first, plain text, under 1500 chars, sent on change and weekly
  - proposed_at stamping (ledger gate, nightly re-gate, revive) and dream_runs counting it
  - the pre-062 path (probe says absent): nothing crashes, the features step aside
  - the scope a session is judged in, by the accept path's own rule
"""

from __future__ import annotations

import json
from datetime import timedelta

import pytest
from psycopg.types.json import Json

import dream.skills.nightly as N
import dream.skills.review_queue as RQ
import dream.skills.skill_ledger as L
from dream.skills.skill_doc import body_hash
from ingestion.schema_check import ColumnProbe
from mcp_server.skill_sync_routes import _proposal_act, _proposals_list


def _skill(name, desc, step="Do the thing."):
    return f"---\nname: {name}\ndescription: {desc}\n---\n# {name}\n\n1. {step}\n"


DOCX = _skill("rq-docx", "Render and edit transcript Word documents.")
DOCX_WIDE = _skill("rq-docx", "Render and edit transcript Word documents, resumes and CVs.")


@pytest.fixture()
def rq(conn):
    def _clean():
        conn.execute("DELETE FROM skills_lane.skill_gap_candidates WHERE name LIKE 'rq-%%'")
        conn.execute("DELETE FROM skills_lane.skill_registry WHERE name LIKE 'rq-%%'")
        conn.execute("DELETE FROM skills_lane.skill_history WHERE name LIKE 'rq-%%'")
        conn.execute("DELETE FROM episodes WHERE session_id LIKE 'rq-%%'")
        conn.execute("DELETE FROM dream_runs WHERE errors @> '[\"rq-test\"]'::jsonb")
        conn.execute(
            "UPDATE skills_lane.skill_scan_cursor SET config = config - 'report_sent_at' WHERE id=1"
        )

    _clean()
    # the shared DB may hold other proposed rows; run each test with the queue to itself
    conn.execute(
        "UPDATE skills_lane.skill_gap_candidates SET status='observe' WHERE status='proposed'"
    )
    yield conn
    _clean()


def _registry(conn, name, body, scope="global", status="active"):
    conn.execute(
        "INSERT INTO skills_lane.skill_registry (name, scope, status, body, description) "
        "VALUES (%s, %s, %s, %s, %s)",
        (name, scope, status, body, body.split("description: ")[1].split("\n")[0]),
    )


def _session(conn, sid, project, day="2026-10-20"):
    conn.execute(
        "INSERT INTO episodes (session_id, sequence, project, platform, content, metadata, "
        "created_at) VALUES (%s, 1, %s, 'claude_code', 'synthetic', %s, %s)",
        (sid, project, Json({"cwd": f"/work/{project}"}), f"{day}T15:00:00Z"),
    )


def _under(sid, missing="resume"):
    return {
        "session_id": sid,
        "class": "judge",
        "signal": "under_trigger",
        "skill": "rq-docx",
        "missing_phrasing": missing,
    }


def _cand(
    conn,
    *,
    name="rq-docx",
    kind="retune",
    direction="widen",
    target="rq-docx",
    body=DOCX_WIDE,
    base=DOCX,
    evidence=None,
    status="proposed",
    proposed_days_ago=1,
):
    ev = evidence if evidence is not None else [_under("rq-s-tai")]
    return conn.execute(
        "INSERT INTO skills_lane.skill_gap_candidates (kind, name, direction, target_skills, "
        "status, summary, evidence, proposal_body, base_body_hash, judge_sessions, judge_weight, "
        "proposed_at) VALUES (%s,%s,%s,%s,%s,'synthetic',%s,%s,%s,1,1.0, "
        "now() - make_interval(days => %s)) RETURNING id",
        (
            kind,
            name,
            direction,
            [target] if target else [],
            status,
            Json(ev),
            body,
            body_hash(base) if base else None,
            proposed_days_ago,
        ),
    ).fetchone()[0]


def _row(conn, cid, cols="status, reject_reason"):
    return conn.execute(
        f"SELECT {cols} FROM skills_lane.skill_gap_candidates WHERE id=%s", (cid,)
    ).fetchone()


# ------------------------------------------------------------------ auto-reject


def test_scope_mismatch_auto_rejects_without_a_human_reject(rq):
    # #76's shape: docx is project:transcribe_ai, half its evidence came from elsewhere
    _registry(rq, "rq-docx", DOCX, scope="project:rq_tai")
    _session(rq, "rq-s-tai", "rq_tai")
    _session(rq, "rq-s-jobs", "rq_jobs")
    cid = _cand(rq, evidence=[_under("rq-s-tai", "author metadata"), _under("rq-s-jobs")])

    res = RQ.auto_reject(rq)

    assert [(i["id"], i["code"]) for i in res["items"]] == [(cid, "scope_mismatch")]
    status, reason, until, ev, gw, body = _row(
        rq, cid, "status, reject_reason, rejected_until, evidence, grounded_weight, proposal_body"
    )
    assert (status, reason) == ("rejected", "auto:scope_mismatch")
    assert timedelta(days=29) < until - _now(rq) <= timedelta(days=30)
    assert [e["session_id"] for e in ev] == ["rq-s-tai"]  # the out-of-scope evidence left
    assert gw == 0 and not any(e.get("signal") == "reject" for e in ev)  # no -3
    assert body is None  # the draft built from the out-of-scope sessions went with it


def test_a_multi_project_session_is_out_of_a_project_skills_scope(rq):
    _registry(rq, "rq-docx", DOCX, scope="project:rq_tai")
    _session(rq, "rq-s-mix", "rq_tai")
    rq.execute(
        "INSERT INTO episodes (session_id, sequence, project, platform, content) "
        "VALUES ('rq-s-mix', 2, 'rq_jobs', 'claude_code', 'synthetic')"
    )
    cid = _cand(rq, evidence=[_under("rq-s-mix")])
    RQ.auto_reject(rq)
    assert _row(rq, cid) == ("rejected", "auto:scope_mismatch")


def test_in_scope_widen_with_a_real_change_stays_proposed(rq):
    _registry(rq, "rq-docx", DOCX, scope="project:rq_tai")
    _session(rq, "rq-s-tai", "rq_tai")
    cid = _cand(rq)
    assert RQ.auto_reject(rq)["auto_rejected"] == 0
    assert _row(rq, cid) == ("proposed", None)


@pytest.mark.parametrize(
    ("setup", "code"),
    [
        ("missing", "no_target"),
        ("retired", "retired"),
        ("renamed", "renamed"),
        ("no_change", "no_change"),
        ("covered", "covered"),
    ],
)
def test_undeliverable_retunes_auto_reject(rq, setup, code):
    _session(rq, "rq-s-tai", "rq_tai")
    body, evidence = DOCX_WIDE, None
    if setup != "missing":
        _registry(rq, "rq-docx", DOCX, status="retired" if setup == "retired" else "active")
    if setup == "renamed":
        body = DOCX_WIDE.replace("name: rq-docx", "name: rq-docx-2")
    if setup == "no_change":
        body = DOCX.replace("1. Do the thing.", "1. Do the thing, twice.")
    if setup == "covered":
        evidence = [_under("rq-s-tai", "transcript Word documents")]
    cid = _cand(rq, body=body, evidence=evidence)
    res = RQ.auto_reject(rq)
    assert [i["code"] for i in res["items"]] == [code]
    assert _row(rq, cid) == ("rejected", f"auto:{code}")


def test_a_stale_draft_is_left_for_the_drafter(rq):
    _registry(rq, "rq-docx", DOCX)
    _session(rq, "rq-s-tai", "rq_tai")
    cid = _cand(rq, base=_skill("rq-docx", "An older body."))
    assert RQ.auto_reject(rq)["auto_rejected"] == 0
    assert _row(rq, cid) == ("proposed", None)


def test_auto_rejected_row_reopens_on_new_evidence_after_the_cooldown(rq):
    _registry(rq, "rq-docx", DOCX)
    cid = _cand(rq, target="rq-gone", name="rq-gone")
    RQ.auto_reject(rq)
    rq.execute(
        "UPDATE skills_lane.skill_gap_candidates SET rejected_until = now() - interval '1 day' "
        "WHERE id=%s",
        (cid,),
    )
    r = L.merge_candidate(
        rq,
        "retune",
        "rq-gone",
        [_under("rq-s-new")],
        direction="widen",
        target_skills=["rq-gone"],
        do_embed=False,
    )
    assert r["id"] == cid and r["status"] == "observe"
    assert _row(rq, cid, "grounded_weight")[0] == 0  # nothing held against it


# ------------------------------------------------------------------ expiry


def test_unanswered_proposals_expire_after_21_days(rq):
    old = _cand(rq, name="rq-old", target="rq-old", proposed_days_ago=22)
    young = _cand(rq, name="rq-young", target="rq-young", proposed_days_ago=20)
    res = RQ.expire(rq)
    assert [(i["id"], i["days"]) for i in res["items"]] == [(old, 22)]
    assert _row(rq, old) == ("retired", "expired")
    assert _row(rq, young) == ("proposed", None)
    # a fresh sighting reopens the expired row instead of starting a new one
    r = L.merge_candidate(
        rq,
        "retune",
        "rq-old",
        [_under("rq-s-new")],
        direction="widen",
        target_skills=["rq-old"],
        do_embed=False,
    )
    assert r["id"] == old and r["status"] == "observe"


# ------------------------------------------------------------------ recommendation


def test_recommendation_is_stored_once_per_draft(rq, monkeypatch):
    _registry(rq, "rq-docx", DOCX)
    cid = _cand(rq)
    prompts = []

    def judge(prompt):
        prompts.append(prompt)
        return 'ok: {"verdict": "Accept", "reason": "Two sessions asked for it — **real** gap."}'

    monkeypatch.setattr(RQ, "_judge_call", judge)
    assert RQ.recommend(rq)["recommended"] == 1
    rec = _row(rq, cid, "recommendation")[0]
    assert rec["verdict"] == "accept" and rec["draft_hash"] == body_hash(DOCX_WIDE)
    assert rec["reason"] == "Two sessions asked for it, real gap." and rec["at"]
    assert "+description: Render and edit transcript Word documents, resumes" in prompts[0]
    assert RQ.recommend(rq)["recommended"] == 0 and len(prompts) == 1  # same draft: no call
    rq.execute(
        "UPDATE skills_lane.skill_gap_candidates SET proposal_body = %s WHERE id=%s",
        (DOCX_WIDE.replace("CVs", "CVs and cover letters"), cid),
    )
    assert RQ.recommend(rq)["recommended"] == 1 and len(prompts) == 2  # new draft: asked again


def test_unusable_recommendation_is_not_stored(rq, monkeypatch):
    _registry(rq, "rq-docx", DOCX)
    cid = _cand(rq)
    monkeypatch.setattr(RQ, "_judge_call", lambda prompt: "I think maybe?")
    assert RQ.recommend(rq) == {"pending": 1, "recommended": 0, "failed": 1, "deferred": 0}
    assert _row(rq, cid, "recommendation")[0] is None


# ------------------------------------------------------------------ report


def _now(conn):
    return conn.execute("SELECT now()").fetchone()[0]


def test_report_lists_tonights_changes_verdict_first(rq, capsys):
    started = _now(rq)
    rec = {"verdict": "accept", "reason": "Two sessions on two days ran the same steps."}
    new = _cand(
        rq,
        name="rq-new",
        kind="derive",
        direction=None,
        target=None,
        base=None,
        body=_skill("rq-new", "New."),
        proposed_days_ago=0,
    )
    rq.execute(
        "UPDATE skills_lane.skill_gap_candidates SET proposed_at = now() + interval '1 second', "
        "recommendation = %s WHERE id=%s",
        (Json(rec), new),
    )
    old = _cand(rq, name="rq-old", target="rq-old", proposed_days_ago=19)
    changes = {
        "auto_rejected": [
            {
                "id": 76,
                "kind": "retune",
                "name": "rq-docx",
                "direction": "widen",
                "target_skills": ["rq-docx"],
                "code": "scope_mismatch",
            }
        ],
        "expired": [
            {
                "id": 142,
                "kind": "retune",
                "name": "rq-bug",
                "direction": "extend",
                "target_skills": ["rq-bug"],
                "days": 26,
                "recommendation": None,
            }
        ],
    }
    res = RQ.report(rq, started, changes, no_discord=True)
    out = capsys.readouterr().out.strip().splitlines()
    assert res["sent"] == 1
    assert out[0] == f"Skills lane {started:%m-%d}: 1 new, 1 auto-rejected, 1 expired, 2 pending."
    assert out[1] == (
        f"accept #{new} new skill rq-new (new tonight): "
        "Two sessions on two days ran the same steps."
    )
    assert out[2].startswith("auto-rejected #76 rq-docx widen: its evidence came from sessions")
    assert out[3] == (
        "expired #142 rq-bug extend after 26d unanswered. Reply 'revive 142' to bring it back."
    )
    assert out[4] == f"no verdict yet #{old} rq-old widen (19d pending)"
    assert out[-1] == "Reply 'approve N', 'reject N' or 'tell me about N'."
    text = "\n".join(out)
    assert "\N{EM DASH}" not in text and "**" not in text and len(text) <= RQ.REPORT_CHARS


def test_report_is_quiet_between_changes_then_reminds_weekly(rq, capsys):
    _cand(rq, name="rq-old", target="rq-old", proposed_days_ago=3)
    started = _now(rq)
    assert RQ.report(rq, started, {}, no_discord=True)["sent"] == 1  # never sent before
    capsys.readouterr()
    assert RQ.report(rq, started + timedelta(days=1), {}, no_discord=True)["sent"] == 0
    assert capsys.readouterr().out.startswith("(report: 1 pending, nothing changed")
    res = RQ.report(rq, started + timedelta(days=7), {}, no_discord=True)
    assert res == {"pending": 1, "sent": 1, "reminder": 1}
    assert "weekly reminder" in capsys.readouterr().out


def test_report_posts_to_the_webhook_and_caps_its_length(rq, monkeypatch):
    for i in range(40):
        _cand(rq, name=f"rq-many-{i:02d}", target=f"rq-many-{i:02d}", proposed_days_ago=2)
    posted = []
    monkeypatch.setattr(RQ.config, "DISCORD_WEBHOOK", "https://discord.invalid/webhook")
    monkeypatch.setattr(
        RQ.urllib.request, "urlopen", lambda req, timeout: posted.append(json.loads(req.data))
    )
    assert RQ.report(rq, _now(rq), {}, no_discord=False)["sent"] == 1
    msg = posted[0]["content"]
    assert len(msg) <= RQ.REPORT_CHARS and "more: skill_review.py list" in msg
    assert msg.splitlines()[0].endswith("40 pending.")


# ------------------------------------------------------------------ proposed_at


def test_the_gate_stamps_proposed_at(rq):
    r = L.merge_candidate(
        rq,
        "retune",
        "rq-quoted",
        [dict(_under("rq-s1"), quote="it never fired")],
        direction="widen",
        target_skills=["rq-quoted"],
        do_embed=False,
    )
    assert r["status"] == "proposed"
    age = _now(rq) - _row(rq, r["id"], "proposed_at")[0]
    assert timedelta(0) <= age < timedelta(minutes=1)


def test_regate_proposes_an_open_derive_once_its_session_days_are_known(rq):
    # a derive written before dates were stamped: two sessions, two days, still observe
    _session(rq, "rq-s-a", "rq_jobs", "2026-10-23")
    _session(rq, "rq-s-b", "rq_jobs", "2026-10-28")
    ev = [{"session_id": s, "class": "judge", "signal": "gap_scan"} for s in ("rq-s-a", "rq-s-b")]
    cid = _cand(
        rq,
        name="rq-tailoring",
        kind="derive",
        direction=None,
        target=None,
        base=None,
        body=None,
        evidence=ev,
        status="observe",
    )
    rq.execute("UPDATE skills_lane.skill_gap_candidates SET proposed_at=NULL WHERE id=%s", (cid,))
    res = L.regate_observe(rq)
    assert cid in res["ids"]
    status, pat, stored = _row(rq, cid, "status, proposed_at, evidence")
    assert status == "proposed" and pat is not None
    assert sorted(e["date"] for e in stored) == ["2026-10-23", "2026-10-28"]


def test_dream_runs_counts_a_row_that_went_proposed_tonight(rq, db_url):
    from dream.__main__ import _dream_collect_proposals

    run_id = rq.execute(
        "INSERT INTO dream_runs (started_at, errors) VALUES (now(), '[\"rq-test\"]') RETURNING id"
    ).fetchone()[0]
    cid = _cand(rq, name="rq-late", target="rq-late", status="observe")
    rq.execute(
        "UPDATE skills_lane.skill_gap_candidates SET created_at = now() - interval '40 days', "
        "status='proposed', proposed_at = now() + interval '1 second' WHERE id=%s",
        (cid,),
    )
    counts: dict = {}
    samples = _dream_collect_proposals(db_url, run_id, counts)
    assert counts["proposals_raised"] >= 1
    assert {"id": f"skill:{cid}", "kind": "skill", "name": "rq-late"} in samples["proposals"]


# ------------------------------------------------------------------ revive + list


def test_revive_brings_back_a_rejected_proposal_and_list_shows_its_age(rq, db_url):
    cid = _cand(rq, status="rejected", proposed_days_ago=40)
    rq.execute(
        "UPDATE skills_lane.skill_gap_candidates SET reject_reason='user_rejected', "
        "rejected_until = now() + interval '20 days', grounded_weight = -3, evidence = "
        'evidence || \'[{"session_id": null, "class": "grounded", "signal": "reject"}]\' '
        "WHERE id=%s",
        (cid,),
    )
    r = _proposal_act(db_url, cid, "revive", None, lambda *_: "")
    assert r["status"] == "proposed" and r["id"] == cid
    status, reason, until, gw, ev = _row(
        rq, cid, "status, reject_reason, rejected_until, grounded_weight, evidence"
    )
    assert (status, reason, until) == ("proposed", None, None)
    assert gw == 0  # the revive cancels the old reject...
    accept = {"session_id": None, "class": "grounded", "signal": "accept"}
    assert L._rollup([*ev, accept])[3] > 0  # ...so a later accept clears the 023 CHECK
    listed = [p for p in _proposals_list(db_url)["proposals"] if p["id"] == cid]
    assert listed and listed[0]["days_pending"] == 0 and "recommendation" in listed[0]
    again = _proposal_act(db_url, cid, "revive", None, lambda *_: "")
    assert again["status"] == "refused"


# ------------------------------------------------------------------ before schema 062


def test_queue_probe_answers_absent_while_062_is_missing(rq):
    probe = ColumnProbe(L.QUEUE_PROBE._sql, "t")
    with pytest.raises(RuntimeError), rq.transaction():
        rq.execute("SET LOCAL lock_timeout = '5s'")  # fail, never hang, if a lock is held
        rq.execute("ALTER TABLE skills_lane.skill_gap_candidates DROP COLUMN recommendation")
        assert not probe.ready(rq, "without-062")
        raise RuntimeError("roll the DROP back")
    assert ColumnProbe(L.QUEUE_PROBE._sql, "t").ready(rq, "with-062")


def test_without_062_the_queue_steps_aside(rq, monkeypatch, capsys):
    monkeypatch.setattr(L, "queue_ready", lambda conn: False)
    cid = _cand(rq, proposed_days_ago=30)
    assert RQ.expire(rq)["skipped"] == "schema 062 not applied"
    assert RQ.recommend(rq)["skipped"] == "schema 062 not applied"
    assert _row(rq, cid) == ("proposed", None)
    assert RQ.report(rq, _now(rq), {}, no_discord=True)["sent"] == 1
    assert f"#{cid} rq-docx widen (pending)" in capsys.readouterr().out
    r = L.merge_candidate(
        rq,
        "retune",
        "rq-quoted",
        [dict(_under("rq-s1"), quote="it never fired")],
        direction="widen",
        target_skills=["rq-quoted"],
        do_embed=False,
    )
    assert r["status"] == "proposed" and _row(rq, r["id"], "proposed_at")[0] is None


# ------------------------------------------------------------------ session scope


def test_session_scopes_follow_the_accept_paths_rule(rq):
    _session(rq, "rq-s-tai", "rq_tai")
    _session(rq, "rq-s-mix", "rq_tai")
    rq.execute(
        "INSERT INTO episodes (session_id, sequence, project, platform, content) "
        "VALUES ('rq-s-mix', 2, 'rq_jobs', 'claude_code', 'synthetic')"
    )
    views = [{"session": "rq-s-tai"}, {"session": "rq-s-mix"}, {"session": "rq-s-none"}]
    N._session_scopes(rq, views)
    assert [v["scope"] for v in views] == ["project:rq_tai", "global", "global"]


# ------------------------------------------------------------------ run counts


def test_run_lane_returns_the_nights_counts(rq, monkeypatch, capsys):
    monkeypatch.setenv("SYNAPSE_SKILLS_DETECTORS", "post_fire")
    monkeypatch.setattr(N.SM, "load_skills", lambda: {})
    monkeypatch.setattr(N, "consolidate_pass", lambda conn, seen: {"embedded": 0})
    monkeypatch.setattr(N, "sync_usage", lambda conn, last: {"fire_events": 2, "dismissals": 1})
    monkeypatch.setattr(N.DB, "sessions_since", lambda last, max_sessions: [])
    monkeypatch.setattr(N, "run_derive", lambda *a: {"gaps": 3, "clusters": 1, "proposed": 0})
    monkeypatch.setattr(N, "run_retune", lambda *a: {"under_trigger": 2, "covered_by_skill": {}})
    monkeypatch.setattr(N, "capture_grounded", lambda *a: {"dismissals": 1})
    monkeypatch.setattr(
        N.post_fire, "run", lambda conn, since: {"fires": 3, "empty_window": 1, "deviation": 1}
    )
    monkeypatch.setattr(N.skill_draft, "draft_pending", lambda conn: {"drafted": 0})
    monkeypatch.setattr(RQ, "_judge_call", lambda prompt: "{}")
    _cand(
        rq,
        name="rq-expiring",
        kind="derive",
        direction=None,
        target=None,
        base=None,
        body=_skill("rq-expiring", "New."),
        proposed_days_ago=30,
    )

    counts = N.run_lane(limit=5, no_discord=True)

    assert counts["sessions"] == 0 and counts["usage_dismissals"] == 1
    assert counts["derive_gaps"] == 3 and counts["retune_under_trigger"] == 2
    assert counts["post_fire_empty_window"] == 1 and counts["post_fire_deviation"] == 1
    assert counts["expired"] >= 1 and counts["auto_rejected"] == 0
    assert counts["digest_sent"] == 1  # the expiry changed the queue
    assert {"candidates_new", "candidates_merged", "to_proposed", "retired"} <= counts.keys()
    assert "covered_by_skill" not in str(counts)  # only integer stats are recorded
    assert "expired #" in capsys.readouterr().out


def test_run_once_records_the_skills_counts_in_dream_runs(monkeypatch):
    import dream.__main__ as M

    recorded = {}
    monkeypatch.setenv("SKILLS_LANE_ENABLED", "1")
    monkeypatch.setenv("CONFIG_LANE_ENABLED", "0")
    monkeypatch.setenv("SYNAPSE_NOTES_LANE", "0")
    monkeypatch.setattr(M, "_dream_record_start", lambda db_url: 1)
    monkeypatch.setattr(M, "_dream_collect_proposals", lambda db_url, run_id, counts: {})
    monkeypatch.setattr(
        M, "_dream_record_finish", lambda *a: recorded.update(stages=a[2], counts=a[3])
    )
    monkeypatch.setattr(N, "run_lane", lambda limit: {"sessions": 4, "post_fire_empty_window": 1})
    M.run_once()
    assert recorded["counts"] == {"skills_sessions": 4, "skills_post_fire_empty_window": 1}
    assert recorded["stages"]["skills"] == {"ran": True, "ok": True}
