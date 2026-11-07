"""The dream->config review queue looks after itself: expiry, stored recommendations, the report
(dream/config/review_queue.py, schema 065). DB-backed, the judge stubbed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from dream.config import nightly as N
from dream.config import review_queue as RQ


@pytest.fixture()
def q(conn):
    conn.execute("TRUNCATE config_lane.config_proposals RESTART IDENTITY")
    conn.execute("UPDATE skills_lane.skill_scan_cursor SET config = '{}'::jsonb WHERE id=1")
    return conn


def seed(conn, rule="Rule one", sessions=("s1", "s2"), status="proposed", days_ago=0):
    ev = [{"session_id": s, "quote": f"user said it in {s}"} for s in sessions]
    return conn.execute(
        "INSERT INTO config_lane.config_proposals "
        "(kind, file_key, diff, summary, evidence, status, proposed_at, updated_at) "
        "VALUES ('add','rules/learned.md',%s,%s,%s::jsonb,%s, "
        "        CASE WHEN %s::text = 'proposed' THEN now() - make_interval(days => %s) END, "
        "        now() - make_interval(days => %s)) RETURNING id",
        (rule, rule, json.dumps(ev), status, status, days_ago, days_ago),
    ).fetchone()[0]


def status_of(conn, rid):
    return conn.execute(
        "SELECT status, reject_reason FROM config_lane.config_proposals WHERE id=%s", (rid,)
    ).fetchone()


# ---------------------------------------------------------------- expiry
def test_a_proposal_unanswered_for_21_days_is_rejected_as_expired(q):
    old = seed(q, "Old rule", days_ago=22)
    fresh = seed(q, "Fresh rule", days_ago=20)
    waiting = seed(q, "Waiting rule", sessions=("s1",), status="observe", days_ago=40)
    q.execute(
        'UPDATE config_lane.config_proposals SET recommendation = \'{"verdict":"accept"}\' '
        "WHERE id=%s",
        (old,),
    )
    out = RQ.expire(q)
    assert out["expired"] == 1
    item = out["items"][0]
    assert (
        item["id"] == old and item["days"] == 22 and item["recommendation"]["verdict"] == "accept"
    )
    assert status_of(q, old) == ("rejected", "expired")
    assert status_of(q, fresh)[0] == "proposed"
    assert status_of(q, waiting)[0] == "observe"  # only proposals expire


def test_expiry_days_come_from_proposed_at_not_created_at(q):
    rid = seed(q, days_ago=5)
    q.execute(
        "UPDATE config_lane.config_proposals SET created_at = now() - interval '90 days' WHERE id=%s",
        (rid,),
    )
    assert (
        RQ.expire(q)["expired"] == 0
    )  # a row that sat in observe starts its 21 days when proposed


# ---------------------------------------------------------------- recommendation
def stub(verdict="accept", reason="A standing preference.", calls=None):
    def _call(prompt):
        if calls is not None:
            calls.append(prompt)
        return json.dumps({"verdict": verdict, "reason": reason})

    return _call


def rec_of(conn, rid):
    return conn.execute(
        "SELECT recommendation FROM config_lane.config_proposals WHERE id=%s", (rid,)
    ).fetchone()[0]


def test_recommendation_is_stored_once_per_rule_and_evidence(q, monkeypatch):
    rid = seed(q, "Write short answers")
    calls: list[str] = []
    monkeypatch.setattr(N, "_judge_call", stub(calls=calls))
    out = RQ.recommend(q)
    assert out["recommended"] == 1 and out["failed"] == 0
    rec = rec_of(q, rid)
    assert rec["verdict"] == "accept" and rec["reason"] == "A standing preference."
    assert rec["key"] and rec["model"] and rec["at"]
    # the judge sees the rule and one quote per session, and is told to reject opposites
    assert "Write short answers" in calls[0] and "user said it in s1" in calls[0]
    assert "user said it in s2" in calls[0] and "(2 sessions)" in calls[0]
    assert "opposite" in calls[0] and "reject" in calls[0]
    # asked once: a second night makes no call
    assert RQ.recommend(q)["pending"] == 0 and len(calls) == 1
    # a new session joins the evidence: asked again
    q.execute(
        "UPDATE config_lane.config_proposals SET evidence = evidence || "
        '\'[{"session_id":"s3","quote":"a third time"}]\'::jsonb WHERE id=%s',
        (rid,),
    )
    assert RQ.recommend(q)["recommended"] == 1 and len(calls) == 2
    assert "a third time" in calls[1] and "(3 sessions)" in calls[1]


def test_only_proposed_rows_are_judged(q, monkeypatch):
    seed(q, "Observed", sessions=("s1",), status="observe")
    seed(q, "Rejected", status="rejected")
    calls: list[str] = []
    monkeypatch.setattr(N, "_judge_call", stub(calls=calls))
    assert RQ.recommend(q)["pending"] == 0 and calls == []


def test_a_failed_judge_call_is_retried_tomorrow(q, monkeypatch):
    rid = seed(q)
    monkeypatch.setattr(N, "_judge_call", lambda p: "I cannot decide")
    assert RQ.recommend(q) == {"pending": 1, "recommended": 0, "failed": 1, "deferred": 0}
    assert rec_of(q, rid) is None

    def boom(p):
        raise RuntimeError("timeout")

    monkeypatch.setattr(N, "_judge_call", boom)
    assert RQ.recommend(q)["failed"] == 1
    monkeypatch.setattr(N, "_judge_call", stub(verdict="reject", reason="Two quotes conflict."))
    assert RQ.recommend(q)["recommended"] == 1 and rec_of(q, rid)["verdict"] == "reject"


def test_recommendation_calls_are_capped_per_night(q, monkeypatch):
    for i in range(4):
        seed(q, f"Rule {i}")
    monkeypatch.setattr(N, "_judge_call", stub())
    out = RQ.recommend(q, cap=3)
    assert out["recommended"] == 3 and out["deferred"] == 1


def test_a_reason_is_one_plain_line(q, monkeypatch):
    rid = seed(q)
    monkeypatch.setattr(
        N, "_judge_call", stub(reason="**Good**  rule\n— it's `general`" + " x" * 200)
    )
    RQ.recommend(q)
    reason = rec_of(q, rid)["reason"]
    assert "\n" not in reason and "**" not in reason and "`" not in reason
    assert "—" not in reason and len(reason) <= 240


# ---------------------------------------------------------------- report text
def pending(**kw):
    p = {
        "id": 1,
        "summary": "Write short answers",
        "evidence": [{"session_id": "s1"}, {"session_id": "s2"}],
        "recommendation": {"verdict": "accept", "reason": "A standing preference."},
        "days": 0,
        "new": True,
    }
    p.update(kw)
    return p


def test_report_text_new_proposal_with_verdict_first():
    msg = RQ.build_report("11-07", [pending()], [], [])
    lines = msg.splitlines()
    assert lines[0] == "Config lane 11-07: 1 new, 1 pending."
    assert lines[1] == (
        "accept #1 Write short answers (new tonight, 2 sessions): A standing preference."
    )
    assert "Ask-only" in lines[-1]


def test_report_text_covered_and_expired_lines():
    covered = [{"id": 4, "summary": "Be brief", "covered_by": "rules/a.md: keep answers short"}]
    expired = [
        {
            "id": 5,
            "summary": "Old rule",
            "days": 21,
            "recommendation": {"verdict": "reject", "reason": "x"},
        }
    ]
    msg = RQ.build_report(
        "11-07", [pending(new=False, days=9, recommendation=None)], covered, expired
    )
    assert msg.splitlines()[0] == (
        "Config lane 11-07: 1 already covered by your rules, 1 expired, 1 pending."
    )
    assert "dropped #4 Be brief: already in your rules (rules/a.md: keep answers short)." in msg
    assert "expired #5 Old rule after 21d unanswered (recommendation was reject)." in msg
    assert "It comes back if you correct it again." in msg
    assert "no verdict yet #1 Write short answers (9d pending, 2 sessions)" in msg


def test_report_weekly_reminder_text():
    assert RQ.build_report("11-07", [], [], []) == (
        "Config lane weekly reminder 11-07: nothing pending."
    )
    reminder = RQ.build_report("11-07", [pending(new=False, days=3)], [], [])
    assert reminder.splitlines()[0] == "Config lane weekly reminder 11-07: 1 pending."


def test_report_is_capped_and_says_how_many_more():
    many = [pending(id=i, summary=f"Rule number {i} " + "word " * 15) for i in range(60)]
    msg = RQ.build_report("11-07", many, [], [])
    assert len(msg) <= RQ.REPORT_CHARS
    assert "more: config_review.py list" in msg and msg.splitlines()[0].startswith(
        "Config lane 11-07: 60 new"
    )


# ---------------------------------------------------------------- report delivery
def cursor_config(conn):
    return conn.execute("SELECT config FROM skills_lane.skill_scan_cursor WHERE id=1").fetchone()[0]


def run_started():
    return datetime.now(UTC)


def test_report_is_stored_next_to_the_skills_key_and_printed(q, capsys):
    q.execute('UPDATE skills_lane.skill_scan_cursor SET config = \'{"last_report": {"at": "x"}}\'')
    rid = seed(q)
    now = run_started() - timedelta(
        hours=1
    )  # the row is proposed after this run began: new tonight
    q.execute("UPDATE config_lane.config_proposals SET proposed_at = now() WHERE id=%s", (rid,))
    stats = RQ.report(q, now, {})
    assert stats == {"pending": 1, "sent": 1, "reminder": 0}
    cfg = cursor_config(q)
    assert cfg["last_report"] == {"at": "x"}  # the skills lane's key is left alone
    assert cfg["config_last_report"]["text"].startswith("Config lane ")
    assert cfg["config_last_report"]["at"] and cfg["config_report_sent_at"]
    assert cfg["config_last_report"]["text"] in capsys.readouterr().out  # no webhook: stdout


def test_a_quiet_night_makes_no_report_until_the_reminder_is_due(q):
    seed(q, days_ago=3)  # pending since before this run
    now = run_started()
    q.execute(
        "UPDATE skills_lane.skill_scan_cursor SET config = %s::jsonb WHERE id=1",
        (json.dumps({"config_report_sent_at": (now - timedelta(days=2)).isoformat()}),),
    )
    stats = RQ.report(q, now, {})
    assert stats == {"pending": 1, "sent": 0, "reminder": 0}
    assert "config_last_report" not in cursor_config(q)
    # a week on, the reminder goes out
    q.execute(
        "UPDATE skills_lane.skill_scan_cursor SET config = %s::jsonb WHERE id=1",
        (json.dumps({"config_report_sent_at": (now - timedelta(days=8)).isoformat()}),),
    )
    stats = RQ.report(q, now, {})
    assert stats == {"pending": 1, "sent": 1, "reminder": 1}
    assert "weekly reminder" in cursor_config(q)["config_last_report"]["text"]


def test_report_posts_to_the_webhook_when_one_is_set(q, monkeypatch):
    seed(q)
    sent = []

    class _Resp:
        def read(self):
            return b""

    def fake_urlopen(req, timeout=0):
        sent.append(json.loads(req.data)["content"])
        return _Resp()

    monkeypatch.setattr(RQ.config, "DISCORD_WEBHOOK", "https://discord.invalid/hook")
    monkeypatch.setattr(RQ.urllib.request, "urlopen", fake_urlopen)
    stats = RQ.report(q, run_started() - timedelta(days=1), {})
    assert stats["sent"] == 1 and sent and sent[0].startswith("Config lane ")
    assert cursor_config(q)["config_last_report"]["text"] == sent[0]


def test_a_failed_post_keeps_the_report_and_does_not_stamp_it_sent(q, monkeypatch):
    seed(q)

    def down(req, timeout=0):
        raise OSError("webhook down")

    monkeypatch.setattr(RQ.config, "DISCORD_WEBHOOK", "https://discord.invalid/hook")
    monkeypatch.setattr(RQ.urllib.request, "urlopen", down)
    stats = RQ.report(q, run_started() - timedelta(days=1), {})
    assert stats["sent"] == 0
    cfg = cursor_config(q)
    assert "config_last_report" in cfg and "config_report_sent_at" not in cfg  # retry tomorrow
