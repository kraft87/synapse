"""skills-lane v2 foundation — DB-backed tests (migration 045 + skill_ledger + review routes).

Runs against the shared Postgres test DB (tagged into the `db` xdist group via conftest).
All rows are synthetic (v2t- prefix) and cleaned up around each test.

Covers:
  - merge_candidate v2 gates (quote / session-day recurrence / salience / legacy score)
  - salience max-ratchet, source_detector + proposed_patch persistence, direction widening
  - the derive-identity resolver never wildcard-matches on empty session/signature keys,
    and takes rejected and recently retired rows (cooldown, -3 and reopen kept)
  - decay_stale: 'proposed' rows are exempt, 'observe' rows retire
  - fire_events resolves fired_at from the episode's own transcript timestamp, and reads
    dismissals from the turn after the skill loader's episode
  (the report that replaced the digest: tests/test_skills_review_queue.py)
  - migration 045 idempotency + the ghost skill_usage repair
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import dream.skills.skill_db_source as DBSRC
import dream.skills.skill_ledger as L
from mcp_server.skill_sync_routes import _proposal_detail, _proposals_list

MIGRATION = Path(__file__).resolve().parent.parent / "schema" / "045_skills_v2_foundation.sql"

_GHOSTS = ("research", "multi-research", "deep-research")


def _mc(conn, kind, name, evidence, **kw):
    kw.setdefault("do_embed", False)
    return L.merge_candidate(conn, kind, name, evidence, **kw)


@pytest.fixture()
def lane(conn):
    def _clean():
        conn.execute("DELETE FROM skills_lane.skill_gap_candidates WHERE name LIKE 'v2t-%'")
        conn.execute(
            "DELETE FROM skills_lane.skill_usage WHERE skill LIKE 'v2t-%' "
            "OR skill IN ('research', 'multi-research', 'deep-research', 'run-research')"
        )
        conn.execute("DELETE FROM episodes WHERE session_id LIKE 'v2t-%'")
        conn.execute(
            "UPDATE skills_lane.skill_scan_cursor SET config = config - 'digest_ids' WHERE id=1"
        )

    _clean()
    yield conn
    _clean()


def _row(conn, cid, cols="status, salience, source_detector, proposed_patch"):
    return conn.execute(
        f"SELECT {cols} FROM skills_lane.skill_gap_candidates WHERE id=%s", (cid,)
    ).fetchone()


# ------------------------------------------------------------------ v2 gates


def test_retune_proposes_on_single_quoted_deviation(lane):
    r = _mc(
        lane,
        "retune",
        "v2t-demo-skill",
        [
            {
                "session_id": "v2t-sess-1",
                "class": "judge",
                "signal": "post_fire_deviation",
                "quote": "synthetic: the agent skipped the verify step",
                "scan_night": "2026-07-18",
                "date": "2026-07-17",
            }
        ],
        direction="fix",
        target_skills=["v2t-demo-skill"],
        summary="post-fire deviation",
        salience=3,
        source_detector="post_fire",
        proposed_patch="- add an explicit verify step after deploy",
    )
    assert r["status"] == "proposed"
    status, sal, det, patch = _row(lane, r["id"])
    assert (status, sal, det) == ("proposed", 3, "post_fire")
    assert "verify step" in patch


def test_retune_without_quote_stays_observe(lane):
    r = _mc(
        lane,
        "retune",
        "v2t-demo-skill",
        [{"session_id": "v2t-sess-1", "class": "judge", "signal": "under_trigger"}],
        direction="widen",
        target_skills=["v2t-demo-skill"],
    )
    assert r["status"] == "observe"


_SIG = "synthetic restic nas backup rclone"


def _gap(sid, night, date=None):
    e = {"session_id": sid, "class": "judge", "signal": "gap_scan", "scan_night": night}
    if date:
        e["date"] = date
    return e


def _session_on(conn, sid, day):
    conn.execute(
        "INSERT INTO episodes (session_id, sequence, platform, content, created_at) "
        "VALUES (%s, 1, 'claude_code', '[user] synthetic', %s)",
        (sid, f"{day}T15:00:00Z"),
    )


def test_derive_one_session_rescanned_next_night_stays_observe(lane):
    # one long session that sessions_since re-scans the next night is not recurrence
    r1 = _mc(
        lane, "derive", "v2t-proc", [_gap("v2t-sess-a", "2026-07-17", "2026-07-16")], signature=_SIG
    )
    r2 = _mc(
        lane, "derive", "v2t-proc", [_gap("v2t-sess-a", "2026-07-18", "2026-07-16")], signature=_SIG
    )
    assert r2["merged"] is True and r2["id"] == r1["id"]
    assert r2["status"] == "observe"
    # and the second night's entry is not a second judge vote (score stays one sighting)
    assert _row(lane, r1["id"], cols="judge_weight, score") == (1.0, 0.5)


def test_derive_proposes_on_a_second_session_on_another_day(lane):
    r1 = _mc(
        lane, "derive", "v2t-proc", [_gap("v2t-sess-a", "2026-10-24", "2026-10-23")], signature=_SIG
    )
    assert r1["status"] == "observe"
    r2 = _mc(
        lane, "derive", "v2t-proc", [_gap("v2t-sess-b", "2026-10-29", "2026-10-28")], signature=_SIG
    )
    assert r2["id"] == r1["id"] and r2["status"] == "proposed"


def test_derive_two_sessions_on_the_same_day_stay_observe(lane):
    _mc(
        lane, "derive", "v2t-proc", [_gap("v2t-sess-a", "2026-07-30", "2026-07-29")], signature=_SIG
    )
    r = _mc(
        lane, "derive", "v2t-proc", [_gap("v2t-sess-b", "2026-07-30", "2026-07-29")], signature=_SIG
    )
    assert r["merged"] is True and r["status"] == "observe"


def test_undated_legacy_evidence_takes_session_days_from_episodes(lane):
    # rows written before the nightly stamped dates: the gate looks the days up from the
    # sessions' episodes and stores them on the entries
    _session_on(lane, "v2t-sess-old", "2026-10-23")
    _session_on(lane, "v2t-sess-new", "2026-10-28")
    legacy = [{"session_id": "v2t-sess-old", "class": "judge", "signal": "gap_scan"}]
    r1 = _mc(lane, "derive", "v2t-proc", legacy, signature=_SIG)
    assert r1["status"] == "observe"
    new = [{"session_id": "v2t-sess-new", "class": "judge", "signal": "gap_scan"}]
    r2 = _mc(lane, "derive", "v2t-proc", new, signature=_SIG)
    assert r2["id"] == r1["id"] and r2["status"] == "proposed"
    ev = _row(lane, r1["id"], cols="evidence")[0]
    assert sorted(e["date"] for e in ev) == ["2026-10-23", "2026-10-28"]


def test_rejected_derive_in_cooldown_absorbs_a_resighting(lane):
    r1 = _mc(lane, "derive", "v2t-proc", [_gap("v2t-sess-a", "2026-07-17")], signature=_SIG)
    lane.execute(
        "UPDATE skills_lane.skill_gap_candidates SET status='rejected', "
        "reject_reason='user_rejected', rejected_until=now() + interval '30 days' WHERE id=%s",
        (r1["id"],),
    )
    r2 = _mc(lane, "derive", "v2t-proc-renamed", [_gap("v2t-sess-a", "2026-09-01")], signature=_SIG)
    assert r2 == {"id": r1["id"], "status": "rejected", "score": None, "merged": False}
    n = lane.execute(
        "SELECT count(*) FROM skills_lane.skill_gap_candidates WHERE name LIKE 'v2t-proc%%'"
    ).fetchone()[0]
    assert n == 1  # no fresh row beside the rejected one


def test_rejected_derive_past_cooldown_reopens_with_its_reject(lane):
    r1 = _mc(lane, "derive", "v2t-proc", [_gap("v2t-sess-a", "2026-07-17")], signature=_SIG)
    reject = {"session_id": None, "class": "grounded", "signal": "reject"}
    lane.execute(
        "UPDATE skills_lane.skill_gap_candidates SET status='rejected', reject_reason='user_rejected', "
        "rejected_until=now() - interval '1 day', evidence = evidence || %s::jsonb, "
        "grounded_weight=-3 WHERE id=%s",
        (json.dumps([reject]), r1["id"]),
    )
    r2 = _mc(lane, "derive", "v2t-proc", [_gap("v2t-sess-b", "2026-09-01")], signature=_SIG)
    assert r2["id"] == r1["id"] and r2["merged"] is True
    gw = _row(lane, r1["id"], cols="grounded_weight")[0]
    assert gw == -3.0  # the human reject came along


@pytest.mark.parametrize(("retired_days_ago", "reopens"), [(30, True), (90, False)])
def test_recently_retired_derive_reopens(lane, retired_days_ago, reopens):
    r1 = _mc(lane, "derive", "v2t-proc", [_gap("v2t-sess-a", "2026-07-17")], signature=_SIG)
    lane.execute(
        "UPDATE skills_lane.skill_gap_candidates SET status='retired', reject_reason='stale', "
        "updated_at = now() - make_interval(days => %s) WHERE id=%s",
        (retired_days_ago, r1["id"]),
    )
    r2 = _mc(lane, "derive", "v2t-proc", [_gap("v2t-sess-b", "2026-09-01")], signature=_SIG)
    assert (r2["id"] == r1["id"]) is reopens
    if reopens:
        assert r2["status"] == "observe" and _row(lane, r1["id"])[0] == "observe"


def test_an_open_row_wins_a_tie_with_a_cooling_rejected_one(lane):
    # the duplicate pair the audit found (#2 rejected, #129 open): a new sighting that
    # matches both equally joins the open row instead of vanishing into the cooldown
    old = _mc(lane, "derive", "v2t-proc", [_gap("v2t-sess-a", "2026-07-17")], signature=_SIG)
    lane.execute(
        "UPDATE skills_lane.skill_gap_candidates SET status='rejected', "
        "rejected_until=now() + interval '30 days' WHERE id=%s",
        (old["id"],),
    )
    lane.execute(
        "INSERT INTO skills_lane.skill_gap_candidates (kind, name, signature_key, evidence) "
        "VALUES ('derive', 'v2t-proc-dup', %s, %s::jsonb)",
        (L.signature_key(_SIG, None), json.dumps([_gap("v2t-sess-b", "2026-08-01")])),
    )
    r = _mc(lane, "derive", "v2t-proc", [_gap("v2t-sess-c", "2026-09-01")], signature=_SIG)
    assert r["id"] != old["id"] and r["merged"] is True


def test_derive_proposes_on_salience(lane):
    r = _mc(
        lane,
        "derive",
        "v2t-painful-proc",
        [
            {
                "session_id": "v2t-sess-b",
                "class": "judge",
                "signal": "struggle_arc",
                "scan_night": "2026-07-18",
            }
        ],
        salience=4,
        source_detector="struggle_arc",
    )
    assert r["status"] == "proposed"


def test_derive_legacy_score_gate_still_works(lane):
    r = _mc(
        lane,
        "derive",
        "v2t-grounded-proc",
        [{"session_id": "v2t-sess-c", "class": "grounded", "signal": "explicit_request"}],
    )
    assert r["status"] == "proposed"  # grounded weight 3.0 >= 1.5


def test_salience_ratchets_up_never_down(lane):
    ev = [
        {
            "session_id": "v2t-sess-d",
            "class": "judge",
            "signal": "struggle_arc",
            "scan_night": "2026-07-17",
        }
    ]
    r = _mc(lane, "derive", "v2t-ratchet", ev, salience=2, signature="synthetic ratchet sig")
    assert _row(lane, r["id"])[1] == 2
    _mc(lane, "derive", "v2t-ratchet", ev, salience=1, signature="synthetic ratchet sig")
    assert _row(lane, r["id"])[1] == 2  # lower new value does not overwrite
    r3 = _mc(lane, "derive", "v2t-ratchet", ev, salience=5, signature="synthetic ratchet sig")
    assert _row(lane, r["id"])[1] == 5
    assert r3["status"] == "proposed"  # 5 >= PROPOSE_SALIENCE


def test_legacy_evidence_without_scan_night_counts_one_bucket(lane):
    # two legacy sightings (no scan_night) = one recurrence bucket -> still observe
    for _ in range(2):
        r = _mc(
            lane,
            "derive",
            "v2t-legacy-proc",
            [{"session_id": "v2t-sess-e", "class": "judge", "signal": "gap_scan"}],
            signature="synthetic legacy sig",
        )
    assert r["status"] == "observe"
    ev = _row(lane, r["id"], cols="evidence")[0]
    assert len(ev) == 1  # legacy dedup collapse preserved


def test_extend_and_fix_directions_accepted(lane):
    for direction in ("extend", "fix"):
        r = _mc(
            lane,
            "retune",
            f"v2t-dir-{direction}",
            [{"session_id": "v2t-sess-f", "class": "judge", "signal": "post_fire_deviation"}],
            direction=direction,
            target_skills=[f"v2t-dir-{direction}"],
        )
        assert _row(lane, r["id"])[0] == "observe"


def test_consolidate_does_not_propose_on_quote(lane):
    r = _mc(
        lane,
        "consolidate",
        "v2t-a+v2t-b",
        [
            {
                "session_id": None,
                "class": "judge",
                "signal": "overlap",
                "quote": "synthetic near-duplicate description",
            }
        ],
        target_skills=["v2t-a", "v2t-b"],
    )
    assert r["status"] == "observe"


def test_empty_keys_never_wildcard_merge(lane):
    anchor = _mc(
        lane,
        "derive",
        "v2t-anchor",
        [{"session_id": "v2t-sess-anchor", "class": "judge", "signal": "gap_scan"}],
        signature="synthetic anchor restic nas",
    )
    # no session ids, no signature: must NOT match (and clobber) the anchor row
    stray = _mc(
        lane,
        "derive",
        "v2t-stray",
        [{"session_id": None, "class": "judge", "signal": "struggle_arc"}],
    )
    assert stray["merged"] is False
    assert stray["id"] != anchor["id"]
    # and an empty-keyed row on file must not swallow later empty-keyed submissions either
    stray2 = _mc(
        lane,
        "derive",
        "v2t-stray-2",
        [{"session_id": None, "class": "judge", "signal": "struggle_arc"}],
    )
    assert stray2["merged"] is False
    assert stray2["id"] != stray["id"]


# ------------------------------------------- terminal-state re-sightings (unique key)
# retune/consolidate share ONE row per (kind, name, direction) regardless of status
# (skill_gap_named_idx has no status filter): a nightly re-derivation of a candidate
# whose row went terminal must merge/skip, never INSERT into a UniqueViolation.


def _under_trigger(sid):
    return [{"session_id": sid, "class": "judge", "signal": "under_trigger"}]


def _retune(conn, sid, name="v2t-terminal-skill"):
    return _mc(
        conn,
        "retune",
        name,
        _under_trigger(sid),
        direction="widen",
        target_skills=[name],
        summary="under-fires: synthetic",
    )


def test_resighting_promoted_row_records_evidence_without_insert(lane):
    r1 = _retune(lane, "v2t-sess-t1")
    # promote the way the review path does: grounded 'accept' evidence first, then the
    # status flip (grounded_required_to_apply needs grounded_weight > 0 on promoted rows)
    lane.execute(
        "UPDATE skills_lane.skill_gap_candidates SET status='promoted', "
        'evidence = evidence || \'[{"session_id": null, "class": "grounded", '
        '"signal": "accept"}]\'::jsonb, grounded_weight=3.0 WHERE id=%s',
        (r1["id"],),
    )
    # the nightly lane re-derives the same (kind, name, direction) candidate
    r2 = _retune(lane, "v2t-sess-t2")
    assert r2["id"] == r1["id"] and r2["merged"] is True
    assert r2["status"] == "promoted"  # sighting recorded, never re-proposed
    status, ev, gw, n = lane.execute(
        "SELECT status, evidence, grounded_weight, "
        "(SELECT count(*) FROM skills_lane.skill_gap_candidates "
        "WHERE name='v2t-terminal-skill') FROM skills_lane.skill_gap_candidates WHERE id=%s",
        (r1["id"],),
    ).fetchone()
    assert status == "promoted" and n == 1
    assert gw > 0  # full-evidence rollup kept the grounded 'accept' — constraint holds
    assert {e["session_id"] for e in ev} == {"v2t-sess-t1", "v2t-sess-t2", None}


def test_resighting_rejected_row_in_cooldown_skips_silently(lane):
    r1 = _retune(lane, "v2t-sess-t1")
    lane.execute(
        "UPDATE skills_lane.skill_gap_candidates SET status='rejected', "
        "reject_reason='user_rejected', rejected_until=now() + interval '30 days' "
        "WHERE id=%s",
        (r1["id"],),
    )
    r2 = _retune(lane, "v2t-sess-t2")
    assert r2["id"] == r1["id"] and r2["merged"] is False
    assert r2["status"] == "rejected"
    status, reason, ev = lane.execute(
        "SELECT status, reject_reason, evidence FROM skills_lane.skill_gap_candidates WHERE id=%s",
        (r1["id"],),
    ).fetchone()
    # untouched: still rejected, cooldown intact, the new sighting NOT merged in
    assert (status, reason) == ("rejected", "user_rejected")
    assert {e["session_id"] for e in ev} == {"v2t-sess-t1"}


def test_resighting_rejected_row_past_cooldown_reopens(lane):
    r1 = _retune(lane, "v2t-sess-t1")
    lane.execute(
        "UPDATE skills_lane.skill_gap_candidates SET status='rejected', "
        "reject_reason='user_rejected', rejected_until=now() - interval '1 day' "
        "WHERE id=%s",
        (r1["id"],),
    )
    r2 = _retune(lane, "v2t-sess-t2")
    assert r2["id"] == r1["id"] and r2["merged"] is True
    assert r2["status"] == "observe"  # reopened; gates re-decide from here
    status, reason, until = lane.execute(
        "SELECT status, reject_reason, rejected_until FROM skills_lane.skill_gap_candidates "
        "WHERE id=%s",
        (r1["id"],),
    ).fetchone()
    assert (status, reason, until) == ("observe", None, None)


# --------------------------------------------------------------------- decay


def test_decay_retires_observe_but_never_proposed(lane):
    obs = _mc(
        lane,
        "derive",
        "v2t-decay-obs",
        [{"session_id": "v2t-sess-g", "class": "judge", "signal": "gap_scan"}],
        signature="synthetic decay observe",
    )
    prop = _mc(
        lane,
        "retune",
        "v2t-decay-prop",
        [
            {
                "session_id": "v2t-sess-h",
                "class": "judge",
                "signal": "post_fire_deviation",
                "quote": "synthetic quote",
            }
        ],
        direction="fix",
        target_skills=["v2t-decay-prop"],
    )
    assert (obs["status"], prop["status"]) == ("observe", "proposed")
    lane.execute(
        "UPDATE skills_lane.skill_gap_candidates SET last_seen = now() - interval '60 days' "
        "WHERE name LIKE 'v2t-decay-%'"
    )
    L.decay_stale(lane)
    assert _row(lane, obs["id"])[0] == "retired"
    assert _row(lane, prop["id"])[0] == "proposed"  # waits for human review


# ------------------------------------------------------------- cursor config


def test_run_config_update_does_not_clobber_digest_state(lane):
    lane.execute(
        "UPDATE skills_lane.skill_scan_cursor SET config = COALESCE(config,'{}'::jsonb) "
        "|| '{\"digest_ids\": [1, 2]}'::jsonb WHERE id=1"
    )
    L.update_cursor(lane, datetime.now(UTC), config={"limit": 40})
    cfg = L.get_cursor(lane)["config"]
    assert cfg["digest_ids"] == [1, 2] and cfg["limit"] == 40


# ------------------------------------------------------- fired_at resolution


def test_fire_events_use_transcript_time_not_ingest_clock(lane, db_url, monkeypatch):
    monkeypatch.setenv("SYNAPSE_DB_URL", db_url)
    event_ts = "2026-05-14T10:00:00Z"
    lane.execute(
        "INSERT INTO episodes (session_id, sequence, platform, content, metadata, created_at) "
        "VALUES ('v2t-fire-sess', 1, 'claude_code', %s, %s::jsonb, now())",
        (
            "[user] synthetic request\n[tool:Skill] {'skill': 'v2t-demo-skill'}",
            json.dumps({"ts": event_ts}),
        ),
    )
    events = [e for e in DBSRC.fire_events(days=30) if e["session_id"] == "v2t-fire-sess"]
    assert len(events) == 1
    assert events[0]["skill"] == "v2t-demo-skill"
    assert events[0]["fired_at"] == datetime(2026, 5, 14, 10, 0, tzinfo=UTC)


def test_fire_events_fall_back_to_created_at(lane, db_url, monkeypatch):
    """A metadata-less episode falls back to created_at for its fire timestamp.

    The row is dated RELATIVE to now (30 days back, inside the 90-day window). It was
    pinned at a literal 2026-06-01, which quietly aged out of that window on 2026-08-30
    and turned a real assertion into a calendar failure — the fallback under test has
    nothing to do with any particular date."""
    monkeypatch.setenv("SYNAPSE_DB_URL", db_url)
    fired_at = datetime.now(UTC).replace(microsecond=0) - timedelta(days=30)
    lane.execute(
        "INSERT INTO episodes (session_id, sequence, platform, content, metadata, created_at) "
        "VALUES ('v2t-fire-sess2', 1, 'claude_code', %s, '{}'::jsonb, %s)",
        ("[tool:Skill] {'skill': 'v2t-demo-skill'}", fired_at),
    )
    events = [e for e in DBSRC.fire_events(days=90) if e["session_id"] == "v2t-fire-sess2"]
    assert len(events) == 1
    assert events[0]["fired_at"] == fired_at


def _fire_session(conn, sid, turns):
    """Episodes 1..n for one session: (content, human_turn) pairs, a minute apart."""
    t0 = datetime.now(UTC).replace(microsecond=0) - timedelta(days=1)
    for i, (content, human) in enumerate(turns, start=1):
        conn.execute(
            "INSERT INTO episodes (session_id, sequence, platform, content, human_turn, "
            "metadata, created_at) VALUES (%s, %s, 'claude_code', %s, %s, '{}'::jsonb, %s)",
            (sid, i, content, human, t0 + timedelta(minutes=i)),
        )


def test_dismissal_reads_the_reply_after_the_loader_episode(lane, db_url, monkeypatch):
    # prod's layout: the call, then the loader's banner + body as its own episode, then
    # the user. The loader episode stood in for the user's reply, so no fire ever dismissed.
    monkeypatch.setenv("SYNAPSE_DB_URL", db_url)
    banner = "Base directory for this skill: C:\\Users\\someone\\.claude\\skills\\v2t-demo-skill"
    _fire_session(
        lane,
        "v2t-dismiss-sess",
        [
            ("[user] deploy it\n\n[tool:Skill] {'skill': 'v2t-demo-skill'}", "deploy it"),
            (f"[user] {banner}\n\n# Demo\n\nsteps\n[assistant] on it", f"{banner}\n\n# Demo"),
            ("[user] no, that's wrong\n[assistant] sorry", "no, that's wrong"),
        ],
    )
    events = [e for e in DBSRC.fire_events(days=30) if e["session_id"] == "v2t-dismiss-sess"]
    assert [(e["skill"], e["dismissed"]) for e in events] == [("v2t-demo-skill", True)]


def test_dismissal_without_a_loader_episode_reads_the_next_turn(lane, db_url, monkeypatch):
    monkeypatch.setenv("SYNAPSE_DB_URL", db_url)
    _fire_session(
        lane,
        "v2t-dismiss-sess2",
        [
            ("[tool:Skill] {'skill': 'v2t-demo-skill'}", "deploy it"),
            ("[user] no, that's wrong", "no, that's wrong"),
        ],
    )
    events = [e for e in DBSRC.fire_events(days=30) if e["session_id"] == "v2t-dismiss-sess2"]
    assert [e["dismissed"] for e in events] == [True]


# ------------------------------------------------- migration 045 / data repair


def test_migration_045_is_idempotent_and_repairs_ghosts(lane):
    ts = "2026-05-14T09:00:00Z"
    for skill in ("run-research", *_GHOSTS):
        lane.execute(
            "INSERT INTO skills_lane.skill_usage (skill, fired_at, session_id) "
            "VALUES (%s, %s, 'v2t-ghost-sess')",
            (skill, ts),
        )
    # a same-named row that does NOT share a run-research timestamp must survive
    lane.execute(
        "INSERT INTO skills_lane.skill_usage (skill, fired_at, session_id) "
        "VALUES ('research', '2026-06-02T09:00:00Z', 'v2t-other-sess')"
    )
    lane.execute(MIGRATION.read_text())  # re-run the full migration (idempotent)
    rows = {
        (r[0], r[1])
        for r in lane.execute(
            "SELECT skill, session_id FROM skills_lane.skill_usage "
            "WHERE session_id LIKE 'v2t-%%' AND skill IN ('run-research', %s, %s, %s)",
            _GHOSTS,
        ).fetchall()
    }
    assert rows == {("run-research", "v2t-ghost-sess"), ("research", "v2t-other-sess")}


# ------------------------------------------------------------- review routes


def test_proposal_routes_serve_v2_fields(lane, db_url):
    r = _mc(
        lane,
        "retune",
        "v2t-route-skill",
        [
            {
                "session_id": "v2t-sess-r",
                "class": "judge",
                "signal": "post_fire_deviation",
                "quote": "synthetic deviation quote",
                "scan_night": "2026-07-18",
            }
        ],
        direction="fix",
        target_skills=["v2t-route-skill"],
        salience=4,
        source_detector="post_fire",
        proposed_patch="- synthetic patch line",
    )
    listed = [p for p in _proposals_list(db_url)["proposals"] if p["name"] == "v2t-route-skill"]
    assert listed and listed[0]["salience"] == 4
    assert listed[0]["source_detector"] == "post_fire"
    detail = _proposal_detail(db_url, r["id"])
    assert detail["proposed_patch"] == "- synthetic patch line"
    assert detail["evidence"][0]["quote"] == "synthetic deviation quote"
