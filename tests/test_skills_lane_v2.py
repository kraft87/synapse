"""skills-lane v2 foundation — DB-backed tests (migration 045 + skill_ledger + review routes).

Runs against the shared Postgres test DB (tagged into the `db` xdist group via conftest).
All rows are synthetic (v2t- prefix) and cleaned up around each test.

Covers:
  - merge_candidate v2 gates (quote / scan-night recurrence / salience / legacy score)
  - salience max-ratchet, source_detector + proposed_patch persistence, direction widening
  - the derive-identity resolver never wildcard-matches on empty session/signature keys
  - decay_stale: 'proposed' rows are exempt, 'observe' rows retire
  - throttled discord digest: top-5 cap and new-proposals-only pinging
  - fire_events resolves fired_at from the episode's own transcript timestamp
  - the window stages read episodes by insertion order, bounded above
  - migration 045 idempotency + the ghost skill_usage repair
"""

from __future__ import annotations

import itertools
import json
import urllib.error
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import dream.lane as lane_mod
import dream.skills.nightly as N
import dream.skills.skill_db_source as DBSRC
import dream.skills.skill_ledger as L
import dream.skills.struggle_arc as SA
from mcp_server.skill_sync_routes import _proposal_detail, _proposals_list

SCHEMA = Path(__file__).resolve().parent.parent / "schema"
MIGRATION = SCHEMA / "045_skills_v2_foundation.sql"
OPEN_IDENTITY_MIGRATION = next(SCHEMA.glob("*_skill_candidate_open_identity.sql"))

_GHOSTS = ("research", "multi-research", "deep-research")


def _mc(conn, kind, name, evidence, **kw):
    kw.setdefault("do_embed", False)
    return L.merge_candidate(conn, kind, name, evidence, **kw)


@pytest.fixture()
def lane(conn):
    def _clean():
        conn.execute(
            "DELETE FROM skills_lane.skill_gap_candidates "
            "WHERE name LIKE 'v2t-%' OR name LIKE 'explicit:v2t-%'"
        )
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


def _arc(night, date):
    return [
        {
            "session_id": "v2t-sess-a",
            "class": "judge",
            "signal": "struggle_arc",
            "scan_night": night,
            "date": date,
        }
    ]


def test_derive_proposes_when_the_work_recurs_on_a_second_day(lane):
    sig = "synthetic restic nas backup rclone"
    r1 = _mc(lane, "derive", "v2t-proc", _arc("2026-07-17", "2026-07-17"), signature=sig)
    assert r1["status"] == "observe"
    # the same arc seen again by a later scan is not recurrence
    r2 = _mc(lane, "derive", "v2t-proc", _arc("2026-07-18", "2026-07-17"), signature=sig)
    assert r2["merged"] is True and r2["id"] == r1["id"]
    assert r2["status"] == "observe"
    r3 = _mc(lane, "derive", "v2t-proc", _arc("2026-07-19", "2026-07-19"), signature=sig)
    assert r3["id"] == r1["id"]
    assert r3["status"] == "proposed"


def test_a_rescan_of_the_same_arc_adds_no_evidence_or_weight(lane):
    # a held cursor re-runs struggle_arc over the same window; each night stamps a new scan_night
    sig = "synthetic restic nas backup rclone"
    r1 = _mc(lane, "derive", "v2t-proc", _arc("2026-07-17", "2026-07-16"), signature=sig)
    before = _row(lane, r1["id"], "score, jsonb_array_length(evidence)")
    for night in ("2026-07-18", "2026-07-19", "2026-07-20"):
        r = _mc(lane, "derive", "v2t-proc", _arc(night, "2026-07-16"), signature=sig)
        assert r["id"] == r1["id"] and r["status"] == "observe"
    assert _row(lane, r1["id"], "score, jsonb_array_length(evidence)") == before


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


def test_legacy_evidence_without_scan_night_is_not_recurrence(lane):
    # two legacy sightings (no scan_night, no date) collapse to one entry -> still observe
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


def _gap(sid, night="2026-07-17"):
    return [{"session_id": sid, "class": "judge", "signal": "gap_scan", "scan_night": night}]


def test_a_sighting_without_a_signature_keeps_the_rows_identity_key(lane):
    sig = "synthetic restic nas backup rclone"
    r1 = _mc(lane, "derive", "v2t-keep-key", _gap("v2t-sess-k1"), signature=sig)
    key = _row(lane, r1["id"], "signature_key, signature")
    r2 = _mc(lane, "derive", "v2t-keep-key", _gap("v2t-sess-k2", "2026-07-18"))
    assert r2["id"] == r1["id"] and r2["merged"] is True
    assert key[0] and _row(lane, r1["id"], "signature_key, signature") == key


def test_trigger_phrasings_accumulate_and_are_never_erased(lane):
    sig = "synthetic restic nas backup rclone"
    r1 = _mc(
        lane,
        "derive",
        "v2t-phr",
        _gap("v2t-sess-p1"),
        signature=sig,
        trigger_phrasings=["back up the nas", "nas backup"],
    )

    def phrasings():
        return _row(lane, r1["id"], "trigger_phrasings")[0]

    # a merge that brings none (procedure_miner, a phrasing-less verdict) leaves them be
    _mc(lane, "derive", "v2t-phr", _gap("v2t-sess-p2", "2026-07-18"), signature=sig)
    assert phrasings() == ["back up the nas", "nas backup"]
    # a merge that brings some adds the new ones, deduped, oldest first
    _mc(
        lane,
        "derive",
        "v2t-phr",
        _gap("v2t-sess-p3", "2026-07-19"),
        signature=sig,
        trigger_phrasings=["nas backup", "snapshot the nas"],
    )
    assert phrasings() == ["back up the nas", "nas backup", "snapshot the nas"]


def test_an_explicit_request_is_its_own_candidate_and_leaves_the_sessions_gap_alone(
    lane, monkeypatch
):
    gap = _mc(
        lane,
        "derive",
        "v2t-session-gap",
        _gap("v2t-sess-x"),
        signature="synthetic restic nas backup rclone",
        summary="back up the nas",
        trigger_phrasings=["nas backup"],
    )
    before = _row(
        lane, gap["id"], "summary, signature_key, trigger_phrasings, grounded_weight, status"
    )
    monkeypatch.setattr(
        N.DB,
        "explicit_skill_requests",
        lambda conn, window: [
            {"session_id": "v2t-sess-x", "ts": None, "phrasing": "make this into a skill"}
        ],
    )
    N.capture_grounded(lane, None, [], set(), catalog=[])
    N.capture_grounded(lane, None, [], set(), catalog=[])  # a re-scan adds nothing
    assert (
        _row(lane, gap["id"], "summary, signature_key, trigger_phrasings, grounded_weight, status")
        == before
    )
    (n, weight) = lane.execute(
        "SELECT count(*), max(grounded_weight) FROM skills_lane.skill_gap_candidates "
        "WHERE kind='derive' AND name LIKE 'explicit:v2t-sess%'"
    ).fetchone()
    assert (n, weight) == (1, 3.0)


def test_a_sighting_joins_its_gap_not_the_explicit_request_made_in_the_same_session(lane):
    sig = "synthetic restic nas backup rclone"
    gap = _mc(lane, "derive", "v2t-rev-nas-backup", _gap("v2t-sess-s0"), signature=sig)
    # the user asks for a skill in another session; capture_grounded stores it as its own row
    request = _mc(
        lane,
        "derive",
        "explicit:v2t-sess-s1",
        [{"session_id": "v2t-sess-s1", "class": "grounded", "signal": "explicit_request"}],
        summary="make this a skill",
    )
    cols = "summary, signature_key, evidence, grounded_weight, status"
    before = _row(lane, request["id"], cols)
    # struggle_arc then sights the same procedure in that session: it belongs to the gap row
    # (same signature), not to the request row that only shares the session
    sighting = _mc(
        lane,
        "derive",
        "v2t-rev-struggle",
        [
            {
                "session_id": "v2t-sess-s1",
                "class": "judge",
                "signal": "struggle_arc",
                "date": "2026-07-18",
            }
        ],
        signature=sig,
        summary="backing up the nas",
    )
    assert sighting["id"] == gap["id"] and sighting["merged"] is True
    assert _row(lane, request["id"], cols) == before


def test_only_a_dismissal_of_a_registered_skill_becomes_a_narrow_candidate(lane, monkeypatch):
    monkeypatch.setattr(N.DB, "explicit_skill_requests", lambda conn, window: [])
    fires = [
        {"skill": "v2t-owned", "session_id": "v2t-sess-d1", "dismissed": True},
        {"skill": "v2t-builtin", "session_id": "v2t-sess-d1", "dismissed": True},
        {"skill": "v2t-owned", "session_id": "v2t-sess-d2", "dismissed": False},
    ]

    stats = N.capture_grounded(lane, None, fires, set(), catalog=[("v2t-owned", "does a thing")])

    assert stats["dismissals"] == 1
    names = lane.execute(
        "SELECT name FROM skills_lane.skill_gap_candidates "
        "WHERE name LIKE 'v2t-%' AND direction = 'narrow'"
    ).fetchall()
    assert names == [("v2t-owned",)]  # the built-in has nothing to draft against


# ------------------------------------------- terminal-state re-sightings (unique key)
# A retune's (name, direction) is unique among OPEN rows only (the open-identity migration: observe, proposed,
# rejected). A promoted or retired row is history: a sighting after it opens a fresh candidate
# instead of being folded into the closed one, and an INSERT never hits a UniqueViolation.


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


def _promote(conn, cid):
    """Promote the way the review path does: grounded 'accept' evidence first, then the status
    flip (grounded_required_to_apply needs grounded_weight > 0 on promoted rows)."""
    conn.execute(
        "UPDATE skills_lane.skill_gap_candidates SET status='promoted', "
        'evidence = evidence || \'[{"session_id": null, "class": "grounded", '
        '"signal": "accept"}]\'::jsonb, grounded_weight=3.0 WHERE id=%s',
        (cid,),
    )


def test_a_sighting_after_an_accept_opens_a_fresh_candidate(lane):
    r1 = _retune(lane, "v2t-sess-t1")
    _promote(lane, r1["id"])
    # the nightly lane sees the same skill under-fire again: a new problem after the change
    r2 = _retune(lane, "v2t-sess-t2")
    assert r2["id"] != r1["id"] and r2["merged"] is False
    assert r2["status"] == "observe"
    assert _row(lane, r1["id"], "status, jsonb_array_length(evidence)") == ("promoted", 2)
    (new_ev,) = _row(lane, r2["id"], "evidence")
    assert {e["session_id"] for e in new_ev} == {"v2t-sess-t2"}  # post-change evidence only
    # and the next sighting joins the open row, not the history
    r3 = _retune(lane, "v2t-sess-t3")
    assert r3["id"] == r2["id"] and r3["merged"] is True


def test_a_sighting_after_a_retirement_opens_a_fresh_candidate(lane):
    r1 = _retune(lane, "v2t-sess-t1")
    lane.execute(
        "UPDATE skills_lane.skill_gap_candidates SET status='retired', reject_reason='stale' "
        "WHERE id=%s",
        (r1["id"],),
    )
    r2 = _retune(lane, "v2t-sess-t2")
    assert r2["id"] != r1["id"] and r2["merged"] is False and r2["status"] == "observe"
    assert _row(lane, r1["id"], "status") == ("retired",)


def test_before_the_open_identity_migration_a_sighting_of_a_closed_retune_is_dropped_loudly(
    lane, caplog
):
    old_index = (
        "CREATE UNIQUE INDEX skill_gap_named_idx ON skills_lane.skill_gap_candidates "
        "(kind, name, COALESCE(direction, '-')) WHERE kind IN ('retune', 'consolidate')"
    )
    lane.execute("DROP INDEX skills_lane.skill_gap_named_idx")
    lane.execute(old_index)
    try:
        r1 = _retune(lane, "v2t-sess-t1")
        _promote(lane, r1["id"])
        r2 = _retune(lane, "v2t-sess-t2")  # the closed row still occupies the key
        assert r2["id"] == r1["id"] and r2["merged"] is False and r2["status"] == "promoted"
        assert _row(lane, r1["id"], "jsonb_array_length(evidence)") == (2,)  # nothing recorded
        assert "sighting of" in caplog.text and "open-identity migration" in caplog.text
        # the same for a retired row, which used to be reopened
        caplog.clear()
        lane.execute(
            "UPDATE skills_lane.skill_gap_candidates SET status='retired' WHERE id=%s", (r1["id"],)
        )
        r3 = _retune(lane, "v2t-sess-t3")
        assert r3["id"] == r1["id"] and r3["status"] == "retired" and r3["merged"] is False
        assert "open-identity migration" in caplog.text
    finally:
        lane.execute(OPEN_IDENTITY_MIGRATION.read_text())


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


# --------------------------------------------- what an accepted retune did to its skill
_usage_ids = itertools.count(1)


def _usage(conn, skill, days_ago, *, dismissed=False, outcome=None):
    conn.execute(
        "INSERT INTO skills_lane.skill_usage (skill, fired_at, session_id, via, dismissed, outcome) "
        "VALUES (%s, now() - make_interval(days => %s), %s, 'tool', %s, %s)",
        (skill, days_ago, f"v2t-eff-{next(_usage_ids)}", dismissed, outcome),
    )


def _promoted(conn, name, days_ago, kind="retune"):
    conn.execute(
        "INSERT INTO skills_lane.skill_gap_candidates "
        "(kind, name, direction, target_skills, status, grounded_weight, updated_at) "
        "VALUES (%s, %s, 'fix', %s, 'promoted', 3.0, now() - make_interval(days => %s))",
        (kind, name, [name], days_ago),
    )


def test_the_post_fire_verdict_is_kept_on_the_usage_row(lane):
    import dream.skills.post_fire as PF

    fired_at = datetime(2026, 7, 18, 12, tzinfo=UTC)
    fire = {"skill": "v2t-kept", "session_id": "v2t-sess-o", "fired_at": fired_at}
    lane.execute(
        "INSERT INTO skills_lane.skill_usage (skill, fired_at, session_id, via, dismissed) "
        "VALUES ('v2t-kept', %s, 'v2t-sess-o', 'tool', false)",
        (fired_at,),
    )
    for verdict, stored in (("deviation", "partial"), ("clean", "worked"), ("fight", "failed")):
        PF._record_outcome(lane, fire, verdict)
        assert lane.execute(
            "SELECT outcome FROM skills_lane.skill_usage WHERE skill='v2t-kept'"
        ).fetchone() == (stored,)


def test_skill_effects_compares_the_span_since_the_accept_with_the_span_before(lane):
    # accepted 14 days ago: 14 days before it against the 14 since
    _promoted(lane, "v2t-eff-skill", 14)
    for ago, kw in ((20, {}), (18, {"dismissed": True}), (16, {"dismissed": True}), (29, {})):
        _usage(lane, "v2t-eff-skill", ago, **kw)  # before; 29 is outside the 14-day span
    for ago, kw in (
        (10, {"outcome": "worked"}),
        (8, {"outcome": "partial"}),
        (3, {"outcome": "worked"}),
        (1, {"outcome": "unknown"}),
    ):
        _usage(lane, "v2t-eff-skill", ago, **kw)
    _usage(lane, "v2t-other-skill", 5)  # another skill's fires are not counted
    # too fresh to say, too old to attribute, and not a retune: all left out
    _promoted(lane, "v2t-eff-fresh", 3)
    _promoted(lane, "v2t-eff-old", 40)
    _promoted(lane, "v2t-eff-derive", 14, kind="derive")

    r = N.skill_effects(lane)

    assert r["measured"] == 1
    assert r["lines"] == [
        "v2t-eff-skill (fix, 14d ago): fires 3 -> 4 · dismissed 2/3 -> 0/4 · deviated - -> 1/3"
    ]


# -------------------------------------------------------------- digest throttle


def _counts(result):
    """The digest's proposal counts, without the served-description size it also reports."""
    return {k: result[k] for k in ("proposed", "new")}


def test_digest_caps_at_five_and_pings_only_on_new(lane, capsys):
    # the shared test DB may hold other proposed rows; run this window empty of them
    lane.execute(
        "UPDATE skills_lane.skill_gap_candidates SET status='observe' WHERE status='proposed'"
    )
    for i in range(7):
        _mc(
            lane,
            "retune",
            f"v2t-digest-{i}",
            [
                {
                    "session_id": f"v2t-sess-d{i}",
                    "class": "judge",
                    "signal": "post_fire_deviation",
                    "quote": f"synthetic quote {i}",
                }
            ],
            direction="fix",
            target_skills=[f"v2t-digest-{i}"],
            salience=(i % 5) + 1,
        )
    r1 = N.discord_digest(lane, no_discord=True)
    out1 = capsys.readouterr().out
    assert _counts(r1) == {"proposed": 7, "new": 7}
    assert out1.count("- `retune`") == 5  # capped at DIGEST_CAP
    assert "+2 more pending (skill_review list)" in out1
    assert "sal 5" in out1.splitlines()[1]  # ranked salience DESC first

    # same pending set again -> no ping, no re-wall
    r2 = N.discord_digest(lane, no_discord=True)
    out2 = capsys.readouterr().out
    assert _counts(r2) == {"proposed": 7, "new": 0}
    assert "- `retune`" not in out2

    # one genuinely new proposal -> pings again
    _mc(
        lane,
        "retune",
        "v2t-digest-new",
        [
            {
                "session_id": "v2t-sess-dn",
                "class": "judge",
                "signal": "post_fire_deviation",
                "quote": "synthetic fresh quote",
            }
        ],
        direction="fix",
        target_skills=["v2t-digest-new"],
    )
    r3 = N.discord_digest(lane, no_discord=True)
    assert r3["proposed"] == 8 and r3["new"] == 1


def test_digest_reports_the_size_of_the_global_descriptions(lane, capsys):
    def size():
        r = N.discord_digest(lane, no_discord=True)
        return r["global_skills"], r["global_description_chars"]

    try:
        n0, chars0 = size()
        lane.execute(
            "INSERT INTO skills_lane.skill_registry (name, scope, body, description, status) "
            "VALUES ('v2t-big', 'global', 'b', %s, 'active'), "
            "('v2t-proj', 'project:v2t', 'b', %s, 'active'), "
            "('v2t-gone', 'global', 'b', %s, 'retired')",
            ("d" * 300, "p" * 50, "r" * 70),
        )
        assert size() == (n0 + 1, chars0 + 300)  # project and retired skills are not counted
        _mc(
            lane,
            "retune",
            "v2t-digest-size",
            [
                {
                    "session_id": "v2t-sess-z",
                    "class": "judge",
                    "signal": "post_fire_deviation",
                    "quote": "synthetic",
                }
            ],
            direction="fix",
            target_skills=["v2t-digest-size"],
        )
        capsys.readouterr()
        N.discord_digest(lane, no_discord=True)
        assert f"{n0 + 1} skills, {chars0 + 300} chars" in capsys.readouterr().out
    finally:
        lane.execute("DELETE FROM skills_lane.skill_registry WHERE name LIKE 'v2t-%'")


def test_digest_shows_what_the_accepted_changes_did(lane, capsys):
    lane.execute(
        "UPDATE skills_lane.skill_gap_candidates SET status='observe' WHERE status='proposed'"
    )
    _mc(
        lane,
        "retune",
        "v2t-digest-effects",
        [
            {
                "session_id": "v2t-sess-ef",
                "class": "judge",
                "signal": "post_fire_deviation",
                "quote": "synthetic",
            }
        ],
        direction="fix",
        target_skills=["v2t-digest-effects"],
    )
    line = "v2t-x (fix, 14d ago): fires 3 -> 4 · dismissed 2/3 -> 0/4 · deviated - -> 1/3"
    N.discord_digest(lane, no_discord=True, effects=[line])
    out = capsys.readouterr().out
    assert "Accepted changes" in out and f"- {line}" in out


def test_a_failed_discord_post_raises_and_the_batch_pings_again_next_run(lane, monkeypatch, capsys):
    lane.execute(
        "UPDATE skills_lane.skill_gap_candidates SET status='observe' WHERE status='proposed'"
    )
    _mc(
        lane,
        "retune",
        "v2t-digest-hook",
        [
            {
                "session_id": "v2t-sess-hook",
                "class": "judge",
                "signal": "post_fire_deviation",
                "quote": "synthetic quote",
            }
        ],
        direction="fix",
        target_skills=["v2t-digest-hook"],
    )
    posts: list[dict] = []

    def urlopen(req, timeout):
        posts.append(json.loads(req.data))
        if len(posts) == 1:
            raise urllib.error.URLError("discord is down")

    monkeypatch.setattr(N.config, "DISCORD_WEBHOOK", "https://discord.example/hook")
    monkeypatch.setattr(N.urllib.request, "urlopen", urlopen)

    with pytest.raises(urllib.error.URLError):  # the stage fails, so the run is recorded not ok
        N.discord_digest(lane, no_discord=False)
    assert "v2t-digest-hook" in capsys.readouterr().out  # the digest is still in the log
    assert _counts(N.discord_digest(lane, no_discord=False)) == {"proposed": 1, "new": 1}  # again
    assert [p["content"].count("v2t-digest-hook") for p in posts] == [1, 1]
    assert _counts(N.discord_digest(lane, no_discord=False)) == {"proposed": 1, "new": 0}
    assert len(posts) == 2


def test_run_config_update_does_not_clobber_digest_state(lane):
    lane.execute(
        "UPDATE skills_lane.skill_scan_cursor SET config = COALESCE(config,'{}'::jsonb) "
        "|| '{\"digest_ids\": [1, 2]}'::jsonb WHERE id=1"
    )
    L.update_cursor(lane, datetime.now(UTC), config={"limit": 40})
    cfg = L.get_cursor(lane)["config"]
    assert cfg["digest_ids"] == [1, 2] and cfg["limit"] == 40


# ------------------------------------------------------- fired_at resolution


def _max_id(conn):
    return conn.execute("SELECT COALESCE(max(id), 0) FROM episodes").fetchone()[0]


def _window_over(conn, insert):
    """A lane window holding exactly the episodes `insert()` adds."""
    after = _max_id(conn)
    insert()
    return lane_mod.Window("v2t", after, _max_id(conn))


def test_fire_events_use_transcript_time_not_ingest_clock(lane):
    event_ts = "2026-05-14T10:00:00Z"
    window = _window_over(
        lane,
        lambda: lane.execute(
            "INSERT INTO episodes (session_id, sequence, platform, content, metadata, created_at) "
            "VALUES ('v2t-fire-sess', 1, 'claude_code', %s, %s::jsonb, now())",
            (
                "[user] synthetic request\n[tool:Skill] {'skill': 'v2t-demo-skill'}",
                json.dumps({"ts": event_ts}),
            ),
        ),
    )
    events = DBSRC.fire_events(lane, window)
    assert len(events) == 1
    assert events[0]["skill"] == "v2t-demo-skill"
    assert events[0]["fired_at"] == datetime(2026, 5, 14, 10, 0, tzinfo=UTC)


def test_fire_events_fall_back_to_created_at(lane):
    """A metadata-less episode falls back to created_at for its fire timestamp."""
    fired_at = datetime.now(UTC).replace(microsecond=0) - timedelta(days=30)
    window = _window_over(
        lane,
        lambda: lane.execute(
            "INSERT INTO episodes (session_id, sequence, platform, content, metadata, created_at) "
            "VALUES ('v2t-fire-sess2', 1, 'claude_code', %s, '{}'::jsonb, %s)",
            ("[tool:Skill] {'skill': 'v2t-demo-skill'}", fired_at),
        ),
    )
    events = DBSRC.fire_events(lane, window)
    assert len(events) == 1
    assert events[0]["fired_at"] == fired_at


def test_fire_events_read_an_offset_less_transcript_time_as_utc(lane):
    """post_fire orders the window's fires by fired_at; a naive timestamp among aware ones
    would make that sort raise."""
    window = _window_over(
        lane,
        lambda: lane.execute(
            "INSERT INTO episodes (session_id, sequence, platform, content, metadata) "
            "VALUES ('v2t-fire-naive', 1, 'claude_code', %s, %s::jsonb)",
            ("[tool:Skill] {'skill': 'v2t-demo-skill'}", json.dumps({"ts": "2026-05-14T10:00:00"})),
        ),
    )
    (event,) = DBSRC.fire_events(lane, window)
    assert event["fired_at"] == datetime(2026, 5, 14, 10, 0, tzinfo=UTC)


def test_window_stages_see_late_uploads_and_nothing_past_the_bound(lane):
    """The window stages (fire_events, explicit_skill_requests, the struggle-arc prescreen)
    read the window by insertion order: a turn uploaded today with a month-old transcript
    timestamp is in it, and a turn stored after the run read its bound is not (it belongs
    to the next run)."""
    old = datetime.now(UTC) - timedelta(days=30)
    ask = "please make this into a skill"

    def insert():
        lane.execute(
            "INSERT INTO episodes (session_id, sequence, platform, human_turn, content, "
            "created_at) VALUES ('v2t-late', 1, 'claude_code', %s, %s, %s)",
            (ask, "[tool:Skill] {'skill': 'v2t-demo-skill'}", old),
        )

    window = _window_over(lane, insert)
    lane.execute(
        "INSERT INTO episodes (session_id, sequence, platform, human_turn, content) "
        "VALUES ('v2t-after', 1, 'claude_code', %s, %s)",
        (ask, "[tool:Skill] {'skill': 'v2t-demo-skill'}"),
    )
    assert [e["session_id"] for e in DBSRC.fire_events(lane, window)] == ["v2t-late"]
    assert [r["session_id"] for r in DBSRC.explicit_skill_requests(lane, window)] == ["v2t-late"]
    assert [e["session_id"] for e in SA._fetch_new_episodes(lane, window)] == ["v2t-late"]


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
