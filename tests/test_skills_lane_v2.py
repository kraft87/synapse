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
  - migration 045 idempotency + the ghost skill_usage repair
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

import dream.skills.nightly as N
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


def test_derive_proposes_on_second_scan_night(lane):
    ev1 = [
        {
            "session_id": "v2t-sess-a",
            "class": "judge",
            "signal": "struggle_arc",
            "scan_night": "2026-07-17",
        }
    ]
    r1 = _mc(lane, "derive", "v2t-proc", ev1, signature="synthetic restic nas backup rclone")
    assert r1["status"] == "observe"
    ev2 = [
        {
            "session_id": "v2t-sess-a",
            "class": "judge",
            "signal": "struggle_arc",
            "scan_night": "2026-07-18",
        }
    ]
    r2 = _mc(lane, "derive", "v2t-proc", ev2, signature="synthetic restic nas backup rclone")
    assert r2["merged"] is True
    assert r2["id"] == r1["id"]
    assert r2["status"] == "proposed"


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


# -------------------------------------------------------------- digest throttle


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
    assert r1 == {"proposed": 7, "new": 7}
    assert out1.count("- `retune`") == 5  # capped at DIGEST_CAP
    assert "+2 more pending (skill_review list)" in out1
    assert "sal 5" in out1.splitlines()[1]  # ranked salience DESC first

    # same pending set again -> no ping, no re-wall
    r2 = N.discord_digest(lane, no_discord=True)
    out2 = capsys.readouterr().out
    assert r2 == {"proposed": 7, "new": 0}
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
    monkeypatch.setenv("SYNAPSE_DB_URL", db_url)
    lane.execute(
        "INSERT INTO episodes (session_id, sequence, platform, content, metadata, created_at) "
        "VALUES ('v2t-fire-sess2', 1, 'claude_code', %s, '{}'::jsonb, "
        "'2026-06-01T12:00:00Z')",
        ("[tool:Skill] {'skill': 'v2t-demo-skill'}",),
    )
    events = [e for e in DBSRC.fire_events(days=90) if e["session_id"] == "v2t-fire-sess2"]
    assert len(events) == 1
    assert events[0]["fired_at"] == datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


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
