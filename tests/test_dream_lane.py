"""dream/lane.py — the episode-id window, per-lane cursor and session walk the dream lanes
share — plus the skills and config lanes' use of it. DB-backed; every row is synthetic
(session ids `lanet-*`, lane keys `lanet*`) and cleaned up around each test.

Covers:
  - the window: from the stored cursor (else the legacy last_scan_at, else 30 days back)
    to the newest episode when the run starts; --backfill ignores the cursor
  - a turn uploaded late with an old transcript timestamp lands in the next window
  - a turn stored after the run read its bound waits for the next run
  - the session walk: oldest last touch first, capped, overflow carried to the next run
  - targeted whole-session views, and the judged-session memo
  - the cursor never moves back; before the migration nothing is stored or skipped
  - both lanes advance their cursor only past what they judged
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import psycopg
import pytest

import dream.config.nightly as CN
import dream.lane as lane
import dream.skills.nightly as N


@pytest.fixture()
def db(conn):
    def _clean():
        conn.execute("DELETE FROM episodes WHERE session_id LIKE 'lanet-%'")
        conn.execute(
            "DELETE FROM dream_lane_cursors WHERE lane LIKE 'lanet%' "
            "OR lane IN ('skills', 'config')"
        )
        conn.execute("DELETE FROM dream_session_memo WHERE session_id LIKE 'lanet-%'")
        conn.execute("DELETE FROM skills_lane.skill_usage WHERE skill = 'lanet-skill'")

    _clean()
    yield conn
    _clean()


def _max_id(conn) -> int:
    return conn.execute("SELECT COALESCE(max(id), 0) FROM episodes").fetchone()[0]


def _ep(conn, sid, seq, *, human="synthetic ask", tools=0, created_at=None, extra=""):
    """Insert one claude_code turn; returns its id."""
    content = f"[user] {human}\n" + "".join(f"[tool:Bash] ls /tmp/{i}\n" for i in range(tools))
    return conn.execute(
        "INSERT INTO episodes (session_id, sequence, platform, human_turn, content, created_at) "
        "VALUES (%s, %s, 'claude_code', %s, %s, COALESCE(%s, now())) RETURNING id",
        (sid, seq, human, content + extra, created_at),
    ).fetchone()[0]


def _seed(conn, name="lanet") -> int:
    """Point `name`'s cursor at the newest episode, so its window holds only what a test
    adds. Returns the cursor."""
    newest = _max_id(conn)
    lane.advance(conn, lane.Window(name, 0, 0), newest)
    return newest


def _open(conn, name="lanet", **kw):
    kw.setdefault("legacy_since", None)
    return lane.open_window(conn, name, **kw)


def _substantive(conn, sid, *, extra="", **kw):
    """A session is_substantive accepts (two user turns, three tool calls); `extra` goes on
    its second turn. Returns that turn's id, the session's last touch."""
    _ep(conn, sid, 1, tools=2, **kw)
    return _ep(conn, sid, 2, tools=1, extra=extra, **kw)


# ------------------------------------------------------------------------ window
def test_window_runs_from_the_stored_cursor_to_the_newest_episode(db):
    cursor = _seed(db)
    _ep(db, "lanet-a", 1)
    last = _ep(db, "lanet-a", 2)
    w = _open(db)
    assert (w.after_id, w.until_id) == (cursor, last)


def test_window_without_a_cursor_starts_at_the_legacy_time(db):
    # far-future event times keep the test independent of whatever else the table holds
    _ep(db, "lanet-old", 1, created_at=datetime(2098, 1, 1, tzinfo=UTC))
    new = _ep(db, "lanet-new", 1, created_at=datetime(2100, 1, 1, tzinfo=UTC))
    w = _open(db, legacy_since=datetime(2099, 1, 1, tzinfo=UTC))
    assert w.after_id == new - 1


def test_window_without_any_cursor_starts_thirty_days_back(db):
    _ep(db, "lanet-ancient", 1, created_at=datetime.now(UTC) - timedelta(days=400))
    recent = _ep(db, "lanet-recent", 1)
    w = _open(db)
    assert w.after_id < recent <= w.until_id


def test_backfill_ignores_the_stored_cursor(db):
    recent = _ep(db, "lanet-recent", 1)
    _seed(db)
    assert _open(db).after_id == recent  # the cursor already passed it
    assert _open(db, backfill=True).after_id < recent


def test_late_upload_with_an_old_timestamp_lands_in_the_next_window(db):
    """The bug: a created_at cursor (event time) never saw a turn uploaded after the run
    that followed its timestamp. By insertion order it is simply the next run's work."""
    _seed(db)
    _substantive(db, "lanet-ontime")
    w1 = _open(db)
    views, carry = lane.sessions(db, w1, limit=10)
    assert [v["session"] for v in views] == ["lanet-ontime"]
    lane.advance(db, w1, carry)

    month_ago = datetime.now(UTC) - timedelta(days=30)
    _substantive(db, "lanet-late", created_at=month_ago)
    views, _ = lane.sessions(db, _open(db), limit=10)
    assert [v["session"] for v in views] == ["lanet-late"]


def test_a_turn_stored_mid_run_waits_for_the_next_run(db):
    _seed(db)
    _ep(db, "lanet-a", 1)
    w1 = _open(db)
    _ep(db, "lanet-b", 1)  # lands while the run is busy
    views, carry = lane.sessions(db, w1, limit=10)
    assert [v["session"] for v in views] == ["lanet-a"]
    assert carry == w1.until_id
    lane.advance(db, w1, carry)
    views, _ = lane.sessions(db, _open(db), limit=10)
    assert [v["session"] for v in views] == ["lanet-b"]


def test_advance_never_moves_the_cursor_back(db):
    lane.advance(db, lane.Window("lanet", 0, 0), 500)
    lane.advance(db, lane.Window("lanet", 0, 0), 100)
    row = db.execute("SELECT last_episode_id FROM dream_lane_cursors WHERE lane = 'lanet'")
    assert row.fetchone()[0] == 500


# ------------------------------------------------------------------------ sessions
def test_capped_walk_takes_the_oldest_and_carries_the_rest(db):
    """At most `limit` wanted sessions, oldest last touch first; the cursor stops before
    the first one left over (passing an unwanted one on the way), so the next run gets it."""
    _seed(db)
    _substantive(db, "lanet-a")
    _substantive(db, "lanet-b")
    skipped = _ep(db, "lanet-thin", 1)  # one turn, no tools: not wanted
    _substantive(db, "lanet-c")
    from dream.skills.skill_measure import is_substantive

    w1 = _open(db)
    views, carry = lane.sessions(db, w1, limit=2, wanted=is_substantive)
    assert [v["session"] for v in views] == ["lanet-a", "lanet-b"]
    assert carry == skipped
    lane.advance(db, w1, carry)

    w2 = _open(db)
    views, carry = lane.sessions(db, w2, limit=2, wanted=is_substantive)
    assert [v["session"] for v in views] == ["lanet-c"]
    assert carry == w2.until_id


def test_a_zero_limit_is_an_error_not_a_silent_skip(db):
    with pytest.raises(ValueError, match="at least 1"):
        lane.sessions(db, _open(db), limit=0)


def test_order_is_by_last_touch_not_session_size(db):
    _seed(db)
    for seq in range(1, 6):
        _ep(db, "lanet-long", seq)
    _ep(db, "lanet-short", 1)
    _ep(db, "lanet-long", 6)  # touched again last
    views, _ = lane.sessions(db, _open(db), limit=1)
    assert [v["session"] for v in views] == ["lanet-short"]


def test_views_are_whole_sessions_up_to_the_bound(db):
    _ep(db, "lanet-a", 1, human="first ask from before the window")
    _ep(db, "lanet-untouched", 1)
    _seed(db)
    _ep(db, "lanet-a", 2, tools=1)
    w = _open(db)
    _ep(db, "lanet-a", 3, human="after the bound")
    views, _ = lane.sessions(db, w, limit=10)
    assert [v["session"] for v in views] == ["lanet-a"]
    (v,) = views
    assert v["first_user"] == "first ask from before the window"
    assert v["user_turns"] == 2 and v["tools"] == {"Bash": 1}


def test_a_judged_session_waits_until_it_grows(db):
    _seed(db)
    _ep(db, "lanet-a", 1)
    _ep(db, "lanet-a", 2)
    w = _open(db)
    views, carry = lane.sessions(db, w, limit=10)
    lane.mark_judged(db, "lanet", views)
    lane.advance(db, w, carry)

    _ep(db, "lanet-a", 3)  # one more turn: not enough to judge it again
    w = _open(db)
    views, carry = lane.sessions(db, w, limit=10)
    assert views == [] and carry == w.until_id  # passed over, the cursor moves on
    lane.advance(db, w, carry)

    for seq in range(4, 4 + lane.REJUDGE_TURNS - 1):
        _ep(db, "lanet-a", seq)
    views, _ = lane.sessions(db, _open(db), limit=10)
    assert [v["session"] for v in views] == ["lanet-a"]
    assert views[0]["user_turns"] == 2 + lane.REJUDGE_TURNS


def test_memo_is_per_lane(db):
    _seed(db, "lanet-x")
    _seed(db, "lanet-y")
    _ep(db, "lanet-a", 1)
    views, _ = lane.sessions(db, _open(db, "lanet-x"), limit=10)
    lane.mark_judged(db, "lanet-x", views)
    assert lane.sessions(db, _open(db, "lanet-x"), limit=10)[0] == []
    assert len(lane.sessions(db, _open(db, "lanet-y"), limit=10)[0]) == 1


def test_before_the_migration_nothing_is_stored_or_skipped(db, monkeypatch):
    monkeypatch.setattr(lane, "_tables_ready", lambda conn: False)
    _ep(db, "lanet-a", 1, created_at=datetime(2100, 1, 1, tzinfo=UTC))
    w = _open(db, legacy_since=datetime(2099, 1, 1, tzinfo=UTC))
    views, carry = lane.sessions(db, w, limit=10)
    assert [v["session"] for v in views] == ["lanet-a"]
    lane.mark_judged(db, "lanet", views)
    lane.advance(db, w, carry)
    monkeypatch.undo()
    assert db.execute("SELECT 1 FROM dream_lane_cursors WHERE lane = 'lanet'").fetchone() is None
    assert (
        db.execute("SELECT 1 FROM dream_session_memo WHERE session_id = 'lanet-a'").fetchone()
        is None
    )


def test_probe_answers_whether_the_tables_exist(db_url):
    with psycopg.connect(db_url, autocommit=True) as c:
        assert lane.LANE_TABLES.ready(c, "lanet-probe-present")
        c.execute("SET search_path TO pg_catalog")  # hides the tables, as before the migration
        assert not lane.LANE_TABLES.ready(c, "lanet-probe-absent")


# ------------------------------------------------------------------------ lanes
def test_config_lane_judges_its_cap_and_carries_the_rest(db, db_url, monkeypatch):
    monkeypatch.setenv("SYNAPSE_DB_URL", db_url)
    judged: list[str] = []
    monkeypatch.setattr(CN, "scan_corrections", lambda s: judged.append(s["session"]) or [])
    _seed(db, "config")
    sids = [f"lanet-c{i}" for i in range(5)]
    for sid in sids:
        _ep(db, sid, 1)

    assert CN.run_lane(limit=1)["sessions"] == 4  # the lane's cap is limit * 4
    assert judged == sids[:4]
    assert CN.run_lane(limit=1)["sessions"] == 1
    assert judged == sids
    assert CN.run_lane(limit=1)["sessions"] == 0  # all judged, nothing new


def test_skills_lane_carries_overflow_and_bounds_every_stage(db, db_url, monkeypatch):
    """With more substantive sessions than `limit`, the lane judges the oldest, every
    window stage covers only up to where that walk stopped, and the next run takes the
    rest. The window's fires are read once and shared with the grounded and post-fire
    stages."""
    monkeypatch.setenv("SYNAPSE_DB_URL", db_url)
    scanned: list[str] = []
    seen: dict[str, list] = {"struggle": [], "post_fire": []}
    fire_reads: list = []
    real_fire_events = N.DB.fire_events

    def fire_events(conn, window):
        fire_reads.append(window)
        return real_fire_events(conn, window)

    monkeypatch.setattr(N, "consolidate_pass", lambda conn, ids: {})
    monkeypatch.setattr(N.SD, "scan_session", lambda s, cat: scanned.append(s["session"]) or [])
    monkeypatch.setattr(N.SD, "cluster_gaps", lambda gaps: [])
    monkeypatch.setattr(N.SM, "judge_session", lambda s, cat: None)
    monkeypatch.setattr(N.DB, "fire_events", fire_events)
    monkeypatch.setattr(
        N.struggle_arc, "run", lambda conn, **kw: seen["struggle"].append(kw["window"]) or {}
    )
    monkeypatch.setattr(
        N.post_fire, "run", lambda conn, **kw: seen["post_fire"].append(kw["fires"]) or {}
    )
    monkeypatch.setattr(N.procedure_miner, "due", lambda conn: False)
    monkeypatch.setattr(N.skill_draft, "draft_pending", lambda conn: {})
    monkeypatch.setattr(N.L, "decay_stale", lambda conn: {})
    monkeypatch.setattr(N, "discord_digest", lambda conn, no_discord: {})

    _seed(db, "skills")
    _substantive(db, "lanet-s1", extra="[tool:Skill] {'skill': 'lanet-skill'}\n")
    second = _substantive(db, "lanet-s2")
    _substantive(db, "lanet-s3", extra="[tool:Skill] {'skill': 'lanet-skill'}\n")

    N.run_lane(limit=2, no_discord=True)
    assert scanned == ["lanet-s1", "lanet-s2"]
    (w1,) = seen["struggle"]
    assert w1.until_id == second  # stops where the session walk stopped
    assert [f["session_id"] for f in seen["post_fire"][0]] == ["lanet-s1"]
    assert len(fire_reads) == 1  # one read of the window's fires per run

    N.run_lane(limit=2, no_discord=True)
    assert scanned == ["lanet-s1", "lanet-s2", "lanet-s3"]
    assert seen["struggle"][1].after_id == second
    assert [f["session_id"] for f in seen["post_fire"][1]] == ["lanet-s3"]
