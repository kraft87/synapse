"""dream/__main__ against a database: the run lock, the dream_runs bookkeeping, the proposals
count, the catch-up check and the telemetry retention. Every row is synthetic (`dreamt-`
names and `dreamt/` file keys, recall_metrics source `dreamt`); dream_runs is emptied around
each test, as the dashboard tests do.

Covers:
  - one run at a time: a run that finds the lock held skips, and the lock frees on close
  - a run closes the dream_runs rows a dead process left open, and records its own
  - proposals_raised counts an older observe row that tonight's run promoted
  - the catch-up check: only a database with history and no finished run since the last window
  - recall_metrics rows older than the retention are pruned nightly
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

import dream.__main__ as dm


@pytest.fixture()
def runs(conn, db_url, monkeypatch):
    def _clean():
        conn.execute("TRUNCATE dream_runs RESTART IDENTITY")
        conn.execute("DELETE FROM skills_lane.skill_gap_candidates WHERE name LIKE 'dreamt-%'")
        conn.execute("DELETE FROM config_lane.config_proposals WHERE file_key LIKE 'dreamt/%'")
        conn.execute("DELETE FROM recall_metrics WHERE source = 'dreamt'")

    _clean()
    monkeypatch.setattr(dm, "_require_db_url", lambda: db_url)
    yield conn
    _clean()


def _rows(conn):
    return conn.execute(
        "SELECT ok, finished_at IS NOT NULL, stages, counts, samples, errors "
        "FROM dream_runs ORDER BY id"
    ).fetchall()


def _lane(result=None, *, effect=None):
    def run():
        if effect:
            effect()
        return result or {"counts": {}, "errors": []}

    return dm.Lane("dreamt", None, run)


# ------------------------------------------------------------------------ the run lock
def test_a_second_run_cannot_take_the_lock_until_the_first_lets_go(runs, db_url):
    with dm._run_lock(db_url) as first:
        assert first is True
        with dm._run_lock(db_url) as second:
            assert second is False
    with dm._run_lock(db_url) as again:
        assert again is True  # closing the connection released it


def test_run_once_skips_while_another_run_holds_the_lock(runs, db_url, monkeypatch):
    ran: list[str] = []
    monkeypatch.setattr(dm, "LANES", (_lane(effect=lambda: ran.append("x")),))
    with dm._run_lock(db_url):
        dm.run_once()
    assert ran == [] and _rows(runs) == []  # no lane, and no dream_runs row either
    dm.run_once()
    assert ran == ["x"] and len(_rows(runs)) == 1


# ------------------------------------------------------------------------ the record
def test_a_run_closes_the_rows_a_dead_process_left_open_and_records_its_own(runs, monkeypatch):
    runs.execute("INSERT INTO dream_runs (started_at) VALUES (now() - interval '1 day')")
    runs.execute(
        "INSERT INTO dream_runs (started_at, finished_at, ok) "
        "VALUES (now() - interval '2 days', now() - interval '2 days', true)"
    )
    monkeypatch.setattr(dm, "LANES", (_lane({"counts": {"dreamt_seen": 4}, "errors": []}),))

    dm.run_once()

    dead, done, mine = _rows(runs)
    assert dead[:2] == (False, False)  # not ok, and still no finish time: nobody knows it
    assert dead[5] == [dm._INTERRUPTED]
    assert done[:2] == (True, True) and done[5] == []  # a finished run is left alone
    assert mine[:2] == (True, True)
    assert mine[2] == {"dreamt": {"ran": True, "ok": True}}
    assert mine[3]["dreamt_seen"] == 4 and mine[5] == []

    dm.run_once()
    assert [r[5] for r in _rows(runs)][:1] == [[dm._INTERRUPTED]]  # not stamped twice


def test_a_failed_stage_is_recorded_as_a_failed_run(runs, monkeypatch):
    monkeypatch.setattr(
        dm, "LANES", (_lane({"counts": {}, "errors": ["judge: RuntimeError: boom"]}),)
    )
    dm.run_once()
    ((ok, finished, stages, _counts, _samples, errors),) = _rows(runs)
    assert (ok, finished) == (False, True)
    assert stages == {"dreamt": {"ran": True, "ok": False}}
    assert errors == ["dreamt lane judge: RuntimeError: boom"]


def test_proposals_raised_counts_an_older_row_that_the_run_promoted(runs, monkeypatch):
    runs.execute(
        "INSERT INTO skills_lane.skill_gap_candidates (kind, name, status, created_at) VALUES "
        "('derive', 'dreamt-old-observe', 'observe', now() - interval '20 days'), "
        "('derive', 'dreamt-already-proposed', 'proposed', now() - interval '20 days')"
    )
    runs.execute(
        "INSERT INTO config_lane.config_proposals (kind, file_key, status, created_at) VALUES "
        "('add', 'dreamt/old-observe.md', 'observe', now() - interval '20 days')"
    )

    def tonight():
        runs.execute(
            "UPDATE skills_lane.skill_gap_candidates SET status = 'proposed', updated_at = now() "
            "WHERE name IN ('dreamt-old-observe', 'dreamt-already-proposed')"
        )
        runs.execute(
            "UPDATE config_lane.config_proposals SET status = 'proposed' "
            "WHERE file_key = 'dreamt/old-observe.md'"
        )
        runs.execute(
            "INSERT INTO skills_lane.skill_gap_candidates (kind, name, status) "
            "VALUES ('derive', 'dreamt-new', 'proposed')"
        )

    monkeypatch.setattr(dm, "LANES", (_lane(effect=tonight),))
    dm.run_once()

    ((_ok, _fin, _stages, counts, samples, _errors),) = _rows(runs)
    assert counts["proposals_raised"] == 3  # the already-proposed row is not new
    names = {p["name"] for p in samples["proposals"]}
    assert names == {"dreamt-old-observe", "dreamt-new", "dreamt/old-observe.md"}
    assert {p["kind"] for p in samples["proposals"]} == {"skill", "config-edit"}


# ------------------------------------------------------------------------ the catch-up check
NOW = datetime(2026, 10, 15, 9, 0, tzinfo=UTC)  # the last window is 02:03 the same day


def _run_at(conn, started_at, *, finished=True):
    conn.execute(
        "INSERT INTO dream_runs (started_at, finished_at, ok) VALUES (%s, %s, %s)",
        (
            started_at,
            started_at + timedelta(minutes=5) if finished else None,
            True if finished else None,
        ),
    )


def test_a_new_install_waits_for_its_first_window(runs, db_url):
    assert dm._missed_window(db_url, NOW) is False


def test_a_run_since_the_last_window_is_not_missed(runs, db_url):
    _run_at(runs, datetime(2026, 10, 15, 2, 3, 4, tzinfo=UTC))
    assert dm._missed_window(db_url, NOW) is False


def test_the_last_nights_run_alone_means_this_night_was_missed(runs, db_url):
    _run_at(runs, datetime(2026, 10, 14, 2, 3, 4, tzinfo=UTC))
    assert dm._missed_window(db_url, NOW) is True


def test_a_run_that_never_finished_does_not_count(runs, db_url):
    _run_at(runs, datetime(2026, 10, 14, 2, 3, 4, tzinfo=UTC))
    _run_at(runs, datetime(2026, 10, 15, 2, 3, 4, tzinfo=UTC), finished=False)  # redeployed mid-run
    assert dm._missed_window(db_url, NOW) is True


def test_a_failed_run_still_counts_as_the_nights_run(runs, db_url):
    runs.execute(
        "INSERT INTO dream_runs (started_at, finished_at, ok) "
        "VALUES ('2026-10-15 02:03:04+00', '2026-10-15 02:04:00+00', false)"
    )
    assert dm._missed_window(db_url, NOW) is False  # no retry loop across restarts


def test_an_unreachable_database_waits_for_the_window(runs):
    assert dm._missed_window("postgresql://nobody@127.0.0.1:1/none", NOW) is False


# ------------------------------------------------------------------------ retention
def test_old_recall_metrics_rows_are_pruned_and_recent_ones_kept(runs):
    keep_days = dm.RECALL_METRICS_KEEP_DAYS
    runs.execute(
        "INSERT INTO recall_metrics (kind, source, created_at) VALUES "
        "('recall', 'dreamt', now() - make_interval(days => %s)), "
        "('recall', 'dreamt', now() - make_interval(days => %s)), "
        "('recall', 'dreamt', now())",
        (keep_days + 1, keep_days - 1),
    )

    result = dm._prune_telemetry()

    assert result["errors"] == [] and result["counts"]["recall_metrics_pruned"] >= 1
    left = runs.execute("SELECT count(*) FROM recall_metrics WHERE source = 'dreamt'").fetchone()
    assert left == (2,)
