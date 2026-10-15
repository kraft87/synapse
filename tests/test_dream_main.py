"""dream/__main__: the lane table, the lanes' adapters to one result shape, and the
scheduler's pure helpers. No DB: the lanes and the dream_runs bookkeeping are stubbed (the
bookkeeping, the run lock and the catch-up query against a database are in test_dream_runs.py).

Covers:
  - every lane is ON by default and `=0` on its switch turns off that lane only
  - counts, samples and errors of every lane land in the run's record
  - a failed lane stage, or a lane that raises, marks the run not ok and the later lanes still run
  - the skills, config and notes lanes' results adapt to counts a Dream report can print
  - a run that finds the run lock taken does nothing
  - the 02:03 window arithmetic, and the scheduler's catch-up run at startup
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime

import pytest

import dream.__main__ as dm

_ran: list[str] = []


@pytest.fixture(autouse=True)
def _reset_ran():
    _ran.clear()


def _lane(name, result=None, *, switch=None):
    """A Lane whose runner records that it ran and returns `result` (raises it, if an exception)."""

    def run():
        _ran.append(name)
        if isinstance(result, Exception):
            raise result
        return result or {"counts": {}, "errors": []}

    return dm.Lane(name, switch, run)


def _run_once(monkeypatch, *lanes, locked=True):
    """Run run_once over `lanes` with the DB bookkeeping stubbed. Returns what the run
    recorded in dream_runs (stages, counts, samples, errors, ok), {} if it recorded nothing."""
    recorded: dict = {}

    @contextmanager
    def run_lock(url):
        yield locked

    monkeypatch.setattr(dm, "_require_db_url", lambda: "postgresql://unused")
    monkeypatch.setattr(dm, "_run_lock", run_lock)
    monkeypatch.setattr(dm, "_dream_close_interrupted", lambda url: None)
    monkeypatch.setattr(dm, "_dream_record_start", lambda url: 1)
    monkeypatch.setattr(dm, "_proposed", lambda url: {})
    monkeypatch.setattr(dm, "_dream_collect_proposals", lambda url, before, counts: {})
    monkeypatch.setattr(
        dm,
        "_dream_record_finish",
        lambda url, run_id, stages, counts, samples, errors, ok: recorded.update(
            stages=stages, counts=counts, samples=samples, errors=errors, ok=ok
        ),
    )
    monkeypatch.setattr(dm, "LANES", lanes)
    dm.run_once()
    return recorded


# ------------------------------------------------------------------------ the lane table
def test_the_lanes_and_their_kill_switches():
    assert [(lane.name, lane.switch) for lane in dm.LANES] == [
        ("skills", "SKILLS_LANE_ENABLED"),
        ("config", "CONFIG_LANE_ENABLED"),
        ("notes", "SYNAPSE_NOTES_LANE"),
        ("retention", None),
    ]


@pytest.mark.parametrize("value", [None, "1", "yes", ""])
def test_a_lane_is_on_unless_its_switch_is_zero(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("DREAMT_SWITCH", raising=False)
    else:
        monkeypatch.setenv("DREAMT_SWITCH", value)
    rec = _run_once(monkeypatch, _lane("a", switch="DREAMT_SWITCH"))
    assert _ran == ["a"]
    assert rec["stages"] == {"a": {"ran": True, "ok": True}} and rec["ok"] is True


def test_zero_turns_off_only_that_lane(monkeypatch):
    monkeypatch.setenv("DREAMT_SWITCH", "0")
    rec = _run_once(monkeypatch, _lane("a", switch="DREAMT_SWITCH"), _lane("b"))
    assert _ran == ["b"]
    assert rec["stages"] == {"b": {"ran": True, "ok": True}} and rec["ok"] is True


# ------------------------------------------------------------------------ the record
def test_counts_samples_and_stages_of_every_lane_are_recorded(monkeypatch):
    a = {"counts": {"a_seen": 3}, "errors": []}
    b = {"counts": {"b_seen": 5}, "samples": [f"s{i}" for i in range(30)], "errors": []}
    rec = _run_once(monkeypatch, _lane("a", a), _lane("b", b))
    assert rec["counts"] == {"a_seen": 3, "b_seen": 5}
    assert rec["samples"] == {"b": [f"s{i}" for i in range(20)]}  # bounded
    assert rec["stages"] == {"a": {"ran": True, "ok": True}, "b": {"ran": True, "ok": True}}
    assert rec["errors"] == [] and rec["ok"] is True


def test_a_failed_lane_stage_marks_the_run_not_ok_and_the_later_lanes_still_run(monkeypatch):
    failed = {"counts": {"a_seen": 1}, "errors": ["drafts: RuntimeError: boom"]}
    rec = _run_once(monkeypatch, _lane("skills", failed), _lane("b"))
    assert _ran == ["skills", "b"]
    assert rec["stages"] == {"skills": {"ran": True, "ok": False}, "b": {"ran": True, "ok": True}}
    assert rec["ok"] is False
    assert rec["errors"] == ["skills lane drafts: RuntimeError: boom"]
    assert rec["counts"] == {"a_seen": 1}  # what the lane did get through is recorded


def test_a_lane_that_raises_is_recorded_and_the_later_lanes_still_run(monkeypatch):
    rec = _run_once(
        monkeypatch,
        _lane("skills", RuntimeError("usage limit")),
        _lane("b", {"counts": {"b_seen": 2}, "errors": []}),
    )
    assert _ran == ["skills", "b"]
    assert rec["stages"] == {"skills": {"ran": True, "ok": False}, "b": {"ran": True, "ok": True}}
    assert rec["ok"] is False and rec["errors"] == ["skills lane: usage limit"]
    assert rec["counts"] == {"b_seen": 2}


def test_a_run_that_finds_the_lock_taken_does_nothing(monkeypatch):
    rec = _run_once(monkeypatch, _lane("a"), locked=False)
    assert _ran == [] and rec == {}


# ------------------------------------------------------------------------ the adapters
def test_skills_lane_counts_are_flattened_from_its_stages(monkeypatch):
    seen: list[int] = []

    def run_lane(limit):
        seen.append(limit)
        return {
            "stages": {
                "usage": {"fire_events": 7, "dismissals": 2},
                "judge": {
                    "sessions": 12,
                    "unparsed": 1,
                    "derive": {"gaps": 9, "clusters": 4, "proposed": 1},
                    "retune": {"under_trigger": 3},
                },
                "struggle_arc": {"judge_failures": 2},
                "post_fire": {"judge_failures": 1, "over_cap": 4},
                "drafts": {"drafted": 2, "redrafted": 1, "errors": 1},
                "decay": {"retired": 5},
            },
            "errors": ["digest: URLError: down"],
            "advanced": True,
        }

    monkeypatch.setattr("dream.skills.nightly.run_lane", run_lane)
    monkeypatch.delenv("SKILLS_LANE_LIMIT", raising=False)
    assert dm._skills_lane() == {
        "counts": {
            "skills_sessions_judged": 12,
            "skills_judge_unparsed": 1,
            "skills_detector_unparsed": 3,
            "skills_gaps": 9,
            "skills_retune_signals": 3,
            "skills_fire_events": 7,
            "skills_fires_over_cap": 4,
            "skills_dismissals": 2,
            "skills_drafted": 3,
            "skills_draft_errors": 1,
            "skills_retired": 5,
        },
        "errors": ["digest: URLError: down"],
    }
    assert seen == [30]


def test_skills_lane_counts_zero_for_a_stage_that_did_not_run(monkeypatch):
    monkeypatch.setattr(
        "dream.skills.nightly.run_lane",
        lambda limit: {"stages": {}, "errors": ["judge: RuntimeError: boom"], "advanced": False},
    )
    res = dm._skills_lane()
    assert res["errors"] == ["judge: RuntimeError: boom"]
    assert set(res["counts"].values()) == {0}


def test_lane_limits_come_from_the_environment(monkeypatch):
    seen: dict[str, int] = {}
    monkeypatch.setenv("SKILLS_LANE_LIMIT", "7")
    monkeypatch.setenv("CONFIG_LANE_LIMIT", "11")
    monkeypatch.setattr(
        "dream.skills.nightly.run_lane",
        lambda limit: seen.update(skills=limit) or {"stages": {}, "errors": []},
    )
    monkeypatch.setattr(
        "dream.config.nightly.run_lane",
        lambda limit: seen.update(config=limit) or _CONFIG_RESULT,
    )
    dm._skills_lane()
    dm._config_lane()
    assert seen == {"skills": 7, "config": 11}


def test_a_malformed_limit_fails_that_lane_not_the_run(monkeypatch):
    monkeypatch.setenv("SKILLS_LANE_LIMIT", "lots")
    rec = _run_once(monkeypatch, dm.Lane("skills", None, dm._skills_lane), _lane("b"))
    assert rec["stages"]["skills"] == {"ran": True, "ok": False}
    assert "b" in rec["stages"] and rec["ok"] is False
    assert rec["errors"][0].startswith("skills lane: invalid literal")


_CONFIG_RESULT = {
    "sessions": 0,
    "found": 0,
    "proposed": 0,
    "recurred": 0,
    "already_known": 0,
    "unparsed": 0,
    "expired": 0,
}


def test_config_lane_counts_keep_their_names(monkeypatch):
    monkeypatch.setattr(
        "dream.config.nightly.run_lane",
        lambda limit: {
            "sessions": 12,
            "found": 5,
            "proposed": 2,
            "recurred": 1,
            "already_known": 3,
            "unparsed": 4,
            "expired": 6,
        },
    )
    assert dm._config_lane() == {
        "counts": {
            "config_sessions_scanned": 12,
            "config_judge_unparsed": 4,
            "config_corrections_found": 5,
            "config_already_known": 3,
            "config_proposals": 2,
            "config_applied_recurrences": 1,
            "config_observe_expired": 6,
        },
        "errors": [],
    }


def test_notes_judge_failures_are_counted_not_stage_errors(monkeypatch):
    monkeypatch.setattr(
        "dream.notes.nightly.run_lane",
        lambda: {
            "counts": {"pairs_judged": 4, "applied_supersede": 1},
            "samples": ["n:1 -> superseded by n:2 (duplicate)"],
            "errors": ["pair 3/4: backend down", "retype 9: backend down"],
        },
    )
    assert dm._notes_lane() == {
        "counts": {"pairs_judged": 4, "applied_supersede": 1, "notes_judge_errors": 2},
        "samples": [
            "n:1 -> superseded by n:2 (duplicate)",
            "error: pair 3/4: backend down",
            "error: retype 9: backend down",
        ],
        "errors": [],  # fail-soft per item: the lane, and the run, stay ok
    }


# ------------------------------------------------------------------------ the 02:03 window
@pytest.mark.parametrize(
    ("now", "window"),
    [
        (datetime(2026, 10, 15, 2, 2, 59, tzinfo=UTC), datetime(2026, 10, 14, 2, 3, tzinfo=UTC)),
        (datetime(2026, 10, 15, 2, 3, tzinfo=UTC), datetime(2026, 10, 15, 2, 3, tzinfo=UTC)),
        (datetime(2026, 10, 15, 23, 59, tzinfo=UTC), datetime(2026, 10, 15, 2, 3, tzinfo=UTC)),
        (datetime(2026, 10, 1, 0, 0, tzinfo=UTC), datetime(2026, 9, 30, 2, 3, tzinfo=UTC)),
    ],
)
def test_last_window_is_the_most_recent_0203(now, window):
    assert dm._last_window(now) == window


@pytest.mark.parametrize(
    ("now", "seconds"),
    [
        (datetime(2026, 10, 15, 1, 3, tzinfo=UTC), 3600),
        (datetime(2026, 10, 15, 2, 3, tzinfo=UTC), 86400),  # at the window itself: tomorrow's
        (datetime(2026, 10, 15, 3, 3, tzinfo=UTC), 23 * 3600),
    ],
)
def test_seconds_until_2am_is_the_gap_to_the_next_window(monkeypatch, now, seconds):
    class _Now(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr(dm, "datetime", _Now)
    assert dm._seconds_until_2am() == seconds


class _Stop(Exception):
    pass


def _main(monkeypatch, argv, *, missed=False, run_once=None):
    """dm.main(argv) with the schema check, logfire, the clock and the run stubbed; the first
    sleep ends the scheduler loop. Returns the calls in order."""
    calls: list = []
    monkeypatch.delenv("LOGFIRE_SERVICE_NAME", raising=False)
    monkeypatch.setattr(dm, "_require_db_url", lambda: "postgresql://unused")
    monkeypatch.setattr("ingestion.schema_check.check_schema_version", lambda url: None)
    monkeypatch.setattr(dm.logfire, "configure", lambda **kw: calls.append(("configure", kw)))
    monkeypatch.setattr(dm.logfire, "instrument_claude_agent_sdk", lambda: None)
    monkeypatch.setattr(dm.logfire, "instrument_httpx", lambda: None)
    monkeypatch.setattr(dm, "_missed_window", lambda url, now: missed)
    monkeypatch.setattr(dm, "run_once", run_once or (lambda: calls.append("run")))

    def sleep(seconds):
        calls.append("sleep")
        raise _Stop

    monkeypatch.setattr(dm.time, "sleep", sleep)
    try:
        dm.main(argv)
    except _Stop:
        pass
    return calls


def test_the_dream_reports_to_logfire_as_its_own_service(monkeypatch):
    (configure, *_) = _main(monkeypatch, [])
    assert configure == (
        "configure",
        {"service_name": "synapse-dream", "send_to_logfire": "if-token-present"},
    )


def test_a_one_shot_run_runs_once_and_does_not_schedule(monkeypatch):
    assert _main(monkeypatch, [], missed=True)[1:] == ["run"]


def test_schedule_mode_waits_for_the_window_when_it_has_not_been_missed(monkeypatch):
    assert _main(monkeypatch, ["--schedule"])[1:] == ["sleep"]


def test_schedule_mode_runs_a_missed_window_at_startup_before_waiting(monkeypatch):
    assert _main(monkeypatch, ["--schedule"], missed=True)[1:] == ["run", "sleep"]


def test_a_failing_catch_up_run_does_not_stop_the_scheduler(monkeypatch):
    def boom():
        raise RuntimeError("database down")

    assert _main(monkeypatch, ["--schedule"], missed=True, run_once=boom)[1:] == ["sleep"]
