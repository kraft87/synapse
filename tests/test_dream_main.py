"""dream/__main__.run_once: the skills lane is on by default with SKILLS_LANE_ENABLED=0 as its
kill switch (like CONFIG_LANE_ENABLED and SYNAPSE_NOTES_LANE), and the stages the lane reports
as failed make the run not ok. No DB: the lanes and the dream_runs bookkeeping are stubbed."""

from __future__ import annotations

import pytest


def _run_once(monkeypatch, lane):
    """Run run_once with only the skills lane enabled, `lane` standing in for its run_lane.
    Returns what the run recorded in dream_runs (stages, errors, ok)."""
    import dream.__main__ as dm  # imported here: it configures logging at import

    recorded: dict = {}
    monkeypatch.setattr(dm, "_require_db_url", lambda: "postgresql://unused")
    monkeypatch.setattr(dm, "_dream_record_start", lambda url: 1)
    monkeypatch.setattr(dm, "_dream_collect_proposals", lambda url, run_id, counts: {})
    monkeypatch.setattr(
        dm,
        "_dream_record_finish",
        lambda url, run_id, stages, counts, samples, errors, ok: recorded.update(
            stages=stages, errors=errors, ok=ok
        ),
    )
    monkeypatch.setenv("CONFIG_LANE_ENABLED", "0")
    monkeypatch.setenv("SYNAPSE_NOTES_LANE", "0")
    monkeypatch.setattr("dream.skills.nightly.run_lane", lane)
    dm.run_once()
    return recorded


def test_skills_lane_runs_by_default(monkeypatch):
    monkeypatch.delenv("SKILLS_LANE_ENABLED", raising=False)
    monkeypatch.delenv("SKILLS_LANE_LIMIT", raising=False)
    calls: list[int] = []

    def lane(limit):
        calls.append(limit)
        return {"stages": {}, "errors": [], "advanced": True}

    rec = _run_once(monkeypatch, lane)
    assert calls == [30]
    assert rec["stages"] == {"skills": {"ran": True, "ok": True}}
    assert rec["ok"] is True and rec["errors"] == []


@pytest.mark.parametrize("value", ["1", "yes", ""])
def test_only_zero_turns_the_skills_lane_off_or_on(monkeypatch, value):
    """ "=0" is the kill switch; any other value leaves the default (on)."""
    monkeypatch.setenv("SKILLS_LANE_ENABLED", value)
    calls: list[int] = []
    _run_once(monkeypatch, lambda limit: calls.append(limit) or {"errors": []})
    assert calls == [30]


def test_skills_lane_kill_switch(monkeypatch):
    monkeypatch.setenv("SKILLS_LANE_ENABLED", "0")
    calls: list[int] = []
    rec = _run_once(monkeypatch, lambda limit: calls.append(limit) or {"errors": []})
    assert calls == [] and rec["stages"] == {} and rec["ok"] is True


def test_a_failed_lane_stage_marks_the_run_not_ok(monkeypatch):
    monkeypatch.delenv("SKILLS_LANE_ENABLED", raising=False)
    rec = _run_once(
        monkeypatch,
        lambda limit: {
            "stages": {"judge": {}},
            "errors": ["drafts: RuntimeError: boom"],
            "advanced": True,
        },
    )
    assert rec["stages"] == {"skills": {"ran": True, "ok": False}}
    assert rec["ok"] is False
    assert rec["errors"] == ["skills lane drafts: RuntimeError: boom"]


def test_a_lane_that_raises_marks_the_run_not_ok(monkeypatch):
    monkeypatch.delenv("SKILLS_LANE_ENABLED", raising=False)

    def lane(limit):
        raise RuntimeError("usage limit")

    rec = _run_once(monkeypatch, lane)
    assert rec["stages"] == {"skills": {"ran": True, "ok": False}}
    assert rec["ok"] is False and rec["errors"] == ["skills lane: usage limit"]
