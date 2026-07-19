"""Pure-logic tests for the post-fire outcome detector (dream/skills/post_fire).

No DB, no LLM: fetchers and the judge call are monkeypatched, merge_candidate is stubbed
to the pinned v2 contract. Guards the two data-quality rules (fire position resolved from
content markers — never fired_at alone; session-tail fires marked unassessable and never
judged), the outcome tally, and the retune call shapes (signal='post_fire_deviation',
proposed_patch bullets, direction fallback). All fixtures synthetic.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta

import pytest

import dream.skills.post_fire as PF

_T0 = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _ep(seq, content="", human="", minutes=0):
    return {
        "sequence": seq,
        "created_at": _T0 + timedelta(minutes=minutes),
        "human_turn": human,
        "content": content,
    }


_FIRE = "[tool:Skill] {'skill': 'widget-deploy'}"


# ------------------------------------------------------------ fire-position resolution
def test_marker_beats_backfilled_timestamp():
    # backfill scenario: fired_at was stamped at ingest time, hours after the episodes
    eps = [
        _ep(1, human="deploy the widget please"),
        _ep(3, content=_FIRE, minutes=1),
        _ep(8, minutes=7),
    ]
    late_fired_at = _T0 + timedelta(hours=9)  # closest episode by time is seq 8 — a lie
    assert PF.resolve_fire_position(eps, "widget-deploy", late_fired_at) == 3


def test_base_directory_marker_resolves():
    eps = [_ep(2, content="[result] Base directory for this skill: /skills/widget-deploy")]
    assert PF.resolve_fire_position(eps, "widget-deploy", _T0) == 2


def test_multiple_fires_pick_closest_to_fired_at():
    eps = [_ep(2, content=_FIRE, minutes=0), _ep(9, content=_FIRE, minutes=60)]
    assert PF.resolve_fire_position(eps, "widget-deploy", _T0 + timedelta(minutes=58)) == 9
    assert PF.resolve_fire_position(eps, "widget-deploy", _T0 + timedelta(minutes=2)) == 2


def test_no_marker_returns_none():
    eps = [_ep(1, content="[tool:Bash] echo hello"), _ep(2, human="thanks")]
    assert PF.resolve_fire_position(eps, "widget-deploy", _T0) is None


def test_other_skills_fire_does_not_match():
    eps = [_ep(1, content="[tool:Skill] {'skill': 'other-skill'}")]
    assert PF.resolve_fire_position(eps, "widget-deploy", _T0) is None


# ------------------------------------------------------------------ small helpers
def test_patch_text_renders_bullets():
    assert PF._patch_text(["add a check", "verify restart"]) == "- add a check\n- verify restart"
    assert PF._patch_text("already bulleted") == "already bulleted"
    assert PF._patch_text([]) is None
    assert PF._patch_text(None) is None


def test_salience_clamp():
    assert PF._clamp_salience(9) == 5
    assert PF._clamp_salience("2") == 2
    assert PF._clamp_salience(None) is None


# --------------------------------------------------------------- run() harness
def _stub_merge(calls):
    """The pinned v2 merge_candidate contract — kwargs are additive on foundation's side."""

    def fake(
        conn,
        kind,
        name,
        evidence_entries,
        *,
        signature=None,
        tools=None,
        summary="",
        trigger_phrasings=None,
        target_skills=None,
        direction=None,
        salience=None,
        source_detector=None,
        proposed_patch=None,
        do_embed=True,
    ):
        calls.append(
            {
                "kind": kind,
                "name": name,
                "evidence": evidence_entries,
                "summary": summary,
                "target_skills": target_skills,
                "direction": direction,
                "salience": salience,
                "source_detector": source_detector,
                "proposed_patch": proposed_patch,
                "do_embed": do_embed,
            }
        )
        return {"id": len(calls), "status": "observe", "score": 0.5, "merged": False}

    return fake


def _wire(monkeypatch, *, fires, episodes, verdict_json, calls, doc=("desc", "body")):
    seen = {}

    def fetch_fires(conn, since):
        seen["since"] = since
        return fires

    monkeypatch.setattr(PF, "_fetch_fires", fetch_fires)
    monkeypatch.setattr(PF, "_fetch_session_episodes", lambda conn, sid: episodes[sid])
    monkeypatch.setattr(PF, "_load_skill_doc", lambda conn, name: doc)
    prompts = []

    def judge(prompt, model=None):
        prompts.append(prompt)
        return verdict_json

    monkeypatch.setattr(PF, "_judge_call", judge)
    monkeypatch.setattr(PF.L, "merge_candidate", _stub_merge(calls))
    return prompts, seen


def _fire(skill="widget-deploy", sid="s1", minutes=1):
    return {"skill": skill, "session_id": sid, "fired_at": _T0 + timedelta(minutes=minutes)}


_SESSION = {
    "s1": [
        _ep(1, human="deploy the widget please"),
        _ep(2, content=_FIRE, minutes=1),
        _ep(3, human="that skipped the restart step", minutes=2),
        _ep(4, content="[assistant] restarting now", minutes=3),
    ]
}

_DEVIATION_VERDICT = {
    "outcome": "deviation",
    "why": "skipped a step the skill mandates",
    "direction": "fix",
    "salience": 3,
    "quote": "that skipped the restart step",
    "summary": "The deploy skill omits the restart verification step.",
    "proposed_patch": ["add explicit restart step", "verify service is up before done"],
}


def test_deviation_emits_retune_call_shape(monkeypatch):
    calls = []
    prompts, seen = _wire(
        monkeypatch,
        fires=[_fire()],
        episodes=_SESSION,
        verdict_json=json.dumps(_DEVIATION_VERDICT),
        calls=calls,
    )
    since = _T0 - timedelta(days=1)
    stats = PF.run(None, since=since)
    assert seen["since"] == since  # watermark passed through untouched
    assert stats == {
        "fires": 1,
        "clean": 0,
        "deviation": 1,
        "fight": 0,
        "unassessable": 0,
        "last_episode": 0,
        "unlocated": 0,
        "candidates": 1,
        "judge_failures": 0,
    }
    c = calls[0]
    assert c["kind"] == "retune" and c["name"] == "widget-deploy"
    assert c["target_skills"] == ["widget-deploy"] and c["direction"] == "fix"
    assert c["salience"] == 3 and c["source_detector"] == "post_fire"
    assert c["proposed_patch"] == "- add explicit restart step\n- verify service is up before done"
    assert c["do_embed"] is False
    (ev,) = c["evidence"]
    assert ev["class"] == "judge" and ev["signal"] == "post_fire_deviation"
    assert ev["session_id"] == "s1" and ev["quote"] == "that skipped the restart step"
    assert _DATE_RE.match(ev["scan_night"]) and _DATE_RE.match(ev["date"])
    # prompt carries the skill doc, the aftermath, and the verbatim negative list
    assert "widget-deploy" in prompts[0] and "body" in prompts[0]
    assert "environment-dependent failures" in prompts[0]
    # the fire episode itself is NOT in the post-fire window
    assert "[seq 2]" not in prompts[0] and "[seq 3]" in prompts[0]


def test_fight_emits_candidate_clean_does_not(monkeypatch):
    for outcome, want in (("fight", 1), ("clean", 0)):
        calls = []
        _wire(
            monkeypatch,
            fires=[_fire()],
            episodes=_SESSION,
            verdict_json=json.dumps(dict(_DEVIATION_VERDICT, outcome=outcome)),
            calls=calls,
        )
        stats = PF.run(None, since=None)
        assert stats[outcome] == 1 and stats["candidates"] == want and len(calls) == want


def test_session_tail_fire_skipped_not_judged(monkeypatch):
    calls = []
    episodes = {"s1": [_ep(1, human="deploy it"), _ep(2, content=_FIRE, minutes=1)]}
    prompts, _ = _wire(
        monkeypatch,
        fires=[_fire()],
        episodes=episodes,
        verdict_json=json.dumps(_DEVIATION_VERDICT),
        calls=calls,
    )
    stats = PF.run(None, since=None)
    assert stats["unassessable"] == 1 and stats["last_episode"] == 1
    assert not prompts and not calls  # never judged, never merged


def test_unlocated_fire_is_unassessable(monkeypatch):
    calls = []
    episodes = {"s1": [_ep(1, content="[tool:Bash] echo no fires here")]}
    prompts, _ = _wire(
        monkeypatch,
        fires=[_fire()],
        episodes=episodes,
        verdict_json="{}",
        calls=calls,
    )
    stats = PF.run(None, since=None)
    assert stats["unassessable"] == 1 and stats["unlocated"] == 1
    assert not prompts and not calls


def test_judge_garbage_counts_failure_and_unassessable(monkeypatch):
    calls = []
    _wire(
        monkeypatch,
        fires=[_fire()],
        episodes=_SESSION,
        verdict_json="no json in sight",
        calls=calls,
    )
    stats = PF.run(None, since=None)
    assert stats["judge_failures"] == 1 and stats["unassessable"] == 1 and not calls


def test_unknown_outcome_falls_back_to_unassessable(monkeypatch):
    calls = []
    _wire(
        monkeypatch,
        fires=[_fire()],
        episodes=_SESSION,
        verdict_json=json.dumps(dict(_DEVIATION_VERDICT, outcome="glorious")),
        calls=calls,
    )
    stats = PF.run(None, since=None)
    assert stats["unassessable"] == 1 and not calls


def test_invalid_direction_defaults_to_fix(monkeypatch):
    calls = []
    _wire(
        monkeypatch,
        fires=[_fire()],
        episodes=_SESSION,
        verdict_json=json.dumps(dict(_DEVIATION_VERDICT, direction="sideways")),
        calls=calls,
    )
    PF.run(None, since=None)
    assert calls[0]["direction"] == "fix"


def test_window_stops_before_next_fire(monkeypatch):
    calls = []
    eps = [
        _ep(1, human="deploy the widget please"),
        _ep(2, content=_FIRE, minutes=1),
        _ep(3, human="that skipped the restart step", minutes=2),
        _ep(4, content="[assistant] restarting now", minutes=3),
        _ep(5, content="[tool:Skill] {'skill': 'other-skill'}", minutes=4),
        _ep(6, human="now the other thing is wrong too", minutes=5),
    ]
    prompts, _ = _wire(
        monkeypatch,
        fires=[_fire()],
        episodes={"s1": eps},
        verdict_json=json.dumps(_DEVIATION_VERDICT),
        calls=calls,
    )
    PF.run(None, since=None)
    # aftermath ends before the NEXT fire: seq 5 (the fire) and seq 6 (its aftermath) excluded
    assert "[seq 3]" in prompts[0] and "[seq 4]" in prompts[0]
    assert "[seq 5]" not in prompts[0] and "[seq 6]" not in prompts[0]
    # attribution guard ships in the prompt
    assert "unrelated later work" in prompts[0]


def test_immediate_refire_is_unassessable(monkeypatch):
    calls = []
    eps = [
        _ep(1, content=_FIRE),
        _ep(2, content="[tool:Skill] {'skill': 'other-skill'}", minutes=1),
        _ep(3, human="more work", minutes=2),
    ]
    prompts, _ = _wire(
        monkeypatch,
        fires=[_fire()],
        episodes={"s1": eps},
        verdict_json=json.dumps(_DEVIATION_VERDICT),
        calls=calls,
    )
    stats = PF.run(None, since=None)
    assert stats["unassessable"] == 1 and stats["last_episode"] == 0
    assert not prompts and not calls


def test_window_caps_at_ten_episodes(monkeypatch):
    calls = []
    eps = [_ep(1, content=_FIRE)] + [_ep(i, human=f"turn {i}", minutes=i) for i in range(2, 20)]
    prompts, _ = _wire(
        monkeypatch,
        fires=[_fire()],
        episodes={"s1": eps},
        verdict_json=json.dumps(dict(_DEVIATION_VERDICT, outcome="clean")),
        calls=calls,
    )
    PF.run(None, since=None)
    assert "[seq 11]" in prompts[0] and "[seq 12]" not in prompts[0]


def test_run_signature_is_keyword_only():
    with pytest.raises(TypeError):
        PF.run(None, None)  # since must be keyword-only
