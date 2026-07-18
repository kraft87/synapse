"""Pure-logic tests for the struggle-arc detector (dream/skills/struggle_arc).

No DB, no LLM: the SQL fetchers and the judge call are monkeypatched, merge_candidate is
stubbed to the pinned v2 contract. Guards the Stage A prescreen (marker scoring, the
soft-marker continuation filter, machinery-noise exclusion, arc grouping + the nightly
cap) and the Stage B plumbing (judge-output parsing, update-first retune routing,
grounded-vs-judge evidence class, merge_candidate call shapes). All fixtures synthetic.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime

import pytest

import dream.skills.struggle_arc as SA

_TS = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ------------------------------------------------------------ Stage A: marker prescreen
def test_hard_marker_flags_any_position():
    assert SA.score_turn("that didn't work, the service is still down", 1) >= SA.FLAG_SCORE
    assert SA.score_turn("same error again after the restart", 4) >= SA.FLAG_SCORE
    assert SA.score_turn("you gotta verify your work", 9) >= SA.FLAG_SCORE


def test_hard_marker_matches_curly_apostrophe():
    assert SA.score_turn("that didn\u2019t work", 2) >= SA.FLAG_SCORE


def test_soft_markers_need_continuation():
    # soft-only on the session's FIRST turn: a fresh request, not a correction
    assert SA.score_turn("something is wrong with the widget importer, please look", 1) == 0
    # same phrasing mid-session counts (one soft marker = 1, below the flag bar alone)
    assert SA.score_turn("nope, still not working", 5) >= SA.FLAG_SCORE  # two soft markers
    assert 0 < SA.score_turn("that looks wrong", 5) < SA.FLAG_SCORE


def test_soft_markers_skip_long_turns():
    long_turn = "please refactor the widget parser " + "and also the exporter " * 30 + "wrong"
    assert SA.score_turn(long_turn, 5) == 0


def test_clean_turn_scores_zero():
    assert SA.score_turn("looks good, ship it", 7) == 0
    assert SA.score_turn("", 3) == 0


def test_noise_excluded():
    assert SA.is_noise("This session is being continued from a previous conversation.", "")
    assert SA.is_noise("Task notification: background agent finished", "")
    assert SA.is_noise("<system-reminder>stale</system-reminder>", "")
    assert SA.is_noise("", "<summary>quoted 'same error' text</summary>")
    assert not SA.is_noise("that didn't work", "[user] that didn't work")


# ------------------------------------------------------------------- arc grouping
def _flag(sid, seq, score=2):
    return {"session_id": sid, "sequence": seq, "score": score, "date": "2026-07-18"}


def test_group_arcs_by_proximity():
    arcs = SA.group_arcs([_flag("s1", 3), _flag("s1", 5), _flag("s1", 8)])
    assert len(arcs) == 1
    assert (arcs[0]["seq_min"], arcs[0]["seq_max"]) == (3, 8)
    assert arcs[0]["score"] == 6  # per-episode scores accumulate


def test_group_arcs_splits_on_gap_and_session():
    arcs = SA.group_arcs([_flag("s1", 3), _flag("s1", 30), _flag("s2", 4)])
    assert len(arcs) == 3


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
                "signature": signature,
                "tools": tools,
                "summary": summary,
                "trigger_phrasings": trigger_phrasings,
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


def _episodes(sid="s1", turns=None):
    turns = turns or ["please fix the widget importer", "that didn't work", "same error again"]
    return [
        {
            "session_id": sid,
            "sequence": i + 1,
            "created_at": _TS,
            "human_turn": t,
            "content_head": f"[user] {t}",
        }
        for i, t in enumerate(turns)
    ]


def _window(sid="s1", turns=None):
    # windows carry tool evidence by default so they pass the topic gate
    return [
        {
            "sequence": e["sequence"],
            "human_turn": e["human_turn"],
            "content": e["content_head"] + "\n[tool:Bash] widgetctl status",
        }
        for e in _episodes(sid, turns)
    ]


def _wire(monkeypatch, *, episodes, window, catalog, verdict_json, calls):
    monkeypatch.setattr(SA, "_fetch_new_episodes", lambda conn, since: episodes)
    monkeypatch.setattr(SA, "_fetch_window", lambda conn, sid, lo, hi: window)
    monkeypatch.setattr(SA, "_load_catalog", lambda conn: catalog)
    prompts = []

    def judge(prompt, model=None):
        prompts.append(prompt)
        return verdict_json

    monkeypatch.setattr(SA, "_judge_call", judge)
    monkeypatch.setattr(SA.L, "merge_candidate", _stub_merge(calls))
    return prompts


_DERIVE_VERDICT = {
    "struggle": True,
    "skill_worthy": True,
    "why": "same fix guessed three times",
    "kind": "derive",
    "name": "widget-import-triage",
    "direction": None,
    "salience": 4,
    "signature": "widget import log triage",
    "summary": "Check the import log before changing config.",
    "quote": "that didn't work",
    "trigger_phrasings": ["widget import fails", "importer broken"],
    "proposed_patch": None,
}


def test_run_derive_call_shape(monkeypatch):
    calls = []
    prompts = _wire(
        monkeypatch,
        episodes=_episodes(),
        window=_window(),
        catalog=[("unrelated-skill", "does something else entirely")],
        verdict_json=json.dumps(_DERIVE_VERDICT),
        calls=calls,
    )
    stats = SA.run(None, since=None)
    assert stats["flagged"] == 2 and stats["arcs"] == 1
    assert stats["derives"] == 1 and stats["retunes"] == 0
    assert len(calls) == 1
    c = calls[0]
    assert c["kind"] == "derive" and c["name"] == "widget-import-triage"
    assert c["salience"] == 4 and c["source_detector"] == "struggle_arc"
    assert c["signature"] == "widget import log triage"  # judge topic key passed through
    assert c["tools"] == ["Bash"]  # observed in the window
    (ev,) = c["evidence"]
    assert ev["class"] == "judge" and ev["signal"] == "struggle_arc"
    assert ev["session_id"] == "s1" and ev["quote"] == "that didn't work"
    assert _DATE_RE.match(ev["scan_night"]) and _DATE_RE.match(ev["date"])
    # design invariant: the negative capture list ships verbatim in every judge prompt
    assert "environment-dependent failures" in prompts[0]
    assert "config lane" in prompts[0]
    assert "unrelated-skill" in prompts[0]  # update-first: catalog is in the prompt


def test_run_retune_call_shape(monkeypatch):
    calls = []
    verdict = dict(
        _DERIVE_VERDICT,
        kind="retune",
        name="widget-deploy",
        direction="extend",
        proposed_patch="- add a log-check step\n- verify the service restarted",
    )
    _wire(
        monkeypatch,
        episodes=_episodes(),
        window=_window(),
        catalog=[("widget-deploy", "deploy the widget service")],
        verdict_json=json.dumps(verdict),
        calls=calls,
    )
    stats = SA.run(None, since=None)
    assert stats["retunes"] == 1 and stats["derives"] == 0
    c = calls[0]
    assert c["kind"] == "retune" and c["target_skills"] == ["widget-deploy"]
    assert c["direction"] == "extend" and c["do_embed"] is False
    assert c["proposed_patch"].startswith("- add a log-check step")


def test_retune_of_unknown_skill_downgrades_to_derive(monkeypatch):
    calls = []
    verdict = dict(_DERIVE_VERDICT, kind="retune", name="No Such Skill!")
    _wire(
        monkeypatch,
        episodes=_episodes(),
        window=_window(),
        catalog=[("widget-deploy", "deploy the widget service")],
        verdict_json=json.dumps(verdict),
        calls=calls,
    )
    stats = SA.run(None, since=None)
    assert stats["derives"] == 1 and stats["retunes"] == 0
    assert calls[0]["kind"] == "derive"
    assert calls[0]["name"] == "nosuchskill"  # kebab-cleaned


def test_explicit_ask_in_window_grounds_evidence(monkeypatch):
    calls = []
    window = _window(
        turns=["that didn't work", "same error", "turn this into a skill so it stops happening"]
    )
    _wire(
        monkeypatch,
        episodes=_episodes(),
        window=window,
        catalog=[],
        verdict_json=json.dumps(_DERIVE_VERDICT),
        calls=calls,
    )
    SA.run(None, since=None)
    assert calls[0]["evidence"][0]["class"] == "grounded"


def test_derive_never_emitted_without_signature(monkeypatch):
    # SEV1 guard: signatureless derives wildcard-merge in the ledger resolver
    calls = []
    verdict = {k: v for k, v in _DERIVE_VERDICT.items() if k != "signature"}
    _wire(
        monkeypatch,
        episodes=_episodes(),
        window=_window(),
        catalog=[],
        verdict_json=json.dumps(verdict),
        calls=calls,
    )
    SA.run(None, since=None)
    assert calls[0]["signature"]  # synthesized fallback, never empty
    assert "widget" in calls[0]["signature"]


def test_arc_without_tool_evidence_skipped(monkeypatch):
    # topic gate: correction phrasing with zero tool work in the window = conversation
    calls = []
    window = [
        {
            "sequence": 2,
            "human_turn": "that's not the case",
            "content": "[user] that's not the case",
        }
    ]
    prompts = _wire(
        monkeypatch,
        episodes=_episodes(),
        window=window,
        catalog=[],
        verdict_json=json.dumps(_DERIVE_VERDICT),
        calls=calls,
    )
    stats = SA.run(None, since=None)
    assert stats["skipped_no_tools"] == 1
    assert not prompts and not calls  # never judged, never merged


def test_tool_evidence_accepts_file_paths():
    assert SA.has_tool_evidence([{"sequence": 1, "content": "[tool:Bash] ls"}])
    assert SA.has_tool_evidence([{"sequence": 1, "content": "edit ~/services/widget/app.py next"}])
    assert not SA.has_tool_evidence([{"sequence": 1, "content": "purely a chat about plans"}])


def test_not_skill_worthy_emits_nothing(monkeypatch):
    calls = []
    verdict = dict(_DERIVE_VERDICT, skill_worthy=False)
    _wire(
        monkeypatch,
        episodes=_episodes(),
        window=_window(),
        catalog=[],
        verdict_json=json.dumps(verdict),
        calls=calls,
    )
    stats = SA.run(None, since=None)
    assert stats["skipped"] == 1 and not calls


def test_judge_garbage_counts_failure(monkeypatch):
    calls = []
    _wire(
        monkeypatch,
        episodes=_episodes(),
        window=_window(),
        catalog=[],
        verdict_json="I could not decide, sorry — no JSON here",
        calls=calls,
    )
    stats = SA.run(None, since=None)
    assert stats["judge_failures"] == 1 and not calls


def test_limit_caps_judged_arcs(monkeypatch):
    calls = []
    episodes = _episodes("s1") + _episodes("s2") + _episodes("s3")
    prompts = _wire(
        monkeypatch,
        episodes=episodes,
        window=_window(),
        catalog=[],
        verdict_json=json.dumps(_DERIVE_VERDICT),
        calls=calls,
    )
    stats = SA.run(None, since=None, limit=1)
    assert stats["arcs"] == 3
    assert len(prompts) == 1  # only the top arc reached the judge


def test_noise_episodes_never_flag(monkeypatch):
    calls = []
    episodes = [
        {
            "session_id": "s1",
            "sequence": 2,
            "created_at": _TS,
            "human_turn": "This session is being continued: earlier we hit the same error twice",
            "content_head": "",
        }
    ]
    _wire(
        monkeypatch,
        episodes=episodes,
        window=[],
        catalog=[],
        verdict_json="{}",
        calls=calls,
    )
    stats = SA.run(None, since=None)
    assert stats["flagged"] == 0 and stats["arcs"] == 0 and not calls


def test_salience_clamp():
    assert SA._clamp_salience(7) == 5
    assert SA._clamp_salience(0) == 1
    assert SA._clamp_salience("3") == 3
    assert SA._clamp_salience(None) is None
    assert SA._clamp_salience("high") is None


def test_run_signature_is_keyword_only():
    with pytest.raises(TypeError):
        SA.run(None, None)  # since must be keyword-only
