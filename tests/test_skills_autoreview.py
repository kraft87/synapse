"""The skills autopilot's pure parts and its LLM seam — no DB, no network.

Covers the stripped `claude -p` helper both the calibration and the live reviewer use (exact
flags and env, so the gate measures what runs), the reviewer/verifier parsing and the
decision table, the evidence hash that keys the memo, and the tiering. All text is synthetic.
"""

from __future__ import annotations

import subprocess

import pytest

import dream.skills.autoreview as A
import dream.skills.claude_cli as CLI

NEW = (
    "---\nname: ap-demo\ndescription: Back up the demo database.\n---\n# ap-demo\n\n"
    "1. Dump the database.\n"
)
WIDE = NEW.replace("Back up the demo database.", "Back up the demo database or restore it.")
EDIT = NEW.replace("1. Dump the database.", "1. Dump the database.\n2. Verify the dump.")


def test_the_reviewer_call_is_stripped(monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"], seen["env"], seen["cwd"] = cmd, kw["env"], kw["cwd"]
        return subprocess.CompletedProcess(cmd, 0, stdout='{"verdict": "accept"}', stderr="")

    monkeypatch.setattr(CLI.subprocess, "run", fake_run)
    assert CLI.run("PROMPT", "some-model") == '{"verdict": "accept"}'
    cmd = seen["cmd"]
    assert cmd[:4] == ["claude", "-p", "PROMPT", "--model"] and cmd[4] == "some-model"
    for flag in ("--no-session-persistence", "--strict-mcp-config"):
        assert flag in cmd
    assert cmd[cmd.index("--setting-sources") + 1] == ""  # no settings, hooks or plugins
    assert cmd[cmd.index("--tools") + 1] == ""  # no tool definitions
    env = seen["env"]
    assert env["ENABLE_CLAUDEAI_MCP_SERVERS"] == "false"
    assert env["DISABLE_NON_ESSENTIAL_MODEL_CALLS"] == "1"
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1"  # a workstation's CLAUDE.md stays out
    assert seen["cwd"] and "synapse-review" in seen["cwd"]  # an empty scratch dir


def test_a_missing_cli_is_an_empty_answer_not_a_crash(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("claude")

    monkeypatch.setattr(CLI.subprocess, "run", boom)
    assert CLI.run("p", "m") == ""


def test_the_stage_reviewer_uses_the_helper_and_the_configured_model(monkeypatch):
    calls = []
    monkeypatch.setattr(CLI, "run", lambda prompt, model, **k: calls.append((prompt, model)) or "x")
    monkeypatch.delenv("SKILLS_AUTOREVIEW_MODEL", raising=False)
    A._reviewer_call("p1")
    monkeypatch.setenv("SKILLS_AUTOREVIEW_MODEL", "a-lighter-model")
    A._reviewer_call("p2")
    assert calls == [("p1", A.config.JUDGE_MODEL), ("p2", "a-lighter-model")]


def test_mode_defaults_and_rejects_nonsense(monkeypatch):
    monkeypatch.delenv("SKILLS_AUTOREVIEW", raising=False)
    assert A.current_mode() == A.DEFAULT_MODE
    for m in A.MODES:
        monkeypatch.setenv("SKILLS_AUTOREVIEW", m.upper())
        assert A.current_mode() == m
    monkeypatch.setenv("SKILLS_AUTOREVIEW", "yolo")
    assert A.current_mode() == "shadow"


def test_decision_table():
    d = A.decide
    assert d("new_skill", "accept", "pass") == "apply"
    assert d("description", "accept", "pass") == "apply"
    assert d("body_edit", "accept", "pass") == "ask"  # a body edit is never auto-applied
    assert d("pinned", "accept", "pass") == "ask"
    assert d("consolidate", "accept", "pass") == "ask"
    assert d("new_skill", "accept", "split") == "ask"  # one order flagged it
    assert d("new_skill", "accept", "fail") == "ask"  # reviewer-verifier disagreement
    assert d("new_skill", "accept", "error") == "ask"
    assert d("description", "reject", "fail") == "reject"
    assert d("body_edit", "reject", "fail") == "reject"
    assert d("description", "reject", "pass") == "ask"
    assert d("description", "reject", "split") == "ask"
    assert d("description", None, "pass") == "ask"


def _clean(**over):
    v = {"traces_to_evidence": True, "invents": False, "steals_from": "", "negative_list": False}
    return A.parse_verifier(__import__("json").dumps({**v, **over, "note": "ok"}), ["a", "b"], "a")


def test_verifier_must_be_clean_in_both_orders():
    clean, flagged = _clean(), _clean(invents=True)
    assert A.combine_verifier(clean, clean) == "pass"
    assert A.combine_verifier(flagged, flagged) == "fail"
    assert A.combine_verifier(clean, flagged) == "split"
    assert A.combine_verifier(clean, None) == "error"
    # stealing from the skill itself, or from a name not installed, is not a flag
    assert _clean(steals_from="a")["clean"] and _clean(steals_from="nope")["clean"]
    assert not _clean(steals_from="b")["clean"]
    assert A.parse_verifier("not json", ["a"], None) is None


def test_reviewer_answer_needs_a_verdict_and_a_reason():
    assert A.parse_reviewer('{"verdict": "Accept", "reason": "fine"}') == {
        "verdict": "accept",
        "reason": "fine",
    }
    assert A.parse_reviewer('{"verdict": "accept"}') is None
    assert A.parse_reviewer('{"verdict": "maybe", "reason": "x"}') is None
    assert A.parse_reviewer("") is None


def test_evidence_hash_ignores_a_rescan_but_sees_new_evidence():
    e1 = {"session_id": "s1", "class": "judge", "signal": "gap", "scan_night": "2026-11-01"}
    rescan = {**e1, "scan_night": "2026-11-02", "quote": "the same sighting again"}
    person = {"session_id": None, "class": "grounded", "signal": "revive"}
    new = {"session_id": "s2", "class": "judge", "signal": "gap"}
    base = A.evidence_hash([e1])
    assert A.evidence_hash([e1, rescan, person]) == base
    assert A.evidence_hash([e1, new]) != base
    assert A.evidence_hash([e1, {**e1, "class": "grounded"}]) != base


def test_a_person_touching_a_row_takes_it_out_of_the_lanes_hands():
    assert A.human_touched([{"session_id": None, "class": "grounded", "signal": "revive"}])
    assert not A.human_touched([{"session_id": "s1", "class": "grounded", "signal": "dismiss"}])
    assert not A.human_touched([{"session_id": "s1", "class": "judge", "signal": "gap"}])


def _derive(evidence, body=NEW):
    return {"kind": "derive", "name": "ap-demo", "evidence": evidence, "proposal_body": body}


def _day(sid, day):
    return {"session_id": sid, "class": "judge", "signal": "gap", "date": day}


def test_tiers():
    twice = [_day("s1", "2026-11-01"), _day("s2", "2026-11-03")]
    assert A.tier_for(_derive(twice), {}, False)[0] == "new_skill"
    same_day = [_day("s1", "2026-11-01"), _day("s2", "2026-11-01")]
    assert A.tier_for(_derive(same_day), {}, False)[0] == "derive_no_recurrence"
    assert A.tier_for(_derive(twice, "no frontmatter"), {}, False)[0] == "derive_unusable"
    assert A.tier_for(_derive(twice, NEW + "x" * 9000), {}, False)[0] == "derive_too_big"
    retune = {"kind": "retune", "name": "ap-demo", "proposal_body": WIDE}
    assert A.tier_for(retune, {"registry_body": NEW}, False)[0] == "description"
    assert A.tier_for(retune, {"registry_body": NEW}, True)[0] == "pinned"
    assert A.tier_for({**retune, "proposal_body": EDIT}, {"registry_body": NEW}, False)[0] == (
        "body_edit"
    )
    assert A.tier_for({"kind": "consolidate", "name": "x"}, {}, False)[0] == "consolidate"
    long_desc = NEW.replace("Back up the demo database.", "x" * 1100)
    assert A.tier_for({**retune, "proposal_body": long_desc}, {"registry_body": NEW}, False)[0] == (
        "description_unusable"
    )


def test_judge_never_raises_and_skips_the_verifier_when_the_reviewer_failed(monkeypatch):
    verifier_calls = []
    monkeypatch.setattr(A, "verifier_model", lambda: "other/model")
    monkeypatch.setattr(A, "_verifier_call", lambda p: verifier_calls.append(p) or "{}")

    def boom(_p):
        raise RuntimeError("cli down")

    monkeypatch.setattr(A, "_reviewer_call", boom)
    row = {**_derive([_day("s1", "2026-11-01")]), "id": 1, "summary": "s"}
    j = A.judge(row, None, [("other", "Something else.")])
    assert j["reviewer"] is None and j["verifier"]["result"] == "error"
    assert verifier_calls == []  # nothing to verify, nothing spent


def test_no_independent_verifier_means_no_auto_apply(monkeypatch):
    monkeypatch.setattr(A, "verifier_model", lambda: None)
    monkeypatch.setattr(A, "_reviewer_call", lambda p: '{"verdict": "accept", "reason": "ok"}')
    row = {**_derive([_day("s1", "2026-11-01")]), "id": 1, "summary": "s"}
    j = A.judge(row, None, [])
    assert j["verifier"]["result"] == "error"
    assert A.decide("new_skill", "accept", j["verifier"]["result"]) == "ask"


@pytest.mark.parametrize("provider", ["", "claude", "anthropic"])
def test_verifier_model_requires_a_non_claude_openai_compatible_stage(monkeypatch, provider):
    monkeypatch.setenv("SYNAPSE_LLM_PROVIDER", provider)
    monkeypatch.setenv("SYNAPSE_SKILLS_VERIFY_MODEL", "z-ai/some-model")
    assert A.verifier_model() is None
    monkeypatch.setenv("SYNAPSE_LLM_PROVIDER", "openai")
    assert A.verifier_model() == "z-ai/some-model"
    monkeypatch.setenv("SYNAPSE_SKILLS_VERIFY_MODEL", "anthropic/claude-sonnet")
    assert A.verifier_model() is None
