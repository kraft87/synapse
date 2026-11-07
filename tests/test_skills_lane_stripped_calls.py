"""Every LLM call the dream->skills lane (and the config lane that shares its judge) makes to
the claude CLI is the stripped invocation of dream/skills/claude_cli.py.

A bare `claude -p <prompt> --model <m>` loads ~16k tokens of built-in tool definitions, can pull
in the claude.ai connector schemas, and makes a hidden side call, on every judge call of every
night. Each test here drives a real call site down to subprocess.run and checks the argv and the
environment it would spawn: no tools, no settings, strict MCP, no persisted session, the
side-call and connector switches off. Model, prompt, timeout and the text answer are untouched.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

import dream.config.nightly as CN
import dream.skills.claude_cli as CLI
import dream.skills.post_fire as PF
import dream.skills.procedure_miner as PM
import dream.skills.review_queue as RQ
import dream.skills.skill_derive as SD
import dream.skills.skill_draft as SK
import dream.skills.skill_measure as SM
import dream.skills.struggle_arc as SA

JSON_ANSWER = '{"verdict": "accept", "reason": "fine", "gaps": [], "clusters": []}'
WINDOW = [{"sequence": 1, "content": "step one", "human_turn": "please do it"}]
SESSION = {
    "session": "abcdef123456",
    "fired": [],
    "tools": {"Bash": 3},
    "bash_heads": ["ls"],
    "user_msgs": ["do the thing"],
}


@pytest.fixture
def spawned(monkeypatch):
    """Records every subprocess.run the code makes (nothing is spawned) and answers JSON."""
    calls: list[dict] = []

    def fake_run(cmd, **kw):
        calls.append({"cmd": list(cmd), **kw})
        return subprocess.CompletedProcess(cmd, 0, stdout=JSON_ANSWER, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.delenv("SKILL_MEASURE_MODEL", raising=False)
    monkeypatch.delenv("SKILL_DERIVE_DRAFT_MODEL", raising=False)
    return calls


def _flag(cmd: list[str], name: str) -> str:
    return cmd[cmd.index(name) + 1]


def assert_stripped(call: dict, *, model: str, timeout: int) -> None:
    cmd, env = call["cmd"], call.get("env") or {}
    assert cmd[0] == "claude" and cmd[1] == "-p" and _flag(cmd, "--model") == model
    assert "--no-session-persistence" in cmd
    assert "--strict-mcp-config" in cmd  # no MCP servers, claude.ai connectors included
    assert "--setting-sources" in cmd and _flag(cmd, "--setting-sources") == ""
    assert "--tools" in cmd and _flag(cmd, "--tools") == ""  # no tool definitions
    for k, v in CLI.STRIP_ENV.items():
        assert env.get(k) == v, k
    assert env["ENABLE_CLAUDEAI_MCP_SERVERS"] == "false"
    assert env["DISABLE_NON_ESSENTIAL_MODEL_CALLS"] == "1"
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert call["timeout"] == timeout
    # the answer is still the plain text on stdout, parsed as before
    assert "--output-format" not in cmd and "--max-turns" not in cmd
    assert call["capture_output"] is True and call["text"] is True
    assert call["cwd"] and "synapse-review" in call["cwd"]  # an empty scratch dir


# ---------------------------------------------------------------- the shared judge
def test_run_judge_default_model_is_stripped(spawned):
    assert SM.run_judge("PROMPT") == JSON_ANSWER
    (call,) = spawned
    assert call["cmd"][2] == "PROMPT"
    assert_stripped(call, model=SM.config.JUDGE_MODEL, timeout=240)


def test_run_judge_explicit_model_and_timeout_are_kept(spawned):
    SM.run_judge("PROMPT", "claude-some-model", timeout=77)
    (call,) = spawned
    assert_stripped(call, model="claude-some-model", timeout=77)


def test_the_openrouter_backend_never_reaches_the_cli(spawned, monkeypatch):
    monkeypatch.setattr(SM, "_openrouter_judge", lambda prompt: "or:" + prompt)
    assert SM.run_judge("P", "deepseek") == "or:P"
    monkeypatch.setenv("SKILL_MEASURE_MODEL", "openrouter")
    assert SM.run_judge("P") == "or:P"
    assert spawned == []


def test_a_missing_cli_or_a_timeout_still_raises_so_detectors_count_a_failure(monkeypatch):
    def boom(cmd, **kw):
        raise FileNotFoundError("claude")

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(FileNotFoundError):
        SM.run_judge("p")
    assert PF.judge_fire("sk", "d", "b", WINDOW) is None
    assert SA.judge_arc(WINDOW, "cat") is None
    assert SM.judge_session(SESSION, "cat") is None

    def slow(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])

    monkeypatch.setattr(subprocess, "run", slow)
    with pytest.raises(subprocess.TimeoutExpired):
        SM.run_judge("p")


# ---------------------------------------------------------------- each call site
def _recommend(monkeypatch):
    """review_queue.recommend over one drafted proposal, without a database."""
    row = {
        "id": 7,
        "kind": "derive",
        "name": "demo-skill",
        "direction": None,
        "target_skills": None,
        "summary": "a demo procedure",
        "evidence": [{"session_id": "s1", "class": "judge", "signal": "gap"}],
        "proposal_body": "---\nname: demo-skill\ndescription: demo\n---\n1. Do it.\n",
        "recommendation": None,
        "registry_body": None,
    }

    class Cur:
        def execute(self, *a, **k):
            return self

        def fetchall(self):
            return [row]

    class Conn:
        def cursor(self, **k):
            return Cur()

        def commit(self):
            pass

    monkeypatch.setattr(RQ.L, "queue_ready", lambda conn: True)
    return RQ.recommend(Conn())


def test_recommend_is_stripped(spawned, monkeypatch):
    assert _recommend(monkeypatch)["recommended"] == 1
    (call,) = spawned
    assert "demo procedure" in call["cmd"][2]  # the lane's own recommendation prompt
    assert_stripped(call, model=SM.config.JUDGE_MODEL, timeout=240)


def test_post_fire_judge_is_stripped(spawned):
    assert PF.judge_fire("sk", "desc", "body", WINDOW) is not None
    assert PF.judge_fire("sk", "desc", "body", WINDOW, model="claude-other") is not None
    first, second = spawned
    assert_stripped(first, model=SM.config.JUDGE_MODEL, timeout=240)
    assert_stripped(second, model="claude-other", timeout=240)


def test_struggle_arc_judge_is_stripped(spawned):
    assert SA.judge_arc(WINDOW, "- skill-a: does a thing") is not None
    SA.judge_arc(WINDOW, "cat", model="claude-other")
    first, second = spawned
    assert_stripped(first, model=SM.config.JUDGE_MODEL, timeout=240)
    assert_stripped(second, model="claude-other", timeout=240)


def test_procedure_miner_confirm_is_stripped(spawned):
    cluster = {"steps": ["a", "b"], "sessions": {"s1", "s2"}}
    out = PM._confirm(cluster, {}, {}, {}, "- skill-a: x", 30, "claude-other")
    assert out is not None
    (call,) = spawned
    assert_stripped(call, model="claude-other", timeout=240)


def test_derive_gap_scan_cluster_and_draft_are_stripped(spawned):
    assert SD.scan_session(SESSION, "- skill-a: x") == []
    gaps = [{"_session": "abcdef123456", "procedure": "p", "what": "w"}]
    SD.cluster_gaps(gaps)
    SD.draft_skill({"what": "w", "signature": "s", "n_sessions": 2})
    scan, cluster, draft = spawned
    assert_stripped(scan, model=SM.config.JUDGE_MODEL, timeout=240)
    assert_stripped(cluster, model=SM.config.JUDGE_MODEL, timeout=240)
    assert_stripped(draft, model="claude-opus-4-8", timeout=300)  # the drafting model, 300s


def test_the_draft_model_override_is_kept(spawned, monkeypatch):
    monkeypatch.setenv("SKILL_DERIVE_DRAFT_MODEL", "claude-custom")
    SD._draft_call("PROMPT")
    (call,) = spawned
    assert call["cmd"][2] == "PROMPT"
    assert_stripped(call, model="claude-custom", timeout=300)


def test_the_widen_and_revise_drafts_are_stripped(spawned):
    current = "---\nname: demo\ndescription: old words\n---\nbody\n"
    SK.draft_widen("demo", current, [{"signal": "under_trigger", "missing_phrasing": "new words"}])
    SK.draft_revision(
        {"direction": "fix", "summary": "s", "proposed_patch": "- change it", "name": "demo"},
        current,
    )
    assert len(spawned) == 2
    for call in spawned:
        assert_stripped(call, model="claude-opus-4-8", timeout=300)


def test_retune_under_trigger_judge_is_stripped(spawned):
    assert SM.judge_session(SESSION, "- skill-a: x") is not None
    (call,) = spawned
    assert_stripped(call, model=SM.config.JUDGE_MODEL, timeout=240)


def test_config_lane_correction_finder_is_stripped(spawned):
    assert CN.scan_corrections(SESSION) == []
    (call,) = spawned
    assert_stripped(call, model=SM.config.JUDGE_MODEL, timeout=240)


# ---------------------------------------------------------------- no new unstripped site
def test_no_dream_module_spawns_a_claude_process_but_claude_cli():
    """claude_cli.py is the one place the dream package shells out; a new call site that
    builds its own `claude -p` argv would carry the full tool definitions again."""
    root = Path(CLI.__file__).resolve().parents[1]
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "claude_cli.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            if any(n.split(".")[0] in ("subprocess", "pty", "asyncio.subprocess") for n in names):
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert offenders == []
