"""The dream lanes' LLM calls (dream/lane.py judge / draft / run_parallel) and the skills
lane's one-analysis-per-session pass. No DB, no real LLM.

Covers:
  - judge and draft go through ingestion.llm_client on their own stage models
    (SYNAPSE_SKILLS_JUDGE_MODEL / SYNAPSE_SKILLS_DRAFT_MODEL, Opus by default)
  - a usage limit or unreachable model RAISES out of the lane's call chain, instead of
    reading as "no gaps" (the old `claude -p` subprocess returned stdout whatever happened)
  - run_parallel: order, real concurrency, first failure stops the rest
  - one analysis call per session feeds both derive and retune
  - the dream prompts are on the harness-prompt backstop list
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import pytest

import dream.config.nightly as CN
import dream.lane as lane
import dream.skills.nightly as N
import dream.skills.post_fire as PF
import dream.skills.procedure_miner as PM
import dream.skills.skill_derive as SD
import dream.skills.skill_draft as DR
import dream.skills.skill_measure as SM
import dream.skills.struggle_arc as SA
import ingestion.llm_client as llm
from ingestion.contamination import is_harness_call


class _FakeClient:
    """Duck-types ClaudeAgentClient: records messages.create calls."""

    def __init__(self, reply="{}", error=None):
        self.calls: list[dict] = []
        self.reply, self.error = reply, error
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        if self.error:
            raise self.error
        return SimpleNamespace(content=[SimpleNamespace(text=self.reply)])


@pytest.fixture()
def fake(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr(lane, "_llm", lambda: client)
    for var in ("SYNAPSE_SKILLS_JUDGE_MODEL", "SYNAPSE_SKILLS_DRAFT_MODEL", "SYNAPSE_LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    return client


# ------------------------------------------------------------------ judge / draft
def test_judge_and_draft_use_their_own_stage_models(fake, monkeypatch):
    fake.reply = "the reply"
    assert lane.judge("judge this") == "the reply"
    assert lane.draft("draft this") == "the reply"
    judge, draft = fake.calls
    assert judge["model"] == draft["model"] == lane.OPUS
    assert judge["messages"] == [{"role": "user", "content": "judge this"}]
    assert draft["messages"] == [{"role": "user", "content": "draft this"}]

    monkeypatch.setenv("SYNAPSE_SKILLS_JUDGE_MODEL", "claude-sonnet-4-6")
    monkeypatch.setenv("SYNAPSE_SKILLS_DRAFT_MODEL", "claude-haiku-4-5")
    lane.judge("again")
    lane.draft("again")
    assert [c["model"] for c in fake.calls[2:]] == ["claude-sonnet-4-6", "claude-haiku-4-5"]


def test_a_bad_model_id_fails_before_any_call(fake, monkeypatch):
    monkeypatch.setenv("SYNAPSE_SKILLS_JUDGE_MODEL", "anthropic/claude-opus-4.8")
    with pytest.raises(ValueError, match="OpenRouter-style"):
        lane.judge("x")
    assert fake.calls == []


@pytest.mark.parametrize(
    "error",
    [
        llm.UsageLimitError("Claude usage limit reached"),
        llm.LLMUnavailableError("not logged in"),
        llm.TransientLLMHTTPError("503", status_code=503),
    ],
)
def test_an_unreachable_model_raises_through_every_judge_path(fake, error):
    """Each lane judge reads its reply through lane.judge/draft; none turns the failure
    into an empty verdict."""
    fake.error = error
    session = {"session": "s", "fired": [], "tools": {}, "bash_heads": [], "user_msgs": ["x"]}
    calls = [
        lambda: SM.analyze_session(session, "(none)"),
        lambda: SD.cluster_gaps([{"_session": "s", "procedure": "p"}]),
        lambda: SD.draft_skill({"what": "w"}),
        lambda: DR.draft_widen(
            "n",
            "---\nname: n\ndescription: d\n---\nbody\n",
            [{"signal": "under_trigger", "missing_phrasing": "m"}],
        ),
        lambda: SA.judge_arc([], "(none)"),
        lambda: PF.judge_fire("s", "d", "b", []),
        lambda: PM._confirm({"steps": ["a"], "sessions": ["s"]}, {}, {}, {}, "(none)", 35),
        lambda: CN.scan_corrections(session),
    ]
    for call in calls:
        with pytest.raises(type(error)):
            call()


def test_config_scan_tells_an_unreadable_reply_from_no_corrections(fake):
    session = {"session": "s1", "user_msgs": ["no em-dashes please"]}
    fake.reply = '{"corrections": []}'
    assert CN.scan_corrections(session) == []  # the judge looked and found none
    fake.reply = "I could not find a JSON answer, sorry"
    assert CN.scan_corrections(session) is None  # told apart: the lane counts these
    fake.reply = 'Here: {"corrections": [{"rule": " Avoid em-dashes ", "quote": "no em-dashes"}, {"rule": ""}]}'
    assert CN.scan_corrections(session) == [
        {"rule": "Avoid em-dashes", "session_id": "s1", "quote": "no em-dashes"}
    ]


def test_usage_limit_text_from_the_cli_raises_end_to_end(monkeypatch):
    """Through the real ClaudeAgentClient: the usage-limit text agent_call detects reaches the
    lane as UsageLimitError (the subprocess path returned it as stdout and parsed it as
    "no gaps")."""
    seen: dict = {}

    async def agent_call(prompt, **kw):
        seen.update(kw, prompt=prompt)
        raise llm.UsageLimitError("Claude usage limit reached")

    monkeypatch.setattr(llm, "agent_call", agent_call)
    monkeypatch.setattr(lane, "_llm", lambda: llm.create_llm_client())
    monkeypatch.delenv("SYNAPSE_SKILLS_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("SYNAPSE_LLM_MODEL", raising=False)
    monkeypatch.delenv("SYNAPSE_LLM_PROVIDER", raising=False)
    with pytest.raises(llm.UsageLimitError):
        lane.judge("p")
    assert seen["model"] == lane.OPUS and seen["prompt"] == "p"


# ------------------------------------------------------------------ run_parallel
def test_run_parallel_keeps_order_and_runs_concurrently():
    barrier = threading.Barrier(3, timeout=5)  # only passes if 3 calls are in flight at once

    def fn(x):
        barrier.wait()
        return x * 10

    assert lane.run_parallel(fn, [1, 2, 3], workers=3) == [10, 20, 30]


def test_run_parallel_stops_at_the_first_failure():
    started: list[int] = []

    def fn(x):
        started.append(x)
        if x == 2:
            raise llm.UsageLimitError("limit")
        if x == 3:
            time.sleep(0.3)  # in flight while the failure is handled
        return x

    with pytest.raises(llm.UsageLimitError):
        lane.run_parallel(fn, [1, 2, 3, 4, 5], workers=1)
    assert started == [1, 2, 3]  # the call already running finishes; 4 and 5 never start


def test_run_parallel_of_nothing():
    assert lane.run_parallel(lambda x: x, []) == []


# ------------------------------------------------------------------ one call per session
_ANALYSIS = {
    "would_have_helped": [
        {
            "skill": "db-query",
            "why": "ran psql",
            "description_covers": False,
            "missing_phrasing": "pgbouncer pool size",
        }
    ],
    "gaps": [
        {
            "procedure": "widget-triage",
            "what": "triage the widget import log",
            "trigger_phrasings": ["widget import fails"],
            "signature": "rg -> Read -> Bash widgetctl",
        }
    ],
}


def _session(sid):
    return {
        "session": sid,
        "first_user": "fix the import",
        "user_turns": 3,
        "user_msgs": ["fix the import", "still broken"],
        "tools": {"Bash": 4},
        "bash_heads": ["widgetctl status"],
        "fired": [],
    }


def test_one_analysis_call_per_session_feeds_derive_and_retune(monkeypatch):
    prompts: list[str] = []

    def judge(prompt):
        prompts.append(prompt)
        if "consolidating candidate" in prompt:  # the cross-session cluster call
            return json.dumps(
                {
                    "clusters": [
                        {
                            "procedure": "widget-triage",
                            "what": "triage the widget import log",
                            "sessions": ["s1", "s2"],
                        }
                    ]
                }
            )
        return json.dumps(_ANALYSIS)

    merged: list[tuple] = []

    def merge(conn, kind, name, evidence, **kw):
        merged.append((kind, name, [e["session_id"] for e in evidence]))
        return {"id": len(merged), "status": "observe", "score": 0.5, "merged": False}

    monkeypatch.setattr(lane, "judge", judge)
    monkeypatch.setattr(N.L, "merge_candidate", merge)
    catalog = [("db-query", "Query the app Postgres with psql.")]

    stats = N.judge_sessions(None, [_session("s1"), _session("s2")], catalog, {}, set())

    # two sessions -> two analysis calls (was four: a scan and a judge each) + one cluster call
    assert len(prompts) == 3
    assert stats["sessions"] == 2 and stats["unparsed"] == 0
    assert stats["derive"]["gaps"] == 2 and stats["derive"]["clusters"] == 1
    assert stats["retune"]["under_trigger"] == 2
    kinds = sorted(k for k, _, _ in merged)
    assert kinds == ["derive", "retune", "retune"]
    assert ("derive", "widget-triage", ["s1", "s2"]) in merged


def test_a_proposed_derive_is_left_to_the_draft_stage_and_keeps_its_bash_heads(monkeypatch):
    cluster = {"clusters": [{"procedure": "widget-triage", "sessions": ["s1", "s2"]}]}
    derived: list[list[dict]] = []

    def merge(conn, kind, name, evidence, **kw):
        derived.append(evidence)
        return {"id": 1, "status": "proposed", "score": 2.0, "merged": False}

    monkeypatch.setattr(lane, "judge", lambda prompt: json.dumps(cluster))
    monkeypatch.setattr(lane, "draft", lambda prompt: pytest.fail("drafted outside the cap"))
    monkeypatch.setattr(N.L, "merge_candidate", merge)
    sessions = [_session("s1"), _session("s2")]

    analyses = [{"gaps": [dict(g) for g in _ANALYSIS["gaps"]]} for _ in sessions]

    stats = N.run_derive(None, sessions, analyses, set())

    assert stats["proposed"] == 1
    # each session's evidence carries the shell commands its draft will show the model
    assert [e["bash"] for e in derived[0]] == [["widgetctl status"], ["widgetctl status"]]


def test_a_session_is_judged_against_the_skills_its_project_is_served(monkeypatch):
    seen: dict[str, str] = {}
    analysis = {
        "would_have_helped": [
            {"skill": s, "why": "w", "description_covers": False, "missing_phrasing": "pgbouncer"}
            for s in ("db-query", "widget-deploy")
        ],
        "gaps": [],
    }

    def judge(prompt):
        session = "a" if "in the widgets repo" in prompt else "b"
        seen[session] = prompt.split("SESSION:")[0]
        return json.dumps(analysis)

    merged: list[tuple[str, str]] = []

    def merge(conn, kind, name, evidence, **kw):
        merged.append((evidence[0]["session_id"], name))
        return {"id": len(merged), "status": "observe", "score": 0.5, "merged": False}

    monkeypatch.setattr(lane, "judge", judge)
    monkeypatch.setattr(N.L, "merge_candidate", merge)
    catalog = [("db-query", "Query Postgres."), ("widget-deploy", "Deploy the widgets.")]
    scopes = {"db-query": "global", "widget-deploy": "project:Widgets"}
    in_widgets = {  # the label's case differs from the scope's
        **_session("session-a"),
        "project": "widgets",
        "user_msgs": ["in the widgets repo"],
    }
    elsewhere = {**_session("session-b"), "project": "Other", "user_msgs": ["somewhere else"]}

    N.judge_sessions(None, [in_widgets, elsewhere], catalog, scopes, set())

    assert "widget-deploy" in seen["a"] and "widget-deploy" not in seen["b"]
    assert "db-query" in seen["a"] and "db-query" in seen["b"]
    # the judge named a project skill for the session that is never served it: no widen
    assert sorted(merged) == [
        ("session-a", "db-query"),
        ("session-a", "widget-deploy"),
        ("session-b", "db-query"),
    ]


def test_an_unparseable_analysis_adds_nothing_and_is_counted(monkeypatch):
    monkeypatch.setattr(lane, "judge", lambda prompt: "Sorry, I cannot help with that.")
    stats = N.judge_sessions(None, [_session("s1")], [], {}, set())
    assert stats["sessions"] == 1 and stats["unparsed"] == 1
    assert stats["derive"]["gaps"] == 0 and stats["retune"]["under_trigger"] == 0


# ------------------------------------------------------------------ harness prompts
@pytest.mark.parametrize(
    "prompt",
    [
        SM._ANALYZE_PROMPT,
        SD._CLUSTER_PROMPT,
        SD._DRAFT_PROMPT,
        DR._WIDEN_PROMPT,
        DR._REVISE_PROMPT,
        SA._JUDGE_PROMPT,
        PF._JUDGE_PROMPT,
        PM._CONFIRM_PROMPT,
        CN._FIND_PROMPT,
    ],
)
def test_every_dream_prompt_is_on_the_harness_backstop(prompt):
    """A lane run by hand on a workstation with the Stop ingest hook must not ingest its own
    judge prompts as memories (ingestion/contamination.py)."""
    assert is_harness_call(f"[user] {prompt}\n\n[assistant] {{}}")
