"""Under-trigger -> retune/widen gate (dream/skills nightly.run_retune + skill_measure).

No DB, no LLM: the session judge is monkeypatched and merge_candidate is stubbed to record
its calls. Guards the precision fix: a would_have_helped skill whose CURRENT description
already covers the request was a match the model didn't invoke, so it must never become a
widen proposal. Only a stated, real vocabulary gap may widen. All skills/phrasings synthetic.
"""

from __future__ import annotations

import pytest

import dream.skills.nightly as N
import dream.skills.skill_measure as SM

_DB_DESC = (
    "Query the app Postgres with psql. Use whenever running SQL against the app database. "
    'Triggers on: "run this SQL against the app db", "psql", "check the table".'
)
_PIPE_DESC = (
    "Edit and validate the build pipelines in pipelines/build_a.yml and pipelines/build_b.yml. "
    'Triggers on: "build pipeline", "pipeline rev".'
)
_CATALOG = {"db-query": _DB_DESC, "pipeline-edit": _PIPE_DESC}


def _session(sid="sess-1", first_user="can you run this SQL for me", fired=()):
    return {
        "session": sid,
        "first_user": first_user,
        "user_turns": 3,
        "user_msgs": [first_user],
        "tools": {"Bash": 4},
        "bash_heads": ["psql -c 'select 1'"],
        "fired": list(fired),
    }


def _wire(monkeypatch, verdict, calls):
    monkeypatch.setattr(N.SM, "judge_session", lambda s, catalog: verdict)

    def fake_merge(conn, kind, name, evidence, **kw):
        calls.append({"kind": kind, "name": name, "evidence": evidence, **kw})
        return {"id": len(calls), "status": "observe", "score": 0.5, "merged": False}

    monkeypatch.setattr(N.L, "merge_candidate", fake_merge)


def _run(sessions):
    seen: set[int] = set()
    stats = N.run_retune(None, sessions, "(catalog)", seen, _CATALOG)
    return stats, seen


# ----------------------------------------------------------------- run_retune gate


def test_covered_request_creates_no_widen_candidate(monkeypatch):
    calls: list[dict] = []
    verdict = {
        "would_have_helped": [
            {
                "skill": "db-query",
                "why": "the session ran psql queries",
                "description_covers": True,
                "missing_phrasing": "",
            }
        ]
    }
    _wire(monkeypatch, verdict, calls)
    stats, seen = _run([_session()])
    assert calls == [] and seen == set()
    assert stats["under_trigger"] == 0
    assert stats["covered_not_fired"] == 1 and stats["backstop_covered"] == 0
    assert stats["covered_by_skill"] == {"db-query": 1}  # telemetry kept, not merged


@pytest.mark.parametrize("covers", [False, "false"])  # LLMs sometimes stringify booleans
def test_real_gap_creates_widen_with_missing_phrasing(monkeypatch, covers):
    calls: list[dict] = []
    verdict = {
        "would_have_helped": [
            {
                "skill": "pipeline-edit",
                "why": "the session edited a third build pipeline",
                "description_covers": covers,
                "missing_phrasing": "pipelines/build_c.yml",
            }
        ]
    }
    _wire(monkeypatch, verdict, calls)
    stats, seen = _run([_session(first_user="update build_c.yml to add a lint stage")])
    assert stats["under_trigger"] == 1 and stats["covered_not_fired"] == 0
    assert seen == {1}
    (c,) = calls
    assert c["kind"] == "retune" and c["name"] == "pipeline-edit"
    assert c["direction"] == "widen" and c["target_skills"] == ["pipeline-edit"]
    (ev,) = c["evidence"]
    assert ev["signal"] == "under_trigger" and ev["class"] == "judge"
    assert ev["missing_phrasing"] == "pipelines/build_c.yml"  # reviewer sees the gap
    assert ev["phrasing"] == "update build_c.yml to add a lint stage"
    assert "build_c.yml" in c["summary"]


def test_backstop_overrides_judge_when_phrasing_is_in_description(monkeypatch):
    calls: list[dict] = []
    verdict = {
        "would_have_helped": [
            {
                "skill": "db-query",
                "why": "user asked to run SQL",
                "description_covers": False,  # judge self-report is wrong
                "missing_phrasing": "Run this SQL against the app DB",
            }
        ]
    }
    _wire(monkeypatch, verdict, calls)
    stats, _ = _run([_session()])
    assert calls == []
    assert stats["under_trigger"] == 0
    assert stats["covered_not_fired"] == 1 and stats["backstop_covered"] == 1


def test_backstop_also_checks_the_session_phrasing(monkeypatch):
    calls: list[dict] = []
    verdict = {
        "would_have_helped": [
            {
                "skill": "db-query",
                "why": "short psql ask",
                "description_covers": False,
                "missing_phrasing": "query helper",  # not literally in the description
            }
        ]
    }
    _wire(monkeypatch, verdict, calls)
    stats, _ = _run([_session(first_user="check the table")])  # a declared trigger
    assert calls == [] and stats["backstop_covered"] == 1


def test_legacy_judge_output_without_gap_fields_does_not_widen(monkeypatch):
    # pre-gate judge output: {skill, why} only. Safe default: no stated gap -> no widen.
    calls: list[dict] = []
    verdict = {"would_have_helped": [{"skill": "db-query", "why": "ran psql"}]}
    _wire(monkeypatch, verdict, calls)
    stats, _ = _run([_session()])
    assert calls == []
    assert stats["no_gap_stated"] == 1 and stats["under_trigger"] == 0
    assert stats["covered_not_fired"] == 0


@pytest.mark.parametrize(
    "item",
    [
        {"description_covers": False, "missing_phrasing": ""},  # gap claimed, none named
        {"description_covers": False, "missing_phrasing": "   "},
        {"missing_phrasing": "pipelines/build_c.yml"},  # flag missing -> not a stated gap
        {"description_covers": "maybe", "missing_phrasing": "pipelines/build_c.yml"},
    ],
)
def test_unstated_or_malformed_gap_does_not_widen(monkeypatch, item):
    calls: list[dict] = []
    verdict = {"would_have_helped": [{"skill": "pipeline-edit", "why": "x", **item}]}
    _wire(monkeypatch, verdict, calls)
    stats, _ = _run([_session(first_user="update build_c.yml")])
    assert calls == [] and stats["no_gap_stated"] == 1


def test_fired_and_uncatalogued_skills_are_skipped(monkeypatch):
    calls: list[dict] = []
    gap = {"description_covers": False, "missing_phrasing": "pipelines/build_c.yml"}
    verdict = {
        "would_have_helped": [
            {"skill": "pipeline-edit", "why": "x", **gap},  # fired this session
            {"skill": "no-such-skill", "why": "x", **gap},  # not in the catalog
        ]
    }
    _wire(monkeypatch, verdict, calls)
    stats, _ = _run([_session(fired=["pipeline-edit"])])
    assert calls == []
    assert stats == {
        "under_trigger": 0,
        "covered_not_fired": 0,
        "backstop_covered": 0,
        "no_gap_stated": 0,
    }


# ------------------------------------------------------ deterministic coverage backstop


@pytest.mark.parametrize(
    ("phrasing", "description", "covered"),
    [
        # literal / normalized substring of the description (case, punctuation, filler)
        ("psql", _DB_DESC, True),
        ("run SQL against the app DB", _DB_DESC, True),
        ("Check the TABLE!", _DB_DESC, True),
        # a `Triggers on:` entry inside the user's wording
        ("look into", 'Research a topic. Triggers on: "look into", "check reddit".', True),
        ("check reddit", 'Research a topic. Triggers on: "look into", "check reddit".', True),
        # identifier parts: the description's longer name covers the shorter knob name
        ("RECALL_FLOOR", "Tune the SYNAPSE_RECALL_FLOOR knob.", True),
        # real gaps stay gaps: a third file, an unnamed knob, an extra concept
        ("pipelines/build_c.yml", _PIPE_DESC, False),
        ("NOTES_FLOOR", "Tune the SYNAPSE_RECALL_FLOOR knob.", False),
        ("psql connection pooling", _DB_DESC, False),
        # degenerate inputs never count as covered
        ("", _DB_DESC, False),
        ("the this", "the this", False),
        ("psql", "", False),
    ],
)
def test_description_covers_phrasing(phrasing, description, covered):
    assert SM.description_covers_phrasing(phrasing, description) is covered


def test_under_trigger_verdict_classes():
    v = SM.under_trigger_verdict
    assert v({"description_covers": True}, _DB_DESC) == "covered"
    assert v({"description_covers": "TRUE", "missing_phrasing": "x"}, _DB_DESC) == "covered"
    assert v({"skill": "db-query", "why": "legacy"}, _DB_DESC) == "no_gap"
    assert v({"description_covers": False, "missing_phrasing": "psql"}, _DB_DESC) == "backstop"
    gap = {"description_covers": False, "missing_phrasing": "pgbouncer pool size"}
    assert v(gap, _DB_DESC) == "gap"
    assert v(gap, _DB_DESC, phrasing="psql") == "backstop"  # session phrasing is covered


# --------------------------------------------------------------------- judge prompt


def test_judge_prompt_carries_descriptions_and_gap_schema(monkeypatch):
    prompts: list[str] = []

    def fake_judge(prompt):
        prompts.append(prompt)
        return '{"would_have_helped": [], "dismissed": []}'

    monkeypatch.setattr(SM, "_run_judge", fake_judge)
    catalog = "\n".join(f"- {n}: {d}" for n, d in sorted(_CATALOG.items()))
    assert SM.judge_session(_session(), catalog) == {"would_have_helped": [], "dismissed": []}
    (p,) = prompts
    assert "- db-query: Query the app Postgres with psql." in p  # description reaches judge
    assert "description_covers" in p and "missing_phrasing" in p
