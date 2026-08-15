"""Pure tests for the dream→notes curation lane (dream/notes/nightly.py).

No database and no network: verdict parsing, the both-orders consensus collapse,
winner selection, and the orchestrator's apply budget, all driven through a fake
DB + a stubbed ``parse_with_retry`` (the same seam ingestion/notes.py's tests use
for NOTES_CONFIRM). The DB-backed half lives in tests/test_notes_curation.py.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

import dream.notes.nightly as lane
from dream.notes.nightly import (
    parse_pair_verdict,
    parse_scope_verdict,
    resolve_pair,
    run_lane,
)
from ingestion.llm_client import MalformedResponseError

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _note(note_id: int, *, type="user", project=None, days=0, hook=None, body="Body."):
    return {
        "id": note_id,
        "hook": hook or f"Example note {note_id}",
        "body": body,
        "type": type,
        "project": project,
        "updated_at": _T0 + timedelta(days=days),
    }


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"verdict": "DISTINCT"}', {"verdict": "DISTINCT", "keep": None}),
        ('{"verdict": "duplicate"}', {"verdict": "DUPLICATE", "keep": None}),
        ('{"verdict": "CORRECTION", "keep": 2}', {"verdict": "CORRECTION", "keep": 2}),
        # Chatty models wrap the object in prose; the extractor takes the outermost braces.
        ('Sure. {"verdict": "DISTINCT"} — both stand.', {"verdict": "DISTINCT", "keep": None}),
        # A stringified index is still an index.
        ('{"verdict": "CORRECTION", "keep": "1"}', {"verdict": "CORRECTION", "keep": 1}),
        # A stray keep on a non-CORRECTION verdict is carried, not an error; the
        # resolver ignores it (only CORRECTION reads a direction).
        ('{"verdict": "DUPLICATE", "keep": 1}', {"verdict": "DUPLICATE", "keep": 1}),
    ],
)
def test_parse_pair_verdict_ok(text, expected):
    assert parse_pair_verdict(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "no json here",
        '{"verdict": "MERGE"}',  # not one of the three
        '{"verdict": "CORRECTION"}',  # direction is mandatory — never guessed
        '{"verdict": "CORRECTION", "keep": 3}',  # out of range -> dropped -> missing
        '{"verdict": "DUPLICATE"',  # truncated
        '["verdict"]',  # not an object
    ],
)
def test_parse_pair_verdict_rejects(text):
    with pytest.raises(MalformedResponseError):
        parse_pair_verdict(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"scope": "GLOBAL"}', {"scope": "GLOBAL", "project": None}),
        ('{"scope": "global", "project": "x"}', {"scope": "GLOBAL", "project": None}),
        (
            '{"scope": "PROJECT", "project": " demo-api "}',
            {"scope": "PROJECT", "project": "demo-api"},
        ),
    ],
)
def test_parse_scope_verdict_ok(text, expected):
    assert parse_scope_verdict(text) == expected


@pytest.mark.parametrize(
    "text", ['{"scope": "PROJECT"}', '{"scope": "PROJECT", "project": "  "}', '{"scope": "MAYBE"}']
)
def test_parse_scope_verdict_rejects(text):
    with pytest.raises(MalformedResponseError):
        parse_scope_verdict(text)


# ---------------------------------------------------------------------------
# Both-orders consensus + winner selection
# ---------------------------------------------------------------------------

_DUP = {"verdict": "DUPLICATE", "keep": None}
_DIS = {"verdict": "DISTINCT", "keep": None}


def _corr(keep):
    return {"verdict": "CORRECTION", "keep": keep}


def test_duplicate_consensus_retires_the_older_note():
    a, b = _note(10, days=0), _note(11, days=5)
    res = resolve_pair(a, b, _DUP, _DUP)
    assert res == {"verdict": "DUPLICATE", "winner": 11, "loser": 10, "reason": "consensus"}


def test_duplicate_winner_is_updated_at_not_id():
    """The newer EDIT wins even when it carries the lower id — a note updated in
    place (remember() restating it) is the current phrasing."""
    a, b = _note(10, days=9), _note(11, days=1)
    res = resolve_pair(a, b, _DUP, _DUP)
    assert res["winner"] == 10 and res["loser"] == 11


def test_duplicate_tie_on_updated_at_breaks_on_id():
    res = resolve_pair(_note(10), _note(11), _DUP, _DUP)
    assert res["winner"] == 11 and res["loser"] == 10


@pytest.mark.parametrize(
    ("v_ab", "v_ba", "reason"),
    [
        (_DIS, _DIS, "both-distinct"),
        (_DUP, _DIS, "order-disagreement"),
        (_DIS, _DUP, "order-disagreement"),
        (_DUP, _corr(1), "order-disagreement"),
    ],
)
def test_disagreement_and_distinct_collapse_to_the_safe_verdict(v_ab, v_ba, reason):
    res = resolve_pair(_note(10), _note(11, days=1), v_ab, v_ba)
    assert res == {"verdict": "DISTINCT", "winner": None, "loser": None, "reason": reason}


def test_cross_type_duplicate_is_left_alone():
    """A 'duplicate' spanning two types is a mis-typing, not a redundancy — merging
    would silently pick one shelf. Stage C owns that case."""
    a = _note(10, type="feedback")
    b = _note(11, type="project", project="demo-api", days=1)
    res = resolve_pair(a, b, _DUP, _DUP)
    assert res["verdict"] == "DISTINCT" and res["reason"] == "cross-type-duplicate"


def test_cross_type_correction_is_allowed():
    a = _note(10, type="feedback")
    b = _note(11, type="project", project="demo-api", days=1)
    # keep=2 in the (a,b) ordering and keep=1 in the (b,a) ordering both name note 11.
    res = resolve_pair(a, b, _corr(2), _corr(1))
    assert res == {"verdict": "CORRECTION", "winner": 11, "loser": 10, "reason": "consensus"}


def test_correction_direction_must_agree_on_the_id_not_the_slot():
    """Both answers say "keep NOTE 1" — but the contents were swapped between the
    calls, so they name different notes. That is position bias, not a verdict."""
    res = resolve_pair(_note(10), _note(11, days=1), _corr(1), _corr(1))
    assert res["verdict"] == "DISTINCT" and res["reason"] == "direction-disagreement"


def test_correction_can_retire_the_newer_note():
    """Direction comes from the model, not from the clock: a note can be written
    first and still be the one that corrects (e.g. a later restatement of a rule
    the user already retracted)."""
    a, b = _note(10, days=9), _note(11, days=1)
    res = resolve_pair(a, b, _corr(2), _corr(1))
    assert res["winner"] == 11 and res["loser"] == 10


def test_correction_without_direction_is_distinct():
    res = resolve_pair(_note(10), _note(11, days=1), _corr(None), _corr(None))
    assert res["verdict"] == "DISTINCT" and res["reason"] == "missing-direction"


# ---------------------------------------------------------------------------
# Orchestration — fake DB, stubbed LLM
# ---------------------------------------------------------------------------


class _FakeDB:
    def __init__(self, pairs=(), retypes=(), projects=("demo-api",)):
        self.pairs = [{"a": a, "b": b, "sim": sim} for a, b, sim in pairs]
        self.retypes = list(retypes)
        self.projects = list(projects)
        self.superseded: list[tuple[int, int]] = []
        self.retyped: list[tuple[int, str, str]] = []
        self.curation: list[dict] = []
        self.closed = False

    def find_note_pair_candidates(self, owner_id, *, sim_floor, limit):
        return [p for p in self.pairs if p["sim"] >= sim_floor][:limit]

    def find_retype_candidates(self, owner_id, *, limit):
        return self.retypes[:limit]

    def known_project_slugs(self, limit=40):
        return list(self.projects)

    def project_slug_exists(self, slug):
        return slug in self.projects

    def supersede_note(self, old_id, new_id):
        self.superseded.append((old_id, new_id))

    def retype_note(self, note_id, *, type, project):
        self.retyped.append((note_id, type, project))

    def record_curation(self, **kw):
        self.curation.append(kw)
        return len(self.curation)

    def close(self):
        self.closed = True


def _stub_llm(monkeypatch, pair_verdicts=(), scope_verdicts=(), fail_pairs=False):
    """Replace parse_with_retry: pair calls pop from one queue, scope calls the other."""
    pairs, scopes, calls = list(pair_verdicts), list(scope_verdicts), []

    def _fake(llm, *, base_prompt, parser, model, max_tokens):
        calls.append(parser.__name__)
        if parser is lane.parse_pair_verdict:
            if fail_pairs:
                raise RuntimeError("judge backend down")
            return pairs.pop(0)
        return scopes.pop(0)

    monkeypatch.setattr(lane, "parse_with_retry", _fake)
    return calls


def test_kill_switch_returns_zeroed_counts_without_touching_anything(monkeypatch):
    monkeypatch.setenv("SYNAPSE_NOTES_LANE", "0")
    db = _FakeDB(pairs=[(_note(1), _note(2, days=1), 0.99)])
    res = run_lane(db=db, llm=object())
    assert res["enabled"] is False
    assert res["counts"] == {
        "pairs_judged": 0,
        "retype_judged": 0,
        "applied_supersede": 0,
        "applied_retype": 0,
    }
    assert db.curation == [] and db.superseded == []


def test_pair_merge_records_audit_and_sample(monkeypatch):
    _stub_llm(monkeypatch, pair_verdicts=[_DUP, _DUP])
    db = _FakeDB(pairs=[(_note(1), _note(2, days=1), 0.94)])
    res = run_lane(db=db, llm=object())
    assert db.superseded == [(1, 2)]
    assert res["counts"]["applied_supersede"] == 1 and res["counts"]["pairs_judged"] == 1
    assert res["samples"] == ["n:1 -> superseded by n:2 (duplicate)"]
    (row,) = db.curation
    assert row["op"] == "pair" and row["note_a"] == 1 and row["note_b"] == 2
    assert row["verdict"] == "DUPLICATE" and row["applied"] is True
    assert row["detail"]["sim"] == 0.94 and row["detail"]["reason"] == "consensus"
    assert len(row["detail"]["orders"]) == 2  # both judgements kept for the audit trail


def test_distinct_pair_is_memoized_but_not_applied(monkeypatch):
    _stub_llm(monkeypatch, pair_verdicts=[_DIS, _DIS])
    db = _FakeDB(pairs=[(_note(1), _note(2), 0.71)])
    res = run_lane(db=db, llm=object())
    assert db.superseded == [] and res["counts"]["applied_supersede"] == 0
    assert db.curation[0]["verdict"] == "DISTINCT" and db.curation[0]["applied"] is False


def test_sim_floor_filters_candidates(monkeypatch):
    _stub_llm(monkeypatch, pair_verdicts=[_DIS, _DIS])
    db = _FakeDB(pairs=[(_note(1), _note(2), 0.90), (_note(3), _note(4), 0.42)])
    res = run_lane(db=db, llm=object(), sim_floor=0.60)
    assert res["counts"]["pairs_judged"] == 1


def test_judge_failure_skips_the_pair_without_memoizing(monkeypatch):
    """A transient backend failure must NOT freeze a DISTINCT verdict in — the pair
    has to come back tomorrow, so no audit row is written."""
    _stub_llm(monkeypatch, fail_pairs=True)
    db = _FakeDB(pairs=[(_note(1), _note(2), 0.95)])
    res = run_lane(db=db, llm=object())
    assert db.curation == [] and db.superseded == []
    assert res["counts"]["pairs_judged"] == 0 and len(res["errors"]) == 1


def test_second_pair_touching_a_retired_note_is_skipped(monkeypatch):
    """Candidates are materialized up front; once note 1 is retired its other pairs
    are stale, so they are skipped rather than judged against a dead row."""
    _stub_llm(monkeypatch, pair_verdicts=[_DUP, _DUP])
    n1, n2, n3 = _note(1), _note(2, days=1), _note(3, days=2)
    db = _FakeDB(pairs=[(n1, n2, 0.95), (n1, n3, 0.80)])
    res = run_lane(db=db, llm=object())
    assert db.superseded == [(1, 2)] and res["counts"]["pairs_judged"] == 1


def test_apply_budget_stops_judging_so_memo_rows_stay_truthful(monkeypatch):
    """The cap stops the loop BEFORE the next judgement, so notes_curation never
    holds a verdict that was silently not acted on."""
    calls = _stub_llm(monkeypatch, pair_verdicts=[_DUP, _DUP, _DUP, _DUP])
    db = _FakeDB(
        pairs=[
            (_note(1), _note(2, days=1), 0.95),
            (_note(3), _note(4, days=1), 0.94),
            (_note(5), _note(6, days=1), 0.93),
        ]
    )
    res = run_lane(db=db, llm=object(), max_apply=2)
    assert res["counts"]["applied_supersede"] == 2
    assert res["counts"]["pairs_judged"] == 2 and len(db.curation) == 2
    assert len(calls) == 4  # two pairs x two orderings; the third was never judged


def test_apply_budget_is_shared_across_both_stages(monkeypatch):
    _stub_llm(monkeypatch, pair_verdicts=[_DUP, _DUP], scope_verdicts=[])
    db = _FakeDB(
        pairs=[(_note(1), _note(2, days=1), 0.95)],
        retypes=[_note(7, type="feedback")],
    )
    res = run_lane(db=db, llm=object(), max_apply=1)
    assert res["counts"]["applied_supersede"] == 1
    assert res["counts"]["retype_judged"] == 0 and db.retyped == []


def test_retype_applies_for_a_known_project(monkeypatch):
    _stub_llm(monkeypatch, scope_verdicts=[{"scope": "PROJECT", "project": "demo-api"}])
    db = _FakeDB(retypes=[_note(7, type="feedback", hook="demo-api runs migrations on boot")])
    res = run_lane(db=db, llm=object())
    assert db.retyped == [(7, "project", "demo-api")]
    assert res["counts"]["applied_retype"] == 1
    assert res["samples"] == ["n:7 feedback -> project:demo-api"]
    (row,) = db.curation
    assert row["verdict"] == "PROJECT" and row["applied"] is True
    assert row["detail"] == {"from_type": "feedback", "project": "demo-api"}


def test_retype_skips_an_unknown_project_slug_but_records_it(monkeypatch):
    _stub_llm(monkeypatch, scope_verdicts=[{"scope": "PROJECT", "project": "ghost-service"}])
    db = _FakeDB(retypes=[_note(7, type="user")], projects=("demo-api",))
    res = run_lane(db=db, llm=object())
    assert db.retyped == [] and res["counts"]["applied_retype"] == 0
    assert db.curation[0]["applied"] is False
    assert db.curation[0]["detail"]["reason"] == "unknown-project-slug"


def test_global_verdict_is_memoized_and_leaves_the_note_alone(monkeypatch):
    _stub_llm(monkeypatch, scope_verdicts=[{"scope": "GLOBAL", "project": None}])
    db = _FakeDB(retypes=[_note(7, type="user", hook="User prefers numbered steps")])
    res = run_lane(db=db, llm=object())
    assert db.retyped == [] and res["counts"]["retype_judged"] == 1
    assert db.curation[0]["verdict"] == "GLOBAL" and db.curation[0]["applied"] is False


def test_injected_db_is_not_closed_by_the_lane(monkeypatch):
    _stub_llm(monkeypatch, pair_verdicts=[_DIS, _DIS])
    db = _FakeDB(pairs=[(_note(1), _note(2), 0.9)])
    run_lane(db=db, llm=object())
    assert db.closed is False  # the caller owns an injected connection


def test_caps_read_from_env_when_not_passed(monkeypatch):
    monkeypatch.setenv("NOTES_LANE_MAX_APPLY", "1")
    monkeypatch.setenv("NOTES_LANE_SIM_FLOOR", "0.85")
    _stub_llm(monkeypatch, pair_verdicts=[_DUP, _DUP])
    db = _FakeDB(pairs=[(_note(1), _note(2, days=1), 0.90), (_note(3), _note(4, days=1), 0.86)])
    res = run_lane(db=db, llm=object())
    assert res["counts"]["applied_supersede"] == 1  # budget of 1 stopped the second


def test_bad_env_value_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("NOTES_LANE_MAX_JUDGE", "not-a-number")
    _stub_llm(monkeypatch, pair_verdicts=[_DIS, _DIS])
    db = _FakeDB(pairs=[(_note(1), _note(2), 0.9)])
    assert run_lane(db=db, llm=object())["counts"]["pairs_judged"] == 1
