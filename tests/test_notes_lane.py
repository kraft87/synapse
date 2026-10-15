"""Pure tests for the dream→notes curation lane (dream/notes/nightly.py).

No database and no network: verdict parsing, the both-orders consensus collapse,
winner selection, the scope guards, and the orchestrator's apply budget, all driven
through a fake DB + a stubbed judge (the pair judge, ``judge_pair``, is shared with
ingestion/notes.py's write path). The DB-backed half lives in tests/test_notes_curation.py.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

import dream.notes.nightly as lane
from dream.notes.nightly import parse_scope_verdict, resolve_pair, run_lane
from ingestion.llm_client import MalformedResponseError
from ingestion.notes import parse_pair_verdict

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _note(
    note_id: int,
    *,
    type="user",
    project=None,
    days=0,
    hook=None,
    body="Body.",
    audience="personal",
):
    return {
        "id": note_id,
        "hook": hook or f"Example note {note_id}",
        "body": body,
        "type": type,
        "project": project,
        "audience": audience,
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
        (_DIS, _DIS, "first-order-distinct"),
        (_DIS, None, "first-order-distinct"),
        (_DUP, _DIS, "order-disagreement"),
        (_DUP, _corr(1), "order-disagreement"),
    ],
)
def test_disagreement_and_distinct_collapse_to_the_safe_verdict(v_ab, v_ba, reason):
    res = resolve_pair(_note(10), _note(11, days=1), v_ab, v_ba)
    assert res == {"verdict": "DISTINCT", "winner": None, "loser": None, "reason": reason}


def test_a_pair_the_first_order_did_not_settle_needs_the_second():
    with pytest.raises(ValueError, match="both orderings"):
        resolve_pair(_note(10), _note(11, days=1), _DUP, None)


def test_cross_type_duplicate_is_left_alone():
    """A 'duplicate' spanning two types is a mis-typing, not a redundancy — merging
    would silently pick one shelf. Stage C owns that case. The first order settles it."""
    a = _note(10, type="feedback")
    b = _note(11, type="project", project="demo-api", days=1)
    for v_ba in (_DUP, None):
        res = resolve_pair(a, b, _DUP, v_ba)
        assert res["verdict"] == "DISTINCT" and res["reason"] == "cross-type-duplicate"


def test_cross_type_correction_is_allowed():
    """A correction may cross types when the winner's board covers the loser's: here the
    global rule corrects a project note that had restated it differently."""
    a = _note(10, type="feedback")
    b = _note(11, type="project", project="demo-api", days=1)
    # keep=1 in the (a,b) ordering and keep=2 in the (b,a) ordering both name note 10.
    res = resolve_pair(a, b, _corr(1), _corr(2))
    assert res == {"verdict": "CORRECTION", "winner": 10, "loser": 11, "reason": "consensus"}


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
# The winner must be visible wherever the loser was
# ---------------------------------------------------------------------------


def test_a_project_note_never_retires_a_note_of_another_project():
    alpha = _note(10, type="project", project="alpha")
    beta = _note(11, type="project", project="beta", days=1)
    res = resolve_pair(alpha, beta, _DUP, _DUP)
    assert res["verdict"] == "DISTINCT" and res["reason"] == "winner-narrower-than-loser"
    res = resolve_pair(alpha, beta, _corr(2), _corr(1))
    assert res["verdict"] == "DISTINCT" and res["reason"] == "winner-narrower-than-loser"


def test_a_project_note_does_not_retire_a_global_rule():
    """A correction filed under one project must not remove the rule from every other."""
    rule = _note(10, type="feedback")
    scoped = _note(11, type="project", project="alpha", days=1)
    res = resolve_pair(rule, scoped, _corr(2), _corr(1))  # keep = the project note
    assert res["verdict"] == "DISTINCT" and res["reason"] == "winner-narrower-than-loser"


def test_a_global_note_may_retire_a_project_note():
    """The reverse direction loses nothing: a global winner is on every board."""
    rule = _note(10, type="feedback")
    scoped = _note(11, type="project", project="alpha", days=1)
    res = resolve_pair(rule, scoped, _corr(1), _corr(2))  # keep = the global rule
    assert res == {"verdict": "CORRECTION", "winner": 10, "loser": 11, "reason": "consensus"}


def test_same_project_notes_merge():
    a = _note(10, type="project", project="alpha")
    b = _note(11, type="project", project="alpha", days=1)
    assert resolve_pair(a, b, _DUP, _DUP)["verdict"] == "DUPLICATE"


def test_a_personal_note_does_not_retire_a_work_safe_one():
    """The work board would lose the line and gain nothing."""
    work = _note(10, audience="work-safe")
    personal = _note(11, days=1)  # the newer wins a duplicate
    res = resolve_pair(work, personal, _DUP, _DUP)
    assert res["verdict"] == "DISTINCT" and res["reason"] == "audience-narrowing"
    # ...but a work-safe winner may retire a personal duplicate, and equal tiers merge.
    res = resolve_pair(personal, _note(12, days=2, audience="work-safe"), _DUP, _DUP)
    assert res["verdict"] == "DUPLICATE" and res["loser"] == 11


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
        self.retyped_audience: list[str | None] = []
        self.curation: list[dict] = []
        self.closed = False

    def find_note_pair_candidates(self, owner_id, *, sim_floor, limit):
        return [p for p in self.pairs if p["sim"] >= sim_floor][:limit]

    def find_retype_candidates(self, owner_id, *, limit):
        return self.retypes[:limit]

    def known_project_slugs(self, limit=40):
        return list(self.projects)

    def supersede_note(self, old_id, new_id):
        self.superseded.append((old_id, new_id))

    def retype_note(self, note_id, *, type, project, audience=None):
        self.retyped.append((note_id, type, project))
        self.retyped_audience.append(audience)

    def restricted_surface_projects(self):
        """No registered restricted surface -> every re-project derives 'personal'."""
        return []

    def record_curation(self, **kw):
        self.curation.append(kw)
        return len(self.curation)

    def close(self):
        self.closed = True


def _stub_llm(monkeypatch, pair_verdicts=(), scope_verdicts=(), fail_pairs=False):
    """Replace the judges: pair calls pop from one queue, scope calls the other."""
    pairs, scopes, calls = list(pair_verdicts), list(scope_verdicts), []

    def _fake_pair(llm, first, second):
        calls.append((first["id"], second["id"]))
        if fail_pairs:
            raise RuntimeError("judge backend down")
        return pairs.pop(0)

    def _fake_scope(llm, *, base_prompt, parser, model, max_tokens):
        calls.append(parser.__name__)
        return scopes.pop(0)

    monkeypatch.setattr(lane, "judge_pair", _fake_pair)
    monkeypatch.setattr(lane, "parse_with_retry", _fake_scope)
    return calls


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


def test_a_first_order_distinct_costs_one_judge_call(monkeypatch):
    """The swapped order could not change a DISTINCT, so it is never asked. The audit row
    keeps the one verdict and says why the pair ended there."""
    calls = _stub_llm(monkeypatch, pair_verdicts=[_DIS])
    db = _FakeDB(pairs=[(_note(1), _note(2), 0.71)])
    run_lane(db=db, llm=object())
    assert calls == [(1, 2)]
    (row,) = db.curation
    assert row["detail"]["reason"] == "first-order-distinct" and row["detail"]["orders"] == [_DIS]


def test_a_cross_type_duplicate_costs_one_judge_call(monkeypatch):
    calls = _stub_llm(monkeypatch, pair_verdicts=[_DUP])
    pair = (_note(1, type="feedback"), _note(2, type="project", project="demo-api"), 0.8)
    db = _FakeDB(pairs=[pair])
    run_lane(db=db, llm=object())
    assert calls == [(1, 2)] and db.superseded == []
    assert db.curation[0]["detail"]["reason"] == "cross-type-duplicate"


def test_a_merge_still_needs_both_orders(monkeypatch):
    calls = _stub_llm(monkeypatch, pair_verdicts=[_DUP, _DUP])
    db = _FakeDB(pairs=[(_note(1), _note(2, days=1), 0.9)])
    run_lane(db=db, llm=object())
    assert calls == [(1, 2), (2, 1)] and db.superseded == [(1, 2)]


def test_a_refused_merge_is_memoized_with_its_reason(monkeypatch):
    """Two project notes would never reach the lane (the SQL drops them); a project winner
    over a global loser can, and is refused after the judge has spoken."""
    _stub_llm(monkeypatch, pair_verdicts=[_corr(2), _corr(1)])
    rule, scoped = _note(1, type="feedback"), _note(2, type="project", project="alpha", days=1)
    db = _FakeDB(pairs=[(rule, scoped, 0.8)])
    res = run_lane(db=db, llm=object())
    assert db.superseded == [] and res["counts"]["applied_supersede"] == 0
    assert db.curation[0]["verdict"] == "DISTINCT"
    assert db.curation[0]["detail"]["reason"] == "winner-narrower-than-loser"


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
    # A re-project re-derives audience by the project rule (schema 053). No restricted
    # surface allows 'demo-api' here, so it lands back on the fail-closed tier.
    assert db.retyped_audience == ["personal"]
    assert res["counts"]["applied_retype"] == 1
    assert res["samples"] == ["n:7 feedback -> project:demo-api"]
    (row,) = db.curation
    assert row["verdict"] == "PROJECT" and row["applied"] is True
    assert row["detail"] == {
        "from_type": "feedback",
        "from_project": None,
        "from_audience": "personal",
        "project": "demo-api",
        "audience": "personal",
    }


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


def test_limits_are_arguments_with_constant_defaults(monkeypatch):
    _stub_llm(monkeypatch, pair_verdicts=[_DUP, _DUP])
    db = _FakeDB(pairs=[(_note(1), _note(2, days=1), 0.90), (_note(3), _note(4, days=1), 0.86)])
    res = run_lane(db=db, llm=object(), max_apply=1, sim_floor=0.85)
    assert res["counts"]["applied_supersede"] == 1  # budget of 1 stopped the second
    db2 = _FakeDB(pairs=[(_note(5), _note(6), 0.70)])
    _stub_llm(monkeypatch, pair_verdicts=[_DIS])
    assert run_lane(db=db2, llm=object(), sim_floor=0.85)["counts"]["pairs_judged"] == 0
    assert run_lane(db=db2, llm=object())["counts"]["pairs_judged"] == 1  # default floor 0.60


def test_retype_slug_match_is_case_folded_and_filed_under_the_stored_casing(monkeypatch):
    _stub_llm(monkeypatch, scope_verdicts=[{"scope": "PROJECT", "project": "demo-api"}])
    db = _FakeDB(retypes=[_note(7, type="feedback")], projects=("Demo-API",))
    res = run_lane(db=db, llm=object())
    assert db.retyped == [(7, "project", "Demo-API")]
    assert res["counts"]["applied_retype"] == 1
    assert db.curation[0]["detail"]["project"] == "Demo-API"


def test_retype_audience_follows_the_case_folded_allowlist(monkeypatch):
    """remember() and the retype stage apply one rule: an allowlist entry 'Synapse' makes a
    note retyped to 'synapse' work-safe."""
    _stub_llm(monkeypatch, scope_verdicts=[{"scope": "PROJECT", "project": "synapse"}])
    db = _FakeDB(retypes=[_note(7, type="feedback")], projects=("synapse",))
    db.restricted_surface_projects = lambda: ["Synapse"]
    run_lane(db=db, llm=object())
    assert db.retyped_audience == ["work-safe"]
