"""The rules of scripts/repair_numbered_entity_magnets.py, without a database.

The old resolver let near-identical numbered names ("PR #290" / "PR #269", "ABC-4100" /
"ABC-4102") resolve onto one node, so one entity collected the facts of dozens of IDs.
These tests pin what the repair moves and what it leaves alone:

* which entity names count as an identifier (whole-name shapes only, one number by the
  resolver guard's own reading);
* what a fact text names (kind words, bare '#N', URLs, ranges, ticket keys);
* the per-endpoint decision (own number / none / several / range / bare / move);
* the plan: targets reused or created (named as written), self-joins and cross-group edges
  skipped, retired edges moved with their successors, summaries of losers cleared;
* supersession candidates: disjoint identifiers only, never when another live edge on
  the pair carries the old fact's identifier, never for an expired edge.
The end-to-end writes, backup and restore are in test_repair_numbered_entity_magnets_db.py.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from ingestion.dedup import ids_conflict

_SPEC = importlib.util.spec_from_file_location(
    "repair_numbered_entity_magnets",
    Path(__file__).resolve().parent.parent / "scripts" / "repair_numbered_entity_magnets.py",
)
rm = importlib.util.module_from_spec(_SPEC)
sys.modules["repair_numbered_entity_magnets"] = rm
_SPEC.loader.exec_module(rm)

Ident, Entity, Edge = rm.Ident, rm.Entity, rm.Edge
G = "technical"


# ---------------------------------------------------------------- entity names


@pytest.mark.parametrize(
    ("name", "want"),
    [
        ("PR #290", Ident("pr", "290")),
        ("pr #0290", Ident("pr", "290")),  # leading zeros dropped, as the guard does
        ("draft PR #188", Ident("pr", "188")),
        ("pull request #7", Ident("pr", "7")),
        ("Issue #33", Ident("issue", "33")),
        ("Task #60", Ident("task", "60")),
        ("ABC-4100", Ident("key:ABC", "4100")),
        ("widget refactor (PR #12)", None),  # decorated: the extra words carry meaning
        ("PR #12 rules", None),
        ("PRs #180-182", None),  # two numbers
        ("abc-123", None),  # a key is upper-case
        ("Python 3.12", None),
        ("GPT-4", None),  # one digit: a version, not a ticket
        ("#5", None),  # a bare number says nothing about its kind
        ("PR", None),
    ],
)
def test_entity_ident_takes_whole_name_identifiers_only(name, want):
    assert rm.entity_ident(name) == want


def test_target_names_are_the_bare_identifier():
    assert rm.is_target_name("PR #218", Ident("pr", "218"))
    assert rm.is_target_name("pull request 218", Ident("pr", "218"))
    assert not rm.is_target_name("draft PR #218", Ident("pr", "218"))
    assert not rm.is_target_name("PR #219", Ident("pr", "218"))
    assert rm.is_target_name("abc-12", Ident("key:ABC", "12"))
    assert not rm.is_target_name("XYZ-12", Ident("key:ABC", "12"))


# ---------------------------------------------------------------- fact text


def _m(text):
    return [(m.kind, m.number, m.written, m.multi) for m in rm.mentions(text)]


def test_mentions_read_kind_words_bare_hashes_urls_and_keys():
    assert _m("Merged PR #218 after #205") == [
        ("pr", "218", "PR #218", False),
        ("bare", "205", "#205", False),
    ]
    assert _m("PRs #12 and pull requests #9") == [
        ("pr", "12", "PR #12", False),
        ("pr", "9", "pull request #9", False),
    ]
    assert _m("pr 44 landed") == [("pr", "44", "pr #44", False)]
    assert _m("see https://example.com/o/r/pull/42 and /issues/7") == [
        ("pr", "42", "PR #42", False),
        ("issue", "7", "issue #7", False),
    ]
    assert _m("ticket abc-123 blocks ABC-124") == [
        ("key:ABC", "123", "ABC-123", False),
        ("key:ABC", "124", "ABC-124", False),
    ]


def test_mentions_skip_prose_numbers_and_flag_ranges():
    # "issue 5" / "task 3" without '#' are prose; '#' inside a word or an entity is not one.
    assert _m("issue 5 and task 3 and widget#851 and &#39;") == []
    assert _m("PRs #117-119 shipped")[0][3] is True
    assert _m("PR #5 to #9")[0][3] is True
    assert _m("ABC-12/13 closed")[0][3] is True


def test_identifier_set_shares_one_namespace_for_hash_kinds_and_expands_ranges():
    assert rm.identifier_set("PR #12 and issue #12") == {("#", "12")}
    assert ("#", "118") in rm.identifier_set("PRs #117-119 merged")
    assert rm.identifier_set("ABC-12 is the epic") == {("key:ABC", "12")}
    assert rm.identifier_set("no ids here") == frozenset()


# ---------------------------------------------------------------- one endpoint


@pytest.mark.parametrize(
    ("fact", "verdict", "number"),
    [
        ("PR #218 was merged", rm.MOVE, "218"),
        ("PR #290 shipped the fix", rm.STAY_OWN, None),
        ("pr #0290 shipped", rm.STAY_OWN, None),  # same number however written
        ("the fix in #290 shipped", rm.STAY_OWN, None),
        ("the branch was rebased", rm.STAY_NONE, None),
        ("issue #218 was closed", rm.STAY_NONE, None),  # another kind is not this one's ID
        ("PR #211 was closed because PR #212 does it", rm.STAY_SEVERAL, None),
        ("PRs #218 and #222 merged", rm.STAY_SEVERAL, None),
        ("PR #218 merged; PR #218 deployed", rm.MOVE, "218"),
        ("PRs #117-119 merged", rm.STAY_RANGE, None),
        ("merged #218 today", rm.STAY_BARE, None),  # kind unknown: leave it
    ],
)
def test_decide_for_a_pr_endpoint(fact, verdict, number):
    got, mention = rm.decide(Ident("pr", "290"), fact)
    assert got == verdict
    assert (mention.number if mention else None) == number


def test_decide_for_a_ticket_key_endpoint():
    key = Ident("key:ABC", "4100")
    assert rm.decide(key, "ABC-4021 is blocked by a vendor")[0] == rm.MOVE
    assert rm.decide(key, "ABC-4100 is the epic")[0] == rm.STAY_OWN
    assert rm.decide(key, "XYZ-4021 is blocked")[0] == rm.STAY_NONE  # other prefix
    assert rm.decide(key, "ABC-4021 replaces ABC-3990")[0] == rm.STAY_SEVERAL
    assert rm.decide(key, "PR #5 closes it")[0] == rm.STAY_NONE


# ---------------------------------------------------------------- the plan


def _ent(uuid, name, *, group=G, summary="", etype="pull_request", project="alpha"):
    return Entity(uuid, "default", group, name, etype, "artifact", etype, project, summary)


def _edge(
    uuid, src, tgt, fact, *, live=True, by=None, group=G, created="2026-01-01", expired=False
):
    return Edge(uuid, "default", group, src, tgt, fact, "RELATES", live, by, expired, created)


def _world():
    ents = [
        _ent("m", "PR #290", summary="a PR that did something else entirely"),
        _ent("p259", "PR #259"),
        _ent("x", "widget service", etype="service"),
        _ent("y", "release branch", etype="branch"),
    ]
    edges = [
        _edge("e1", "m", "x", "PR #218 was merged into the widget service"),
        _edge("e2", "m", "y", "PR #218 was rebased onto the release branch"),
        _edge("e3", "m", "x", "PR #290 changed the widget service"),
        _edge("e4", "m", "x", "PR #259 added a retry to the widget service"),
        _edge("e5", "m", "y", "the release branch was cut"),
    ]
    return ents, edges


def test_plan_moves_each_edge_to_the_entity_named_for_its_id():
    ents, edges = _world()
    plan = rm.plan_repair(ents, edges)
    moved = {m.edge: m for m in plan.moves}
    assert set(moved) == {"e1", "e2", "e4"}
    by_name = {t.name: t for t in plan.targets.values()}
    # PR #218 has no entity yet: one is created, both facts land on it.
    new = by_name["PR #218"]
    assert not new.existing and new.moves == 2
    assert moved["e1"].to_uuid == moved["e2"].to_uuid == new.uuid
    assert new.template.entity_type == "pull_request"
    # PR #259 exists: reused, never duplicated.
    assert moved["e4"].to_uuid == "p259" and by_name["PR #259"].existing
    assert all(m.side == "src" and m.from_uuid == "m" for m in plan.moves)
    # The shipped guard agrees every move separates two different numbers.
    assert all(ids_conflict("PR #290", t.name) for t in plan.targets.values())
    assert plan.clear_summaries == ["m"]


def test_plan_lands_on_an_existing_entity_spelled_another_way():
    # 'PR 140' and 'abc-123' are the identifiers 'PR #140' and 'ABC-123': the moves land
    # there instead of creating a second entity for the same number.
    ents = [
        _ent("m", "PR #290"),
        _ent("p140", "PR 140"),
        _ent("km", "ABC-100", etype="ticket"),
        _ent("k123", "abc-123", etype="ticket"),
        _ent("x", "widget"),
    ]
    edges = [
        _edge("a", "m", "x", "PR #140 was merged"),
        _edge("b", "km", "x", "ABC-123 tracks the widget"),
        _edge("c", "p140", "x", "PR #12 was opened"),
    ]
    plan = rm.plan_repair(ents, edges)
    assert {m.edge: m.to_uuid for m in plan.moves} == {"a": "p140", "b": "k123"}
    assert plan.new_targets() == []
    # A landing is not a magnet: what sits on 'PR 140' is not read for other numbers.
    assert rm.entity_ident("PR 140") is None


def test_plan_names_a_new_entity_as_the_facts_write_it():
    ents = [_ent("m", "PR #290"), _ent("x", "widget")]
    edges = [
        _edge("a", "m", "x", "pull request #31 merged"),
        _edge("b", "m", "x", "PR #31 deployed"),
        _edge("c", "m", "x", "PR #31 reverted"),
    ]
    (target,) = rm.plan_repair(ents, edges).targets.values()
    assert target.name == "PR #31"


def test_plan_never_joins_an_edge_to_itself_or_crosses_groups():
    ents = [
        _ent("m", "PR #290"),
        _ent("p218", "PR #218"),
        _ent("other", "PR #77", group="personal"),
        _ent("x", "widget"),
    ]
    edges = [
        # Moving the magnet end onto PR #218 would make PR #218 -> PR #218.
        _edge("self", "m", "p218", "PR #218 was reviewed"),
        # The entity sits in another group than the edge: nothing to say about it here.
        _edge("cross", "other", "x", "PR #12 was merged"),
    ]
    plan = rm.plan_repair(ents, edges)
    assert plan.moves == []
    assert plan.stats["pr #"]["stay: move would join the edge to itself"] == 1
    assert plan.stats["pr #"]["stay: edge in another group"] == 1


def test_both_endpoints_move_when_each_names_its_own_kind():
    ents = [_ent("m", "PR #290"), _ent("k", "ABC-100", etype="ticket")]
    edges = [_edge("e", "m", "k", "PR #12 implements ABC-123")]
    plan = rm.plan_repair(ents, edges)
    sides = {
        m.side: plan.targets[next(k for k, t in plan.targets.items() if t.uuid == m.to_uuid)].name
        for m in plan.moves
    }
    assert sides == {"src": "PR #12", "tgt": "ABC-123"}


def test_a_linked_supersession_about_one_id_moves_together():
    """Retired edges move by the same rule, so a correction about PR #218 stays on one
    (src, tgt) pair, which the pair-joined supersession reader needs."""
    ents = [_ent("m", "PR #290"), _ent("x", "widget")]
    edges = [
        _edge("old", "m", "x", "PR #218 is waiting on review", live=False, by="new"),
        _edge("new", "m", "x", "PR #218 was approved", created="2026-01-02"),
    ]
    plan = rm.plan_repair(ents, edges)
    tos = {m.edge: m.to_uuid for m in plan.moves}
    assert tos["old"] == tos["new"]
    assert plan.stats["pr #"]["linked pair: kept together"] == 1
    assert plan.candidates == []  # same identifier on both sides: a real update


def test_disjoint_supersession_is_a_candidate():
    ents = [_ent("m", "PR #290"), _ent("x", "widget")]
    edges = [
        _edge("old", "m", "x", "PR #205 is live on prod", live=False, by="new"),
        _edge("new", "m", "x", "PR #218 was merged and deployed", created="2026-01-02"),
    ]
    plan = rm.plan_repair(ents, edges)
    assert [(c.old, c.new) for c in plan.candidates] == [("old", "new")]
    assert plan.stats["pr #"]["linked pair: separated"] == 1


def test_supersession_is_no_candidate_when_its_id_has_another_live_fact_on_the_pair():
    """'No other valid invalidation': another live fact about PR #205 on the pair the
    retired one ends up on is a plausible real successor, so the retirement stands."""
    ents = [_ent("m", "PR #290"), _ent("x", "widget")]
    edges = [
        _edge("old", "m", "x", "PR #205 is live on prod", live=False, by="new"),
        _edge("new", "m", "x", "PR #218 was merged", created="2026-01-02"),
        _edge("later", "m", "x", "PR #205 was rolled back", created="2026-01-03"),
    ]
    plan = rm.plan_repair(ents, edges)
    assert plan.candidates == []
    key = "  ...kept: another live edge on its pair names its identifier"
    assert plan.stats["pr #"][key] == 1


@pytest.mark.parametrize(
    ("old_fact", "new_fact", "expired", "shared_is_id"),
    [
        ("PR #205 is live", "the widget was deployed", False, True),  # one side names none
        ("PR #205 is live", "PR #205 was rolled back", False, True),  # shared id
        ("PR #205 is live", "PR #218 was merged", True, True),  # t_expired: forgotten on purpose
        ("PR #205 is live", "PR #218 was merged", False, False),  # not on an ID entity
    ],
)
def test_supersessions_that_are_not_candidates(old_fact, new_fact, expired, shared_is_id):
    a = _ent("m", "PR #290") if shared_is_id else _ent("m", "deploy pipeline", etype="tool")
    ents = [a, _ent("x", "widget")]
    edges = [
        _edge("old", "m", "x", old_fact, live=False, by="new", expired=expired),
        _edge("new", "m", "x", new_fact, created="2026-01-02"),
    ]
    assert rm.plan_repair(ents, edges).candidates == []


def test_entities_that_lose_nothing_keep_their_summary():
    ents = [_ent("m", "PR #290", summary="about PR #290"), _ent("x", "widget")]
    edges = [_edge("e", "m", "x", "PR #290 shipped")]
    plan = rm.plan_repair(ents, edges)
    assert plan.moves == [] and plan.clear_summaries == []


def test_plan_is_deterministic_apart_from_new_uuids():
    ents, edges = _world()
    a, b = rm.plan_repair(ents, edges), rm.plan_repair(ents, list(reversed(edges)))
    assert [(m.edge, m.side, m.from_uuid) for m in a.moves] == [
        (m.edge, m.side, m.from_uuid) for m in b.moves
    ]
    assert sorted(t.name for t in a.targets.values()) == sorted(t.name for t in b.targets.values())


def test_private_outputs_are_refused_inside_the_repository(tmp_path):
    with pytest.raises(SystemExit):
        rm._outside_repo(str(Path(rm.__file__).resolve().parent / "plan.json"))
    assert rm._outside_repo(str(tmp_path / "plan.json")) == (tmp_path / "plan.json").resolve()
