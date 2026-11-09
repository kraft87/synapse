"""The statements of the fact-audience rule and the owner-seed exclusion (schema 066).

No DB: every statement is recorded from a scripted cursor. What each statement does
against a real table is in test_kg_fact_audience_db.py. Pinned here:

  * the restricted predicate: the old provenance string byte for byte with the audience
    switched off, one bound parameter either way, and the alias on every column
  * every restricted leg carries the label rule, and its arguments do not change
  * full trust never names a 066 column, and its statements are unchanged
  * the seed statement is unchanged without a flagged owner, and binds the owner uuids
    right after the group pair with one
  * the cross-group helpers stay out of a restricted read
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from mcp_server import kg_pg
from tests.helpers.embed import GROUP, onehot

ALLOWED = ["work-proj"]

_PRE_066 = "source_projects IS NOT NULL AND cardinality(source_projects) > 0 AND source_projects <@ %s::text[]"


class _Recorder:
    """Records every statement with its parameters; answers with no rows."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def execute(self, sql: str, params: Any = None) -> _Recorder:
        self.calls.append((sql, params))
        return self

    def fetchall(self) -> list[Any]:
        return []

    @property
    def sql(self) -> list[str]:
        return [s for s, _ in self.calls]


def _search(cur: _Recorder, *, allowed: list[str] | None = None, **kw: Any) -> None:
    kg_pg.search_kg_postgres(cur, "widget q", onehot(0), "default", GROUP, [], 5, allowed, **kw)


def _seed_statement(cur: _Recorder) -> tuple[str, Any]:
    (call,) = [c for c in cur.calls if c[0].startswith("WITH seeds")]
    return call


# ---------------------------------------------------------------------------
# scope_predicate
# ---------------------------------------------------------------------------


def test_the_pre_066_predicate_is_the_provenance_rule_byte_for_byte():
    assert kg_pg.scope_predicate(fact_audience=False) == _PRE_066
    assert kg_pg.scope_predicate("o", fact_audience=False) == _PRE_066.replace(
        "source_projects", "o.source_projects"
    )


def test_the_default_predicate_adds_the_label_rule_to_the_provenance_rule():
    pred = kg_pg.scope_predicate()
    assert pred.startswith(_PRE_066 + " AND (audience = 'work-safe' OR (audience IS NULL AND ")
    assert "synapse_meta" in pred and kg_pg.FACT_AUDIENCE_CUTOVER_KEY in pred
    assert pred.count("%s") == 1, "the allowlist stays the only bound parameter"


@pytest.mark.parametrize("alias", ["", "o", "n"])
def test_every_column_of_the_predicate_carries_the_alias(alias):
    pred = kg_pg.scope_predicate(alias)
    prefix = f"{alias}." if alias else ""
    for col in ("audience", "source_projects", "id"):
        assert f"{prefix}{col}" in pred
    if alias:
        # no bare reference to the outer row's columns: inside the join, an unqualified
        # ``audience`` or ``id`` would be ambiguous or, worse, resolve to the other side
        assert not re.search(r"(?<![\w.])(audience|id|source_projects)\b", pred)


def test_the_cutover_is_read_by_a_subquery_not_a_bound_value():
    pred = kg_pg.scope_predicate("a")
    assert "(SELECT value::bigint FROM synapse_meta WHERE key = " in pred
    # unqualified inside the subquery on purpose: its own FROM is the only table in scope
    assert "a.value" not in pred and "a.key" not in pred


# ---------------------------------------------------------------------------
# The restricted read
# ---------------------------------------------------------------------------


def test_every_restricted_statement_carries_the_label_rule():
    cur = _Recorder()
    _search(cur, allowed=ALLOWED)
    # vector, BM25, seed/degree (the recorder returns no seeds, so there is no hop query)
    assert len(cur.sql) == 3
    for sql in cur.sql:
        assert "audience = 'work-safe'" in sql and "synapse_meta" in sql, sql
        assert "source_projects <@ %s::text[]" in sql
    # the degree gate counts both arms of its UNION, each filtered
    assert cur.sql[2].count("audience = 'work-safe'") == 2


def test_the_arguments_do_not_change_with_the_label_rule():
    on, off = _Recorder(), _Recorder()
    _search(on, allowed=ALLOWED)
    _search(off, allowed=ALLOWED, fact_audience=False)
    assert [p for _, p in on.calls] == [p for _, p in off.calls]
    for sql, params in on.calls:
        assert sql.count("%s") == len(params)
    # and the statements differ only by the predicate
    for (a, _), (b, _) in zip(on.calls, off.calls, strict=True):
        assert "audience" not in b and "synapse_meta" not in b
        assert a != b


def test_a_restricted_call_without_the_label_rule_sends_the_056_statements():
    cur = _Recorder()
    _search(cur, allowed=ALLOWED, fact_audience=False)
    for sql in cur.sql:
        assert _PRE_066 in sql
        assert "audience" not in sql and "synapse_meta" not in sql


# ---------------------------------------------------------------------------
# Full trust
# ---------------------------------------------------------------------------


def test_full_trust_statements_never_mention_schema_066():
    cur = _Recorder()
    _search(cur)
    kg_pg.search_kg(cur, "widget q", onehot(0), "default", [GROUP, "personal"], [], 5)
    assert len(cur.sql) >= 6
    for sql in cur.sql:
        for word in ("audience", "synapse_meta", "is_self", "pg_attribute"):
            assert word not in sql, (word, sql)


def test_full_trust_ignores_the_label_rule_argument():
    a, b = _Recorder(), _Recorder()
    _search(a)
    _search(b, fact_audience=False)
    assert a.calls == b.calls


# ---------------------------------------------------------------------------
# The owner exclusion
# ---------------------------------------------------------------------------


def test_without_a_flagged_owner_the_seed_statement_is_the_one_it_always_was():
    for allowed in (None, ALLOWED):
        plain, empty = _Recorder(), _Recorder()
        _search(plain, allowed=allowed)
        _search(empty, allowed=allowed, exclude_seeds=())
        assert plain.calls == empty.calls
        sql, params = _seed_statement(plain)
        assert ") e WHERE owner_id = %s AND group_id = %s   ORDER BY dist LIMIT 25" in sql
        assert "ALL(" not in sql
        assert params[3:5] == ("default", GROUP)


@pytest.mark.parametrize("allowed", [None, ALLOWED])
def test_a_flagged_owner_is_filtered_out_of_the_seed_candidates(allowed):
    cur = _Recorder()
    _search(cur, allowed=allowed, exclude_seeds=["own-1", "own-2"])
    sql, params = _seed_statement(cur)
    assert (
        ") e WHERE owner_id = %s AND group_id = %s   AND uuid <> ALL(%s::text[])   ORDER BY" in sql
    )
    # bound right after the first owner/group pair, as a list
    assert params[3:6] == ("default", GROUP, ["own-1", "own-2"])
    assert sql.count("%s") == len(params)
    # no other statement is touched: the owner's facts are still served by the vector and
    # BM25 legs, and the hop leg only ever sees the seeds this statement returns
    plain = _Recorder()
    _search(plain, allowed=allowed)
    others = [c for c in cur.calls if not c[0].startswith("WITH seeds")]
    assert others == [c for c in plain.calls if not c[0].startswith("WITH seeds")]


def test_search_kg_hands_both_arguments_to_every_group_read(monkeypatch):
    seen: list[dict[str, Any]] = []

    def _fake(cur, *a, **kw):
        seen.append({"fact_audience": kw["fact_audience"], "exclude_seeds": kw["exclude_seeds"]})
        return [], []

    monkeypatch.setattr(kg_pg, "search_kg_postgres", _fake)
    monkeypatch.setattr(kg_pg, "fact_vector_walk", lambda *a, **k: [])
    monkeypatch.setattr(kg_pg, "cross_group_facts", lambda *a, **k: [])
    kw: dict[str, Any] = {"fact_audience": False, "exclude_seeds": ("own",)}
    kg_pg.search_kg(None, "q", onehot(0), "default", GROUP, [], 5, **kw)
    kg_pg.search_kg(None, "q", onehot(0), "default", [GROUP, "personal"], [], 5, **kw)
    kg_pg.search_kg(None, "q", onehot(0), "default", [GROUP], [], 5, ALLOWED, **kw)
    assert seen == [kw, kw, kw]


def test_a_restricted_read_never_reaches_the_cross_group_helpers(monkeypatch):
    """The cross-group extras take neither 066 argument, which is safe only because a
    restricted caller never gets there: it returns after its own group's read. Pin it, so
    a future change that lets one through fails here and not on a work laptop."""

    def _boom(*a, **k):
        raise AssertionError("cross-group helper reached by a restricted read")

    for name in ("fact_vector_walk", "split_walk", "cross_vec_rows", "cross_bm25_rows"):
        monkeypatch.setattr(kg_pg, name, _boom)
    monkeypatch.setattr(kg_pg, "cross_group_facts", _boom)
    cur = _Recorder()
    # a list is what a full-trust caller gets from read_groups; a restricted caller gets a
    # string, and even handed a list the allowlist makes search_kg stop at its own group
    kg_pg.search_kg(cur, "q", onehot(0), "default", GROUP, [], 5, ALLOWED)
    kg_pg.search_kg(cur, "q", onehot(0), "default", [GROUP, "personal"], [], 5, ALLOWED)
    assert kg_pg.read_groups(GROUP, True) == GROUP
