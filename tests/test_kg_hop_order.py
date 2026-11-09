"""The 1-hop leg picks each seed's 8 edges by query distance, not by recency.

A supernode seed (the owner identity node, thousands of live edges) used to offer its 8
newest edges whatever the question was. The statement now computes the exact distance of
every live edge of the seed to the query inside a MATERIALIZED fence (defensive: no index
can serve this ORDER BY today, and the fence keeps a future one from cutting the filtered
set short) and keeps the 8 nearest. Edges with no embedding stay eligible and rank after
every embedded one. No DB: the statements are recorded from a scripted cursor; the
behaviour against a real table is in test_kg_hop_order_db.py.
"""

from __future__ import annotations

from typing import Any

from mcp_server import kg_pg

OWNER, GROUP = "owner", "grp"
QUERY_EMB = [0.25, 0.5]
SEEDS = [("s1", "One", None, 0.1, 5), ("s2", "Two", None, 0.2, 5)]


class _RecordingCursor:
    """Records every statement; the seed statement answers with ``SEEDS``, the rest with no rows."""

    def __init__(self) -> None:
        self.statements: list[tuple[str, Any]] = []
        self._rows: list[tuple[Any, ...]] = []

    def execute(self, sql: str, params: Any = None) -> _RecordingCursor:
        self.statements.append((sql, params))
        self._rows = SEEDS if "WITH seeds" in sql else []
        return self

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._rows)

    def hops(self) -> list[tuple[str, Any]]:
        return [(s, p) for s, p in self.statements if "src_uuid = %s OR tgt_uuid = %s" in s]


def _run(
    allowed_projects: list[str] | None = None, *, fact_audience: bool = True
) -> _RecordingCursor:
    cur = _RecordingCursor()
    kg_pg.search_kg_postgres(
        cur,
        "",
        QUERY_EMB,
        OWNER,
        GROUP,
        [],
        5,
        allowed_projects=allowed_projects,
        fact_audience=fact_audience,
    )
    return cur


def test_the_hop_statement_orders_by_distance_inside_a_materialized_fence():
    (sql, params), _ = _run().hops()
    assert sql.startswith("WITH cand AS MATERIALIZED (")
    # the stored vector, not the HNSW legs' halfvec cast, which converts every candidate row
    assert "fact_embedding <=> %s::vector AS dist" in sql
    assert "halfvec" not in sql
    # the filter lives inside the fence; the order and the cut sit outside it
    fence, outer = sql.split(") SELECT ", 1)
    for clause in (
        "owner_id = %s",
        "group_id = %s",
        "t_invalid IS NULL",
        "web_artifact_id IS NULL",
    ):
        assert clause in fence
    assert "(src_uuid = %s OR tgt_uuid = %s)" in fence
    assert outer.endswith("FROM cand ORDER BY dist ASC NULLS LAST, t_valid DESC LIMIT 8")
    # unembedded edges stay eligible: nothing filters fact_embedding IS NOT NULL
    assert "fact_embedding IS NOT NULL" not in sql
    assert sql.count("%s") == len(params)
    assert params == (kg_pg._vec_literal(QUERY_EMB), OWNER, GROUP, "s1", "s1")


def test_one_statement_per_seed_in_seed_order():
    hops = _run().hops()
    assert [p[3] for _, p in hops] == ["s1", "s2"]
    assert all(p[0] == kg_pg._vec_literal(QUERY_EMB) for _, p in hops)


def test_a_restricted_caller_filters_inside_the_fence_and_passes_the_allowlist():
    allowed = ["work-proj"]
    hops = _run(allowed).hops()
    assert len(hops) == 2
    for sql, params in hops:
        fence = sql.split(") SELECT ", 1)[0]
        assert kg_pg.scope_predicate() in fence
        assert sql.count("%s") == len(params)
        assert params[-1] == allowed
        assert params[:3] == (kg_pg._vec_literal(QUERY_EMB), OWNER, GROUP)
    assert [p[3] for _, p in hops] == ["s1", "s2"]


def test_the_hop_fence_carries_the_provenance_rule_alone_before_schema_066():
    hops = _run(["work-proj"], fact_audience=False).hops()
    assert len(hops) == 2
    for sql, params in hops:
        fence = sql.split(") SELECT ", 1)[0]
        assert kg_pg.scope_predicate(fact_audience=False) in fence
        assert "audience" not in sql
        assert sql.count("%s") == len(params)


def test_full_trust_hop_statement_carries_no_scope_predicate():
    (sql, params), _ = _run().hops()
    assert "source_projects" not in sql
    assert len(params) == 5
