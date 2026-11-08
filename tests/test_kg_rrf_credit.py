"""A uuid is credited once per ranked leg of the KG fact fusion, at its best rank.

The 1-hop leg returns a fact once per seed it touches, so an edge between two seeds
appeared twice in the hop list and ``_rrf_fuse`` credited both occurrences: one fact at
hop ranks 4 and 6 outweighed a fact that led the vector leg. The vector and BM25 legs are
one row per fact (``kg_rel_uuid`` is unique), so they cannot repeat a uuid; the fusion
itself skips a repeat, which covers every leg the same way. No DB: the search runs against
a scripted cursor.
"""

from __future__ import annotations

import math
from typing import Any

from mcp_server import kg_pg

K = kg_pg._RRF_K


def _credit(rank: int) -> float:
    return 1.0 / (K + rank + 1)


def test_a_uuid_repeated_in_one_leg_is_credited_once_at_its_first_rank():
    fused = kg_pg._rrf_fuse([["a", "b", "c", "b", "d", "b"]])
    assert math.isclose(fused["b"], _credit(1))
    # a skipped repeat leaves every other uuid at the rank it had
    assert math.isclose(fused["c"], _credit(2))
    assert math.isclose(fused["d"], _credit(4))


def test_a_uuid_in_two_legs_is_still_credited_by_each():
    fused = kg_pg._rrf_fuse([["a", "b"], ["b", "a", "b"]])
    assert math.isclose(fused["a"], _credit(0) + _credit(1))
    assert math.isclose(fused["b"], _credit(1) + _credit(0))


def test_legs_without_repeats_fuse_as_before():
    fused = kg_pg._rrf_fuse([["a", "b"], [], ["c", "a"]])
    assert math.isclose(fused["a"], _credit(0) + _credit(1))
    assert math.isclose(fused["b"], _credit(1))
    assert math.isclose(fused["c"], _credit(0))
    assert set(fused) == {"a", "b", "c"}


class _ScriptedCursor:
    """Answers each statement of ``search_kg_postgres`` from canned rows, by what it reads."""

    def __init__(
        self,
        vec: list[tuple[Any, ...]],
        seeds: list[tuple[Any, ...]],
        hops: dict[str, list[tuple[Any, ...]]],
    ) -> None:
        self._vec, self._seeds, self._hops = vec, seeds, hops
        self._rows: list[tuple[Any, ...]] = []

    def execute(self, sql: str, params: Any = None) -> _ScriptedCursor:
        if "FROM kg_relationships" in sql and "fact_embedding" in sql:
            self._rows = self._vec
        elif "paradedb" in sql:
            self._rows = []
        elif "WITH seeds" in sql:
            self._rows = self._seeds
        elif "src_uuid = %s OR tgt_uuid = %s" in sql:
            self._rows = self._hops[params[2]]
        else:  # pragma: no cover - a statement the scripted legs do not cover
            raise AssertionError(sql)
        return self

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._rows)


def _row(uuid: str) -> tuple[Any, ...]:
    return (uuid, f"fact {uuid}", None, False, None)


def test_a_fact_between_two_seeds_gets_the_hop_credit_of_one_occurrence():
    # "link" joins both seeds, so each seed's hop list names it: hop ranks 2 and 4.
    # Credited twice it beats "lead", the only vector hit (1/61 against 1/62 + 1/64);
    # credited once it ranks below it.
    cur = _ScriptedCursor(
        vec=[_row("lead")],
        seeds=[("s1", "One", None, 0.1, 5), ("s2", "Two", None, 0.2, 5)],
        hops={
            "s1": [_row("x1"), _row("link")],
            "s2": [_row("x2"), _row("link")],
        },
    )
    facts, seeds = kg_pg.search_kg_postgres(cur, "", [0.0], "owner", "grp", [], limit=1)
    assert [s["uuid"] for s in seeds] == ["s1", "s2"]
    assert [f["_uuid"] for f in facts] == ["lead"]


def test_the_fact_between_two_seeds_still_ranks_by_its_best_hop_rank():
    # no vector hit to beat: "link" at hop rank 1 (via s1) leads the hop-only facts
    cur = _ScriptedCursor(
        vec=[],
        seeds=[("s1", "One", None, 0.1, 5), ("s2", "Two", None, 0.2, 5)],
        hops={
            "s1": [_row("link"), _row("x1")],
            "s2": [_row("x2"), _row("link")],
        },
    )
    facts, _ = kg_pg.search_kg_postgres(cur, "", [0.0], "owner", "grp", [], limit=4)
    assert [f["_uuid"] for f in facts] == ["link", "x1", "x2"]
