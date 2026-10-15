"""The KG leg's graph hop: one statement for all seeds, the same facts as one query per seed.

Seeds are the entities nearest the query that still have a live edge; each contributes its
8 most recently valid live edges in either direction. The seed degree gate only needs to
know whether an edge exists, not how many.
"""

from __future__ import annotations

from typing import Any

import pytest

from ingestion.vector import vec_literal
from mcp_server import kg_pg
from tests.helpers.embed import GROUP, onehot

pytestmark = pytest.mark.db

# A query token no fact says, so the BM25 leg is empty; facts carry no fact embedding, so the
# vector leg is empty too. What comes back is the hop leg alone.
_QUERY = "zzqqxx"


@pytest.fixture()
def graph(conn):
    conn.execute("TRUNCATE kg_relationships, kg_entities RESTART IDENTITY CASCADE")
    for uid, axis in (("hub", 0), ("small", 1), ("lonely", 2)):
        conn.execute(
            "INSERT INTO kg_entities (uuid, owner_id, group_id, name, normalized_name, embedding) "
            "VALUES (%s, 'default', %s, %s, %s, %s::vector)",
            (uid, GROUP, uid, uid, vec_literal(onehot(axis))),
        )
    yield conn
    conn.execute("TRUNCATE kg_relationships, kg_entities RESTART IDENTITY CASCADE")


def _edge(conn, uid: str, src: str, tgt: str, day: int, *, live: bool = True) -> None:
    conn.execute(
        "INSERT INTO kg_relationships (uuid, owner_id, group_id, src_uuid, tgt_uuid, name, fact, "
        "  t_valid, t_invalid) VALUES (%s, 'default', %s, %s, %s, 'RELATES', %s, "
        "  '2026-01-01T00:00:00+00:00'::timestamptz + make_interval(days => %s), %s)",
        (uid, GROUP, src, tgt, f"fact {uid}", day, None if live else "2026-06-01T00:00:00+00:00"),
    )


class _CountingCursor:
    def __init__(self, cur: Any) -> None:
        self._cur = cur
        self.statements = 0

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        self.statements += 1
        return self._cur.execute(*args, **kwargs)

    def fetchall(self) -> Any:
        return self._cur.fetchall()


def _hops(conn, seed_axis: int = 0, limit: int = 60) -> tuple[list[str], int]:
    cur = _CountingCursor(conn.cursor())
    facts, _ = kg_pg.search_kg_postgres(cur, _QUERY, onehot(seed_axis), "default", GROUP, [], limit)
    return [f["_uuid"] for f in facts], cur.statements


def test_each_seed_contributes_its_eight_most_recent_live_edges(graph):
    for d in range(12):  # hub as source: days 0..11
        _edge(graph, f"hub-out-{d}", "hub", f"x{d}", d)
    for d in range(12, 15):  # hub as target: newer than any outgoing edge
        _edge(graph, f"hub-in-{d}", f"y{d}", "hub", d)
    _edge(graph, "hub-dead", "hub", "z", 99, live=False)  # retired: never a hop
    for d in range(3):
        _edge(graph, f"small-{d}", "small", f"w{d}", d)
    got, _ = _hops(graph)
    hub = {u for u in got if u.startswith("hub")}
    # 3 incoming (days 12-14) + the 5 newest outgoing (days 7-11)
    assert hub == {f"hub-in-{d}" for d in (12, 13, 14)} | {f"hub-out-{d}" for d in range(7, 12)}
    assert {u for u in got if u.startswith("small")} == {"small-0", "small-1", "small-2"}


def test_a_self_loop_is_one_hop_not_two(graph):
    _edge(graph, "loop", "hub", "hub", 5)
    _edge(graph, "out", "hub", "x", 1)
    got, _ = _hops(graph)
    assert sorted(got) == ["loop", "out"]


def test_an_edge_between_two_seeds_counts_for_each_seed(graph):
    # The RRF fuse rewards a fact reached from both seeds; the single statement keeps the
    # per-seed lists, so the edge still appears under each.
    _edge(graph, "bridge", "hub", "small", 3)
    _edge(graph, "hub-only", "hub", "x", 1)
    _edge(graph, "small-only", "small", "y", 2)
    got, _ = _hops(graph)
    assert set(got) == {"bridge", "hub-only", "small-only"}
    assert got[0] == "bridge"  # fused first: two ranked appearances


def test_seed_without_a_live_edge_is_not_a_seed(graph):
    _edge(graph, "gone", "lonely", "x", 1, live=False)
    _edge(graph, "hub-edge", "hub", "x", 2)
    got, _ = _hops(graph, seed_axis=2)  # "lonely" is the nearest entity but has no live edge
    assert got == ["hub-edge"]


def test_the_leg_costs_four_statements_however_many_seeds(graph):
    for d in range(6):
        _edge(graph, f"hub-{d}", "hub", f"x{d}", d)
        _edge(graph, f"small-{d}", "small", f"y{d}", d)
    _, statements = _hops(graph)
    assert statements == 4  # vector, BM25, seeds, hops
