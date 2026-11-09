"""The 1-hop leg, against a real table: each seed contributes its 8 edges nearest the query.

test_kg_hop_order.py records the statement from a scripted cursor; this file runs it. One
supernode seed carries more live edges than the per-seed cut, laid out so the answer is
not the old one: the nearest edges are the OLDEST, an edge with no embedding is the NEWEST,
and every embedded edge sits at a strictly different distance from the query (the cos/sin
construction of test_kg_both_groups, so no ties and one possible order). The hop rows are
captured from the statement itself rather than read off the fused facts, because the vector
leg returns the same edges and would blur what the hop leg picked.

  * full trust: exactly the 8 nearest, in distance order; the unembedded edge is pushed out
  * a restricted caller: the nearest edge has out-of-allowlist provenance, so it is absent
    and the next 8 fill the slots (the scope filter runs before the cut, not after)
  * an unembedded edge stays eligible: with fewer than 8 embedded edges it fills a slot
"""

from __future__ import annotations

import math
import uuid
from typing import Any

import psycopg
import pytest

from mcp_server import kg_pg
from tests.helpers.embed import GROUP, onehot

WORK = "work-proj"
ELSEWHERE = "elsewhere-proj"
SEED = "supernode"
HOP_MARK = "src_uuid = %s OR tgt_uuid = %s"
CUT = 8


@pytest.fixture()
def kg(conn):
    """A clean episodes + KG slate. KG first: truncating episodes fires 056's truncate
    trigger, which only has work to do while facts still exist."""

    def _wipe() -> None:
        conn.execute("TRUNCATE kg_relationships, kg_entities RESTART IDENTITY CASCADE")
        conn.execute("TRUNCATE episodes RESTART IDENTITY CASCADE")

    _wipe()
    yield conn
    _wipe()


def _ep(conn, project: str) -> int:
    """Episodes go in before the facts: schema 056 computes source_projects from them at
    the fact's insert, and a fact written first has none (NULL, never served restricted)."""
    return conn.execute(
        "INSERT INTO episodes (session_id, sequence, project, content) "
        "VALUES (%s, 1, %s, 'a turn') RETURNING id",
        (f"hop-{uuid.uuid4().hex[:10]}", project),
    ).fetchone()[0]


def _seed_entity(conn) -> None:
    """The entity step 3 picks: its embedding is the query, and it has live edges."""
    conn.execute(
        "INSERT INTO kg_entities (uuid, owner_id, group_id, name, normalized_name, summary, "
        "  embedding) VALUES (%s, 'default', %s, 'Supernode', 'supernode', 's', %s::vector)",
        (SEED, GROUP, kg_pg._vec_literal(onehot(0))),
    )


def _edge(
    conn,
    uid: str,
    ep: int,
    *,
    rank: int | None,
    day: int,
    seed_is_src: bool,
    audience: str | None = "work-safe",
) -> None:
    """One live edge of the seed. Its distance from onehot(0) grows strictly with ``rank``
    (``None``: no embedding). ``day`` is its valid-from, so recency can be set against
    distance. The seed is the edge's source or its target (the hop is undirected).
    ``audience`` (schema 066) is work-safe, so only provenance decides a restricted read."""
    vec = None
    if rank is not None:
        v = [0.0] * len(onehot(0))
        v[0], v[1] = math.cos(rank * 0.02), math.sin(rank * 0.02)
        vec = kg_pg._vec_literal(v)
    src, tgt = (SEED, f"far-{uid}") if seed_is_src else (f"far-{uid}", SEED)
    conn.execute(
        "INSERT INTO kg_relationships (uuid, owner_id, group_id, src_uuid, tgt_uuid, name, "
        "  fact, fact_embedding, episodes, t_valid, audience) "
        "VALUES (%s, 'default', %s, %s, %s, 'RELATES', %s, %s::vector, %s::jsonb, "
        "        '2026-01-01T00:00:00+00:00'::timestamptz + make_interval(days => %s), %s)",
        (uid, GROUP, src, tgt, f"fact {uid}", vec, f"[{ep}]", day, audience),
    )


def _supernode(conn, first_ep: int, ep: int, *, embedded: int = 12) -> list[str]:
    """``embedded`` edges, rank 0 nearest, named e00.. by rank and valid from day ``rank``
    (so the nearest is the oldest), then ``e-null``: no embedding, valid last of all. The
    old ``ORDER BY t_valid DESC LIMIT 8`` would have served e-null and the 7 farthest
    ranks. ``first_ep`` is the nearest edge's episode, ``ep`` every other edge's. Returns
    the uuids of the embedded edges by rank."""
    _seed_entity(conn)
    ids = [f"e{rank:02d}" for rank in range(embedded)]
    for rank, uid in enumerate(ids):
        _edge(
            conn,
            uid,
            first_ep if rank == 0 else ep,
            rank=rank,
            day=rank,
            seed_is_src=rank % 2 == 0,
        )
    _edge(conn, "e-null", ep, rank=None, day=100, seed_is_src=True)
    return ids


def _hop_rows(db_url: str, allowed_projects: list[str] | None) -> list[list[str]]:
    """Run step 1-5 on a real cursor and return, per hop statement, the uuids it returned
    in the order it returned them."""
    hops: list[list[str]] = []

    class _Rec:
        def __init__(self, cur: Any) -> None:
            self._cur = cur
            self._is_hop = False

        def execute(self, sql: str, params: Any = None) -> _Rec:
            self._is_hop = HOP_MARK in sql
            self._cur.execute(sql, params)
            return self

        def fetchall(self) -> list[tuple[Any, ...]]:
            rows = self._cur.fetchall()
            if self._is_hop:
                hops.append([r[0] for r in rows])
            return rows

    with psycopg.connect(db_url) as c, c.cursor() as cur:
        kg_pg.search_kg_postgres(
            _Rec(cur), "q", onehot(0), "default", GROUP, [], 12, allowed_projects
        )
    return hops


def test_the_hop_leg_returns_the_eight_nearest_edges_in_distance_order(kg, db_url):
    ids = _supernode(kg, _ep(kg, WORK), _ep(kg, WORK))
    live = kg.execute(
        "SELECT count(*) FROM kg_relationships WHERE t_invalid IS NULL AND "
        "(src_uuid = %s OR tgt_uuid = %s)",
        (SEED, SEED),
    ).fetchone()[0]
    assert live == len(ids) + 1
    assert live > CUT + 1
    # One seed, one statement. The nearest edges are the oldest, so recency would have
    # served e-null first and the 7 farthest ranks after it; distance serves e00..e07.
    assert _hop_rows(db_url, None) == [ids[:CUT]]


def test_a_restricted_caller_loses_an_out_of_allowlist_edge_before_the_cut(kg, db_url):
    """The nearest edge's only source is another project's episode: it is filtered inside
    the fence, so the next 8 take the slots (a filter after the cut would return 7)."""
    ids = _supernode(kg, _ep(kg, ELSEWHERE), _ep(kg, WORK))
    precondition = kg.execute(
        "SELECT uuid, source_projects FROM kg_relationships WHERE uuid IN (%s, %s, %s)",
        (ids[0], ids[1], "e-null"),
    ).fetchall()
    assert dict(precondition) == {ids[0]: [ELSEWHERE], ids[1]: [WORK], "e-null": [WORK]}
    (rows,) = _hop_rows(db_url, [WORK])
    assert ids[0] not in rows
    assert rows == ids[1 : CUT + 1]
    assert "e-null" not in rows


def test_an_unembedded_edge_stays_eligible_and_ranks_after_the_embedded_ones(kg, db_url):
    """Three embedded edges and two without: the embedded three in distance order, then the
    unembedded ones newest first. Nothing filters the NULL embedding out of the candidates."""
    ep = _ep(kg, WORK)
    ids = _supernode(kg, ep, ep, embedded=3)
    _edge(kg, "e-null-old", ep, rank=None, day=50, seed_is_src=False)
    assert _hop_rows(db_url, None) == [[*ids, "e-null", "e-null-old"]]
