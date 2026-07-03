"""DB-backed test for recall._surface_supersessions — the supersession fact leg.

A query that matches a now-INVALID fact should still return today's answer: the leg finds a
superseded edge near the query that carries a precise invalidated_by link (schema 028), resolves
the live successor, and surfaces THAT — deduped, distance-gated, go-forward only, never the stale
fact itself.
"""

from __future__ import annotations

from mcp_server.recall import Recall

GROUP = "technical"
DIM = 2048


def _axis(i: int) -> str:
    v = [0.0] * DIM
    v[i] = 1.0
    return "[" + ",".join(map(str, v)) + "]"


def _axis_list(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i] = 1.0
    return v


def _seed(conn) -> None:
    conn.execute("TRUNCATE kg_relationships RESTART IDENTITY CASCADE")
    conn.execute(
        "INSERT INTO kg_relationships (uuid, owner_id, group_id, src_uuid, tgt_uuid, name, fact, "
        "  fact_embedding, t_valid, t_invalid, invalidated_by) VALUES "
        # live successor N — query for axis(1) should NOT match it directly (it's on axis 0)
        "('n-1', 'default', %(g)s, 'a', 'b', 'USES', 'Synapse uses Postgres now', %(v0)s, "
        "  '2026-06-10T00:00:00+00:00', NULL, NULL), "
        # superseded predecessor P — on axis(1); links to n-1 via invalidated_by
        "('p-1', 'default', %(g)s, 'a', 'b', 'USES', 'Synapse uses FalkorDB', %(v1)s, "
        "  '2026-05-01T00:00:00+00:00', '2026-06-10T00:00:00+00:00', 'n-1')",
        {"g": GROUP, "v0": _axis(0), "v1": _axis(1)},
    )


def test_query_matching_invalid_fact_surfaces_successor(conn, db_url):
    _seed(conn)
    r = Recall(db_url, "")
    # query aligns with the SUPERSEDED fact P (axis 1) -> surface its successor N.
    out = r._surface_supersessions(_axis_list(1), GROUP, served_uuids=set())
    assert len(out) == 1
    assert out[0]["_uuid"] == "n-1"
    assert out[0]["fact"] == "Synapse uses Postgres now"  # the CURRENT fact, not the stale one
    assert str(out[0]["_date"]).startswith("2026-06-10")  # successor's t_valid


def test_already_served_successor_is_deduped(conn, db_url):
    _seed(conn)
    r = Recall(db_url, "")
    out = r._surface_supersessions(_axis_list(1), GROUP, served_uuids={"n-1"})
    assert out == []  # successor already in the facts bucket -> not added again


def test_off_topic_superseded_fact_is_gated_out(conn, db_url):
    _seed(conn)
    r = Recall(db_url, "")
    # query aligns with the SUCCESSOR (axis 0); the superseded P (axis 1) is cosine-distance 1.0
    # away -> beyond _SUP_MAX_DIST -> nothing surfaced (the live leg already owns this case).
    assert r._surface_supersessions(_axis_list(0), GROUP, served_uuids=set()) == []
