"""DB-backed test for recall._surface_supersessions — the supersession fact leg.

A query that matches a now-INVALID fact should still return today's answer: the leg finds a
superseded edge near the query that carries a precise invalidated_by link (schema 028), resolves
the live successor, and surfaces THAT — deduped, distance-gated, go-forward only, never the stale
fact itself.
"""

from __future__ import annotations

import uuid

from mcp_server.recall import Recall
from tests.helpers.embed import GROUP

DIM = 2048


def _axis(i: int) -> str:
    v = [0.0] * DIM
    v[i] = 1.0
    return "[" + ",".join(map(str, v)) + "]"


def _axis_list(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i] = 1.0
    return v


def _mix(i: int, j: int, wi: float) -> str:
    """Unit vector mostly along axis ``i`` (weight ``wi``), the rest along ``j``."""
    v = [0.0] * DIM
    v[i], v[j] = wi, (1.0 - wi * wi) ** 0.5
    return "[" + ",".join(map(str, v)) + "]"


def _seed(conn) -> None:
    conn.execute("TRUNCATE kg_relationships RESTART IDENTITY CASCADE")
    conn.execute(
        "INSERT INTO kg_relationships (uuid, owner_id, group_id, src_uuid, tgt_uuid, name, fact, "
        "  fact_embedding, t_valid, t_invalid, invalidated_by) VALUES "
        # live successor N: about the same thing as P (cosine 0.9) but not on the query's axis
        "('n-1', 'default', %(g)s, 'a', 'b', 'USES', 'Synapse uses Postgres now', %(vn)s, "
        "  '2026-06-10T00:00:00+00:00', NULL, NULL), "
        # superseded predecessor P — on axis(1); links to n-1 via invalidated_by
        "('p-1', 'default', %(g)s, 'a', 'b', 'USES', 'Synapse uses FalkorDB', %(v1)s, "
        "  '2026-05-01T00:00:00+00:00', '2026-06-10T00:00:00+00:00', 'n-1')",
        {"g": GROUP, "vn": _mix(1, 0, 0.9), "v1": _axis(1)},
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


def _seed_pairs(conn) -> None:
    """One served live edge A on (a,b) plus three retired edges on the same endpoints:
    linked to A with another predicate, unlinked with A's predicate, unlinked with a
    different predicate. Only the first two are genuine predecessors."""
    conn.execute("TRUNCATE kg_relationships RESTART IDENTITY CASCADE")
    conn.execute(
        "INSERT INTO kg_relationships (uuid, owner_id, group_id, src_uuid, tgt_uuid, name, fact, "
        "  t_valid, t_invalid, invalidated_by) VALUES "
        "('a', 'default', %(g)s, 'a', 'b', 'CORRECTED', 'User corrected the assistant on 10-08', "
        "  '2026-10-08T00:00:00+00:00', NULL, NULL), "
        "('linked', 'default', %(g)s, 'a', 'b', 'TOLD', 'User told the assistant X (linked)', "
        "  '2026-09-01T00:00:00+00:00', '2026-10-08T00:00:00+00:00', 'a'), "
        "('same-name', 'default', %(g)s, 'a', 'b', 'CORRECTED', 'User corrected the assistant on 09-08', "
        "  '2026-09-08T00:00:00+00:00', '2026-09-09T00:00:00+00:00', NULL), "
        "('unrelated', 'default', %(g)s, 'a', 'b', 'TOLD_TO_STOP', 'User told the assistant on 08-29 to stop X', "
        "  '2026-08-29T00:00:00+00:00', '2026-08-29T00:00:00+00:00', NULL)",
        {"g": GROUP},
    )


def test_superseded_pair_needs_a_link_or_the_same_predicate(conn, db_url):
    _seed_pairs(conn)
    r = Recall(db_url, "")
    out = r._fetch_superseded_pairs_pg(GROUP, ["a"], 10)
    # DISTINCT ON keeps the most recently retired genuine predecessor: the linked one.
    assert [x["id"] for x in out] == ["f:linked"]
    assert out[0]["superseded_by"] == "User corrected the assistant on 10-08"


def test_unrelated_retired_edge_on_the_same_endpoints_is_never_paired(conn, db_url):
    _seed_pairs(conn)
    conn.execute("DELETE FROM kg_relationships WHERE uuid IN ('linked', 'same-name')")
    r = Recall(db_url, "")
    assert r._fetch_superseded_pairs_pg(GROUP, ["a"], 10) == []


def test_same_predicate_without_a_link_still_pairs(conn, db_url):
    _seed_pairs(conn)
    conn.execute("DELETE FROM kg_relationships WHERE uuid = 'linked'")
    r = Recall(db_url, "")
    assert [x["id"] for x in r._fetch_superseded_pairs_pg(GROUP, ["a"], 10)] == ["f:same-name"]


def _seed_far_link(conn) -> dict[str, int]:
    """P (axis 1) retired by an UNRELATED successor (axis 5): a false contradiction verdict."""
    _seed(conn)
    eid = conn.execute(
        "INSERT INTO episodes (session_id, sequence, content) VALUES (%s, 1, 'x') RETURNING id",
        (f"far-link-{uuid.uuid4().hex[:10]}",),
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO kg_relationships (uuid, owner_id, group_id, src_uuid, tgt_uuid, name, fact, "
        "  fact_embedding, episodes, t_valid, t_invalid, invalidated_by) VALUES "
        "('n-far', 'default', %(g)s, 'c', 'd', 'R', 'An unrelated later fact', %(v5)s, NULL, "
        "  '2026-10-09T00:00:00+00:00', NULL, NULL), "
        "('p-far', 'default', %(g)s, 'c', 'd', 'R', 'A fact wrongly judged contradicted', %(v1)s, "
        "  %(eps)s::jsonb, '2026-09-01T00:00:00+00:00', '2026-10-09T00:00:00+00:00', 'n-far')",
        {"g": GROUP, "v1": _axis(1), "v5": _axis(5), "eps": f"[{eid}]"},
    )
    return {"episode": eid}


def test_far_successor_link_is_not_believed_by_any_leg(conn, db_url):
    ids = _seed_far_link(conn)
    r = Recall(db_url, "")
    surfaced = {
        x["_uuid"] for x in r._surface_supersessions(_axis_list(1), GROUP, served_uuids=set())
    }
    assert "n-far" not in surfaced and "n-1" in surfaced  # the close link still surfaces
    assert r._episode_supersessions([ids["episode"]], GROUP) == {}
    assert r._fetch_superseded_pairs_pg(GROUP, ["n-far"], 10) == []


def test_close_successor_link_still_overlays_its_episode(conn, db_url):
    _seed(conn)
    eid = conn.execute(
        "INSERT INTO episodes (session_id, sequence, content) VALUES (%s, 1, 'x') RETURNING id",
        (f"close-link-{uuid.uuid4().hex[:10]}",),
    ).fetchone()[0]
    conn.execute(
        "UPDATE kg_relationships SET episodes = %s::jsonb WHERE uuid = 'p-1'", (f"[{eid}]",)
    )
    r = Recall(db_url, "")
    assert r._episode_supersessions([eid], GROUP) == {eid: ["Synapse uses Postgres now"]}
    pairs = r._fetch_superseded_pairs_pg(GROUP, ["n-1"], 10)
    assert [x["id"] for x in pairs] == ["f:p-1"]


def test_overlay_never_serves_a_retired_successor(conn, db_url):
    _seed(conn)
    eid = conn.execute(
        "INSERT INTO episodes (session_id, sequence, content) VALUES (%s, 1, 'x') RETURNING id",
        (f"stale-succ-{uuid.uuid4().hex[:10]}",),
    ).fetchone()[0]
    conn.execute(
        "UPDATE kg_relationships SET episodes = %s::jsonb WHERE uuid = 'p-1'", (f"[{eid}]",)
    )
    conn.execute("UPDATE kg_relationships SET t_invalid = now() WHERE uuid = 'n-1'")
    assert Recall(db_url, "")._episode_supersessions([eid], GROUP) == {}


def test_overlay_needs_the_served_text_to_state_the_retired_claim(conn, db_url):
    """A chunk-sourced fact cites every episode of its chunk: the successor attaches only
    to a passage that actually states the retired claim, not to every cited episode."""
    _seed(conn)
    on, off = (
        conn.execute(
            "INSERT INTO episodes (session_id, sequence, content) VALUES (%s, 1, 'x') RETURNING id",
            (f"ovl-{uuid.uuid4().hex[:10]}",),
        ).fetchone()[0]
        for _ in range(2)
    )
    conn.execute(
        "UPDATE kg_relationships SET episodes = %s::jsonb WHERE uuid = 'p-1'", (f"[{on}, {off}]",)
    )
    served = {
        on: "[assistant] For now Synapse stores the graph in FalkorDB.",
        off: "[user] set up restic backups to the nas please",
    }
    r = Recall(db_url, "")
    assert r._episode_supersessions([on, off], GROUP, served_text=served) == {
        on: ["Synapse uses Postgres now"]
    }
    # Without served text (drill-down callers) the citation alone still links.
    assert set(r._episode_supersessions([on, off], GROUP)) == {on, off}


def test_claim_support_scoring():
    from mcp_server.recall_presentation import claim_support

    claim = "Kyle restarted vitamin D at 10,000 IU daily (meaning 2026-01-05)"
    assert claim_support(claim, "I restarted my vitamin d, 10000 iu daily now") == 1.0
    assert claim_support(claim, "restic backup schedule for the nas") == 0.0
    # A path in the claim matches when its parts appear in the text.
    assert claim_support("rreading-glasses in compose.yml", "/opt/docker/plex/compose.yml") == 0.5
    assert claim_support("the and of", "anything") == 1.0  # nothing to check against
