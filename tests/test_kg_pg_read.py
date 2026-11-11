"""DB-backed tests for ingestion/kg_pg_read.py (#67 PR 2 write-side read port).

Exercises every reader against the real kg_entities / kg_relationships schema
(017) on the shared test DB, asserting the FalkorDB-mirroring contracts the
callers depend on: valid_at as ISO string, fact_embedding as list[float],
vector score as cosine distance, live-edge (t_invalid IS NULL) filtering, and
owner scoping (a row still carrying the old 'personal' group is the owner's
memory like any other).
"""

from __future__ import annotations

import pytest

from ingestion import dedup
from ingestion.kg_pg_read import KGPostgresReader, _emb_list, _iso
from tests.helpers.embed import GROUP
from tests.helpers.kg import DIM, _axis_list


def _axis(i: int) -> str:
    """Unit vector along axis i as a pgvector literal (cosine-orthogonal set)."""
    v = [0.0] * DIM
    v[i] = 1.0
    return "[" + ",".join(map(str, v)) + "]"


@pytest.fixture()
def kg_tables(conn, monkeypatch, db_url):
    """Clean KG tables, point the reader at the test DB, seed a small graph."""
    monkeypatch.setenv("SYNAPSE_DB_URL", db_url)
    conn.execute("TRUNCATE kg_entities, kg_relationships RESTART IDENTITY CASCADE")

    conn.execute(
        "INSERT INTO kg_entities (uuid, owner_id, group_id, name, normalized_name, "
        "summary, embedding) VALUES "
        "('e-syn', 'default', %(g)s, 'Synapse', 'synapse', 'memory layer', %(v0)s), "
        "('e-fdb', 'default', %(g)s, 'FalkorDB', 'falkordb', 'graph db', %(v1)s), "
        # pre-migration node: normalized_name missing, name needs lower(trim())
        "('e-old', 'default', %(g)s, '  Voyage AI ', NULL, NULL, NULL), "
        # another tenant: must never surface
        "('e-other-owner', 'tenant2', %(g)s, 'Synapse', 'synapse', NULL, %(v0)s), "
        # left in the old personal group: still this owner's memory
        "('e-legacy', 'default', 'personal', 'Lantern', 'lantern', 'night light', %(v2)s)",
        {"g": GROUP, "v0": _axis(0), "v1": _axis(1), "v2": _axis(2)},
    )
    conn.execute(
        "INSERT INTO kg_relationships (uuid, owner_id, group_id, src_uuid, tgt_uuid, "
        "name, fact, fact_embedding, valid_at, t_valid, t_invalid) VALUES "
        "('r-live', 'default', %(g)s, 'e-syn', 'e-fdb', 'USES', "
        " 'Synapse uses FalkorDB as its graph layer', %(v0)s, "
        " '2026-06-01T00:00:00+00:00', '2026-06-01T00:00:00+00:00', NULL), "
        "('r-dead', 'default', %(g)s, 'e-syn', 'e-fdb', 'USES', "
        " 'Synapse uses OpenAI embeddings', %(v0)s, "
        " '2026-05-01T00:00:00+00:00', '2026-05-01T00:00:00+00:00', "
        " '2026-06-01T00:00:00+00:00'), "
        "('r-far', 'default', %(g)s, 'e-fdb', 'e-syn', 'RUNS_ON', "
        " 'FalkorDB runs on the docker VM', %(v1)s, NULL, NULL, NULL), "
        "('r-noemb', 'default', %(g)s, 'e-syn', 'e-fdb', 'USES', "
        " 'Synapse stores episodes in Postgres', NULL, NULL, NULL, NULL), "
        "('r-other', 'tenant2', %(g)s, 'e-syn', 'e-fdb', 'USES', "
        " 'tenant2 edge must not surface', %(v0)s, NULL, NULL, NULL), "
        "('r-legacy', 'default', 'personal', 'e-legacy', 'e-syn', 'LIGHTS', "
        " 'Lantern lights the harbor office', %(v2)s, NULL, NULL, NULL)",
        {"g": GROUP, "v0": _axis(0), "v1": _axis(1), "v2": _axis(2)},
    )
    yield KGPostgresReader()


class TestHelpers:
    def test_emb_list_parses_pgvector_text(self):
        assert _emb_list("[1,2.5,3]") == [1.0, 2.5, 3.0]
        assert _emb_list(None) is None
        assert _emb_list([1, 2]) == [1.0, 2.0]
        assert _emb_list("garbage") is None

    def test_iso(self):
        from datetime import UTC, datetime

        assert _iso(None) is None
        assert _iso(datetime(2026, 6, 1, tzinfo=UTC)) == "2026-06-01T00:00:00+00:00"


class TestEntityReads:
    def test_find_similar_nodes_orders_by_distance(self, kg_tables):
        hits = kg_tables.find_similar_nodes(_axis_list(0), limit=5)
        uuids = [h["uuid"] for h in hits]
        assert uuids[0] == "e-syn"  # identical vector, distance 0
        assert hits[0]["score"] == pytest.approx(0.0, abs=1e-6)
        assert hits[0]["name"] == "Synapse"
        # orthogonal vector still returned, ranked after
        assert "e-fdb" in uuids
        # a node left in the old personal group is found like any other
        assert "e-legacy" in uuids
        # another tenant never surfaces
        assert "e-other-owner" not in uuids

    def test_normalized_name_exact_and_fallback(self, kg_tables):
        assert kg_tables.entity_uuid_by_normalized_name("synapse") == "e-syn"
        # pre-migration node: lower(trim(name)) fallback path
        assert kg_tables.entity_uuid_by_normalized_name("voyage ai") == "e-old"
        assert kg_tables.entity_uuid_by_normalized_name("nonexistent") is None

    def test_a_node_in_either_legacy_group_is_found_by_name(self, kg_tables):
        assert kg_tables.entity_uuid_by_normalized_name("lantern") == "e-legacy"

    def test_load_entities_scoped_tuples(self, kg_tables):
        rows = kg_tables.load_entities()
        by_uuid = {u: (n, s) for u, n, s, _supertype in rows}
        assert set(by_uuid) == {"e-syn", "e-fdb", "e-old", "e-legacy"}
        assert by_uuid["e-syn"] == ("Synapse", "memory layer")
        assert by_uuid["e-old"] == ("  Voyage AI ", "")  # NULLs coalesced


class TestEdgeReads:
    def test_find_edges_by_pair_live_only_with_embedding_list(self, kg_tables):
        hits = kg_tables.find_edges_by_pair("e-syn", "e-fdb")
        uuids = {h["uuid"] for h in hits}
        assert "r-live" in uuids and "r-noemb" in uuids
        assert "r-dead" not in uuids  # invalidated edges must not re-enter the pool
        assert "r-other" not in uuids
        live = next(h for h in hits if h["uuid"] == "r-live")
        # contradiction detector calls list(cand_emb) and cosine() on this
        assert isinstance(live["fact_embedding"], list)
        assert len(live["fact_embedding"]) == DIM
        assert live["fact_embedding"][0] == 1.0
        assert live["valid_at"] == "2026-06-01T00:00:00+00:00"  # ISO string, not datetime
        noemb = next(h for h in hits if h["uuid"] == "r-noemb")
        assert noemb["fact_embedding"] is None

    def test_edges_left_in_the_old_personal_group_are_found(self, kg_tables):
        (pair,) = kg_tables.find_edges_by_pair("e-legacy", "e-syn")
        assert pair["uuid"] == "r-legacy"
        similar = kg_tables.find_similar_edges(_axis_list(2), distance_threshold=0.20)
        assert [h["uuid"] for h in similar] == ["r-legacy"]
        fulltext = kg_tables.find_edges_by_fulltext("harbor office")
        assert [h["uuid"] for h in fulltext] == ["r-legacy"]

    def test_find_similar_edges_threshold_gate(self, kg_tables):
        hits = kg_tables.find_similar_edges(_axis_list(0), distance_threshold=0.20)
        uuids = {h["uuid"] for h in hits}
        assert uuids == {"r-live"}  # r-dead invalidated, r-far orthogonal (dist 1.0)
        assert hits[0]["score"] == pytest.approx(0.0, abs=1e-6)
        # widening the threshold admits the orthogonal edge
        wide = kg_tables.find_similar_edges(_axis_list(0), distance_threshold=1.5)
        assert {h["uuid"] for h in wide} >= {"r-live", "r-far"}

    def test_fulltext_multiword_or_semantics(self, kg_tables):
        # The FalkorDB leg returned 0 on ALL multi-word queries (AND/phrase
        # semantics) — the ParadeDB leg matching here is the dead-leg fix.
        hits = kg_tables.find_edges_by_fulltext("FalkorDB graph layer")
        uuids = {h["uuid"] for h in hits}
        assert "r-live" in uuids
        assert "r-dead" not in uuids  # live-only
        assert "r-other" not in uuids
        assert all(isinstance(h["score"], float) for h in hits)

    def test_fulltext_reports_distance_to_a_supplied_embedding(self, kg_tables):
        # "Synapse ... FalkorDB" lexically matches r-live (axis 0), r-far (axis 1) and
        # r-noemb (no embedding); the distance is what tells the topical hit apart.
        hits = kg_tables.find_edges_by_fulltext(
            "Synapse FalkorDB Postgres", fact_embedding=_axis_list(0)
        )
        dist = {h["uuid"]: h["dist"] for h in hits}
        assert dist["r-live"] == pytest.approx(0.0, abs=1e-3)
        assert dist["r-far"] == pytest.approx(1.0, abs=1e-3)
        assert dist["r-noemb"] is None
        plain = kg_tables.find_edges_by_fulltext("Synapse FalkorDB Postgres")
        assert all(h["dist"] is None for h in plain)

    def test_fulltext_empty_query_short_circuits(self, kg_tables):
        assert kg_tables.find_edges_by_fulltext("") == []
        assert kg_tables.find_edges_by_fulltext("!!! ---") == []


def _tilted(cos: float) -> list[float]:
    """Unit vector at cosine ``cos`` to axis 0 (cosine distance 1 - cos)."""
    v = [0.0] * DIM
    v[0], v[1] = cos, (1.0 - cos * cos) ** 0.5
    return v


class TestNearestLiveEdges:
    """Stage 6c's batched lookup: per embedding, the nearest live in-scope edges inside
    the distance bound, nearest first, in one statement."""

    @pytest.fixture()
    def reader(self, conn, monkeypatch, db_url):
        monkeypatch.setenv("SYNAPSE_DB_URL", db_url)
        conn.execute("TRUNCATE kg_entities, kg_relationships RESTART IDENTITY CASCADE")
        rows = [
            # uuid, owner, group, vector, retired
            ("n-same", "default", GROUP, _tilted(1.0), False),
            ("n-close", "default", GROUP, _tilted(0.95), False),
            ("n-edge", "default", GROUP, _tilted(0.91), False),
            ("n-far", "default", GROUP, _tilted(0.80), False),
            ("n-retired", "default", GROUP, _tilted(1.0), True),
            ("n-legacy", "default", "personal", _tilted(0.97), False),
            ("n-other-owner", "tenant2", GROUP, _tilted(1.0), False),
            ("n-noemb", "default", GROUP, None, False),
        ]
        for uuid, owner, group, vec, retired in rows:
            conn.execute(
                "INSERT INTO kg_relationships (uuid, owner_id, group_id, src_uuid, tgt_uuid, "
                "name, fact, fact_embedding, t_invalid) VALUES "
                "(%s, %s, %s, 'e-s', 'e-t', 'R', %s, %s::vector, %s)",
                (
                    uuid,
                    owner,
                    group,
                    f"fact {uuid}",
                    None if vec is None else "[" + ",".join(map(str, vec)) + "]",
                    "2026-06-01T00:00:00+00:00" if retired else None,
                ),
            )
        return KGPostgresReader()

    def test_in_scope_live_neighbours_inside_the_bound_nearest_first(self, reader):
        (hits,) = reader.nearest_live_edges([_tilted(1.0)], max_distance=0.10, limit=5)
        # the edge left in the old personal group is as much this owner's as the others
        assert [h["uuid"] for h in hits] == ["n-same", "n-legacy", "n-close", "n-edge"]
        assert hits[0]["fact"] == "fact n-same"
        assert hits[0]["score"] == pytest.approx(0.0, abs=1e-3)
        assert hits[2]["score"] == pytest.approx(0.05, abs=1e-3)

    def test_one_result_list_per_embedding_in_input_order(self, reader):
        out = reader.nearest_live_edges(
            [_axis_list(7), _tilted(0.95), _tilted(1.0)], max_distance=0.10, limit=1
        )
        assert [[h["uuid"] for h in hits] for hits in out] == [[], ["n-close"], ["n-same"]]

    def test_empty_input_issues_no_query(self, reader):
        assert reader.nearest_live_edges([], 0.10, 5) == []


class TestNameResolutionParity:
    """A name that several rows hold resolves to one row, whichever way it is looked up.

    The SQL lookup (the live database) and the in-memory index (``load_entities`` through
    ``NodeDeduper``, built once per process) must pick the same entity, or a mention would
    resolve differently depending on whether the index has been built yet. The row is the
    owner's flagged entity, else the oldest, across the old personal and technical groups."""

    def _entity(self, conn, uuid, name, group, *, norm=None, is_self=False):
        conn.execute(
            "INSERT INTO kg_entities (uuid, owner_id, group_id, name, normalized_name, is_self) "
            "VALUES (%s, 'default', %s, %s, %s, %s)",
            (uuid, group, name, name.lower() if norm is None else norm, is_self),
        )

    def test_the_lookup_and_the_index_pick_the_same_row(self, conn, monkeypatch, db_url):
        monkeypatch.setenv("SYNAPSE_DB_URL", db_url)
        conn.execute("TRUNCATE kg_entities, kg_relationships RESTART IDENTITY CASCADE")
        self._entity(conn, "p-twin", "Twin", "personal")
        self._entity(conn, "t-twin", "Twin", GROUP)
        self._entity(conn, "t-flag", "Flag", GROUP)
        self._entity(conn, "p-flag", "Flag", "personal", is_self=True)
        self._entity(conn, "t-blank", "Omega", GROUP, norm="")  # written before the column
        self._entity(conn, "p-omega", "Omega", "personal")
        expected = {"twin": "p-twin", "flag": "p-flag", "omega": "t-blank"}

        reader = KGPostgresReader()
        index = dedup.NodeDeduper(reader)._build_index()

        assert {n: reader.entity_uuid_by_normalized_name(n) for n in expected} == expected
        assert {n: index.uuid_by_normalized_name[n] for n in expected} == expected
