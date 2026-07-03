"""DB-backed tests for ingestion/kg_pg_read.py (#67 PR 2 write-side read port).

Exercises every reader against the real kg_entities / kg_relationships schema
(017) on the shared test DB, asserting the FalkorDB-mirroring contracts the
callers depend on: valid_at as ISO string, fact_embedding as list[float],
vector score as cosine distance, live-edge (t_invalid IS NULL) filtering, and
owner/group scoping.
"""

from __future__ import annotations

import pytest

from ingestion.kg_pg_read import KGPostgresReader, _emb_list, _iso

GROUP = "technical"
DIM = 2048


def _axis(i: int) -> str:
    """Unit vector along axis i as a pgvector literal (cosine-orthogonal set)."""
    v = [0.0] * DIM
    v[i] = 1.0
    return "[" + ",".join(map(str, v)) + "]"


def _axis_list(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i] = 1.0
    return v


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
        # other tenant / other group: must never surface
        "('e-other-owner', 'tenant2', %(g)s, 'Synapse', 'synapse', NULL, %(v0)s), "
        "('e-other-group', 'default', 'personal', 'Synapse', 'synapse', NULL, %(v0)s)",
        {"g": GROUP, "v0": _axis(0), "v1": _axis(1)},
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
        " 'tenant2 edge must not surface', %(v0)s, NULL, NULL, NULL)",
        {"g": GROUP, "v0": _axis(0), "v1": _axis(1)},
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
        hits = kg_tables.find_similar_nodes(_axis_list(0), GROUP, limit=5)
        uuids = [h["uuid"] for h in hits]
        assert uuids[0] == "e-syn"  # identical vector, distance 0
        assert hits[0]["score"] == pytest.approx(0.0, abs=1e-6)
        assert hits[0]["name"] == "Synapse"
        # orthogonal vector still returned, ranked after
        assert "e-fdb" in uuids
        # other tenant / group never surface
        assert "e-other-owner" not in uuids
        assert "e-other-group" not in uuids

    def test_normalized_name_exact_and_fallback(self, kg_tables):
        assert kg_tables.entity_uuid_by_normalized_name("synapse", GROUP) == "e-syn"
        # pre-migration node: lower(trim(name)) fallback path
        assert kg_tables.entity_uuid_by_normalized_name("voyage ai", GROUP) == "e-old"
        assert kg_tables.entity_uuid_by_normalized_name("nonexistent", GROUP) is None

    def test_load_entities_scoped_tuples(self, kg_tables):
        rows = kg_tables.load_entities(GROUP)
        by_uuid = {u: (n, s) for u, n, s, _supertype in rows}
        assert set(by_uuid) == {"e-syn", "e-fdb", "e-old"}
        assert by_uuid["e-syn"] == ("Synapse", "memory layer")
        assert by_uuid["e-old"] == ("  Voyage AI ", "")  # NULLs coalesced


class TestEdgeReads:
    def test_find_edges_by_pair_live_only_with_embedding_list(self, kg_tables):
        hits = kg_tables.find_edges_by_pair("e-syn", "e-fdb", GROUP)
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

    def test_find_similar_edges_threshold_gate(self, kg_tables):
        hits = kg_tables.find_similar_edges(_axis_list(0), GROUP, distance_threshold=0.20)
        uuids = {h["uuid"] for h in hits}
        assert uuids == {"r-live"}  # r-dead invalidated, r-far orthogonal (dist 1.0)
        assert hits[0]["score"] == pytest.approx(0.0, abs=1e-6)
        # widening the threshold admits the orthogonal edge
        wide = kg_tables.find_similar_edges(_axis_list(0), GROUP, distance_threshold=1.5)
        assert {h["uuid"] for h in wide} >= {"r-live", "r-far"}

    def test_fulltext_multiword_or_semantics(self, kg_tables):
        # The FalkorDB leg returned 0 on ALL multi-word queries (AND/phrase
        # semantics) — the ParadeDB leg matching here is the dead-leg fix.
        hits = kg_tables.find_edges_by_fulltext("FalkorDB graph layer", GROUP)
        uuids = {h["uuid"] for h in hits}
        assert "r-live" in uuids
        assert "r-dead" not in uuids  # live-only
        assert "r-other" not in uuids
        assert all(isinstance(h["score"], float) for h in hits)

    def test_fulltext_empty_query_short_circuits(self, kg_tables):
        assert kg_tables.find_edges_by_fulltext("", GROUP) == []
        assert kg_tables.find_edges_by_fulltext("!!! ---", GROUP) == []
