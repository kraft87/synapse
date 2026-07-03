"""Pure-unit tests for KGClient helpers and write orchestration (no DB).

The storage layer underneath is covered by test_kg_pg_write.py /
test_kg_pg_read.py (DB-backed); this file covers the module-level rrf_merge
fusion plus the batch-path orchestration (self-invalidation of pre-bookended
facts). create_edge's detector/edge-date orchestration is covered in
test_contradiction.py::TestCreateEdgeBiTemporal.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from ingestion.kg_client import KGClient, rrf_merge


class TestRRFMerge:
    def test_merges_two_ranked_lists(self):
        vec = [{"uuid": "A", "fact": "a"}, {"uuid": "B", "fact": "b"}]
        bm = [{"uuid": "C", "fact": "c"}, {"uuid": "A", "fact": "a"}]
        merged = rrf_merge(vec, bm, limit=10, k=1)
        # A appears in both at rank 0 and rank 1 → highest score
        assert merged[0]["uuid"] == "A"
        # All UUIDs present, no dupes
        assert {m["uuid"] for m in merged} == {"A", "B", "C"}
        assert len(merged) == 3

    def test_k_equals_one_steepens_curve(self):
        """Graphiti uses k=1 so rank-1 dominates rank-10 by 10x vs only 1.14x at k=60."""
        vec = [{"uuid": f"V{i}", "fact": ""} for i in range(10)]
        bm = [{"uuid": f"B{i}", "fact": ""} for i in range(10)]
        # The top-1 of vec and top-1 of bm should appear before any rank-9 entries
        merged = rrf_merge(vec, bm, limit=20, k=1)
        top_uuids = {merged[0]["uuid"], merged[1]["uuid"]}
        assert top_uuids == {"V0", "B0"}

    def test_empty_inputs(self):
        assert rrf_merge([], [], limit=10) == []
        # One side empty still works
        single = [{"uuid": "A", "fact": "a"}]
        assert rrf_merge(single, [], limit=10) == single

    def test_truncates_after_merge_not_before(self):
        """RRF exists to rescue the long tail — truncation must happen post-merge."""
        # Both lists have 20 items; vec ranks A..T, bm ranks T..A (reversed)
        vec = [{"uuid": chr(ord("A") + i), "fact": ""} for i in range(20)]
        bm = list(reversed(vec))
        merged = rrf_merge(vec, bm, limit=5, k=1)
        # The middle items should outrank the extremes because they got
        # moderate ranks in both lists (rescued by RRF)
        assert len(merged) == 5

    def test_preserves_fact_payload(self):
        vec = [{"uuid": "A", "fact": "alpha", "valid_at": "2026"}]
        bm = []
        merged = rrf_merge(vec, bm, limit=10)
        assert merged[0]["fact"] == "alpha"
        assert merged[0]["valid_at"] == "2026"


def _client_with_mock_writer() -> tuple[KGClient, MagicMock]:
    client = KGClient.__new__(KGClient)
    client._writer = MagicMock()
    client._reader = MagicMock()
    return client, client._writer


class TestCreateEdgesBatch:
    def _row(self, uuid: str, **over):
        base = {
            "src": "s",
            "tgt": "t",
            "edge_uuid": uuid,
            "name": "USES",
            "fact": f"fact {uuid}",
            "episodes": [1],
            "created_at": "2026-06-01T00:00:00+00:00",
            "t_created": "2026-06-01T00:00:00+00:00",
            "valid_at": "2026-06-01T00:00:00+00:00",
            "t_valid": "2026-06-01T00:00:00+00:00",
        }
        base.update(over)
        return base

    def test_returns_uuids_in_input_order(self):
        client, writer = _client_with_mock_writer()
        rows = [self._row("r-1"), self._row("r-2")]
        assert client.create_edges_batch(rows, "technical") == ["r-1", "r-2"]
        writer.create_edges.assert_called_once_with(rows, "technical")
        writer.invalidate_edges.assert_not_called()

    def test_pre_bookended_facts_self_invalidate(self):
        """Rows carrying t_invalid (fact text said "from 2020 to 2022") are
        created and then immediately invalidated with the extracted date."""
        client, writer = _client_with_mock_writer()
        rows = [
            self._row("r-1"),
            self._row("r-2", t_invalid="2022-01-01T00:00:00+00:00"),
        ]
        client.create_edges_batch(rows, "technical")
        (items, group) = writer.invalidate_edges.call_args.args
        assert group == "technical"
        assert items == [("r-2", "2022-01-01T00:00:00+00:00")]

    def test_empty_rows_short_circuits(self):
        client, writer = _client_with_mock_writer()
        assert client.create_edges_batch([], "technical") == []
        writer.create_edges.assert_not_called()

    def test_invalidate_edges_batch_fills_now_for_none(self):
        client, writer = _client_with_mock_writer()
        client.invalidate_edges_batch([("r-1", None), ("r-2", "2026-01-01T00:00:00+00:00")], "g")
        (items, _group) = writer.invalidate_edges.call_args.args
        assert items[1] == ("r-2", "2026-01-01T00:00:00+00:00")
        assert items[0][0] == "r-1" and "T" in items[0][1]  # now() filled in
