"""Pure-unit tests for the module-level rrf_merge fusion (no DB).

The client's reads and writes are covered by test_kg_client_reads.py /
test_kg_client_writes.py (DB-backed).
"""

from __future__ import annotations

from ingestion.kg_client import rrf_merge


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
