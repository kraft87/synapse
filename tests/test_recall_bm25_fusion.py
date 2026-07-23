"""Unit tests for the recall BM25-fusion of the final episode order (_fuse_bm25_order).

Validated 2026-07-23: RRF-fusing the cross-encoder rerank order with the pool's BM25 order
recovers answer episodes the web-trained reranker buries (exact lexical matches). Pure logic,
no DB / no API.
"""

from __future__ import annotations

from mcp_server.recall import _RECALL_BM25_FUSE, Recall


def _eps(specs: list[tuple[int, float | None]]) -> list[dict]:
    """specs = [(id, bm25_score_or_None), ...] in RERANK order (index 0 = top reranked)."""
    return [{"id": f"e:{i}", "doc_type": "episode", "bm25_score": bm} for i, bm in specs]


def test_flag_defaults_on() -> None:
    assert _RECALL_BM25_FUSE is True


def test_fuse_lifts_lexical_match_the_reranker_buried() -> None:
    # Answer e:99 is LAST in rerank order (pos 20) but the strongest BM25 hit -> must rise.
    eps = _eps([(i, None) for i in range(20)] + [(99, 10.0)])
    out = Recall._fuse_bm25_order(eps)
    ids = [e["id"] for e in out]
    assert ids.index("e:99") < 5, f"expected e:99 lifted into top-5, got {ids[:6]}"


def test_fuse_noop_when_no_bm25_hits() -> None:
    # Vector-only pool (no bm25_score anywhere) -> order unchanged.
    eps = _eps([(i, None) for i in range(6)])
    assert [e["id"] for e in Recall._fuse_bm25_order(eps)] == [e["id"] for e in eps]


def test_fuse_short_list_passthrough() -> None:
    eps = _eps([(1, 5.0)])
    assert Recall._fuse_bm25_order(eps) == eps


def test_fuse_preserves_membership() -> None:
    eps = _eps([(0, 1.0), (1, None), (2, 9.0), (3, None), (4, 3.0)])
    out = Recall._fuse_bm25_order(eps)
    assert {e["id"] for e in out} == {e["id"] for e in eps}
    assert len(out) == len(eps)
