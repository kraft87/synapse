"""Web bucket gate (_search_web_reranked): fuse BM25+vector, cross-encoder rerank, drop
chunks below an absolute floor, dedupe by parent page. Pure-logic — no DB, no Voyage
(the BM25/vector/rerank legs are stubbed). Unlike the fact floor there is NO keep->=1
backstop: an all-subfloor result serves [], which self-gates web on intent."""

from __future__ import annotations

import mcp_server.recall as recall_mod
from mcp_server.recall import Recall


def _bare() -> Recall:
    return object.__new__(Recall)


def _chunks(specs: list[tuple[int, int]]) -> list[dict]:
    """specs = [(chunk_id, artifact_id), ...] -> web-row dicts as the search legs return."""
    return [
        {"id": f"w:{cid}", "web_artifact_id": aid, "content": f"c{cid}", "url": f"http://x/{cid}"}
        for cid, aid in specs
    ]


def _wire(r: Recall, fused: list[dict], scored: list[tuple[int, float]]) -> None:
    # bm25 empty so _merge_rrf preserves `fused` order (recency/feedback mults are 1.0 with
    # created_at/retrieval_count absent); rerank returns caller-controlled (idx, score) pairs.
    r._search_bm25_web = lambda q, n: []  # type: ignore[method-assign]
    r._search_vector_web = lambda emb, n: fused  # type: ignore[method-assign]
    r._rerank_pool_scored = lambda q, pool: scored  # type: ignore[method-assign]


def test_web_floor_drops_subfloor(monkeypatch):
    monkeypatch.setattr(recall_mod, "_WEB_FLOOR", 0.60)
    r = _bare()
    _wire(
        r,
        _chunks([(0, 10), (1, 11), (2, 12), (3, 13)]),
        [(0, 0.91), (1, 0.55), (2, 0.50), (3, 0.30)],
    )
    out = r._search_web_reranked("q", [0.0])
    assert [c["id"] for c in out] == ["w:0"]  # only the 0.91 clears 0.60


def test_web_all_subfloor_serves_empty(monkeypatch):
    # THE key differentiator from the fact floor: no keep->=1, so web can serve nothing.
    monkeypatch.setattr(recall_mod, "_WEB_FLOOR", 0.60)
    r = _bare()
    _wire(r, _chunks([(0, 10), (1, 11), (2, 12)]), [(0, 0.55), (1, 0.40), (2, 0.30)])
    assert r._search_web_reranked("q", [0.0]) == []


def test_web_dedupe_by_artifact(monkeypatch):
    monkeypatch.setattr(recall_mod, "_WEB_FLOOR", 0.60)
    r = _bare()
    # two above-floor chunks share artifact 10 -> only the higher-ranked survives dedupe
    _wire(r, _chunks([(0, 10), (1, 10), (2, 11)]), [(0, 0.90), (1, 0.80), (2, 0.70)])
    assert [c["id"] for c in r._search_web_reranked("q", [0.0])] == ["w:0", "w:2"]


def test_web_degraded_rerank_serves_fused(monkeypatch):
    # reranker down/disabled -> top score 0.0 -> skip floor, serve fused order deduped (never blank)
    monkeypatch.setattr(recall_mod, "_WEB_FLOOR", 0.60)
    r = _bare()
    _wire(
        r, _chunks([(0, 10), (1, 11), (2, 12), (3, 13)]), [(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)]
    )
    assert [c["id"] for c in r._search_web_reranked("q", [0.0])] == ["w:0", "w:1", "w:2"]


def test_web_floor_disabled_serves_topk(monkeypatch):
    monkeypatch.setattr(recall_mod, "_WEB_FLOOR", 0.0)
    r = _bare()
    _wire(
        r,
        _chunks([(0, 10), (1, 11), (2, 12), (3, 13)]),
        [(0, 0.30), (1, 0.20), (2, 0.10), (3, 0.05)],
    )
    # floor off -> serve reranked top-_WEB_LIMIT regardless of score
    assert [c["id"] for c in r._search_web_reranked("q", [0.0])] == ["w:0", "w:1", "w:2"]


def test_web_floor_default_on():
    # validated 2026-07-24 (100% noise suppression at 100% helpful retention) -> shipped ON
    assert recall_mod._WEB_FLOOR == 0.60
