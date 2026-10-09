"""Fact-relevance gate (_floor_facts): drop off-topic served facts below an absolute
cross-encoder floor. Pure-logic — no DB, no Voyage (the embedder is stubbed). OFF by
default in recall() via the _RECALL_FACT_FLOOR > 0 guard; these test the filter itself."""

from __future__ import annotations

import mcp_server.recall as recall_mod
from mcp_server.recall import Recall


def _bare() -> Recall:
    return object.__new__(Recall)


class _Emb:
    def __init__(self, scored):
        self._scored = scored

    def rerank_scored(self, query, docs):
        if self._scored == "raise":
            raise RuntimeError("voyage down")
        return self._scored


def _facts(n: int) -> list[dict]:
    return [{"fact": f"fact{i}", "_uuid": f"u{i}"} for i in range(n)]


def test_floor_facts_drops_subfloor(monkeypatch):
    monkeypatch.setattr(recall_mod, "_RECALL_FACT_FLOOR", 0.40)
    r = _bare()
    facts = _facts(5)
    r._reranker = _Emb([(0, 0.83), (1, 0.55), (2, 0.44), (3, 0.39), (4, 0.28)])
    out = r._floor_facts("q", facts)
    assert [f["fact"] for f in out] == ["fact0", "fact1", "fact2"]  # 0.39, 0.28 dropped


def test_floor_facts_keeps_at_least_one(monkeypatch):
    # all below the floor -> never blank the bucket, keep the single best
    monkeypatch.setattr(recall_mod, "_RECALL_FACT_FLOOR", 0.99)
    r = _bare()
    facts = _facts(3)
    r._reranker = _Emb([(0, 0.6), (1, 0.5), (2, 0.4)])
    out = r._floor_facts("q", facts)
    assert [f["fact"] for f in out] == ["fact0"]


def test_floor_facts_keeps_all_on_rerank_failure(monkeypatch):
    monkeypatch.setattr(recall_mod, "_RECALL_FACT_FLOOR", 0.40)
    r = _bare()
    facts = _facts(4)
    r._reranker = _Emb("raise")
    out = r._floor_facts("q", facts)
    assert out == facts


def test_fact_floor_default_off():
    # the prod default must be OFF (0) — recall() only gates when > 0
    assert recall_mod._RECALL_FACT_FLOOR == 0.0


# --- shared helper _floor_by_rerank: the generalized contract both wrappers delegate to ---


def test_floor_by_rerank_custom_text_key_and_keep_min():
    r = _bare()
    items = [{"body": f"i{i}"} for i in range(4)]
    r._reranker = _Emb([(0, 0.7), (1, 0.5), (2, 0.3), (3, 0.1)])
    # non-default text_key + keep_min=0 -> pure floor, may return fewer
    out = r._floor_by_rerank("q", items, 0.45, text_key="body", keep_min=0)
    assert [i["body"] for i in out] == ["i0", "i1"]


def test_floor_by_rerank_keep_min_backstops_empty():
    r = _bare()
    items = [{"fact": f"f{i}"} for i in range(3)]
    r._reranker = _Emb([(0, 0.3), (1, 0.2), (2, 0.1)])  # all below floor
    # keep_min=0 -> [] ; keep_min=2 -> top 2 survive the blanket
    assert r._floor_by_rerank("q", items, 0.5, keep_min=0) == []
    assert [i["fact"] for i in r._floor_by_rerank("q", items, 0.5, keep_min=2)] == ["f0", "f1"]
