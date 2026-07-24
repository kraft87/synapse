"""Inline timeline relevance gate (_floor_timeline): drop off-topic events below an absolute
cross-encoder floor. Pure-logic — no DB, no Voyage (reranker stubbed). Unlike _floor_facts
there is NO keep->=1 backstop, so an all-subfloor result serves [] (self-gates like web).
The standalone recall_timeline() deep path is not floored."""

from __future__ import annotations

import mcp_server.recall as recall_mod
from mcp_server.recall import Recall


def _bare() -> Recall:
    return object.__new__(Recall)


class _RR:
    def __init__(self, scored):
        self._scored = scored

    def rerank_scored(self, query, docs):
        if self._scored == "raise":
            raise RuntimeError("voyage down")
        return self._scored


def _events(n: int) -> list[dict]:
    return [{"fact": f"event{i}", "_id": i, "salience": 1} for i in range(n)]


def test_timeline_floor_drops_subfloor(monkeypatch):
    monkeypatch.setattr(recall_mod, "_TIMELINE_FLOOR", 0.40)
    r = _bare()
    r._reranker = _RR([(0, 0.74), (1, 0.48), (2, 0.41), (3, 0.30), (4, 0.20)])
    out = r._floor_timeline("q", _events(5))
    assert [e["fact"] for e in out] == ["event0", "event1", "event2"]  # 0.30, 0.20 dropped


def test_timeline_all_subfloor_serves_empty(monkeypatch):
    # differentiator from the fact floor: no keep->=1, timeline may serve nothing
    monkeypatch.setattr(recall_mod, "_TIMELINE_FLOOR", 0.40)
    r = _bare()
    r._reranker = _RR([(0, 0.38), (1, 0.30), (2, 0.22)])
    assert r._floor_timeline("q", _events(3)) == []


def test_timeline_floor_keeps_all_on_rerank_failure(monkeypatch):
    monkeypatch.setattr(recall_mod, "_TIMELINE_FLOOR", 0.40)
    r = _bare()
    r._reranker = _RR("raise")
    ev = _events(4)
    assert r._floor_timeline("q", ev) == ev


def test_timeline_floor_reranker_disabled_keeps_all(monkeypatch):
    monkeypatch.setattr(recall_mod, "_TIMELINE_FLOOR", 0.40)
    r = _bare()
    r._reranker = None
    ev = _events(3)
    assert r._floor_timeline("q", ev) == ev


def test_timeline_floor_default_on():
    # validated 2026-07-24: 86% noise suppression at 87% helpful retention -> shipped ON
    assert recall_mod._TIMELINE_FLOOR == 0.40
