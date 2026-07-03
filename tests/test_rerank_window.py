"""Long-episode rerank windowing (recall._bm25_best_window / _rerank_docs / max-collapse).

Pure-logic tests — no DB, no Voyage. Guard the tail-recall fix: the reranker must also see the
BM25-relevant window of long episodes, and an episode must be scored by max(head, window) so a
tail answer can win without ever lowering the head's score (the conservative property that makes
it a strict, no-regression change).
"""

from __future__ import annotations

import mcp_server.recall as recall_mod
from mcp_server.recall import _RERANK_DOC_CAP, Recall, _bm25_best_window, _bm25_tokenize


def _mk(content: str) -> dict:
    return {"content": content, "id": "e:1"}


# --- _bm25_best_window -----------------------------------------------------
def test_window_short_content_returns_whole():
    txt = "a short turn under the cap"
    assert _bm25_best_window(txt, ["short"], _RERANK_DOC_CAP) == txt


def test_window_locates_query_region_past_the_head():
    # ~9k chars of filler (no query terms), then the answer region — well past the 4k head.
    filler = "alpha beta gamma delta epsilon " * 320
    answer = " the special marker zebrafish lives in this exact spot " * 6
    content = filler + answer + (" trailing padding text " * 60)
    assert len(content) > _RERANK_DOC_CAP
    win = _bm25_best_window(content, _bm25_tokenize("zebrafish marker"), _RERANK_DOC_CAP)
    assert "zebrafish" in win  # window found the answer region, which the first-4k head misses
    assert "zebrafish" not in content[:_RERANK_DOC_CAP]
    assert len(win) <= _RERANK_DOC_CAP


# --- _rerank_docs ----------------------------------------------------------
def test_rerank_docs_short_items_head_only(monkeypatch):
    monkeypatch.setattr(recall_mod, "_RERANK_WINDOW", True)
    r = Recall("", "")
    docs, owner = r._rerank_docs("q", [_mk("a" * 100), _mk("b" * 100)])
    assert owner == [0, 1]
    assert len(docs) == 2


def test_rerank_docs_long_item_adds_window(monkeypatch):
    monkeypatch.setattr(recall_mod, "_RERANK_WINDOW", True)
    r = Recall("", "")
    long = "x " * (_RERANK_DOC_CAP)  # > cap chars
    docs, owner = r._rerank_docs("q", [_mk("short"), _mk(long)])
    assert owner == [0, 1, 1]  # short -> head; long -> head + window
    assert len(docs) == 3


def test_rerank_docs_flag_off_head_only(monkeypatch):
    monkeypatch.setattr(recall_mod, "_RERANK_WINDOW", False)
    r = Recall("", "")
    long = "x " * (_RERANK_DOC_CAP)
    docs, owner = r._rerank_docs("q", [_mk(long)])
    assert owner == [0]
    assert len(docs) == 1


# --- max-collapse in _rerank_pool_scored -----------------------------------
class _FakeEmb:
    def __init__(self, scores: dict[int, float]) -> None:
        self._scores = scores

    def rerank_scored(self, query, docs):
        return [(i, self._scores[i]) for i in range(len(docs))]


def test_pool_scored_scores_episode_by_max_of_its_docs(monkeypatch):
    monkeypatch.setattr(recall_mod, "_RERANK_WINDOW", True)
    r = Recall("", "")
    long = "x " * (_RERANK_DOC_CAP)
    pool = [_mk("short"), _mk(long)]
    # docs = [short-head(0), long-head(1), long-window(2)]; the long episode's WINDOW scores high,
    # its head low. max() must let the long episode win, proving tail recovery without head-lowering.
    r._reranker = _FakeEmb({0: 0.2, 1: 0.1, 2: 0.9})
    scored = r._rerank_pool_scored("q", pool)
    assert scored[0] == (1, 0.9)  # long episode first, scored by its window
    assert dict(scored)[0] == 0.2  # short episode keeps its head score
    assert len(scored) == 2  # collapsed back to one entry per pool item


def test_pool_scored_degrades_to_rrf_on_rerank_error(monkeypatch):
    monkeypatch.setattr(recall_mod, "_RERANK_WINDOW", True)
    r = Recall("", "")

    class _Boom:
        def rerank_scored(self, q, d):
            raise RuntimeError("rate limit")

    r._reranker = _Boom()
    scored = r._rerank_pool_scored("q", [_mk("a"), _mk("b" * (_RERANK_DOC_CAP + 10))])
    assert scored == [(0, 0.0), (1, 0.0)]  # RRF order preserved, 0.0 signals fallback
