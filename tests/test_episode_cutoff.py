"""Adaptive episode score-cutoff serving (recall_episodes variable-k).

Pure-logic tests — no DB, no Voyage. Cover the _cutoff_k math and that
_select_episodes is byte-identical to fixed-k when disabled (the prod default)
and applies the relative cutoff (with graceful degrade) when enabled.
"""

from __future__ import annotations

import mcp_server.recall as recall_mod
from mcp_server.recall import Recall, _cutoff_k


# --- _cutoff_k -------------------------------------------------------------
def test_cutoff_k_empty():
    assert _cutoff_k([], 0.5, 3, 8) == 0


def test_cutoff_k_dominant_top_floored_to_min_k():
    # only the top doc clears tau*top, but min_k floors the count
    assert _cutoff_k([1.0, 0.1, 0.05], 0.5, 3, 8) == 3


def test_cutoff_k_partial_cut():
    # tau*top = 0.45 -> 0.9, 0.8, 0.5 qualify; 0.2, 0.1 do not
    assert _cutoff_k([0.9, 0.8, 0.5, 0.2, 0.1], 0.5, 1, 8) == 3


def test_cutoff_k_many_relevant_clamped_to_max_k():
    assert _cutoff_k([0.9] * 20, 0.5, 3, 8) == 8


def test_cutoff_k_degraded_top_zero_uses_max_k():
    # rerank degraded to RRF order (all 0.0); caller treats this as a fallback signal,
    # but _cutoff_k itself returns a bounded count.
    assert _cutoff_k([0.0, 0.0, 0.0], 0.5, 3, 8) == 3


def test_cutoff_k_respects_pool_size():
    assert _cutoff_k([0.9, 0.1], 0.5, 5, 8) == 2  # min_k=5 but only 2 docs


# --- _select_episodes ------------------------------------------------------
def _bare_recall() -> Recall:
    # bypass __init__ (no DB/Voyage); _select_episodes only needs the rerank methods
    return object.__new__(Recall)


def _pool(n: int) -> list[dict]:
    return [{"id": f"e:{i}", "content": f"doc{i}"} for i in range(n)]


def test_select_episodes_fixed_k_when_disabled(monkeypatch):
    monkeypatch.setattr(recall_mod, "_EPISODE_CUTOFF_TAU", 0.0)
    r = _bare_recall()
    pool = _pool(10)
    r._rerank_pool = lambda q, p: p  # identity rerank
    out = r._select_episodes("q", pool, limit=5)
    assert out == pool[:5]


def test_select_episodes_cutoff_when_enabled(monkeypatch):
    monkeypatch.setattr(recall_mod, "_EPISODE_CUTOFF_TAU", 0.5)
    monkeypatch.setattr(recall_mod, "_EPISODE_CUTOFF_MIN_K", 1)
    monkeypatch.setattr(recall_mod, "_EPISODE_CUTOFF_MAX_K", 8)
    r = _bare_recall()
    pool = _pool(6)
    # top dominates: 0.9, 0.5 clear tau*top=0.45 -> serve 2 (fewer than limit=5)
    r._rerank_pool_scored = lambda q, p: [
        (0, 0.9),
        (1, 0.5),
        (2, 0.2),
        (3, 0.1),
        (4, 0.05),
        (5, 0.01),
    ]
    out = r._select_episodes("q", pool, limit=5)
    assert [d["id"] for d in out] == ["e:0", "e:1"]


def test_select_episodes_falls_back_to_fixed_k_on_degraded_rerank(monkeypatch):
    monkeypatch.setattr(recall_mod, "_EPISODE_CUTOFF_TAU", 0.5)
    r = _bare_recall()
    pool = _pool(10)
    r._rerank_pool_scored = lambda q, p: [(i, 0.0) for i in range(len(p))]  # degraded
    out = r._select_episodes("q", pool, limit=5)
    assert out == pool[:5]


def test_select_episodes_empty_pool():
    r = _bare_recall()
    assert r._select_episodes("q", [], limit=5) == []
