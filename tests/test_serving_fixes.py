"""Serving-side fixes over the reranked episode leg (pure-logic — no DB, no Voyage):

  1. Post-rerank recency re-injection (_apply_rerank_recency): the cross-encoder is
     recency-blind, so re-weight the FINAL ordering by created_at (14-day half-life),
     floored so old-but-relevant content is dampened, never annihilated.
  2. Query-echo suppression (_echo_suppressed_indices): drop served episodes that are
     the prompt quoting itself (compaction copies / re-ingested repeats), backfilling
     the freed slots from the next-ranked candidates.

Both ship ON; each has an env kill switch that restores the prior ordering exactly.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import mcp_server.recall as recall_mod
from mcp_server.recall import Recall


def _bare() -> Recall:
    # bypass __init__ (no DB/Voyage) — these helpers only touch the pool + module knobs
    return object.__new__(Recall)


def _ago(days: float) -> datetime:
    return datetime.now(UTC) - timedelta(days=days)


# --- Fix 1: post-rerank recency -------------------------------------------------


def test_newer_lower_score_outranks_older_higher_within_floor():
    # old (14d, exactly one half-life) 0.80 -> 0.40; new (today) 0.70 -> 0.70.
    # The newer doc wins despite a lower RAW rerank score, and neither is floor-clamped
    # (0.5 multiplier > 0.25 floor), so this is genuine recency tie-breaking.
    r = _bare()
    pool = [
        {"id": "e:old", "content": "old", "created_at": _ago(14)},
        {"id": "e:new", "content": "new", "created_at": _ago(0)},
    ]
    scored = [(0, 0.80), (1, 0.70)]  # reranker order: old first
    out = r._apply_rerank_recency(scored, pool)
    assert [pool[i]["id"] for i, _ in out] == ["e:new", "e:old"]


def test_recency_kill_switch_is_byte_identical(monkeypatch):
    monkeypatch.setattr(recall_mod, "_RERANK_RECENCY", False)
    r = _bare()
    pool = [
        {"id": "e:old", "content": "old", "created_at": _ago(120)},
        {"id": "e:new", "content": "new", "created_at": _ago(0)},
    ]
    scored = [(0, 0.80), (1, 0.70)]
    # disabled -> returns the incoming (raw rerank) order, unchanged
    assert r._apply_rerank_recency(scored, pool) == scored


def test_recency_floor_prevents_annihilation():
    # A very old (180d) but highly-relevant (1.0) doc would decay to ~1e-4 without a
    # floor and fall below a fresh, low-relevance (0.10) doc. The 0.25 floor clamps it
    # to 0.25 so it stays ranked ABOVE the fresh-but-weak doc.
    r = _bare()
    pool = [
        {"id": "e:old_relevant", "content": "relevant", "created_at": _ago(180)},
        {"id": "e:fresh_weak", "content": "weak", "created_at": _ago(0)},
    ]
    scored = [(0, 1.0), (1, 0.10)]
    out = r._apply_rerank_recency(scored, pool)
    by_id = {pool[i]["id"]: s for i, s in out}
    assert by_id["e:old_relevant"] == 0.25  # 1.0 * max(floor, tiny) == floor
    assert [pool[i]["id"] for i, _ in out] == ["e:old_relevant", "e:fresh_weak"]


def test_recency_degraded_rerank_untouched():
    # all-0.0 scores signal the degraded/RRF-order fallback — recency must not reorder it
    r = _bare()
    pool = [
        {"id": "e:0", "content": "a", "created_at": _ago(0)},
        {"id": "e:1", "content": "b", "created_at": _ago(90)},
    ]
    scored = [(0, 0.0), (1, 0.0)]
    assert r._apply_rerank_recency(scored, pool) == scored


# --- Fix 2: query-echo suppression ----------------------------------------------

# A long query and an episode whose content is that query quoted verbatim (the
# self-pollution the backtest measured). Normalized length is well over the 40-char gate.
_ECHO_QUERY = "what did we decide about the postgres connection pool sizing for the recall legs"


def test_echo_index_flags_verbatim_quote():
    r = _bare()
    items = [
        {"id": "e:echo", "content": f"[user] {_ECHO_QUERY}\n[assistant] noted"},
        {"id": "e:real", "content": "we sized the pool at four workers after profiling"},
    ]
    assert r._echo_suppressed_indices(_ECHO_QUERY, items) == [0]


def test_echo_short_query_untouched():
    # below _ECHO_MIN_QUERY_LEN the heuristic is meaningless — never suppress
    r = _bare()
    items = [{"id": "e:0", "content": "postgres postgres postgres"}]
    assert r._echo_suppressed_indices("postgres", items) == []


def test_echo_kill_switch_disables(monkeypatch):
    monkeypatch.setattr(recall_mod, "_SUPPRESS_QUERY_ECHO", False)
    r = _bare()
    items = [{"id": "e:echo", "content": _ECHO_QUERY}]
    assert r._echo_suppressed_indices(_ECHO_QUERY, items) == []


def test_echo_dropped_and_backfilled_via_select(monkeypatch):
    # recall's shared selection path: the rank-0 echo is dropped and the freed slot
    # backfills from the next-ranked candidates so the served count does not shrink.
    monkeypatch.setattr(recall_mod, "_EPISODE_CUTOFF_TAU", 0.0)  # fixed-k path
    r = _bare()
    pool = [
        {"id": "e:echo", "content": _ECHO_QUERY},  # the query quoting itself
        {"id": "e:1", "content": "sized the pool at four workers"},
        {"id": "e:2", "content": "hnsw ef_search set to two hundred"},
        {"id": "e:3", "content": "halfvec index on the embedding column"},
    ]
    r._rerank_pool_scored = lambda q, p: [(i, 1.0 - i * 0.01) for i in range(len(p))]
    out, n_echo = r._select_episodes(_ECHO_QUERY, pool, limit=2)
    assert n_echo == 1
    assert [d["id"] for d in out] == ["e:1", "e:2"]  # echo gone, backfilled to full count


def test_echo_kill_switch_restores_prior_selection(monkeypatch):
    monkeypatch.setattr(recall_mod, "_EPISODE_CUTOFF_TAU", 0.0)
    monkeypatch.setattr(recall_mod, "_SUPPRESS_QUERY_ECHO", False)
    r = _bare()
    pool = [
        {"id": "e:echo", "content": _ECHO_QUERY},
        {"id": "e:1", "content": "sized the pool at four workers"},
        {"id": "e:2", "content": "hnsw ef_search set to two hundred"},
    ]
    r._rerank_pool_scored = lambda q, p: [(i, 1.0 - i * 0.01) for i in range(len(p))]
    out, n_echo = r._select_episodes(_ECHO_QUERY, pool, limit=2)
    assert n_echo == 0
    assert [d["id"] for d in out] == ["e:echo", "e:1"]  # prior behavior: echo served


# --- defaults -------------------------------------------------------------------


def test_serving_fix_defaults_on():
    # both validated fixes ship ON; the env vars are kill switches, not opt-ins
    assert recall_mod._RERANK_RECENCY is True
    assert recall_mod._SUPPRESS_QUERY_ECHO is True
    assert recall_mod._RERANK_RECENCY_FLOOR == 0.25
