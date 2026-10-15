"""RecallConfig: one set of defaults, built once, shared by recall() and recall_full_turns.

Pure-logic: every leg is stubbed (no DB, no Voyage)."""

from __future__ import annotations

import dataclasses
from typing import Any

from ingestion.surfaces import FULL_TRUST
from mcp_server.recall import Recall
from mcp_server.recall_config import RecallConfig

_NO_DB = "postgresql://synapse@127.0.0.1:1/nonexistent"


class _Emb:
    def embed(self, texts, task="query"):
        return [[0.0] * 4 for _ in texts]


def _episode(i: int) -> dict[str, Any]:
    return {
        "id": f"e:{i}",
        "content": f"pool doc {i}",
        "doc_type": "episode",
        "project": None,
        "created_at": None,
        "retrieval_count": 0,
        "session_id": f"s{i}",
    }


def _engine(config: RecallConfig):
    r = Recall(_NO_DB, "", config=config)
    seen: dict[str, Any] = {"fetch": [], "pool": []}

    def _bm25(q, proj, limit, sid=None, allowed=None, own=None):
        seen["fetch"].append(limit)
        return [_episode(i) for i in range(10)]

    r._ensure_embedder = lambda: _Emb()
    r._search_bm25_episodes = _bm25
    r._search_vector_episodes = lambda emb, proj, limit, sid=None, allowed=None, own=None: []
    r._search_web_reranked = lambda q, emb, **_k: []
    r._search_kg = lambda *a, **k: ([], [])
    r._search_notes = lambda q, emb, proj, audience=None: []
    r._fetch_superseded_pairs_pg = lambda gid, uuids, cap: []
    r._surface_supersessions = lambda *a, **k: []
    r._episode_supersessions = lambda *a, **k: {}
    r._compact_to_passages = lambda q, eps, n, **_k: [{"id": e["id"], "content": "p"} for e in eps]

    def _rerank(q, pool):
        seen["pool"].append(len(pool))
        return [(i, 0.9 - 0.01 * i) for i in range(len(pool))]

    r._rerank_pool_scored = _rerank
    r._increment_fact_retrieval_counts = lambda *a, **k: None
    r._increment_retrieval_counts = lambda ids: None
    r._record_metrics = lambda m: None
    return r, seen


def test_one_config_sizes_the_pool_for_recall_and_recall_full_turns():
    config = dataclasses.replace(RecallConfig(), episode_fetch=7, episode_rerank_pool=3)
    r, seen = _engine(config)
    r.recall("where is the gizmo", trust=FULL_TRUST)
    r.recall_episodes("where is the gizmo", trust=FULL_TRUST)
    assert seen["fetch"] == [7, 7]  # both paths read the one fetch depth
    assert seen["pool"] == [3, 3]  # and cut the pool at the one rerank width


def test_the_defaults_are_the_measured_policy(monkeypatch):
    for name in (
        "SYNAPSE_RECALL_FLOOR",
        "SYNAPSE_RECALL_WEB_FLOOR",
        "SYNAPSE_RECALL_NOTES_FLOOR",
        "SYNAPSE_RECALL_CHUNK_CANDIDATES",
        "SYNAPSE_RECALL_PASSAGE_N",
    ):
        monkeypatch.delenv(name, raising=False)
    cfg = RecallConfig()
    assert (cfg.episode_fetch, cfg.episode_rerank_pool) == (100, 100)
    assert (cfg.fact_limit, cfg.recall_episode_limit, cfg.recall_passage_n) == (12, 5, 6)
    # The floors are reranker-calibrated: off for the abstention floor, 0.60 (rerank-3-lite
    # presets pin it too) for the web and notes gates. Chunk candidates are opt-in.
    assert (cfg.recall_floor, cfg.web_floor, cfg.notes_floor) == (0.0, 0.60, 0.60)
    assert cfg.chunk_candidates is False


def test_the_deployment_knobs_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("SYNAPSE_RECALL_FLOOR", "0.55")
    monkeypatch.setenv("SYNAPSE_RECALL_WEB_FLOOR", "0.7")
    monkeypatch.setenv("SYNAPSE_RECALL_NOTES_FLOOR", "0")
    monkeypatch.setenv("SYNAPSE_RECALL_CHUNK_CANDIDATES", "1")
    monkeypatch.setenv("SYNAPSE_RECALL_PASSAGE_N", "4")
    cfg = RecallConfig()
    assert (cfg.recall_floor, cfg.web_floor, cfg.notes_floor) == (0.55, 0.7, 0.0)
    assert cfg.chunk_candidates is True and cfg.recall_passage_n == 4
