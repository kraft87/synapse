"""recall() starts each second-wave call as soon as its own inputs are ready.

The episode rerank needs only the fused BM25 + vector pool, so it runs while the KG and
web legs are still in flight. The fact-floor rerank needs only the KG leg's facts, so it
runs while the episode rerank does. Moving the waits must not change what is served: the
payload and the telemetry envelope are pinned across several leg-timing permutations.

Pure-logic tests — no DB, no Voyage: every leg is stubbed (same style as the other
recall tests) and _record_metrics captures the telemetry row instead of writing it.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

import mcp_server.recall as recall_mod
from ingestion.surfaces import FULL_TRUST
from mcp_server.recall import Recall

# Poison DSN: an unstubbed leg fails loudly instead of finding a database.
_NO_DB = "postgresql://synapse@127.0.0.1:1/nonexistent"
_WAIT = 2.0  # how long a stub waits for an overlap that should already be happening


class _FakeEmbedder:
    def embed(self, texts, task="query"):
        return [[0.0] * 4 for _ in texts]


_POOL = [
    {"id": f"e:{i}", "content": f"pool doc {i}", "doc_type": "episode", "created_at": None}
    for i in range(4)
]
_FACTS = [
    {"fact": f"fact {i}", "_uuid": f"u{i}", "_date": f"2026-10-0{i + 1}", "_ongoing": i == 1}
    for i in range(4)
]
_WEB = [{"id": "w:7", "content": "web chunk", "url": "https://x.example/a", "title": "A"}]
_PAIRS = [{"id": "f:old", "fact": "u0 used to be different", "superseded_by": "fact 0"}]
_NOTES = [{"id": "n:3", "hook": "A note hook", "note": "A note body."}]


def _wired(
    monkeypatch,
    delays: dict[str, float] | None = None,
    hooks: dict[str, Any] | None = None,
):
    """A Recall with every leg stubbed. ``delays`` sleeps a stub before it answers;
    ``hooks`` runs a callable at the start of a stub (the overlap probes)."""
    monkeypatch.setattr(recall_mod, "_RECALL_FACT_FLOOR", 0.42)
    delays = delays or {}
    hooks = hooks or {}

    def leg(name: str, answer: Any):
        def run(*_a, **_k):
            if name in hooks:
                hooks[name]()
            time.sleep(delays.get(name, 0.0))
            return answer(*_a) if callable(answer) else answer

        return run

    r = Recall(_NO_DB, "")
    r._ensure_embedder = lambda: _FakeEmbedder()
    r._search_bm25_episodes = leg("bm25", lambda *a: [dict(e) for e in _POOL])
    r._search_vector_episodes = leg("vec", lambda *a: [dict(e) for e in _POOL[2:]])
    r._search_web_reranked = leg("web", lambda *a: [dict(w) for w in _WEB])
    r._search_kg = leg("kg", lambda *a: ([dict(f) for f in _FACTS], []))
    r._search_notes = leg("notes", lambda *a: [dict(n) for n in _NOTES])
    r._fetch_superseded_pairs_pg = leg("superseded", lambda *a: [dict(p) for p in _PAIRS])
    # (pool index, score) in rerank order, deliberately not the pool order.
    r._rerank_pool_scored = leg("rerank", lambda q, pool: [(1, 0.9), (2, 0.7), (0, 0.5), (3, 0.3)])
    # The floor keeps the facts with an even index.
    r._floor_facts = leg(
        "floor", lambda q, facts: [f for f in facts if int(f["_uuid"][1:]) % 2 == 0]
    )
    r._surface_supersessions = lambda *a, **k: []
    r._episode_supersessions = lambda *a, **k: {}
    r._compact_to_passages = lambda q, eps, n, **_k: [
        {"id": e["id"], "content": f"passage {e['id']}"} for e in eps[:n]
    ]
    r._increment_fact_retrieval_counts = lambda *a, **k: None
    r._increment_retrieval_counts = lambda ids: None
    captured: list[dict] = []
    r._record_metrics = captured.append
    return r, captured


def _served(out: dict, row: dict) -> tuple[dict, dict]:
    """The payload and the non-timing telemetry: everything a reordering could change."""
    telemetry = {k: v for k, v in row.items() if not k.startswith("ms_")}
    return out, telemetry


def test_episode_rerank_runs_while_kg_and_web_are_in_flight(monkeypatch):
    rerank_started = threading.Event()
    saw: dict[str, bool] = {}
    r, _ = _wired(
        monkeypatch,
        hooks={
            "rerank": rerank_started.set,
            "kg": lambda: saw.__setitem__("kg", rerank_started.wait(_WAIT)),
            "web": lambda: saw.__setitem__("web", rerank_started.wait(_WAIT)),
        },
    )
    r.recall("q", trust=FULL_TRUST)
    assert saw == {"kg": True, "web": True}, "the episode rerank waited for the KG/web legs"


def test_fact_floor_rerank_runs_while_the_episode_rerank_is_in_flight(monkeypatch):
    floor_started = threading.Event()
    saw: dict[str, bool] = {}
    r, _ = _wired(
        monkeypatch,
        hooks={
            "floor": floor_started.set,
            "rerank": lambda: saw.__setitem__("rerank", floor_started.wait(_WAIT)),
        },
    )
    r.recall("q", trust=FULL_TRUST)
    assert saw == {"rerank": True}, "the fact floor waited for the episode rerank"


_PERMUTATIONS = [
    {},
    {"kg": 0.15},
    {"web": 0.15},
    {"rerank": 0.15},
    {"floor": 0.15},
    {"superseded": 0.15, "notes": 0.1},
    {"bm25": 0.1, "kg": 0.05, "rerank": 0.1, "floor": 0.05},
    {"vec": 0.1, "web": 0.2, "floor": 0.15},
]


@pytest.mark.parametrize("delays", _PERMUTATIONS)
def test_served_output_is_independent_of_leg_timing(monkeypatch, delays):
    r, captured = _wired(monkeypatch, delays=delays)
    out, telemetry = _served(r.recall("q", trust=FULL_TRUST), captured[0])
    # The fused pool is e:2, e:3, e:0, e:1; the stub rerank orders it 1, 2, 0, 3.
    assert [e["id"] for e in out["episodes"]] == ["e:3", "e:0", "e:2", "e:1"]
    assert out["facts"] == [
        {"fact": "fact 0", "id": "f:u0", "date": "2026-10-01"},
        {"fact": "fact 2", "id": "f:u2", "date": "2026-10-03"},
    ]
    assert out["web"] == [
        {"id": "w:7", "excerpt": "web chunk", "url": "https://x.example/a", "title": "A"}
    ]
    assert out["superseded_facts"] == _PAIRS
    assert out["notes"] == _NOTES
    assert "warnings" not in out
    assert telemetry["served_ids"]["facts"] == ["u0", "u2"]
    assert telemetry["kg_candidates"] == 4 and telemetry["n_facts"] == 2

    base_r, base_captured = _wired(monkeypatch)
    assert (out, telemetry) == _served(base_r.recall("q", trust=FULL_TRUST), base_captured[0])
