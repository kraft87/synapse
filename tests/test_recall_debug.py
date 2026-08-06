"""recall(debug=True) — the phase-2 dashboard debug envelope.

Pure-logic tests (no DB, no Voyage): every search leg, the embedder, the reranker, and
the metrics writer are stubbed so recall() runs end-to-end in-process. Guards the two
contracts the dashboard console depends on:
  * debug=True attaches a `debug` envelope surfacing the SAME numbers the engine measures
    for recall_metrics (per-leg ms, pool sizes, rerank model + top score, est_tokens);
  * debug=False omits the key AND leaves the rest of the payload byte-identical (a live
    call-rate A/B relies on the non-debug path being unchanged).
"""

from __future__ import annotations

from typing import Any

import mcp_server.recall as recall_module
from mcp_server.recall import Recall

# Poison DSN, not "": libpq reads its PG* env vars for an empty conninfo, so an
# unstubbed leg would quietly connect on any host that has them set (that is exactly
# how CI hid the unstubbed _search_bm25_web leg while it failed on dev boxes). A
# refused port turns a missing stub into an immediate, obvious failure everywhere.
_NO_DB = "postgresql://synapse@127.0.0.1:1/nonexistent"


class _FakeEmbedder:
    def embed(self, texts: list[str], task: str | None = None) -> list[list[float]]:
        return [[0.0, 0.0] for _ in texts]


class _FakeReranker:
    def rerank_scored(self, query, documents, top_k=None):  # pragma: no cover - unused here
        return [(i, 1.0 - 0.01 * i) for i in range(len(documents))]


class _FakeTimeline:
    def recall_timeline(self, **kwargs: Any) -> dict[str, Any]:
        return {"items": []}


def _episode(eid: int, content: str) -> dict[str, Any]:
    return {
        "id": f"e:{eid}",
        "content": content,
        "project": "synapse",
        "created_at": "2026-07-10T12:00:00+00:00",
        "doc_type": "episode",
        "retrieval_count": 0,
    }


def _wire_stubs(r: Recall) -> None:
    """Replace every DB/network touchpoint recall() reaches with an in-process fake."""
    r._embedder = _FakeEmbedder()
    r._reranker = _FakeReranker()
    r._timeline_engine = _FakeTimeline()
    r._search_bm25_episodes = lambda q, p, n: [_episode(1, "alpha episode body")]  # type: ignore[method-assign]
    r._search_vector_episodes = lambda emb, p, n: [_episode(2, "beta episode body")]  # type: ignore[method-assign]
    # BOTH halves of the web leg: _search_web_reranked fuses BM25 + vector, so stubbing
    # only the vector half leaves _search_bm25_web reaching for a real connection.
    r._search_vector_web = lambda emb, n: []  # type: ignore[method-assign]
    r._search_bm25_web = lambda q, n: []  # type: ignore[method-assign]
    r._search_kg = lambda q, emb, g, sf, fact_limit: ([], [])  # type: ignore[method-assign]
    r._rerank_pool_scored = lambda q, pool: [(i, 0.9 - 0.1 * i) for i in range(len(pool))]  # type: ignore[method-assign]
    r._fetch_history_pairs_pg = lambda g, uuids, cap: []  # type: ignore[method-assign]
    # Stub compaction empty so no chunker/reranker runs in the passage stage. There's no
    # full-episode fallback anymore, so the episode bucket is omitted — these tests assert
    # on the debug envelope + est_tokens self-consistency, not on episode presence.
    r._compact_to_passages = lambda q, eps, n: []  # type: ignore[method-assign]
    r._surface_supersessions = lambda emb, g, served, cap=None: []  # type: ignore[method-assign]
    r._episode_supersessions = lambda ids, g, cap=6: {}  # type: ignore[method-assign]
    r._record_metrics = lambda m: None  # type: ignore[method-assign]  # no background DB write


def test_debug_true_attaches_envelope():
    r = Recall(_NO_DB, "")
    _wire_stubs(r)
    out = r.recall("connection pooling decisions", debug=True)

    assert "debug" in out
    d = out["debug"]
    assert isinstance(d["total_ms"], float)
    # Timed legs: the six always-on legs plus timeline (enabled by default). The prefs
    # leg was removed 2026-07-27 — it must never reappear in the waterfall.
    assert set(d["legs_ms"]) == {
        "embed",
        "bm25",
        "vector",
        "kg",
        "web",
        "rerank",
        "timeline",
        "notes",
    }
    assert "preferences" not in out  # the prefs bucket is gone from the served payload
    assert all(isinstance(v, float) for v in d["legs_ms"].values())
    # Pool sizes mirror the fused bm25(1)+vector(1)=2 pool; KG stub served no candidates.
    assert d["pool_sizes"] == {"bm25": 1, "vector": 1, "fused": 2, "kg_candidates": 0}
    assert d["rerank"]["top_score"] == 0.9  # RAW pre-recency top from the rerank stub
    assert d["rerank"]["model"] == recall_module._embedding._RERANK_MODEL
    assert (
        d["est_tokens"]
        == recall_module._served_chars({k: v for k, v in out.items() if k != "debug"}) // 4
    )


def test_debug_false_omits_key_and_is_byte_identical():
    r = Recall(_NO_DB, "")
    _wire_stubs(r)
    with_dbg = r.recall("connection pooling decisions", debug=True)
    without = r.recall("connection pooling decisions", debug=False)

    assert "debug" not in without
    # The served payload is identical apart from the debug envelope.
    assert {k: v for k, v in with_dbg.items() if k != "debug"} == without


def test_disabled_legs_are_omitted_from_legs_ms(monkeypatch):
    # A disabled timeline leg has no timed future, so its key is dropped — the
    # console renders it as untimed/skipped rather than a spurious 0 ms bar.
    monkeypatch.setattr(recall_module, "_TIMELINE_IN_RECALL", False)
    monkeypatch.setattr(recall_module, "_NOTES_IN_RECALL", False)
    r = Recall(_NO_DB, "")
    _wire_stubs(r)
    d = r.recall("q", debug=True)["debug"]
    assert set(d["legs_ms"]) == {"embed", "bm25", "vector", "kg", "web", "rerank"}
    assert "timeline" not in d["legs_ms"] and "prefs" not in d["legs_ms"]
    assert "notes" not in d["legs_ms"]
