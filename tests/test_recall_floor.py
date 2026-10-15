"""Abstention floor (SYNAPSE_RECALL_FLOOR, RecallConfig.recall_floor).

Marker: when real rerank scores exist and the RAW pre-recency top score is strictly below
the floor, the recall_metrics served_ids envelope gains {"would_abstain": true,
"floor": <float>}.

Enforcement: recall() drops the EPISODE bucket when the raw top rerank score is in
(0, floor) under working retrieval (query_emb present). Facts are unaffected.
recall_episodes() (drill-down) never enforces — the caller asked for turns, but it records
the marker. Telemetry and payload agree on WHEN a recall was below the floor.

Pure-logic tests — no DB, no Voyage: every search leg is stubbed (same style as the other
recall tests) and _record_metrics captures the telemetry row instead of writing it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from mcp_server.recall import Recall
from mcp_server.recall_config import RecallConfig

# Poison DSN, not "": libpq reads its PG* env vars for an empty conninfo, so an
# unstubbed leg would quietly connect on any host that has them set (that is exactly
# how CI hid the unstubbed _search_bm25_web leg while it failed on dev boxes). A
# refused port turns a missing stub into an immediate, obvious failure everywhere.
_NO_DB = "postgresql://synapse@127.0.0.1:1/nonexistent"


class _FakeEmbedder:
    def __init__(self, fail: bool = False) -> None:
        self._fail = fail

    def embed(self, texts, task="query"):
        if self._fail:
            raise RuntimeError("embedding backend down")
        return [[0.0] * 4 for _ in texts]


class _FakeTimeline:
    def recall_timeline(self, **kwargs):
        return {"items": []}


def _pool(n: int = 4) -> list[dict]:
    return [
        {
            "id": f"e:{i}",
            "content": f"pool doc {i}",
            "doc_type": "episode",
            "project": None,
            "created_at": None,
            "retrieval_count": 0,
        }
        for i in range(n)
    ]


def _wired(
    scores: list[float],
    *,
    embed_fail: bool = False,
    pool: list[dict] | None = None,
    floor: float = 0.0,
):
    """A Recall with every network/DB leg stubbed; returns (recall, captured_metric_rows).

    ``scores`` are the raw rerank scores handed back for the pool, by pool index (extra
    entries ignored when the pool is shorter). ``floor`` is the abstention floor (0 = off).
    """
    r = Recall(_NO_DB, "", config=RecallConfig(recall_floor=floor))
    p = _pool() if pool is None else pool
    r._ensure_embedder = lambda: _FakeEmbedder(fail=embed_fail)
    r._ensure_timeline = lambda: _FakeTimeline()
    r._search_bm25_episodes = lambda q, proj, limit, sid=None, allowed=None, own=None: list(p)
    r._search_vector_episodes = lambda emb, proj, limit, sid=None, allowed=None, own=None: []
    # BOTH halves of the web leg: _search_web_reranked fuses BM25 + vector, so stubbing
    # only the vector half leaves _search_bm25_web reaching for a real connection.
    r._search_vector_web = lambda emb, n: []
    r._search_bm25_web = lambda q, n: []
    r._search_kg = lambda *a, **k: ([], [])
    r._fetch_history_pairs_pg = lambda gid, uuids, cap: []
    r._surface_supersessions = lambda *a, **k: []
    r._episode_supersessions = lambda *a, **k: {}
    # Compaction serves compact passages of the top reranked eps (the real path mines
    # them from chunks). Stub returns them directly so the payload-invariance tests run
    # against a non-empty episode bucket; the empty-return case is exercised explicitly
    # by test_recall_no_passages_yields_empty_episode_bucket.
    r._compact_to_passages = lambda q, eps, n, **_k: [
        {"id": e["id"], "content": f"passage {e['id']}"} for e in eps[:n]
    ]
    r._increment_fact_retrieval_counts = lambda *a, **k: None
    r._increment_retrieval_counts = lambda ids: None
    r._rerank_pool_scored = lambda q, pl: [(i, scores[i]) for i in range(min(len(scores), len(pl)))]
    captured: list[dict] = []
    r._record_metrics = captured.append
    return r, captured


# --- _floor_shadow unit ----------------------------------------------------------


def _shadow(floor: float, rerank_top: float, *, emb_ok: bool = True) -> dict:
    env: dict = {"episodes": [], "n_echo_suppressed": 0}
    Recall(_NO_DB, "", config=RecallConfig(recall_floor=floor))._floor_shadow(
        env, rerank_top, emb_ok=emb_ok
    )
    return env


def test_floor_shadow_marks_below_floor():
    env = _shadow(0.58, 0.31)
    assert env["would_abstain"] is True
    assert env["floor"] == 0.58


def test_floor_shadow_strict_less_than():
    assert _shadow(0.58, 0.58) == {"episodes": [], "n_echo_suppressed": 0}  # exactly at the floor


def test_floor_shadow_guards():
    clean = {"episodes": [], "n_echo_suppressed": 0}
    assert _shadow(0.58, 0.0) == clean  # degraded/disabled/empty-pool sentinel
    assert _shadow(0.58, 0.31, emb_ok=False) == clean  # embedding failure
    assert _shadow(0.0, 0.31) == clean  # floor disabled


# --- recall(): marker + payload invariance ---------------------------------------


def test_recall_below_floor_drops_episodes_and_marks_envelope():
    r, captured = _wired([0.31, 0.22, 0.11, 0.05], floor=0.58)  # top 0.31 < floor
    out = r.recall("q")
    assert "episodes" not in out  # enforced: weak top -> episode bucket dropped
    (m,) = captured
    assert m["rerank_top_score"] == 0.31
    assert m["served_ids"]["would_abstain"] is True
    assert m["served_ids"]["floor"] == 0.58
    assert m["served_ids"]["episodes"] == []  # telemetry agrees: nothing served


def test_recall_no_passages_yields_empty_episode_bucket():
    # Compaction-gate contract (2026-07-23), isolated from the floor: scores here are ABOVE
    # the floor, so enforcement does not fire — the ONLY reason the bucket is empty is that
    # passage compaction produced nothing. recall() does NOT fall back to full episodes;
    # the bucket is omitted (empty container). Drill-down still returns full turns.
    r, captured = _wired([0.91, 0.80, 0.60, 0.59], floor=0.58)  # above floor -> floor does not fire
    r._compact_to_passages = lambda q, eps, n, **_k: []  # but no passage is produced
    out = r.recall("q")
    assert "episodes" not in out  # empty container from the compaction gate, no fallback
    assert captured[0]["served_ids"]["episodes"] == []  # telemetry agrees: nothing served
    assert "would_abstain" not in captured[0]["served_ids"]  # above floor -> no marker


def test_recall_floor_disabled_serves_episodes():
    # Floor 0 disables both the marker AND enforcement: weak scores still serve episodes.
    r, captured = _wired([0.31, 0.22, 0.11, 0.05])
    out = r.recall("q")
    assert out["episodes"]  # floor off -> no enforcement, episodes served
    assert "would_abstain" not in captured[0]["served_ids"]
    assert "floor" not in captured[0]["served_ids"]


def test_recall_above_floor_no_marker():
    r, captured = _wired([0.91, 0.80, 0.60, 0.59], floor=0.58)
    r.recall("q")
    (m,) = captured
    assert m["rerank_top_score"] == 0.91
    assert "would_abstain" not in m["served_ids"]
    assert "floor" not in m["served_ids"]


def test_recall_exactly_at_floor_no_marker():
    r, captured = _wired([0.58, 0.40, 0.30, 0.20], floor=0.58)
    r.recall("q")
    assert "would_abstain" not in captured[0]["served_ids"]


def test_recall_marker_uses_raw_pre_recency_top():
    # Recency reweighting can reorder/deflate scores, but the floor compares the RAW top:
    # an old top-scored doc decays (14d half-life) yet the marker must NOT appear, because
    # the raw 0.60 clears the 0.58 floor. rerank_top_score records the same raw value.
    old = (datetime.now(UTC) - timedelta(days=120)).isoformat()
    pool = _pool()
    for row in pool:
        row["created_at"] = old
    r, captured = _wired([0.60, 0.20, 0.10, 0.05], pool=pool, floor=0.58)
    r.recall("q")
    (m,) = captured
    assert m["rerank_top_score"] == 0.6
    assert "would_abstain" not in m["served_ids"]


def test_recall_rerank_disabled_no_marker():
    # SYNAPSE_RERANK_PROVIDER=none path: _ensure_reranker() resolves to None and the real
    # _rerank_pool_scored returns the all-0.0 RRF-order sentinel — no scores, no marker.
    r, captured = _wired([], floor=0.58)
    del r._rerank_pool_scored  # use the real method
    r._reranker = None  # what create_reranker() yields for provider "none"
    out = r.recall("q")
    assert out["episodes"]  # fusion-order serving still works
    (m,) = captured
    assert m["rerank_top_score"] == 0.0
    assert "would_abstain" not in m["served_ids"]


def test_recall_degraded_rerank_no_marker():
    # Rerank outage degrades to all-0.0 scores — never mark a degraded call.
    r, captured = _wired([0.0, 0.0, 0.0, 0.0], floor=0.58)
    r.recall("q")
    (m,) = captured
    assert m["rerank_top_score"] == 0.0
    assert "would_abstain" not in m["served_ids"]


def test_recall_embed_failure_no_marker():
    # Embedding down -> BM25-only retrieval. Real rerank scores may exist, but a weak top
    # under crippled retrieval is not abstention evidence — guard on emb_ok.
    r, captured = _wired([0.31, 0.22, 0.11, 0.05], embed_fail=True, floor=0.58)
    out = r.recall("q")
    assert out["episodes"]
    (m,) = captured
    assert m["emb_ok"] is False
    assert "would_abstain" not in m["served_ids"]


def test_recall_empty_pool_no_marker_no_crash():
    r, captured = _wired([], pool=[], floor=0.58)
    out = r.recall("q")
    assert "episodes" not in out
    (m,) = captured
    assert m["rerank_top_score"] == 0.0
    assert "would_abstain" not in m["served_ids"]


# --- recall_episodes(): rerank_top_score + marker ---------------------------------


def test_recall_episodes_records_rerank_top_score():
    r, captured = _wired([0.7712, 0.50, 0.40, 0.30], floor=0.58)
    out = r.recall_episodes("q")
    assert len(out["episodes"]) == 4
    (m,) = captured
    assert m["kind"] == "episodes"
    assert m["rerank_top_score"] == 0.7712  # the column recall() always recorded
    assert "would_abstain" not in m["served_ids"]


def test_recall_episodes_below_floor_marks_envelope():
    r, captured = _wired([0.31, 0.22, 0.11, 0.05], floor=0.58)
    out = r.recall_episodes("q")
    assert len(out["episodes"]) == 4  # served in full — shadow only
    (m,) = captured
    assert m["rerank_top_score"] == 0.31
    assert m["served_ids"]["would_abstain"] is True
    assert m["served_ids"]["floor"] == 0.58


def test_recall_episodes_payload_identical_with_and_without_floor():
    scores = [0.31, 0.22, 0.11, 0.05]
    r1, c1 = _wired(scores, floor=0.58)
    out_marked = r1.recall_episodes("q")  # floor ON — the marked leg
    r2, c2 = _wired(scores, floor=0.0)  # floor off
    out_plain = r2.recall_episodes("q")
    assert out_marked == out_plain  # the full response object — byte-identical payload
    assert c1[0]["served_ids"]["would_abstain"] is True  # marked leg really ran with floor on
    assert "would_abstain" not in c2[0]["served_ids"]
    assert "floor" not in c2[0]["served_ids"]


def test_recall_episodes_exactly_at_floor_no_marker():
    r, captured = _wired([0.58, 0.40, 0.30, 0.20], floor=0.58)
    r.recall_episodes("q")
    assert "would_abstain" not in captured[0]["served_ids"]


def test_recall_episodes_empty_pool_no_marker_no_crash():
    r, captured = _wired([], pool=[], floor=0.58)
    out = r.recall_episodes("q")
    assert out["episodes"] == []
    (m,) = captured
    assert m["rerank_top_score"] == 0.0
    assert "would_abstain" not in m["served_ids"]


# --- env parsing / defaults --------------------------------------------------------


def test_floor_defaults_off_and_reads_the_env(monkeypatch):
    # The threshold is reranker-specific and lives in .env, so the code default is off.
    monkeypatch.delenv("SYNAPSE_RECALL_FLOOR", raising=False)
    assert RecallConfig().recall_floor == 0.0
    monkeypatch.setenv("SYNAPSE_RECALL_FLOOR", "0.7")
    assert RecallConfig().recall_floor == 0.7
