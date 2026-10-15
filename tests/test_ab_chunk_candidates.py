"""scripts/ab_chunk_candidates.py: the scoring, the golden loader, and one query through a
real Recall engine whose legs are stubbed (no database, no Voyage).

What is pinned: the loader skips lines it cannot score, the numbers mean what the header
says, and the "on" arm is the only one that reaches the chunk sources.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from mcp_server.recall import Recall

_SPEC = importlib.util.spec_from_file_location(
    "ab_chunk_candidates",
    Path(__file__).resolve().parent.parent / "scripts" / "ab_chunk_candidates.py",
)
assert _SPEC and _SPEC.loader
script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(script)

_NO_DB = "postgresql://synapse@127.0.0.1:1/nonexistent"


def test_goldens_skip_lines_without_a_query_or_a_gold_turn():
    goldens, skipped = script.load_goldens(
        [
            '{"query": "port of the proxy", "gold": [12, "e:13"], "project": "synapse"}',
            "",
            '{"query": "no gold here"}',
            '{"query": "  ", "gold": ["e:1"]}',
        ]
    )
    assert goldens == [
        {"query": "port of the proxy", "gold": ["e:12", "e:13"], "project": "synapse"}
    ]
    assert skipped == 2


def test_a_query_is_scored_on_its_pool_and_its_ranking():
    scores = script.score_query(
        gold=["e:1", "e:2"], pool=["e:9", "e:1", "e:8"], ranked=["e:9", "e:8", "e:1"], ks=(2, 3)
    )
    assert scores == {"pool_recall": 0.5, "mrr": 1 / 3, "hit@2": 0.0, "hit@3": 1.0}
    missed = script.score_query(["e:1"], ["e:9"], ["e:9"], (5,))
    assert missed == {"pool_recall": 0.0, "mrr": 0.0, "hit@5": 0.0}


def _row(query: str, **scores: float) -> dict[str, Any]:
    return {
        "query": query,
        "scores": scores,
        "pool_ms": 10.0,
        "total_ms": 20.0,
        "notices": [],
        "degraded": False,
    }


def test_movers_are_the_queries_whose_score_changed():
    off = [_row("a", pool_recall=0.0), _row("b", pool_recall=1.0), _row("c", pool_recall=0.5)]
    on = [_row("a", pool_recall=1.0), _row("b", pool_recall=0.5), _row("c", pool_recall=0.5)]
    assert script.movers(off, on, "pool_recall") == (["a"], ["b"])


def test_the_report_carries_the_deltas_and_the_notices():
    rows = {
        "off": [_row("a", pool_recall=0.0, mrr=0.0, **{"hit@5": 0.0})],
        "on": [
            {
                **_row("a", pool_recall=1.0, mrr=0.5, **{"hit@5": 1.0}),
                "notices": ["chunk vector source skipped"],
            }
        ],
    }
    text = "\n".join(script.report(rows, n_skipped=1, ks=(5,)))
    assert "1 goldens (1 skipped)" in text
    assert "pool_recall" in text and "+1.000" in text
    assert "gained a gold turn in the pool with chunks on: 1" in text
    assert "chunk vector source skipped" in text


def test_the_report_labels_the_notices_of_both_arms_and_counts_degraded_reranks():
    rows = {
        "off": [
            {
                **_row("a", pool_recall=0.0, mrr=0.0, **{"hit@5": 0.0}),
                "notices": ["rerank failed (voyage: 429): serving RRF fusion order"],
                "degraded": True,
            }
        ],
        "on": [_row("a", pool_recall=1.0, mrr=0.5, **{"hit@5": 1.0})],
    }
    text = "\n".join(script.report(rows, n_skipped=0, ks=(5,)))
    assert "notices from the 'off' arm" in text
    assert "rerank failed (voyage: 429)" in text
    assert "notices from the 'on' arm" not in text  # that arm had none
    assert "queries with a degraded rerank (RRF order, score 0.0): off 1, on 0" in text


def test_the_engine_is_built_with_batch_patience():
    seen: dict[str, Any] = {}

    def fake_embedder(*_a, **kw):
        seen["embedder"] = kw
        return _Emb()

    def fake_reranker(*_a, **kw):
        seen["reranker"] = kw
        return None

    import ingestion.embedding as embedding

    original = embedding.create_embedder, embedding.create_reranker
    embedding.create_embedder, embedding.create_reranker = fake_embedder, fake_reranker
    try:
        engine = script.build_engine(_NO_DB, "key")
    finally:
        embedding.create_embedder, embedding.create_reranker = original
    # Offline measurement: a Voyage 429 is ridden out, not silently turned into RRF order.
    assert seen["embedder"]["interactive"] is False
    assert seen["reranker"]["interactive"] is False
    assert engine._ensure_reranker() is None


class _Emb:
    def embed(self, texts, task="query"):
        return [[0.0] * 4 for _ in texts]


def _ep(n: int) -> dict[str, Any]:
    return {
        "id": f"e:{n}",
        "content": f"turn {n}",
        "doc_type": "episode",
        "project": None,
        "created_at": None,
        "retrieval_count": 0,
        "session_id": f"s{n}",
    }


def _engine(rerank: Any, chunk_hits: list[dict[str, Any]]) -> Recall:
    r = Recall(_NO_DB, "")
    r._ensure_embedder = lambda: _Emb()  # type: ignore[method-assign]
    r._search_bm25_episodes = lambda q, proj, limit, sid=None, allowed=None, own=None: [  # type: ignore[method-assign]
        _ep(1),
        _ep(2),
    ]
    r._search_vector_episodes = lambda emb, proj, limit, sid=None, allowed=None, own=None: []  # type: ignore[method-assign]
    r._search_chunks_bm25 = lambda q, proj, allowed=None, own=None: chunk_hits  # type: ignore[method-assign]
    r._search_chunks_vector = lambda emb, proj, allowed=None, own=None: []  # type: ignore[method-assign]
    r._rerank_pool_scored = rerank  # type: ignore[method-assign]
    return r


def _graded(q: str, pool: list[dict[str, Any]]) -> list[tuple[int, float]]:
    return [(i, 0.9 - 0.1 * i) for i in range(len(pool))]


def _degraded(q: str, pool: list[dict[str, Any]]) -> list[tuple[int, float]]:
    """What _rerank_pool_scored returns when the reranker errors: RRF order, every score 0.0."""
    return [(i, 0.0) for i in range(len(pool))]


def test_only_the_on_arm_reaches_the_chunk_sources():
    r = _engine(_graded, [_ep(3)])
    golden = {"query": "where is the gizmo", "gold": ["e:3"], "project": None}

    off = script.run_arm(r, golden, False, (5, 10))
    on = script.run_arm(r, golden, True, (5, 10))
    off_again = script.run_arm(r, golden, False, (5, 10))

    assert off["scores"]["pool_recall"] == 0.0 and off["scores"]["hit@10"] == 0.0
    assert on["scores"]["pool_recall"] == 1.0 and on["scores"]["hit@5"] == 1.0
    assert 0.0 < on["scores"]["mrr"] <= 1.0
    assert off_again["scores"] == off["scores"]  # the setting is per run, not left switched on
    assert on["pool_ms"] <= on["total_ms"]
    assert not off["degraded"] and not on["degraded"]


def test_an_arm_whose_rerank_degraded_is_flagged():
    golden = {"query": "where is the gizmo", "gold": ["e:1"], "project": None}
    assert script.run_arm(_engine(_degraded, []), golden, False, (5,))["degraded"] is True
    assert script.run_arm(_engine(_graded, []), golden, False, (5,))["degraded"] is False


def test_the_script_exits_nonzero_when_any_rerank_degraded(tmp_path, monkeypatch, capsys):
    goldens = tmp_path / "goldens.jsonl"
    goldens.write_text('{"query": "where is the gizmo", "gold": ["e:1"]}\n')
    argv = [str(goldens), "--db-url", _NO_DB, "--k", "5"]

    monkeypatch.setattr(script, "build_engine", lambda *_a: _engine(_graded, []))
    assert script.main(argv) == 0
    monkeypatch.setattr(script, "build_engine", lambda *_a: _engine(_degraded, []))
    assert script.main(argv) == 1
    assert "degraded rerank (RRF order, score 0.0): off 1, on 1" in capsys.readouterr().out
