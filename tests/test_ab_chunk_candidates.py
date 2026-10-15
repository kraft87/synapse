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
    return {"query": query, "scores": scores, "pool_ms": 10.0, "total_ms": 20.0, "notices": []}


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


def test_only_the_on_arm_reaches_the_chunk_sources():
    r = Recall(_NO_DB, "")
    r._ensure_embedder = lambda: _Emb()  # type: ignore[method-assign]
    r._search_bm25_episodes = lambda q, proj, limit, sid=None, allowed=None, own=None: [  # type: ignore[method-assign]
        _ep(1),
        _ep(2),
    ]
    r._search_vector_episodes = lambda emb, proj, limit, sid=None, allowed=None, own=None: []  # type: ignore[method-assign]
    r._search_chunks_bm25 = lambda q, proj, allowed=None, own=None: [_ep(3)]  # type: ignore[method-assign]
    r._search_chunks_vector = lambda emb, proj, allowed=None, own=None: []  # type: ignore[method-assign]
    r._rerank_pool_scored = lambda q, pool: [(i, 0.9 - 0.1 * i) for i in range(len(pool))]  # type: ignore[method-assign]
    golden = {"query": "where is the gizmo", "gold": ["e:3"], "project": None}

    off = script.run_arm(r, golden, False, (5, 10))
    on = script.run_arm(r, golden, True, (5, 10))
    off_again = script.run_arm(r, golden, False, (5, 10))

    assert off["scores"]["pool_recall"] == 0.0 and off["scores"]["hit@10"] == 0.0
    assert on["scores"]["pool_recall"] == 1.0 and on["scores"]["hit@5"] == 1.0
    assert 0.0 < on["scores"]["mrr"] <= 1.0
    assert off_again["scores"] == off["scores"]  # the setting is per run, not left switched on
    assert on["pool_ms"] <= on["total_ms"]
