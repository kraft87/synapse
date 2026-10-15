#!/usr/bin/env python3
"""A/B the chunk candidate source (SYNAPSE_RECALL_CHUNK_CANDIDATES) on a set of goldens.

recall() builds a pool of candidate episodes (keyword and embedding search over single turns),
and the cross-encoder reranks it. With the setting on, the top chunks (four consecutive turns)
by keyword and by embedding also name their episodes into that pool. This runs each golden
query through the real pool build and rerank twice, off and on, and reports whether the extra
candidates put a gold turn into the pool and whether it ends up ranked higher:

    pool recall   share of a query's gold turns that are in the pool before the rerank
    hit@k         a gold turn is among the first k after the rerank (k from --k)
    MRR           mean of 1 / rank of the first gold turn after the rerank
    ms            wall time of the pool build, and of pool + rerank (median and p95)

and which queries gained or lost a gold turn in the pool or in the top k. Nothing is
written: no recall_metrics row, no retrieval-count bump. It calls the real Voyage embedder
and reranker (VOYAGE_API_KEY, or the SYNAPSE_EMBED_* / SYNAPSE_RERANK_* provider settings
the server uses) and the database, so it costs a rerank call per query per arm. Both clients
keep the batch patience (a Voyage 429 is ridden out, not failed fast as in recall), and a
query whose rerank still failed (RRF order, every score 0.0) is counted per arm and makes
the script exit 1: that arm's numbers say nothing about the rerank.

Goldens: one JSON object per line,

    {"query": "what port did we pick for the proxy", "gold": ["e:1234", "e:1240"], "project": "synapse"}

``gold`` lists the episodes an answer comes from (any one is a hit; ``e:`` prefix optional);
``project`` is optional. A line without a query or gold is skipped and counted.

The vector half of the chunk source needs the chunks vector index (see its migration); built
without it, that half says so and serves nothing, and this script prints the notice, so a run
that measured only the keyword half cannot pass for the whole source. The two arms run in
alternating order per query, so the embedding cache does not favour one of them.

Usage:
    scripts/ab_chunk_candidates.py GOLDENS.jsonl [--db-url DSN] [--limit N] [--k 5 10]
                                   [--json OUT.json]

The DSN comes from --db-url, else $SYNAPSE_DB_URL. The loser of the A/B gets deleted: if the
setting loses, the chunk sources in mcp_server/recall_episode_search.py and the optional
chunks vector index go; if it wins, the setting goes and the sources stay on.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_ARMS = ("off", "on")
_DEFAULT_KS = (5, 10)


def load_goldens(lines: list[str]) -> tuple[list[dict[str, Any]], int]:
    """``(goldens, n_skipped)`` from JSONL lines; blank lines are ignored, a line without a
    query or any gold turn is skipped."""
    goldens: list[dict[str, Any]] = []
    skipped = 0
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        gold = [_episode_id(g) for g in row.get("gold") or []]
        if not str(row.get("query") or "").strip() or not gold:
            skipped += 1
            continue
        goldens.append({"query": row["query"], "gold": gold, "project": row.get("project")})
    return goldens, skipped


def _episode_id(value: Any) -> str:
    text = str(value).strip()
    return text if text.startswith("e:") else f"e:{text}"


def score_query(
    gold: list[str], pool: list[str], ranked: list[str], ks: tuple[int, ...]
) -> dict[str, float]:
    """One query's numbers for one arm: ``pool`` and ``ranked`` are episode ids before and
    after the rerank, best first."""
    wanted = set(gold)
    first = next((rank for rank, eid in enumerate(ranked, start=1) if eid in wanted), None)
    scores = {
        "pool_recall": len(wanted & set(pool)) / len(wanted),
        "mrr": 0.0 if first is None else 1.0 / first,
    }
    for k in ks:
        scores[f"hit@{k}"] = float(first is not None and first <= k)
    return scores


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Means of every score, and the median and p95 of the two timings, over one arm's rows."""
    out = {key: statistics.fmean(row["scores"][key] for row in rows) for key in rows[0]["scores"]}
    for key in ("pool_ms", "total_ms"):
        times = [row[key] for row in rows]
        out[f"{key}_p50"] = statistics.median(times)
        out[f"{key}_p95"] = _percentile(times, 0.95)
    return out


def movers(
    off: list[dict[str, Any]], on: list[dict[str, Any]], key: str
) -> tuple[list[str], list[str]]:
    """``(gained, lost)``: queries whose ``key`` score is higher / lower with the setting on."""
    gained, lost = [], []
    for before, after in zip(off, on, strict=True):
        delta = after["scores"][key] - before["scores"][key]
        if delta > 0:
            gained.append(before["query"])
        elif delta < 0:
            lost.append(before["query"])
    return gained, lost


def run_arm(
    engine: Any, golden: dict[str, Any], chunks_on: bool, ks: tuple[int, ...]
) -> dict[str, Any]:
    """One golden through the real pool build and rerank with the setting on or off."""
    from mcp_server.recall_warnings import warn_sink

    query, project = golden["query"], golden["project"]
    engine._config = dataclasses.replace(engine._config, chunk_candidates=chunks_on)
    notices: list[str] = []
    started = time.perf_counter()
    with warn_sink(notices):
        pool = engine._join_episode_pool(engine._start_episode_legs(query, project))
        pool_ms = (time.perf_counter() - started) * 1000.0
        scored = engine._rerank_pool_scored(query, pool.items)
        ranked, _top, _n_echo = engine._rank_episodes(query, pool.items, scored, max(ks))
    total_ms = (time.perf_counter() - started) * 1000.0
    pool_ids = [item["id"] for item in pool.items]
    ranked_ids = [item["id"] for item in ranked]
    return {
        "query": query,
        "scores": score_query(golden["gold"], pool_ids, ranked_ids, ks),
        "pool_ms": pool_ms,
        "total_ms": total_ms,
        "notices": notices,
        # The reranker erroring (or off) serves the pool in RRF order with every score 0.0.
        "degraded": bool(scored) and max(score for _, score in scored) == 0.0,
    }


def report(rows: dict[str, list[dict[str, Any]]], n_skipped: int, ks: tuple[int, ...]) -> list[str]:
    off, on = summarize(rows["off"]), summarize(rows["on"])
    lines = [f"{len(rows['off'])} goldens ({n_skipped} skipped)", ""]
    lines.append(f"{'':16}{'off':>10}{'on':>10}{'delta':>10}")
    for key in ("pool_recall", *(f"hit@{k}" for k in ks), "mrr"):
        lines.append(f"{key:16}{off[key]:>10.3f}{on[key]:>10.3f}{on[key] - off[key]:>+10.3f}")
    for key in ("pool_ms_p50", "pool_ms_p95", "total_ms_p50", "total_ms_p95"):
        lines.append(f"{key:16}{off[key]:>10.0f}{on[key]:>10.0f}{on[key] - off[key]:>+10.0f}")
    for key, what in (("pool_recall", "a gold turn in the pool"), (f"hit@{ks[0]}", f"hit@{ks[0]}")):
        gained, lost = movers(rows["off"], rows["on"], key)
        lines += ["", f"gained {what} with chunks on: {len(gained)}", *(f"  {q}" for q in gained)]
        lines += [f"lost {what} with chunks on: {len(lost)}", *(f"  {q}" for q in lost)]
    for arm in _ARMS:
        notices = sorted({n for row in rows[arm] for n in row["notices"]})
        if notices:
            lines += [
                "",
                f"notices from the '{arm}' arm (a source that did not run is not measured):",
            ]
            lines += [f"  {n}" for n in notices]
    degraded = {arm: sum(row["degraded"] for row in rows[arm]) for arm in _ARMS}
    lines += [
        "",
        "queries with a degraded rerank (RRF order, score 0.0): "
        + ", ".join(f"{arm} {n}" for arm, n in degraded.items()),
    ]
    return lines


def build_engine(db_url: str, voyage_key: str) -> Any:
    """A Recall engine whose embedder and reranker keep the batch patience: this is offline
    measurement, so a Voyage 429 is ridden out instead of failing fast into RRF order."""
    from ingestion import embedding
    from mcp_server.recall import Recall

    engine = Recall(db_url, voyage_key)
    engine._embedder = embedding.create_embedder(
        voyage_api_key=voyage_key, db_url=db_url, interactive=False
    )
    engine._reranker = embedding.create_reranker(voyage_api_key=voyage_key, interactive=False)
    return engine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("goldens", type=Path, help="JSONL: query, gold episode ids, project?")
    parser.add_argument("--db-url", default=os.environ.get("SYNAPSE_DB_URL"))
    parser.add_argument("--limit", type=int, default=0, help="only the first N goldens")
    parser.add_argument(
        "--k", type=int, nargs="+", default=list(_DEFAULT_KS), help="hit@k cut-offs"
    )
    parser.add_argument("--json", type=Path, help="also write every query's numbers here")
    args = parser.parse_args(argv)
    if not args.db_url:
        parser.error("no database: pass --db-url or set SYNAPSE_DB_URL")
    goldens, n_skipped = load_goldens(args.goldens.read_text().splitlines())
    if args.limit:
        goldens = goldens[: args.limit]
    if not goldens:
        parser.error("no usable goldens (each line needs a query and gold episode ids)")
    ks = tuple(sorted(set(args.k)))

    engine = build_engine(args.db_url, os.environ.get("VOYAGE_API_KEY", ""))
    rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in _ARMS}
    for index, golden in enumerate(goldens):
        for arm in _ARMS if index % 2 == 0 else reversed(_ARMS):
            rows[arm].append(run_arm(engine, golden, arm == "on", ks))
    print("\n".join(report(rows, n_skipped, ks)))
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
    # A degraded rerank means the arm's ranking is the RRF order, not the cross-encoder's.
    return 1 if any(row["degraded"] for arm_rows in rows.values() for row in arm_rows) else 0


if __name__ == "__main__":
    sys.exit(main())
