"""RecallConfig: every tunable of the recall engine, in one place.

Built once per engine (``Recall(config=...)``); a benchmark or test overrides a field with
``dataclasses.replace``. Fields read from the environment are the deployment's own
(a score threshold belongs to the reranker in use, a distance gate to the embedder); the
rest are the measured policy values, fixed in code.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field

from ingestion import embedding as _embedding


def _env_float(name: str, default: float) -> Callable[[], float]:
    return lambda: float(os.getenv(name) or default)


def _env_int(name: str, default: int) -> Callable[[], int]:
    return lambda: int(os.getenv(name) or default)


@dataclass(frozen=True)
class RecallConfig:
    kg_owner: str = field(default_factory=lambda: os.environ.get("SYNAPSE_KG_OWNER_ID", "default"))
    # Embedding width for every vector/halfvec SQL cast. Must match the width the schema was
    # provisioned with or the halfvec HNSW index expressions don't apply (validated against
    # synapse_meta, see ingestion/embedding.py).
    embed_dims: int = field(default_factory=_embedding.embed_dims)

    # --- Episode pool -----------------------------------------------------------------------
    # Per-leg fetch depth and the fused pool the cross-encoder reranks (WIN1, 2026-06-03): exact-
    # fact misses were pure truncation, golds ranking 16-88 in a leg never entered a 15-deep
    # pool. Deep fetch plus a wide-pool rerank lifted answerability 90.5% -> 95.2% (pool 50:
    # 92.9%). Pool size is a quality/latency knob, not a truncation risk: Voyage's ~32k-token
    # limit is per query+doc pair, not per request.
    episode_fetch: int = 100
    episode_rerank_pool: int = 100
    # Chunks as an extra candidate source for the episode pool (off: the A/B in
    # scripts/ab_chunk_candidates.py decides). Chunks (4 consecutive turns, advanced 3 at a
    # time) are embedded and BM25-indexed at ingest, but the episode legs search single turns,
    # so a turn whose own text misses the question's words while its neighbours carry them never
    # entered the pool. With this on, the top chunks by keyword and by embedding name their
    # episodes, which join the pool through the same project and surface predicates as the
    # episode legs. Chunks are never served, only their episodes. The vector half needs the
    # optional chunks vector index (without it that half says so in `warnings`).
    chunk_candidates: bool = field(
        default_factory=lambda: os.getenv("SYNAPSE_RECALL_CHUNK_CANDIDATES", "0") == "1"
    )

    # --- Rerank and ranking -----------------------------------------------------------------
    # Per-doc char cap (~1k tokens) sent to the reranker, so no single doc is silently truncated.
    rerank_doc_cap: int = 4000
    # The cross-encoder is recency-blind: an old *definitive* statement out-scores a newer
    # *transition* one, so its scores are multiplied by a recency decay afterwards. Shorter
    # half-life than the pre-rerank fusion (the corpus churns within a month); the floor keeps
    # a ~3-month-old episode at 1/4 of its score instead of ~0.01x, so a query that asks for old
    # content still gets it.
    rerank_recency_half_life_days: int = 14
    rerank_recency_floor: float = field(
        default_factory=_env_float("SYNAPSE_RERANK_RECENCY_FLOOR", 0.25)
    )
    # Query-echo suppression: a served episode containing a long verbatim run of the query is the
    # prompt quoting itself (74% of top-5s in the 2026-07-08 backtest), so it is dropped and the
    # next-ranked backfills. Skipped for queries shorter than this (the overlap threshold means
    # nothing); at most this many chars of a doc are scanned (bounds SequenceMatcher work).
    echo_min_query_len: int = 40
    echo_content_cap: int = 8000
    # Abstention floor on the RAW (pre-recency) top rerank score of the episodes that get mined:
    # in (0, floor), recall() serves no episode bucket (low relevance costs fewer tokens, not the
    # least-bad six) and served_ids records {"would_abstain": true, "floor": f}. 0 = off.
    # The threshold is a property of the RERANKER, not of recall: scores are not comparable across
    # models, and a floor replayed on another backend blanked 39-56% of episode buckets on the
    # LME A/B set (2026-09-10). Each examples/env preset sets the value calibrated for its
    # reranker (0.60 for rerank-3-lite). recall_episodes() never enforces: the caller asked for turns.
    recall_floor: float = field(default_factory=_env_float("SYNAPSE_RECALL_FLOOR", 0.0))

    # --- recall() episode bucket (passages) -------------------------------------------------
    # recall() serves the most query-relevant PASSAGES (markdown chunks) of the top reranked
    # episodes, picked by a second cross-encoder rerank: ~1/4 the tokens of whole episodes at
    # near-baseline answer survival (scripts/passage_bench_v*, 2026-06-26). Five episodes
    # (episode_count_sweep, 2026-06-04: broad synthesis peaks at 5); mining the top 20 recovers
    # answers the reranker ranks at 11-20 (2026-07-23); at most 80 chunks are reranked, shared
    # across the episodes. 6 passages is the bucket's token budget.
    recall_episode_limit: int = 5
    recall_passage_src_k: int = 20
    recall_passage_cand: int = 80
    recall_passage_n: int = field(default_factory=_env_int("SYNAPSE_RECALL_PASSAGE_N", 6))

    # --- Facts ------------------------------------------------------------------------------
    # Raised 5 -> 12 (2026-06-03): the KG owns relational questions, and serving more of its
    # existing facts lifted relational answer quality 0.746 -> 0.826 for ~200 tokens.
    fact_limit: int = 12
    superseded_limit: int = 2
    # Supersession surface: a query that matches a now-INVALID fact still gets the CURRENT answer,
    # through the fact's precise successor link (invalidated_by). Distance-gated so only on-topic
    # superseded facts pull their correction in; at most `sup_candidates` nearest edges are
    # considered and `sup_limit` successors added beyond fact_limit.
    sup_candidates: int = 10
    sup_limit: int = 3
    sup_max_dist: float = field(default_factory=_env_float("SYNAPSE_SUPERSEDE_MAX_DIST", 0.45))
    # A successor link is believed only when the successor is about the same thing as the fact it
    # retired (2026-10-09): Stage-6b verdicts raised links past 0.35 apart from 6% to 58%, and
    # every such pair read as junk. Calibrated on voyage-4-large; recalibrate with the embedder.
    sup_link_max_dist: float = field(
        default_factory=_env_float("SYNAPSE_SUPERSEDE_LINK_MAX_DIST", 0.35)
    )

    # --- Web (fed by the out-of-repo capture cron) ------------------------------------------
    # Fused BM25 + vector pool, cross-encoder rerank, then an absolute floor with NO keep-one
    # backstop: the bucket may serve zero, which self-gates web on intent. 0.60 was calibrated
    # on rerank-2.5-lite (all 41 noise ids scored <= 0.559, the one helpful id 0.914; n_helpful
    # = 1) and is a property of the reranker like recall_floor. 0 = off.
    web_limit: int = 3
    web_fetch: int = 50
    web_rerank_pool: int = 60
    web_floor: float = field(default_factory=_env_float("SYNAPSE_RECALL_WEB_FLOOR", 0.60))

    # --- Notes ------------------------------------------------------------------------------
    # Hook-embedding KNN over the live notes (board scoping: global types + the caller's
    # project), cross-encoder floor on "hook — body" text, top notes_limit served. The floor
    # self-gates like web (an all-subfloor result serves no bucket), and is reranker-calibrated
    # the same way (0.60 on rerank-2.5-lite; 0 = off). A note already on the board may be served
    # again. Bodies are capped; fetch("n:N") returns the full note.
    notes_limit: int = 3
    notes_fetch: int = 24
    notes_body_cap: int = 700
    notes_floor: float = field(default_factory=_env_float("SYNAPSE_RECALL_NOTES_FLOOR", 0.60))
