"""RecallRerankMixin operations."""

from __future__ import annotations

import logging
from typing import Any

from ingestion import embedding as _embedding
from mcp_server.recall_query import bm25_best_window as _bm25_best_window
from mcp_server.recall_query import bm25_tokenize as _bm25_tokenize
from mcp_server.recall_query import normalize_whitespace as _norm_ws
from mcp_server.recall_query import query_shingles as _query_shingles
from mcp_server.recall_ranking import cutoff_k as _cutoff_k
from mcp_server.recall_ranking import recency_multiplier as _recency_multiplier
from mcp_server.recall_warnings import config_hint as _config_hint
from mcp_server.recall_warnings import error_brief as _err_brief
from mcp_server.recall_warnings import warn as _warn

logger = logging.getLogger(__name__)


class RecallRerankMixin:
    def _rerank_docs(self, query: str, pool: list[dict[str, Any]]) -> tuple[list[str], list[int]]:
        """Build the docs fed to the reranker + their pool owner index.

        Every item contributes its head (first _RERANK_DOC_CAP chars). With _RERANK_WINDOW on,
        items longer than the cap ALSO contribute their BM25-relevant window, so the caller can
        score the episode by max(head, window) and recover answers in the truncated tail (see the
        _RERANK_WINDOW note). ``owner[k]`` is the pool index doc ``k`` belongs to."""
        settings = self._settings()
        docs: list[str] = []
        owner: list[int] = []
        q_tokens = _bm25_tokenize(query) if settings._RERANK_WINDOW else []
        for i, c in enumerate(pool):
            content = c.get("content") or ""
            docs.append(content[: settings._RERANK_DOC_CAP])
            owner.append(i)
            if settings._RERANK_WINDOW and len(content) > settings._RERANK_DOC_CAP:
                docs.append(_bm25_best_window(content, q_tokens, settings._RERANK_DOC_CAP))
                owner.append(i)
        return docs, owner

    def _rerank_pool(self, query: str, pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Reorder a pooled candidate list by a Voyage cross-encoder. Degrades
        gracefully to the incoming RRF order if the reranker errors (rate limit,
        outage) — recall must never hard-fail on the rerank leg."""
        if len(pool) <= 1:
            return pool
        scored = self._rerank_pool_scored(query, pool)
        return [pool[i] for i, _ in scored] if scored else pool

    def _rerank_pool_scored(
        self, query: str, pool: list[dict[str, Any]]
    ) -> list[tuple[int, float]]:
        """Like _rerank_pool but returns (pool_index, relevance_score) pairs in
        rerank order, so callers can apply a relative score cutoff. Each episode is scored by the
        MAX over its rerank docs (head + optional BM25 window — see _rerank_docs). Degrades to the
        incoming RRF order with score 0.0 if the reranker errors — recall must never hard-fail on
        the rerank leg. A 0.0 top score signals the caller to fall back to fixed-k serving."""
        if not pool:
            return []
        if len(pool) == 1:
            return [(0, 1.0)]
        reranker = self._ensure_reranker()
        if reranker is None:
            # Rerank disabled (SYNAPSE_RERANK_PROVIDER=none): serve the incoming
            # fusion (RRF) order. Score 0.0 = the same "fixed-k" signal as the
            # degraded path below. Logged once at startup by create_reranker().
            return [(i, 0.0) for i in range(len(pool))]
        docs, owner = self._rerank_docs(query, pool)
        try:
            scored = reranker.rerank_scored(query, docs)
        except Exception as e:
            logger.warning("Scored rerank failed, using RRF order: %s", e)
            detail = _err_brief(e)
            backend = _embedding.rerank_provider()
            _warn(
                f"rerank failed ({backend}: {detail}): serving RRF fusion order, "
                f"result ranking is degraded."
                + _config_hint(detail, backend=backend, env_prefix="SYNAPSE_RERANK")
            )
            return [(i, 0.0) for i in range(len(pool))]
        best: dict[int, float] = {}
        for di, s in scored:
            if 0 <= di < len(owner):
                oi = owner[di]
                if oi not in best or s > best[oi]:
                    best[oi] = s
        return sorted(best.items(), key=lambda x: x[1], reverse=True)

    def _apply_rerank_recency(
        self, scored: list[tuple[int, float]], pool: list[dict[str, Any]]
    ) -> list[tuple[int, float]]:
        """Re-inject recency into the post-rerank ordering (see _RERANK_RECENCY_HALF_LIFE_DAYS).

        Multiplies each candidate's rerank score by _recency_multiplier(created_at) at the
        14-day half-life, FLOORED at _RERANK_RECENCY_FLOOR so old-but-relevant content is
        dampened at most 1/floor rather than annihilated, then re-sorts descending. Any
        downstream tau cutoff then operates on THESE recency-adjusted scores (that is the
        intended contract — recency is part of the final relevance, not a post-cut tweak).
        No-op (returns ``scored`` unchanged) when disabled (SYNAPSE_RERANK_RECENCY=0) or when
        the reranker degraded to RRF order (top score 0.0 — leave that fallback ordering
        untouched; a 0.0 top is the same signal _cutoff_k reads)."""
        settings = self._settings()
        if not settings._RERANK_RECENCY or not scored or scored[0][1] <= 0.0:
            return scored
        floor = settings._RERANK_RECENCY_FLOOR
        adjusted = [
            (
                i,
                s
                * max(
                    floor,
                    _recency_multiplier(
                        pool[i].get("created_at"), settings._RERANK_RECENCY_HALF_LIFE_DAYS
                    ),
                ),
            )
            for i, s in scored
        ]
        adjusted.sort(key=lambda x: x[1], reverse=True)
        return adjusted

    def _filter_query_echo(
        self, query: str, items: list[dict[str, Any]], need: int
    ) -> tuple[list[int], int]:
        """Echo suppression over a ranked list: (indices to keep, echoes dropped).

        A served episode whose content shares a long verbatim run with the query is an echo
        (compaction copy / re-ingested repeat), not recalled memory — drop it. Threshold:
        longest common substring >= min(60, max(40, len(query_norm)//2)) chars (the heuristic
        validated in the 2026-07-08 backtest), via difflib.SequenceMatcher (autojunk=False)
        over whitespace-collapsed lowercase, scanning at most _ECHO_CONTENT_CAP chars per doc.

        Cost containment — the pool is ~90 docs and SequenceMatcher is quadratic, so a naive
        full-pool scan costs seconds on long (800+ char) prompt-sized queries:
          - LAZY: walk in rank order and stop scanning once ``need`` survivors accumulate —
            items past that point can never be served, so they're kept unscanned.
          - Shingle pre-filter (_query_shingles): only docs containing one of the query's
            word shingles (C-level ``in``) reach the SequenceMatcher confirm (_echo_lcs_len);
            echoes are rare in the pool, so the confirm almost never runs.
        Keeps everything when disabled, when the query is too short for the threshold to be
        meaningful (< _ECHO_MIN_QUERY_LEN), or when the query yields no usable shingles."""
        settings = self._settings()
        keep_all = list(range(len(items)))
        if not settings._SUPPRESS_QUERY_ECHO or not items or need <= 0:
            return keep_all, 0
        q = _norm_ws(query)
        if len(q) < settings._ECHO_MIN_QUERY_LEN:
            return keep_all, 0
        shingles = _query_shingles(q)
        if not shingles:  # all-short-word query — the pre-filter can't attest, fail open
            return keep_all, 0
        thr = min(60, max(40, len(q) // 2))
        keep: list[int] = []
        dropped = 0
        for i, it in enumerate(items):
            if len(keep) >= need:
                keep.extend(range(i, len(items)))  # unscanned tail — can't be served
                break
            content = _norm_ws(it.get("content") or "")[: settings._ECHO_CONTENT_CAP]
            if (
                content
                and any(sh in content for sh in shingles)
                and self._echo_overlap(content, q) >= thr
            ):
                dropped += 1
                continue
            keep.append(i)
        return keep, dropped

    def _floor_by_rerank(
        self,
        query: str,
        items: list[dict[str, Any]],
        floor: float,
        *,
        text_key: str = "fact",
        keep_min: int = 0,
    ) -> list[dict[str, Any]]:
        """Shared per-item relevance gate: drop served items the cross-encoder scores below
        ``floor`` (off-topic). Backs _floor_facts and the notes leg — one of several recall()
        relevance gates (see the "Relevance gates" map by _RECALL_FACT_FLOOR).

        These buckets hold SHORT items (facts/events), scored at full length — unlike episodes
        (flat-high at full length), off-topic short items genuinely score low, so an absolute
        floor separates. ``keep_min`` > 0 backstops the bucket to its top-N when everything is
        subfloor (facts: never blank the bucket); keep_min=0 lets it serve [] (timeline/web:
        self-gate on intent). Degrades to keeping ALL items on reranker outage/disable — a floor
        must never hard-fail recall. Callers gate on ``floor`` > 0, so this runs only when armed."""
        settings = self._settings()
        reranker = self._ensure_reranker()
        if reranker is None:  # rerank disabled — no score signal, keep all
            return items
        texts = [(it.get(text_key) or "")[: settings._RERANK_DOC_CAP] for it in items]
        try:
            scored = reranker.rerank_scored(query, texts)
        except Exception as e:
            logger.warning("Relevance-floor rerank failed, keeping all items: %s", e)
            detail = _err_brief(e)
            backend = _embedding.rerank_provider()
            _warn(
                f"relevance-floor rerank failed ({backend}: {detail}): serving all items "
                f"unfiltered." + _config_hint(detail, backend=backend, env_prefix="SYNAPSE_RERANK")
            )
            return items
        kept = [items[i] for i, s in scored if s >= floor]
        if not kept and keep_min > 0:
            kept = [items[i] for i, _ in scored[:keep_min]]
        return kept

    def _floor_facts(self, query: str, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Fact relevance gate (_RECALL_FACT_FLOOR): drop off-topic facts, keep >=1 so the bucket
        is never blanked. Thin wrapper over _floor_by_rerank; caller gates on the floor > 0."""
        settings = self._settings()
        return self._floor_by_rerank(query, facts, settings._RECALL_FACT_FLOOR, keep_min=1)

    def _select_episodes(
        self, query: str, pool: list[dict[str, Any]], limit: int
    ) -> tuple[list[dict[str, Any]], int, float]:
        """Rerank `pool`, re-inject recency + suppress query echo, and pick what to serve.

        Returns ``(episodes, n_echo_suppressed, rerank_top)``. ``rerank_top`` is the RAW
        pre-recency top rerank score — the value recall_metrics.rerank_top_score records and
        the shadow abstention floor compares against — 0.0 when the pool is empty or the
        rerank degraded/was disabled. Default (SYNAPSE_EPISODE_CUTOFF_TAU <= 0):
        fixed top-`limit` in the recency-adjusted rerank order. With tau > 0: adaptive relative
        score-cutoff (keep score >= tau*top, clamped [_EPISODE_CUTOFF_MIN_K, _EPISODE_CUTOFF_MAX_K])
        over the recency-adjusted scores. Echoed episodes (the query quoting itself) are dropped
        BEFORE the final slice/cutoff so the next-ranked candidates backfill the freed slots. Always
        degrades to RRF order on rerank failure; never hard-fails."""
        settings = self._settings()
        if not pool:
            return [], 0, 0.0
        scored = self._rerank_pool_scored(query, pool)
        if not scored:
            return [], 0, 0.0
        rerank_top = scored[0][1]  # RAW top score (telemetry + shadow floor) — pre-recency
        degraded = rerank_top <= 0.0  # reranker down/disabled -> RRF order, fixed-k
        scored = self._apply_rerank_recency(scored, pool)  # no-op when disabled/degraded
        ranked = [pool[i] for i, _ in scored]
        ranked_scores = [s for _, s in scored]
        fixed_k = settings._EPISODE_CUTOFF_TAU <= 0 or degraded
        # Echo suppression's lazy scan only needs enough survivors to cover the serve
        # window: `limit` on the fixed-k path, at most _EPISODE_CUTOFF_MAX_K on the tau
        # path (k is clamped there, so ranked[:k] never reaches past the scanned prefix).
        need = limit if fixed_k else max(limit, settings._EPISODE_CUTOFF_MAX_K)
        keep, n_echo = self._filter_query_echo(query, ranked, need)
        if n_echo:
            ranked = [ranked[i] for i in keep]
            ranked_scores = [ranked_scores[i] for i in keep]
        if fixed_k:
            return ranked[:limit], n_echo, rerank_top
        k = _cutoff_k(
            ranked_scores,
            settings._EPISODE_CUTOFF_TAU,
            settings._EPISODE_CUTOFF_MIN_K,
            settings._EPISODE_CUTOFF_MAX_K,
        )
        return ranked[:k], n_echo, rerank_top
