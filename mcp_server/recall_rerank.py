"""RecallRerankMixin operations."""

from __future__ import annotations

import logging
from typing import Any

from ingestion import embedding as _embedding
from mcp_server.recall_query import bm25_best_window as _bm25_best_window
from mcp_server.recall_query import bm25_tokenize as _bm25_tokenize
from mcp_server.recall_query import normalize_whitespace as _norm_ws
from mcp_server.recall_query import query_shingles as _query_shingles
from mcp_server.recall_ranking import recency_multiplier as _recency_multiplier
from mcp_server.recall_warnings import config_hint as _config_hint
from mcp_server.recall_warnings import error_brief as _err_brief
from mcp_server.recall_warnings import warn as _warn

logger = logging.getLogger(__name__)


class RecallRerankMixin:
    def _rerank_docs(self, query: str, pool: list[dict[str, Any]]) -> tuple[list[str], list[int]]:
        """Build the docs fed to the reranker + their pool owner index.

        Every item contributes its head (first ``rerank_doc_cap`` chars). Items longer than the
        cap ALSO contribute their BM25-relevant window, so the caller can score the episode by
        max(head, window) and recover answers in the truncated tail: the cross-encoder only sees
        each doc's head, and on a back-half golden that capped retr@5 at 0.139 against a 0.917
        pool ceiling (scripts/passage_bench_v*, 2026-06-26). Window selection is pure-Python BM25
        (~ms), and max() can only RAISE an episode's score, so short episodes are unaffected
        (tail retr@5 0.139 -> 0.889, natural 0.929 -> 0.929). ``owner[k]`` is the pool index doc
        ``k`` belongs to."""
        cap = self._config.rerank_doc_cap
        docs: list[str] = []
        owner: list[int] = []
        q_tokens = _bm25_tokenize(query)
        for i, c in enumerate(pool):
            content = c.get("content") or ""
            docs.append(content[:cap])
            owner.append(i)
            if len(content) > cap:
                docs.append(_bm25_best_window(content, q_tokens, cap))
                owner.append(i)
        return docs, owner

    def _rerank_pool_scored(
        self, query: str, pool: list[dict[str, Any]]
    ) -> list[tuple[int, float]]:
        """Rerank a pooled candidate list with the cross-encoder: (pool_index, relevance_score)
        pairs in rerank order. Each episode is scored by the MAX over its rerank docs (head + optional BM25 window — see _rerank_docs). Degrades to the
        incoming RRF order with score 0.0 if the reranker errors — recall must never hard-fail on
        the rerank leg. A 0.0 top score means "no real scores"."""
        if not pool:
            return []
        if len(pool) == 1:
            return [(0, 1.0)]
        reranker = self._ensure_reranker()
        if reranker is None:
            # Rerank disabled (SYNAPSE_RERANK_PROVIDER=none): serve the incoming
            # fusion (RRF) order. Score 0.0 = the same "no real scores" signal as the
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
        """Re-inject recency into the post-rerank ordering (see RecallConfig.rerank_recency_*).

        Multiplies each candidate's rerank score by the recency decay of its created_at at the
        14-day half-life, FLOORED at ``rerank_recency_floor`` so old-but-relevant content is
        dampened at most 1/floor rather than annihilated, then re-sorts descending. No-op
        (returns ``scored`` unchanged) when the reranker degraded to RRF order (top score 0.0):
        that fallback ordering stays untouched."""
        if not scored or scored[0][1] <= 0.0:
            return scored
        cfg = self._config
        adjusted = [
            (
                i,
                s
                * max(
                    cfg.rerank_recency_floor,
                    _recency_multiplier(
                        pool[i].get("created_at"), cfg.rerank_recency_half_life_days
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
        over whitespace-collapsed lowercase, scanning at most ``echo_content_cap`` chars per doc.

        Cost containment — the pool is ~90 docs and SequenceMatcher is quadratic, so a naive
        full-pool scan costs seconds on long (800+ char) prompt-sized queries:
          - LAZY: walk in rank order and stop scanning once ``need`` survivors accumulate —
            items past that point can never be served, so they're kept unscanned.
          - Shingle pre-filter (_query_shingles): only docs containing one of the query's
            word shingles (C-level ``in``) reach the SequenceMatcher confirm (_echo_lcs_len);
            echoes are rare in the pool, so the confirm almost never runs.
        Keeps everything when the query is too short for the threshold to be meaningful
        (< ``echo_min_query_len``), or when the query yields no usable shingles."""
        cfg = self._config
        keep_all = list(range(len(items)))
        if not items or need <= 0:
            return keep_all, 0
        q = _norm_ws(query)
        if len(q) < cfg.echo_min_query_len:
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
            content = _norm_ws(it.get("content") or "")[: cfg.echo_content_cap]
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
        self, query: str, items: list[dict[str, Any]], floor: float, *, text_key: str = "fact"
    ) -> list[dict[str, Any]]:
        """Per-item relevance gate for a bucket of SHORT items (the notes leg): drop the items the
        cross-encoder scores below ``floor``. Short items are scored at full length, so off-topic
        ones genuinely score low and an absolute floor separates them (episodes score flat-high,
        so theirs is a whole-bucket top-score gate instead; the web bucket applies its floor
        inline on the pool it already reranked). The bucket may serve [] (self-gates on intent).
        Degrades to keeping ALL items on reranker outage/disable: a floor must never hard-fail
        recall. Callers gate on ``floor`` > 0, so this runs only when armed."""
        reranker = self._ensure_reranker()
        if reranker is None:  # rerank disabled — no score signal, keep all
            return items
        cap = self._config.rerank_doc_cap
        texts = [(it.get(text_key) or "")[:cap] for it in items]
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
        return [items[i] for i, s in scored if s >= floor]

    def _rank_episodes(
        self,
        query: str,
        pool: list[dict[str, Any]],
        scored: list[tuple[int, float]],
        need: int,
    ) -> tuple[list[dict[str, Any]], float, int]:
        """The ranking contract shared by recall() and recall_episodes(): re-inject recency
        into the rerank order and suppress query echoes.

        ``scored`` is ``_rerank_pool_scored``'s output for ``pool``; ``need`` is how many
        episodes the caller can serve (the echo scan's lazy bound). Returns ``(ranked,
        rerank_top, n_echo)``: the episodes in final order, the echoes dropped, and ``rerank_top``, the RAW
        pre-recency top rerank score among the first ``need`` survivors. That is the value
        recall_metrics.rerank_top_score records and the abstention floor judges, so an echo
        of the prompt that scores high cannot hold the floor open for passages that are
        weak. 0.0 when nothing survives or the rerank degraded/was disabled. Echoed
        episodes are dropped BEFORE the caller slices, so the next-ranked candidates
        backfill the freed slots. The caller excludes its own session's turns from ``pool``
        first (``_join_episode_pool``), so they never take a slot nor a rerank token."""
        raw = dict(scored)
        adjusted = self._apply_rerank_recency(scored, pool)  # no-op when degraded
        order = [i for i, _ in adjusted]
        ranked = [pool[i] for i in order]
        keep, n_echo = self._filter_query_echo(query, ranked, need)
        if n_echo:
            order = [order[i] for i in keep]
            ranked = [ranked[i] for i in keep]
        return ranked, max((raw[i] for i in order[:need]), default=0.0), n_echo

    def _select_episodes(
        self, query: str, pool: list[dict[str, Any]], limit: int
    ) -> tuple[list[dict[str, Any]], int, float]:
        """Rerank `pool` (see ``_rank_episodes``) and serve the top `limit`.

        Returns ``(episodes, n_echo_suppressed, rerank_top)``. Always degrades to RRF order on
        rerank failure; never hard-fails."""
        if not pool:
            return [], 0, 0.0
        scored = self._rerank_pool_scored(query, pool)
        if not scored:
            return [], 0, 0.0
        ranked, rerank_top, n_echo = self._rank_episodes(query, pool, scored, limit)
        return ranked[:limit], n_echo, rerank_top
