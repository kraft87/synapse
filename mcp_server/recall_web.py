"""RecallWebMixin: the web bucket (research pages captured into ``web_chunks``).

BM25 and vector legs over web chunks, fused, reranked, floored and deduped by page.
A restricted caller reads only artifacts labelled with its audience tier (schema 063,
``web_artifacts.audience``); full trust reads them all.
"""

from __future__ import annotations

import logging
from typing import Any

from ingestion.surfaces import WEB_AUDIENCE_PROBE
from mcp_server.kg_pg import _vec_literal
from mcp_server.recall_ranking import merge_rrf as _merge_rrf
from mcp_server.recall_warnings import error_brief as _err_brief
from mcp_server.recall_warnings import warn as _warn

logger = logging.getLogger(__name__)


class RecallWebMixin:
    def _web_audience_ready(self) -> bool:
        """True when restricted web serving can run, i.e. schema 063 is applied.

        Answers from the probe's cache when it can; otherwise borrows a connection to
        probe. Fail-closed: a probe error or a missing column answers False, and the
        restricted caller is served no web chunks, the pre-063 posture. Full-trust
        callers never ask: their statements do not name the column."""
        cached = WEB_AUDIENCE_PROBE.cached(self._db_url)
        if cached is not None:
            return cached
        try:
            with self._pg() as pg:
                return WEB_AUDIENCE_PROBE.ready(pg, self._db_url)
        except Exception as e:
            logger.warning("web audience probe failed (%s); restricted web stays off", e)
            return False

    def _web_audience_gone(self, audience: str | None, e: Exception) -> bool:
        """True when ``e`` is the restricted predicate hitting a missing column (063 rolled
        back after the probe cached it). The leg then serves nothing, never unfiltered."""
        return audience is not None and WEB_AUDIENCE_PROBE.note_error(e, self._db_url)

    def _search_bm25_web(
        self, query: str, limit: int, audience: str | None = None
    ) -> list[dict[str, Any]]:
        """BM25 over web chunks. ``audience`` (a restricted caller's tier, schema 063)
        keeps only chunks whose artifact carries exactly that label; None (full trust)
        is the unfiltered statement."""
        # Same no-alphanumeric-tokens short-circuit as _bm25_table.
        if not any(c.isalnum() for c in query):
            return []
        audience_sql = "AND a.audience = %s" if audience is not None else ""
        params: tuple[Any, ...] = (
            (query, audience, limit) if audience is not None else (query, limit)
        )
        try:
            with self._pg() as pg:
                rows = pg.execute(
                    f"""
                    SELECT c.id, c.content, c.context_prefix, c.web_artifact_id,
                           c.content_ts AS created_at,
                           a.url, a.title, a.tool_name,
                           paradedb.score(c.id) AS bm25_score
                    FROM web_chunks c
                    JOIN web_artifacts a ON a.id = c.web_artifact_id
                    WHERE c.id @@@ paradedb.match('content', %s)
                    {audience_sql}
                    ORDER BY bm25_score DESC LIMIT %s
                    """,
                    params,
                ).fetchall()
            return [{**dict(r), "doc_type": "web", "id": f"w:{r['id']}"} for r in rows]
        except Exception as e:
            if self._web_audience_gone(audience, e):
                return []
            logger.warning("BM25 web search failed: %s", e)
            _warn(f"BM25 web search failed ({_err_brief(e)}): web bucket is degraded.")
            return []

    def _search_vector_web(
        self, query_emb: list[float], limit: int, audience: str | None = None
    ) -> list[dict[str, Any]]:
        """Vector search over web chunks; ``audience`` as in :meth:`_search_bm25_web`.
        The distance is over the full ``vector`` column, which no index serves, so the
        audience predicate filters an exact scan rather than thinning an ANN result."""
        emb_literal = _vec_literal(query_emb)
        audience_sql = "AND a.audience = %s" if audience is not None else ""
        params: tuple[Any, ...] = (
            (emb_literal, audience, limit) if audience is not None else (emb_literal, limit)
        )
        try:
            with self._pg() as pg:
                rows = pg.execute(
                    f"""
                    SELECT c.id, c.content, c.context_prefix, c.web_artifact_id,
                           c.content_ts AS created_at,
                           a.url, a.title, a.tool_name,
                           (c.embedding <=> %s::vector) AS vec_distance
                    FROM web_chunks c
                    JOIN web_artifacts a ON a.id = c.web_artifact_id
                    WHERE c.is_embedded = TRUE
                    {audience_sql}
                    ORDER BY vec_distance ASC LIMIT %s
                    """,
                    params,
                ).fetchall()
            return [{**dict(r), "doc_type": "web", "id": f"w:{r['id']}"} for r in rows]
        except Exception as e:
            if self._web_audience_gone(audience, e):
                return []
            logger.warning("Vector web search failed: %s", e)
            _warn(f"vector web search failed ({_err_brief(e)}): web bucket is degraded.")
            return []

    @staticmethod
    def _dedupe_by_artifact(chunks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        """Keep only the highest-ranked chunk per parent web_artifact_id.

        Input is assumed already RRF-ordered. First occurrence of an
        artifact_id wins; subsequent chunks from the same page drop.
        """
        seen: set[int] = set()
        out: list[dict[str, Any]] = []
        for c in chunks:
            aid = c.get("web_artifact_id")
            if aid in seen:
                continue
            if aid is not None:
                seen.add(aid)
            out.append(c)
            if len(out) >= limit:
                break
        return out

    def _search_web_reranked(
        self, query: str, query_emb: list[float], audience: str | None = None
    ) -> list[dict[str, Any]]:
        """Web bucket (2026-07-24): fuse BM25 + vector via RRF, cross-encoder rerank the pool,
        DROP chunks below _WEB_FLOOR (off-topic collisions), dedupe by parent page, cap to
        _WEB_LIMIT. May return [] when nothing clears the floor — that self-gates web on intent.

        Replaces the old vector-only leg: BM25 was written (_search_bm25_web) but never wired,
        and the raw bi-encoder ranking served ~2.6% precision (feedback). The rerank is the active
        ingredient (it reads content topicality, not surface tokens); fusion widens the candidate
        pool the reranker sees. Degrades safe: on reranker outage/disable (top score 0.0) it serves
        the fused RRF order deduped rather than blanking the bucket or hard-failing.

        ``audience`` is a restricted caller's tier (schema 063): both legs then read only
        artifacts labelled exactly that. None is full trust and the unfiltered legs."""
        settings = self._settings()
        if audience is None:
            bm25 = self._search_bm25_web(query, settings._WEB_FETCH)
            vec = self._search_vector_web(query_emb, settings._WEB_FETCH)
        else:
            bm25 = self._search_bm25_web(query, settings._WEB_FETCH, audience=audience)
            vec = self._search_vector_web(query_emb, settings._WEB_FETCH, audience=audience)
        fused = _merge_rrf(bm25, vec, id_key="id")[: settings._WEB_RERANK_POOL]
        if not fused:
            return []
        scored = self._rerank_pool_scored(query, fused)
        if settings._WEB_FLOOR > 0 and scored and scored[0][1] > 0.0:
            ranked = [fused[i] for i, s in scored if s >= settings._WEB_FLOOR]
        else:  # floor disabled, or reranker degraded to RRF order (no score signal)
            ranked = [fused[i] for i, _ in scored] if scored else fused
        return self._dedupe_by_artifact(ranked, settings._WEB_LIMIT)
