"""RecallWebMixin operations: the web bucket, fed by the out-of-repo capture cron."""

from __future__ import annotations

import logging
from typing import Any

from mcp_server.kg_pg import _vec_literal
from mcp_server.recall_ranking import merge_rrf as _merge_rrf
from mcp_server.recall_warnings import error_brief as _err_brief
from mcp_server.recall_warnings import warn as _warn

logger = logging.getLogger(__name__)


def vector_web_sql(dims: int) -> str:
    """The web vector KNN, nearest first; placeholders: query vector (distance), query vector
    (order), limit.

    The ORDER BY is the verbatim ``embedding::halfvec(N)`` expression
    ``web_chunks_embedding_hnsw`` is built on (schema 014), so the index serves it; ordering
    by the full-precision ``vector`` column reads every embedded chunk. It sits in a LIMITed
    subquery so the join to web_artifacts only touches the rows that survive."""
    return f"""
        SELECT c.id, c.content, c.context_prefix, c.web_artifact_id,
               c.content_ts AS created_at, a.url, a.title, a.tool_name, c.vec_distance
        FROM (
            SELECT id, content, context_prefix, web_artifact_id, content_ts,
                   (embedding::halfvec({dims}) <=> %s::halfvec({dims})) AS vec_distance
            FROM web_chunks
            WHERE is_embedded = TRUE
            ORDER BY embedding::halfvec({dims}) <=> %s::halfvec({dims}) LIMIT %s
        ) c
        JOIN web_artifacts a ON a.id = c.web_artifact_id
        ORDER BY c.vec_distance
        """


class RecallWebMixin:
    def _search_bm25_web(self, query: str, limit: int) -> list[dict[str, Any]]:
        # Same no-alphanumeric-tokens short-circuit as _bm25_table.
        if not any(c.isalnum() for c in query):
            return []
        try:
            with self._pg() as pg:
                rows = pg.execute(
                    """
                    SELECT c.id, c.content, c.context_prefix, c.web_artifact_id,
                           c.content_ts AS created_at,
                           a.url, a.title, a.tool_name,
                           paradedb.score(c.id) AS bm25_score
                    FROM web_chunks c
                    JOIN web_artifacts a ON a.id = c.web_artifact_id
                    WHERE c.id @@@ paradedb.match('content', %s)
                    ORDER BY bm25_score DESC LIMIT %s
                    """,
                    (query, limit),
                ).fetchall()
            return [{**dict(r), "doc_type": "web", "id": f"w:{r['id']}"} for r in rows]
        except Exception as e:
            logger.warning("BM25 web search failed: %s", e)
            _warn(f"BM25 web search failed ({_err_brief(e)}): web bucket is degraded.")
            return []

    def _search_vector_web(self, query_emb: list[float], limit: int) -> list[dict[str, Any]]:
        emb_literal = _vec_literal(query_emb)
        try:
            with self._pg() as pg:
                rows = pg.execute(
                    vector_web_sql(self._settings()._EMBED_DIMS),
                    (emb_literal, emb_literal, limit),
                ).fetchall()
            return [{**dict(r), "doc_type": "web", "id": f"w:{r['id']}"} for r in rows]
        except Exception as e:
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

    def _search_web_reranked(self, query: str, query_emb: list[float]) -> list[dict[str, Any]]:
        """Web bucket (2026-07-24): fuse BM25 + vector via RRF, cross-encoder rerank the pool,
        DROP chunks below _WEB_FLOOR (off-topic collisions), dedupe by parent page, cap to
        _WEB_LIMIT. May return [] when nothing clears the floor — that self-gates web on intent.

        Replaces the old vector-only leg: BM25 was written (_search_bm25_web) but never wired,
        and the raw bi-encoder ranking served ~2.6% precision (feedback). The rerank is the active
        ingredient (it reads content topicality, not surface tokens); fusion widens the candidate
        pool the reranker sees. Degrades safe: on reranker outage/disable (top score 0.0) it serves
        the fused RRF order deduped rather than blanking the bucket or hard-failing."""
        settings = self._settings()
        bm25 = self._search_bm25_web(query, settings._WEB_FETCH)
        vec = self._search_vector_web(query_emb, settings._WEB_FETCH)
        fused = _merge_rrf(bm25, vec, id_key="id")[: settings._WEB_RERANK_POOL]
        if not fused:
            return []
        scored = self._rerank_pool_scored(query, fused)
        if settings._WEB_FLOOR > 0 and scored and scored[0][1] > 0.0:
            ranked = [fused[i] for i, s in scored if s >= settings._WEB_FLOOR]
        else:  # floor disabled, or reranker degraded to RRF order (no score signal)
            ranked = [fused[i] for i, _ in scored] if scored else fused
        return self._dedupe_by_artifact(ranked, settings._WEB_LIMIT)
