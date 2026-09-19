"""RecallEpisodeSearchMixin operations."""

from __future__ import annotations

import logging
from typing import Any

from mcp_server.kg_pg import _vec_literal
from mcp_server.recall_ranking import merge_rrf as _merge_rrf
from mcp_server.recall_settings import _EPISODE_FETCH, _EPISODE_RERANK_POOL
from mcp_server.recall_warnings import error_brief as _err_brief
from mcp_server.recall_warnings import warn as _warn

logger = logging.getLogger(__name__)


class RecallEpisodeSearchMixin:
    @staticmethod
    def _ts_col(table: str) -> str:
        """Timestamp column name varies by table."""
        return "generated_at" if table == "synth_documents" else "created_at"

    @staticmethod
    def _extra_cols(table: str) -> str:
        # Slim SELECT: only fields read by the ranking pipeline. Drill-down
        # columns (source_ids, sequence ranges, synth_type) are dropped — caller
        # never sees them, so don't waste DB→Python wire bytes.
        # retrieval_count IS kept for episodes because _feedback_multiplier reads it.
        # session_id IS kept for episodes because the passage serve-cap reads it to break
        # self/recency domination (one session monopolising the served bucket). It is still
        # dropped from the caller-facing item by _to_recall_item — internal ranking use only.
        if table == "episodes":
            return ", retrieval_count, session_id"
        if table == "chunks":
            return ", episode_ids"  # chunk = retrieval signal; serve its episodes
        return ""

    def _bm25_table(
        self,
        table: str,
        query: str,
        project: str | None,
        limit: int,
        doc_type: str,
        session_id: str | None = None,
        allowed_projects: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        # A query with no alphanumeric content tokenizes to nothing — skip the
        # round-trip instead of burning it on a guaranteed-empty (or erroring)
        # match. Same "no tokens -> no candidates" semantics as the KG BM25
        # legs' sanitize-then-empty check.
        if not any(c.isalnum() for c in query):
            return []
        pg = self._ensure_pg()
        ts = self._ts_col(table)
        extra = self._extra_cols(table)
        where = ["id @@@ paradedb.match('content', %s)"]
        params: list[Any] = [query]
        if project:
            where.append("project = %s")
            params.append(project)
        if session_id:  # episodes only — the session-scoped drill-down filter
            where.append("session_id = %s")
            params.append(session_id)
        if allowed_projects is not None:  # restricted surface (schema 053)
            # ANY('{}') is false for every row, NULL project included — an unknown
            # surface serves nothing rather than everything. That is the fail-closed
            # serve, not a bug to "fix" with an emptiness special case.
            where.append("project = ANY(%s)")
            params.append(allowed_projects)
        try:
            rows = pg.execute(
                f"""
                SELECT id, content, project,
                       {ts} AS created_at{extra},
                       paradedb.score(id) AS bm25_score
                FROM {table}
                WHERE {" AND ".join(where)}
                ORDER BY bm25_score DESC LIMIT %s
                """,
                (*params, limit),
            ).fetchall()
            return [
                {**dict(r), "doc_type": doc_type, "id": f"{doc_type[0]}:{r['id']}"} for r in rows
            ]
        except Exception as e:
            logger.warning("BM25 %s search failed: %s", table, e)
            _warn(f"BM25 {table} search failed ({_err_brief(e)}): lexical leg served nothing.")
            return []

    def _search_bm25_episodes(
        self,
        query: str,
        project: str | None,
        limit: int,
        session_id: str | None = None,
        allowed_projects: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return self._bm25_table(
            "episodes", query, project, limit, "episode", session_id, allowed_projects
        )

    def _vector_table(
        self,
        table: str,
        emb_literal: str,
        project: str | None,
        limit: int,
        doc_type: str,
        session_id: str | None = None,
        allowed_projects: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        settings = self._settings()
        pg = self._ensure_pg()
        ts = self._ts_col(table)
        extra = self._extra_cols(table)
        # Distance is computed over halfvec(N), NOT vector(N). The default embeddings are
        # 2048-dim (voyage-4-large), which exceeds pgvector's 2000-dim limit for HNSW on
        # the full `vector` type — so the HNSW indexes are built on `embedding::halfvec(N)`
        # (half-precision, index limit 4000 dims), with N = _EMBED_DIMS as provisioned.
        # The ORDER BY expression must match that index expression verbatim to be served
        # from it; the alias form is NOT index-eligible.
        # Half precision is loss-free for what's served: recall@10 vs exact scan = 1.000,
        # recall@100 = 0.981 (and the reranker re-scores the pool). 878ms -> 23ms on episodes.
        where = ["is_embedded = TRUE"]
        params: list[Any] = [emb_literal]
        if project:
            where.append("project = %s")
            params.append(project)
        if session_id:  # episodes only — the session-scoped drill-down filter
            where.append("session_id = %s")
            params.append(session_id)
        if allowed_projects is not None:  # restricted surface (schema 053) — see _bm25_table
            where.append("project = ANY(%s)")
            params.append(allowed_projects)
        try:
            rows = pg.execute(
                f"""
                SELECT id, content, project,
                       {ts} AS created_at{extra},
                       (embedding::halfvec({settings._EMBED_DIMS}) <=> %s::halfvec({settings._EMBED_DIMS})) AS vec_distance
                FROM {table}
                WHERE {" AND ".join(where)}
                ORDER BY embedding::halfvec({settings._EMBED_DIMS}) <=> %s::halfvec({settings._EMBED_DIMS}) ASC LIMIT %s
                """,
                (*params, emb_literal, limit),
            ).fetchall()
            return [
                {**dict(r), "doc_type": doc_type, "id": f"{doc_type[0]}:{r['id']}"} for r in rows
            ]
        except Exception as e:
            logger.warning("Vector %s search failed: %s", table, e)
            _warn(f"vector {table} search failed ({_err_brief(e)}): semantic leg served nothing.")
            return []

    def _search_vector_episodes(
        self,
        query_emb: list[float],
        project: str | None,
        limit: int,
        session_id: str | None = None,
        allowed_projects: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        emb_literal = _vec_literal(query_emb)
        return self._vector_table(
            "episodes", emb_literal, project, limit, "episode", session_id, allowed_projects
        )

    def _episode_pool(
        self,
        query: str,
        query_emb: list[float] | None,
        project: str | None,
        fetch: int = _EPISODE_FETCH,
        pool_size: int = _EPISODE_RERANK_POOL,
        session_id: str | None = None,
        allowed_projects: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Fused BM25+vector episode candidate pool, PRE-rerank (WIN1 deep-fetch).

        Shared primitive: recall_episodes() reranks it and serves top-k (deep
        drill-down); recall() merges it with the summary candidates and co-reranks
        in a single cross-encoder pass, then partitions by doc_type. The deep fetch
        is what lets the reranker recover a rank-16..88 gold; plain RRF can't.
        BM25-only when the query embedding is unavailable. ``session_id`` scopes
        both legs to one conversation (the fetch_session/Grep drill-down).
        ``allowed_projects`` applies a restricted surface's project allowlist to BOTH
        legs — filtering the pool, not the served slice, so the allowlisted content
        still gets the full deep-fetch width."""
        bm25_eps = self._search_bm25_episodes(query, project, fetch, session_id, allowed_projects)
        vec_eps: list[dict[str, Any]] = []
        if query_emb is not None:
            vec_eps = self._search_vector_episodes(
                query_emb, project, fetch, session_id, allowed_projects
            )
        return _merge_rrf(bm25_eps, vec_eps, id_key="id")[:pool_size]
