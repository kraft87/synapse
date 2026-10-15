"""RecallEpisodeSearchMixin operations."""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, NamedTuple

from ingestion import embedding as _embedding
from ingestion.schema_check import ColumnProbe
from ingestion.surfaces import (
    OWN_SURFACE_PROBE,
    SurfaceTrust,
    episode_scope_sql,
    own_surface_if_ready,
)
from mcp_server.kg_pg import _vec_literal
from mcp_server.recall_ranking import merge_rrf as _merge_rrf
from mcp_server.recall_ranking import timed as _timed
from mcp_server.recall_settings import RecallSettings
from mcp_server.recall_warnings import config_hint as _config_hint
from mcp_server.recall_warnings import error_brief as _err_brief
from mcp_server.recall_warnings import submit_with_context as _submit_ctx
from mcp_server.recall_warnings import warn as _warn

logger = logging.getLogger(__name__)

#: Chunks taken from each chunk source (a chunk is 4 consecutive episodes, advanced 3 at a
#: time), so each source adds at most ~40 episodes to a pool the episode legs fill with 200.
_CHUNK_FETCH = 10

#: The optional chunks vector index (its migration builds it CONCURRENTLY). Without it the
#: chunk vector source would scan every chunk's 2048-dim embedding, so it stays off until
#: the index is valid; an interrupted CONCURRENTLY build leaves an invalid one.
CHUNK_VECTOR_INDEX = ColumnProbe(
    "SELECT 1 FROM pg_index WHERE indexrelid = to_regclass('chunks_embedding_hnsw') AND indisvalid",
    "the chunks vector index",
)


class EpisodeLegs(NamedTuple):
    """The episode candidate legs in flight (see ``_start_episode_legs``)."""

    f_bm25: Future[tuple[list[dict[str, Any]], float]]
    f_vec: Future[tuple[list[dict[str, Any]], float]] | None  # None: no query embedding
    query_emb: list[float] | None
    ms_embed: float
    # The chunk candidate sources (SYNAPSE_RECALL_CHUNK_CANDIDATES): each is a ranked list
    # of episodes, like the legs above. None: the setting is off, or the source cannot run.
    f_chunk_bm25: Future[list[dict[str, Any]]] | None = None
    f_chunk_vec: Future[list[dict[str, Any]]] | None = None


class EpisodePool(NamedTuple):
    """The rerank input and what it took to build (see ``_join_episode_pool``)."""

    items: list[dict[str, Any]]
    n_bm25: int
    n_vector: int
    n_self_excluded: int
    ms_bm25: float
    ms_vector: float


class RecallEpisodeSearchMixin:
    _db_url: str
    _leg_executor: ThreadPoolExecutor

    if TYPE_CHECKING:

        def _pg(self) -> AbstractContextManager[Any]: ...
        def _settings(self) -> RecallSettings: ...
        def _ensure_embedder(self) -> Any: ...
        @staticmethod
        def _exclude_self(
            ranked_eps: list[dict[str, Any]], self_session: str
        ) -> list[dict[str, Any]]: ...

    def _own_surface(self, st: SurfaceTrust) -> str | None:
        """The caller's own-surface widening (schema 057), or None.

        ``st.own_surface`` is set only for a credential-bound restricted caller; it is
        used only once the provenance column exists. Everyone else (full trust, unknown,
        self-reported ids) gets None without a probe, i.e. the pre-057 statements."""
        if not st.own_surface:
            return None
        try:
            with self._pg() as pg:
                return own_surface_if_ready(pg, self._db_url, st.own_surface)
        except Exception as e:  # no connection: the legs will report it themselves
            logger.warning("own-surface probe skipped (%s)", e)
            return None

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
        own_surface: str | None = None,
    ) -> list[dict[str, Any]]:
        # A query with no alphanumeric content tokenizes to nothing — skip the
        # round-trip instead of burning it on a guaranteed-empty (or erroring)
        # match. Same "no tokens -> no candidates" semantics as the KG BM25
        # legs' sanitize-then-empty check.
        if not any(c.isalnum() for c in query):
            return []
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
        scope = episode_scope_sql(allowed_projects, own_surface)
        if scope is not None:  # restricted surface (schema 053/057)
            # ANY('{}') is false for every row, NULL project included — an unknown
            # surface serves nothing rather than everything. That is the fail-closed
            # serve, not a bug to "fix" with an emptiness special case. A known
            # restricted surface also matches the rows it ingested (own_surface).
            where.append(scope[0])
            params.extend(scope[1])
        try:
            with self._pg() as pg:
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
            if own_surface and OWN_SURFACE_PROBE.note_error(e, self._db_url):
                # Schema 057 rolled back under a running engine: the pre-057 serve.
                return self._bm25_table(
                    table, query, project, limit, doc_type, session_id, allowed_projects
                )
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
        own_surface: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._bm25_table(
            "episodes", query, project, limit, "episode", session_id, allowed_projects, own_surface
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
        own_surface: str | None = None,
    ) -> list[dict[str, Any]]:
        settings = self._settings()
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
        scope = episode_scope_sql(allowed_projects, own_surface)
        if scope is not None:  # restricted surface (schema 053/057) — see _bm25_table
            where.append(scope[0])
            params.extend(scope[1])
        try:
            with self._pg() as pg:
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
            if own_surface and OWN_SURFACE_PROBE.note_error(e, self._db_url):
                return self._vector_table(
                    table, emb_literal, project, limit, doc_type, session_id, allowed_projects
                )
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
        own_surface: str | None = None,
    ) -> list[dict[str, Any]]:
        emb_literal = _vec_literal(query_emb)
        return self._vector_table(
            "episodes",
            emb_literal,
            project,
            limit,
            "episode",
            session_id,
            allowed_projects,
            own_surface,
        )

    def _chunk_episodes(
        self,
        chunks: list[dict[str, Any]],
        project: str | None,
        allowed_projects: list[str] | None,
        own_surface: str | None,
    ) -> list[dict[str, Any]]:
        """The episodes of ranked ``chunks``, in rank order and the shape the other episode
        legs return them in. A chunk is never served; it only names episodes. They load
        through the same project and surface predicates as the episode legs, so a chunk
        cannot lead a restricted caller to a turn it could not have searched for."""
        ids = list(dict.fromkeys(i for chunk in chunks for i in chunk["episode_ids"]))
        if not ids:
            return []
        where = ["id = ANY(%s)"]
        params: list[Any] = [ids]
        if project:
            where.append("project = %s")
            params.append(project)
        scope = episode_scope_sql(allowed_projects, own_surface)
        if scope is not None:
            where.append(scope[0])
            params.extend(scope[1])
        with self._pg() as pg:
            rows = pg.execute(
                "SELECT id, content, project, created_at, retrieval_count, session_id "
                f"FROM episodes WHERE {' AND '.join(where)}",
                params,
            ).fetchall()
        by_id = {r["id"]: r for r in rows}
        return [{**by_id[i], "doc_type": "episode", "id": f"e:{i}"} for i in ids if i in by_id]

    def _search_chunks_bm25(
        self,
        query: str,
        project: str | None,
        allowed_projects: list[str] | None = None,
        own_surface: str | None = None,
    ) -> list[dict[str, Any]]:
        """Episodes of the chunks whose text best matches ``query`` by keyword: a candidate
        source for episodes whose own text misses the words a neighbouring turn carries."""
        if not any(c.isalnum() for c in query):  # see _bm25_table
            return []
        where = ["id @@@ paradedb.match('content', %s)"]
        params: list[Any] = [query]
        if project:
            where.append("project = %s")
            params.append(project)
        try:
            with self._pg() as pg:
                chunks = pg.execute(
                    "SELECT id, episode_ids, paradedb.score(id) AS bm25_score FROM chunks "
                    f"WHERE {' AND '.join(where)} ORDER BY bm25_score DESC LIMIT %s",
                    (*params, _CHUNK_FETCH),
                ).fetchall()
            return self._chunk_episodes(chunks, project, allowed_projects, own_surface)
        except Exception as e:
            logger.warning("BM25 chunks search failed: %s", e)
            _warn(
                f"BM25 chunks search failed ({_err_brief(e)}): chunk keyword source served nothing."
            )
            return []

    def _search_chunks_vector(
        self,
        query_emb: list[float],
        project: str | None,
        allowed_projects: list[str] | None = None,
        own_surface: str | None = None,
    ) -> list[dict[str, Any]]:
        """Episodes of the chunks nearest to the query embedding. Needs the optional chunks
        vector index; without it the source says so and serves nothing, because the
        alternative is an exact scan of every chunk on every recall."""
        dims = self._settings()._EMBED_DIMS
        emb_literal = _vec_literal(query_emb)
        where = ["is_embedded = TRUE"]
        params: list[Any] = []
        if project:
            where.append("project = %s")
            params.append(project)
        try:
            with self._pg() as pg:
                if not CHUNK_VECTOR_INDEX.ready(pg, self._db_url):
                    _warn(
                        "chunk vector source skipped: the chunks vector index is not built "
                        "(optional migration), and without it every recall would scan all chunks."
                    )
                    return []
                chunks = pg.execute(
                    "SELECT id, episode_ids FROM chunks "
                    f"WHERE {' AND '.join(where)} "
                    f"ORDER BY embedding::halfvec({dims}) <=> %s::halfvec({dims}) LIMIT %s",
                    (*params, emb_literal, _CHUNK_FETCH),
                ).fetchall()
            return self._chunk_episodes(chunks, project, allowed_projects, own_surface)
        except Exception as e:
            logger.warning("Vector chunks search failed: %s", e)
            _warn(
                f"vector chunks search failed ({_err_brief(e)}): chunk semantic source served nothing."
            )
            return []

    def _embed_query(self, query: str) -> tuple[list[float] | None, float]:
        """Embed the query for the vector legs: ``(embedding, ms)``. A dead backend yields
        ``None`` and a degradation notice; the legs then run BM25-only."""
        started = time.perf_counter()
        try:
            query_emb = self._ensure_embedder().embed([query], task="query")[0]
        except Exception as e:
            logger.error("Embedding query failed: %s", e)
            detail = _err_brief(e)
            backend = _embedding.embed_provider()
            _warn(
                f"embedding failed ({backend}: {detail}): vector legs skipped, "
                f"results are BM25-only."
                + _config_hint(detail, backend=backend, env_prefix="SYNAPSE_EMBED")
            )
            query_emb = None
        return query_emb, (time.perf_counter() - started) * 1000.0

    def _start_episode_legs(
        self,
        query: str,
        project: str | None,
        session_id: str | None = None,
        allowed_projects: list[str] | None = None,
        own_surface: str | None = None,
    ) -> EpisodeLegs:
        """Start the BM25 and vector episode legs on the leg executor (WIN1 deep-fetch).

        BM25 is pure text search: it starts FIRST so its ~165ms fetch overlaps the ~170ms
        Voyage query-embedding call, and the vector leg follows the embedding. The caller
        can start its own embedding-dependent legs before joining with
        ``_join_episode_pool``. ``session_id`` scopes both legs to one conversation (the
        fetch_session/Grep drill-down). ``allowed_projects`` applies a restricted surface's
        project allowlist to BOTH legs: it filters the pool, not the served slice, so the
        allowlisted content still gets the full deep-fetch width. ``own_surface`` widens
        that filter to the rows the calling surface ingested itself (schema 057)."""
        settings = self._settings()
        fetch = settings._EPISODE_FETCH
        # Chunk sources feed more candidates into the same pool. A session-scoped drill-down
        # greps one conversation, which the episode legs already cover.
        chunks = settings._CHUNK_CANDIDATES and session_id is None
        f_bm25 = _submit_ctx(
            self._leg_executor,
            _timed,
            self._search_bm25_episodes,
            query,
            project,
            fetch,
            session_id,
            allowed_projects,
            own_surface,
        )
        f_chunk_bm25 = (
            _submit_ctx(
                self._leg_executor,
                self._search_chunks_bm25,
                query,
                project,
                allowed_projects,
                own_surface,
            )
            if chunks
            else None
        )
        query_emb, ms_embed = self._embed_query(query)
        f_vec = (
            _submit_ctx(
                self._leg_executor,
                _timed,
                self._search_vector_episodes,
                query_emb,
                project,
                fetch,
                session_id,
                allowed_projects,
                own_surface,
            )
            if query_emb is not None
            else None
        )
        f_chunk_vec = (
            _submit_ctx(
                self._leg_executor,
                self._search_chunks_vector,
                query_emb,
                project,
                allowed_projects,
                own_surface,
            )
            if chunks and query_emb is not None
            else None
        )
        return EpisodeLegs(f_bm25, f_vec, query_emb, ms_embed, f_chunk_bm25, f_chunk_vec)

    def _join_episode_pool(self, legs: EpisodeLegs, self_session: str | None = None) -> EpisodePool:
        """Fuse the legs (RRF) into the PRE-rerank pool the cross-encoder scores.

        The deep fetch is what lets the reranker recover a rank-16..88 gold; plain RRF
        can't. ``self_session`` drops the calling session's own turns BEFORE the pool is
        cut, so their slots backfill with real history instead of costing rerank tokens
        and a place in the pool. Pass it only when the caller wants self-exclusion (a
        session-scoped drill-down never does)."""
        bm25_eps, ms_bm25 = legs.f_bm25.result()
        vec_eps, ms_vec = legs.f_vec.result() if legs.f_vec is not None else ([], 0.0)
        chunk_bm25 = legs.f_chunk_bm25.result() if legs.f_chunk_bm25 is not None else []
        chunk_vec = legs.f_chunk_vec.result() if legs.f_chunk_vec is not None else []
        fused = _merge_rrf(bm25_eps, vec_eps, chunk_bm25, chunk_vec, id_key="id")
        n_self_excluded = 0
        if self_session:
            kept = self._exclude_self(fused, self_session)
            n_self_excluded = len(fused) - len(kept)
            fused = kept
        pool = fused[: self._settings()._EPISODE_RERANK_POOL]
        return EpisodePool(pool, len(bm25_eps), len(vec_eps), n_self_excluded, ms_bm25, ms_vec)
