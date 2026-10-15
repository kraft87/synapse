"""recall() — the core retrieval function for Synapse.

What recall() serves:
  1. Episodes  — passages of the best-matching user-turn exchanges, from a wide-pool rerank
  2. KG facts  — relationship edges (kg_relationships), with their superseded predecessors
  3. Notes     — the curated notes store, by hook similarity (a rerank floor gates the bucket)
  4. Web       — captured web chunks (out-of-repo cron), gated the same way

Search strategy:
  - BM25 (ParadeDB) + ANN cosine over episodes, fused with RRF
  - KG vector seed + graph traversal for facts
  - ONE cross-encoder rerank over the episode pool

The KG legs serve from Postgres (kg_entities / kg_relationships); a KG-leg failure
degrades to an empty facts bucket rather than failing the whole recall.

recall_episodes() exposes the raw episode drill-down (the recall_full_turns tool). Chunks
come back as a candidate source only: with RecallConfig.chunk_candidates on, the episodes of
the best-matching chunks join the pool the cross-encoder reranks.

Every tunable lives in recall_config.RecallConfig.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from typing import Any

from psycopg.rows import dict_row

from ingestion import embedding as _embedding
from ingestion import pg_pool
from ingestion.surfaces import SurfaceTrust, lookup_surface
from mcp_server.recall_config import RecallConfig
from mcp_server.recall_episode_search import RecallEpisodeSearchMixin
from mcp_server.recall_episodes import RecallEpisodesMixin
from mcp_server.recall_fetch import RecallFetchMixin
from mcp_server.recall_metrics import RecallMetricsMixin
from mcp_server.recall_overview import RecallOverviewMixin
from mcp_server.recall_passages import RecallPassagesMixin
from mcp_server.recall_presentation import apply_supersessions as _apply_supersessions
from mcp_server.recall_presentation import parse_episode_ids as _parse_episode_ids
from mcp_server.recall_presentation import passage_role as _passage_role
from mcp_server.recall_presentation import role_spans as _role_spans
from mcp_server.recall_presentation import to_recall_item as _to_recall_item
from mcp_server.recall_presentation import to_web_recall_item as _to_web_recall_item
from mcp_server.recall_query import bm25_best_window as _bm25_best_window
from mcp_server.recall_query import bm25_tokenize as _bm25_tokenize
from mcp_server.recall_query import echo_lcs_len as _echo_lcs_len
from mcp_server.recall_query import normalize_whitespace as _norm_ws
from mcp_server.recall_query import query_shingles as _query_shingles
from mcp_server.recall_ranking import merge_rrf as _merge_rrf
from mcp_server.recall_ranking import recency_multiplier as _recency_multiplier
from mcp_server.recall_ranking import served_chars as _served_chars
from mcp_server.recall_ranking import timed as _timed
from mcp_server.recall_rerank import RecallRerankMixin
from mcp_server.recall_sources import RecallSourcesMixin
from mcp_server.recall_warnings import config_hint as _config_hint
from mcp_server.recall_warnings import error_brief as _err_brief
from mcp_server.recall_warnings import submit_with_context as _submit_ctx
from mcp_server.recall_warnings import warn as _warn
from mcp_server.recall_warnings import warn_sink as _warn_sink
from mcp_server.recall_web import RecallWebMixin

# Sentinel for the lazily-built reranker: ``None`` is a VALID resolved value
# (SYNAPSE_RERANK_PROVIDER=none — fusion-only serving), so "unset" needs its own marker.
_RERANKER_UNSET = object()

logger = logging.getLogger(__name__)


def _resolve(db_url: str, surface: str | None, trust: SurfaceTrust | None) -> SurfaceTrust:
    """The serving verdict for one call: a pre-resolved one wins, else look up the id.

    ``trust`` is what the MCP server and the HTTP routes pass: the caller's verdict,
    resolved from its credential, so the engine must not go re-derive one from a
    self-reported string. ``surface`` is for in-process callers and tests: a
    server-derived id, never a value a caller supplied. It fails closed the same way.
    """
    return trust if trust is not None else lookup_surface(db_url, surface)


class Recall(
    RecallEpisodeSearchMixin,
    RecallRerankMixin,
    RecallPassagesMixin,
    RecallSourcesMixin,
    RecallWebMixin,
    RecallOverviewMixin,
    RecallEpisodesMixin,
    RecallFetchMixin,
    RecallMetricsMixin,
):
    """Stateful retrieval engine. One instance per MCP server process."""

    def _floor_shadow(self, served_ids: dict[str, Any], rerank_top: float, emb_ok: bool) -> None:
        """Mark the recall_metrics ``served_ids`` envelope when the abstention floor fires.

        Telemetry only, never touches the served payload: real rerank scores exist and the RAW
        pre-recency top score is strictly below ``recall_floor``. recall() drops its episode
        bucket on the same condition; recall_episodes() only records it. No marker when: the floor
        is disabled (<= 0); the rerank is disabled or degraded (the all-0.0 sentinel, no real
        scores), which also covers the empty candidate pool (rerank_top 0.0); or the query
        embedding failed (emb_ok False: a weak top under crippled retrieval says nothing about
        whether relevant memory exists)."""
        if emb_ok and 0.0 < rerank_top < self._config.recall_floor:
            served_ids["would_abstain"] = True
            served_ids["floor"] = self._config.recall_floor

    @staticmethod
    def _echo_overlap(content: str, query: str) -> int:
        return _echo_lcs_len(content, query)

    def _resolve_trust(self, surface: str | None, trust: SurfaceTrust | None) -> SurfaceTrust:
        return _resolve(self._db_url, surface, trust)

    def __init__(
        self,
        db_url: str,
        voyage_api_key: str,
        config: RecallConfig | None = None,
    ) -> None:
        self._db_url = db_url
        self._voyage_key = voyage_api_key
        self._config = config or RecallConfig()
        self._embedder: Any = None
        self._reranker: Any = _RERANKER_UNSET
        # Background executor for fire-and-forget feedback writes (retrieval
        # count bumps). One worker is enough for single-user scale; queueing
        # on a hot recall burst is cheaper than blocking the response.
        self._async_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="recall-feedback"
        )
        # Leg executor: runs recall()'s independent legs (episode BM25 + vector, web, KG,
        # supersession surface, notes, then the rerank and the superseded pairs) concurrently,
        # each borrowing its own connection (_pg) or making its own Voyage call. Legs never
        # submit here, so concurrent recalls queue rather than deadlock. 8 = one recall's
        # widest wave; the connection pool (ingestion/pg_pool) holds 16.
        self._leg_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="recall-leg")

    def _pg(self) -> AbstractContextManager[Any]:
        """A dict-row autocommit connection for one query or a short run of them: borrowed
        from the server's pool, or opened for the block where there is none (see
        ingestion/pg_pool, which also sets hnsw.ef_search = 200 on every connection).
        Each search leg borrows its own, so parallel legs never share a psycopg handle.
        Hold it only around the statements: nothing inside the block may borrow again."""
        return pg_pool.connection(self._db_url, row_factory=dict_row)

    def _ensure_embedder(self) -> Any:
        if self._embedder is None:
            # Interactive: a Voyage 429 gives up within seconds and recall degrades to
            # BM25 / RRF order, instead of the 3-minute patience batch work keeps.
            self._embedder = _embedding.create_embedder(
                voyage_api_key=self._voyage_key, db_url=self._db_url, interactive=True
            )
        return self._embedder

    def _ensure_reranker(self) -> Any:
        """Lazily-built rerank backend. ``None`` = rerank disabled
        (SYNAPSE_RERANK_PROVIDER=none) — callers serve the fusion (RRF) order."""
        if self._reranker is _RERANKER_UNSET:
            self._reranker = _embedding.create_reranker(
                voyage_api_key=self._voyage_key, interactive=True
            )
        return self._reranker

    @staticmethod
    def _exclude_self(ranked_eps: list[dict[str, Any]], self_session: str) -> list[dict[str, Any]]:
        """Drop the calling session's own episodes from the serving pool.

        Runs whenever the call carries ``self_session`` (injected by the client's
        PreToolUse hook). The caller already holds its own turns in context; serving
        them back wastes tokens, crowds out older real history, and was the top
        measured recall_feedback noise driver. A session-scoped drill-down
        (``session_id`` set, or fetch) is never excluded.
        """
        return [e for e in ranked_eps if e.get("session_id") != self_session]


__all__ = [
    "Recall",
    "_apply_supersessions",
    "_bm25_best_window",
    "_bm25_tokenize",
    "_config_hint",
    "_echo_lcs_len",
    "_err_brief",
    "_merge_rrf",
    "_norm_ws",
    "_parse_episode_ids",
    "_passage_role",
    "_query_shingles",
    "_recency_multiplier",
    "_role_spans",
    "_served_chars",
    "_submit_ctx",
    "_timed",
    "_to_recall_item",
    "_to_web_recall_item",
    "_warn",
    "_warn_sink",
]
