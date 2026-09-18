"""RecallEpisodesMixin operations."""

from __future__ import annotations

import logging
import time
from typing import Any

from ingestion import embedding as _embedding
from ingestion.surfaces import SurfaceTrust
from mcp_server.recall_presentation import to_recall_item as _to_recall_item
from mcp_server.recall_ranking import served_chars as _served_chars
from mcp_server.recall_settings import _EPISODE_LIMIT
from mcp_server.recall_warnings import config_hint as _config_hint
from mcp_server.recall_warnings import error_brief as _err_brief
from mcp_server.recall_warnings import warn as _warn
from mcp_server.recall_warnings import warn_sink as _warn_sink

logger = logging.getLogger(__name__)


class RecallEpisodesMixin:
    def recall_episodes(
        self,
        query: str,
        project: str | None = None,
        limit: int = _EPISODE_LIMIT,
        source: str | None = None,
        self_session: str | None = None,
        session_id: str | None = None,
        surface: str | None = None,
        trust: SurfaceTrust | None = None,
    ) -> dict[str, Any]:
        """Raw episode drill-down: individual conversation turns.

        Best for 'show me exactly what was said about X' queries.
        Returns full episode content ranked by relevance + recency.

        ``session_id`` scopes the search to ONE conversation (the Grep-within-a-
        session drill-down): same BM25+vector+rerank pipeline, pool restricted to
        that session, and self-exclusion is skipped — an explicit session ask
        must never be suppressed, even for the caller's own session.

        Served on the MCP surface as the recall_full_turns tool (standalone again
        since 2026-07-26 — its docstring needed more room than a mode inside
        recall's 2KB budget allowed); the telemetry kind stays 'episodes' so
        historical per-tool metrics remain comparable.

        ``surface`` applies the same project allowlist recall() does (schema 053): a
        restricted caller must not reach whole turns it cannot reach passages of.

        Carries the same ``warnings`` list recall() does, and on the same terms: present
        only when a leg degraded, absent otherwise.
        """
        t_start = time.perf_counter()
        warnings: list[str] = []
        with _warn_sink(warnings):
            return self._recall_episodes_inner(
                query=query,
                warnings=warnings,
                project=project,
                limit=limit,
                source=source,
                self_session=self_session,
                session_id=session_id,
                surface=surface,
                trust=trust,
                t_start=t_start,
            )

    def _recall_episodes_inner(
        self,
        query: str,
        warnings: list[str],
        project: str | None,
        limit: int,
        source: str | None,
        self_session: str | None,
        session_id: str | None,
        surface: str | None,
        trust: SurfaceTrust | None,
        t_start: float,
    ) -> dict[str, Any]:
        """recall_episodes()'s body, run with ``warnings`` bound as the degradation sink."""
        settings = self._settings()
        allowed = self._resolve_trust(surface, trust).project_filter
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

        # Deep-fetch both legs (golds rank up to ~88 in a single leg), fuse, then
        # rerank the WIDE pool and select what to serve (WIN1 — see _episode_pool).
        # Fixed top-`limit` by default; adaptive score-cutoff when enabled (see
        # _select_episodes / _EPISODE_CUTOFF_TAU).
        pool = self._episode_pool(
            query, query_emb, project, session_id=session_id, allowed_projects=allowed
        )
        # Same self-exclusion as recall(): the caller's own turns are already in
        # its context window. Pool-level so excluded slots backfill before rerank.
        # Skipped under a session filter — the caller explicitly asked for that
        # session's turns, including when it is its own.
        n_self_excluded = 0
        if settings._RECALL_SELF_EXCLUDE and self_session and not session_id:
            pre_excl = len(pool)
            pool = self._exclude_self(pool, self_session)
            n_self_excluded = pre_excl - len(pool)
        episodes, n_echo_suppressed, rerank_top = self._select_episodes(query, pool, limit)

        # Increment feedback counts BEFORE slimming (we lose the parseable id afterwards)
        ep_ids = [
            int(r["id"].split(":")[1])
            for r in episodes
            if isinstance(r.get("id"), str) and r["id"].startswith("e:")
        ]
        if ep_ids:
            self._increment_retrieval_counts(ep_ids)

        out: dict[str, Any] = {
            "query": query,
            "episodes": [_to_recall_item(r) for r in episodes],
        }
        # Same contract as recall(): key present only when a leg degraded.
        if warnings:
            out["warnings"] = list(warnings)
        served_ids: dict[str, Any] = {
            "episodes": [r["id"] for r in episodes if r.get("id")],
            "n_echo_suppressed": n_echo_suppressed,
            "limit": limit,  # requested ceiling — its distribution tunes the default
        }
        # Same conditional envelope as recall(): keys only when the hook delivered an id.
        if self_session:
            served_ids["self_session"] = self_session
            served_ids["n_self_excluded"] = n_self_excluded
        if session_id:
            served_ids["session_id"] = session_id  # session-scoped drill-down call
        # Shadow abstention floor (telemetry only) — same RAW pre-recency score contract
        # as recall(); the served episodes above are untouched.
        self._floor_shadow(served_ids, float(rerank_top), emb_ok=query_emb is not None)
        chars = _served_chars(out)
        self._record_metrics(
            {
                "kind": "episodes",
                "source": source or "mcp",
                "query": query[:200],
                "ms_total": round((time.perf_counter() - t_start) * 1000.0, 1),
                "n_episodes": len(episodes),
                "chars": chars,
                "est_tokens": chars // 4,
                "rerank_top_score": round(float(rerank_top), 4),
                "emb_ok": query_emb is not None,
                "served_ids": served_ids,
            }
        )
        return out
