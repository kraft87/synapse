"""Best-effort feedback counters and recall telemetry persistence."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import psycopg
from psycopg.types.json import Json as PgJson

logger = logging.getLogger(__name__)

_METRIC_COLS = (
    "kind",
    "source",
    "query",
    "group_id",
    "write_feedback",
    "ms_total",
    "ms_embed",
    "ms_bm25",
    "ms_vector",
    "ms_kg",
    "ms_web",
    "ms_rerank",
    "n_facts",
    "n_episodes",
    "n_entities",
    "n_web",
    "n_history",
    "chars",
    "est_tokens",
    "pool_bm25",
    "pool_vector",
    "pool_fused",
    "kg_candidates",
    "rerank_model",
    "rerank_top_score",
    "emb_ok",
    "n_timeline",
    "ms_timeline",
    "n_prefs",
    "ms_prefs",
    "n_notes",
    "ms_notes",
    "served_ids",
)


class RecallMetricsMixin:
    """Persistence methods mixed into the stateful recall engine."""

    _db_url: str
    _async_executor: Any
    _kg_owner: str

    if TYPE_CHECKING:

        def _ensure_pg(self) -> Any: ...

    def _increment_retrieval_counts(self, episode_ids: list[int]) -> None:
        if not episode_ids:
            return
        connection = self._ensure_pg()
        try:
            placeholders = ",".join(["%s"] * len(episode_ids))
            connection.execute(
                f"UPDATE episodes SET retrieval_count = retrieval_count + 1 WHERE id IN ({placeholders})",
                episode_ids,
            )
        except Exception as error:
            logger.warning("Failed to increment retrieval counts: %s", error)

    def _increment_fact_retrieval_counts(self, edge_uuids: list[str], group_id: str) -> None:
        """Bump surfaced fact counters asynchronously."""
        if edge_uuids:
            self._async_executor.submit(self._do_increment, list(edge_uuids), group_id)

    def _do_increment(self, edge_uuids: list[str], group_id: str) -> None:
        try:
            with psycopg.connect(self._db_url, autocommit=True) as connection:
                connection.execute(
                    "UPDATE kg_relationships "
                    "SET retrieval_count = COALESCE(retrieval_count, 0) + 1 "
                    "WHERE owner_id = %s AND group_id = %s AND uuid = ANY(%s)",
                    (self._kg_owner, group_id, edge_uuids),
                )
        except Exception as error:
            logger.debug("Background fact-bump (PG) failed: %s", error)

    def _record_metrics(self, metrics: dict[str, Any]) -> None:
        """Persist a metrics row asynchronously so recall never waits on telemetry."""
        self._async_executor.submit(self._do_record, metrics)

    def _do_record(self, metrics: dict[str, Any]) -> None:
        try:
            placeholders = ",".join(["%s"] * len(_METRIC_COLS))
            values = [metrics.get(column) for column in _METRIC_COLS]
            served_ids_index = _METRIC_COLS.index("served_ids")
            if values[served_ids_index] is not None:
                values[served_ids_index] = PgJson(values[served_ids_index])
            with psycopg.connect(self._db_url, autocommit=True) as connection:
                connection.execute(
                    f"INSERT INTO recall_metrics ({','.join(_METRIC_COLS)}) VALUES ({placeholders})",
                    values,
                )
        except Exception as error:
            logger.debug("recall_metrics write failed: %s", error)

    def record_event(
        self,
        kind: str,
        *,
        source: str | None = None,
        query: str | None = None,
        group_id: str | None = None,
        ms_total: float | None = None,
        chars: int | None = None,
        est_tokens: int | None = None,
        served_ids: dict[str, Any] | None = None,
    ) -> None:
        """Record telemetry for non-recall callers through the same writer."""
        self._record_metrics(
            {
                "kind": kind,
                "source": source,
                "query": query,
                "group_id": group_id,
                "ms_total": ms_total,
                "chars": chars,
                "est_tokens": est_tokens,
                "served_ids": served_ids,
            }
        )
