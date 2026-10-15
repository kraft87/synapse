"""Postgres-backed knowledge-graph client (#67 PR 3 — FalkorDB decommissioned).

Replaces ``ingestion/falkordb_client.py``. The write-side pipeline talks to the
graph exclusively through this class: mutations go through
``kg_pg_write.KGPostgresWriter`` (strict — a failed write fails the extraction
item so the queue retries it), reads through ``kg_pg_read.KGPostgresReader``.
Both target the ``kg_entities`` / ``kg_relationships`` tables (schema/017),
scoped by ``owner_id`` + ``group_id``.

The ``SYNAPSE_KG_READ`` / ``SYNAPSE_KG_DUAL_WRITE`` env seams from PRs #128/#129
are gone: with FalkorDB decommissioned there is nothing to dispatch to or fall
back on. Read failures now RAISE instead of degrading — same retry semantics
as writes.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from ingestion.kg_pg_read import KGPostgresReader
from ingestion.kg_pg_write import KGPostgresWriter

logger = logging.getLogger(__name__)


def rrf_merge(
    vector_hits: list[dict[str, Any]],
    fulltext_hits: list[dict[str, Any]],
    limit: int = 20,
    k: int = 1,
) -> list[dict[str, Any]]:
    """Rank-based reciprocal rank fusion of two candidate lists.

    Score per uuid = sum over lists of 1/(k + rank), rank 0-indexed.
    Default k=1 matches Graphiti's `rrf()` — steeper than the Cormack k=60,
    appropriate when top-1 dominance matters (contradiction detection).

    Truncation happens AFTER the merge — the long tail is what RRF exists
    to rescue, so do not pre-truncate the input lists. Pass the FULL pool
    from each retrieval modality and let RRF promote items that scored
    moderately well in both.

    Dedupe by uuid; preserve the payload from the first list that yielded it.
    """
    scores: dict[str, float] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for hits in (vector_hits, fulltext_hits):
        for rank, item in enumerate(hits):
            uid = item.get("uuid")
            if not uid:
                continue
            scores[uid] = scores.get(uid, 0.0) + 1.0 / (k + rank)
            payloads.setdefault(uid, item)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [payloads[uid] for uid, _ in ordered[:limit]]


class KGClient:
    def __init__(self) -> None:
        self._writer = KGPostgresWriter()
        self._reader = KGPostgresReader()

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def upsert_node(
        self,
        node_uuid: str,
        name: str,
        entity_type: str,
        summary: str,
        group_id: str,
        project: str | None = None,
        embedding: list[float] | None = None,
        supertype: str | None = None,
    ) -> str:
        """Create or update a node. Returns the node UUID.

        ``normalized_name`` (lowercase, whitespace-collapsed, leading/trailing
        punctuation stripped) is recomputed on every write so
        ``ingestion.dedup.NodeDeduper`` can serve the exact-name short-circuit
        via an indexed lookup, and a rename propagates to the dedup index
        immediately.
        """
        from ingestion.dedup import _normalize_name

        now = datetime.now(UTC).isoformat()
        self._writer.upsert_node(
            uuid=node_uuid,
            name=name,
            normalized_name=_normalize_name(name),
            entity_type=entity_type,
            summary=summary,
            group_id=group_id,
            project=project,
            created_at=now,
            valid_at=now,
            embedding=embedding,
            supertype=supertype,
        )
        return node_uuid

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def invalidate_edges_batch(
        self,
        items: list[tuple[str, str | None]],
        group_id: str,
        invalidated_by: str | None = None,
    ) -> None:
        """Batch-invalidate edges in a single writer round-trip.

        ``items`` is a list of ``(edge_uuid, invalid_at)`` pairs. ``invalid_at``
        of None falls back to now() per-item so callers can mix "invalidate as
        of fact-extracted date" rows with "invalidate as of right now" rows in
        the same call. Empty list is a no-op.

        ``invalidated_by`` (the superseding edge's uuid) applies to every item in
        the call (schema 028) — callers group same-superseder edges into one call.
        """
        if not items:
            return
        now = datetime.now(UTC).isoformat()
        self._writer.invalidate_edges(
            [(u, ts or now) for u, ts in items], group_id, invalidated_by=invalidated_by
        )

    def create_edges_batch(
        self,
        rows: list[dict[str, Any]],
        group_id: str,
    ) -> None:
        """Batch-create relationship edges in one writer round-trip.

        Each row must already contain:
          - ``src``: source entity uuid
          - ``tgt``: target entity uuid
          - ``edge_uuid``: pre-generated UUID for the new edge
          - ``name``: relationship name
          - ``fact``: natural-language fact text
          - ``episodes``: list of episode IDs
          - ``created_at`` / ``t_created`` (timestamps)
          - ``valid_at`` / ``t_valid`` (timestamps)
          - optional ``emb``: list[float] fact embedding
          - optional ``web_artifact_id``: web-lane provenance (task #68).

        Edges are never born invalid: ``t_invalid`` is set only by a later
        fact superseding them (``invalidate_edges_batch``).
        """
        if not rows:
            return
        self._writer.create_edges(rows, group_id)

    def reinforce_edges(self, items: list[tuple[Any, ...]], group_id: str) -> None:
        """Capture dedup hits: bump ``mention_count`` + union source episodes on
        each existing edge a newly-extracted duplicate fact matched. ``items`` is
        a list of ``(edge_uuid, source_episode_ids)`` or, with schema-058 support
        evidence, ``(edge_uuid, source_episode_ids, support_at, support_episode)``
        where a non-None ``support_at`` (a USER-attributed re-assertion) refreshes
        the edge's ``last_supported_at``. Empty list is a no-op."""
        if not items:
            return
        self._writer.reinforce_edges(items, group_id)

    def delete_orphan_entities(self, min_age_hours: int = 24) -> int:
        """Delete edge-less entities older than ``min_age_hours``; returns the count."""
        return self._writer.delete_orphan_entities(min_age_hours)

    # ------------------------------------------------------------------
    # Reads (entity resolution, dedup, contradiction + dedup pools)
    # ------------------------------------------------------------------

    def find_similar_nodes(
        self,
        query_embedding: list[float],
        group_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return top-k nodes by embedding similarity. Score is cosine distance (0=identical)."""
        hits: list[dict[str, Any]] = self._reader.find_similar_nodes(
            query_embedding, group_id, limit=limit
        )
        return hits

    def entity_uuid_by_normalized_name(self, normalized: str, group_id: str) -> str | None:
        """Exact normalized-name entity lookup (NodeDeduper short-circuit).

        Falls back to a lower(trim(name)) equality check for pre-migration
        rows that lack the indexed property.
        """
        result: str | None = self._reader.entity_uuid_by_normalized_name(normalized, group_id)
        return result

    def load_entities(self, group_id: str) -> list[tuple[str, str, str, str | None]]:
        """(uuid, name, summary, entity_supertype) for every entity in the group (LSH hydrate)."""
        result: list[tuple[str, str, str, str | None]] = self._reader.load_entities(group_id)
        return result

    def load_type_map(self) -> dict[str, str]:
        """subtype -> supertype from kg_entity_types (entity-dedup type gate)."""
        result: dict[str, str] = self._reader.load_type_map()
        return result

    def find_similar_edges(
        self,
        fact_embedding: list[float],
        group_id: str,
        distance_threshold: float = 0.20,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Return active edges whose fact_embedding is within distance_threshold.

        distance_threshold=0.20 corresponds to cosine similarity >= 0.80.
        """
        hits: list[dict[str, Any]] = self._reader.find_similar_edges(
            fact_embedding, group_id, distance_threshold=distance_threshold, limit=limit
        )
        return hits

    def nearest_live_edges(
        self,
        fact_embeddings: list[list[float]],
        group_id: str,
        max_distance: float,
        limit: int,
    ) -> list[list[dict[str, Any]]]:
        """Per embedding, its nearest live edges within ``max_distance`` (cosine
        distance), nearest first, in one round-trip (Stage 6c restatement recheck)."""
        hits: list[list[dict[str, Any]]] = self._reader.nearest_live_edges(
            fact_embeddings, group_id, max_distance=max_distance, limit=limit
        )
        return hits

    def find_edges_by_pair(
        self,
        source_uuid: str,
        target_uuid: str,
        group_id: str,
    ) -> list[dict[str, Any]]:
        """Return active edges between a specific source/target pair.

        Live edges only: already-invalidated edges must NOT re-enter the
        candidate pool, or Stage 6b could re-contradict an edge that was
        already retired.
        """
        hits: list[dict[str, Any]] = self._reader.find_edges_by_pair(
            source_uuid, target_uuid, group_id
        )
        return hits

    def find_edges_by_fulltext(
        self,
        query: str,
        group_id: str,
        limit: int = 20,
        fact_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """BM25 candidates over live fact text (ParadeDB, OR semantics).

        No score floor: rank-only RRF (k=1) plus the stage-6b LLM confirm gate
        the pool instead (see ``kg_pg_read.find_edges_by_fulltext``). Pass
        ``fact_embedding`` to get each hit's cosine ``dist`` to it.
        """
        hits: list[dict[str, Any]] = self._reader.find_edges_by_fulltext(
            query, group_id, limit=limit, fact_embedding=fact_embedding
        )
        return hits
