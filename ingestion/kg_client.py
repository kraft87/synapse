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

Orchestration preserved verbatim from the FalkorDB client: bi-temporal
edge-date enrichment, the write-time contradiction safety net, and the batched
self-invalidation pass for facts that arrive already bookended ("from 2020 to
2022"). Only the storage calls underneath changed.
"""

from __future__ import annotations

import logging
import uuid
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

    def create_edge(
        self,
        source_uuid: str,
        target_uuid: str,
        relationship: str,
        fact: str,
        episode_ids: list[int],
        group_id: str,
        fact_embedding: list[float] | None = None,
        *,
        t_valid: str | None = None,
        t_invalid: str | None = None,
        detector: Any | None = None,
        extracted_fact: Any | None = None,
        edge_date_extractor: Any | None = None,
        reference_time: str | None = None,
    ) -> str:
        """Create a relationship edge. Returns the new edge UUID.

        Bi-temporal model (Phase 3):
        - ``t_created`` (== ``created_at``) -- when the edge was written. Set
          on every create. Never updated.
        - ``t_valid``   (== ``valid_at``)   -- when the fact began being true.
          Defaults to now() when the caller does not pass an explicit value;
          callers extracting an LLM-supplied valid date should pass it via
          ``t_valid``.
        - ``t_invalid`` (== ``invalid_at``) -- when the fact stopped being
          true. NULL on create; set later via ``invalidate_edge`` when a
          contradiction is confirmed.
        - ``t_expired`` -- soft-delete marker for facts the system decides
          are no longer relevant even if not contradicted. NULL on create;
          populated by future selective-forgetting work in the dream
          pipeline.

        When ``detector`` is supplied, the bi-temporal contradiction safety
        net runs BEFORE the create: any live edge it confirms the new fact
        contradicts has its ``t_invalid`` set in-place. The new edge is then
        always written -- contradictions invalidate the *old* edge, never
        block the new one.

        ``edge_date_extractor`` (Phase 4) is an optional
        ``EdgeDateExtractor`` that runs only when ``t_valid`` was NOT
        supplied by the caller. It pulls ``valid_at`` and ``invalid_at``
        from the fact text itself using the verbatim Graphiti
        ``extract_edge_dates`` prompt. Failures (LLM error, malformed JSON,
        empty result) fall through silently to the legacy "now()" default
        so a write is never blocked by date extraction.
        """
        now = datetime.now(UTC).isoformat()
        edge_uuid = str(uuid.uuid4())

        # Phase 4 bi-temporal enrichment. When the caller pre-computed
        # ``t_valid`` / ``t_invalid`` via a batched extractor (PR #38) we
        # use those and skip the per-fact LLM call entirely. Otherwise fall
        # back to the per-fact ``edge_date_extractor`` (best-effort: both
        # fields fall back to None on failure and we then use now() for
        # valid_at — legacy behavior preserved).
        invalid_at_ts: str | None = t_invalid
        if t_valid is None and edge_date_extractor is not None:
            try:
                v, i = edge_date_extractor.extract(fact, reference_time=reference_time)
            except Exception:
                v, i = (None, None)
            if v:
                t_valid = v
            if i and invalid_at_ts is None:
                invalid_at_ts = i

        valid_at_ts = t_valid or now

        # Bi-temporal safety net: invalidate confirmed-contradicted live
        # edges BEFORE writing the new one. Detector failures never block
        # the write -- the detector swallows exceptions and returns [].
        if detector is not None and extracted_fact is not None:
            contradicted_uuids = detector.detect_contradictions(
                extracted_fact,
                source_uuid,
                target_uuid,
                group_id,
                fact_embedding=fact_embedding,
            )
            for old_uuid in contradicted_uuids:
                # Record the superseder: edge_uuid is the new edge that contradicted old_uuid.
                self.invalidate_edge(old_uuid, group_id, by_uuid=edge_uuid)

        self._writer.create_edges(
            [
                {
                    "src": source_uuid,
                    "tgt": target_uuid,
                    "edge_uuid": edge_uuid,
                    "name": relationship,
                    "fact": fact,
                    "episodes": episode_ids,
                    "created_at": now,
                    "t_created": now,
                    "valid_at": valid_at_ts,
                    "t_valid": valid_at_ts,
                    "emb": fact_embedding,
                }
            ],
            group_id,
        )

        # Phase 4: if the edge-date extractor came back with an explicit
        # invalid_at, mark the edge as already-invalidated. The fact was
        # ALREADY contradicted at write time (e.g. "a person worked at X from
        # 2020 to 2022") — preserve that lifecycle bookend immediately
        # rather than wait for a future contradiction to flip it.
        if invalid_at_ts:
            self.invalidate_edge(edge_uuid, group_id, invalid_at=invalid_at_ts)

        return edge_uuid

    def invalidate_edge(
        self,
        edge_uuid: str,
        group_id: str,
        invalid_at: str | None = None,
        by_uuid: str | None = None,
    ) -> None:
        """Set invalid_at on an edge (marks it as contradicted/superseded).

        Bi-temporal (Phase 3): writes BOTH the legacy ``invalid_at`` AND the
        canonical ``t_invalid`` so existing readers and the future t_*-only
        readers both see the invalidation. Edges are NEVER deleted -- this
        is the only mutation on the bi-temporal lifecycle (apart from the
        future ``t_expired`` soft-delete marker).

        ``by_uuid`` records the superseding edge (schema 028); set only by the
        contradiction path, where the new edge that killed this one is known.
        """
        ts = invalid_at or datetime.now(UTC).isoformat()
        self._writer.invalidate_edges([(edge_uuid, ts)], group_id, invalidated_by=by_uuid)

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
    ) -> list[str]:
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
          - optional ``t_invalid``: pre-known invalidation timestamp (e.g. fact
            text said "from 2020 to 2022") — applied via a follow-up
            ``invalidate_edges_batch`` so the bi-temporal lifecycle bookend
            is recorded immediately.
          - optional ``web_artifact_id``: web-lane provenance (task #68).

        Returns the list of edge UUIDs that were written, in input order.
        """
        if not rows:
            return []
        self._writer.create_edges(rows, group_id)

        # Apply any pre-known invalidations in one more round-trip. This
        # collapses the create_edge() per-fact "edge was already contradicted
        # at write time" pattern into the batched path.
        self_invalidations: list[tuple[str, str | None]] = [
            (r["edge_uuid"], r.get("t_invalid")) for r in rows if r.get("t_invalid")
        ]
        if self_invalidations:
            self.invalidate_edges_batch(self_invalidations, group_id)

        return [r["edge_uuid"] for r in rows]

    def reinforce_edges(self, items: list[tuple[str, list[int]]], group_id: str) -> None:
        """Capture dedup hits: bump ``mention_count`` + union source episodes on
        each existing edge a newly-extracted duplicate fact matched. ``items`` is
        a list of ``(edge_uuid, source_episode_ids)``. Empty list is a no-op."""
        if not items:
            return
        self._writer.reinforce_edges(items, group_id)

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

    def find_edges_by_pair(
        self,
        source_uuid: str,
        target_uuid: str,
        group_id: str,
    ) -> list[dict[str, Any]]:
        """Return active edges between a specific source/target pair.

        ContradictionDetector relies on this call returning ONLY live edges;
        already-invalidated edges must NOT re-enter the candidate pool, or
        we could re-contradict an edge that was already retired.
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
    ) -> list[dict[str, Any]]:
        """BM25 candidates over live fact text (ParadeDB, OR semantics).

        No score floor: rank-only RRF (k=1) plus the stage-6b LLM confirm gate
        the pool instead (see ``kg_pg_read.find_edges_by_fulltext``).
        """
        hits: list[dict[str, Any]] = self._reader.find_edges_by_fulltext(
            query, group_id, limit=limit
        )
        return hits
