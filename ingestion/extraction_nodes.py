"""Entity extraction, resolution, and node persistence stages."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from ingestion.extraction_entities import DeterministicExtractor, EntityResolver
from ingestion.extraction_llm import LLMExtractor
from ingestion.models import ExtractedEntity, ExtractedFact, ExtractionResult

logger = logging.getLogger(__name__)


@dataclass
class PendingNodes:
    """One group's NEW entities, held back until Stage 7's write decision is known.

    ``entities`` are the ones Stage 4 resolved to a "new:" uuid; the rest is what
    ``_stage5_write_nodes`` needs to write them."""

    entities: list[ExtractedEntity]
    project: str | None
    embeddings: dict[str, list[float]]
    deduper: Any | None


class ExtractionNodesMixin:
    _det: DeterministicExtractor
    _llm: LLMExtractor
    _resolver: EntityResolver
    _kg: Any
    _dedupers: dict[str, Any]

    def _stage2_deterministic(self, episodes: list[dict[str, Any]]) -> list[ExtractedEntity]:
        import logfire

        with logfire.span("stage2_deterministic episodes={n}", n=len(episodes)) as span:
            entities = self._det.extract(episodes)
            span.set_attribute("entities", len(entities))
            return entities

    def _stage3_llm(
        self,
        summary: str,
        det_entities: list[ExtractedEntity],
        session_date: str | None = None,
    ) -> ExtractionResult:
        import logfire

        with logfire.span(
            "stage3_llm_extract summary_chars={chars} det={det}",
            chars=len(summary),
            det=len(det_entities),
        ) as span:
            result = self._llm.extract(
                summary=summary, context_entities=det_entities, session_date=session_date
            )
            span.set_attribute("entities", len(result.entities))
            span.set_attribute("facts", len(result.facts))
            return result

    def _stage4_resolve(
        self,
        entities: list[ExtractedEntity],
        group_id: str,
        deduper: Any | None = None,
    ) -> dict[str, str]:
        import logfire

        with logfire.span(
            "stage4_resolve grp={grp} entities={n}",
            grp=group_id,
            n=len(entities),
        ):
            return self._resolver.resolve(entities, self._kg, group_id, deduper=deduper)

    def _stage5_write_nodes(
        self,
        entities: list[ExtractedEntity],
        uuid_map: dict[str, str],
        project: str | None,
        group_id: str,
        embeddings: dict[str, list[float]] | None = None,
        deduper: Any | None = None,
    ) -> None:
        from ingestion.dedup import NodeDeduper

        for entity in entities:
            raw_uuid = uuid_map.get(entity.name, f"new:{uuid.uuid4()}")
            is_new = raw_uuid.startswith("new:")
            clean_uuid = raw_uuid.removeprefix("new:")
            emb = (embeddings or {}).get(entity.name)

            # Same-name guard: Stage 4 resolved this name as new, but another worker (or
            # an earlier entity of this batch with the same normalized name) may have
            # written it since. Handle it as the exact-name hit Stage 4 would find now:
            # route the batch's edges to that entity and update it instead of inserting
            # a second one.
            if is_new and deduper is not None:
                existing = deduper.exact_match(entity.name)
                if existing is not None:
                    uuid_map[entity.name] = existing
                    is_new, clean_uuid = False, existing

            # When dedup matched an EXISTING entity, prefer the longer
            # summary so the more-detailed text survives. Without this
            # the freshly-extracted (often shorter) summary would
            # overwrite the canonical one. ``merge_summary`` is the same
            # rule used by the nightly dream pipeline.
            summary_to_write = entity.summary
            if not is_new:
                existing_summary = ""
                if deduper is not None:
                    existing_summary = deduper.summary_of(clean_uuid)
                summary_to_write = NodeDeduper.merge_summary(existing_summary, entity.summary)

            # Auto-type: roll the extracted subtype up to a canonical supertype via the
            # taxonomy map (already loaded on the deduper). Unknown subtype -> 'other'
            # (queryable as the to-map backlog); no map -> None (backfill fills later).
            supertype = None
            type_map = deduper.type_map if deduper is not None else {}
            if type_map:
                supertype = type_map.get(entity.type, "other")
            self._kg.upsert_node(
                node_uuid=clean_uuid,
                name=entity.name,
                entity_type=entity.type,
                summary=summary_to_write,
                group_id=group_id,
                project=project,
                embedding=emb,
                supertype=supertype,
            )
            # Register the freshly-INSERTED node in the deduper so any
            # later extraction in the same run dedupes against it instead
            # of writing a duplicate. Updates (non-new) are already in
            # the deduper's exact-name index from its initial hydration.
            if is_new and deduper is not None:
                deduper.register(entity.name, clean_uuid, entity.summary)

    def _stage5_write_new_nodes(
        self,
        facts: list[ExtractedFact],
        uuid_map: dict[str, str],
        skip_indices: set[int],
        group_id: str,
        pending: PendingNodes,
    ) -> None:
        """Write the group's NEW entities that a fact Stage 7 will create links, no others.

        Runs right after Stage 6c, inside Stage 7 before its edge writes, with the same
        create guard Stage 7 applies (not skipped, both endpoints resolved). A new entity whose facts were all
        dropped (cross-group, same-batch twin, 6b or 6c duplicate, unresolved endpoint) is
        never written, so it cannot be left without an edge. ``uuid_map`` is updated in
        place when the same-name guard routes a name to an entity written meanwhile."""
        linked: set[str] = set()
        for idx, fact in enumerate(facts):
            if idx in skip_indices or not (uuid_map.get(fact.source) and uuid_map.get(fact.target)):
                continue
            linked.update((fact.source, fact.target))
        to_write = [
            e
            for e in pending.entities
            if e.name in linked and uuid_map.get(e.name, "").startswith("new:")
        ]
        held = len(pending.entities) - len(to_write)
        if held:
            logger.debug("Held back %d new entities no written fact links", held)
        if to_write:
            self._stage5_write_nodes(
                to_write,
                uuid_map,
                pending.project,
                group_id,
                pending.embeddings,
                deduper=pending.deduper,
            )

    def _deduper_for(self, group_id: str) -> Any:
        """Cached per-group NodeDeduper (thin shell; see ingestion.dedup).

        The expensive part — the O(all entities) LSH index — is no longer
        on the deduper: it lives in a process-shared cache inside the dedup
        module, one copy per group per PROCESS instead of one per worker
        thread (4 threads x ~366MB of index copies is what OOM-killed the
        pollers against their 1GiB caps). The shared index carries its own
        TTL (SYNAPSE_DEDUP_CACHE_TTL_SECONDS) bounding staleness from other
        processes' writes — and only for the fuzzy-name assist, since the
        exact-name short-circuit and the embedding-similarity candidates
        always query the live DB. This per-pipeline cache just avoids
        re-constructing the shell (client refs + config) per item.
        """
        from ingestion.dedup import NodeDeduper

        if group_id not in self._dedupers:
            self._dedupers[group_id] = NodeDeduper(
                kg_client=self._kg,
                group_id=group_id,
                llm_client=self._llm._client,
            )
        return self._dedupers[group_id]

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------
