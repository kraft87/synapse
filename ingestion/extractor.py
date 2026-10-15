from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ingestion.models import (
    ExtractedEntity,
    ExtractedFact,
)
from ingestion.scope import active_groups

if TYPE_CHECKING:
    from ingestion.embedding import EmbeddingModel

from ingestion.extraction_edges import ExtractionEdgesMixin
from ingestion.extraction_entities import DeterministicExtractor, EntityResolver
from ingestion.extraction_llm import LLMExtractor
from ingestion.extraction_nodes import ExtractionNodesMixin, PendingNodes
from ingestion.extraction_policy import (
    _apply_canonical_aliases,
    _classify_entity_group,
    _default_group_for_project,
    _reconcile_fact_groups,
)
from ingestion.extraction_recheck import recheck_restated
from ingestion.extraction_twins import collapse_batch_twins
from ingestion.localtime import local_date, local_iso

logger = logging.getLogger(__name__)


class ExtractionPipeline(ExtractionNodesMixin, ExtractionEdgesMixin):
    """Orchestrates the full extraction pipeline for a single queue item."""

    def __init__(
        self,
        db: Any,
        llm_client: Any,
        embedder: EmbeddingModel,
        kg_client: Any,
        llm_model: str | None = None,
    ) -> None:
        from ingestion.edge_dates import EdgeDateExtractor
        from ingestion.llm_client import DEFAULT_MODEL, stage_model

        # Per-stage model resolution (issue #8): SYNAPSE_<STAGE>_MODEL env
        # beats SYNAPSE_LLM_MODEL beats the ``llm_model`` param / code default.
        base = llm_model or DEFAULT_MODEL

        self._db = db
        self._det = DeterministicExtractor()
        self._llm = LLMExtractor(llm_client=llm_client, model=stage_model("EXTRACTOR", base))
        self._resolver = EntityResolver(embedder=embedder, llm_client=llm_client)
        # Stage-6b write-time contradiction/duplicate confirm calls.
        self._contradiction_model = stage_model("CONTRADICTION", base)
        self._kg = kg_client
        self._embedder = embedder
        # LLM-driven temporal-bounds extractor (Graphiti's `extract_timestamps`):
        # reads valid_at out of the fact text. A response that never validates
        # falls back to the segment time; an outage fails the item.
        self._edge_date_extractor = EdgeDateExtractor(
            llm_client=llm_client,
            model=stage_model("EDGE_DATES", base),
        )
        # Per-group NodeDeduper cache. Dedupers are thin shells over a
        # process-shared hydrated index (see ingestion.dedup._GroupIndex),
        # so this cache only avoids re-constructing the shell per item;
        # index build/TTL policy lives in the dedup module.
        self._dedupers: dict[str, Any] = {}
        # Timeline chat gate (schema 033): per-turn "did something happen?" check on
        # episode-type items -> naked dated events in timeline_events. Fail-soft and
        # env-gated (SYNAPSE_TIMELINE_GATE=0); orthogonal to KG extraction.
        from ingestion.timeline_gate import TimelineGate

        self._timeline_gate = TimelineGate(
            db=db, llm_client=llm_client, embedder=embedder, model=stage_model("TIMELINE", base)
        )
        # Preferences chat gate (schema 035): per-turn "did the user assert a durable
        # preference?" check on episode-type items -> reconciled rows in `preferences`.
        # Same fail-soft, env-gated (SYNAPSE_PREFS_GATE=0) shape as the timeline gate;
        # kept out of the KG so preferences don't rebuild the User-supernode.
        from ingestion.preferences_gate import PreferencesGate

        self._preferences_gate = PreferencesGate(
            db=db, llm_client=llm_client, embedder=embedder, model=stage_model("PREFERENCES", base)
        )

    # ------------------------------------------------------------------
    # Stage methods
    # ------------------------------------------------------------------

    def process_item(self, item: dict[str, Any]) -> None:
        """Process one extraction queue item. Raises on failure (caller handles retry).

        Group routing: each entity is assigned to either the technical or personal
        graph via _classify_entity_group (project tag → item default, then a per-entity
        override from its name), or to technical alone when the personal scope is off. A
        fact whose endpoints disagree moves the odd one into the item's default group
        (_reconcile_fact_groups), so an edge always has both endpoints in one graph.
        Entities are partitioned by group and resolved/written independently against
        their target graph.
        """
        content_type: str = item["content_type"]
        project: str | None = item.get("project")
        session_id: str | None = item.get("session_id")
        default_group = _default_group_for_project(project)
        web_provenance: dict[str, Any] | None = None

        # What the item extracts is read from the row it names: the queue stores no text
        # for an episode, a remember() note or (once the chunk_id migration is applied) a
        # chunk, so a turn deleted since it was queued is gone here too. Drill-back: the
        # turn's own id for an episode or note, the window's source episodes for a chunk.
        episode_ids: list[int] = []
        content: str = item["content"]
        if item.get("episode_id"):
            episode_ids = [item["episode_id"]]
            episode_text = self._db.get_episode_content(item["episode_id"])
            if episode_text is None:
                logger.info(
                    "queue item %s: its episode is gone; nothing to extract", item.get("id")
                )
                return
            content = episode_text
        elif content_type == "chunk":
            # A database error fails the item (the queue retries it): swallowing it wrote
            # edges with no provenance and t_valid=now.
            if item.get("chunk_id"):
                chunk = self._db.get_chunk(item["chunk_id"])
                if chunk is not None:
                    content, episode_ids = chunk["content"], chunk["episode_ids"]
            elif session_id:
                # A row enqueued before the chunk_id column carries the chunk's text.
                episode_ids = self._db.get_chunk_episode_ids(session_id, content)
            if not episode_ids:
                # Deleted with its episodes, or rebuilt: its facts would have no provenance.
                logger.info("queue item %s: its chunk is gone; nothing to extract", item.get("id"))
                return
        item = {**item, "content": content}

        # Per-turn gates ride the episode item (a chunk spans turns and would blur an
        # event's date or a preference's author). They emit no KG facts, so there is
        # nothing else to do for the turn.
        if content_type == "episode":
            self._timeline_gate.process(item)
            self._preferences_gate.process(item)
            return

        # Segment date (conversation time, NOT ingest wall-clock) = the source episodes'
        # latest created_at. Resolved BEFORE Stage 3, so the extraction prompt can anchor
        # in-text date mentions against it; reused below as the default fact valid-time.
        segment_valid_at = self._db.get_episodes_valid_at(episode_ids) if episode_ids else None
        # The user's calendar day, not the UTC one: a turn at 20:30 EDT is stored as
        # 00:30 UTC the next day.
        session_date = local_date(segment_valid_at) if segment_valid_at else None

        # Stage 2: deterministic extraction, from the item's OWN text — not a full-session
        # fetch per item — then full LLM extraction on it.
        if content_type in ("chunk", "manual"):
            # Chunk = a 3-5 turn window (task #63); manual = one remember() note.
            det_entities = self._stage2_deterministic([{"content": content, "metadata": {}}])
            llm_result = self._stage3_llm(content, det_entities, session_date)
        elif content_type == "web_chunk":
            # Web chunk = ~400-token slice of a scraped page or research brief
            # (task #68). Third-party content: extraction uses the web prompt
            # variant (attribution firewall + closed type vocab + salience bar).
            # No episode backlink — provenance is the parent web_artifact,
            # carried onto edges via the kg_shadow mirror below.
            if item.get("web_chunk_id"):
                web_provenance = self._db.get_web_chunk_provenance(item["web_chunk_id"])
            if web_provenance is None:
                logger.warning(
                    "web_chunk queue item %s has no resolvable provenance; skipping",
                    item.get("id"),
                )
                return
            det_entities = self._stage2_deterministic([{"content": content, "metadata": {}}])
            llm_result = self._llm.extract_web(content, det_entities, web_provenance)
        else:
            raise ValueError(f"queue item {item.get('id')}: unknown content_type {content_type!r}")

        all_entities = det_entities + [
            e for e in llm_result.entities if e.name not in {x.name for x in det_entities}
        ]

        # Task #49: canonicalize identity aliases (User / full-name spellings -> owner hub)
        # before any filtering or resolution, so every downstream consumer
        # (orphan filter, group classification, embeddings, uuid_map, edges)
        # sees only the canonical name.
        all_entities = _apply_canonical_aliases(all_entities, llm_result.facts)

        # --- Pre-resolve orphan filter --------------------------------------
        # Only entities that appear as the source or target of an extracted
        # fact can be written at all: an existing one is updated in Stage 5
        # below, a new one only when Stage 7 will create a fact linking it
        # (_stage5_write_new_nodes, per group, right before Stage 7 writes). But the
        # deterministic extractor emits hundreds-to-thousands of entity
        # mentions per summary (file paths, URLs, identifiers): on a real
        # corpus summary that's ~1300 entities backing only ~6-12 facts.
        # Resolving every one of them in Stage 4 (per-entity vector search +
        # up to 4 LLM "same entity?" confirms, each a ~30s claude-CLI
        # subprocess) cost ~90 min/summary -- almost all of it spent on
        # entities no fact ever references, only to be dropped before write.
        # Prune to the fact-referenced set HERE, before Stage 4, so we resolve
        # dozens not thousands. det_entities still informed Stage 3 extraction
        # (above); we only drop them from the resolve/write path. Facts
        # reference entities by name, so the filter is name-keyed. Episodes
        # produce no facts -> empty set -> nothing resolved (they already
        # wrote zero nodes via the post-resolve orphan-drop; this just skips
        # the wasted resolution).
        referenced_names = {f.source for f in llm_result.facts} | {
            f.target for f in llm_result.facts
        }
        all_entities = [e for e in all_entities if e.name in referenced_names]

        # Per-entity group classification (name-based override of item default), then one
        # graph per fact: decided before Stage 4 so both endpoints resolve against it.
        entity_groups: dict[str, str] = {
            e.name: _classify_entity_group(e.name, default_group) for e in all_entities
        }
        moved = _reconcile_fact_groups(llm_result.facts, entity_groups, default_group)
        if moved:
            logger.info(
                "Moved %d entit%s into the %s group so the facts joining them keep both "
                "endpoints in one graph",
                moved,
                "y" if moved == 1 else "ies",
                default_group,
            )

        # Filled by Stage 4 with the vector of every name the dedup did not settle exactly,
        # for its vector search and for Stage 5 to store on the entities that are new.
        entity_embeddings: dict[str, list[float]] = {}

        # Resolve nodes separately per group, defer the write until after the
        # orphan-drop pass below. Resolving (Stage 4) first is safe — it only
        # reads from the graph. Writing (Stage 5) is what we need to gate.
        #
        # Per-group ``NodeDeduper`` instances are cached on the Extractor
        # (``_deduper_for``) so the LSH index — an O(all entities) MinHash
        # build — is constructed once per worker and reused across items.
        # ``register`` keeps the in-memory index in sync after Stage 5
        # writes each new node, so repeat names dedupe against
        # freshly-inserted nodes both within an item and across items, and
        # the same-name guard in ``_stage5_write_nodes`` re-checks the name
        # right before each insert.
        from ingestion.dedup import NodeDeduper

        uuid_map: dict[str, str] = {}
        grp_entities_map: dict[str, list[ExtractedEntity]] = {}
        dedupers: dict[str, NodeDeduper] = {}
        for grp in active_groups():
            grp_entities = [e for e in all_entities if entity_groups[e.name] == grp]
            if not grp_entities:
                continue
            dedupers[grp] = self._deduper_for(grp)
            grp_uuid_map = self._stage4_resolve(
                grp_entities, grp, deduper=dedupers[grp], embeddings=entity_embeddings
            )
            uuid_map.update(grp_uuid_map)
            grp_entities_map[grp] = grp_entities

        # --- Orphan-drop: skip writes for entities no fact references ---
        # Mirrors Graphiti's combined_extraction.py:280-295. Before we burn a
        # graph write per entity, check that the LLM extractor actually
        # produced a fact referencing it. Without this filter the
        # deterministic extractor + LLM extractor produce ~92% zero-edge
        # orphan nodes. A referenced fact can still be dropped later
        # (same-batch twin, 6b/6c duplicate, unresolved endpoint), which used to
        # leave a third of the fresh entities edge-less: so NEW entities are held
        # back here and written per group right before Stage 7 writes edges, only
        # for the facts it creates. Maintenance.sweep_orphan_entities stays as the daily safety net
        # (edges deleted with their episode, a crash between the two writes).
        referenced_uuids: set[str] = set()
        for fact in llm_result.facts:
            src_uuid = uuid_map.get(fact.source)
            tgt_uuid = uuid_map.get(fact.target)
            if src_uuid:
                referenced_uuids.add(src_uuid.removeprefix("new:"))
            if tgt_uuid:
                referenced_uuids.add(tgt_uuid.removeprefix("new:"))

        # Stage 5 — only entities whose resolved UUID shows up as the source or
        # target of some fact in the same response. EXISTING ones get their
        # summary-merge update now; NEW ones wait in ``new_nodes`` for their group's
        # Stage 7 write decision.
        orphan_count = 0
        new_nodes: dict[str, PendingNodes] = {}
        for grp, grp_entities in grp_entities_map.items():
            grp_uuid_map = {e.name: uuid_map[e.name] for e in grp_entities if e.name in uuid_map}
            kept_entities = [
                e
                for e in grp_entities
                if grp_uuid_map.get(e.name, "").removeprefix("new:") in referenced_uuids
            ]
            orphan_count += len(grp_entities) - len(kept_entities)
            fresh = [e for e in kept_entities if grp_uuid_map[e.name].startswith("new:")]
            new_nodes[grp] = PendingNodes(fresh, project, entity_embeddings, dedupers.get(grp))
            existing = [e for e in kept_entities if e not in fresh]
            if not existing:
                continue
            # Matched entities only: nothing is inserted here, so nothing to register.
            self._stage5_write_nodes(
                existing,
                grp_uuid_map,
                project,
                grp,
                entity_embeddings,
                deduper=dedupers.get(grp),
            )

        if orphan_count:
            logger.info(
                "Dropped %d orphan entit%s (no fact references resolved UUID)",
                orphan_count,
                "y" if orphan_count == 1 else "ies",
            )

        if not llm_result.facts:
            return

        # Partition facts by the group of their (source, target) entity pair: the same
        # group for both, after the reconcile above, unless an endpoint is no entity.
        facts_by_group: dict[str, list[ExtractedFact]] = {g: [] for g in active_groups()}
        dropped = 0
        for fact in llm_result.facts:
            src_grp = entity_groups.get(fact.source)
            tgt_grp = entity_groups.get(fact.target)
            if src_grp and tgt_grp and src_grp == tgt_grp and src_grp in facts_by_group:
                facts_by_group[src_grp].append(fact)
            else:
                dropped += 1
        if dropped:
            logger.info("Dropped %d fact(s) whose endpoints resolve to no common group", dropped)

        # Web provenance → edge attrs: the artifact id rides every created edge
        # (kg_shadow mirror column), and the page's published/fetched date is the
        # default t_valid for facts whose text carries no date of its own — web
        # claims age with their source, not with ingestion time.
        web_artifact_id: int | None = None
        default_valid_at: str | None = None
        reference_time: str | None = None
        if web_provenance:
            web_artifact_id = web_provenance.get("web_artifact_id")
            dt = web_provenance.get("published_at") or web_provenance.get("fetched_at")
            if dt is not None:
                default_valid_at = dt.isoformat() if hasattr(dt, "isoformat") else str(dt)
            reference_time = default_valid_at
        else:
            # Conversation facts: default valid-time = the SEGMENT's own timestamp
            # (max created_at of its source episodes), NOT ingest wall-clock. Without
            # this a fact carrying no in-text date got t_valid=now() — coincidentally
            # right for live ingestion (now ≈ conversation time) but wrong for any
            # backfilled/retro transcript. Mirrors how web provenance dates its facts.
            # Computed once up-front (segment_valid_at) and reused here.
            default_valid_at = segment_valid_at
            # Same instant on the user's wall clock, so "yesterday" resolves against
            # the local day the session date names, not the UTC one.
            reference_time = local_iso(segment_valid_at) if segment_valid_at else None

        # Stage 6 + 7 run separately per group — same code path, different graph.
        # reference_time is the relative-date anchor (the segment timestamp), so
        # "last week" resolves against the conversation, not ingest.
        # A group with no facts writes none of its new entities.
        for grp in active_groups():
            grp_facts = facts_by_group[grp]
            if not grp_facts:
                continue
            self._process_facts_for_group(
                grp_facts,
                uuid_map,
                episode_ids,
                grp,
                web_artifact_id=web_artifact_id,
                default_valid_at=default_valid_at,
                reference_time=reference_time,
                new_nodes=new_nodes.get(grp),
            )

    def _process_facts_for_group(
        self,
        facts: list[ExtractedFact],
        uuid_map: dict[str, str],
        episode_ids: list[int],
        group_id: str,
        web_artifact_id: int | None = None,
        default_valid_at: str | None = None,
        reference_time: str | None = None,
        new_nodes: PendingNodes | None = None,
    ) -> None:
        """Stage 6 + 7 for one group's facts (extracted from process_item to keep it readable).

        ``new_nodes`` are the group's entities Stage 4 resolved as new; only those a fact
        Stage 7 creates links are written, right before its edges."""
        import logfire

        with logfire.span(
            "process_facts_for_group {group_id} ({facts_n} facts)",
            group_id=group_id,
            facts_n=len(facts),
        ):
            # Embed the fact texts once: the twin pass, Stage 6a retrieval and the
            # Stage 7 write all use these (6a used to embed the same texts again).
            with logfire.span("voyage_embed_facts {n}", n=len(facts)):
                fact_embeddings_list = self._embedder.embed(
                    [f.fact for f in facts], task="document"
                )

            # Same-batch twins: Stage 6 compares a fact only with edges already in the
            # graph, so a fact this batch states twice (usually one sentence fanned out
            # over every entity it names) would land twice. See ingestion.extraction_twins.
            with logfire.span("collapse_batch_twins") as twin_span:
                facts, fact_embeddings_list, twins = collapse_batch_twins(
                    facts, fact_embeddings_list
                )
                twin_span.set_attribute("collapsed", twins)
            if twins:
                logger.info("Collapsed %d same-batch twin fact(s) in group %s", twins, group_id)

            # Stage 6a: find contradiction candidates (no LLM)
            with logfire.span("stage6a_embedding_filter"):
                candidates_map = self._stage6a_embedding_filter(
                    facts, uuid_map, group_id, fact_embeddings_list
                )

            # Stage 6b: ONE batched LLM call for every fact with candidates,
            # instead of N serial ~30s claude-CLI subprocesses. Mirrors PR #83's
            # stage-4 batch treatment.
            with logfire.span(
                "stage6b_batch_confirm cands={cands}",
                cands=len(candidates_map),
            ) as span:
                skip_indices, invalidate, reinforce, llm_ok = self._stage6b_batch_confirm(
                    facts, candidates_map
                )
                span.set_attribute("skipped", len(skip_indices))
                span.set_attribute("invalidated", sum(len(v) for v in invalidate.values()))
                # False: the response never validated and these facts go in with no
                # duplicate or contradiction verdicts (the warning names the count).
                span.set_attribute("llm_ok", llm_ok)

            # Saturation continuation: facts that contradicted most of their
            # pool get fresh retrieval rounds so a sweeping supersession can
            # invalidate beyond the _SEMANTIC_POOL_LIMIT cap. The round-1 pools
            # (candidates_map) seed its seen-set so nothing is judged twice.
            if llm_ok and invalidate:
                with logfire.span("stage6b_saturation_rounds") as sat_span:
                    extra = self._stage6b_saturation_rounds(
                        facts, candidates_map, invalidate, fact_embeddings_list, group_id
                    )
                    sat_span.set_attribute("extra_invalidated", extra)

            # Stage 6c + the deferred Stage 5 for new entities run INSIDE Stage 7, after its
            # LLM pre-pass and right before its first write, so the window in which an
            # overlapping window's freshly written edge goes unseen is DB round-trips, not
            # LLM calls. 6c rechecks the facts about to be created against the live graph
            # as it is NOW (an edge written since 6a pooled, or one the judge missed); then
            # only the new entities a fact still being created links are written.
            inserted: list[tuple[str, str]] = []

            def recheck_then_write_new_nodes() -> None:
                with logfire.span("stage6c_recheck") as recheck_span:
                    restated = recheck_restated(
                        facts,
                        fact_embeddings_list,
                        uuid_map,
                        group_id,
                        self._kg,
                        skip_indices,
                        invalidate,
                        reinforce,
                    )
                    recheck_span.set_attribute("restated", restated)
                if restated:
                    logger.info(
                        "Stage 6c: %d fact(s) restate a live edge in %s", restated, group_id
                    )
                if new_nodes is not None and new_nodes.entities:
                    with logfire.span("stage5_write_new_nodes"):
                        inserted.extend(
                            self._stage5_write_new_nodes(
                                facts, uuid_map, skip_indices, group_id, new_nodes
                            )
                        )

            # Stage 7: write edges
            with logfire.span("stage7_write_edges {n}", n=len(facts)):
                self._stage7_write_edges(
                    facts,
                    uuid_map,
                    episode_ids,
                    group_id,
                    skip_indices,
                    invalidate,
                    fact_embeddings_list,
                    web_artifact_id=web_artifact_id,
                    default_valid_at=default_valid_at,
                    reference_time=reference_time,
                    reinforce=reinforce,
                    before_write=recheck_then_write_new_nodes,
                )

            # Stage 7 wrote the new entities in its transaction. Only now that it has
            # committed may later items dedupe against them: a rolled-back entity found
            # by name would give the retried item edges to a node that does not exist.
            if new_nodes is not None and new_nodes.deduper is not None:
                for name, node_uuid in inserted:
                    new_nodes.deduper.register(name, node_uuid)
