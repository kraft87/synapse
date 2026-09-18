from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from ingestion.models import (
    ExtractedEntity,
    ExtractedFact,
    ExtractionResult,
)
from ingestion.scope import active_groups

if TYPE_CHECKING:
    from ingestion.embedding import EmbeddingModel

from ingestion.extraction_edges import ExtractionEdgesMixin
from ingestion.extraction_entities import DeterministicExtractor, EntityResolver
from ingestion.extraction_llm import LLMExtractor
from ingestion.extraction_nodes import ExtractionNodesMixin
from ingestion.extraction_policy import _BATCH_CONTRADICTION_PROMPT as _BATCH_CONTRADICTION_PROMPT
from ingestion.extraction_policy import _CONTRADICTION_PROMPT as _CONTRADICTION_PROMPT
from ingestion.extraction_policy import _ERROR_RE as _ERROR_RE
from ingestion.extraction_policy import _FILE_PATH_RE as _FILE_PATH_RE
from ingestion.extraction_policy import _KNOWN_TOOLS as _KNOWN_TOOLS
from ingestion.extraction_policy import _OWNER as _OWNER
from ingestion.extraction_policy import _OWNER_POSSESSIVE as _OWNER_POSSESSIVE
from ingestion.extraction_policy import _PERSONAL_NAME_PATTERN as _PERSONAL_NAME_PATTERN
from ingestion.extraction_policy import _PERSONAL_PROJECTS as _PERSONAL_PROJECTS
from ingestion.extraction_policy import _SATURATION_MAX_ROUNDS as _SATURATION_MAX_ROUNDS
from ingestion.extraction_policy import _SATURATION_MIN as _SATURATION_MIN
from ingestion.extraction_policy import _SEMANTIC_POOL_LIMIT as _SEMANTIC_POOL_LIMIT
from ingestion.extraction_policy import _TECHNICAL_NAME_PATTERN as _TECHNICAL_NAME_PATTERN
from ingestion.extraction_policy import _URL_RE as _URL_RE
from ingestion.extraction_policy import (
    _apply_canonical_aliases as _apply_canonical_aliases,
)
from ingestion.extraction_policy import (
    _apply_gate_enforce as _apply_gate_enforce,
)
from ingestion.extraction_policy import (
    _classify_entity_group as _classify_entity_group,
)
from ingestion.extraction_policy import _cosine_similarity as _cosine_similarity
from ingestion.extraction_policy import (
    _dedup_gate_mode as _dedup_gate_mode,
)
from ingestion.extraction_policy import (
    _dedup_gate_thresholds as _dedup_gate_thresholds,
)
from ingestion.extraction_policy import (
    _default_group_for_project as _default_group_for_project,
)
from ingestion.extraction_policy import (
    _gate_decisions as _gate_decisions,
)
from ingestion.extraction_policy import (
    _gate_shadow_rows as _gate_shadow_rows,
)
from ingestion.extraction_policy import (
    build_batch_resolution_prompt as build_batch_resolution_prompt,
)
from ingestion.extraction_policy import build_resolution_prompt as build_resolution_prompt
from ingestion.extraction_policy import dedupe_pools as dedupe_pools

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
        from ingestion.contradiction import ContradictionDetector
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
        # Phase 3: writer-side bi-temporal contradiction safety net. Runs
        # immediately before create_edge writes, layered on top of Stage 6's
        # extractor-level contradiction prompt -- catches edges written via
        # paths that bypass Stage 6 (dream pipeline, manual writes, future
        # ingestion sources) AND any same-pair contradictions Stage 6's
        # broader retrieval missed.
        self._contradiction_detector = ContradictionDetector(
            kg_client=kg_client,
            embedder=embedder,
            llm_client=llm_client,
            model=self._contradiction_model,
        )
        # Phase 4: LLM-driven temporal-bounds extractor (Graphiti verbatim
        # `extract_timestamps`). Reads valid_at / invalid_at out of the
        # fact text itself when the caller doesn't pre-supply t_valid.
        # Best-effort: failures fall back to now() inside create_edge so
        # the write path is never blocked by date extraction.
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
        graph via _classify_entity_group (project tag → item default, then per-entity
        regex override), or to technical alone when the personal scope is off. Entities are partitioned by group and resolved/written
        independently against their target graph. Edges are only written when both
        endpoints landed in the same group; cross-group facts are dropped (rare —
        a sign of borderline content the regex misclassified).
        """
        content_type: str = item["content_type"]
        content: str = item["content"]
        project: str | None = item.get("project")
        session_id: str | None = item.get("session_id")
        default_group = _default_group_for_project(project)
        web_provenance: dict[str, Any] | None = None

        # Segment date (conversation time, NOT ingest wall-clock) = the source
        # episodes' latest created_at. Resolved up-front, BEFORE Stage 3, so the
        # extraction prompt can anchor in-text date mentions against it; reused below
        # as the edge backlink and the default fact valid-time. Drill-back: the turn's
        # own id for episodes, else the segment's source episodes.
        episode_ids: list[int] = []
        if item.get("episode_id"):
            episode_ids = [item["episode_id"]]
        elif content_type == "summary" and session_id:
            # Summaries -> the synth_document's source_ids so edges trace back to the
            # episodes they were derived from.
            try:
                episode_ids = self._db.get_synth_document_source_ids(session_id, content)
            except Exception:
                episode_ids = []
        elif content_type == "chunk" and session_id:
            # Chunks -> the window the chunk was built from (task #63).
            try:
                episode_ids = self._db.get_chunk_episode_ids(session_id, content)
            except Exception:
                episode_ids = []
        # Episodes emit no facts and skip Stage 3, so their segment date is never
        # consumed — skip the lookup for them to keep the per-turn hot path lean.
        segment_valid_at = (
            self._db.get_episodes_valid_at(episode_ids)
            if (episode_ids and content_type != "episode")
            else None
        )
        session_date = segment_valid_at[:10] if segment_valid_at else None

        # Stage 2: deterministic extraction
        if content_type == "episode":
            episodes = [{"content": content, "metadata": json.loads(item.get("metadata") or "{}")}]
            det_entities = self._stage2_deterministic(episodes)
            llm_result = ExtractionResult(entities=[], facts=[])
            # Timeline chat gate rides the per-turn item (chunks span turns and
            # would blur the event's date). Fail-soft; never blocks KG work.
            self._timeline_gate.process(item)
            # Preferences gate rides the same per-turn item (a preference is stated in
            # one turn, not spread across a window). Fail-soft; never blocks KG work.
            self._preferences_gate.process(item)
        elif content_type == "chunk":
            # Chunk = a 3-5 turn window (task #63). Deterministic entities come
            # from the chunk's OWN text — not a full-session fetch per chunk —
            # then full LLM extraction runs on the window. Edge backlink to the
            # source episodes is resolved below via get_chunk_episode_ids.
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
            # summary or manual — full pipeline
            episodes_raw: list[dict[str, Any]] = []
            if session_id:
                episodes_raw = self._db.get_session_episodes(session_id)
            det_entities = self._stage2_deterministic(episodes_raw)
            llm_result = self._stage3_llm(content, det_entities, session_date)

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
        # fact can survive Stage 5 (the orphan-drop below enforces that). But
        # the deterministic extractor emits hundreds-to-thousands of entity
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

        # Per-entity group classification (regex-based override of item default)
        entity_groups: dict[str, str] = {
            e.name: _classify_entity_group(e.name, e.summary, default_group) for e in all_entities
        }

        # Pre-embed all entity names once (re-used per-group below)
        entity_embeddings: dict[str, list[float]] = {}
        if all_entities:
            names = [e.name for e in all_entities]
            embs = self._embedder.embed(names, task="entity")
            entity_embeddings = dict(zip(names, embs, strict=True))

        # Resolve nodes separately per group, defer the write until after the
        # orphan-drop pass below. Resolving (Stage 4) first is safe — it only
        # reads from the graph. Writing (Stage 5) is what we need to gate.
        #
        # Per-group ``NodeDeduper`` instances are cached on the Extractor
        # (``_deduper_for``) so the LSH index — an O(all entities) MinHash
        # build — is constructed once per worker and reused across items.
        # ``register`` keeps the in-memory index in sync after Stage 5
        # writes each new node, so repeat names dedupe against
        # freshly-inserted nodes both within an item and across items.
        from ingestion.dedup import NodeDeduper

        uuid_map: dict[str, str] = {}
        grp_entities_map: dict[str, list[ExtractedEntity]] = {}
        dedupers: dict[str, NodeDeduper] = {}
        for grp in active_groups():
            grp_entities = [e for e in all_entities if entity_groups[e.name] == grp]
            if not grp_entities:
                continue
            dedupers[grp] = self._deduper_for(grp)
            grp_uuid_map = self._stage4_resolve(grp_entities, grp, deduper=dedupers[grp])
            uuid_map.update(grp_uuid_map)
            grp_entities_map[grp] = grp_entities

        # --- Orphan-drop: skip writes for entities no fact references ---
        # Mirrors Graphiti's combined_extraction.py:280-295. Before we burn a
        # graph write per entity, check that the LLM extractor actually
        # produced a fact referencing it. Without this filter the
        # deterministic extractor + LLM extractor produce ~92% zero-edge
        # orphan nodes that the nightly cleanup later has to delete.
        referenced_uuids: set[str] = set()
        for fact in llm_result.facts:
            src_uuid = uuid_map.get(fact.source)
            tgt_uuid = uuid_map.get(fact.target)
            if src_uuid:
                referenced_uuids.add(src_uuid.removeprefix("new:"))
            if tgt_uuid:
                referenced_uuids.add(tgt_uuid.removeprefix("new:"))

        # Stage 5 — write only the entities whose resolved UUID shows up as
        # the source or target of some fact in the same response.
        orphan_count = 0
        for grp, grp_entities in grp_entities_map.items():
            grp_uuid_map = {e.name: uuid_map[e.name] for e in grp_entities if e.name in uuid_map}
            kept_entities = [
                e
                for e in grp_entities
                if grp_uuid_map.get(e.name, "").removeprefix("new:") in referenced_uuids
            ]
            orphan_count += len(grp_entities) - len(kept_entities)
            if not kept_entities:
                continue
            self._stage5_write_nodes(
                kept_entities,
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

        # Partition facts by the group of their (source, target) entity pair. Drop
        # cross-group facts — they imply the regex misclassified one endpoint.
        facts_by_group: dict[str, list[ExtractedFact]] = {g: [] for g in active_groups()}
        cross_group_dropped = 0
        for fact in llm_result.facts:
            src_grp = entity_groups.get(fact.source)
            tgt_grp = entity_groups.get(fact.target)
            if src_grp and tgt_grp and src_grp == tgt_grp and src_grp in facts_by_group:
                facts_by_group[src_grp].append(fact)
            else:
                cross_group_dropped += 1
        if cross_group_dropped:
            logger.debug("Dropped %d cross-group facts (entities span groups)", cross_group_dropped)

        # Web provenance → edge attrs: the artifact id rides every created edge
        # (kg_shadow mirror column), and the page's published/fetched date is the
        # default t_valid for facts whose text carries no date of its own — web
        # claims age with their source, not with ingestion time.
        web_artifact_id: int | None = None
        default_valid_at: str | None = None
        if web_provenance:
            web_artifact_id = web_provenance.get("web_artifact_id")
            dt = web_provenance.get("published_at") or web_provenance.get("fetched_at")
            if dt is not None:
                default_valid_at = dt.isoformat() if hasattr(dt, "isoformat") else str(dt)
        else:
            # Conversation facts: default valid-time = the SEGMENT's own timestamp
            # (max created_at of its source episodes), NOT ingest wall-clock. Without
            # this a fact carrying no in-text date got t_valid=now() — coincidentally
            # right for live ingestion (now ≈ conversation time) but wrong for any
            # backfilled/retro transcript. Mirrors how web provenance dates its facts.
            # Computed once up-front (segment_valid_at) and reused here.
            default_valid_at = segment_valid_at

        # Stage 6 + 7 run separately per group — same code path, different graph.
        # default_valid_at doubles as the relative-date reference_time (the segment
        # timestamp), so "last week" resolves against the conversation, not ingest.
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
                reference_time=default_valid_at,
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
    ) -> None:
        """Stage 6 + 7 for one group's facts (extracted from process_item to keep it readable)."""
        import logfire

        with logfire.span(
            "process_facts_for_group {group_id} ({facts_n} facts)",
            group_id=group_id,
            facts_n=len(facts),
        ):
            # Stage 6a: find contradiction candidates (no LLM)
            with logfire.span("stage6a_embedding_filter"):
                candidates_map = self._stage6a_embedding_filter(facts, uuid_map, group_id)

            # Gray-zone gate (issue #14): triage candidates on the similarity 6a
            # already computed. shadow = log would-be decisions, change nothing;
            # enforce = only the gray zone reaches the LLM confirm below.
            gate_mode = _dedup_gate_mode()
            gate_info: dict[int, list[tuple[dict[str, Any], str, float | None, str]]] = {}
            pre_skip: set[int] = set()
            pre_reinforce: dict[int, list[str]] = {}
            llm_map = candidates_map
            if gate_mode != "off" and candidates_map:
                high, low = _dedup_gate_thresholds()
                gate_info = {
                    idx: _gate_decisions(pair_pool, semantic_pool, high, low)
                    for idx, (pair_pool, semantic_pool) in candidates_map.items()
                }
                if gate_mode == "enforce":
                    llm_map, pre_skip, pre_reinforce = _apply_gate_enforce(gate_info)

            # Pre-embed fact texts for Stage 7
            with logfire.span("voyage_embed_facts {n}", n=len(facts)):
                fact_embeddings_list = self._embedder.embed(
                    [f.fact for f in facts], task="document"
                )

            # Stage 6b: ONE batched LLM call for every fact with candidates,
            # instead of N serial ~30s claude-CLI subprocesses. Mirrors PR #83's
            # stage-4 batch treatment; per-fact _stage6b_llm_confirm is retained
            # for callers that need single-fact confirmation (e.g. dream writes).
            with logfire.span(
                "stage6b_batch_confirm cands={cands}",
                cands=len(llm_map),
            ) as span:
                skip_indices, invalidate, reinforce, llm_ok = self._stage6b_batch_confirm(
                    facts, llm_map
                )
                skip_indices |= pre_skip
                for idx, uuids in pre_reinforce.items():
                    reinforce[idx] = uuids
                span.set_attribute("skipped", len(skip_indices))
                span.set_attribute("invalidated", sum(len(v) for v in invalidate.values()))
                span.set_attribute("gate_mode", gate_mode)
                span.set_attribute("gate_pre_skipped", len(pre_skip))

            # Saturation continuation: facts that contradicted most of their
            # pool get fresh retrieval rounds so a sweeping supersession can
            # invalidate beyond the _SEMANTIC_POOL_LIMIT cap. Uses the FULL
            # round-1 pools (candidates_map, not the gate-shrunk llm_map) for
            # its seen-set so enforcement-dropped candidates aren't re-judged.
            if llm_ok and invalidate:
                with logfire.span("stage6b_saturation_rounds") as sat_span:
                    extra = self._stage6b_saturation_rounds(
                        facts, candidates_map, invalidate, fact_embeddings_list, group_id
                    )
                    sat_span.set_attribute("extra_invalidated", extra)

            # Shadow log: one row per (fact, candidate) with the gate's would-be
            # decision next to the LLM's actual verdict — the threshold-picking
            # data for enforcement. Best-effort; never blocks the pipeline.
            if gate_info:
                try:
                    self._db.log_dedup_gate_shadow(
                        _gate_shadow_rows(
                            facts, gate_info, llm_map, group_id, invalidate, reinforce, llm_ok
                        )
                    )
                except Exception as e:
                    logger.debug("dedup gate shadow log failed: %s", e)

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
                )
