from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ingestion.kg_client import rrf_merge
from ingestion.llm_client import LLM_TRANSPORT_ERRORS
from ingestion.models import (
    ExtractedFact,
)

if TYPE_CHECKING:
    from ingestion.extraction_llm import LLMExtractor

from ingestion.extraction_policy import (
    _SATURATION_MAX_ROUNDS,
    _SATURATION_MIN,
    _SEMANTIC_POOL_LIMIT,
    _cosine_similarity,
    build_batch_resolution_prompt,
    build_resolution_prompt,
    dedupe_pools,
)

logger = logging.getLogger(__name__)


class ExtractionEdgesMixin:
    _embedder: Any
    _kg: Any
    _llm: LLMExtractor
    _contradiction_model: str
    _edge_date_extractor: Any
    _contradiction_detector: Any

    def _stage6a_embedding_filter(
        self,
        facts: list[ExtractedFact],
        uuid_map: dict[str, str],
        group_id: str,
    ) -> dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
        """Find duplicate/contradiction candidates for each fact.

        Returns {fact_index: (pair_pool, semantic_pool)} where:
        - pair_pool: edges sharing the new fact's source+target entities
          (strongest prior for *duplicate* detection).
        - semantic_pool: RRF-merged BM25 + vector hits over fact text
          (strongest prior for *contradiction/supersession* detection,
          including drop-in replacements where the target entity differs).

        Pools are deduped so any uuid in pair_pool is removed from semantic_pool.
        Facts with both pools empty are omitted from the returned dict.
        """
        if not facts:
            return {}

        # Embed all fact texts in one batch
        fact_texts = [f.fact for f in facts]
        fact_embeddings = self._embedder.embed(fact_texts, task="document")

        per_fact_pools: dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}

        for idx, (fact, fact_emb) in enumerate(zip(facts, fact_embeddings, strict=True)):
            src_uuid = uuid_map.get(fact.source, "").removeprefix("new:")
            tgt_uuid = uuid_map.get(fact.target, "").removeprefix("new:")

            # Pair pool — same source/target endpoints.
            pair_pool: list[dict[str, Any]] = []
            if src_uuid and tgt_uuid:
                pair_pool = list(self._kg.find_edges_by_pair(src_uuid, tgt_uuid, group_id))
            # Gray-zone gate signal (issue #14): tag every candidate that has an
            # embedding with its cosine similarity to the new fact. Pair-pool rows
            # return their stored embedding; computed here, in-process, no extra I/O.
            for cand in pair_pool:
                emb = cand.get("fact_embedding")
                cand["_sim"] = _cosine_similarity(fact_emb, emb) if emb else None

            # Semantic pool — RRF over vector + BM25 hits. Pull 2x the eventual
            # cap from each modality so the long tail of moderate-rank entries
            # in both lists has a chance to win the merge.
            _src_limit = _SEMANTIC_POOL_LIMIT * 2
            vector_hits = self._kg.find_similar_edges(fact_emb, group_id, limit=_src_limit)
            # Vector hits carry cosine DISTANCE as "score" (kg_pg_read) — convert once
            # here so the gate sees one signal. BM25-only hits stay untagged (_sim
            # None -> always gray/LLM-confirmed; their score isn't comparable).
            for cand in vector_hits:
                cand["_sim"] = 1.0 - float(cand.get("score") or 0.0)
            fulltext_hits = self._kg.find_edges_by_fulltext(fact.fact, group_id, limit=_src_limit)
            semantic_pool = rrf_merge(vector_hits, fulltext_hits, limit=_SEMANTIC_POOL_LIMIT, k=1)

            # Drop any pair-pool uuid from the semantic pool so the LLM doesn't
            # see the same fact under two labels.
            _, semantic_pool = dedupe_pools(pair_pool, semantic_pool)

            if pair_pool or semantic_pool:
                per_fact_pools[idx] = (pair_pool, semantic_pool)

        return per_fact_pools

    def _stage6b_llm_confirm(
        self,
        fact: ExtractedFact,
        pair_pool: list[dict[str, Any]],
        semantic_pool: list[dict[str, Any]],
    ) -> tuple[bool, list[str]]:
        """Ask LLM which candidates the new fact duplicates and/or contradicts.

        Pools are passed separately so the prompt can render them under
        distinct labels (`EXISTING FACTS` vs `INVALIDATION CANDIDATES`),
        giving the LLM the structural prior — same-endpoint → likely
        duplicate, semantic-neighbour → likely contradiction.

        Returns (skip_write, contradicted_uuids).
        skip_write=True when a pure-duplicate idx is present in the response.
        contradicted_uuids lists every edge to invalidate; an idx in both
        duplicate_facts and contradicted_facts is treated as a drop-in
        replacement (skip nothing, invalidate the old).
        """
        from ingestion.llm_client import structured_call
        from ingestion.llm_schemas import ResolutionResult

        prompt, idx_to_uuid = build_resolution_prompt(fact.fact, pair_pool, semantic_pool)
        if not idx_to_uuid:
            return False, []
        try:
            # Triple classification on short fact pairs — Haiku is sufficient
            # and ~10x cheaper than Sonnet.
            resolution = structured_call(
                self._llm._client,
                output_model=ResolutionResult,
                base_prompt=prompt,
                model=self._contradiction_model,
                max_tokens=300,
            )
            dup_idx = [i for i in resolution.duplicate_facts if i in idx_to_uuid]
            contradicted_idx = [i for i in resolution.contradicted_facts if i in idx_to_uuid]
        except LLM_TRANSPORT_ERRORS:
            raise  # never answered: fail the item, don't record a non-decision
        except Exception:
            return False, []
        dup_uuids = [idx_to_uuid[i] for i in dup_idx]
        contradicted = [idx_to_uuid[i] for i in contradicted_idx]

        # Skip the new write when ANY duplicate is purely-restated (in dup
        # list but not also contradicted) — the existing edge already covers
        # this information. Still emit any contradictions so they get
        # invalidated regardless.
        pure_duplicates = [u for u in dup_uuids if u not in contradicted]
        return bool(pure_duplicates), contradicted

    def _stage6b_batch_confirm(
        self,
        facts: list[ExtractedFact],
        candidates_map: dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]]]],
    ) -> tuple[set[int], dict[int, list[str]], dict[int, list[str]], bool]:
        """Batched version of _stage6b_llm_confirm.

        Collects every (fact, pools) entry with at least one candidate and
        issues ONE LLM call instead of N ~30s claude-CLI subprocesses. On
        any parse/transport failure, fails closed: returns empty skip set
        and empty invalidate dict (same conservative behaviour as the
        per-fact version's bare except — a missed contradiction is
        recoverable; blocking the new write is not).

        Returns (skip_indices, invalidate, reinforce, ok): skip_indices = new
        facts to skip (pure duplicates), invalidate = {new_idx: [contradicted
        edge uuids]}, reinforce = {new_idx: [matched existing edge uuids]} for
        the skipped duplicates — the dedup-hit signal Stage 7 uses to bump
        mention_count + union episodes instead of dropping the re-assertion.
        ok=False means the LLM call/parse failed (fail-closed empties) — the
        gray-zone gate's shadow log uses it to keep failed batches out of the
        threshold-analysis data (issue #14).
        """
        if not candidates_map:
            return set(), {}, {}, True

        items = [
            {
                "id": idx,
                "new_fact": facts[idx].fact,
                "existing_pool": pair_pool,
                "candidate_pool": semantic_pool,
            }
            for idx, (pair_pool, semantic_pool) in candidates_map.items()
        ]
        from ingestion.llm_client import structured_call
        from ingestion.llm_schemas import BatchResolutionResult

        prompt, per_item_maps = build_batch_resolution_prompt(items)

        try:
            batch = structured_call(
                self._llm._client,
                output_model=BatchResolutionResult,
                base_prompt=prompt,
                model=self._contradiction_model,
                max_tokens=300 * max(1, len(items)),
            )
        except LLM_TRANSPORT_ERRORS:
            raise  # never answered: fail the item, don't record a non-decision
        except Exception:
            return set(), {}, {}, False

        # Index coercion (digit strings, {"index": N} wrappers — seen on
        # deepseek-v4-flash 2026-07-18) happens inside BatchResolutionResult's
        # tolerant validators; by here every idx is a bare int.
        skip_indices: set[int] = set()
        invalidate: dict[int, list[str]] = {}
        reinforce: dict[int, list[str]] = {}
        for r in batch.results:
            fid = r.id
            if fid not in per_item_maps:
                continue
            idx_to_uuid = per_item_maps[fid]
            dup_idx = [i for i in r.duplicate_facts if i in idx_to_uuid]
            contradicted_idx = [i for i in r.contradicted_facts if i in idx_to_uuid]
            dup_uuids = [idx_to_uuid[i] for i in dup_idx]
            contradicted = [idx_to_uuid[i] for i in contradicted_idx]
            pure_duplicates = [u for u in dup_uuids if u not in contradicted]
            if pure_duplicates:
                skip_indices.add(fid)
                # the matched existing edges this fact re-asserts -> Stage 7
                # reinforces them (mention_count++ + episodes union).
                reinforce[fid] = pure_duplicates
            if contradicted:
                invalidate[fid] = contradicted
        return skip_indices, invalidate, reinforce, True

    def _stage6b_saturation_rounds(
        self,
        facts: list[ExtractedFact],
        candidates_map: dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]]]],
        invalidate: dict[int, list[str]],
        fact_embeddings: list[list[float]],
        group_id: str,
    ) -> int:
        """Continuation rounds for facts whose verdict saturated their pool.

        A fact that contradicts >= _SATURATION_MIN of its candidates is very
        likely a sweeping supersession whose true contradiction set exceeds
        _SEMANTIC_POOL_LIMIT. For each such fact, re-run retrieval excluding
        every candidate already judged (contradicted edges aren't invalidated
        in the DB until Stage 7, so exclusion must be by seen-set, not by
        retrieval filters) and issue further contradiction-only confirms
        until a round returns fewer than _SATURATION_MIN fresh hits or
        _SATURATION_MAX_ROUNDS extra rounds have run.

        Only ``contradicted`` verdicts are consumed from continuation rounds;
        duplicate/skip verdicts are ignored — the write decision was already
        made against the (higher-ranked) round-1 pools, and dropping a new
        edge on a rank-9+ "duplicate" would be trusting a weaker signal.

        Mutates ``invalidate`` in place. Returns the number of extra edges
        marked for invalidation (for span attributes). Never raises.
        """
        seen: dict[int, set[str]] = {}
        active: list[int] = []
        for idx, uuids in invalidate.items():
            pools = candidates_map.get(idx)
            if not pools or len(uuids) < _SATURATION_MIN:
                continue
            pair_pool, semantic_pool = pools
            seen[idx] = {c["uuid"] for c in pair_pool} | {c["uuid"] for c in semantic_pool}
            active.append(idx)
        if not active:
            return 0

        extra = 0
        try:
            for _round in range(_SATURATION_MAX_ROUNDS):
                round_map: dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
                for idx in active:
                    # Over-fetch by the seen-set size so exclusion can't
                    # starve the pool while earlier hits still dominate
                    # the ranking.
                    src_limit = _SEMANTIC_POOL_LIMIT * 2 + len(seen[idx])
                    vector_hits = self._kg.find_similar_edges(
                        fact_embeddings[idx], group_id, limit=src_limit
                    )
                    fulltext_hits = self._kg.find_edges_by_fulltext(
                        facts[idx].fact, group_id, limit=src_limit
                    )
                    pool = [
                        c
                        for c in rrf_merge(vector_hits, fulltext_hits, limit=src_limit, k=1)
                        if c["uuid"] not in seen[idx]
                    ][:_SEMANTIC_POOL_LIMIT]
                    if pool:
                        seen[idx].update(c["uuid"] for c in pool)
                        round_map[idx] = ([], pool)
                if not round_map:
                    break
                _, round_invalidate, _, ok = self._stage6b_batch_confirm(facts, round_map)
                if not ok:
                    break
                next_active: list[int] = []
                for idx, uuids in round_invalidate.items():
                    fresh = [u for u in uuids if u not in invalidate.get(idx, [])]
                    if not fresh:
                        continue
                    invalidate.setdefault(idx, []).extend(fresh)
                    extra += len(fresh)
                    if len(fresh) >= _SATURATION_MIN:
                        next_active.append(idx)
                active = next_active
                if not active:
                    break
        except Exception as exc:
            # Same posture as the writer-side detector: a missed
            # contradiction is recoverable, blocking the write path is not.
            logger.warning("stage6b saturation rounds failed group=%s: %s", group_id, exc)
        return extra

    def _stage7_write_edges(
        self,
        facts: list[ExtractedFact],
        uuid_map: dict[str, str],
        episode_ids: list[int],
        group_id: str,
        skip_indices: set[int],
        invalidate: dict[int, list[str]],
        fact_embeddings: list[list[float]] | None = None,
        web_artifact_id: int | None = None,
        default_valid_at: str | None = None,
        reference_time: str | None = None,
        reinforce: dict[int, list[str]] | None = None,
    ) -> None:
        """Write fact edges, skipping duplicates and invalidating contradictions.

        Invalidations run independently of skips: if a new fact is a pure
        duplicate of one existing edge but also contradicts another (drop-in
        replacement case), we skip the redundant write but still mark the
        superseded edge as invalid.
        """
        # Pre-extract (valid_at, invalid_at) for ALL facts in ONE LLM call
        # (PR #89). Previously create_edge fired EdgeDateExtractor.extract
        # once per fact — at ~30s/call on N=26 facts that was ~13 min of
        # serial LLM work per summary. Batching collapses it to one call.
        # On failure the helper returns (None, None) per fact; create_edge
        # then falls back to now() for valid_at exactly as before.
        eligible_facts: list[str] = []
        eligible_srcs: list[str] = []
        eligible_tgts: list[str] = []
        eligible_embs: list[list[float] | None] = []
        for idx, fact in enumerate(facts):
            if idx in skip_indices:
                eligible_facts.append("")
                eligible_srcs.append("")
                eligible_tgts.append("")
                eligible_embs.append(None)
                continue
            src_uuid = uuid_map.get(fact.source)
            tgt_uuid = uuid_map.get(fact.target)
            ok = bool(src_uuid and tgt_uuid)
            eligible_facts.append(fact.fact if ok else "")
            eligible_srcs.append(src_uuid.removeprefix("new:") if (ok and src_uuid) else "")
            eligible_tgts.append(tgt_uuid.removeprefix("new:") if (ok and tgt_uuid) else "")
            eligible_embs.append(
                fact_embeddings[idx]
                if (ok and fact_embeddings and idx < len(fact_embeddings))
                else None
            )
        batched_dates = self._edge_date_extractor.extract_batch(
            eligible_facts, reference_time=reference_time
        )

        # Batched contradiction detection (one LLM call covering every fact
        # whose (src, tgt) pair already has a live edge above the similarity
        # threshold). Replaces the per-fact detector firing inside
        # create_edge: stage7 wall p95 was 110s, max 186s — variance was
        # the detector firing serially when contradictions exist. Batching
        # collapses N detector LLM calls to ONE.
        batched_contradictions = self._contradiction_detector.detect_contradictions_batch(
            facts, eligible_srcs, eligible_tgts, group_id, fact_embeddings=eligible_embs
        )

        # Assemble the FULL invalidation list before writing edges so we
        # dispatch ONE UNWIND-batched MATCH+SET round-trip instead of N.
        # Three sources contribute, in the same ordering the previous
        # per-fact code used:
        #   1. ``invalidate`` (dedup invalidations from Stage 6) — independent
        #      of whether the new edge gets written; always applied.
        #   2. ``batched_contradictions[idx]`` — old live edges the new fact
        #      supersedes. Applied BEFORE the new CREATE (preserves the
        #      semantic ordering from the per-fact create_edge detector path).
        #   3. ``t_invalid_pre`` from the batched edge-date extractor — the
        #      new edge was already-contradicted at extraction time (e.g. the
        #      fact text said "the user worked at X from 2020 to 2022"). Applied
        #      AFTER the create_edges_batch via create_edges_batch's own
        #      follow-up invalidate_edges_batch call.
        # Pre-generate the new edge uuid for each fact that WILL be created (same guards as the
        # create_rows loop below), so a contradiction can record WHICH new edge supersedes the old
        # one (schema 028, invalidated_by). Facts that won't be created (no resolved src/tgt) can
        # still contradict an old edge but leave no recoverable superseder -> invalidated_by NULL.
        new_uuid_by_idx: dict[int, str] = {}
        for idx, fact in enumerate(facts):
            if idx in skip_indices:
                continue
            if uuid_map.get(fact.source) and uuid_map.get(fact.target):
                new_uuid_by_idx[idx] = str(uuid.uuid4())

        # 1. Dedup invalidations (Stage 6) — no superseder.
        dedup_invalidations: list[tuple[str, str | None]] = [
            (edge_uuid, None) for uuids in invalidate.values() for edge_uuid in uuids
        ]
        if dedup_invalidations:
            self._kg.invalidate_edges_batch(dedup_invalidations, group_id)
        # 2. Contradiction invalidations — group old edges by their superseding new edge so each
        #    group records invalidated_by in one round-trip; orphans (uncreated fact) stay NULL.
        by_superseder: dict[str, list[tuple[str, str | None]]] = {}
        orphan_contradictions: list[tuple[str, str | None]] = []
        for idx in range(len(facts)):
            if idx in skip_indices:
                continue
            sup = new_uuid_by_idx.get(idx)
            for old_uuid in batched_contradictions[idx]:
                if sup:
                    by_superseder.setdefault(sup, []).append((old_uuid, None))
                else:
                    orphan_contradictions.append((old_uuid, None))
        for sup, olds in by_superseder.items():
            self._kg.invalidate_edges_batch(olds, group_id, invalidated_by=sup)
        if orphan_contradictions:
            self._kg.invalidate_edges_batch(orphan_contradictions, group_id)

        # Build CREATE rows for every eligible fact in one pass, then dispatch
        # one batched MATCH+CREATE round-trip (two if some facts lack
        # embeddings — see create_edges_batch). Replaces N per-fact create_edge
        # calls, each of which fired its own MATCH+CREATE Cypher hop. Stage 7
        # wall on a 16-fact item was ~30s dominated by graph round-trips;
        # this collapses them.
        now = datetime.now(UTC).isoformat()
        create_rows: list[dict[str, Any]] = []
        for idx, fact in enumerate(facts):
            if idx in skip_indices:
                continue
            src_uuid = uuid_map.get(fact.source)
            tgt_uuid = uuid_map.get(fact.target)
            if not src_uuid or not tgt_uuid:
                continue
            src_clean = src_uuid.removeprefix("new:")
            tgt_clean = tgt_uuid.removeprefix("new:")
            emb = fact_embeddings[idx] if fact_embeddings and idx < len(fact_embeddings) else None
            t_valid_pre, t_invalid_pre = batched_dates[idx]
            # Date precedence: explicit date in the fact text > source-page date
            # (web lane) > extraction time.
            valid_at_ts = t_valid_pre or default_valid_at or now
            create_rows.append(
                {
                    "src": src_clean,
                    "tgt": tgt_clean,
                    "edge_uuid": new_uuid_by_idx[idx],
                    "name": fact.relationship,
                    "fact": fact.fact,
                    "episodes": episode_ids,
                    "created_at": now,
                    "valid_at": valid_at_ts,
                    "t_created": now,
                    "t_valid": valid_at_ts,
                    "emb": emb,
                    # Non-None => create_edges_batch fires a follow-up
                    # invalidate_edges_batch with these uuids so the new edge
                    # is born already-invalidated (preserves bi-temporal
                    # lifecycle bookend from the per-fact create_edge path).
                    "t_invalid": t_invalid_pre,
                    # Web provenance (task #68). None on the episode lane.
                    "web_artifact_id": web_artifact_id,
                }
            )
        if create_rows:
            self._kg.create_edges_batch(create_rows, group_id)

        # Capture dedup hits: a fact skipped as a pure duplicate still ASSERTED
        # the matched edge(s). Bump their mention_count + union this chunk's
        # source episodes (provenance) instead of dropping the re-assertion.
        # Forward-only — historical dupes are already gone. The read-side ranking
        # boost on mention_count is a later phase.
        if reinforce:
            reinforce_items: list[tuple[str, list[int]]] = [
                (existing_uuid, episode_ids)
                for idx in skip_indices
                for existing_uuid in reinforce.get(idx, [])
            ]
            if reinforce_items:
                self._kg.reinforce_edges(reinforce_items, group_id)
