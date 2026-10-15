from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ingestion.kg_client import rrf_merge
from ingestion.llm_client import LLM_TRANSPORT_ERRORS, MalformedResponseError
from ingestion.models import (
    ExtractedFact,
)

if TYPE_CHECKING:
    from ingestion.extraction_llm import LLMExtractor

from ingestion.extraction_policy import (
    _FULLTEXT_MIN_SIM,
    _PAIR_POOL_LIMIT,
    _SATURATION_MAX_ROUNDS,
    _SATURATION_MIN,
    _SEMANTIC_POOL_LIMIT,
    build_batch_resolution_prompt,
    dedupe_pools,
)

logger = logging.getLogger(__name__)


def _on_topic(fulltext_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """BM25 hits close enough to the new fact to be judged against it (_FULLTEXT_MIN_SIM).

    A hit with no stored embedding cannot be checked and is dropped."""
    return [
        h
        for h in fulltext_hits
        if h.get("dist") is not None and 1.0 - h["dist"] >= _FULLTEXT_MIN_SIM
    ]


class ExtractionEdgesMixin:
    _embedder: Any
    _kg: Any
    _llm: LLMExtractor
    _contradiction_model: str
    _edge_date_extractor: Any

    def _stage6a_embedding_filter(
        self,
        facts: list[ExtractedFact],
        uuid_map: dict[str, str],
        group_id: str,
        fact_embeddings: list[list[float]] | None = None,
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
        ``fact_embeddings`` reuses the caller's embeddings of the same texts; without
        them the facts are embedded here.
        """
        if not facts:
            return {}

        if fact_embeddings is None:
            fact_embeddings = self._embedder.embed([f.fact for f in facts], task="document")

        per_fact_pools: dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}

        for idx, (fact, fact_emb) in enumerate(zip(facts, fact_embeddings, strict=True)):
            src_uuid = uuid_map.get(fact.source, "")
            tgt_uuid = uuid_map.get(fact.target, "")

            # Pair pool — same source/target endpoints. An endpoint that is new has no
            # edges by construction, so there is no pair to look up.
            pair_pool: list[dict[str, Any]] = []
            if (
                src_uuid
                and tgt_uuid
                and not (src_uuid.startswith("new:") or tgt_uuid.startswith("new:"))
            ):
                pair_pool = list(
                    self._kg.find_edges_by_pair(
                        src_uuid, tgt_uuid, group_id, fact_emb, limit=_PAIR_POOL_LIMIT
                    )
                )

            # Semantic pool — RRF over vector + BM25 hits. Pull 2x the eventual
            # cap from each modality so the long tail of moderate-rank entries
            # in both lists has a chance to win the merge.
            _src_limit = _SEMANTIC_POOL_LIMIT * 2
            vector_hits = self._kg.find_similar_edges(fact_emb, group_id, limit=_src_limit)
            fulltext_hits = _on_topic(
                self._kg.find_edges_by_fulltext(
                    fact.fact, group_id, limit=_src_limit, fact_embedding=fact_emb
                )
            )
            semantic_pool = rrf_merge(vector_hits, fulltext_hits, limit=_SEMANTIC_POOL_LIMIT, k=1)

            # Drop any pair-pool uuid from the semantic pool so the LLM doesn't
            # see the same fact under two labels.
            _, semantic_pool = dedupe_pools(pair_pool, semantic_pool)

            if pair_pool or semantic_pool:
                per_fact_pools[idx] = (pair_pool, semantic_pool)

        return per_fact_pools

    def _stage6b_batch_confirm(
        self,
        facts: list[ExtractedFact],
        candidates_map: dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]]]],
    ) -> tuple[set[int], dict[int, list[str]], dict[int, list[str]], bool]:
        """Judge every (fact, pools) entry with at least one candidate in ONE LLM call.

        A response that never validates gets one retry (the validation error quoted
        back to the model), then fails closed: empty skip set and empty invalidate
        dict, logged. A missed contradiction is recoverable; blocking the new write
        is not. A transport failure is not a response at all and propagates, so the
        queue retries the item.

        Returns (skip_indices, invalidate, reinforce, ok): skip_indices = new
        facts to skip (pure duplicates), invalidate = {new_idx: [contradicted
        edge uuids]}, reinforce = {new_idx: [matched existing edge uuids]} for
        the skipped duplicates — the dedup-hit signal Stage 7 uses to bump
        mention_count + union episodes instead of dropping the re-assertion.
        ok=False means the response never validated (fail-closed empties), so no
        saturation round runs on the result.
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
                max_attempts=2,
            )
        except MalformedResponseError as exc:
            logger.warning(
                "stage 6b: no usable verdicts for %d fact(s) after a retry; writing them "
                "without duplicate or contradiction checks: %s",
                len(items),
                exc,
            )
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
        marked for invalidation (for span attributes). Raises only transport errors.
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
                    fulltext_hits = _on_topic(
                        self._kg.find_edges_by_fulltext(
                            facts[idx].fact,
                            group_id,
                            limit=src_limit,
                            fact_embedding=fact_embeddings[idx],
                        )
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
        except LLM_TRANSPORT_ERRORS:
            raise  # never answered: fail the item, don't write with a partial verdict set
        except Exception as exc:
            # A missed contradiction is recoverable, blocking the write path is not.
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
        before_write: Callable[[], None] | None = None,
    ) -> None:
        """Write fact edges, skipping duplicates and invalidating contradictions.

        Invalidations run independently of skips: if a new fact is a pure
        duplicate of one existing edge but also contradicts another (drop-in
        replacement case), we skip the redundant write but still mark the
        superseded edge as invalid.

        ``before_write`` runs after the LLM pre-pass (edge dates) and before the first
        write. It may add to ``skip_indices`` and ``reinforce`` and remap ``uuid_map``;
        everything below reads them after it. The extractor runs Stage 6c and the
        new-entity writes there, so an edge another worker writes while this batch waits
        on the date call is still seen.
        """
        # Pre-extract (valid_at, invalid_at) for ALL facts in ONE LLM call
        # (PR #89): one call per summary instead of one per fact at ~30s each.
        # A fact the call returns no date for gets the default valid-time below.
        eligible_facts: list[str] = []
        for idx, fact in enumerate(facts):
            ok = idx not in skip_indices and bool(
                uuid_map.get(fact.source) and uuid_map.get(fact.target)
            )
            eligible_facts.append(fact.fact if ok else "")
        batched_dates = self._edge_date_extractor.extract_batch(
            eligible_facts, reference_time=reference_time
        )
        if before_write is not None:
            before_write()

        # Assemble the FULL invalidation list before writing edges so we
        # dispatch ONE UNWIND-batched MATCH+SET round-trip instead of N.
        # ``invalidate`` (the Stage 6b verdicts: old live edges the new fact
        # supersedes) is independent of whether the new edge gets written;
        # always applied, before the new CREATE.
        # The edge-date extractor's ``invalid_at`` is deliberately NOT a source
        # (2026-10-09). It used to be applied at birth, so a fact whose text
        # expressed an end ("User told the assistant on 2026-08-29 to stop X", "the
        # outage ran Aug-Sep") was written already-retired and dropped out of the
        # live index on the spot: 616 such edges in prod, with no successor link,
        # invisible to every recall leg. A fact that records a dated happening or a
        # bounded past state is a TRUE statement for as long as the graph exists;
        # ``t_invalid`` means "superseded by a later fact", and only supersession
        # (Stage 6b) sets it. Currency for the reader
        # rides ``t_valid`` + the schema-058 fields instead.
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

        # Effective valid-from per fact, decided ONCE and shared by the CREATE rows below
        # and by every invalidation the fact causes. Date precedence: explicit date in
        # the fact text > source-segment / source-page date > extraction time.
        now = datetime.now(UTC).isoformat()
        valid_at_by_idx: dict[int, str] = {}
        for idx in range(len(facts)):
            t_valid_pre, _ = batched_dates[idx] if idx < len(batched_dates) else (None, None)
            valid_at_by_idx[idx] = t_valid_pre or default_valid_at or now

        # Every old edge a new fact retires is invalidated AS OF that fact's evidenced
        # valid-from (the correction's own date, else the conversation time it was said
        # in) — not the write clock, which for a backfilled or late-processed chunk can
        # sit months after the change. The writer clamps t_invalid to >= the old edge's
        # own t_valid so a correction dated before the stale edge was even written
        # cannot produce a negative lifetime.
        #
        # Superseder resolution — the schema-028 invalidated_by link that recall's
        # supersession legs REQUIRE to surface a correction (a retired edge without it
        # is invisible at read time): (a) the new edge when the fact is written; (b) the
        # existing edge the fact was judged a pure duplicate of when the write is
        # skipped (drop-in replacement: "now uses X" re-asserting an existing X edge
        # while retiring the Y edge); (c) none when neither exists (unresolved
        # endpoints) — the old edge still retires, link NULL.
        def _superseder(idx: int) -> str | None:
            if idx in skip_indices:
                dups = (reinforce or {}).get(idx) or []
                return dups[0] if dups else None
            return new_uuid_by_idx.get(idx)

        # Stage 6 verdicts ({new fact idx: [old edge uuids]}) were once flattened
        # and written with NO superseder, which is why corrections found at
        # extraction never reached recall's successor lookup.
        by_superseder: dict[str, list[tuple[str, str | None]]] = {}
        orphan_invalidations: list[tuple[str, str | None]] = []
        seen_old: set[tuple[str, str | None]] = set()
        for idx in range(len(facts)):
            # A fact 6c absorbed keeps its verdicts, linked to the absorbing edge.
            olds = invalidate.get(idx, [])
            if not olds:
                continue
            sup = _superseder(idx)
            inv_at = valid_at_by_idx[idx]
            for old_uuid in olds:
                key = (old_uuid, sup)
                if key in seen_old:
                    continue
                seen_old.add(key)
                if sup:
                    by_superseder.setdefault(sup, []).append((old_uuid, inv_at))
                else:
                    orphan_invalidations.append((old_uuid, inv_at))
        for sup, retired in by_superseder.items():
            self._kg.invalidate_edges_batch(retired, group_id, invalidated_by=sup)
        if orphan_invalidations:
            self._kg.invalidate_edges_batch(orphan_invalidations, group_id)

        # Build CREATE rows for every eligible fact in one pass, then dispatch
        # one batched INSERT (see create_edges_batch) instead of a round-trip per
        # fact.
        # Evidence metadata (schema 058): support time = when the chunk was SAID
        # (segment timestamp), never the extracted valid-from — "User ran a marathon
        # in 2019", said today, is supported today and valid from 2019. Only a
        # user-attributed fact counts as support; an assistant-stated one is written
        # with last_supported_at NULL (never user-confirmed) and shows as such.
        support_at = default_valid_at or now
        support_by = max(episode_ids) if episode_ids else None
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
            valid_at_ts = valid_at_by_idx[idx]
            user_stated = getattr(fact, "attribution", "unknown") == "user"
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
                    # Never born-invalidated (see the note above the invalidation
                    # block); create_edges_batch still honours a t_invalid for
                    # callers that genuinely know one, this lane does not pass it.
                    "t_invalid": None,
                    # Web provenance (task #68). None on the episode lane.
                    "web_artifact_id": web_artifact_id,
                    # Evidence metadata (schema 058).
                    "ongoing": bool(getattr(fact, "ongoing", False)),
                    "last_supported_at": support_at if user_stated else None,
                    "last_supported_by": support_by if user_stated else None,
                }
            )
        if create_rows:
            self._kg.create_edges_batch(create_rows, group_id)

        # Capture dedup hits: a fact skipped as a pure duplicate still ASSERTED
        # the matched edge(s). Bump their mention_count + union this chunk's
        # source episodes (provenance) instead of dropping the re-assertion.
        # Forward-only — historical dupes are already gone. The read-side ranking
        # boost on mention_count is a later phase. A USER-attributed re-assertion
        # also refreshes the edge's last_supported_at (schema 058); an assistant
        # repeat unions provenance but leaves currency untouched.
        if reinforce:
            reinforce_items: list[tuple[str, list[int], str | None, int | None]] = []
            for idx in skip_indices:
                user_stated = getattr(facts[idx], "attribution", "unknown") == "user"
                for existing_uuid in reinforce.get(idx, []):
                    reinforce_items.append(
                        (
                            existing_uuid,
                            episode_ids,
                            support_at if user_stated else None,
                            support_by if user_stated else None,
                        )
                    )
            if reinforce_items:
                self._kg.reinforce_edges(reinforce_items, group_id)
