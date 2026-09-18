from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, cast

from ingestion.llm_client import LLM_TRANSPORT_ERRORS
from ingestion.models import (
    ExtractedEntity,
)

if TYPE_CHECKING:
    from ingestion.embedding import EmbeddingModel

from ingestion.extraction_policy import (
    _ERROR_RE,
    _FILE_PATH_RE,
    _URL_RE,
)

logger = logging.getLogger(__name__)


class DeterministicExtractor:
    """Extract entities from episode content and metadata without LLM calls."""

    def extract(self, episodes: list[dict[str, Any]]) -> list[ExtractedEntity]:
        seen: set[str] = set()
        results: list[ExtractedEntity] = []

        def add(name: str, etype: str, summary: str = "") -> None:
            key = f"{etype}:{name.lower()}"
            if key not in seen:
                seen.add(key)
                results.append(ExtractedEntity(name=name, type=etype, summary=summary))

        for ep in episodes:
            content: str = ep.get("content") or ""
            metadata: dict[str, Any] = ep.get("metadata") or {}

            for tool in metadata.get("tools_used", []):
                if isinstance(tool, str) and tool:
                    add(tool.lower(), "Tool")

            for match in _FILE_PATH_RE.finditer(content):
                path = match.group(0)
                parts = [p for p in path.replace("~", "").split("/") if p]
                name = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1] if parts else path
                add(name, "File")

            for match in _URL_RE.finditer(content):
                hostname = match.group(1).split("/")[0]
                add(hostname, "URL")

            for match in _ERROR_RE.finditer(content):
                add(match.group(1), "Issue")

        return results


# ---------------------------------------------------------------------------
# Stage 3 — LLM extractor
# ---------------------------------------------------------------------------


class EntityResolver:
    """Deduplicate new entities against existing KG nodes via vector similarity.

    Returns a mapping of entity name → UUID:
      - "new:<uuid>" if the entity is new
      - "<existing_uuid>" if merged with an existing node
    """

    def __init__(
        self,
        embedder: EmbeddingModel,
        llm_client: Any,
        similarity_threshold: float = 0.85,
        autoconfirm_threshold: float = 0.95,
    ) -> None:
        from ingestion.llm_client import stage_model

        self._embedder = embedder
        self._llm = llm_client
        self._threshold = similarity_threshold
        self._autoconfirm = autoconfirm_threshold
        self._confirm_model = stage_model("DEDUP", self._CONFIRM_MODEL)

    def resolve(
        self,
        entities: list[ExtractedEntity],
        kg_client: Any,
        group_id: str,
        deduper: Any | None = None,
    ) -> dict[str, str]:
        """Resolve entities using write-time dedup then Postgres vector search.

        When ``deduper`` is supplied (a :class:`ingestion.dedup.NodeDeduper`),
        each entity is first run through the 4-strategy write-time dedup
        (exact normalized-name → entropy gate → MinHash/LSH → LLM confirm).
        A hit short-circuits the vector path entirely; a miss falls through
        to the existing vector-similarity logic for backward compatibility.

        The write-time dedup is the conservative path — it catches the
        exact-name and high-Jaccard cases the vector search misses,
        especially when names share a normalized form but the embeddings
        drift apart (long file paths, timestamped filenames, slight
        rewordings between sessions).
        """
        if not entities:
            return {}

        mapping: dict[str, str] = {}
        used_new_uuids: set[str] = set()

        def _new_uuid() -> str:
            new_id = f"new:{uuid.uuid4()}"
            while new_id in used_new_uuids:
                new_id = f"new:{uuid.uuid4()}"
            used_new_uuids.add(new_id)
            return new_id

        # ``pending`` collects every entity that needs an LLM "same entity?"
        # decision, paired with its candidate list. We gather across BOTH the
        # write-time dedup (LSH) path and the vector path, then resolve them
        # all in ONE batched LLM call (Phase 2) instead of one ~30s claude-CLI
        # subprocess per entity. Each candidate dict is {uuid, name, summary}.
        pending: list[tuple[ExtractedEntity, list[dict[str, str]]]] = []

        # Phase 1a: per-entity write-time dedup classification (no LLM).
        # ``classify`` settles exact-name hits, surfaces LSH candidates for
        # confirmation, or returns "none" so the entity falls through to the
        # vector search.
        vector_needed: list[ExtractedEntity] = []
        if deduper is not None:
            for entity in entities:
                kind, payload = deduper.classify(
                    entity.name, entity.summary, entity_type=entity.type
                )
                if kind == "exact":
                    mapping[entity.name] = cast(str, payload)
                elif kind == "candidates":
                    cands = cast("list[tuple[str, str, str, float]]", payload)
                    pending.append(
                        (entity, [{"uuid": u, "name": n, "summary": s} for (u, n, s, _j) in cands])
                    )
                else:  # "none"
                    vector_needed.append(entity)
        else:
            vector_needed = list(entities)

        # Phase 1b: vector-similarity pass for the deduper misses. Only this
        # tail gets an embedding — saves Voyage calls when the deduper settled
        # the majority exactly.
        if vector_needed:
            names = [e.name for e in vector_needed]
            embeddings = self._embedder.embed(names, task="entity")
            for entity, emb in zip(vector_needed, embeddings, strict=True):
                candidates = kg_client.find_similar_nodes(emb, group_id, limit=5)
                if not candidates:
                    mapping[entity.name] = _new_uuid()
                    continue
                # Vector score is cosine distance (0=identical).
                best = min(candidates, key=lambda c: c["score"])
                similarity = 1.0 - float(best["score"])
                if similarity < self._threshold:
                    mapping[entity.name] = _new_uuid()
                elif similarity >= self._autoconfirm:
                    # Trust the embedding, skip the LLM (near-identical names).
                    mapping[entity.name] = cast(str, best["uuid"])
                else:
                    pending.append(
                        (
                            entity,
                            [
                                {
                                    "uuid": cast(str, best["uuid"]),
                                    "name": cast(str, best["name"]),
                                    "summary": "",
                                }
                            ],
                        )
                    )

        # Phase 2: ONE batched confirm for everything that needs a decision.
        if pending:
            decided = self._batch_confirm(pending)
            for entity, _cands in pending:
                matched = decided.get(entity.name)
                mapping[entity.name] = matched if matched else _new_uuid()

        return mapping

    # Binary "same entity?" classification — use Haiku, not Sonnet.
    # ~10x cheaper, accuracy is fine for a short structured decision.
    # Overridable via SYNAPSE_DEDUP_MODEL (it's a duplicate decision).
    _CONFIRM_MODEL = "claude-haiku-4-5"

    def _batch_confirm(
        self, pending: list[tuple[ExtractedEntity, list[dict[str, str]]]]
    ) -> dict[str, str]:
        """Resolve every (entity, candidates) pair in ONE LLM call.

        Returns ``{entity_name: matched_uuid}`` for entities the model judged a
        duplicate; entities absent from the result are new. Candidates are
        scoped per-entity, so this is behaviour-equivalent to the old per-pair
        confirm — it just collapses N ~30s claude-CLI subprocesses into one.

        Failure policy is CONSERVATIVE: if the call errors or the response
        can't be parsed (e.g. the SDK returns an empty body — the old code's
        ``Expecting value: line 1 column 1`` case), we return ``{}`` so every
        pending entity becomes a NEW node. A spurious new node is cheap (the
        nightly dedup sweeps it); a wrong merge silently corrupts the graph.
        """
        if not pending:
            return {}

        # No LLM available (tests / bench): trust the top candidate, mirroring
        # the prior no-LLM path in both the deduper and the vector match.
        if self._llm is None:
            return {e.name: c[0]["uuid"] for e, c in pending if c}

        from ingestion.llm_client import structured_call
        from ingestion.llm_schemas import BatchNodeDedupResult
        from ingestion.prompts.dedupe_nodes import build_batch_prompt

        items: list[dict[str, Any]] = []
        for i, (entity, cands) in enumerate(pending):
            items.append(
                {
                    "id": i,
                    "name": entity.name,
                    "summary": (entity.summary or "")[:600],
                    "candidates": [
                        {
                            "candidate_id": j,
                            "name": c["name"],
                            "summary": (c.get("summary") or "")[:600],
                        }
                        for j, c in enumerate(cands)
                    ],
                }
            )

        max_tokens = min(4096, 128 + 24 * len(pending))
        try:
            batch = structured_call(
                self._llm,
                output_model=BatchNodeDedupResult,
                messages=build_batch_prompt(items),
                model=self._confirm_model,
                max_tokens=max_tokens,
            )
        except LLM_TRANSPORT_ERRORS:
            # "Treat all as distinct" is a defensible reading of a model that answered
            # badly. It is not a defensible reading of a model that never answered —
            # that would write a fragmented graph and mark the item done.
            raise
        except Exception as e:
            logger.warning(
                "batch dedup confirm failed for %d entit%s (%s); treating all as distinct",
                len(pending),
                "y" if len(pending) == 1 else "ies",
                str(e)[:120],
            )
            return {}

        decided: dict[str, str] = {}
        for r in batch.results:
            if not (0 <= r.id < len(pending)):
                continue
            entity, cands = pending[r.id]
            if 0 <= r.duplicate_candidate_id < len(cands):
                decided[entity.name] = cands[r.duplicate_candidate_id]["uuid"]
        return decided


# ---------------------------------------------------------------------------
# ExtractionPipeline — orchestrates stages 2-7
# ---------------------------------------------------------------------------
