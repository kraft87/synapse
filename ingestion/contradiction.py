"""Write-time contradiction detector for the bi-temporal edge model.

Phase 3 of the Graphiti -> Synapse port. Provides ``ContradictionDetector``,
the safety net that runs immediately before ``KGClient.create_edge``
writes a RELATES_TO edge to the graph. When it confirms that an incoming
fact contradicts one or more *live* (``t_invalid IS NULL``) existing edges,
those edges have their ``t_invalid`` set instead of being deleted -- the
bi-temporal model is additive-only and preserves audit history.

Pipeline (mirrors Graphiti's ``resolve_extracted_edge`` shape, slimmed for
the writer-side hook):

1. Structural filter -- ``find_edges_by_pair`` returns live edges sharing
   the new fact's (source, target). Only same-pair candidates are
   considered for contradiction; a fact about a different node pair
   cannot logically contradict the new one.
2. Vector similarity gate -- candidates' ``fact_embedding`` is compared to
   the new fact's embedding via cosine. Below the threshold (default 0.7)
   the candidate is dropped *before* the LLM step. This keeps the LLM
   call rare and cheap.
3. LLM confirmation -- Haiku 4.5 receives the new fact + the surviving
   candidates and returns a JSON list of UUIDs the new fact contradicts.
   No UUIDs are sent into the prompt (only ``idx`` -> ``uuid`` map
   maintained by the caller) so the LLM cannot hallucinate UUID strings.

The detector does NOT write to the graph. ``detect_contradictions`` only
returns the UUIDs to invalidate; the caller (``KGClient.create_edge``)
calls ``invalidate_edge`` on each.

This is deliberately decoupled from the extractor's Stage 6 contradiction
prompt: Stage 6 sees a much richer candidate pool (BM25 + vector RRF over
ALL live edges) and decides both duplicate AND contradiction at extraction
time. The writer-side detector is the safety net for callers that bypass
the extractor (dream pipeline, manual writes, future ingestion sources).
"""

from __future__ import annotations

import json
import logging
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ingestion.embedding import EmbeddingModel
    from ingestion.kg_client import KGClient
    from ingestion.models import ExtractedFact

logger = logging.getLogger(__name__)


# Cosine similarity threshold below which candidates are dropped before the
# LLM step. 0.7 matches the spec; empirically anything below ~0.65 on
# voyage-4-large fact embeddings is unrelated text. Above 0.7 we still let
# the LLM make the call -- it can rule out false positives that share
# surface form but mean different things ("uses X for A" vs "uses X for B"
# embed close but are not contradictions).
_DEFAULT_SIMILARITY_THRESHOLD = 0.7

# Top-K candidates passed to the LLM after the similarity gate. 5 is plenty
# for the same-pair filter -- most pairs have <5 live edges total.
_DEFAULT_TOP_K = 5


def _cosine(a: list[float], b: list[float]) -> float:
    """Plain cosine similarity, returns 0.0 for zero-magnitude vectors."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# Phase 4: the hand-rolled stub prompt has been replaced with the
# verbatim Graphiti-ported contradiction prompt
# (``ingestion.prompts.invalidate_edges``). The structured-response schema
# below matches Graphiti's `EdgeDuplicate.contradicted_facts` shape and is
# used as `response_format` to constrain Haiku's output.

_CONTRADICTION_SCHEMA: dict[str, Any] = {
    "type": "json",
    "schema": {
        "type": "object",
        "properties": {
            "contradicted_facts": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["contradicted_facts"],
        "additionalProperties": False,
    },
}

_BATCH_CONTRADICTION_SCHEMA: dict[str, Any] = {
    "type": "json",
    "schema": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "contradicted_facts": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["id", "contradicted_facts"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    },
}


class ContradictionDetector:
    """Writer-side bi-temporal contradiction safety net.

    Instantiate once per ingestion process and pass to ``KGClient.create_edge``
    via its ``detector`` kwarg. The detector is stateless -- all state lives in
    Postgres -- so a single instance is shared across all groups/writes.
    """

    def __init__(
        self,
        kg_client: KGClient,
        embedder: EmbeddingModel,
        llm_client: Any,
        *,
        similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
        top_k: int = _DEFAULT_TOP_K,
        model: str = "claude-haiku-4-5",
    ) -> None:
        self._kg = kg_client
        self._embedder = embedder
        self._llm = llm_client
        self._threshold = similarity_threshold
        self._top_k = top_k
        self._model = model

    def detect_contradictions(
        self,
        new_fact: ExtractedFact,
        source_uuid: str,
        target_uuid: str,
        group_id: str,
        *,
        fact_embedding: list[float] | None = None,
    ) -> list[str]:
        """Return UUIDs of LIVE edges the new fact contradicts.

        Always returns a list (possibly empty). Never raises -- on any
        unexpected error the detector logs and returns [] so a contradiction
        miss never blocks an edge write.

        ``fact_embedding`` may be supplied by the caller (the extractor
        already embeds fact texts in batch) to avoid a second embed call.
        When None, the detector embeds ``new_fact.fact`` itself.
        """
        try:
            # Step 1: structural filter -- live edges sharing the same pair.
            # ``find_edges_by_pair`` filters on ``t_invalid IS NULL`` so
            # already-retired edges can't re-enter the candidate pool.
            candidates = self._kg.find_edges_by_pair(source_uuid, target_uuid, group_id)
            if not candidates:
                return []

            # Step 2: vector similarity gate. Skip the LLM entirely when no
            # candidate clears the threshold.
            if fact_embedding is None:
                fact_embedding = self._embedder.embed([new_fact.fact], task="document")[0]

            scored: list[tuple[float, dict[str, Any]]] = []
            for cand in candidates:
                cand_emb = cand.get("fact_embedding")
                if not cand_emb:
                    # No embedding stored -- be conservative and include it,
                    # the LLM step will still gate. Score=1.0 ensures it
                    # survives the top-K cut.
                    scored.append((1.0, cand))
                    continue
                sim = _cosine(list(fact_embedding), list(cand_emb))
                if sim >= self._threshold:
                    scored.append((sim, cand))

            if not scored:
                return []

            # Top-K by similarity desc.
            scored.sort(key=lambda kv: kv[0], reverse=True)
            top = [c for _, c in scored[: self._top_k]]

            # Step 3: LLM confirm via Graphiti's verbatim contradiction
            # prompt (Phase 4). UUIDs never enter the prompt — only the
            # caller-owned idx -> uuid map.
            idx_to_uuid: dict[int, str] = {}
            lines: list[str] = []
            for i, cand in enumerate(top):
                idx_to_uuid[i] = cand["uuid"]
                lines.append(f"[{i}] {cand['fact']}")

            from .prompts.invalidate_edges import build_prompt

            messages = build_prompt(
                {
                    "new_fact": new_fact.fact,
                    "existing_facts": "\n".join(lines),
                }
            )

            response = self._llm.messages.create(
                model=self._model,
                max_tokens=200,
                messages=messages,
                response_format=_CONTRADICTION_SCHEMA,
            )
            raw = response.content[0].text.strip()
            start = raw.find("{")
            if start < 0:
                raise ValueError(f"no JSON object in response: {raw[:200]!r}")
            data, _ = json.JSONDecoder().raw_decode(raw[start:])
            contradicted_idx = data.get("contradicted_facts", [])
            return [idx_to_uuid[i] for i in contradicted_idx if i in idx_to_uuid]
        except Exception as exc:
            # Never let a detector failure block the write. Log and return [].
            logger.warning(
                "ContradictionDetector failed for fact=%r group=%s: %s",
                new_fact.fact[:80],
                group_id,
                exc,
            )
            return []

    def detect_contradictions_batch(
        self,
        new_facts: list[ExtractedFact],
        source_uuids: list[str],
        target_uuids: list[str],
        group_id: str,
        *,
        fact_embeddings: list[list[float] | None] | None = None,
    ) -> list[list[str]]:
        """Batched same-shape variant of detect_contradictions.

        For every (fact, src, tgt) triple in the input lists, return a list
        of edge UUIDs the new fact contradicts. Output is parallel to input:
        result[i] is the UUIDs for new_facts[i]. Pure-novel facts (no live
        edges sharing the pair, or all candidates below the similarity
        threshold) get an empty list with NO LLM call.

        One LLM call total: facts that survive the pair + similarity gates
        are pooled into a single prompt with per-fact idx scoping (same
        shape as ``build_batch_resolution_prompt`` in extractor.py and the
        edge-dates batch prompt). Each surviving fact carries its OWN
        candidate list; the LLM returns contradicted idx per fact id.

        Never raises. Any failure (no candidates, gate-filtered all,
        transport error, parse error) collapses to ``[]`` for that fact,
        matching the single-call detect_contradictions's posture.

        ``fact_embeddings`` is parallel to ``new_facts``; ``None`` slots
        embed on-demand. The extractor already embeds fact texts in
        batch (see ``_process_facts_for_group``), so pass them through.
        """
        n = len(new_facts)
        out: list[list[str]] = [[] for _ in range(n)]
        if n == 0:
            return out
        if not (len(source_uuids) == len(target_uuids) == n):
            return out
        embs = fact_embeddings or [None] * n
        if len(embs) != n:
            embs = [None] * n

        # Per-fact pre-LLM filter: structural pair match + similarity gate.
        # Items below either bar never enter the batched prompt.
        per_fact: list[tuple[int, ExtractedFact, list[dict[str, Any]], dict[int, str]] | None] = [
            None
        ] * n
        items_for_prompt: list[dict[str, Any]] = []
        try:
            for i, (fact, src, tgt) in enumerate(
                zip(new_facts, source_uuids, target_uuids, strict=False)
            ):
                if not src or not tgt:
                    continue
                candidates = self._kg.find_edges_by_pair(src, tgt, group_id)
                if not candidates:
                    continue
                emb = embs[i]
                if emb is None:
                    emb = self._embedder.embed([fact.fact], task="document")[0]
                scored: list[tuple[float, dict[str, Any]]] = []
                for cand in candidates:
                    cand_emb = cand.get("fact_embedding")
                    if not cand_emb:
                        scored.append((1.0, cand))
                        continue
                    sim = _cosine(list(emb), list(cand_emb))
                    if sim >= self._threshold:
                        scored.append((sim, cand))
                if not scored:
                    continue
                scored.sort(key=lambda kv: kv[0], reverse=True)
                top = [c for _, c in scored[: self._top_k]]
                idx_to_uuid = {j: c["uuid"] for j, c in enumerate(top)}
                per_fact[i] = (i, fact, top, idx_to_uuid)
                items_for_prompt.append(
                    {
                        "id": i,
                        "new_fact": fact.fact,
                        "existing_facts": [
                            {"idx": j, "fact": c["fact"]} for j, c in enumerate(top)
                        ],
                    }
                )
        except Exception as exc:
            logger.warning(
                "ContradictionDetector.batch pre-LLM filter failed group=%s: %s",
                group_id,
                exc,
            )
            return out

        if not items_for_prompt:
            return out

        # Single batched LLM call covering every surviving fact's candidates.
        try:
            import json as _json

            from .prompts.invalidate_edges import build_batch_prompt

            messages = build_batch_prompt(items_for_prompt)
            response = self._llm.messages.create(
                model=self._model,
                max_tokens=max(300, 80 * len(items_for_prompt)),
                messages=messages,
                response_format=_BATCH_CONTRADICTION_SCHEMA,
            )
            raw = response.content[0].text.strip()
            start = raw.find("{")
            if start < 0:
                return out
            data, _ = _json.JSONDecoder().raw_decode(raw[start:])
        except Exception as exc:
            logger.warning(
                "ContradictionDetector.batch LLM call failed group=%s n=%d: %s",
                group_id,
                len(items_for_prompt),
                exc,
            )
            return out

        for r in data.get("results", []):
            fid = r.get("id")
            if not isinstance(fid, int) or not (0 <= fid < n) or per_fact[fid] is None:
                continue
            _, _fact, _top, idx_to_uuid = per_fact[fid]  # type: ignore[misc]
            contradicted_idx = r.get("contradicted_facts", [])
            out[fid] = [idx_to_uuid[j] for j in contradicted_idx if j in idx_to_uuid]
        return out
