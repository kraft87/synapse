"""Write-time entity deduplication for KG ingestion.

Ports Graphiti's 4-strategy node dedup (``graphiti_core/utils/maintenance/
dedup_helpers.py`` + ``node_operations.resolve_extracted_nodes``) into a
single ``NodeDeduper`` class scoped to one KG group at a time.

The strategies, tried in order, return the first match:

1. **Exact normalized name** — Cypher lookup against ``Entity.normalized_name``
   (must be indexed; see ``scripts/migrate_normalized_name.py``).
2. **Entropy gate** — names with low Shannon entropy or fewer than
   ``_MIN_NAME_LENGTH`` characters short-circuit *out* of fuzzy matching.
   Two distinct "api" entities should stay distinct; the LSH+Jaccard
   shingles on a 3-letter string are too unreliable to trust.
3. **MinHash/LSH Jaccard** — character-3-gram MinHash signatures + LSH
   keyed on entity names produce a candidate set, then Jaccard is
   recomputed exactly on the shingles. Default threshold 0.7.
4. **LLM confirm** — top-K (default 3) LSH candidates that survive the
   entropy filter are sent to Haiku 4.5 with a "are these the same
   entity? yes/no/uncertain" prompt, one pair per call. Returns the
   canonical UUID for the first ``yes``.

If no strategy matches, ``find_or_none`` returns ``None`` and the caller
should INSERT the entity as new. If any strategy matches, the caller
should treat the entity as a duplicate and route edges to the returned
UUID instead — and may merge the candidate's summary (longer wins).

Why this layer matters: the retired FalkorDB-era cleanup CLI
(``dream/entity_dedup.py``, deleted in #67 PR 3) used to catch ~1,603
duplicates (~29% of the technical graph), the largest single cluster
type being exact-name (301 clusters covering 1,603 nodes). Catching
exact-name duplicates at write time prevents those duplicates from
ever entering the graph. Merge tooling for the cases this layer still
misses (names that aren't normalized-identical but whose Jaccard sits
just below the LLM-confirm gate) gets rebuilt on Postgres with the
task #49 alias-normalization work.
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from ingestion.kg_client import KGClient

logger = logging.getLogger(__name__)

# --- Tunables (mirror Graphiti's dedup_helpers.py constants) ---------------

# Shannon entropy over character histogram. Graphiti uses 1.5; we adopt 2.0
# (the spec asks for ~2.0). Names below this entropy are treated as
# "too ambiguous to fuzzy-match" — the exact-name pass still runs, but the
# MinHash/LSH path is skipped so two distinct one-word entities (e.g. two
# different "api" entities from different contexts) stay distinct.
_NAME_ENTROPY_THRESHOLD = 2.0

# A name shorter than this AND with fewer than _MIN_TOKEN_COUNT tokens also
# fails the entropy gate regardless of its entropy score — too short to
# trust shingles.
_MIN_NAME_LENGTH = 6
_MIN_TOKEN_COUNT = 2

# Jaccard similarity floor for LSH hits. Below this we drop the candidate
# silently; at or above it we send to the LLM for a final confirm.
_LSH_JACCARD_THRESHOLD = 0.7

# Number of MinHash permutations — direct port of Graphiti's value.
_MINHASH_PERMUTATIONS = 128

# Number of top LSH candidates passed to the LLM confirm pass.
_LLM_CONFIRM_TOP_K = 3

# Default Haiku model for the LLM confirm step. Background-work standard,
# per PR #11 and the project conventions (~10x cheaper than Sonnet, fine
# for binary yes/no on short pairs). Overridable via SYNAPSE_DEDUP_MODEL.
_DEFAULT_LLM_MODEL = "claude-haiku-4-5"

# --- Canonical identity aliases (task #49) ---------------------------------
# The extractor keeps minting fresh 'User' / 'Kyle Doucette' nodes that the
# write-time dedup can't catch (cos(User, Kyle) = 0.53, far below threshold;
# names aren't normalized-identical either), re-fragmenting the identity hub
# that the 2026-05-28 one-shot merge consolidated. Map known aliases to the
# canonical node name BEFORE resolution so the exact-name short-circuit lands
# on the existing hub. Keys are _normalize_name() output; values are the
# canonical entity name as written.
#
# Deliberately NOT mapped: the assistant cluster (Claude / neuron / Neuron) —
# whether those are one identity is a pending decision, and merging them is
# not reversible at this layer.
CANONICAL_ALIASES: dict[str, str] = {
    "user": "Kyle",
    "the user": "Kyle",
    "kyle doucette": "Kyle",
}

# --- Type-compatibility gate for FUZZY merge candidates (entity taxonomy) -----
# Once entities carry a supertype (schema 020), drop a fuzzy (non-exact-name) merge
# candidate whose supertype is confidently INCOMPATIBLE with the new entity's — a
# "Database" named synapse shouldn't even be a merge candidate for the "Service" named
# Synapse. SOFT by design (Oracle): the exact-name short-circuit (Strategy 1) runs FIRST
# and always merges identical names regardless of type, so this only ever filters fuzzy
# candidates; and it only DROPS when BOTH sides are non-permissive, different, and not
# allowlisted — so an untyped/unknown/Concept side, or the LLM typing the same thing two
# ways, never blocks a real merge (it degrades to current behavior). Default ON;
# SYNAPSE_DEDUP_TYPE_GATE=0 disables for instant rollback.
_TYPE_GATE_ON = os.getenv("SYNAPSE_DEDUP_TYPE_GATE", "1") not in ("0", "false", "")
# Permissive supertypes never block a merge (abstract/catch-all/untyped).
_TYPE_PERMISSIVE = frozenset({None, "", "other", "Concept"})
# Cross-supertype pairs that MAY still merge (genuinely fungible at the boundary).
_TYPE_ALLOWLIST = frozenset(
    frozenset(p)
    for p in (
        # genuinely-fungible software boundaries (Oracle's examples). Deliberately
        # NOT Service<->Database or Service<->Agent — those are the false-merge cases
        # the gate exists to PREVENT (Database "synapse" vs app "Synapse"; Agent=actor).
        ("Tool", "Service"),
        ("Tool", "Library"),
        ("Library", "Service"),
        ("Project", "Product"),
    )
)


def _type_compatible(a: str | None, b: str | None) -> bool:
    """True if two supertypes may merge: either permissive/untyped, equal, or allowlisted.
    Conservative — only False when BOTH are confidently-typed, different, and not paired."""
    if a in _TYPE_PERMISSIVE or b in _TYPE_PERMISSIVE:
        return True
    if a == b:
        return True
    return frozenset((a, b)) in _TYPE_ALLOWLIST


def canonical_name(name: str) -> str | None:
    """The canonical entity name for a known alias, or None if not aliased."""
    return CANONICAL_ALIASES.get(_normalize_name(name))


# --- Normalization helpers --------------------------------------------------


def _normalize_name(name: str) -> str:
    """Lowercase, strip leading/trailing whitespace and punctuation.

    Mirrors Graphiti's ``_normalize_string_exact`` (dedup_helpers.py:39-42)
    but additionally trims leading/trailing punctuation so trailing periods
    or quote marks don't fragment the exact-name bucket. Spec calls this
    "lowercase, strip whitespace + leading/trailing punctuation."
    """
    if not name:
        return ""
    # Lowercase, collapse internal whitespace into single spaces.
    collapsed = re.sub(r"\s+", " ", name.lower())
    # Strip leading/trailing whitespace and a handful of punctuation chars
    # that commonly bookend names from extractor output.
    return collapsed.strip(" \t\n\r\f\v.,;:!?\"'()[]{}<>")


def _name_entropy(normalized_name: str) -> float:
    """Approximate text specificity via Shannon entropy over characters.

    Spaces are dropped first so two-word names aren't penalized for the
    separator; everything else is counted as-is.
    """
    if not normalized_name:
        return 0.0

    counts: dict[str, int] = {}
    for char in normalized_name.replace(" ", ""):
        counts[char] = counts.get(char, 0) + 1

    total = sum(counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy


def has_high_entropy(normalized_name: str) -> bool:
    """Decide whether a name carries enough information to trust fuzzy matching.

    Returns False for names that are both short AND single-token (those need
    an exact match or LLM disambiguation; LSH on 3-grams of "api" is noise).
    Otherwise gates on the Shannon entropy threshold.
    """
    token_count = len(normalized_name.split())
    if len(normalized_name) < _MIN_NAME_LENGTH and token_count < _MIN_TOKEN_COUNT:
        return False
    return _name_entropy(normalized_name) >= _NAME_ENTROPY_THRESHOLD


def _normalize_for_fuzzy(name: str) -> str:
    """Tighter normalization for shingling.

    Replaces every non-alphanumeric character (besides apostrophes) with
    a space, then collapses whitespace and strips. Mirrors Graphiti's
    ``_normalize_name_for_fuzzy`` so "synapse-poller" and "synapse poller"
    produce identical shingle sets after the dash is treated as a separator.
    """
    base = _normalize_name(name)
    if not base:
        return ""
    fuzzed = re.sub(r"[^a-z0-9' ]", " ", base)
    return re.sub(r"\s+", " ", fuzzed).strip()


def _shingles(normalized_name: str, n: int = 3) -> set[str]:
    """Character n-gram shingles for MinHash/Jaccard.

    Caller should pass a fuzzy-normalized string (``_normalize_for_fuzzy``)
    so punctuation has been turned into whitespace first; spaces are then
    stripped here so "synapse poller" and "synapsepoller" produce the same
    shingle set. Names shorter than ``n`` are turned into a single-element
    set so they can still participate in MinHash (rarely collides).
    """
    cleaned = normalized_name.replace(" ", "")
    if not cleaned:
        return set()
    if len(cleaned) < n:
        return {cleaned}
    return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    """Exact Jaccard similarity over shingle sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


# --- Optional LLM confirm response schema ----------------------------------
# Phase 4: the legacy "yes/no/uncertain" mini-prompt has been retired in
# favor of Graphiti's verbatim NodeDuplicate prompt. The prompt body lives
# in `ingestion/prompts/dedupe_nodes.py`; the structured-response schema
# below mirrors Graphiti's `NodeDuplicate` shape and is sent as
# `response_format` to constrain Haiku's output.

_NODE_DEDUP_SCHEMA: dict[str, Any] = {
    "type": "json",
    "schema": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "name": {"type": "string"},
            "duplicate_candidate_id": {"type": "integer"},
        },
        "required": ["id", "name", "duplicate_candidate_id"],
        "additionalProperties": False,
    },
}


# ---------------------------------------------------------------------------
# NodeDeduper
# ---------------------------------------------------------------------------


class NodeDeduper:
    """Per-group write-time entity deduplicator.

    Instantiate one per group_id — both the index
    state and the LLM client are held on the instance. The first call to
    ``find_or_none`` builds the LSH index by scanning every Entity in the
    group; subsequent calls reuse it. ``register`` updates the in-memory
    index when the caller decides to INSERT (i.e. find_or_none returned
    None) so later calls within the same run see the freshly-written node.

    Public surface:
        - ``find_or_none(name, summary)`` → existing UUID or None
        - ``register(name, uuid)`` — call after INSERT so the index stays warm
        - ``merge_summary(existing_summary, new_summary)`` → str (longer wins)
    """

    def __init__(
        self,
        kg_client: KGClient,
        group_id: str,
        llm_client: Any | None = None,
        *,
        llm_model: str | None = None,
        lsh_threshold: float = _LSH_JACCARD_THRESHOLD,
        top_k: int = _LLM_CONFIRM_TOP_K,
    ) -> None:
        from ingestion.llm_client import stage_model

        self._kg = kg_client
        self._group_id = group_id
        self._llm = llm_client
        self._llm_model = llm_model or stage_model("DEDUP", _DEFAULT_LLM_MODEL)
        self._lsh_threshold = lsh_threshold
        self._top_k = top_k

        # Lazy-built LSH state. None means "not yet hydrated."
        self._lsh: Any | None = None
        self._shingles_by_uuid: dict[str, set[str]] = {}
        self._summary_by_uuid: dict[str, str] = {}
        self._name_by_uuid: dict[str, str] = {}
        self._uuid_by_normalized_name: dict[str, str] = {}
        self._supertype_by_uuid: dict[str, str | None] = {}
        self._type_map: dict[str, str] = {}  # subtype -> supertype (taxonomy gate)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self, name: str, summary: str | None = None, entity_type: str | None = None
    ) -> tuple[str, str | None] | tuple[str, list[tuple[str, str, str, float]]]:
        """Run the NO-LLM strategies (1-3) and classify ``name`` for the caller.

        Returns one of:
          - ``("exact", uuid)`` — settled by exact normalized-name (or an
            in-run ``register``); the caller should merge, no LLM needed.
          - ``("candidates", [(uuid, name, summary, jaccard), ...])`` — the
            top-K LSH candidates above the Jaccard floor that need an LLM
            "same entity?" decision. Caller decides how to confirm (one pair
            at a time via :meth:`find_or_none`, or batched across many
            entities by the resolver).
          - ``("none", None)`` — no exact hit, entropy-gated, or no LSH
            candidate; the caller should treat the name as new (or fall
            through to its own vector search).

        Splitting candidate generation (cheap, deterministic) from the LLM
        confirm (a ~30s claude-CLI subprocess per call) lets the resolver
        gather every entity needing a decision and confirm them all in ONE
        batched call instead of one subprocess per entity.
        """
        normalized = _normalize_name(name)
        if not normalized:
            return ("none", None)

        # Strategy 1: exact normalized-name match (Cypher / in-mem).
        exact = self._exact_name_lookup(normalized)
        if exact is not None:
            logger.debug("dedup: exact-name hit %r → %s", name, exact[:8])
            return ("exact", exact)

        # Strategy 2: entropy gate. Below threshold → too ambiguous to merge.
        if not has_high_entropy(normalized):
            logger.debug("dedup: %r below entropy threshold; not fuzzy-matching", name)
            return ("none", None)

        # Strategy 3: MinHash/LSH candidate generation (no LLM yet).
        self._ensure_lsh_loaded()
        candidates = self._lsh_candidates(_normalize_for_fuzzy(name))
        if not candidates:
            return ("none", None)

        # Type-compatibility gate (soft): drop fuzzy candidates whose supertype is
        # confidently incompatible with the new entity's. Only fires when the NEW entity
        # is confidently typed; _type_compatible lets untyped/Concept/allowlisted pairs
        # through. Exact-name (Strategy 1) already returned above, so identical names are
        # never gated.
        if _TYPE_GATE_ON and entity_type and self._type_map:
            new_super = self._type_map.get(entity_type)
            if new_super not in _TYPE_PERMISSIVE:
                candidates = [
                    (u, j)
                    for (u, j) in candidates
                    if _type_compatible(new_super, self._supertype_by_uuid.get(u))
                ]
                if not candidates:
                    return ("none", None)

        enriched = [
            (
                cand_uuid,
                self._name_by_uuid.get(cand_uuid, ""),
                self._summary_by_uuid.get(cand_uuid, ""),
                jacc,
            )
            for cand_uuid, jacc in candidates[: self._top_k]
        ]
        return ("candidates", enriched)

    def find_or_none(self, name: str, summary: str | None = None) -> str | None:
        """Run the 4 strategies in order and return a canonical UUID or None.

        The contract: if this returns a non-None UUID, the caller MUST NOT
        INSERT a new node — instead, route incoming edges to the returned
        UUID and (if appropriate) update the existing node's summary via
        ``merge_summary``.

        Strategies 1-3 live in :meth:`classify`; this method adds the
        per-pair LLM confirm (strategy 4). The resolver uses ``classify``
        directly so it can batch the confirm across many entities; this
        single-pair path remains for standalone callers and tests.
        """
        kind, payload = self.classify(name, summary)
        if kind == "exact":
            return cast("str", payload)
        if kind == "none":
            return None

        candidates = cast("list[tuple[str, str, str, float]]", payload)

        # If no LLM available, fall back to "Jaccard alone above threshold
        # is enough." This is the no-LLM path — still useful in tests and
        # for the bench-mark case where Haiku is unavailable.
        if self._llm is None:
            cand_uuid, cand_name, _cand_summary, jacc = candidates[0]
            if jacc >= self._lsh_threshold:
                logger.debug(
                    "dedup: LSH match (no LLM) %r ~ %r jacc=%.2f → %s",
                    name,
                    cand_name or "?",
                    jacc,
                    cand_uuid[:8],
                )
                return cand_uuid
            return None

        # Send top-K to Haiku one pair at a time. First "yes" wins.
        for cand_uuid, cand_name, cand_summary, jacc in candidates:
            try:
                verdict = self._llm_confirm(name, summary or "", cand_name, cand_summary)
            except Exception as exc:
                logger.warning(
                    "dedup: LLM confirm failed for %r vs %r: %s",
                    name,
                    cand_name or "?",
                    exc,
                )
                # On LLM failure, mirror the existing extractor's policy:
                # trust the strong-Jaccard signal rather than crash the run.
                if jacc >= self._lsh_threshold:
                    return cand_uuid
                continue
            if verdict == "yes":
                logger.debug(
                    "dedup: LLM confirmed %r ~ %r jacc=%.2f → %s",
                    name,
                    cand_name or "?",
                    jacc,
                    cand_uuid[:8],
                )
                return cand_uuid
            # "no" or "uncertain" → keep trying the next LSH candidate.
        return None

    def register(self, name: str, node_uuid: str, summary: str = "") -> None:
        """Add a freshly-inserted node to the in-memory LSH state.

        Call after a successful INSERT so that subsequent ``find_or_none``
        calls in the same run can dedupe against this node. Safe to call
        even before ``_ensure_lsh_loaded`` has run; it just primes the
        dicts and (if the LSH is already built) inserts into it.
        """
        normalized = _normalize_name(name)
        if not normalized:
            return
        self._uuid_by_normalized_name.setdefault(normalized, node_uuid)
        self._name_by_uuid[node_uuid] = name
        self._summary_by_uuid[node_uuid] = summary or ""
        shingles = _shingles(_normalize_for_fuzzy(name))
        self._shingles_by_uuid[node_uuid] = shingles
        if self._lsh is not None:
            try:
                mh = self._minhash(shingles)
                # MinHashLSH requires unique keys; skip if already present.
                if node_uuid not in self._lsh:
                    self._lsh.insert(node_uuid, mh)
            except Exception as exc:
                logger.debug("dedup: register %s failed to update LSH: %s", node_uuid[:8], exc)

    @staticmethod
    def merge_summary(existing: str | None, incoming: str | None) -> str:
        """Pick the longer of two summaries, preferring non-empty over empty.

        Conservative: doesn't synthesize, doesn't call an LLM. The longer
        summary is almost always the one with more detail. If the caller
        wants LLM-merged summaries, that's the nightly-dream pipeline's
        job — write-time dedup stays cheap.
        """
        a = (existing or "").strip()
        b = (incoming or "").strip()
        if not a and not b:
            return ""
        if not a:
            return b
        if not b:
            return a
        return b if len(b) > len(a) else a

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _exact_name_lookup(self, normalized: str) -> str | None:
        """Indexed exact lookup on ``normalized_name`` via the graph client.

        Includes the slow lower(trim(name)) fallback for pre-migration rows
        that lack the property.
        """
        # Honor in-memory hits first — covers entities registered this run.
        in_mem = self._uuid_by_normalized_name.get(normalized)
        if in_mem is not None:
            return in_mem

        try:
            return self._kg.entity_uuid_by_normalized_name(normalized, self._group_id)
        except Exception:
            return None

    def _ensure_lsh_loaded(self) -> None:
        """Hydrate the in-memory LSH index from every Entity in the group.

        Lazy because building the LSH costs O(N) MinHash inserts. We pay
        that cost once per pipeline run per group; subsequent dedup calls
        reuse the warm index.
        """
        if self._lsh is not None:
            return

        try:
            from datasketch import MinHashLSH
        except ImportError as exc:
            # No datasketch → skip fuzzy strategy. Exact-name short-circuit
            # still works.
            logger.warning("dedup: datasketch unavailable (%s); skipping LSH", exc)
            self._lsh = _NullLSH()
            return

        try:
            rows = self._kg.load_entities(self._group_id)
        except Exception as exc:
            logger.warning("dedup: failed to load entities for LSH (%s); skipping", exc)
            self._lsh = _NullLSH()
            return

        # Taxonomy map for the type-compatibility gate (empty if unseeded -> gate no-ops).
        if _TYPE_GATE_ON:
            try:
                self._type_map = self._kg.load_type_map()
            except Exception as exc:
                logger.warning("dedup: failed to load type map (%s); type gate off", exc)
                self._type_map = {}

        lsh = MinHashLSH(threshold=self._lsh_threshold, num_perm=_MINHASH_PERMUTATIONS)
        for row in rows:
            node_uuid = row[0]
            name = row[1] or ""
            summary = row[2] or ""
            self._supertype_by_uuid[node_uuid] = row[3] if len(row) > 3 else None
            normalized = _normalize_name(name)
            if not normalized:
                continue
            self._name_by_uuid[node_uuid] = name
            self._summary_by_uuid[node_uuid] = summary
            self._uuid_by_normalized_name.setdefault(normalized, node_uuid)
            shingles = _shingles(_normalize_for_fuzzy(name))
            if not shingles:
                continue
            self._shingles_by_uuid[node_uuid] = shingles
            try:
                mh = self._minhash(shingles)
                # Tolerate duplicate keys from any stragglers (already-merged).
                if node_uuid not in lsh:
                    lsh.insert(node_uuid, mh)
            except Exception as exc:
                logger.debug("dedup: skipping %s in LSH (%s)", node_uuid[:8], exc)
        self._lsh = lsh

    def _minhash(self, shingles: set[str]) -> Any:
        """Construct a MinHash for a shingle set (datasketch import-deferred)."""
        from datasketch import MinHash

        mh = MinHash(num_perm=_MINHASH_PERMUTATIONS)
        for shingle in shingles:
            mh.update(shingle.encode("utf-8"))
        return mh

    def _lsh_candidates(self, normalized: str) -> list[tuple[str, float]]:
        """Return [(uuid, jaccard), ...] sorted by descending Jaccard.

        LSH returns an approximate candidate set; we recompute exact
        Jaccard on the shingles per candidate so the returned similarity
        is trustworthy. Filters out candidates below ``_lsh_threshold``
        — those that slipped through as approximate LSH hits but don't
        meet the true similarity floor.
        """
        if self._lsh is None or isinstance(self._lsh, _NullLSH):
            return []
        query_shingles = _shingles(normalized)
        if not query_shingles:
            return []
        try:
            mh = self._minhash(query_shingles)
            keys = list(self._lsh.query(mh))
        except Exception as exc:
            logger.debug("dedup: LSH query failed: %s", exc)
            return []

        scored: list[tuple[str, float]] = []
        for key in keys:
            candidate_shingles = self._shingles_by_uuid.get(key)
            if not candidate_shingles:
                continue
            score = _jaccard(query_shingles, candidate_shingles)
            if score >= self._lsh_threshold:
                scored.append((key, score))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored

    def _llm_confirm(
        self,
        new_name: str,
        new_summary: str,
        existing_name: str,
        existing_summary: str,
    ) -> str:
        """Return 'yes', 'no', or 'uncertain' from a single Haiku call.

        Phase 4: routes through Graphiti's verbatim node-dedup prompt
        (``ingestion.prompts.dedupe_nodes``) with a single-element
        ``existing_nodes`` list whose ``candidate_id`` is 0. We then map the
        structured response back onto the legacy ``yes/no/uncertain`` API so
        the surrounding LSH top-K loop and the existing tests keep working:

        - ``duplicate_candidate_id == 0``  → ``yes`` (the one candidate matches)
        - ``duplicate_candidate_id == -1`` → ``no``  (Graphiti's "no match")
        - anything else / parse failure    → ``uncertain``

        The legacy stub asked the model "yes/no/uncertain"; the Graphiti
        prompt also explicitly allows the model to return -1 ("no match or
        unsure") — that's our `uncertain` path. The semantic is preserved.
        """
        assert self._llm is not None  # narrowed by find_or_none
        # Both summaries are truncated at 600 chars to match the legacy
        # stub's budget; long auto-generated summaries would otherwise blow
        # past Haiku's context.
        new_summary_trim = (new_summary or "")[:600]
        existing_summary_trim = (existing_summary or "")[:600]

        from .prompts.dedupe_nodes import build_prompt

        messages = build_prompt(
            {
                "previous_episodes": [],
                "episode_content": "",
                "extracted_node": {
                    "name": new_name,
                    "summary": new_summary_trim,
                },
                "entity_type_description": "Entity",
                "existing_nodes": [
                    {
                        "candidate_id": 0,
                        "name": existing_name,
                        "entity_types": ["Entity"],
                        "summary": existing_summary_trim,
                    }
                ],
            }
        )

        response = self._llm.messages.create(
            model=self._llm_model,
            max_tokens=200,
            messages=messages,
            response_format=_NODE_DEDUP_SCHEMA,
        )
        text = str(response.content[0].text).strip()

        # Two parse paths: structured JSON (the new normal) and the legacy
        # yes/no/uncertain plain-text shape (the test fixtures still produce
        # this; we keep the parser tolerant so we don't break them).
        if text.lower().startswith("yes"):
            return "yes"
        if text.lower().startswith("no"):
            return "no"
        try:
            import json as _json

            data = _json.loads(text)
        except Exception:
            return "uncertain"

        try:
            cid = int(data.get("duplicate_candidate_id", -1))
        except (TypeError, ValueError):
            return "uncertain"
        if cid == 0:
            return "yes"
        if cid == -1:
            return "no"
        # Any other id is treated as a parse miss — the only candidate we
        # sent has candidate_id=0, so a non-zero non-(-1) answer is noise.
        return "uncertain"


class _NullLSH:
    """Sentinel for "datasketch unavailable / graph load failed."

    ``find_or_none`` treats this as "no LSH candidates ever" so the
    pipeline degrades gracefully to exact-name-only matching.
    """

    def __contains__(self, key: str) -> bool:  # pragma: no cover - trivial
        return False

    def insert(self, key: str, mh: Any) -> None:  # pragma: no cover - trivial
        return None

    def query(self, mh: Any) -> list[str]:  # pragma: no cover - trivial
        return []


# ---------------------------------------------------------------------------
# Internal collection helper — used by ingestion.extractor to group existing
# candidates by normalized name for ambiguous-match disambiguation.
# ---------------------------------------------------------------------------


def index_by_normalized_name(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Bucket existing-node rows by ``_normalize_name(row['name'])``.

    Used by callers that want to detect the "two candidates share the same
    normalized name" ambiguity case without re-issuing a Cypher query.
    Returns {normalized_name: [uuid, ...]}.
    """
    bucket: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        normalized = _normalize_name(str(row.get("name", "")))
        if not normalized:
            continue
        bucket[normalized].append(str(row.get("uuid", "")))
    return dict(bucket)


__all__ = [
    "NodeDeduper",
    "_jaccard",
    "_name_entropy",
    "_normalize_for_fuzzy",
    "_normalize_name",
    "_shingles",
    "has_high_entropy",
    "index_by_normalized_name",
]
