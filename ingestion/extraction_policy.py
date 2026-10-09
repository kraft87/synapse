from __future__ import annotations

import json
import logging
import os
import re
from typing import TYPE_CHECKING, Any

from ingestion.models import (
    ExtractedEntity,
    ExtractedFact,
)
from ingestion.scope import personal_scope_enabled

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Group routing (technical vs personal)
# ---------------------------------------------------------------------------
#
# Replaces the prior hardcoded `group_id = "technical"`. Entities and edges are
# routed per-item via project tag, then per-entity overridden by content
# heuristics. Patterns mirror scripts/survey_cross_graph_leaks.py — same
# regexes that identified the post-hoc leakers — so the writer-side classifier
# matches the cleanup-side classifier.

# The whole split is off when SYNAPSE_PERSONAL_SCOPE=0 (ingestion/scope.py): every
# item and entity below routes to "technical" and the personal graph is never
# written. Work-only deployments want that: a technical entity tripping the
# personal regex lands in a graph the default search never reads.

# Projects whose extraction items default to the personal graph. Deployment-
# specific slugs (e.g. a project named after the owner) are config, not code:
# SYNAPSE_PERSONAL_PROJECTS adds comma-separated slugs (issue #41).
_PERSONAL_PROJECTS: frozenset[str] = frozenset({"jobs", "personal", "email-templates"}) | frozenset(
    s.strip().lower()
    for s in (os.environ.get("SYNAPSE_PERSONAL_PROJECTS") or "").split(",")
    if s.strip()
)

# Owner-name possessive ("<name>'s ...") routes content to the personal graph.
# Derived from SYNAPSE_OWNER_NAME; skipped for the generic default "User" —
# "user's" is everywhere in technical text and would over-route.
_OWNER = (os.environ.get("SYNAPSE_OWNER_NAME") or "User").strip() or "User"
_OWNER_POSSESSIVE = f"{re.escape(_OWNER.lower())}'?s|" if _OWNER.lower() != "user" else ""

_PERSONAL_NAME_PATTERN = re.compile(
    r"\b("
    + _OWNER_POSSESSIVE
    + r"family|sister|brother|mom|mother|dad|father|parent|cousin|uncle|aunt|nephew|niece|"
    r"girlfriend|boyfriend|ex[- ]girlfriend|ex[- ]boyfriend|wife|husband|partner|spouse|dating|dated|"
    r"friend|friends|cottage|vacation|holiday|birthday|wedding|"
    r"doctor|dentist|appointment|medication|drinking|gym|exercise|sleep|diet|insurance|"
    r"book|novel|movie|film|tv show|chess\.com|"
    r"recruiter|hiring|interview|applied to|job posting|job position|role at|engineer at|"
    r"job opportunity|job offer"
    r")\b",
    re.IGNORECASE,
)

_TECHNICAL_NAME_PATTERN = re.compile(
    r"\b("
    r"python|javascript|typescript|rust|golang|kotlin|java|"
    r"docker|kubernetes|k8s|postgres|postgresql|pgvector|falkordb|redis|"
    r"mongodb|cassandra|elasticsearch|kafka|"
    r"react|django|flask|fastapi|nextjs|svelte|"
    r"aws|azure|gcp|terraform|ansible|helm|"
    r"async|asyncio|threading|semaphore|mutex|coroutine|"
    r"\.py|\.js|\.ts|\.yaml|\.toml|\.sql|\.json"
    r")\b",
    re.IGNORECASE,
)


def _default_group_for_project(project: str | None) -> str:
    """Item-level default group derived from project tag.

    With the personal scope off (SYNAPSE_PERSONAL_SCOPE=0) the project list is
    never consulted: there is one graph, so every item defaults to technical.
    """
    if not personal_scope_enabled():
        return "technical"
    if project and project.lower() in _PERSONAL_PROJECTS:
        return "personal"
    return "technical"


def _classify_entity_group(name: str, summary: str | None, default_group: str) -> str:
    """Per-entity override of the item-level default group.

    A personal-flavored entity name (family, dating, health, job-search) lands
    in personal even when the surrounding session is technical — and vice
    versa for clearly-technical names that surface in personal sessions.
    Borderline cases stay with the default. Matches the cleanup-pass regex
    in scripts/survey_cross_graph_leaks.py so writer and survey agree.

    With the personal scope off (SYNAPSE_PERSONAL_SCOPE=0) neither pattern runs:
    a technical entity that trips the personal regex would otherwise land in a
    graph nothing reads.
    """
    if not personal_scope_enabled():
        return "technical"
    text = f"{name}\n{summary or ''}"
    if _PERSONAL_NAME_PATTERN.search(text):
        return "personal"
    if _TECHNICAL_NAME_PATTERN.search(text):
        return "technical"
    return default_group


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    import math

    dot = sum(x * y for x, y in zip(a, b, strict=True))
    mag_a = math.sqrt(sum(x**2 for x in a))
    mag_b = math.sqrt(sum(x**2 for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


_FILE_PATH_RE = re.compile(r"(?:~/|/)[^\s\"'<>|:,]+\.(?:py|ts|js|go|rs|sql|yaml|yml|toml|json|md)")
_URL_RE = re.compile(r"https?://([^\s\"'<>)]+)")
_ERROR_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:Error|Exception|Violation|Failure))\b")
_KNOWN_TOOLS = frozenset(
    [
        "bash",
        "read",
        "edit",
        "write",
        "grep",
        "find",
        "glob",
        "python",
        "uv",
        "git",
        "docker",
        "psql",
        "ssh",
        "curl",
        "jq",
        "ruff",
        "mypy",
        "pytest",
        "gh",
    ]
)

# Number of semantic (BM25+vector) contradiction/duplicate candidates shown to
# the LLM per new fact in stage 6b. Was 20; each candidate is a full fact line,
# so 20 inflated the dedup/contradiction prompt to ~30K tokens/call — the single
# largest input driver in the pipeline (measured 2026-06-05). Candidates are
# similarity-ranked (RRF of BM25+vector), so a true duplicate/contradiction
# almost always lands in the top handful; the rank 9-20 tail is mostly noise.
# The source modalities each pull 2x this cap so the merge has a real tail to
# choose from. Tune here.
_SEMANTIC_POOL_LIMIT = 8

# Saturation continuation (2026-08-14): a sweeping supersession fact ("the
# entire X stack was torn down") can contradict far more live edges than fit
# in one pool — the 2026-08-08 adult-stack teardown fact invalidated 7 of its
# 8 candidates and left 17 stale facts live because they never ranked top-8
# for ANY of the note's facts. When a round contradicts at least
# _SATURATION_MIN of a fact's pool, retrieval re-runs excluding every
# candidate already judged and another contradiction round fires, until a
# round comes back below the threshold or _SATURATION_MAX_ROUNDS extra
# rounds have run (worst case 8 + 8*8 = 72 invalidations per fact).
#
# Rounds cap sized empirically (2026-08-14): replaying the real teardown fact
# against the reconstructed pre-teardown graph (44 live adult-stack edges),
# the loop needed ~5 continuation rounds to go dry — it invalidated 40/44,
# caught all 17 facts prod had missed, and correctly spared the 4
# product-knowledge facts. A cap of 3 truncated the same replay at 28/44
# with the fact still saturating. 8 covers the worst observed sweep with
# ~2x margin; _SATURATION_MIN remains the real stop condition.
_SATURATION_MIN = 4
#: A BM25-only hit reaches the contradiction judge only when its stored embedding is at
#: least this similar to the new fact (2026-10-09). Vector hits already need >= 0.80
#: (find_similar_edges' distance_threshold); BM25 hits had no floor, and the judge
#: retired facts that merely shared an identifier (a ticket key, a PR number) with the
#: new one: 79% of October's contradiction verdicts came from BM25-only candidates. A
#: shared entity is not enough on its own, since hub entities sit on most facts.
_FULLTEXT_MIN_SIM = 0.70
_SATURATION_MAX_ROUNDS = 8

_CONTRADICTION_PROMPT = """\
You are a knowledge graph deduplication assistant. Decide which existing facts the NEW FACT duplicates and which it contradicts. A single existing fact CAN be both — e.g. "X uses A" supersedes "X uses A (v1)" (duplicate predicate, but the new one updates/replaces).

NEW FACT: "{new_fact}"

EXISTING FACTS share the same source/target entity pair as the new fact — strongest prior for duplicate detection:
{existing_section}

INVALIDATION CANDIDATES were retrieved by hybrid (BM25 + vector) similarity — strongest prior for contradiction/supersession detection:
{candidates_section}

Indices are continuous across both sections. Refer to facts by their `idx`, not by any other identifier.

Rules:
- duplicate_facts: idx values whose information is restated by the NEW FACT (same relationship, same meaning, may differ only by phrasing).
- contradicted_facts: idx values whose claim is incompatible with or superseded by the NEW FACT. This INCLUDES drop-in replacements ("X uses A" → "X uses B") even when the target entity differs.
- A status change of the SAME thing IS a contradiction ("PR 12 is open" → "PR 12 is merged"). NOT a contradiction: a different fact about a related thing (another ticket, PR, file, step or person in the same project or thread), or added detail or later progress that leaves the old claim still true. Sharing an identifier or a topic is not enough.
- An idx MAY appear in BOTH lists when the new fact restates the old predicate while correcting/superseding the value.
- Both lists may be empty if the NEW FACT is genuinely novel.
"""

# Response shape enforced via pydantic-ai structured outputs — see
# ``ingestion.llm_schemas.ResolutionResult``.


def build_resolution_prompt(
    new_fact: str,
    existing_pool: list[dict[str, Any]],
    candidate_pool: list[dict[str, Any]],
) -> tuple[str, dict[int, str]]:
    """Render the dual-section prompt with continuous idx numbering.

    Returns (prompt_text, idx_to_uuid_map). The map lets the caller translate
    the LLM's idx-based response back to edge UUIDs without including the
    UUIDs in the prompt itself — saves tokens and prevents the LLM from
    hallucinating UUID characters.
    """
    idx_to_uuid: dict[int, str] = {}
    existing_lines: list[str] = []
    candidate_lines: list[str] = []

    i = 0
    for item in existing_pool:
        idx_to_uuid[i] = item["uuid"]
        existing_lines.append(f"[{i}] {item['fact']}")
        i += 1
    for item in candidate_pool:
        idx_to_uuid[i] = item["uuid"]
        candidate_lines.append(f"[{i}] {item['fact']}")
        i += 1

    existing_section = "\n".join(existing_lines) if existing_lines else "(none)"
    candidates_section = "\n".join(candidate_lines) if candidate_lines else "(none)"
    prompt = _CONTRADICTION_PROMPT.format(
        new_fact=new_fact,
        existing_section=existing_section,
        candidates_section=candidates_section,
    )
    return prompt, idx_to_uuid


_BATCH_CONTRADICTION_PROMPT = """\
You are a knowledge graph deduplication assistant. For EACH new fact below, decide which of ITS OWN existing/candidate facts it duplicates and which it contradicts. Each new fact has a unique `id` and its OWN indexed candidate lists. An idx ONLY refers to facts under the same fact `id`.

<NEW FACTS WITH CANDIDATES>
{items}
</NEW FACTS WITH CANDIDATES>

For each new fact:
- existing_facts share the same source/target entity pair (strongest duplicate prior).
- invalidation_candidates were retrieved by hybrid (BM25 + vector) similarity (strongest contradiction/supersession prior).
- idx values are continuous within ONE fact's lists; never mix idx across facts.

Rules (apply per-fact):
- duplicate_facts: idx of facts the NEW FACT restates (same relationship, same meaning, may differ only by phrasing).
- contradicted_facts: idx of facts the NEW FACT supersedes or is incompatible with. This INCLUDES drop-in replacements ("X uses A" -> "X uses B") even when the target differs.
- A status change of the SAME thing IS a contradiction ("PR 12 is open" -> "PR 12 is merged"). NOT a contradiction: a different fact about a related thing (another ticket, PR, file, step or person in the same project or thread), or added detail or later progress that leaves the old claim still true. Sharing an identifier or a topic is not enough.
- An idx MAY appear in both lists when the new fact restates a predicate while correcting/superseding the value.
- Both lists may be empty if the NEW FACT is genuinely novel.

Return exactly one result object per new fact `id`.
"""

# Response shape enforced via pydantic-ai structured outputs — see
# ``ingestion.llm_schemas.BatchResolutionResult``.


def build_batch_resolution_prompt(
    items: list[dict[str, Any]],
) -> tuple[str, dict[int, dict[int, str]]]:
    """Render a BATCHED stage-6b prompt: many new facts in one LLM call.

    Each entry in ``items`` is one new fact with its OWN candidate pools::

        {"id": 0, "new_fact": "...",
         "existing_pool": [{"uuid": "...", "fact": "..."}, ...],
         "candidate_pool": [{"uuid": "...", "fact": "..."}, ...]}

    Idx values are scoped PER fact (fact 0's idx 0 != fact 1's idx 0). The
    LLM is told this explicitly so it never references across facts.

    Returns (prompt_text, idx_map) where ``idx_map[fact_id][idx] -> uuid``.
    Mirrors the build_batch_prompt pattern from dedupe_nodes.py (PR #83).
    """
    per_item_maps: dict[int, dict[int, str]] = {}
    rendered: list[dict[str, Any]] = []

    for item in items:
        fid = int(item["id"])
        idx_to_uuid: dict[int, str] = {}
        existing_lines: list[dict[str, Any]] = []
        candidate_lines: list[dict[str, Any]] = []
        i = 0
        for cand in item.get("existing_pool", []):
            idx_to_uuid[i] = cand["uuid"]
            existing_lines.append({"idx": i, "fact": cand["fact"]})
            i += 1
        for cand in item.get("candidate_pool", []):
            idx_to_uuid[i] = cand["uuid"]
            candidate_lines.append({"idx": i, "fact": cand["fact"]})
            i += 1
        per_item_maps[fid] = idx_to_uuid
        rendered.append(
            {
                "id": fid,
                "new_fact": item["new_fact"],
                "existing_facts": existing_lines,
                "invalidation_candidates": candidate_lines,
            }
        )

    prompt = _BATCH_CONTRADICTION_PROMPT.format(items=json.dumps(rendered, ensure_ascii=False))
    return prompt, per_item_maps


def dedupe_pools(
    pair_pool: list[dict[str, Any]],
    semantic_pool: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove any uuid that already appears in pair_pool from semantic_pool.

    The pair-pool prior is strictly stronger (same source/target entities), so
    when an edge clears both checks we keep it in the pair pool and drop it
    from the semantic pool. Prevents the LLM from seeing the same fact under
    two labels and second-guessing itself.
    """
    pair_uuids = {item["uuid"] for item in pair_pool}
    filtered_semantic = [item for item in semantic_pool if item["uuid"] not in pair_uuids]
    return pair_pool, filtered_semantic


# ---------------------------------------------------------------------------
# Stage 2 — Deterministic extractor
# ---------------------------------------------------------------------------


def _apply_canonical_aliases(
    entities: list[ExtractedEntity], facts: list[ExtractedFact]
) -> list[ExtractedEntity]:
    """Rewrite known identity aliases to their canonical entity (task #49).

    Runs BEFORE resolution so 'User' / full-name mentions land on the
    existing owner hub (SYNAPSE_OWNER_NAME) via the exact-name short-circuit instead of minting a
    fresh node that neither LSH (Jaccard too low) nor vector similarity
    (alias spellings sit ~0.5 apart) would ever re-merge. Facts are re-pointed in
    place; entities are renamed and collapsed (longest summary wins) when the
    rewrite makes two extracted entities the same name. Facts that become
    self-loops after the rewrite (e.g. an extracted 'User asked <owner> ...')
    are dropped — a hub->hub edge carries no relational signal.

    Returns the (possibly smaller) entity list; mutates entities/facts in place.
    """
    from ingestion.dedup import canonical_name

    renamed = False
    for e in entities:
        canon = canonical_name(e.name)
        if canon is not None and e.name != canon:
            e.name = canon
            renamed = True
    for f in facts:
        for attr in ("source", "target"):
            canon = canonical_name(getattr(f, attr))
            if canon is not None:
                setattr(f, attr, canon)
    self_loops = [f for f in facts if f.source == f.target]
    for f in self_loops:
        logger.debug("Dropping self-loop fact after alias rewrite: %s", f.fact[:120])
        facts.remove(f)
    if not renamed:
        return entities
    by_name: dict[str, ExtractedEntity] = {}
    for e in entities:
        prev = by_name.get(e.name)
        if prev is None or len(e.summary) > len(prev.summary):
            by_name[e.name] = e
    return list(by_name.values())
