"""
Phase 2b extraction pipeline — extracts entities and facts from session summaries
and writes to the Postgres KG (kg_entities / kg_relationships).

Pipeline stages:
  Stage 2 — DeterministicExtractor: regex + metadata → entities (no LLM)
  Stage 3 — LLMExtractor: session summary → residual entities + facts
  Stage 4 — EntityResolver: dedup new entities against existing KG nodes
  Stage 5 — write_nodes: upsert entities to the KG
  Stage 6a — embedding_filter: vector + structural candidate search (no LLM)
  Stage 6b — llm_confirm: LLM decides duplicate/contradiction/novel for candidates
  Stage 7 — write_edges: create RELATES_TO edges (skip duplicates, invalidate contradictions)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import orjson
from pydantic import ValidationError

from ingestion.kg_client import rrf_merge
from ingestion.llm_client import MalformedResponseError, parse_with_retry
from ingestion.models import (
    CombinedExtraction,
    ExtractedEntity,
    ExtractedFact,
    ExtractionResult,
)

if TYPE_CHECKING:
    from ingestion.embedding import EmbeddingModel

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
    """Item-level default group derived from project tag."""
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
    """
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
- An idx MAY appear in BOTH lists when the new fact restates the old predicate while correcting/superseding the value.
- Both lists may be empty if the NEW FACT is genuinely novel.
"""

_CONTRADICTION_SCHEMA: dict[str, Any] = {
    "type": "json",
    "schema": {
        "type": "object",
        "properties": {
            "duplicate_facts": {"type": "array", "items": {"type": "integer"}},
            "contradicted_facts": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["duplicate_facts", "contradicted_facts"],
        "additionalProperties": False,
    },
}


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
- An idx MAY appear in both lists when the new fact restates a predicate while correcting/superseding the value.
- Both lists may be empty if the NEW FACT is genuinely novel.

Return exactly one result object per new fact `id`.
"""

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
                        "duplicate_facts": {"type": "array", "items": {"type": "integer"}},
                        "contradicted_facts": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["id", "duplicate_facts", "contradicted_facts"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    },
}


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
# Stage 6 gray-zone gate (issue #14) — embedding-similarity triage BEFORE the
# stage-6b LLM confirm. Most dedup/contradiction candidate pairs aren't close
# calls: similarity >= HIGH is an auto-merge, <= LOW an auto-new; only the gray
# zone needs the LLM (the fattest per-item cost, 1-2 of the 3-6 round-trips).
# Modes: 'shadow' (default — log would-be decisions vs actual LLM verdicts to
# dedup_gate_shadow, change NOTHING) -> pick thresholds empirically -> 'enforce'.
# ---------------------------------------------------------------------------


def _dedup_gate_mode() -> str:
    """SYNAPSE_DEDUP_GATE: 'shadow' (default) | 'enforce' | 'off'."""
    mode = os.environ.get("SYNAPSE_DEDUP_GATE", "shadow").strip().lower()
    return mode if mode in ("shadow", "enforce", "off") else "shadow"


def _dedup_gate_thresholds() -> tuple[float, float]:
    """(high, low) similarity bounds; env-tunable so enforcement can adopt the
    thresholds the shadow window picks without a code change."""
    try:
        return (
            float(os.environ.get("SYNAPSE_DEDUP_GATE_HIGH", "0.95")),
            float(os.environ.get("SYNAPSE_DEDUP_GATE_LOW", "0.70")),
        )
    except ValueError:
        return 0.95, 0.70


def _gate_decisions(
    pair_pool: list[dict[str, Any]],
    semantic_pool: list[dict[str, Any]],
    high: float,
    low: float,
) -> list[tuple[dict[str, Any], str, float | None, str]]:
    """Per-candidate would-be decision from embedding similarity alone.

    Returns (candidate, pool_name, sim, decision) tuples; decision is 'merge'
    (sim >= high), 'new' (sim <= low) or 'gray'. A candidate with no ``_sim``
    (BM25-only hit — no embedding signal) is always 'gray': the gate must never
    silently drop a candidate it has no evidence about.
    """
    out: list[tuple[dict[str, Any], str, float | None, str]] = []
    for pool_name, pool in (("pair", pair_pool), ("semantic", semantic_pool)):
        for cand in pool:
            sim = cand.get("_sim")
            if sim is None:
                decision = "gray"
            elif sim >= high:
                decision = "merge"
            elif sim <= low:
                decision = "new"
            else:
                decision = "gray"
            out.append((cand, pool_name, sim, decision))
    return out


def _apply_gate_enforce(
    gate_info: dict[int, list[tuple[dict[str, Any], str, float | None, str]]],
) -> tuple[
    dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]]]],
    set[int],
    dict[int, list[str]],
]:
    """Enforcement: shrink the LLM confirm to the gray zone.

    A fact with any 'merge' candidate is resolved without the LLM: pre-skipped
    as a duplicate, reinforcing every merge-zone match (the assert-count bump
    still fires — Stage 7 consumes the reinforce map exactly as for an LLM-
    confirmed duplicate, per PR #151). 'new' candidates are dropped from the
    pools; a fact left with only gray candidates goes to the LLM as usual, and
    one left with none skips the confirm entirely (auto-new).
    """
    gray_map: dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    pre_skip: set[int] = set()
    pre_reinforce: dict[int, list[str]] = {}
    for idx, decisions in gate_info.items():
        merged = [c["uuid"] for c, _pool, _sim, d in decisions if d == "merge"]
        if merged:
            pre_skip.add(idx)
            pre_reinforce[idx] = merged
            continue
        gray_pair = [c for c, pool, _sim, d in decisions if d == "gray" and pool == "pair"]
        gray_sem = [c for c, pool, _sim, d in decisions if d == "gray" and pool == "semantic"]
        if gray_pair or gray_sem:
            gray_map[idx] = (gray_pair, gray_sem)
    return gray_map, pre_skip, pre_reinforce


def _gate_shadow_rows(
    facts: list[ExtractedFact],
    gate_info: dict[int, list[tuple[dict[str, Any], str, float | None, str]]],
    llm_map: dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]]]],
    group_id: str,
    invalidate: dict[int, list[str]],
    reinforce: dict[int, list[str]],
    llm_ok: bool,
) -> list[tuple[Any, ...]]:
    """dedup_gate_shadow rows: the gate's would-be decision beside the LLM's
    actual verdict, one row per (fact, candidate). Verdict columns are NULL
    unless the LLM batch succeeded AND this candidate was in the pools it saw
    (llm_ran) — failed batches and enforcement-dropped candidates must not
    pollute the threshold analysis as false "LLM said no" rows."""
    rows: list[tuple[Any, ...]] = []
    for idx, decisions in gate_info.items():
        sent = llm_map.get(idx)
        sent_uuids = {c["uuid"] for pool in sent for c in pool} if sent else set()
        dup = set(reinforce.get(idx, []))
        contra = set(invalidate.get(idx, []))
        for cand, pool_name, sim, decision in decisions:
            cand_uuid = cand.get("uuid")
            if not cand_uuid:
                continue
            ran = bool(llm_ok and cand_uuid in sent_uuids)
            rows.append(
                (
                    group_id,
                    facts[idx].fact[:500],
                    cand_uuid,
                    (cand.get("fact") or "")[:500],
                    pool_name,
                    round(sim, 4) if sim is not None else None,
                    decision,
                    (cand_uuid in dup) if ran else None,
                    (cand_uuid in contra) if ran else None,
                    ran,
                )
            )
    return rows


# ---------------------------------------------------------------------------
# Stage 2 — Deterministic extractor
# ---------------------------------------------------------------------------


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

_EXTRACTION_PROMPT = """\
These facts are the ONLY memory the system keeps from this conversation — anything you \
do not capture here is forgotten, so extract completely. The user is the authoritative \
source about their own work, life, and preferences: treat a user assertion as a fact even \
when phrased casually ("I'll just go with Postgres" -> a decision), but a user QUESTION is \
NOT an assertion — never mint a fact from something the user only asked about.

Session date: {session_date}. Resolve every dated mention in the facts against it.

Given the session summary and pre-identified entities below, extract any additional \
entities and the relationships between all entities.

Pre-identified entities:
{context_entities}

Session summary:
{summary}

Output ONLY valid JSON (no explanation, no markdown fence):
{{
  "entities": [{{"name": "...", "type": "...", "summary": "..."}}],
  "facts": [{{"source": "...", "target": "...", "relationship": "...", "fact": "..."}}]
}}

Rules:
- Entity types are open-ended — use whatever fits (Tool, Project, Decision, Issue, Config, etc.)
- fact text must be phrased as a self-contained searchable statement ("X uses Y for Z")
- Include rationale for decisions ("chose X because Y")
- Extract every distinct, durable, self-contained fact the summary genuinely
  contains — there is NO fixed number. Do not pad to reach a count, and do not omit a
  substantive fact to stay small. Let the content set the count: a thin turn yields a
  few, a dense one yields many.
- NEVER record transient or operational actions as facts. A fact must be reusable
  knowledge that stays true beyond this session — a decision and its rationale, a
  config value, a root cause, a relationship between components. BAD (never emit):
  "the inbox was checked and found empty", "discord-send.sh was used to post the
  briefing", "the calendar was clear", "X was run". GOOD: "the briefing pipeline posts
  to Discord #briefings via discord-send.sh", "Voyage embeddings are 2048-dim".
- PRESERVE EXACT QUANTITIES. A fact carrying a number, price, percentage, count, or
  duration MUST keep the figure VERBATIM in the fact text ("Women hold 20% of
  leadership positions at User's company", "User spent $120 on a helmet"). Never
  paraphrase a figure and never convert it — "20%" must NOT become "20 people" or
  "20 women". If one statement carries several distinct quantities, emit one fact
  per figure rather than folding them together.
- NAME WHAT CHANGED. When a fact records a switch, upgrade, migration, or a replaced
  choice/config/state, name what it REPLACED in the fact text — "switched the embedder to
  voyage-4-large, replacing voyage-3", not "uses voyage-4-large". Same for moves and
  cancellations ("moved the prod compose to /opt/docker/synapse, no longer at
  ~/synapse"). Naming the superseded value is what lets a later reader tell
  current state from stale.
- ANCHOR EVERY DATE. When a fact's text carries a dated reference, append its resolved
  absolute date in parentheses right after the original wording. Resolve an ABSOLUTE
  mention against the session date's year unless the text states one ("my flight is
  January 31" -> "... January 31 (meaning 2026-01-31)"); resolve a RELATIVE mention against
  the session date ("two days from now", "last Tuesday" -> that calendar date, e.g.
  "(meaning 2026-01-17)"). Keep the user's original wording and add "(meaning YYYY-MM-DD)"
  after it. If the session date is unknown or the reference is too vague to pin
  ("recently", "a while ago"), leave it unanchored. When one statement carries SEVERAL
  dated events, SPLIT it into one fact per event so each fact carries exactly one date.

ENTITY/FACT CONSISTENCY (strict — orphan entities are dropped on the server):
- Every entity you list in `entities` MUST appear as the `source` or `target` of at
  least one fact. Every fact's `source` and `target` MUST exactly match the `name`
  of an entity declared in `entities`. If you can't write a meaningful fact about
  an entity, don't list it.
"""


_WEB_EXTRACTION_PROMPT = """\
The text below is an excerpt from {source_kind}. Extract entities and the \
relationships between them.

Source: {source_desc}
Pre-identified entities:
{context_entities}

Excerpt:
{summary}

Output ONLY valid JSON (no explanation, no markdown fence):
{{
  "entities": [{{"name": "...", "type": "...", "summary": "..."}}],
  "facts": [{{"source": "...", "target": "...", "relationship": "...", "fact": "..."}}]
}}

Rules:
- ATTRIBUTION FIREWALL (strict): this is third-party content. NEVER phrase a fact
  as something the user said, did, decided, owns, or prefers. The source
  makes claims; the user does not appear in them. BAD: "the user uses Trafilatura",
  "the chosen approach is X". GOOD: "Trafilatura achieves F1 0.93 on boilerplate
  removal benchmarks", "Mem0 removed its graph backend in its v3 rewrite".
- Entity types: use ONLY these — Person, Organization, Product, Technology,
  Technique, Benchmark, Publication, Event, Location. Pick the closest fit;
  do not invent new types.
- SALIENCE BAR: extract only entities that would plausibly matter beyond this one
  page — named tools, products, techniques, organizations, people, measured results.
  Skip page furniture (bylines, categories, related-article titles), code variable
  names, and generic concepts ("performance", "users", "the team").
- fact text must be a self-contained searchable claim ("X achieves Y on Z",
  "X replaced Y because Z"). Include numbers, versions, and dates verbatim when
  the source states them.
- Extract every distinct, durable claim the excerpt genuinely contains — no fixed
  count. A navigation-heavy excerpt may yield zero; return empty lists rather than
  padding.
- NEVER record transient page events ("the article lists 10 tools", "the post was
  updated") — only reusable knowledge claims.

ENTITY/FACT CONSISTENCY (strict — orphan entities are dropped on the server):
- Every entity in `entities` MUST appear as the `source` or `target` of at least
  one fact, and every fact's `source`/`target` MUST exactly match a declared
  entity name.
"""


# Phase 5: the previous ``_MalformedExtractionResponse`` shim has been
# consolidated into ``ingestion.llm_client.MalformedResponseError`` — same
# semantics (carries the raw response text), but reused by every caller
# that opts into ``parse_with_retry``. Kept as a module-local alias so
# existing test imports continue to work.
_MalformedExtractionResponse = MalformedResponseError


class LLMExtractor:
    """Extract entities + facts from session summaries via LLM call."""

    # Up to 2 retries (3 total attempts) on malformed JSON, matching the spec.
    _MAX_ATTEMPTS = 3

    def __init__(self, llm_client: Any, model: str = "claude-haiku-4-5") -> None:
        self._client = llm_client
        self._model = model

    def extract(
        self,
        summary: str,
        context_entities: list[ExtractedEntity],
        session_date: str | None = None,
    ) -> ExtractionResult:
        """Run the extractor LLM, then validate + cross-ref the response.

        ``session_date`` (YYYY-MM-DD, the segment's conversation date) is the
        anchor the prompt resolves in-text date mentions against; None renders
        "unknown" and the prompt leaves relative dates unanchored.

        Phase 5 consolidation: the tenacity ``@retry`` wrapper that used to
        live inline here is gone. Two retry concerns now compose at separate
        layers:

        * Transient wire errors (rate-limit, timeout) → handled by tenacity
          inside ``_MessagesProxy.create`` (see ``ingestion.llm_client``).
        * Malformed JSON / Pydantic validation errors → handled by
          ``parse_with_retry``'s feedback loop.

        The load-bearing Phase 1 logic — the Pydantic ``CombinedExtraction``
        cross-reference validator that drops facts referencing undeclared
        entities — survives untouched in ``_parse_response`` below.
        """
        context_str = (
            ", ".join(f"{e.name} ({e.type})" for e in context_entities)
            if context_entities
            else "none"
        )
        base_prompt = _EXTRACTION_PROMPT.format(
            context_entities=context_str,
            summary=summary,
            session_date=session_date or "unknown",
        )
        return self._run(base_prompt)

    def extract_web(
        self,
        content: str,
        context_entities: list[ExtractedEntity],
        provenance: dict[str, Any],
    ) -> ExtractionResult:
        """Extract from third-party web content (task #68).

        Same parse/retry machinery as ``extract``, different prompt: the web
        variant carries the attribution firewall (claims belong to the source,
        never to the user), a closed entity-type vocabulary, and a salience
        bar — third-party pages are noisier and less trustworthy substrate
        than the user's own conversation chunks.
        """
        context_str = (
            ", ".join(f"{e.name} ({e.type})" for e in context_entities)
            if context_entities
            else "none"
        )
        synthesized = bool(provenance.get("synthesized"))
        if provenance.get("kind") == "research_brief":
            source_kind = "a multi-source research brief compiled by an AI assistant"
        elif synthesized:
            source_kind = "an AI-generated answer about a web page (secondhand, not the raw page)"
        else:
            source_kind = "a scraped web page"
        source_desc = " | ".join(
            str(v)
            for v in (
                provenance.get("title"),
                provenance.get("url"),
                provenance.get("published_at") or provenance.get("fetched_at"),
            )
            if v
        )
        base_prompt = _WEB_EXTRACTION_PROMPT.format(
            source_kind=source_kind,
            source_desc=source_desc or "unknown",
            context_entities=context_str,
            summary=content,
        )
        return self._run(base_prompt)

    def _run(self, base_prompt: str) -> ExtractionResult:
        try:
            combined = parse_with_retry(
                self._client,
                base_prompt=base_prompt,
                parser=self._parse_response,
                model=self._model,
                # No fixed fact cap (content sets the count) + the anti-filler
                # quality bar; dense turns can run 15-25 facts. Dense infra
                # chunks proved 3072 too small — outputs truncated mid-array
                # at ~12K chars and failed all parse retries (2026-07-17), so
                # the headroom is 8192; a truncated tail silently drops the
                # last facts.
                max_tokens=8192,
                max_attempts=self._MAX_ATTEMPTS,
            )
        except MalformedResponseError as exc:
            logger.warning(
                "LLM extraction failed after %d attempts (malformed JSON): %s",
                self._MAX_ATTEMPTS,
                exc,
            )
            return ExtractionResult()

        if combined.dropped_facts:
            logger.info(
                "Dropped %d fact(s) referencing entities not in extractor output",
                len(combined.dropped_facts),
            )
        if combined.dropped_entities:
            logger.debug(
                "Dropped %d entity row(s) with empty name during validation",
                len(combined.dropped_entities),
            )

        return ExtractionResult(entities=combined.entities, facts=combined.facts)

    @staticmethod
    def _parse_response(raw: str) -> CombinedExtraction:
        """Parse the LLM response into a validated ``CombinedExtraction``.

        Raises ``MalformedResponseError`` when the response isn't valid
        JSON or doesn't satisfy the schema; ``parse_with_retry`` catches
        that exception and re-fires the call with the failure quoted back
        as feedback.
        """
        stripped = raw.strip()
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
        if fence_match:
            stripped = fence_match.group(1)
        else:
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start == -1 or end == -1:
                raise MalformedResponseError("no JSON object found in response", raw_response=raw)
            stripped = stripped[start : end + 1]

        try:
            data = orjson.loads(stripped)
        except (orjson.JSONDecodeError, ValueError) as exc:
            raise MalformedResponseError(f"JSON decode error: {exc}", raw_response=raw) from exc

        if not isinstance(data, dict):
            raise MalformedResponseError("top-level JSON value must be an object", raw_response=raw)

        # Coerce each entity/fact row independently — bad rows are skipped, not
        # fatal — then hand the cleaned lists to the Pydantic model so the
        # cross-ref validator can drop facts pointing at undeclared entities.
        entities: list[ExtractedEntity] = []
        for item in data.get("entities", []) or []:
            if not isinstance(item, dict) or "name" not in item:
                continue
            entities.append(
                ExtractedEntity(
                    name=item["name"],
                    type=item.get("type", "Topic"),
                    summary=item.get("summary", ""),
                )
            )

        facts: list[ExtractedFact] = []
        for item in data.get("facts", []) or []:
            if not isinstance(item, dict):
                continue
            if not all(k in item for k in ("source", "target", "relationship", "fact")):
                continue
            facts.append(
                ExtractedFact(
                    source=item["source"],
                    target=item["target"],
                    relationship=item["relationship"],
                    fact=item["fact"],
                )
            )

        try:
            return CombinedExtraction(entities=entities, facts=facts)
        except ValidationError as exc:
            raise MalformedResponseError(
                f"schema validation error: {exc}", raw_response=raw
            ) from exc


# ---------------------------------------------------------------------------
# Stage 4 — Entity resolver
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

        from ingestion.prompts.dedupe_nodes import (
            BATCH_NODE_DEDUP_SCHEMA,
            build_batch_prompt,
        )

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
            response = self._llm.messages.create(
                model=self._confirm_model,
                max_tokens=max_tokens,
                messages=build_batch_prompt(items),
                response_format=BATCH_NODE_DEDUP_SCHEMA,
            )
            data = json.loads(str(response.content[0].text))
            results = data.get("results", []) if isinstance(data, dict) else []
        except Exception as e:
            logger.warning(
                "batch dedup confirm failed for %d entit%s (%s); treating all as distinct",
                len(pending),
                "y" if len(pending) == 1 else "ies",
                str(e)[:120],
            )
            return {}

        decided: dict[str, str] = {}
        for r in results:
            try:
                idx = int(r["id"])
                cid = int(r["duplicate_candidate_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (0 <= idx < len(pending)):
                continue
            entity, cands = pending[idx]
            if 0 <= cid < len(cands):
                decided[entity.name] = cands[cid]["uuid"]
        return decided


# ---------------------------------------------------------------------------
# ExtractionPipeline — orchestrates stages 2-7
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


class ExtractionPipeline:
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
        # Per-group NodeDeduper cache — the fuzzy-name LSH inside it is an
        # O(all entities) MinHash build, far too expensive to redo per item.
        # See _deduper_for for the refresh policy.
        self._dedupers: dict[str, Any] = {}
        self._dedupers_built_at: dict[str, float] = {}
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

            # When dedup matched an EXISTING entity, prefer the longer
            # summary so the more-detailed text survives. Without this
            # the freshly-extracted (often shorter) summary would
            # overwrite the canonical one. ``merge_summary`` is the same
            # rule used by the nightly dream pipeline.
            summary_to_write = entity.summary
            if not is_new:
                existing_summary = ""
                if deduper is not None:
                    existing_summary = deduper._summary_by_uuid.get(clean_uuid, "")
                summary_to_write = NodeDeduper.merge_summary(existing_summary, entity.summary)

            # Auto-type: roll the extracted subtype up to a canonical supertype via the
            # taxonomy map (already loaded on the deduper). Unknown subtype -> 'other'
            # (queryable as the to-map backlog); no map -> None (backfill fills later).
            supertype = None
            if deduper is not None and deduper._type_map:
                supertype = deduper._type_map.get(entity.type, "other")
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
        prompt, idx_to_uuid = build_resolution_prompt(fact.fact, pair_pool, semantic_pool)
        if not idx_to_uuid:
            return False, []
        try:
            # Triple classification on short fact pairs — Haiku is sufficient
            # and ~10x cheaper than Sonnet.
            response = self._llm._client.messages.create(
                model=self._contradiction_model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
                response_format=_CONTRADICTION_SCHEMA,
            )
            raw = response.content[0].text.strip()
            start = raw.find("{")
            if start < 0:
                return False, []
            data, _ = json.JSONDecoder().raw_decode(raw[start:])
            dup_idx = [i for i in data.get("duplicate_facts", []) if i in idx_to_uuid]
            contradicted_idx = [i for i in data.get("contradicted_facts", []) if i in idx_to_uuid]
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
        prompt, per_item_maps = build_batch_resolution_prompt(items)

        try:
            response = self._llm._client.messages.create(
                model=self._contradiction_model,
                max_tokens=300 * max(1, len(items)),
                messages=[{"role": "user", "content": prompt}],
                response_format=_BATCH_CONTRADICTION_SCHEMA,
            )
            raw = response.content[0].text.strip()
            start = raw.find("{")
            if start < 0:
                return set(), {}, {}, False
            data, _ = json.JSONDecoder().raw_decode(raw[start:])
        except Exception:
            return set(), {}, {}, False

        skip_indices: set[int] = set()
        invalidate: dict[int, list[str]] = {}
        reinforce: dict[int, list[str]] = {}
        for r in data.get("results", []):
            fid = r.get("id")
            if not isinstance(fid, int) or fid not in per_item_maps:
                continue
            idx_to_uuid = per_item_maps[fid]
            dup_idx = [i for i in r.get("duplicate_facts", []) if i in idx_to_uuid]
            contradicted_idx = [i for i in r.get("contradicted_facts", []) if i in idx_to_uuid]
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

    def _deduper_for(self, group_id: str) -> Any:
        """Cached per-group NodeDeduper, rebuilt only when the TTL lapses.

        Constructing a NodeDeduper per item rebuilt its LSH index — an
        O(all entities) MinHash pass — every time, which came to dominate
        item latency once extraction reliably produced entities (~24 min/item
        at 44K entities, 2026-07-17). One deduper per worker keeps the index
        warm; ``register`` folds in this worker's own writes as they happen.
        The TTL bounds staleness from OTHER workers' writes — and only for
        the fuzzy-name assist, since the exact-name short-circuit and the
        embedding-similarity candidates always query the live DB.
        """
        from ingestion.dedup import NodeDeduper

        ttl = float(os.environ.get("SYNAPSE_DEDUP_CACHE_TTL_SECONDS", "900"))
        now = time.monotonic()
        built = self._dedupers_built_at.get(group_id)
        if group_id not in self._dedupers or built is None or now - built > ttl:
            self._dedupers[group_id] = NodeDeduper(
                kg_client=self._kg,
                group_id=group_id,
                llm_client=self._llm._client,
            )
            self._dedupers_built_at[group_id] = now
        return self._dedupers[group_id]

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def process_item(self, item: dict[str, Any]) -> None:
        """Process one extraction queue item. Raises on failure (caller handles retry).

        Group routing: each entity is assigned to either the technical or personal
        graph via _classify_entity_group (project tag → item default, then per-entity
        regex override). Entities are partitioned by group and resolved/written
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
        for grp in ("technical", "personal"):
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
        facts_by_group: dict[str, list[ExtractedFact]] = {"technical": [], "personal": []}
        cross_group_dropped = 0
        for fact in llm_result.facts:
            src_grp = entity_groups.get(fact.source)
            tgt_grp = entity_groups.get(fact.target)
            if src_grp and tgt_grp and src_grp == tgt_grp:
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
        for grp in ("technical", "personal"):
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
