from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


# Field-length caps for LLM-extracted free-form strings. A structural backstop
# against meta-reasoning or schema-description text bleeding into a field the
# prompt could not fully guard: models have been observed dumping multi-KB
# deliberation into "summary"-shaped fields. Caps sit far above anything a
# legitimate extraction produces (a dense date-anchored fact runs ~400 chars),
# so a breach means bleed, not a long-but-real value.
MAX_ENTITY_NAME_LEN = 200
MAX_RELATIONSHIP_LEN = 100
MAX_ENTITY_SUMMARY_LEN = 1200
MAX_FACT_LEN = 2000


# --- Banned entity names (structural backstop for the ENTITY NAMES prompt rules) ---
# The prompt already forbids two name shapes, and the model still mints them a few
# times per hundred entities. Both produce nodes nothing can ever retrieve:
#
#   figure names       "$10,000 premium", "210 lbs deadlift", "6 PM", "7 business days"
#                      — a figure is a PROPERTY of the thing it measures; it belongs in
#                        the fact text, not in a node.
#   abstraction names  "hiring decision", "exit strategy", "fit question" — proposition
#                      stand-ins minted to receive an edge, with no referent in the
#                      user's world.
#
# Ported from the 2026-08-07 graph-reification pass (~/workspace/reify/build_plan.py),
# where it gated every reparent endpoint; the audit of that run is in
# backup_dehub.reify_ops. Two deliberate escapes keep real names alive:
#   * a leading 4-digit year is not a figure — "2025 World Series", "2021 Canada
#     Benefits Summary PDF", and a leading digit that is part of a word ("1Password GRC")
#     all survive.
#   * abstraction is judged on the HEAD noun (the LAST word), so "Statement of Defence"
#     and "Certificate of Insurance" survive while a bare "statement" does not.
#
# SYNAPSE_BANNED_NAME_FILTER=0 disables the drop for instant rollback (and lets the
# prompt A/B harness run a filter-free control arm against the same code).
_UNIT = (
    r"%|am|pm|lbs?|kgs?|kms?|mb|gb|tb|hours?|hrs?|min(?:ute)?s?|days?|weeks?|months?|"
    r"years?|business\s+days?|cases|applicants|reps?|pipelines?|candidates?|"
    r"place|st|nd|rd|th"
)
# (?!\w) rather than the port's \b: "%" is not a word char, so "\b" never matched a
# name ENDING in a percent sign and the bare "20%" the prompt explicitly bans sailed
# through. (?!\w) is satisfied at end-of-string and before punctuation/space, and
# leaves every survivor case intact ("1Password GRC", "3M respirator", "2nd place").
_FIGURE = re.compile(rf"^\s*(?:[\$€£]|\d[\d,.\-+/]*\s*(?:{_UNIT})(?!\w))", re.I)
_YEAR_LEAD = re.compile(r"^\s*(?:19|20)\d\d\b")
_ABSTRACT_HEAD = frozenset(
    {
        "decision",
        "decisions",
        "plan",
        "plans",
        "approach",
        "approaches",
        "issue",
        "issues",
        "question",
        "questions",
        "request",
        "requests",
        "strategy",
        "strategies",
        "statement",
        "statements",
        "claim",
        "claims",
        "preference",
        "preferences",
        "discussion",
        "discussions",
        "idea",
        "ideas",
        "proposal",
        "proposals",
        "concern",
        "concerns",
        "expectation",
        "expectations",
        "conclusion",
        "assumption",
        "assumptions",
    }
)


def banned_name(name: str) -> str | None:
    """Return a short reason if ``name`` is a policy-banned entity name, else None.

    Pure and side-effect free — the caller decides what to do with the verdict.
    """
    s = (name or "").strip()
    if not s:
        return "empty"
    if not _YEAR_LEAD.match(s) and _FIGURE.match(s):
        return "figure-name"
    head = re.sub(r"[^a-z]", "", s.split()[-1].lower())
    if head in _ABSTRACT_HEAD:
        return "abstraction-name"
    return None


def _name_filter_on() -> bool:
    """Read the rollback flag per call, not at import.

    The A/B harness flips it between arms inside one process; a module-level
    constant would freeze whichever value happened to be set at import time.
    """
    return os.getenv("SYNAPSE_BANNED_NAME_FILTER", "1") not in ("0", "false", "")


def _normalize_entity_name(name: str) -> str:
    """Case-insensitive, whitespace-collapsed normalization for cross-ref matching.

    Mirrors Graphiti's `_normalize_string_exact` (dedup_helpers.py:39-42) so a
    fact written as ``source="James"`` and an entity declared as ``name="james"``
    or ``name="James "`` still cross-reference cleanly.
    """
    return " ".join((name or "").lower().split())


class Episode(BaseModel):
    session_id: str
    sequence: int
    project: str | None = None
    platform: str | None = None  # claude_code | cursor | claude_ai
    model: str | None = None
    human_turn: str | None = None
    assistant_turn: str | None = None
    content: str  # concatenated human+assistant for search
    span_id: str | None = None  # Logfire span_id for deduplication
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str | None = None
    created_at: datetime | None = None


class SessionSummary(BaseModel):
    session_id: str
    project: str | None = None
    platform: str | None = None
    summary: str
    last_summarized_sequence: int


class LogfireSpan(BaseModel):
    span_id: str
    trace_id: str
    message: str
    model: str | None = None
    input_messages: Any = None  # raw JSON from Logfire
    output_messages: Any = None  # raw JSON from Logfire
    start_timestamp: str


class ExtractionItem(BaseModel):
    episode_id: int | None = None  # set for episode-type items
    session_id: str | None = None  # set for summary-type items
    content: str
    content_type: str = "episode"  # episode | summary | manual
    project: str | None = None


class ExtractedEntity(BaseModel):
    name: str
    type: str  # open-ended: Tool, Project, Decision, Issue, etc.
    summary: str = ""


class ExtractedFact(BaseModel):
    source: str  # entity name
    target: str  # entity name
    relationship: str  # e.g. USES, DECIDED, HAS_ISSUE
    fact: str  # full searchable statement: "X uses Y for Z"


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    facts: list[ExtractedFact] = Field(default_factory=list)


class CombinedExtraction(BaseModel):
    """Validated LLM response: entities + facts with cross-reference consistency.

    Mirrors Graphiti's ``CombinedExtraction`` (prompts/extract_nodes_and_edges.py:51-55)
    and the orphan-drop pass in ``combined_extraction.py:280-295``. The
    post-validation step enforces two invariants graceful-degradation style
    (drop, never raise):

    1. Every fact's ``source`` and ``target`` must normalize-exact-match the
       ``name`` of at least one entity in ``entities``. Facts that reference
       unknown entities are DROPPED into ``dropped_facts`` rather than raised
       — bad facts shouldn't take the whole extraction down with them.

    2. After fact-pruning, ``entities`` is left as-is here; the orphan-drop
       (removing entities not referenced by any surviving fact) happens
       downstream in ``process_item`` because deterministic-extractor entities
       merge in later and the orphan pass needs to see the combined pool.
    """

    entities: list[ExtractedEntity] = Field(default_factory=list)
    facts: list[ExtractedFact] = Field(default_factory=list)
    dropped_facts: list[ExtractedFact] = Field(default_factory=list)
    dropped_entities: list[ExtractedEntity] = Field(default_factory=list)

    @model_validator(mode="after")
    def _enforce_entity_fact_consistency(self) -> CombinedExtraction:
        """Drop facts whose source/target don't match a declared entity name.

        Case- and whitespace-insensitive match. The dropped facts and any
        empty-named entities are recorded on ``dropped_facts`` /
        ``dropped_entities`` so the caller can log counts without re-walking
        the raw response.

        Also drops policy-banned entity names (see :func:`banned_name` — bare
        figures and abstraction nodes), which the prompt forbids but the model
        still emits occasionally. Same drop semantics as everything else here:
        the entity goes to ``dropped_entities`` and any fact using it as an
        ENDPOINT falls out via the cross-reference pass below; a fact that only
        mentions the banned string inside its text is kept.

        Also applies the field-length caps (module constants above), graceful
        degradation style: an over-cap entity name drops the entity (a
        200+-char "name" is never a real referent — and its facts fall out via
        the cross-reference pass), an over-cap summary is blanked (the entity
        itself is still real), and an over-cap fact or relationship drops the
        fact. Lengths are logged; content is not.
        """
        # Filter out empty- and pathological-named entities up front; blank
        # over-cap summaries in place.
        valid_entities: list[ExtractedEntity] = []
        dropped_entities: list[ExtractedEntity] = []
        for entity in self.entities:
            if not entity.name or not entity.name.strip():
                dropped_entities.append(entity)
                continue
            if len(entity.name) > MAX_ENTITY_NAME_LEN:
                logger.info(
                    "Dropped entity with over-cap name (len=%d cap=%d)",
                    len(entity.name),
                    MAX_ENTITY_NAME_LEN,
                )
                dropped_entities.append(entity)
                continue
            reason = banned_name(entity.name) if _name_filter_on() else None
            if reason:
                # Facts using this entity as an endpoint fall out below via the
                # cross-reference pass (an edge cannot survive a missing
                # endpoint); facts that merely MENTION the name in their text
                # are untouched.
                logger.info("Dropped banned entity name (%s)", reason)
                dropped_entities.append(entity)
                continue
            if len(entity.summary) > MAX_ENTITY_SUMMARY_LEN:
                logger.info(
                    "Blanked over-cap entity summary (len=%d cap=%d)",
                    len(entity.summary),
                    MAX_ENTITY_SUMMARY_LEN,
                )
                entity.summary = ""
            valid_entities.append(entity)

        entity_name_keys: set[str] = {_normalize_entity_name(e.name) for e in valid_entities}

        kept_facts: list[ExtractedFact] = []
        dropped_facts: list[ExtractedFact] = []
        for fact in self.facts:
            src_key = _normalize_entity_name(fact.source)
            tgt_key = _normalize_entity_name(fact.target)
            if not src_key or not tgt_key:
                dropped_facts.append(fact)
                continue
            if src_key not in entity_name_keys or tgt_key not in entity_name_keys:
                dropped_facts.append(fact)
                continue
            if len(fact.fact) > MAX_FACT_LEN or len(fact.relationship) > MAX_RELATIONSHIP_LEN:
                logger.info(
                    "Dropped over-cap fact (fact_len=%d relationship_len=%d)",
                    len(fact.fact),
                    len(fact.relationship),
                )
                dropped_facts.append(fact)
                continue
            kept_facts.append(fact)

        # Reassign via __dict__ to bypass validator recursion.
        self.__dict__["entities"] = valid_entities
        self.__dict__["facts"] = kept_facts
        self.__dict__["dropped_facts"] = dropped_facts
        self.__dict__["dropped_entities"] = dropped_entities
        return self
