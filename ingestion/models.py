from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


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
        """
        # Filter out empty-named entities up front; they can't be referenced.
        valid_entities: list[ExtractedEntity] = []
        dropped_entities: list[ExtractedEntity] = []
        for entity in self.entities:
            if not entity.name or not entity.name.strip():
                dropped_entities.append(entity)
                continue
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
            kept_facts.append(fact)

        # Reassign via __dict__ to bypass validator recursion.
        self.__dict__["entities"] = valid_entities
        self.__dict__["facts"] = kept_facts
        self.__dict__["dropped_facts"] = dropped_facts
        self.__dict__["dropped_entities"] = dropped_entities
        return self
