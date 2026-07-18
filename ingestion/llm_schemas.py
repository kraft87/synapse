"""Pydantic output models for every structured-output LLM call in the pipeline.

One model per LLM response shape, replacing the hand-written JSON-schema dicts
that used to live next to each call site (``_CONTRADICTION_SCHEMA``,
``_EDGE_DATES_SCHEMA``, ``_NODE_DEDUP_SCHEMA``, ...). Each model is used as a
pydantic-ai ``output_type`` (native structured outputs — OpenRouter/Fireworks
``response_format: json_schema`` with ``strict: true``) and doubles as the
validator for responses coming back over the legacy text surfaces (the Claude
Agent SDK path and test doubles).

Field names mirror the prompts' JSON contracts EXACTLY — do not rename a field
without changing the prompt that asks for it.

Strict-mode notes (OpenAI strict json_schema requires every property in
``required`` and ``additionalProperties: false``):

* All models here are strict-compatible. Optional fields (``valid_at``,
  ``date``, ...) are declared ``str | None`` so the strict transform emits
  them as required-but-nullable — a schema-following model must emit the key,
  possibly ``null``, which the python-side defaults treat identically to a
  missing key.
* Tolerance lives python-side, in validators, NOT in the wire schema: the
  schema asks for the clean shape (e.g. ``list[int]``), while validators
  accept the junk smaller models actually return (deepseek-v4-flash wraps
  indices in ``{"index": N}`` objects or digit strings — seen 2026-07-18) and
  degrade per-item instead of failing the whole response.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, ClassVar

from pydantic import (
    BaseModel,
    BeforeValidator,
    ValidationError,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# Shared tolerant validators
# ---------------------------------------------------------------------------


def _coerce_index_list(val: Any) -> list[int]:
    """Coerce a model-returned index list to bare ints; drop junk.

    The schema asks for bare ints, but smaller models (deepseek-v4-flash,
    2026-07-18) return digit strings or wrap each index in an object
    (``{"index": 3}`` / ``{"id": 3}`` / ``{"fact_index": 3}``); a raw dict
    used to crash the ``in`` membership test with ``TypeError: unhashable``.
    Unrecognized shapes are dropped, matching the conservative no-op default
    of every consuming stage. Kept as defense-in-depth even though native
    structured outputs make wrapped indices unlikely at the provider level.
    """
    out: list[int] = []
    for v in val if isinstance(val, list) else []:
        if isinstance(v, dict):
            v = v.get("index", v.get("id", v.get("fact_index")))
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            out.append(v)
        elif isinstance(v, str) and v.strip().lstrip("-").isdigit():
            out.append(int(v.strip()))
    return out


# Wire schema: plain ``list[int]``. Python-side: the tolerant coercion above.
IndexList = Annotated[list[int], BeforeValidator(_coerce_index_list)]


def _drop_invalid_items[M: BaseModel](item_model: type[M], val: Any) -> list[M]:
    """Validate each list item independently; drop the ones that fail.

    Preserves the legacy dict-walking posture where one malformed result row
    was skipped (``continue``) without discarding the rest of the batch.
    """
    if not isinstance(val, list):
        return []
    out: list[M] = []
    for item in val:
        try:
            out.append(item_model.model_validate(item))
        except ValidationError:
            continue
    return out


# ---------------------------------------------------------------------------
# Stage 6b — duplicate + contradiction resolution (extractor.py)
# ---------------------------------------------------------------------------


class ResolutionResult(BaseModel):
    """Single-fact stage-6b verdict: which existing facts the new fact
    duplicates and/or contradicts (idx values into the prompt's pools)."""

    duplicate_facts: IndexList = []
    contradicted_facts: IndexList = []


class BatchResolutionItem(BaseModel):
    """Per-fact entry in the batched stage-6b response."""

    id: int
    duplicate_facts: IndexList = []
    contradicted_facts: IndexList = []


class BatchResolutionResult(BaseModel):
    """Batched stage-6b response: one item per new-fact ``id``."""

    results: list[BatchResolutionItem] = []

    @field_validator("results", mode="before")
    @classmethod
    def _tolerant_items(cls, v: Any) -> list[BatchResolutionItem]:
        return _drop_invalid_items(BatchResolutionItem, v)


# ---------------------------------------------------------------------------
# Writer-side contradiction detector (contradiction.py)
# ---------------------------------------------------------------------------


class ContradictionVerdict(BaseModel):
    """Contradiction-only verdict for a single fact pair."""

    contradicted_facts: IndexList = []


class BatchContradictionItem(BaseModel):
    id: int
    contradicted_facts: IndexList = []


class BatchContradictionResult(BaseModel):
    results: list[BatchContradictionItem] = []

    @field_validator("results", mode="before")
    @classmethod
    def _tolerant_items(cls, v: Any) -> list[BatchContradictionItem]:
        return _drop_invalid_items(BatchContradictionItem, v)


# ---------------------------------------------------------------------------
# Edge dates (edge_dates.py)
# ---------------------------------------------------------------------------


def _clean_date(v: Any) -> str | None:
    """Empty strings and non-string junk collapse to None (today's ``or None``
    posture, extended so a junk-typed date degrades instead of failing)."""
    if isinstance(v, str) and v:
        return v
    return None


class EdgeDatesResult(BaseModel):
    """Temporal bounds for one fact — ISO 8601 strings or null."""

    valid_at: str | None = None
    invalid_at: str | None = None

    @field_validator("valid_at", "invalid_at", mode="before")
    @classmethod
    def _empty_to_none(cls, v: Any) -> str | None:
        return _clean_date(v)


class BatchEdgeDatesItem(BaseModel):
    id: int
    valid_at: str | None = None
    invalid_at: str | None = None

    @field_validator("valid_at", "invalid_at", mode="before")
    @classmethod
    def _empty_to_none(cls, v: Any) -> str | None:
        return _clean_date(v)


class BatchEdgeDatesResult(BaseModel):
    results: list[BatchEdgeDatesItem] = []

    @field_validator("results", mode="before")
    @classmethod
    def _tolerant_items(cls, v: Any) -> list[BatchEdgeDatesItem]:
        return _drop_invalid_items(BatchEdgeDatesItem, v)


# ---------------------------------------------------------------------------
# Node dedup (dedup.py single-pair + extractor.py batch confirm)
# ---------------------------------------------------------------------------


class NodeDedupVerdict(BaseModel):
    """Node-dedup verdict. Defaults keep partial responses
    parseable (the legacy parser only ever read ``duplicate_candidate_id``);
    the strict wire schema still requires every key."""

    id: int = 0
    name: str = ""
    duplicate_candidate_id: int = -1


class BatchNodeDedupItem(BaseModel):
    id: int
    duplicate_candidate_id: int


class BatchNodeDedupResult(BaseModel):
    results: list[BatchNodeDedupItem] = []

    @field_validator("results", mode="before")
    @classmethod
    def _tolerant_items(cls, v: Any) -> list[BatchNodeDedupItem]:
        return _drop_invalid_items(BatchNodeDedupItem, v)


# ---------------------------------------------------------------------------
# Timeline gate (timeline_gate.py)
# ---------------------------------------------------------------------------

_EVENT_TYPES = ("decision", "action", "finding", "milestone")
_DOMAINS = ("personal", "technical")


class TimelineGateEvent(BaseModel):
    """One naked past-tense timeline event. Invalid enum-ish values degrade
    to their legacy defaults instead of failing the turn (unlabeled fails
    OPEN at read time — a wrong default would hide events from serving)."""

    event: str
    salience: int = 1
    event_type: str | None = None
    domain: str | None = None
    date: str | None = None

    @field_validator("event", mode="before")
    @classmethod
    def _strip(cls, v: Any) -> str:
        return str(v).strip()

    @field_validator("salience", mode="before")
    @classmethod
    def _clamp(cls, v: Any) -> int:
        return v if isinstance(v, int) and 0 <= v <= 2 else 1

    @field_validator("event_type", mode="before")
    @classmethod
    def _known_type(cls, v: Any) -> str | None:
        return v if v in _EVENT_TYPES else None

    @field_validator("domain", mode="before")
    @classmethod
    def _known_domain(cls, v: Any) -> str | None:
        return v if v in _DOMAINS else None

    @field_validator("date", mode="before")
    @classmethod
    def _shape_only(cls, v: Any) -> str | None:
        # Shape only (non-empty string); range/parse validation stays in
        # timeline_gate._resolve_event_date.
        return v.strip() if isinstance(v, str) and v.strip() else None


class TimelineGateEvents(BaseModel):
    """Gate response: up to MAX_EVENTS happenings for one turn.

    Also accepts the legacy single-event shape (``{"event": ...}`` at top
    level) so a model that regresses to the old contract degrades to one
    event instead of a retry loop. A non-list ``events`` value raises so the
    retry-with-feedback loop can correct it — item-level junk is dropped
    silently instead.
    """

    MAX_EVENTS: ClassVar[int] = 3

    events: list[TimelineGateEvent] = []

    @model_validator(mode="before")
    @classmethod
    def _legacy_single_event(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("events") is None:
            return {"events": [data] if data.get("event") else []}
        return data

    @field_validator("events", mode="before")
    @classmethod
    def _cap_and_filter(cls, v: Any) -> list[Any]:
        if not isinstance(v, list):
            raise ValueError("'events' is not a list")
        out: list[Any] = []
        # Cap BEFORE filtering — mirrors the legacy parser's raw[:3] slice.
        for item in v[: cls.MAX_EVENTS]:
            if isinstance(item, TimelineGateEvent):
                out.append(item)
            elif isinstance(item, dict) and item.get("event"):
                out.append(item)
        return out


# ---------------------------------------------------------------------------
# Stage 3 — entity/fact extraction (extractor.py)
# ---------------------------------------------------------------------------


class ExtractionEntityRow(BaseModel):
    """One extracted entity row. ``type``/``summary`` fall back to their
    legacy defaults when missing or null."""

    name: str
    type: str = "Topic"
    summary: str = ""

    @field_validator("type", mode="before")
    @classmethod
    def _default_type(cls, v: Any) -> Any:
        return v if v is not None else "Topic"

    @field_validator("summary", mode="before")
    @classmethod
    def _default_summary(cls, v: Any) -> Any:
        return v if v is not None else ""


class ExtractionFactRow(BaseModel):
    """One extracted fact row — all four keys required, as in the prompt."""

    source: str
    target: str
    relationship: str
    fact: str


class ExtractionOutput(BaseModel):
    """Stage-3 extraction response: entities + facts.

    Bad rows are skipped, never fatal — the same graceful degradation the
    legacy ``_parse_response`` dict-walk applied. Cross-reference validation
    (facts pointing at undeclared entities) happens downstream in
    ``ingestion.models.CombinedExtraction``.
    """

    entities: list[ExtractionEntityRow] = []
    facts: list[ExtractionFactRow] = []

    @field_validator("entities", mode="before")
    @classmethod
    def _tolerant_entities(cls, v: Any) -> list[ExtractionEntityRow]:
        return _drop_invalid_items(ExtractionEntityRow, v)

    @field_validator("facts", mode="before")
    @classmethod
    def _tolerant_facts(cls, v: Any) -> list[ExtractionFactRow]:
        return _drop_invalid_items(ExtractionFactRow, v)


# ---------------------------------------------------------------------------
# Schema + parsing helpers
# ---------------------------------------------------------------------------


def first_json_object(raw: str) -> str:
    """Extract the first JSON object from ``raw``, tolerating leading and
    trailing prose (``raw_decode`` parses the first object; models can ramble
    after it). Raises ``ValueError`` when no object parses.
    """
    start = raw.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in response: {raw[:200]!r}")
    data, _ = json.JSONDecoder().raw_decode(raw[start:])
    return json.dumps(data)


__all__ = [
    "BatchContradictionItem",
    "BatchContradictionResult",
    "BatchEdgeDatesItem",
    "BatchEdgeDatesResult",
    "BatchNodeDedupItem",
    "BatchNodeDedupResult",
    "BatchResolutionItem",
    "BatchResolutionResult",
    "ContradictionVerdict",
    "EdgeDatesResult",
    "ExtractionEntityRow",
    "ExtractionFactRow",
    "ExtractionOutput",
    "IndexList",
    "NodeDedupVerdict",
    "ResolutionResult",
    "TimelineGateEvent",
    "TimelineGateEvents",
    "first_json_object",
]
