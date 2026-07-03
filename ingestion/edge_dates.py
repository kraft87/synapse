"""LLM-driven temporal-bounds extractor for newly-written edges.

Wraps the verbatim Graphiti prompt at
``ingestion.prompts.extract_edge_dates`` with the structured-response
schema and error handling needed at write time. Used by
``KGClient.create_edge`` to populate ``t_valid`` / ``t_invalid`` from
fact text when the caller hasn't supplied them already.

Why this lives outside ``contradiction.py``: the contradiction detector and
edge-date extractor share the bi-temporal write path but are otherwise
independent — different prompts, different failure modes, different
inputs. Keeping them in separate modules makes the dependency graph
obvious and prevents one regression from masking another.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# Temporal-marker pre-filter. The date extractor is the single most
# expensive stage per item (Logfire 2026-05-28: 20-68s/call, up to ~9.6K
# output tokens), yet the corpus is overwhelmingly non-temporal structural
# facts ("config_loader.py defines EligibilityConfig class") for which the
# LLM just echoes valid_at=REFERENCE_TIME, invalid_at=null.
#
# Safety: ``extract_batch`` defaults reference_time to now(), and the
# extractor's fallback for a (None, None) result is ALSO now()
# (``valid_at_ts = t_valid_pre or now``). So for an ongoing/present-tense
# fact the LLM's answer (valid_at=now, invalid_at=null) is bit-identical to
# skipping the call and letting the now() fallback fill in. Only facts whose
# TEXT carries a resolvable date / offset / validity-flip can produce a
# different answer — those still go to the LLM.
#
# Bias is toward sending to the LLM: a false positive (non-temporal fact
# sent anyway) just wastes tokens (status quo); a false negative (temporal
# fact skipped) would write a wrong date. The set below is broad on purpose.
_TEMPORAL_RE = re.compile(
    r"""
    \b(?:19|20)\d{2}\b                              # 4-digit years 1900-2099
    | \d{1,2}:\d{2}                                 # clock times HH:MM
    | \d{4}-\d{2}-\d{2}                             # ISO dates
    | \b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b              # slash dates
    | \b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)
        (?:uary|ruary|ch|il|e|y|ust|tember|ober|ember)?\b   # month names/abbrevs
    | \b(?:mon|tues|wednes|thurs|fri|satur|sun)day\b        # weekdays
    | \b(?:yesterday|today|tonight|tomorrow|ago|since|until|till)\b
    | \bas\ of\b
    | \b(?:last|next|this|past|coming|previous|prior|recent)\s+
        (?:year|month|week|day|quarter|decade|hour|minute|night|morning|evening)s?\b
    | \b\d+[-\s]?(?:year|month|week|day|hour|minute|second|decade|quarter)s?\b   # "8-year", "3 months"
    | \b(?:year|month|week|day|hour|minute|decade|quarter)s?\b                   # bare duration units
    | \b(?:annually|daily|weekly|monthly|hourly|yearly)\b
    | \b(?:started|stopped|began|begun|ended|ending|quit|resumed|joined|
        retired|launched|deprecated|expired|expires|originally|initially|
        previously|formerly|former)\b
    | \bno\ longer\b
    | \bused\ to\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _has_temporal_markers(fact: str) -> bool:
    """True if the fact text could carry a resolvable date / offset / validity-flip.

    Conservative: when in doubt it returns True so the fact reaches the LLM.
    Returns False only for text with no temporal tokens at all — the common
    structural-fact case the pre-filter exists to short-circuit.
    """
    return bool(_TEMPORAL_RE.search(fact))


# Structured-response schema mirroring ``ingestion.prompts.models.EdgeDates``.
# Sent as `response_format` to constrain Haiku's output.
_EDGE_DATES_SCHEMA: dict[str, Any] = {
    "type": "json",
    "schema": {
        "type": "object",
        "properties": {
            "valid_at": {"type": ["string", "null"]},
            "invalid_at": {"type": ["string", "null"]},
        },
        "required": ["valid_at", "invalid_at"],
        "additionalProperties": False,
    },
}

_BATCH_EDGE_DATES_SCHEMA: dict[str, Any] = {
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
                        "valid_at": {"type": ["string", "null"]},
                        "invalid_at": {"type": ["string", "null"]},
                    },
                    "required": ["id", "valid_at", "invalid_at"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    },
}

_DEFAULT_MODEL = "claude-haiku-4-5"


class EdgeDateExtractor:
    """LLM-backed bi-temporal date extractor for fact edges.

    Stateless; safe to share one instance across all writes. Never raises —
    on any failure ``extract`` returns ``(None, None)`` so the caller falls
    back to ``datetime.now(UTC)`` for ``t_valid``.
    """

    def __init__(self, llm_client: Any, *, model: str = _DEFAULT_MODEL) -> None:
        self._llm = llm_client
        self._model = model

    def extract(
        self, fact: str, reference_time: str | None = None
    ) -> tuple[str | None, str | None]:
        """Return ``(valid_at, invalid_at)`` as ISO 8601 strings or ``None``.

        ``reference_time`` defaults to ``datetime.now(UTC).isoformat()`` so
        relative expressions ("last week") resolve against the moment of the
        write. Callers that have a per-episode timestamp (the typical
        extractor path) should pass it through verbatim.
        """
        if not fact or not fact.strip():
            return (None, None)
        if self._llm is None:
            return (None, None)
        # Non-temporal facts resolve to the caller's now() fallback anyway —
        # skip the LLM. See _TEMPORAL_RE.
        if not _has_temporal_markers(fact):
            return (None, None)
        ref = reference_time or datetime.now(UTC).isoformat()
        try:
            from .prompts.extract_edge_dates import build_prompt

            messages = build_prompt({"fact": fact, "reference_time": ref})
            response = self._llm.messages.create(
                model=self._model,
                max_tokens=200,
                messages=messages,
                response_format=_EDGE_DATES_SCHEMA,
            )
            raw = response.content[0].text.strip()
            if not raw:
                return (None, None)
            data = json.loads(raw)
            valid_at = data.get("valid_at")
            invalid_at = data.get("invalid_at")
            # Normalize empty strings to None — some LLM responses come back
            # with "" instead of null even with the schema.
            return (valid_at or None, invalid_at or None)
        except Exception as exc:
            logger.debug("EdgeDateExtractor failed for fact=%r: %s", (fact or "")[:80], exc)
            return (None, None)

    def extract_batch(
        self, facts: list[str], reference_time: str | None = None
    ) -> list[tuple[str | None, str | None]]:
        """Extract (valid_at, invalid_at) for many facts in ONE LLM call.

        Returns a list parallel to ``facts`` — index i in the return matches
        ``facts[i]``. Facts missing in the LLM response (or any parse failure
        of the whole batch) fall back to ``(None, None)``, matching the
        single-fact ``extract``'s conservative posture so a date-extraction
        miss never blocks an edge write.

        Empty/blank fact strings are skipped without an LLM call and return
        ``(None, None)``; the index alignment with the input list is
        preserved.
        """
        if not facts:
            return []
        n = len(facts)
        out: list[tuple[str | None, str | None]] = [(None, None)] * n
        # Build the indexed item list, skipping blanks AND facts with no
        # temporal markers. The latter resolve to (now(), null) via the
        # caller's fallback, identical to what the LLM would return — so
        # they never need an LLM call. See _TEMPORAL_RE for the safety
        # argument.
        items: list[dict[str, Any]] = []
        for i, fact in enumerate(facts):
            if fact and fact.strip() and _has_temporal_markers(fact):
                items.append({"id": i, "fact": fact})
        if not items or self._llm is None:
            return out

        ref = reference_time or datetime.now(UTC).isoformat()
        try:
            from .prompts.extract_edge_dates import build_batch_prompt

            messages = build_batch_prompt(items, ref)
            response = self._llm.messages.create(
                model=self._model,
                # Allow headroom per fact for two ISO timestamps + the
                # wrapping JSON. 80 tokens/fact is generous.
                max_tokens=max(200, 80 * len(items)),
                messages=messages,
                response_format=_BATCH_EDGE_DATES_SCHEMA,
            )
            raw = response.content[0].text.strip()
            if not raw:
                return out
            # Tolerate trailing prose / code fences (same pattern as the
            # batched stage 6b path).
            start = raw.find("{")
            if start < 0:
                return out
            data, _ = json.JSONDecoder().raw_decode(raw[start:])
        except Exception as exc:
            logger.debug("EdgeDateExtractor.extract_batch failed (n=%d): %s", len(items), exc)
            return out

        for r in data.get("results", []):
            fid = r.get("id")
            if not isinstance(fid, int) or fid < 0 or fid >= n:
                continue
            valid = r.get("valid_at") or None
            invalid = r.get("invalid_at") or None
            out[fid] = (valid, invalid)
        return out


__all__ = ["EdgeDateExtractor"]
