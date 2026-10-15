"""LLM-driven valid-from extractor for newly-written edges.

Wraps the Graphiti-derived prompt at ``ingestion.prompts.extract_edge_dates``
with the structured-response schema and error handling needed at write time.
Stage 7 of the extraction pipeline calls it once per fact batch to read
``valid_at`` out of the fact text. (The prompt used to ask for ``invalid_at`` too;
a fact is never born retired, so it is no longer asked for.)
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime, time
from typing import Any

from ingestion.llm_client import MalformedResponseError, structured_call
from ingestion.llm_schemas import BatchEdgeDatesResult
from ingestion.prompts.extract_edge_dates import build_batch_prompt

logger = logging.getLogger(__name__)

# Temporal-marker pre-filter. Before it, the date extractor was the single most
# expensive stage per item (Logfire 2026-05-28: 20-68s/call, up to ~9.6K
# output tokens), yet the corpus is overwhelmingly non-temporal structural
# facts ("config_loader.py defines EligibilityConfig class") for which the
# LLM just echoes valid_at=REFERENCE_TIME.
#
# Safety: ``extract_batch`` defaults reference_time to now(), and Stage 7's
# fallback for a None result is the segment time or now(). So for an
# ongoing/present-tense fact the LLM's answer is the same as skipping the
# call and letting the fallback fill in. Only facts whose
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
    | \b(?:jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)
        (?:uary|ruary|ch|il|e|y|ust|tember|ober|ember)?\b   # month names/abbrevs
    | \bmay\s+\d                                   # "May 3": the month, not the modal
    | \b(?:in|since|until|till|of|last|next|early|late|mid|during)\s+may\b
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


def _noon_if_date_only(value: str | None) -> str | None:
    """``value`` as an ISO instant, a bare date read as noon UTC; None for junk.

    The prompt asks for noon when the text names only a date, but a model also answers a
    date alone, or midnight, and a stored UTC midnight is the previous evening for a viewer
    west of UTC (the exact bug ``localtime`` fixed for episodes). Noon UTC lands on the same
    calendar day anywhere from UTC-11 to UTC+11, the convention the timeline gate already
    writes for a date it resolves. Nothing in the pipeline records a time of day for a
    date-only fact, so a value at UTC midnight is taken to be one.
    """
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    instant = (parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)).astimezone(UTC)
    if instant.time() == time(0):
        instant = instant.replace(hour=12)
    return instant.isoformat()


def _has_temporal_markers(fact: str) -> bool:
    """True if the fact text could carry a resolvable date / offset / validity-flip.

    Conservative: when in doubt it returns True so the fact reaches the LLM.
    Returns False only for text with no temporal tokens at all — the common
    structural-fact case the pre-filter exists to short-circuit.
    """
    return bool(_TEMPORAL_RE.search(fact))


# The response shape is enforced via pydantic-ai structured outputs — see
# ``ingestion.llm_schemas.BatchEdgeDatesResult``.

_DEFAULT_MODEL = "claude-haiku-4-5"


class EdgeDateExtractor:
    """LLM-backed bi-temporal date extractor for fact edges.

    Stateless; safe to share one instance across all writes. A response that
    does not validate costs the batch its dates (the caller falls back to the
    segment time); an outage propagates, so the queue retries the item.
    """

    def __init__(self, llm_client: Any, *, model: str = _DEFAULT_MODEL) -> None:
        self._llm = llm_client
        self._model = model

    def extract_batch(
        self, facts: list[str], reference_time: str | None = None
    ) -> list[str | None]:
        """Extract valid_at for many facts in ONE LLM call.

        Returns a list parallel to ``facts`` — index i in the return matches
        ``facts[i]``. Each date is an ISO instant, a date with no time at noon UTC
        (``_noon_if_date_only``). Facts missing in the LLM response (or a response that
        never validates) fall back to ``None``, so a date-extraction miss never blocks an
        edge write. A transport failure raises: writing the batch would stamp every fact
        with the segment time for good.

        Empty/blank fact strings are skipped without an LLM call and return ``None``;
        the index alignment with the input list is preserved.
        """
        if not facts:
            return []
        n = len(facts)
        out: list[str | None] = [None] * n
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
        messages = build_batch_prompt(items, ref)
        try:
            batch = structured_call(
                self._llm,
                output_model=BatchEdgeDatesResult,
                messages=messages,
                model=self._model,
                # Headroom per fact for an ISO timestamp + the wrapping JSON.
                max_tokens=max(200, 80 * len(items)),
            )
        except MalformedResponseError as exc:
            logger.warning(
                "edge dates: no usable response for %d fact(s); they take the segment time: %s",
                len(items),
                exc,
            )
            return out

        for r in batch.results:
            if r.id < 0 or r.id >= n:
                continue
            out[r.id] = _noon_if_date_only(r.valid_at)
        return out


__all__ = ["EdgeDateExtractor"]
