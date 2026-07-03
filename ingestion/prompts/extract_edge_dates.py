"""Verbatim port of Graphiti's single-fact temporal-bounds extractor.

Upstream: ``graphiti_core/prompts/extract_edges.py`` (the
``extract_timestamps`` function — single fact, single REFERENCE TIME).

See ``NOTICE.md`` for Apache 2.0 attribution.
"""

# Copyright 2024, Zep Software, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Synapse modifications: context keys preserved verbatim
# (fact, reference_time). Prompt text and rules unchanged.

from __future__ import annotations

from typing import Any

from .models import EdgeDates, Message


def build_prompt(context: dict[str, Any]) -> list[dict[str, str]]:
    """Render the temporal-bounds extraction prompt.

    Required context keys:
      - ``fact`` (str)            — natural-language fact text
      - ``reference_time`` (str)  — ISO 8601 (typically the episode timestamp);
        used to resolve relative expressions like "last week".
    """
    messages = [
        Message(
            role="system",
            content="You extract temporal bounds from facts. NEVER hallucinate dates.",
        ),
        Message(
            role="user",
            content=f"""Given a FACT and its REFERENCE TIME, determine when the fact became true
(valid_at) and when it stopped being true (invalid_at).

Rules:
- Resolve relative expressions ("last week", "2 years ago", "yesterday") using REFERENCE TIME.
- If the fact is ongoing (present tense), set valid_at to REFERENCE TIME.
- If a change or end is expressed, set invalid_at to the relevant time.
- Leave both null if no time is stated or resolvable.
- If only a date is mentioned (no time), assume 00:00:00.
- Use ISO 8601 with Z suffix (e.g., 2025-04-30T00:00:00Z).
- Do NOT hallucinate or infer dates from unrelated events.

<FACT>
{context["fact"]}
</FACT>

<REFERENCE TIME>
{context["reference_time"]}
</REFERENCE TIME>
""",
        ),
    ]
    return [m.model_dump() for m in messages]


def build_batch_prompt(items: list[dict[str, Any]], reference_time: str) -> list[dict[str, str]]:
    """Render a BATCHED temporal-bounds extraction prompt.

    Each entry in ``items`` is one fact with a stable id::

        {"id": 0, "fact": "Synapse switched from Neo4j to FalkorDB"}

    Rules mirror the single-fact ``build_prompt`` verbatim so the LLM
    judgement is identical — only the input/output shape changes from one
    fact per call to many. Returns ``[{"role", "content"}]`` dicts.
    """
    import json as _json

    messages = [
        Message(
            role="system",
            content="You extract temporal bounds from facts. NEVER hallucinate dates.",
        ),
        Message(
            role="user",
            content=f"""For EACH fact below, determine when the fact became true (valid_at)
and when it stopped being true (invalid_at). Each fact has a unique `id` and
results are returned per-id.

<REFERENCE TIME>
{reference_time}
</REFERENCE TIME>

<FACTS>
{_json.dumps(items, ensure_ascii=False)}
</FACTS>

Rules (apply per-fact):
- Resolve relative expressions ("last week", "2 years ago", "yesterday") using REFERENCE TIME.
- If the fact is ongoing (present tense), set valid_at to REFERENCE TIME.
- If a change or end is expressed, set invalid_at to the relevant time.
- Leave both null if no time is stated or resolvable.
- If only a date is mentioned (no time), assume 00:00:00.
- Use ISO 8601 with Z suffix (e.g., 2025-04-30T00:00:00Z).
- Do NOT hallucinate or infer dates from unrelated events.

Return ONLY JSON of the form:
{{"results": [{{"id": <int>, "valid_at": <iso8601-or-null>, "invalid_at": <iso8601-or-null>}}, ...]}}
with exactly one object per fact id.
""",
        ),
    ]
    return [m.model_dump() for m in messages]


# Re-export the response model so callers can do
# `from .extract_edge_dates import build_prompt, EdgeDatesResponse`.
EdgeDatesResponse = EdgeDates


__all__ = ["EdgeDatesResponse", "build_batch_prompt", "build_prompt"]
