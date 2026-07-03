"""Verbatim port of Graphiti's contradiction-detection prompt (carve-out).

Upstream: ``graphiti_core/prompts/dedupe_edges.py`` (the ``resolve_edge``
function). Graphiti folds duplicate-detection and contradiction-detection
into a single call. Synapse's writer-side ``ContradictionDetector`` only
needs the contradiction half — duplicates are already filtered upstream by
the extractor's Stage 6 — so we carve out a slim prompt with identical
rules, examples, and idx semantics, but only one input list and the
``contradicted_facts`` output field.

Why we split: write-time invalidation hooks fire from places (dream
pipeline, manual writes) that have NO duplicate candidate pool. Forcing
those callers to supply an EXISTING FACTS / INVALIDATION CANDIDATES split
would be artificial. The wording, edge cases, and contradiction examples
are preserved verbatim so the LLM judgement is identical to Graphiti's.

See ``NOTICE.md`` for Apache 2.0 attribution.
"""

# Copyright 2024, Zep Software, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Synapse modifications:
#   - Single input list `existing_facts` (Graphiti uses two lists with
#     continuous idx numbering); the rules apply identically to the
#     single-list case.
#   - Response schema slimmed to `contradicted_facts` only.
#   - Contradiction examples from Graphiti's dedupe_edges.py preserved
#     verbatim (the duplicate-detection example dropped since this prompt
#     does not output a duplicate list).

from __future__ import annotations

from typing import Any

from .models import EdgeContradiction, Message


def build_prompt(context: dict[str, Any]) -> list[dict[str, str]]:
    """Render the contradiction-only prompt.

    Required context keys:
      - ``new_fact`` (str)         — the natural-language fact being written
      - ``existing_facts`` (str)   — newline-joined ``"[idx] fact text"``
        candidate list. Caller is responsible for maintaining the
        ``idx -> uuid`` map so the LLM never sees UUIDs.
    """
    messages = [
        Message(
            role="system",
            content=(
                "You are a fact deduplication assistant. "
                "NEVER mark facts with key differences as duplicates."
            ),
        ),
        Message(
            role="user",
            content=f"""
NEVER mark facts as duplicates if they have key differences, particularly around numeric values, dates, or key qualifiers.

IMPORTANT constraints:
- contradicted_facts: idx values from the EXISTING FACTS list below.

<EXISTING FACTS>
{context["existing_facts"]}
</EXISTING FACTS>

<NEW FACT>
{context["new_fact"]}
</NEW FACT>

CONTRADICTION DETECTION:
- Determine which facts the NEW FACT contradicts from the EXISTING FACTS list.
- A fact can be both semantically the same as the NEW FACT AND contradicted (e.g., the new fact updates/supersedes it).
- Return all contradicted idx values in contradicted_facts.
- If no contradictions, return an empty list for contradicted_facts.

<EXAMPLE>
EXISTING FACT: idx=0, "Alice joined Acme Corp in 2020"
NEW FACT: "Alice joined Acme Corp in 2020"
Result: contradicted_facts=[] (identical factual information — not a contradiction)

EXISTING FACT: idx=1, "Alice works at Acme Corp as a software engineer"
NEW FACT: "Alice works at Acme Corp as a senior engineer"
Result: contradicted_facts=[1] (same relationship but updated title — contradiction)

EXISTING FACT: idx=2, "Bob ran 5 miles on Tuesday"
NEW FACT: "Bob ran 3 miles on Wednesday"
Result: contradicted_facts=[] (different events on different days — neither duplicate nor contradiction)
</EXAMPLE>
""",
        ),
    ]
    return [m.model_dump() for m in messages]


def build_batch_prompt(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Render a BATCHED contradiction-only prompt: many new facts in ONE call.

    Each entry in ``items`` is one new fact with its OWN candidate list::

        {"id": 0, "new_fact": "...",
         "existing_facts": [{"idx": 0, "fact": "..."}, ...]}

    Idx values are scoped per-fact (fact 0's idx 0 is NOT fact 1's idx 0).
    Mirrors the contradiction rules from the single-fact ``build_prompt``
    verbatim; only the input/output shape changes.

    Returns ``[{"role", "content"}]`` dicts.
    """
    import json as _json

    messages = [
        Message(
            role="system",
            content=(
                "You are a fact deduplication assistant. "
                "NEVER mark facts with key differences as duplicates."
            ),
        ),
        Message(
            role="user",
            content=f"""For EACH new fact below, identify which of ITS OWN existing facts it contradicts.
Each new fact has a unique `id` and its own indexed `existing_facts` list. An
idx ONLY refers to facts under the SAME new-fact `id`.

NEVER mark facts as duplicates if they have key differences, particularly around
numeric values, dates, or key qualifiers.

<ITEMS>
{_json.dumps(items, ensure_ascii=False)}
</ITEMS>

CONTRADICTION DETECTION (apply per-item):
- Determine which existing_facts the new_fact contradicts.
- A fact can be both semantically the same AND contradicted (e.g., the new
  fact updates/supersedes it).
- Return ALL contradicted idx values per id in `contradicted_facts`.
- If no contradictions for an id, return an empty list for that id.

<EXAMPLE>
ITEMS:
[{{"id": 0, "new_fact": "Alice works at Acme Corp as a senior engineer",
   "existing_facts": [{{"idx": 0, "fact": "Alice works at Acme Corp as a software engineer"}}]}},
 {{"id": 1, "new_fact": "Bob ran 3 miles on Wednesday",
   "existing_facts": [{{"idx": 0, "fact": "Bob ran 5 miles on Tuesday"}}]}}]
Result: {{"results": [{{"id": 0, "contradicted_facts": [0]}}, {{"id": 1, "contradicted_facts": []}}]}}
</EXAMPLE>

Return ONLY JSON of the form:
{{"results": [{{"id": <int>, "contradicted_facts": [<int>]}}, ...]}}
with exactly one object per new-fact id.
""",
        ),
    ]
    return [m.model_dump() for m in messages]


# Re-export the response model so callers can do
# `from .invalidate_edges import build_prompt, EdgeContradictionResponse`.
EdgeContradictionResponse = EdgeContradiction


__all__ = ["EdgeContradictionResponse", "build_batch_prompt", "build_prompt"]
