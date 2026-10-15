"""Port of Graphiti's node-dedup prompt, batched.

Upstream: ``graphiti_core/prompts/dedupe_nodes.py`` (the ``node`` function —
one new entity vs its candidate list; ``build_batch_prompt`` renders many of
those in one call).

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
#   - context keys preserved verbatim: previous_episodes, episode_content,
#     extracted_node, entity_type_description, existing_nodes.
#   - Prompt text, examples, and rule structure unchanged.
#   - to_prompt_json copied verbatim from graphiti_core/prompts/prompt_helpers.py
#     (same defaults: ensure_ascii=False, indent=None) so JSON serialization is
#     bit-for-bit identical to Graphiti's prompt rendering.

from __future__ import annotations

import json
from typing import Any

from .models import Message


def _to_prompt_json(data: Any) -> str:
    """Verbatim port of Graphiti's ``to_prompt_json`` (prompt_helpers.py)."""
    return json.dumps(data, ensure_ascii=False, indent=None)


def build_batch_prompt(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Render a BATCHED node-dedup prompt: many new entities in one call.

    Each entry in ``items`` is one new entity carrying its OWN candidate set:

        {"id": 0, "name": "...", "summary": "...",
         "candidates": [{"candidate_id": 0, "name": "...", "summary": "..."}]}

    Candidates are scoped per-entity (entity ``id``'s ``duplicate_candidate_id``
    indexes into *that* entity's ``candidates`` list), so this is behaviour-
    preserving versus confirming each (entity, candidate) pair separately —
    it just collapses N ~30s claude-CLI subprocess calls into one.

    The dedup rules mirror Graphiti's ``dedupe_nodes.node`` verbatim. Returns
    ``[{"role", "content"}]`` dicts.
    """
    messages = [
        Message(
            role="system",
            content=(
                "You are an entity deduplication assistant. "
                "NEVER fabricate entity names or mark distinct entities as duplicates."
            ),
        ),
        Message(
            role="user",
            content=f"""\
For EACH new entity below, decide whether it is a duplicate of one of ITS OWN
candidate entities. Each new entity has a unique `id` and its own list of
`candidates` (each with a `candidate_id`).

<NEW ENTITIES WITH CANDIDATES>
{_to_prompt_json(items)}
</NEW ENTITIES WITH CANDIDATES>

Two entities are duplicates ONLY if they refer to the *same real-world object
or concept*. Semantic Equivalence: if a descriptive label clearly refers to a
named entity, treat them as duplicates.

NEVER mark entities as duplicates if:
- They are related but distinct.
- They have similar names or purposes but refer to separate instances or concepts.

Task: for every new entity `id`, return the `candidate_id` of the matching
candidate from THAT entity's own candidate list, or `-1` when there is no match
or you are unsure. A candidate_id only refers to candidates listed under the
same entity `id`.

<EXAMPLE>
NEW ENTITIES WITH CANDIDATES:
[{{"id": 0, "name": "NYC", "summary": "", "candidates": [{{"candidate_id": 0, "name": "New York City", "summary": ""}}, {{"candidate_id": 1, "name": "New York Knicks", "summary": "NBA team"}}]}},
 {{"id": 1, "name": "Java", "summary": "programming language", "candidates": [{{"candidate_id": 0, "name": "Java", "summary": "An island in Indonesia"}}]}}]
Result: {{"results": [{{"id": 0, "duplicate_candidate_id": 0}}, {{"id": 1, "duplicate_candidate_id": -1}}]}}
</EXAMPLE>

Return ONLY JSON of the form {{"results": [{{"id": <int>, "duplicate_candidate_id": <int>}}, ...]}}
with exactly one object per new entity `id`.
""",
        ),
    ]
    return [m.model_dump() for m in messages]


__all__ = ["build_batch_prompt"]
