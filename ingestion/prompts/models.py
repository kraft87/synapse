"""Pydantic response models for the verbatim-ported Graphiti prompts.

Ported from:
- ``graphiti_core/prompts/models.py``       (``Message``)
- ``graphiti_core/prompts/dedupe_nodes.py`` (``NodeDuplicate``)
- ``graphiti_core/prompts/dedupe_edges.py`` (``EdgeDuplicate``)
- ``graphiti_core/prompts/extract_edges.py`` (``EdgeTimestamps``)

See ``NOTICE.md`` in this directory for Apache 2.0 attribution.
"""

# Copyright 2024, Zep Software, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Synapse modification: combined Graphiti's per-prompt response models into
# one models module so all prompt modules can import from a single place.
# Field descriptions and defaults preserved verbatim.

from __future__ import annotations

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in the prompt's message list. Mirrors Graphiti's shape."""

    role: str
    content: str


# ---------------------------------------------------------------------------
# dedupe_nodes — single-pair response
# ---------------------------------------------------------------------------


class NodeDuplicate(BaseModel):
    """Response shape for ``dedupe_nodes.node`` — single new-vs-existing pair.

    ``duplicate_candidate_id`` is the ``candidate_id`` (an integer index into
    the EXISTING ENTITIES list sent in the prompt) of the matching existing
    entity, or ``-1`` for "no match found / uncertain."
    """

    id: int = Field(..., description="integer id of the entity")
    name: str = Field(
        ...,
        description=(
            "Name of the entity. Should be the most complete and descriptive "
            "name of the entity. Do not include any JSON formatting in the "
            "Entity name such as {}."
        ),
    )
    duplicate_candidate_id: int = Field(
        ...,
        description=("candidate_id of the matching EXISTING ENTITY, or -1 if no duplicate exists."),
    )


# ---------------------------------------------------------------------------
# dedupe_edges — duplicate + contradiction sweep response
# ---------------------------------------------------------------------------


class EdgeDuplicate(BaseModel):
    """Response shape for ``dedupe_edges.resolve_edge``.

    Continuous idx numbering across both EXISTING FACTS and FACT INVALIDATION
    CANDIDATES lists is preserved per Graphiti's contract — see the prompt
    body for the exact rules.
    """

    duplicate_facts: list[int] = Field(
        ...,
        description=(
            "List of idx values of duplicate facts (only from EXISTING FACTS "
            "range). Empty list if none."
        ),
    )
    contradicted_facts: list[int] = Field(
        ...,
        description=(
            "List of idx values of contradicted facts (from full idx range). Empty list if none."
        ),
    )


# ---------------------------------------------------------------------------
# invalidate_edges — Synapse-only carve-out of dedupe_edges
#
# Synapse splits Graphiti's combined "duplicate + contradiction" response
# because the writer-side detector (ingestion.contradiction) ONLY needs the
# contradicted_facts list — duplicates are caught upstream in Stage 6 of the
# extractor. We keep the field name verbatim so the same Pydantic shape
# parses both Graphiti's response and ours.
# ---------------------------------------------------------------------------


class EdgeContradiction(BaseModel):
    """Subset of ``EdgeDuplicate`` returned by the contradiction-only prompt."""

    contradicted_facts: list[int] = Field(
        ...,
        description=(
            "List of idx values of contradicted facts. Empty list if the "
            "new fact contradicts none of them."
        ),
    )


# ---------------------------------------------------------------------------
# extract_edge_dates — temporal-bounds extraction for a single fact
# ---------------------------------------------------------------------------


class EdgeDates(BaseModel):
    """Temporal bounds extracted from a single fact.

    Mirrors Graphiti's ``EdgeTimestamps`` (``extract_edges.py``). Both fields
    accept ``None`` when the fact carries no resolvable temporal information.
    """

    valid_at: str | None = Field(
        None,
        description=(
            "When the fact became true. ISO 8601 with Z suffix (e.g., 2025-04-30T00:00:00Z)"
        ),
    )
    invalid_at: str | None = Field(
        None,
        description=(
            "When the fact stopped being true. ISO 8601 with Z suffix (e.g., 2025-04-30T00:00:00Z)"
        ),
    )


__all__ = [
    "EdgeContradiction",
    "EdgeDates",
    "EdgeDuplicate",
    "Message",
    "NodeDuplicate",
]
