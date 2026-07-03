"""Verbatim ports of Graphiti's prompt templates.

Each module pairs the formatted message list (a function that takes a
``context`` dict and returns ``[{"role": ..., "content": ...}, ...]``)
with the matching Pydantic response model. Use the response model as the
``response_format`` for structured-output LLM calls.

See ``NOTICE.md`` in this directory for Apache 2.0 attribution.
"""

from __future__ import annotations

__all__ = [
    "dedupe_edges",
    "dedupe_nodes",
    "extract_edge_dates",
    "invalidate_edges",
    "models",
]
