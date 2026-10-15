"""The message type the ported Graphiti prompt modules build their lists from.

Ported from ``graphiti_core/prompts/models.py`` (``Message``). The response
shapes live in ``ingestion.llm_schemas``.

See ``NOTICE.md`` in this directory for Apache 2.0 attribution.
"""

# Copyright 2024, Zep Software, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from pydantic import BaseModel


class Message(BaseModel):
    """A single message in the prompt's message list. Mirrors Graphiti's shape."""

    role: str
    content: str


__all__ = ["Message"]
