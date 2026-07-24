"""Text safety helpers for values headed into Postgres.

Postgres TEXT columns reject NUL (0x00) bytes outright, and jsonb rejects the
``\\u0000`` escape — one stray NUL in a transcript or an LLM response fails
the whole INSERT. Strip them at the write boundary instead of letting a
single byte poison an episode or a batch of graph edges.
"""

from __future__ import annotations

from typing import Any


def strip_nul(value: Any) -> Any:
    """Recursively remove NUL bytes from strings inside any str/dict/list/tuple.

    Non-string leaves pass through untouched.
    """
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, dict):
        return {key: strip_nul(item) for key, item in value.items()}
    if isinstance(value, list):
        return [strip_nul(item) for item in value]
    if isinstance(value, tuple):
        return tuple(strip_nul(item) for item in value)
    return value
