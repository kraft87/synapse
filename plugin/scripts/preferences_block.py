#!/usr/bin/env python3
# mypy: ignore-errors
"""Claude Code ``SessionStart`` hook → print the user's standing preferences into context.

The "make preferences present" half of the wiring (schema 035): a BOUNDED factual block —
the top standing user preferences, max 8 lines — printed to stdout so it lands in the
session's context. Not query-scoped, so it avoids the relevance-mismatch problem that got
query-blind recall injection unwired (#171); preferences are few and apply to every turn.

Reads the machine-token-gated ``/preferences/top`` route (thin client, no DSN). Disable
with SYNAPSE_PREFS_BLOCK=0. Fail-open: any error prints nothing and exits 0 — session
start never breaks.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import _cfg, get_json

# Header counts toward the block, so cap the preference lines so the whole block is <= 8 lines.
_MAX_LINES = 7
_MARK = {"like": "likes", "dislike": "dislikes", "rule": "rule"}


def main() -> None:
    if _cfg("SYNAPSE_PREFS_BLOCK", "1") == "0":
        return
    try:
        r = get_json("/preferences/top", {"limit": 8}, timeout=10)
        items = r.get("items") or []
    except Exception:
        return  # fail-open: no block, no noise
    if not items:
        return
    lines = ["[Synapse preferences]"]
    for it in items[:_MAX_LINES]:
        pref = it.get("pref")
        if not pref:
            continue
        tag = _MARK.get(it.get("polarity"), it.get("polarity") or "")
        lines.append(f"  - ({tag}) {pref}")
    if len(lines) > 1:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
