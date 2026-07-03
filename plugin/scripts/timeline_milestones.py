#!/usr/bin/env python3
# mypy: ignore-errors
"""Claude Code ``SessionStart`` hook → print recent timeline milestones into context.

The "make the timeline present" half of the usage wiring: a BOUNDED factual block —
the last 7 days' salience-2 events, max 5 lines — printed to stdout so it lands in
the session's context. Time-scoped, not query-scoped, so it doesn't have the
relevance-mismatch problem that got query-blind recall injection unwired (#171).

Reads the machine-token-gated ``/timeline/recent`` route (thin client, no DSN).
Disable with SYNAPSE_TIMELINE_MILESTONES=0. Fail-open: any error prints nothing
and exits 0 — session start never breaks.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import _cfg, post_json


def main() -> None:
    if _cfg("SYNAPSE_TIMELINE_MILESTONES", "1") == "0":
        return
    try:
        r = post_json("/timeline/recent", {"days": 7, "min_salience": 2, "limit": 5}, timeout=10)
        items = r.get("items") or []
    except Exception:
        return  # fail-open: no block, no noise
    if not items:
        return
    lines = ["[Timeline] Recent milestones (7d):"]
    for it in items:
        proj = f" ({it['project']})" if it.get("project") else ""
        lines.append(f"  {it['date']}{proj}: {it['fact']}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
