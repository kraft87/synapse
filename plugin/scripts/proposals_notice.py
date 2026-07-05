#!/usr/bin/env python3
# mypy: ignore-errors
"""SessionStart hook — surface pending dream→skills proposals (issue #11).

The skills lane's proposals sit invisible in skills_lane.skill_gap_candidates until
someone runs /synapse:skill-review — three sat unseen for weeks. This fetches the
pending list via the existing /skills/proposals route and prints a one-line notice
when any are waiting. Read-only, so it is NOT gated on SYNAPSE_SKILLS_SYNC (the
notice is how you learn reviews exist); kill switch SYNAPSE_PROPOSALS_NOTICE=0.
Fail-open + silent when zero or the server is unreachable.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import _cfg, post_json


def main() -> None:
    if _cfg("SYNAPSE_PROPOSALS_NOTICE", "1") == "0":
        return
    try:
        rows = post_json("/skills/proposals", {}, timeout=8.0).get("proposals", [])
    except Exception:
        return  # fail-open: server/token/network issues never break session start
    if not rows:
        return
    names = ", ".join(r.get("name") or "?" for r in rows[:3])
    more = f" (+{len(rows) - 3} more)" if len(rows) > 3 else ""
    print(
        f"[Synapse] {len(rows)} dream→skills proposal(s) pending review: "
        f"{names}{more} — run /synapse:skill-review"
    )


if __name__ == "__main__":
    main()
