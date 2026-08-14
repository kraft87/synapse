#!/usr/bin/env python3
# mypy: ignore-errors
"""Codex ``PostToolUse`` hook — fires after synapse recall() returns.

Port of plugin/scripts/recall_feedback_nudge.py: a static reminder to close
the retrieval-quality loop with recall_feedback(). Feedback is offline labeled
data the model has no in-band reason to produce; the nudge captures it from
every install.

In-script guard keeps this on recall/recall_full_turns only (both serve
rateable ids), NOT recall_feedback/fetch, regardless of matcher breadth.
Kill switch: SYNAPSE_RECALL_FEEDBACK_NUDGE=0.
"""

from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from common import _cfg

_RATEABLE = re.compile(r"^mcp__.*__(recall|recall_full_turns)$")

_NUDGE = (
    "[Synapse] recall() returned. If you used any of the served ids (e:/n:/f:/t:/w:) "
    "in your answer, call recall_feedback(query=<verbatim query>, helpful=[...], "
    'noise=[...], comment="...") ONCE to close the retrieval-quality loop: '
    "helpful=load-bearing ids, noise=irrelevant ids, comment=free text for anything "
    "else (too much served, buried hit, bad ordering). ONLY IF you needed specific "
    "content memory likely holds and it was not served, add missing=what it was + "
    "found_via=where you got it instead (full_turns/filesystem/user/nowhere...) — "
    "omit both otherwise. Skip the call only if you used none of the results."
)


def main() -> None:
    if _cfg("SYNAPSE_RECALL_FEEDBACK_NUDGE", "1") == "0":
        return
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return
    if not _RATEABLE.match(payload.get("tool_name") or ""):
        return
    print(
        json.dumps(
            {
                "suppressOutput": True,
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": _NUDGE,
                },
            }
        )
    )


if __name__ == "__main__":
    main()
