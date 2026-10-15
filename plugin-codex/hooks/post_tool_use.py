#!/usr/bin/env python3
# mypy: ignore-errors
"""Codex ``PostToolUse`` hook — fires after synapse recall() returns.

Port of plugin/scripts/recall_feedback_nudge.py, which owns the text and the switch: a
static reminder to close the retrieval-quality loop with recall_feedback(). Feedback is
offline labeled data the model has no in-band reason to produce, so, as on Claude, the
nudge is OFF unless a dev box opts in (`synapse config recall_feedback_nudge on`, or
SYNAPSE_RECALL_FEEDBACK_NUDGE=1): for everyone else it is noise after every recall.

In-script guard keeps this on recall/recall_full_turns only (both serve
rateable ids), NOT recall_feedback/fetch, regardless of matcher breadth.
"""

from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import common  # noqa: F401  (loads the shared config recall_feedback_nudge imports)

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "plugin", "scripts")
)
import recall_feedback_nudge as nudge

_RATEABLE = re.compile(r"^mcp__.*__(recall|recall_full_turns)$")


def main() -> None:
    if not nudge.enabled():
        return
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return
    if _RATEABLE.match(payload.get("tool_name") or ""):
        print(nudge.envelope())


if __name__ == "__main__":
    main()
