#!/usr/bin/env python3
# mypy: ignore-errors
"""Codex ``PostToolUse`` hook — fires after synapse recall() returns.

The Claude plugin's recall_feedback_nudge.py for Codex, with the same text (the client
core's, synapse_client.context.recall_feedback_nudge): a static reminder to close the
retrieval-quality loop with recall_feedback(). The labels are offline tuning data the
model has no in-band reason to produce, useful only to whoever tunes the retrieval
stack, so the nudge is a developer setting: OFF unless SYNAPSE_RECALL_FEEDBACK_NUDGE=1
is in Codex's environment, as in the Claude plugin.

In-script guard keeps this on recall/recall_full_turns only (both serve
rateable ids), NOT recall_feedback/fetch, regardless of matcher breadth.
"""

from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from common import _cfg
from synapse_client.context import recall_feedback_nudge

_RATEABLE = re.compile(r"^mcp__.*__(recall|recall_full_turns)$")


def main() -> None:
    if _cfg("SYNAPSE_RECALL_FEEDBACK_NUDGE", "0") != "1":
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
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": recall_feedback_nudge(),
                },
            }
        )
    )


if __name__ == "__main__":
    main()
