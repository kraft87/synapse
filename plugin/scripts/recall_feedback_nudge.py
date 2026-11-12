#!/usr/bin/env python3
# mypy: ignore-errors
"""PostToolUse hook — fires after synapse recall() returns.

Injects a one-line reminder to close the retrieval-quality loop with recall_feedback().
recall_feedback is offline labeled data (eval goldens, reranker tuning) that never changes
recall's ranking (only the board's within-type note order reads it), so the model has no
in-band reason to call it and won't unprompted.

OFF by default since 2026-08-26 (was on): the labeled data only serves whoever tunes the
retrieval stack — for everyone else it's ~60 tokens of noise after every recall plus an
extra tool call. Dev boxes opt in with SYNAPSE_RECALL_FEEDBACK_NUDGE=1.

Mirrors recall_nudge.py (the UserPromptSubmit "use Synapse" reminder): a static ~60-token
directive, zero latency, zero API calls — the model still decides which ids were load-bearing.

The anchored matcher in hooks.json ensures this fires ONLY on recall() and
recall_full_turns() (both serve rateable ids), NOT on recall_feedback /
fetch.

Opt-in: SYNAPSE_RECALL_FEEDBACK_NUDGE=1 (env or plugin install options).
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import _cfg

# The client core, reached the way synapse_project.py reaches it: plugin/ first on sys.path.
# The text is the core's, shared with the Codex hook.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synapse_client.context import recall_feedback_nudge


def main() -> None:
    if _cfg("SYNAPSE_RECALL_FEEDBACK_NUDGE", "0") != "1":
        return
    msg = recall_feedback_nudge()
    print(
        json.dumps(
            {"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": msg}}
        )
    )


if __name__ == "__main__":
    main()
