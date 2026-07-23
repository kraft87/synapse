#!/usr/bin/env python3
# mypy: ignore-errors
"""PostToolUse hook — fires after synapse recall() returns.

Injects a one-line reminder to close the retrieval-quality loop with recall_feedback().
recall_feedback is offline labeled data (eval goldens, reranker tuning) that never changes
live ranking, so the model has no in-band reason to call it and won't unprompted — which
means feedback only ever gets captured on machines that wire this nudge by hand. Shipping it
in the plugin captures organic feedback from every install, not just the author's box.

Mirrors recall_nudge.py (the UserPromptSubmit "use Synapse" reminder): a static ~60-token
directive, zero latency, zero API calls — the model still decides which ids were load-bearing.

The anchored matcher (mcp__..._recall$) in hooks.json ensures this fires ONLY on recall(),
NOT on recall_feedback / recall_episodes / recall_timeline.

Kill switch: SYNAPSE_RECALL_FEEDBACK_NUDGE=0 (env or plugin install options).
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import _cfg


def main() -> None:
    if _cfg("SYNAPSE_RECALL_FEEDBACK_NUDGE", "1") == "0":
        return
    msg = (
        "[Synapse] recall() returned. If you used any of the served ids (e:N / n:N) in your "
        "answer, call recall_feedback(query=<verbatim query>, helpful=[...], noise=[...], "
        'missing="...") ONCE to close the retrieval-quality loop: helpful=load-bearing ids, '
        "noise=irrelevant ids, missing=one line on what was not served. Skip only if you used "
        "none of the results."
    )
    print(
        json.dumps(
            {"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": msg}}
        )
    )


if __name__ == "__main__":
    main()
