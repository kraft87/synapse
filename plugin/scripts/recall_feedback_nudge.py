#!/usr/bin/env python3
# mypy: ignore-errors
"""PostToolUse hook — fires after synapse recall() returns.

Injects a one-line reminder to close the retrieval-quality loop with recall_feedback().
recall_feedback is offline labeled data (eval goldens, reranker tuning) that never changes
live ranking, so the model has no in-band reason to call it and won't unprompted.

OFF by default since 2026-08-26 (was on), on both harnesses: the labeled data only serves
whoever tunes the retrieval stack — for everyone else it's ~60 tokens of noise after every
recall plus an extra tool call. Dev boxes opt in with `synapse config recall_feedback_nudge
on` (or SYNAPSE_RECALL_FEEDBACK_NUDGE=1). The hook is still spawned after every recall; it
imports only config (no HTTP stack) and exits at the switch.

Mirrors recall_nudge.py (the UserPromptSubmit "use Synapse" reminder): a static ~60-token
directive, zero latency, zero API calls — the model still decides which ids were load-bearing.

The anchored matcher in hooks.json ensures this fires ONLY on recall() and
recall_full_turns() (both serve rateable ids), NOT on recall_feedback /
fetch.

The Codex hook (plugin-codex/hooks/post_tool_use.py) prints the same NUDGE.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import _cfg

NUDGE = (
    "[Synapse] recall() returned. If you used any of the served ids (e:/n:/f:/t:/w:) "
    "in your answer, call recall_feedback(query=<verbatim query>, helpful=[...], "
    'noise=[...], comment="...") ONCE to close the retrieval-quality loop: '
    "helpful=load-bearing ids, noise=irrelevant ids, comment=free text for anything "
    "else (too much served, buried hit, bad ordering). ONLY IF you needed specific "
    "content memory likely holds and it was not served, add missing=what it was + "
    "found_via=where you got it instead (full_turns/filesystem/user/nowhere...) — "
    "omit both otherwise. Skip the call only if you used none of the results."
)


def enabled() -> bool:
    return _cfg("SYNAPSE_RECALL_FEEDBACK_NUDGE", "0") == "1"


def envelope() -> str:
    return json.dumps(
        {"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": NUDGE}}
    )


def main() -> None:
    if enabled():
        print(envelope())


if __name__ == "__main__":
    main()
