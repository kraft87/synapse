#!/usr/bin/env python3
# mypy: ignore-errors
"""UserPromptSubmit hook — a one-line per-prompt reminder to use Synapse.

CLAUDE.md-style standing instructions sit at the top of a long context and get
ignored under pressure; a reminder injected WITH each prompt rides recency and
makes recall/remember use consistent. Deliberately NOT the retired auto-recall
push (which injected multi-second recall RESULTS with no relevance gate and
degraded reasoning): this is a static ~40-token directive — zero latency, zero
API calls, the model still decides relevance.

Wording (v2, 2026-07-12): imperative + motivating rationale + conditional guard —
the shipped-system pattern (Anthropic's memory-tool directive pairs ALWAYS with
"ASSUME INTERRUPTION"; the openfang #583 fix made "recall first" conditional on
context). Soft pointers under-trigger; suppression adverbs are measured no-ops —
bias reads ON, guard with the already-in-context condition, not softeners.
~45 tokens/turn (v1 was ~22); recall_metrics call-rate decides if it earns its keep.

Kill switch: SYNAPSE_RECALL_NUDGE=0 (env or plugin install options).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import _cfg


def main() -> None:
    if _cfg("SYNAPSE_RECALL_NUDGE", "1") == "0":
        return
    print(
        "[Synapse] IMPORTANT: the user's cross-session history is searchable — assume "
        "it has context on any past work, project, device, or person they name. BEFORE "
        "answering anything that references the past: synapse:recall (skip only if the "
        "answer is already in this conversation). About to say 'noted' or 'I'll "
        "remember'? Call synapse:remember FIRST, then reply."
    )


if __name__ == "__main__":
    main()
