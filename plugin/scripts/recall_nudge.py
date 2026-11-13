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

Also injects a wall-clock timestamp line ("[Wed 2026-08-26 15:55:01 EDT]",
host-local time): remember() and timeline references need the current moment,
and the model otherwise only knows the session-start date. The weekday is spelled
out because deriving it from a bare date is exactly the arithmetic models get wrong.

The two lines are the client core's (synapse_client.context.prompt_nudge), with the
tools named as Claude Code shows them to the model; this hook prints them in the
UserPromptSubmit envelope.

Kill switches: SYNAPSE_RECALL_NUDGE=0, SYNAPSE_PROMPT_TIMESTAMP=0
(env or plugin install options).
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# The client core, reached the way synapse_project.py reaches it: plugin/ first on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synapse_client import context


def main() -> None:
    text = context.prompt_nudge(
        config.live_settings(), recall="synapse:recall", remember="synapse:remember"
    )
    if text:
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "UserPromptSubmit",
                        "additionalContext": text,
                    }
                }
            )
        )


if __name__ == "__main__":
    main()
