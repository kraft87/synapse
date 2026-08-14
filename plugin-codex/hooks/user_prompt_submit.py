#!/usr/bin/env python3
# mypy: ignore-errors
"""Codex ``UserPromptSubmit`` hook — per-prompt reminder to use Synapse.

Port of plugin/scripts/recall_nudge.py: standing instructions at the top of a
long context get ignored under pressure; a static ~45-token directive injected
WITH each prompt rides recency and keeps recall/remember use consistent. Zero
latency, zero API calls — the model still decides relevance.

Codex reads the JSON envelope, not plain stdout.
Kill switch: SYNAPSE_RECALL_NUDGE=0.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from common import _cfg

_NUDGE = (
    "[Synapse] IMPORTANT: the user's cross-session history is searchable — assume "
    "it has context on any past work, project, device, or person they name. BEFORE "
    "answering anything that references the past: synapse recall (skip only if the "
    "answer is already in this conversation). About to say 'noted' or 'I'll "
    "remember'? Call synapse remember FIRST, then reply."
)


def main() -> None:
    if _cfg("SYNAPSE_RECALL_NUDGE", "1") == "0":
        return
    print(
        json.dumps(
            {
                "suppressOutput": True,
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": _NUDGE,
                },
            }
        )
    )


if __name__ == "__main__":
    main()
