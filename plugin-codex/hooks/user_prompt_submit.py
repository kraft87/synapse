#!/usr/bin/env python3
# mypy: ignore-errors
"""Codex ``UserPromptSubmit`` hook — per-prompt reminder to use Synapse.

The client core's prompt nudge (synapse_client.context.prompt_nudge), as the Claude
plugin's recall_nudge.py prints it: a wall-clock line, so remember() and timeline
references know the moment, and a static ~45-token directive. Standing instructions at
the top of a long context get ignored under pressure; one injected WITH each prompt
rides recency and keeps recall/remember use consistent. Zero latency, zero API calls —
the model still decides relevance. The tool names are the ones Codex's model sees.

Codex reads the JSON envelope, not plain stdout.
Kill switches, each "0" to disable: SYNAPSE_PROMPT_TIMESTAMP, SYNAPSE_RECALL_NUDGE.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from common import live_settings
from synapse_client import context


def main() -> None:
    text = context.prompt_nudge(
        live_settings(), recall="synapse recall", remember="synapse remember"
    )
    if not text:
        return
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": text,
                },
            }
        )
    )


if __name__ == "__main__":
    main()
