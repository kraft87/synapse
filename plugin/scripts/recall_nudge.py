#!/usr/bin/env python3
# mypy: ignore-errors
"""UserPromptSubmit hook — a one-line per-prompt reminder to use Synapse.

CLAUDE.md-style standing instructions sit at the top of a long context and get
ignored under pressure; a reminder injected WITH each prompt rides recency and
makes recall/remember use consistent. Deliberately NOT the retired auto-recall
push (which injected multi-second recall RESULTS with no relevance gate and
degraded reasoning): this is a static ~40-token directive — zero latency, zero
API calls, the model still decides relevance.

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
        "[Synapse] If this prompt involves a device, project, tool, person, purchase, "
        "or past decision/work, call synapse:recall BEFORE answering from general "
        "knowledge. When the user asks to remember something durable, call "
        "synapse:remember."
    )


if __name__ == "__main__":
    main()
