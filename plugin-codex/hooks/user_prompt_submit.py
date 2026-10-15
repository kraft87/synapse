#!/usr/bin/env python3
# mypy: ignore-errors
"""Codex ``UserPromptSubmit`` hook — per-prompt reminder to use Synapse.

The Codex wording of plugin/scripts/recall_nudge.py, which builds the output (the
timestamp line, then the directive) and owns the switches: standing instructions at the
top of a long context get ignored under pressure; a directive injected WITH each prompt
rides recency and keeps recall/remember use consistent. Zero latency, zero API calls —
the model still decides relevance.

Codex reads the JSON envelope, not plain stdout.
Switches: SYNAPSE_RECALL_NUDGE=0, SYNAPSE_PROMPT_TIMESTAMP=0.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import common  # noqa: F401  (loads the shared config recall_nudge imports)

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "plugin", "scripts")
)
from recall_nudge import envelope

_DIRECTIVE = (
    "[Synapse] IMPORTANT: the user's cross-session history is searchable — assume "
    "it has context on any past work, project, device, or person they name. BEFORE "
    "answering anything that references the past: synapse recall (skip only if the "
    "answer is already in this conversation). About to say 'noted' or 'I'll "
    "remember'? Call synapse remember FIRST, then reply."
)


def main() -> None:
    out = envelope(_DIRECTIVE)
    if out:
        print(out)


if __name__ == "__main__":
    main()
