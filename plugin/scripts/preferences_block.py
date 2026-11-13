#!/usr/bin/env python3
# mypy: ignore-errors
"""Claude Code ``SessionStart`` hook → print the user's standing preferences into context.

The "make preferences present" half of the wiring (schema 035): a BOUNDED factual block —
the top standing user preferences, max 8 lines — printed to stdout so it lands in the
session's context. Not query-scoped, so it avoids the relevance-mismatch problem that got
query-blind recall injection unwired (#171); preferences are few and apply to every turn.

Reads the machine-token-gated ``/preferences/top`` route (thin client, no DSN). The block
is the client core's (synapse_client.context.preferences_part); this hook prints it in
the SessionStart envelope. Disable with SYNAPSE_PREFS_BLOCK=0. Fail-open: any error
prints nothing and exits 0 — session start never breaks.
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
from synapse_client.api import Client


def main() -> None:
    try:
        settings = config.live_settings()
        block = context.preferences_part(Client(settings, config.USER_AGENT), settings)
        if block:
            print(
                json.dumps(
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "SessionStart",
                            "additionalContext": block,
                        }
                    }
                )
            )  # inside the guard: rendering must not break the session
    except Exception:
        return  # fail-open: no block, no noise


if __name__ == "__main__":
    main()
