#!/usr/bin/env python3
# mypy: ignore-errors
"""Codex ``PreToolUse`` hook → inject the calling session's id into synapse tools.

Port of plugin/scripts/self_session_inject.py. recall / recall_full_turns
serving excludes the calling session's own episodes (already in context); the
model can't know its own session id, so this adds it as ``self_session`` via
Codex's ``updatedInput``. recall_feedback and remember get it as
``session_id`` (their existing param — neither accepts ``self_session``, which
would fail their pydantic validation) so reports group by session and
remembered episodes attach to the writing session.

The ``surface`` injection is GONE (schema 054): trust is bound to the per-device token
the client authenticates with, so a host name asserted by the client identifies nothing
and there is no param left to inject.

Codex requires ``permissionDecision: "allow"`` alongside ``updatedInput``.
That auto-approves the call — acceptable here because the matched tools are
memory lookups plus remember() (a write the user asked for either way), and the
config matcher plus the in-script guard below both scope this to synapse tools.

Fail-open: on any error, no output — the call proceeds unmodified.
"""

from __future__ import annotations

import json
import os
import sys

# The matching and the field logic are the Claude hook's. Dependency-free, so no config.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "plugin", "scripts")
)
from self_session_inject import inject


def main() -> None:
    try:
        tool_input = inject(json.load(sys.stdin))
        if tool_input is None:
            return
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "allow",
                        "permissionDecisionReason": "synapse session-id injection",
                        "updatedInput": tool_input,
                    },
                }
            )
        )
    except Exception:
        pass  # fail open: never block the tool call


if __name__ == "__main__":
    main()
