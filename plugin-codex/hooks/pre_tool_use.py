#!/usr/bin/env python3
# mypy: ignore-errors
"""Codex ``PreToolUse`` hook → inject the calling session's id into synapse tools.

Port of plugin/scripts/self_session_inject.py. recall / recall_full_turns
serving excludes the calling session's own episodes (already in context); the
model can't know its own session id, so this adds it as ``self_session`` via
Codex's ``updatedInput``. recall_feedback gets it as ``session_id`` so reports
group by session.

Codex requires ``permissionDecision: "allow"`` alongside ``updatedInput``.
That auto-approves the call — acceptable here because the matched tools are
all read-only memory lookups (the config matcher plus the in-script guard
below both scope this to synapse recall-family tools only).

Fail-open: on any error, no output — the call proceeds unmodified.
"""

from __future__ import annotations

import json
import re
import sys

_SYNAPSE_TOOL = re.compile(r"^mcp__.*__(recall|recall_full_turns|recall_feedback|fetch_session)$")


def main() -> None:
    try:
        payload = json.load(sys.stdin)
        tool_name = payload.get("tool_name") or ""
        # In-script guard mirrors the config matcher so a broader-than-intended
        # matcher can never auto-allow or rewrite an unrelated tool.
        if not _SYNAPSE_TOOL.match(tool_name):
            return
        session_id = payload.get("session_id")
        tool_input = payload.get("tool_input") or {}
        if not session_id or not isinstance(tool_input, dict):
            return
        field = "session_id" if tool_name.endswith("recall_feedback") else "self_session"
        if tool_input.get(field):
            return  # already set (retry or nested call) — don't overwrite
        tool_input[field] = session_id
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "allow",
                        "permissionDecisionReason": "synapse self-session injection",
                        "updatedInput": tool_input,
                    }
                }
            )
        )
    except Exception:
        pass  # fail open: never block the tool call


if __name__ == "__main__":
    main()
