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
import re
import sys

# `fetch` is absent: it has no session dimension, and the surface lane it used to
# receive was removed with schema 054.
_SYNAPSE_TOOL = re.compile(
    r"^mcp__.*__(recall|recall_full_turns|recall_feedback|fetch_session|remember)$"
)


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
        if not isinstance(tool_input, dict):
            return
        base = tool_name.rsplit("__", 1)[-1]
        changed = False

        if session_id:
            field = "session_id" if base in ("recall_feedback", "remember") else "self_session"
            if not tool_input.get(field):  # already set (retry or nested call)
                tool_input[field] = session_id
                changed = True

        if not changed:
            return
        print(
            json.dumps(
                {
                    "suppressOutput": True,
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
