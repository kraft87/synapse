#!/usr/bin/env python3
# mypy: ignore-errors
"""Codex ``PreToolUse`` hook → inject the calling session's id and host into synapse tools.

Port of plugin/scripts/self_session_inject.py. recall / recall_full_turns
serving excludes the calling session's own episodes (already in context); the
model can't know its own session id, so this adds it as ``self_session`` via
Codex's ``updatedInput``. recall_feedback gets it as ``session_id`` so reports
group by session.

It also injects ``surface`` — this HOST's id — into recall, recall_full_turns,
fetch, fetch_session and remember, which is what audience scoping (schema 053)
keys per-host trust on. recall_feedback takes no such param and is excluded.

Codex requires ``permissionDecision: "allow"`` alongside ``updatedInput``.
That auto-approves the call — acceptable here because the matched tools are
memory lookups plus remember() (a write the user asked for either way), and the
config matcher plus the in-script guard below both scope this to synapse tools.

Fail-open: on any error, no output — the call proceeds unmodified. That is still
fail-CLOSED server-side: a call arriving without ``surface`` is treated as restricted.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys

_SYNAPSE_TOOL = re.compile(
    r"^mcp__.*__(recall|recall_full_turns|recall_feedback|fetch|fetch_session|remember)$"
)
_SURFACE_TOOLS = ("recall", "recall_full_turns", "fetch", "fetch_session", "remember")


def _surface() -> str | None:
    """This host's id, or None if config is unavailable. The sys.path entry is scoped to
    the import (see the Claude Code hook for why a permanent one is a hazard)."""
    added = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
    sys.path.insert(0, added)
    try:
        from common import SURFACE

        return SURFACE or None
    except Exception:
        return None
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(added)


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

        # fetch has no session dimension — it gets `surface` only.
        if session_id and base != "fetch":
            field = "session_id" if base == "recall_feedback" else "self_session"
            if not tool_input.get(field):  # already set (retry or nested call)
                tool_input[field] = session_id
                changed = True

        if base in _SURFACE_TOOLS and not tool_input.get("surface"):
            surface = _surface()
            if surface:
                tool_input["surface"] = surface
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
                        "permissionDecisionReason": "synapse session/surface injection",
                        "updatedInput": tool_input,
                    },
                }
            )
        )
    except Exception:
        pass  # fail open: never block the tool call


if __name__ == "__main__":
    main()
