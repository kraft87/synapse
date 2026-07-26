#!/usr/bin/env python3
# mypy: ignore-errors
"""Claude Code `PreToolUse` hook → inject the calling session's id into recall.

The recall tool's session-cap suppresses SELF-session domination (the calling
conversation's own just-ingested turns crowding its recall results — content
that is already in the model's context window). The model cannot know its own
session id, so this hook adds it to the tool input as `self_session` via the
PreToolUse `updatedInput` mechanism. Server-side, the cap only fires on the
session id this field carries (SYNAPSE_RECALL_SESSION_CAP_SCOPE=self), so two
concurrent sessions never suppress each other's genuinely relevant content.

Fail-open by design: on any error, exit 0 with no output — the tool call
proceeds unmodified and the server simply skips self-suppression.
"""

import json
import sys


def main() -> None:
    try:
        payload = json.load(sys.stdin)
        session_id = payload.get("session_id")
        tool_input = payload.get("tool_input") or {}
        if not session_id or not isinstance(tool_input, dict):
            return
        # recall carries the caller id as self_session (serving-side self-exclusion);
        # recall_feedback's existing session_id param groups reports so noise can be
        # classified by session later (all 96 historical rows were NULL — unmeasurable).
        tool_name = payload.get("tool_name") or ""
        field = "session_id" if tool_name.endswith("recall_feedback") else "self_session"
        if tool_input.get(field):
            return  # already set (retry or nested call) — don't overwrite
        tool_input[field] = session_id
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "updatedInput": tool_input,
                    }
                }
            )
        )
    except Exception:
        pass  # fail open: never block the tool call


if __name__ == "__main__":
    main()
