#!/usr/bin/env python3
# mypy: ignore-errors
"""Claude Code `PreToolUse` hook → inject the calling session's id into synapse tools.

recall / recall_full_turns: serving excludes the calling session's own episodes
entirely (SYNAPSE_RECALL_SELF_EXCLUDE) — that content is already in the model's
context window. The model cannot know its own session id, so this hook adds it
to the tool input as `self_session` via the PreToolUse `updatedInput` mechanism;
the exclusion is keyed to exactly that id, so two concurrent sessions never
suppress each other's genuinely relevant content.

recall_feedback: the same id lands in the existing `session_id` param so
feedback reports group by session (same-session noise analysis needs it).

Fail-open by design: on any error, exit 0 with no output — the tool call
proceeds unmodified and the server simply skips self-exclusion / grouping.
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
