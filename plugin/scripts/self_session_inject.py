#!/usr/bin/env python3
# mypy: ignore-errors
"""Claude Code `PreToolUse` hook → inject the calling session's id into synapse tools.

recall / recall_full_turns: serving excludes the calling session's own episodes
entirely (SYNAPSE_RECALL_SELF_EXCLUDE) — that content is already in the model's
context window. The model cannot know its own session id, so this hook adds it
to the tool input as `self_session` via the PreToolUse `updatedInput` mechanism;
the exclusion is keyed to exactly that id, so two concurrent sessions never
suppress each other's genuinely relevant content.

recall_feedback / remember: the same id lands in each tool's existing
`session_id` param — feedback reports group by session (same-session noise
analysis needs it) and remembered episodes attach to the writing session.
Neither tool accepts `self_session`; sending it fails their pydantic
validation and the whole call errors out.

It used to inject a `surface` (this host's name) as well, which is how schema 053
keyed per-host trust. Schema 054 removed that lane entirely: trust now rides on the
per-device TOKEN the client authenticates with, so there is nothing for a client to
assert and nothing for a model to hallucinate. The server still accepts the param for
one release, from root-token callers only, purely so a 0.16.x plugin keeps working
during the migration.

Fail-open by design: on any error, exit 0 with no output — the tool call
proceeds unmodified and the server simply skips self-exclusion / grouping.
"""

import json
import re
import sys

# In-script guard mirroring hooks.json's PreToolUse matcher (and the Codex port's), so a
# broader-than-intended matcher can never rewrite an unrelated tool's input. `fetch` is
# absent: it has no session dimension, and since the surface lane was removed there is
# nothing left to inject into it.
_SYNAPSE_TOOL = re.compile(
    r"^mcp__.*__(recall|recall_full_turns|recall_feedback|fetch_session|remember)$"
)


def main() -> None:
    try:
        payload = json.load(sys.stdin)
        tool_name = payload.get("tool_name") or ""
        if not _SYNAPSE_TOOL.match(tool_name):
            return
        session_id = payload.get("session_id")
        tool_input = payload.get("tool_input") or {}
        if not isinstance(tool_input, dict):
            return
        base = tool_name.rsplit("__", 1)[-1]
        changed = False

        # recall carries the caller id as self_session (serving-side self-exclusion);
        # recall_feedback's existing session_id param groups reports so noise can be
        # classified by session later (all 96 historical rows were NULL — unmeasurable).
        if session_id:
            field = "session_id" if base in ("recall_feedback", "remember") else "self_session"
            if not tool_input.get(field):  # already set (retry/nested call) — don't overwrite
                tool_input[field] = session_id
                changed = True

        if not changed:
            return
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
