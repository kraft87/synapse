#!/usr/bin/env python3
# mypy: ignore-errors
"""Claude Code `PreToolUse` hook → inject the calling session's id into synapse tools.

recall / recall_full_turns: serving excludes the calling session's own episodes
entirely — that content is already in the model's context window. The model cannot know its own session id, so this hook adds it
to the tool input as `self_session` via the PreToolUse `updatedInput` mechanism;
the exclusion is keyed to exactly that id, so two concurrent sessions never
suppress each other's genuinely relevant content.

recall / recall_full_turns also get `self_since`, the time of this session's last
compaction, read from the payload's `transcript_path` (synapse_client.compaction): the
turns before it have left the context, so serving hides only the ones since. Only when
`self_session` is this session's id, and never on the other three tools, whose schemas
reject an unknown argument.

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
proceeds unmodified and the server simply skips self-exclusion / grouping. A failure
to read the cut only loses the cut.
"""

import json
import os
import re
import sys

# In-script guard mirroring hooks.json's PreToolUse matcher (and the Codex port's), so a
# broader-than-intended matcher can never rewrite an unrelated tool's input. `fetch` is
# absent: it has no session dimension, and since the surface lane was removed there is
# nothing left to inject into it.
_SYNAPSE_TOOL = re.compile(
    r"^mcp__.*__(recall|recall_full_turns|recall_feedback|fetch_session|remember)$"
)
_SINCE_TOOLS = ("recall", "recall_full_turns")


def _cut(transcript_path):
    """The session's last compaction time, or None. The client core is imported the way
    synapse_project.py does, here so that a core that cannot load costs only the cut."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from synapse_client.compaction import claude_code_cut

        return claude_code_cut(transcript_path)
    except Exception:
        return None


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
            transcript = payload.get("transcript_path")
            if (
                base in _SINCE_TOOLS
                and transcript
                and tool_input.get("self_session") == session_id
                and not tool_input.get("self_since")
            ):
                cut = _cut(transcript)
                if cut:
                    tool_input["self_since"] = cut
                    changed = True

        if not changed:
            return
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "allow",
                        "updatedInput": tool_input,
                    }
                }
            )
        )
    except Exception:
        pass  # fail open: never block the tool call


if __name__ == "__main__":
    main()
