#!/usr/bin/env python3
# mypy: ignore-errors
"""Claude Code `PreToolUse` hook → inject the calling session's id and host into synapse tools.

recall / recall_full_turns: serving excludes the calling session's own episodes
entirely (SYNAPSE_RECALL_SELF_EXCLUDE) — that content is already in the model's
context window. The model cannot know its own session id, so this hook adds it
to the tool input as `self_session` via the PreToolUse `updatedInput` mechanism;
the exclusion is keyed to exactly that id, so two concurrent sessions never
suppress each other's genuinely relevant content.

recall_feedback: the same id lands in the existing `session_id` param so
feedback reports group by session (same-session noise analysis needs it).

`surface` (schema 053, audience scoping): recall, recall_full_turns, fetch,
fetch_session and remember also get this HOST's id, because trust is per-host —
a work laptop must not be served personal notes, and a note written there must
stay readable there. The model cannot know its own hostname either, and it must
not choose one: injecting server-side of the model is what stops a hallucinated
"surface": "<trusted-host>" from widening what a session can read. recall_feedback
is deliberately excluded — it takes no `surface` param, and passing an unknown
argument would fail the call.

Fail-open by design: on any error, exit 0 with no output — the tool call
proceeds unmodified and the server simply skips self-exclusion / grouping. Note
that failing open here is still fail-CLOSED on the server: a tool call that
arrives with no `surface` is treated as restricted.
"""

import contextlib
import json
import os
import re
import sys

# In-script guard mirroring hooks.json's PreToolUse matcher (and the Codex port's), so a
# broader-than-intended matcher can never rewrite an unrelated tool's input. The matcher
# grew to six tools with audience scoping; without this, one config typo would start
# injecting unknown arguments into whatever else it caught.
_SYNAPSE_TOOL = re.compile(
    r"^mcp__.*__(recall|recall_full_turns|recall_feedback|fetch|fetch_session|remember)$"
)

# Tools that accept a `surface` param. recall_feedback is absent on purpose.
_SURFACE_TOOLS = ("recall", "recall_full_turns", "fetch", "fetch_session", "remember")


def _surface() -> str | None:
    """This host's id, or None if config is unavailable (then we simply don't inject).

    The sys.path entry is added and removed around the import rather than left in place
    at module scope: this directory shadows third-party module names (filelock.py), so a
    permanent entry would leak into any process that imports this file — the test suite
    included."""
    added = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, added)
    try:
        from config import SURFACE

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
        # fetch has no session dimension at all, so it gets `surface` only.
        if session_id and base != "fetch":
            field = "session_id" if base == "recall_feedback" else "self_session"
            if not tool_input.get(field):  # already set (retry/nested call) — don't overwrite
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
