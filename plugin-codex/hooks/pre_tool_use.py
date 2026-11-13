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

recall / recall_full_turns also get ``self_since``, the time of this session's last
compaction, read from the rollout file Codex passes as ``transcript_path``
(synapse_client.compaction, in the Claude plugin beside this one): the turns before it
have left the context, so serving hides only the ones since. Only when ``self_session``
is this session's id, and never on the other three tools.

The ``surface`` injection is GONE (schema 054): trust is bound to the per-device token
the client authenticates with, so a host name asserted by the client identifies nothing
and there is no param left to inject.

Codex requires ``permissionDecision: "allow"`` alongside ``updatedInput``.
That auto-approves the call — acceptable here because the matched tools are
memory lookups plus remember() (a write the user asked for either way), and the
config matcher plus the in-script guard below both scope this to synapse tools.

Fail-open: on any error, no output — the call proceeds unmodified. A failure to read
the cut only loses the cut.
"""

from __future__ import annotations

import json
import os
import re
import sys

# `fetch` is absent: it has no session dimension, and the surface lane it used to
# receive was removed with schema 054.
_SYNAPSE_TOOL = re.compile(
    r"^mcp__.*__(recall|recall_full_turns|recall_feedback|fetch_session|remember)$"
)
_SINCE_TOOLS = ("recall", "recall_full_turns")


def _cut(rollout_path):
    """The session's last compaction time, or None. The client core lives in the Claude
    plugin beside this one; ``plugin/`` goes first on sys.path, as synapse_project.py
    puts it, and the import is here so that a core that cannot load costs only the cut."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(here, "..", "..", "plugin"))
        from synapse_client.compaction import codex_cut

        return codex_cut(rollout_path)
    except Exception:
        return None


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
