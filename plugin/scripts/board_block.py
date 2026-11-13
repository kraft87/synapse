#!/usr/bin/env python3
# mypy: ignore-errors
"""Claude Code ``SessionStart`` hook → print the board into context.

The board (schema 041) is a small always-injected index of explicit memories: curated
note hooks, the last week's milestones, and a banner saying what memory exists at all.
It replaces the timeline-milestones block — the server renders the milestones INSIDE
the board now, so one block covers both. Server-rendered and hard-capped server-side;
this hook just fetches and prints, so caps and layout evolve without a plugin release.

Reads the machine-token-gated ``GET /context`` route (thin client, no DSN). The project
scope comes from the hook payload's ``cwd``, labeled exactly as the ingest hook labels
the episodes it sends (``synapse_project``: marker file, git toplevel, folder name).

No ``surface`` param is sent any more (schema 054): the token identifies the caller, and
a hostname the client asserts is no longer evidence of anything. Enrollment itself is
interactive — it prints a sign-in code and waits for a human — so this hook never runs
it; it only says so when this machine holds no device credential, because an empty
session start with no explanation is the one outcome worse than a restricted one.

It also carries the one-line credential notice (config.token_override_notice): when an
enrolled machine's device token overrode a SYNAPSE_INGEST_TOKEN from the environment,
say so and where to remove it, to the user (systemMessage) and the model. That line is
local, needs no server, and is printed even with the board disabled or the server down,
since a silent override is the bug it exists to prevent.

What it prints is the client core's (synapse_client.context): the notice, the board
part, with the enrollment explainers in this plugin's words (enroll.py), assembled. The
hook adds the stdin payload's project and the JSON envelope.

Disable the board with SYNAPSE_BOARD=0. Fail-open: any error prints nothing beyond that
notice and exits 0 — a broken board must never break a session start.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import enroll
import synapse_project

# The client core, reached the way synapse_project.py reaches it: plugin/ first on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synapse_client import context
from synapse_client.api import Client


def _emit_context(text: str, user_notice: str = "") -> None:
    """``user_notice`` also goes out as ``systemMessage``, which Claude Code shows to the
    user directly rather than only to the model."""
    if not text:
        return
    out: dict = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": text,
        }
    }
    if user_notice:
        out["systemMessage"] = user_notice
    print(json.dumps(out))


def _project_label() -> str | None:
    """Project label for the hook payload's cwd; falls back to the process cwd. The same
    resolver labels the episodes the ingest hook sends, so the board's project section
    reads exactly the project its sessions are filed under."""
    try:
        cwd = json.loads(sys.stdin.read() or "{}").get("cwd")
    except Exception:
        cwd = None
    return synapse_project.project_label(cwd) or synapse_project.project_label(os.getcwd())


def _board(settings) -> str:
    """The server-rendered board, plus the enrollment explainer when one applies: a 401
    to a machine with no device credential, or a 200 that serves it as ``restricted``
    (the silent version of the same thing). "" when the server is down."""
    return context.board_part(
        Client(settings, config.USER_AGENT),
        _project_label(),
        settings,
        timeout=10,
        not_enrolled=enroll.not_enrolled_block(),
        restricted=enroll.restricted_block(),
    )


def main() -> None:
    try:
        notice = config.token_override_notice()
    except Exception:
        notice = ""
    board = ""
    try:
        settings = config.live_settings()
        # Checked before the payload is read: a disabled board reads nothing, sends nothing.
        if settings.flag("SYNAPSE_BOARD", "1") != "0":
            board = _board(settings)
    except Exception:
        pass  # fail-open: no board, no noise
    try:
        # guarded: a print that raises must not break the session
        _emit_context(context.assemble((notice, board)), user_notice=notice)
    except Exception:
        return


if __name__ == "__main__":
    main()
