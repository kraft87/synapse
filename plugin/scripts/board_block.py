#!/usr/bin/env python3
# mypy: ignore-errors
"""Claude Code ``SessionStart`` hook → print the board into context.

The board (schema 041) is a small always-injected index of explicit memories: curated
note hooks, the last week's milestones, and a banner saying what memory exists at all.
It replaces the timeline-milestones block — the server renders the milestones INSIDE
the board now, so one block covers both. Server-rendered and hard-capped server-side;
this hook just fetches and prints, so caps and layout evolve without a plugin release.

Reads the machine-token-gated ``GET /context`` route (thin client, no DSN). The project
scope comes from the hook payload's ``cwd``, labeled the same way the ingest path labels
episodes (mirror of ``ingestion.jsonl_client._cwd_to_project`` — basename of cwd).

This hook also owns device enrollment (schema 054), because it is the first thing that
needs a credential: it calls ``enroll.ensure_enrolled()`` before fetching. A device
still awaiting approval is served nothing, so it prints the PAIRING block instead of a
board — an empty session start with no explanation is the one outcome worse than a
restricted one. No ``surface`` param is sent any more: the token identifies the caller,
and a hostname the client asserts is no longer evidence of anything.

Disable with SYNAPSE_BOARD=0. Fail-open: any error prints nothing and exits 0 — a
broken board must never break a session start.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import enroll
from config import _cfg, get_json


def _cwd_to_project(cwd: str | None) -> str | None:
    """Mirror of ``ingestion.jsonl_client._cwd_to_project`` — kept inline so the hook
    stays dependency-free (it runs under the CLI's bare Python, off the repo path).
    Must stay in lockstep: the board's project scope has to match how episodes are
    labeled, or the project section goes empty."""
    if not cwd:
        return None
    return cwd.rstrip("/").rsplit("/", 1)[-1] or None


def _project_label() -> str | None:
    """Project label from the hook payload's cwd; falls back to the process cwd."""
    try:
        cwd = json.loads(sys.stdin.read() or "{}").get("cwd")
    except Exception:
        cwd = None
    return _cwd_to_project(cwd) or _cwd_to_project(os.getcwd())


def main() -> None:
    if _cfg("SYNAPSE_BOARD", "1") == "0":
        return
    try:
        project = _project_label()
        state = enroll.ensure_enrolled()
        if state.get("status") == "pending":
            print(enroll.pairing_block(state))
            return
        params = {"project": project} if project else {}
        r = get_json("/context", params, timeout=10)
        text = r.get("text") if r.get("status") == "ok" else None
        if text:
            # A served board is the only proof of approval the client ever gets — the
            # server does not notify, so "it worked" is the signal.
            enroll.mark_approved()
            print(text)  # inside the guard: a print that raises must not break the session
    except Exception:
        # 401 lands here too: a revoked or still-pending device. Print nothing rather
        # than a scary traceback; the pairing block above covers the case we can name.
        return  # fail-open: no block, no noise


if __name__ == "__main__":
    main()
