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

No ``surface`` param is sent any more (schema 054): the token identifies the caller, and
a hostname the client asserts is no longer evidence of anything. Enrollment itself is
interactive — it prints a sign-in code and waits for a human — so this hook never runs
it; it only says so when this machine holds no device credential, because an empty
session start with no explanation is the one outcome worse than a restricted one.

Disable with SYNAPSE_BOARD=0. Fail-open: any error prints nothing and exits 0 — a
broken board must never break a session start.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error

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
        params = {"project": project} if project else {}
        try:
            r = get_json("/context", params, timeout=10)
        except urllib.error.HTTPError as e:
            # 401 with no device credential is the one error worth explaining: this
            # machine has not enrolled (or its token was revoked), so it is served
            # nothing and will go on being served nothing until someone signs in.
            if e.code == 401 and not enroll.is_enrolled():
                print(enroll.not_enrolled_block())
            return
        text = r.get("text") if r.get("status") == "ok" else None
        if text:
            print(text)  # inside the guard: a print that raises must not break the session
    except Exception:
        return  # fail-open: no block, no noise


if __name__ == "__main__":
    main()
