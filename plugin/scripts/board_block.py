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
Disable with SYNAPSE_BOARD=0. Fail-open: any error prints nothing and exits 0 — a
broken board must never break a session start.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
        r = get_json("/context", params, timeout=10)
        text = r.get("text") if r.get("status") == "ok" else None
    except Exception:
        return  # fail-open: no block, no noise
    if text:
        print(text)


if __name__ == "__main__":
    main()
