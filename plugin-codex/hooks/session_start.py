#!/usr/bin/env python3
# mypy: ignore-errors
"""Codex ``SessionStart`` hook → board + preferences into context, catchup sweep.

Folds three Claude-plugin SessionStart hooks into one script (Codex runs each
hook entry as a separate process; one fetch pass is cheaper):

  * board_block     — GET /context, the always-injected index of explicit
                      memories, project-scoped by cwd basename
  * preferences_block — GET /preferences/top, max 8 lines
  * ingest catchup  — detached ``synapse_stop_hook.py --catchup`` sweep that
                      ships any rollout tails the live hook missed

Output is Codex's JSON envelope: {"hookSpecificOutput": {"hookEventName":
"SessionStart", "additionalContext": ...}} — Codex does not read plain stdout.
Disable pieces with SYNAPSE_BOARD=0 / SYNAPSE_PREFS_BLOCK=0 /
SYNAPSE_CODEX_CATCHUP=0. Fail-open everywhere: a broken board must never
break a session start.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from common import SURFACE, _cfg, get_json

_MAX_PREF_LINES = 7
_PREF_MARK = {"like": "likes", "dislike": "dislikes", "rule": "rule"}


def _cwd_to_project(cwd: str | None) -> str | None:
    if not cwd:
        return None
    return cwd.rstrip("/").rsplit("/", 1)[-1] or None


def _board_text(project: str | None) -> str | None:
    if _cfg("SYNAPSE_BOARD", "1") == "0":
        return None
    try:
        params = {"project": project} if project else {}
        params["surface"] = SURFACE  # host trust (schema 053) — the server does the filtering
        r = get_json("/context", params, timeout=10)
        return r.get("text") if r.get("status") == "ok" else None
    except Exception:
        return None


def _prefs_text() -> str | None:
    if _cfg("SYNAPSE_PREFS_BLOCK", "1") == "0":
        return None
    try:
        items = (get_json("/preferences/top", {"limit": 8}, timeout=10)).get("items") or []
        lines = ["[Synapse preferences]"]
        for it in items[:_MAX_PREF_LINES]:
            if it.get("pref"):
                tag = _PREF_MARK.get(it.get("polarity"), it.get("polarity") or "")
                lines.append(f"  - ({tag}) {it['pref']}")
        return "\n".join(lines) if len(lines) > 1 else None
    except Exception:
        return None


def _spawn_catchup() -> None:
    if _cfg("SYNAPSE_CODEX_CATCHUP", "1") == "0":
        return
    stop_hook = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synapse_stop_hook.py")
    try:
        subprocess.Popen(
            [sys.executable, stop_hook, "--catchup"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:
        pass


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception:
        payload = {}
    _spawn_catchup()
    project = _cwd_to_project(payload.get("cwd")) or _cwd_to_project(os.getcwd())
    parts = [t for t in (_prefs_text(), _board_text(project)) if t]
    if not parts:
        return
    print(
        json.dumps(
            {
                "suppressOutput": True,
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": "\n\n".join(parts),
                },
            }
        )
    )


if __name__ == "__main__":
    main()
