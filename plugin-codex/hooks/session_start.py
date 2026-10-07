#!/usr/bin/env python3
# mypy: ignore-errors
"""Codex ``SessionStart`` hook → board + preferences into context, catchup sweep.

Folds three Claude-plugin SessionStart hooks into one script (Codex runs each
hook entry as a separate process; one fetch pass is cheaper):

  * board_block     — GET /context, the always-injected index of explicit
                      memories, project-scoped by cwd basename
  * preferences_block — GET /preferences/top, max 8 lines
  * ingest catchup  — detached ``synapse_stop_hook.py --catchup`` sweep that
                      ships any rollout tails the live hook missed, and retries
                      failed uploads however old
  * upload warning  — when uploads keep failing, one line saying since when and
                      why (local cursor state only, no network)
  * skills sync     — ``scripts/skills_sync.py`` (opt-in, SYNAPSE_SKILLS_SYNC=1),
                      run inline under a wall-clock budget so the synced skills
                      exist before Codex scans ~/.agents/skills

Output is Codex's JSON envelope: {"hookSpecificOutput": {"hookEventName":
"SessionStart", "additionalContext": ...}} — Codex does not read plain stdout.
The upload warning also goes out as the top-level ``systemMessage``, which
Codex surfaces to the user as a warning.
Disable pieces with SYNAPSE_BOARD=0 / SYNAPSE_PREFS_BLOCK=0 /
SYNAPSE_CODEX_CATCHUP=0 / SYNAPSE_SKILLS_SYNC=0. Fail-open everywhere: a broken board must never
break a session start.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from common import _cfg, get_json, token_override_notice

_SCRIPTS = sys.path[0]
_STOP_HOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synapse_stop_hook.py")

_MAX_PREF_LINES = 7
_PREF_MARK = {"like": "likes", "dislike": "dislikes", "rule": "rule"}


def _cwd_to_project(cwd: str | None) -> str | None:
    if not cwd:
        return None
    return cwd.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or None


def _board_text(project: str | None) -> str | None:
    if _cfg("SYNAPSE_BOARD", "1") == "0":
        return None
    try:
        # No `surface` param since schema 054: the bearer identifies the caller, and a
        # host name the client asserts identifies nothing. This mirror reads whatever
        # credential is configured — including the device token the Claude Code plugin's
        # enroll.py writes into the shared settings.json, so a machine running both
        # plugins is ONE surface rather than two.
        r = get_json("/context", {"project": project} if project else {}, timeout=10)
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


def _upload_warning() -> str | None:
    """The Stop hook's sustained-upload-failure line ("Synapse uploads have been
    failing since ..."), or None while uploads are healthy. Same rules and
    wording as the Claude plugin (shared synapse_ingest_retry). Reads the local
    cursor file only; fail-open."""
    try:
        spec = importlib.util.spec_from_file_location("synapse_codex_stop_hook", _STOP_HOOK)
        stop_hook = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stop_hook)
        return stop_hook.failure_warning()
    except Exception:
        return None


def _spawn_catchup(skip_path: str = "") -> None:
    """Detached sweep; ``skip_path`` is this session's own rollout, which its
    Stop hook owns."""
    if _cfg("SYNAPSE_CODEX_CATCHUP", "1") == "0":
        return
    try:
        subprocess.Popen(
            [sys.executable, _STOP_HOOK, "--catchup", skip_path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:
        pass


def _sync_skills() -> None:
    if _cfg("SYNAPSE_SKILLS_SYNC", "0") in ("", "0"):
        return
    # Bounded: the hook block's timeout is 20s and the board fetch still has to fit. A kill
    # mid-pull is safe — the engine's next pass re-mirrors the skill and a push can never
    # shrink the server's file set.
    try:
        budget = float(_cfg("SYNAPSE_CODEX_SKILLS_SYNC_TIMEOUT", "12"))
        subprocess.run(
            [sys.executable, os.path.join(_SCRIPTS, "skills_sync.py")],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=budget,
        )
    except Exception:
        pass


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception:
        payload = {}
    # Read before the sweep starts, so the line reflects the streak as it stood;
    # a sweep that succeeds ends the streak.
    warning = _upload_warning()
    _spawn_catchup(str(payload.get("transcript_path") or ""))
    _sync_skills()
    project = _cwd_to_project(payload.get("cwd")) or _cwd_to_project(os.getcwd())
    # The credential notice is local and leads: an env token silently overridden by the
    # device token is exactly the split this line exists to surface. The upload warning,
    # also local, follows it.
    parts = [
        t for t in (token_override_notice(), warning, _prefs_text(), _board_text(project)) if t
    ]
    if not parts:
        return
    out = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": "\n\n".join(parts),
        },
    }
    if warning:
        out = {"systemMessage": warning, **out}  # the user sees it; the model gets it in context
    print(json.dumps(out))


if __name__ == "__main__":
    main()
