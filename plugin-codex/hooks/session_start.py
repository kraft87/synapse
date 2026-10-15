#!/usr/bin/env python3
# mypy: ignore-errors
"""Codex ``SessionStart`` hook → board + preferences into context, catchup sweep.

Folds three Claude-plugin SessionStart hooks into one script (Codex runs each
hook entry as a separate process; one fetch pass is cheaper):

  * board_block     — GET /context, the always-injected index of explicit
                      memories, project-scoped by the shared project resolver
                      (plugin/scripts/synapse_project.py), as every hook labels it
  * preferences_block — GET /preferences/top, max 8 lines
  * ingest catchup  — detached ``synapse_stop_hook.py --catchup`` sweep that
                      ships any rollout tails the live hook missed, and retries
                      failed uploads however old
  * upload warning  — when uploads keep failing, one line saying since when and
                      why (local cursor state only, no network)
  * skills sync     — ``scripts/skills_sync.py --cwd <cwd>`` (opt-in: ``synapse
                      login`` asks), run inline under a wall-clock budget so the
                      synced skills exist before Codex scans ~/.agents/skills and
                      the project's <root>/.agents/skills

The board fetch, the preferences fetch and the skills sync run side by side under one
budget (_BUDGET, inside the hook's 20 s timeout): run one after another, a blackholed
server used up the whole timeout, Codex killed the hook, and even the local lines below
were lost. Whatever has not arrived when the budget ends is left out, and the local lines
(the token notice and the upload warning) always print. For a /clear or a compaction
(``source``) only the board, the preferences and the local lines run: the sync and the
sweep ran when the session started.

Output is Codex's JSON envelope: {"hookSpecificOutput": {"hookEventName":
"SessionStart", "additionalContext": ...}} — Codex does not read plain stdout.
The upload warning also goes out as the top-level ``systemMessage``, which
Codex surfaces to the user as a warning.
Disable pieces with SYNAPSE_BOARD=0 / SYNAPSE_PREFS_BLOCK=0 (or `synapse config board off` /
`prefs_block off`) / SYNAPSE_SKILLS_SYNC=0. Fail-open everywhere: a broken board must never
break a session start.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
# The project resolver is the Claude plugin's, so both plugins label a folder the same
# way. Appended, not prepended, so nothing there can shadow a module this hook imports.
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "plugin", "scripts")
)
import synapse_project
from common import _cfg, get_json, token_override_notice

_SCRIPTS = sys.path[0]
_STOP_HOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synapse_stop_hook.py")

_BUDGET = 15.0  # seconds the fetches and the sync share; the hook's own timeout is 20
_SKILLS_SYNC_TIMEOUT = 12.0
_MAX_PREF_LINES = 7
_PREF_MARK = {"like": "likes", "dislike": "dislikes", "rule": "rule"}


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


def _sync_skills(cwd: str | None = None) -> None:
    if _cfg("SYNAPSE_SKILLS_SYNC", "0") in ("", "0"):
        return
    # Bounded: it shares the hook's budget with the fetches. A kill mid-pull is safe — the
    # engine's next pass re-mirrors the skill and a push can never shrink the server's
    # file set.
    try:
        subprocess.run(
            [sys.executable, os.path.join(_SCRIPTS, "skills_sync.py")]
            + (["--cwd", cwd] if cwd else []),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_SKILLS_SYNC_TIMEOUT,
        )
    except Exception:
        pass


def _concurrently(jobs: dict[str, Callable[[], object]]) -> dict[str, object]:
    """Run each job in its own thread and wait for them all, but no longer than _BUDGET
    in total. Returns the results of the jobs that finished in time; a job that is still
    running is left behind (daemon threads do not hold the process open)."""
    results: dict[str, object] = {}

    def run(name: str, job: Callable[[], object]) -> None:
        try:
            results[name] = job()
        except Exception:
            pass  # fail-open: the jobs catch their own errors; this is the backstop

    threads = [threading.Thread(target=run, args=item, daemon=True) for item in jobs.items()]
    deadline = time.monotonic() + _BUDGET
    for t in threads:
        t.start()
    for t in threads:
        t.join(max(0.0, deadline - time.monotonic()))
    return dict(results)


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception:
        payload = {}
    # A /clear or a compaction opens a new context window, not a new session.
    resumed = payload.get("source") in ("clear", "compact")
    # Read before the sweep starts, so the line reflects the streak as it stood;
    # a sweep that succeeds ends the streak.
    warning = _upload_warning()
    if not resumed:
        _spawn_catchup(str(payload.get("transcript_path") or ""))
    cwd = str(payload.get("cwd") or os.getcwd())
    project = synapse_project.project_label(payload.get("cwd")) or synapse_project.project_label(
        os.getcwd()
    )
    jobs = {"prefs": _prefs_text, "board": lambda: _board_text(project)}
    if not resumed:
        jobs["skills"] = lambda: _sync_skills(cwd)
    fetched = _concurrently(jobs)
    # The credential notice is local and leads: an env token silently overridden by the
    # device token is exactly the split this line exists to surface. The upload warning,
    # also local, follows it.
    parts = [
        t
        for t in (token_override_notice(), warning, fetched.get("prefs"), fetched.get("board"))
        if t
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
