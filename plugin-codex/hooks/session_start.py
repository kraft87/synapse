#!/usr/bin/env python3
# mypy: ignore-errors
"""Codex ``SessionStart`` hook → board + preferences into context, catchup sweep.

Folds the Claude plugin's SessionStart hooks into one script (Codex runs each hook entry
as a separate process; one pass is cheaper). The text is the client core's
(synapse_client.context), the same as every harness gets:

  * token notice    — the device token overrode a configured one, or is withheld
  * upload warning  — when uploads keep failing, one line saying since when and why
                      (the Stop hook's local cursor state, no network)
  * board           — GET /context, the always-injected index of explicit memories,
                      project-scoped by the shared project resolver
                      (plugin/synapse_client/project.py); for a machine that is not
                      enrolled, or is served as ``restricted`` with no device token,
                      the explainer that says how to fix it
  * preferences     — the top standing preferences, max 7 lines, rendered by the
                      server and returned by the same GET /context

and, around them:

  * ingest catchup  — detached ``synapse_stop_hook.py --catchup`` sweep that ships any
                      rollout tails the live hook missed, and retries failed uploads
                      however old
  * skills sync     — ``scripts/skills_sync.py --cwd <cwd>`` (opt-in: ``synapse login``
                      asks), run inline so the synced skills exist before Codex scans
                      ~/.agents/skills and the project's <root>/.agents/skills

Time: Codex kills this hook at 20 s (install.py's block). The board and preferences are
one request, bounded at 10 s, made while the skills sync runs (at most 12 s,
``SYNAPSE_CODEX_SKILLS_SYNC_TIMEOUT``); everything is cut off 15 s after the hook
starts, and whatever is ready by then is printed.

Output is Codex's JSON envelope: {"hookSpecificOutput": {"hookEventName":
"SessionStart", "additionalContext": ...}} — Codex does not read plain stdout. The upload
warning also goes out as the top-level ``systemMessage``, which Codex surfaces to the
user as a warning. Disable pieces with SYNAPSE_BOARD=0 / SYNAPSE_PREFS_BLOCK=0 /
SYNAPSE_CODEX_CATCHUP=0 / SYNAPSE_SKILLS_SYNC=0. Fail-open everywhere: a broken board
must never break a session start.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
# The project resolver is the Claude plugin's, so both plugins label a folder the same
# way. Appended, not prepended, so nothing there can shadow a module this hook imports.
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "plugin", "scripts")
)
import synapse_project
from common import _cfg, client, live_settings, token_override_notice
from synapse_client import context

_SCRIPTS = sys.path[0]
_STOP_HOOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synapse_stop_hook.py")

#: How a Codex user signs in, for the not-enrolled and restricted explainers.
_LOGIN_HINT = context.DEFAULT_LOGIN_HINT
#: Each request's own bound, as before the move onto the core.
_FETCH_TIMEOUT = 10.0
#: Everything is done this long after the hook starts: 5 s of margin under Codex's 20 s
#: for the interpreter's start, the print, and a slow machine.
_DEADLINE_S = 15.0
_STARTED = time.monotonic()


def _left() -> float:
    """Seconds left before the hook's deadline (negative once past it)."""
    return _STARTED + _DEADLINE_S - time.monotonic()


def _served(project: str | None) -> tuple[str, str]:
    # The board and the preferences block, from one request. No `surface` param since
    # schema 054: the bearer identifies the caller. Each is "" when it is switched off
    # (SYNAPSE_BOARD=0, SYNAPSE_PREFS_BLOCK=0) or anything fails (context.server_parts).
    return context.server_parts(
        client(), project, live_settings(), timeout=_FETCH_TIMEOUT, login_hint=_LOGIN_HINT
    )


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


def _sync_skills(cwd: str | None = None) -> None:
    if _cfg("SYNAPSE_SKILLS_SYNC", "0") in ("", "0"):
        return
    # Bounded by its own setting and by the hook's deadline: Codex kills the hook at 20s.
    # A kill mid-pull is safe — the engine's next pass re-mirrors the skill and a push can
    # never shrink the server's file set.
    try:
        budget = min(float(_cfg("SYNAPSE_CODEX_SKILLS_SYNC_TIMEOUT", "12")), _left())
        if budget <= 0:
            return
        subprocess.run(
            [sys.executable, os.path.join(_SCRIPTS, "skills_sync.py")]
            + (["--cwd", cwd] if cwd else []),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=budget,
        )
    except Exception:
        pass


def _fetch(project: str | None) -> tuple[threading.Thread, list[tuple[str, str]]]:
    """Start the request for the board and the preferences; the pair lands in the
    returned list when it is back. A daemon thread: the hook prints and exits at its
    deadline without waiting for a request that is still out."""
    ready: list[tuple[str, str]] = []

    def run() -> None:
        try:
            ready.append(_served(project))
        except Exception:
            pass

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, ready


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception:
        payload = {}
    # Read before the sweep starts, so the line reflects the streak as it stood;
    # a sweep that succeeds ends the streak.
    warning = _upload_warning()
    _spawn_catchup(str(payload.get("transcript_path") or ""))
    project = synapse_project.project_label(payload.get("cwd")) or synapse_project.project_label(
        os.getcwd()
    )
    thread, ready = _fetch(project)
    _sync_skills(str(payload.get("cwd") or os.getcwd()))
    thread.join(max(0.0, _left()))
    # What arrived by the deadline; a late reply is dropped.
    board, prefs = ready[0] if ready else ("", "")
    # The local lines lead (the credential notice, then the upload warning), so they show
    # even with the server down; then the board, then the preferences.
    text = context.assemble((token_override_notice(), warning, board, prefs))
    if not text:
        return
    out = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": text,
        },
    }
    if warning:
        out = {"systemMessage": warning, **out}  # the user sees it; the model gets it in context
    print(json.dumps(out))


if __name__ == "__main__":
    main()
