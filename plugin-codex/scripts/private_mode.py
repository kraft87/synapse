#!/usr/bin/env python3
# mypy: ignore-errors
"""Private mode toggle for Codex sessions — port of plugin/scripts/private_mode.py.

    private_mode.py on  <session_id>          # nothing from this session ever ingests
    private_mode.py off <session_id>          # stop skipping locally (row stays)
    private_mode.py off <session_id> --forget # ALSO delete the server row
    private_mode.py status <session_id>
    private_mode.py --session-end             # SessionEnd hook: payload on stdin

Same two-write contract as the Claude plugin, against the same marker dir and
server routes (Codex and Claude session ids share the private_sessions table):

  1. marker file at ~/.synapse/private/<session_id> — the Stop hook stats it
     before every POST;
  2. row in private_sessions on the server — the durable half that keeps a
     later catchup/backfill from ingesting what the hook skipped.

`on` verifies BOTH landed and exits nonzero loudly if either did not.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

PRIVATE_DIR = common.PRIVATE_DIR


def _marker(session_id: str):
    return PRIVATE_DIR / session_id


def _write_marker(session_id: str) -> None:
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    _marker(session_id).write_text("", encoding="utf-8")


def _remove_marker(session_id: str) -> bool:
    try:
        _marker(session_id).unlink()
    except FileNotFoundError:
        pass
    return not _marker(session_id).exists()


def _server(method: str, session_id: str) -> dict:
    return common.request_json(method, f"/private-sessions/{session_id}", timeout=15)


def _server_private(session_id: str) -> bool:
    return bool(_server("GET", session_id).get("private"))


def _fail(msg: str) -> int:
    print(f"private mode FAILED: {msg}", file=sys.stderr)
    print("This session is NOT off the record. Do not tell the user it is.", file=sys.stderr)
    return 1


def _on(session_id: str) -> int:
    try:
        _write_marker(session_id)
    except OSError as e:
        return _fail(f"could not write marker {_marker(session_id)}: {e}")
    try:
        _server("PUT", session_id)
        if not _server_private(session_id):
            return _fail("server did not record the session as private (read-back said false)")
    except (urllib.error.URLError, OSError, ValueError) as e:
        return _fail(
            f"server flag not set ({type(e).__name__}: {str(e)[:160]}). Local marker is in "
            "place, but a later backfill could still ingest this session."
        )
    print(f"private mode ON for {session_id}")
    return 0


def _off(session_id: str, forget: bool) -> int:
    if not _remove_marker(session_id):
        return _fail(f"could not remove marker {_marker(session_id)}")
    if not forget:
        print(f"private mode OFF for {session_id} (marker removed)")
        print("  server: row KEPT — those turns stay uningestable. --forget to delete it.")
        return 0
    try:
        _server("DELETE", session_id)
        if _server_private(session_id):
            return _fail("server still reports the session as private after DELETE")
    except (urllib.error.URLError, OSError, ValueError) as e:
        return _fail(f"server row not deleted ({type(e).__name__}: {str(e)[:160]})")
    print(f"private mode OFF for {session_id} (marker removed, server row deleted)")
    return 0


def _status(session_id: str) -> int:
    marker = _marker(session_id).exists()
    try:
        server = "yes" if _server_private(session_id) else "no"
    except Exception as e:
        server = f"unknown ({type(e).__name__})"
    print(f"{session_id}: marker={'yes' if marker else 'no'} server={server}")
    return 0


def _session_end() -> int:
    """SessionEnd hook: drop the marker for the ending session, keep the row.
    Fail-soft, always exit 0 — a hook must not break session teardown."""
    try:
        session_id = (json.load(sys.stdin) or {}).get("session_id") or ""
    except Exception:
        return 0
    if session_id:
        try:
            _remove_marker(session_id)
        except OSError:
            pass
    return 0


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--session-end":
        return _session_end()
    if len(argv) < 2 or argv[0] not in ("on", "off", "status"):
        print("usage: private_mode.py on|off|status <session_id> [--forget]", file=sys.stderr)
        return 2
    action, session_id = argv[0], argv[1].strip()
    if not session_id or "/" in session_id or session_id in (".", ".."):
        return _fail(f"invalid session id {session_id!r}")
    if session_id.startswith("-"):
        return _fail(f"session id {session_id!r} looks like a flag")
    if action == "on":
        return _on(session_id)
    if action == "off":
        return _off(session_id, forget="--forget" in argv[2:])
    return _status(session_id)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
