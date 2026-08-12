#!/usr/bin/env python3
# mypy: ignore-errors
"""Private mode toggle — take a Claude Code session off the record.

    private_mode.py on  <session_id>          # nothing from this session ever ingests
    private_mode.py off <session_id>          # stop skipping locally (row stays)
    private_mode.py off <session_id> --forget # ALSO delete the server row (see below)
    private_mode.py status <session_id>
    private_mode.py --session-end             # SessionEnd hook: reads the payload on stdin

Two writes, both required, because they cover different failure modes:

  1. a marker file at ~/.synapse/private/<session_id> — the Stop hook stats it before
     every POST, so the turns never leave this machine, even with the server down;
  2. a row in private_sessions on the server — the durable half. The transcript stays
     on disk long after the marker expires, so without the row a later catch-up sweep
     or `python -m ingestion.backfill` would ingest exactly what the hook skipped.

`on` therefore verifies BOTH landed (the server flag is read back, not assumed) and
exits nonzero, loudly, if either did not. Only exit 0 means "off the record" is true.

`off` removes the marker and LEAVES the row. Turning private mode off does not
retroactively make already-private turns ingestable — those turns are still on disk and
the row is the only thing keeping them out. Since a session id is never reused, the row
also means the rest of that session stays private, which is the honest reading of
"we were off the record". `--forget` is the explicit escape hatch that deletes the row
(and re-exposes the session to future backfills); the SessionEnd hook never uses it.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

PRIVATE_DIR = config.PRIVATE_DIR


def _marker(session_id: str):
    return PRIVATE_DIR / session_id


def _write_marker(session_id: str) -> None:
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    _marker(session_id).write_text("", encoding="utf-8")


def _remove_marker(session_id: str) -> bool:
    """Delete the marker. True if it's gone afterwards (already-absent counts)."""
    try:
        _marker(session_id).unlink()
    except FileNotFoundError:
        pass
    return not _marker(session_id).exists()


def _server(method: str, session_id: str) -> dict:
    return config.request_json(method, f"/private-sessions/{session_id}", timeout=15)


def _server_private(session_id: str) -> bool:
    """Read the durable flag back from the server. Raises on transport/HTTP error."""
    return bool(_server("GET", session_id).get("private"))


def _fail(msg: str) -> int:
    print(f"private mode FAILED: {msg}", file=sys.stderr)
    print("This session is NOT off the record. Do not tell the user it is.", file=sys.stderr)
    return 1


def _on(session_id: str) -> int:
    # Marker first: it is the half that works with no network, and it protects the very
    # next turn. If the server write then fails we say so loudly rather than silently
    # relying on a marker that expires in 12h.
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
            f"server flag not set ({type(e).__name__}: {str(e)[:160]}). Local marker is in place, but a later backfill could still ingest this session."
        )
    print(f"private mode ON for {session_id}")
    print(f"  marker: {_marker(session_id)}")
    print("  server: private_sessions row written (permanent)")
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
        flagged = _server_private(session_id)
        server = "yes" if flagged else "no"
    except Exception as e:
        server = f"unknown ({type(e).__name__})"
    print(f"{session_id}: marker={'yes' if marker else 'no'} server={server}")
    return 0


def _session_end() -> int:
    """SessionEnd hook: drop the marker for the ending session, keep the server row.

    The row is the durable record that those turns must never ingest — including via a
    backfill months from now — so cleanup is local-only. Fail-soft and always exit 0:
    a hook must not break a session teardown, and a leftover marker expires on its own.
    """
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
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        print("usage: private_mode.py on|off|status <session_id> [--forget]", file=sys.stderr)
        return 2
    action, session_id = argv[0], argv[1].strip()
    if not session_id or "/" in session_id or session_id in (".", ".."):
        return _fail(f"invalid session id {session_id!r}")
    if action == "on":
        return _on(session_id)
    if action == "off":
        return _off(session_id, forget="--forget" in argv[2:])
    return _status(session_id)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
