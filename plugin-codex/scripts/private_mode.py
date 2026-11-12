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
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

PRIVATE_DIR = common.PRIVATE_DIR


def _plain_name(session_id: str) -> bool:
    """One ordinary path component: no separator, not ``.``/``..``, no NUL. The client
    core's rule (synapse_client/private.py), copied: the id is joined into the marker's
    path, and ``../x`` would write or delete a file outside PRIVATE_DIR."""
    seps = {"/", "\\", os.sep, os.altsep or "/"}
    return (
        session_id not in (".", "..")
        and "\0" not in session_id
        and not any(sep in session_id for sep in seps)
    )


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
    sid = urllib.parse.quote(session_id, safe="")
    return common.request_json(method, f"/private-sessions/{sid}", timeout=15)


def _server_private(session_id: str) -> bool:
    return bool(_server("GET", session_id).get("private"))


def _fail(msg: str) -> int:
    print(f"private mode FAILED: {msg}", file=sys.stderr)
    print("This session is NOT off the record. Do not tell the user it is.", file=sys.stderr)
    return 1


def _marker_removed_note() -> None:
    """Last line of every failed --forget. _off removes the marker before it sends the
    DELETE, so a --forget that does not finish leaves the session as a plain `off` does:
    the Stop hook sends its turns again, and the server drops them while the row is there."""
    print(
        "The local marker was removed. While the server row is kept, this session is in the "
        "same state as after a plain `off`: its turns are sent and the server discards them.",
        file=sys.stderr,
    )


def _forget_failed(
    msg: str,
    closing: str = "The server row was not deleted; the session stays off the record.",
) -> int:
    """A --forget that left the server row in place: the session stays off the record, so
    _fail's closing line ("NOT off the record") would say the opposite. ``closing`` is
    replaced when the DELETE may still be in flight and the row's fate is not settled."""
    print(f"private mode FAILED: {msg}", file=sys.stderr)
    print(closing, file=sys.stderr)
    _marker_removed_note()
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
    delete_error = ""
    delete_answered = False  # the server replied with an HTTP status, so it is done with it
    try:
        _server("DELETE", session_id)
    except (urllib.error.URLError, OSError, ValueError) as e:
        if isinstance(e, urllib.error.HTTPError) and e.code == 403:
            return _forget_failed(
                "taking a session back on the record must be done from a full-trust device."
            )
        # Not a clear refusal: a timeout or a 5xx may come after the server committed the
        # delete, so say nothing about the row until it has been read back.
        delete_error = f"{type(e).__name__}: {str(e)[:160]}"
        delete_answered = isinstance(e, urllib.error.HTTPError)
    try:
        still_private = _server_private(session_id)
    except (urllib.error.URLError, OSError, ValueError) as e:
        # The DELETE went out but its effect could not be read back: state unknown, so
        # neither "deleted" nor "stays off the record" can be claimed.
        print(
            f"private mode FAILED: could not read back the server row after DELETE "
            f"({type(e).__name__}: {str(e)[:160]}). Check with: private_mode.py status {session_id}",
            file=sys.stderr,
        )
        _marker_removed_note()
        return 1
    if still_private:
        if delete_error and not delete_answered:
            # The request may not have reached the server, or may still be working through
            # it: the row was there at read-back and the DELETE can yet remove it.
            return _forget_failed(
                f"server DELETE did not finish ({delete_error}) and the row was still "
                "present when read back.",
                "The DELETE may still be applying on the server. "
                f"Check with: private_mode.py status {session_id}",
            )
        return _forget_failed(
            f"server DELETE failed ({delete_error})."
            if delete_error
            else "server still reports the session as private after DELETE."
        )
    if delete_error:
        print(
            f"private mode: the DELETE reported an error ({delete_error}), but the server "
            "no longer lists the session as private.",
            file=sys.stderr,
        )
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
    if isinstance(session_id, str) and session_id and _plain_name(session_id):
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
    if not session_id or not _plain_name(session_id):
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
