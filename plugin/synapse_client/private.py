"""Whether a session is off the record, from its local marker file.

Private mode is a marker at ``<private_dir>/<session_id>``, written by the private-mode
command alongside the durable server-side flag. This local check is the fast half: it
keeps the turns from ever leaving the machine, and it works while the server is down.
A port of plugin/scripts/ingest_hook.py's ``_is_private``.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable

from .settings import Settings

DEFAULT_TTL_HOURS = 12.0


def _ttl_hours(settings: Settings) -> float:
    try:
        return float(settings.flag("SYNAPSE_PRIVATE_TTL_HOURS", str(DEFAULT_TTL_HOURS)))
    except ValueError:
        return DEFAULT_TTL_HOURS


def _say(log: Callable[[str], None] | None, line: str) -> None:
    if log is None:
        return
    try:
        log(line)
    except Exception:
        pass


def _plain_name(session_id: str) -> bool:
    """One ordinary path component: no separator, not ``.``/``..``, no NUL."""
    seps = {"/", "\\", os.sep, os.altsep or "/"}
    return (
        session_id not in (".", "..")
        and "\0" not in session_id
        and not any(sep in session_id for sep in seps)
    )


def is_private(
    settings: Settings,
    session_id: str,
    log: Callable[[str], None] | None = None,
    *,
    now: float | None = None,
) -> bool:
    """True when ``session_id`` is in private mode: ship NOTHING for it. Never raises.

    - No marker (or no private dir): not private.
    - A marker older than ``SYNAPSE_PRIVATE_TTL_HOURS`` (default 12): not private, and
      the marker is deleted. A marker outlives its session when the session-end step
      crashes; the server-side row is the durable record, the marker only a live flag.
    - A marker that cannot be read (permissions, I/O): PRIVATE. If a session cannot be
      shown not to be private, capturing it is the failure that matters; a deferred turn
      is not.
    - A session id that is not a single plain file name (``../x``, ``a/b``): not private,
      and nothing is touched. No marker can be named that, and the hook's version would
      join it onto the private dir and could delete an unrelated file older than the TTL.

    ``log`` receives one line per event worth keeping (same wording as the hook)."""
    if not session_id:
        return False
    if not _plain_name(session_id):
        _say(log, f"PRIVATE ignored a session id that is not a file name: {session_id[:80]!r}")
        return False
    marker = os.path.join(str(settings.private_dir), session_id)
    try:
        age = (time.time() if now is None else now) - os.stat(marker).st_mtime
    except FileNotFoundError:
        return False
    except OSError as err:
        _say(log, f"PRIVATE unreadable {session_id}: {type(err).__name__} — skipping ingestion")
        return True
    if age > _ttl_hours(settings) * 3600:
        try:
            os.unlink(marker)
        except OSError:
            pass
        _say(log, f"PRIVATE stale marker removed {session_id} (age {int(age)}s)")
        return False
    return True
