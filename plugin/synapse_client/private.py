"""Whether a session is off the record, from its local marker file.

Private mode is a marker at ``<private_dir>/<session_id>``, written by the private-mode
toggle (private_toggle.py) alongside the durable server-side flag. This local check is
the fast half: it keeps the turns from ever leaving the machine, and it works while the
server is down. ``mark`` and ``unmark`` write and remove the marker.

The check sits where a payload leaves: ``Client.ingest`` runs ``check_payload`` before
its request, and so do the Claude Code and Codex hooks' own senders
(plugin/scripts/ingest_hook.py, plugin-codex/hooks/synapse_stop_hook.py). An adapter that
uploads through the core cannot leave it out.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from .settings import Settings

DEFAULT_TTL_HOURS = 12.0
#: What ``state`` answers: on the record, in private mode, or a marker that cannot be read.
OPEN = "open"
PRIVATE = "private"
UNKNOWN = "unknown"


class PrivateSession(Exception):
    """An upload was refused before any request: ``session_id`` is in private mode, or,
    when ``sure`` is False, its marker could not be read."""

    def __init__(self, session_id: str, *, sure: bool) -> None:
        super().__init__("the session is in private mode: nothing was uploaded")
        self.session_id = session_id
        self.sure = sure


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


def plain_name(session_id: str) -> bool:
    """One ordinary path component: no separator, not ``.``/``..``, no NUL. The id is
    joined onto the private dir, and ``../x`` would name a file outside it."""
    seps = {"/", "\\", os.sep, os.altsep or "/"}
    return (
        session_id not in (".", "..")
        and "\0" not in session_id
        and not any(sep in session_id for sep in seps)
    )


def marker(settings: Settings, session_id: str) -> Path:
    """Where ``session_id``'s marker is kept. RAISES ValueError for an id that is empty
    or not a plain file name: nothing may be written or removed by such a name."""
    if not session_id or not plain_name(session_id):
        raise ValueError(f"not a session id: {session_id[:80]!r}")
    return Path(settings.private_dir) / session_id


def mark(settings: Settings, session_id: str) -> Path:
    """Write ``session_id``'s marker, the local half of private mode, and return its
    path. A marker already there is written again, so its age starts over. RAISES
    OSError, or ValueError for an id ``marker`` refuses."""
    path = marker(settings, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


def unmark(settings: Settings, session_id: str) -> bool:
    """Remove ``session_id``'s marker. True when none is there afterwards (one that was
    never there counts). RAISES ValueError for an id ``marker`` refuses."""
    path = marker(settings, session_id)
    try:
        path.unlink()
    except OSError:
        pass
    return not path.exists()


def state(
    settings: Settings,
    session_id: str,
    log: Callable[[str], None] | None = None,
    *,
    now: float | None = None,
) -> str:
    """OPEN, PRIVATE or UNKNOWN for ``session_id``. Never raises.

    - No marker (or no private dir): OPEN.
    - A marker older than ``SYNAPSE_PRIVATE_TTL_HOURS`` (default 12): OPEN, and the
      marker is deleted. A marker outlives its session when the session-end step
      crashes; the server-side row is the durable record, the marker only a live flag.
    - A marker that cannot be read (permissions, I/O): UNKNOWN. If a session cannot be
      shown not to be private, capturing it is the failure that matters; a deferred turn
      is not. So nothing is sent for it, and nothing queued for it is thrown away.
    - A session id that is not a single plain file name (``../x``, ``a/b``): OPEN, and
      nothing is touched. No marker can be named that, and the hook's version would
      join it onto the private dir and could delete an unrelated file older than the TTL.

    ``log`` receives one line per event worth keeping (same wording as the hook)."""
    if not session_id:
        return OPEN
    if not plain_name(session_id):
        _say(log, f"PRIVATE ignored a session id that is not a file name: {session_id[:80]!r}")
        return OPEN
    marker = os.path.join(str(settings.private_dir), session_id)
    try:
        age = (time.time() if now is None else now) - os.stat(marker).st_mtime
    except FileNotFoundError:
        return OPEN
    except OSError as err:
        _say(log, f"PRIVATE unreadable {session_id}: {type(err).__name__} — skipping ingestion")
        return UNKNOWN
    if age > _ttl_hours(settings) * 3600:
        try:
            os.unlink(marker)
        except OSError:
            pass
        _say(log, f"PRIVATE stale marker removed {session_id} (age {int(age)}s)")
        return OPEN
    return PRIVATE


def is_private(
    settings: Settings,
    session_id: str,
    log: Callable[[str], None] | None = None,
    *,
    now: float | None = None,
) -> bool:
    """True when ``session_id`` is in private mode, or cannot be shown not to be
    (``state``): ship NOTHING for it. Never raises."""
    return state(settings, session_id, log, now=now) != OPEN


def sessions_of(payload: Mapping[str, Any]) -> list[str]:
    """Every session an ingest body names, in the order met: its own ``session_id``
    (the ``messages`` and ``codex`` formats) and each record's ``sessionId`` (Claude
    Code's transcript records, which carry no id beside them)."""
    found: dict[str, None] = {}
    own = payload.get("session_id")
    if isinstance(own, str) and own:
        found[own] = None
    records = payload.get("records")
    for record in records if isinstance(records, list) else ():
        sid = record.get("sessionId") if isinstance(record, Mapping) else None
        if isinstance(sid, str) and sid:
            found[sid] = None
    return list(found)


def check_payload(
    settings: Settings,
    payload: Mapping[str, Any],
    sessions: Iterable[str] = (),
    log: Callable[[str], None] | None = None,
) -> None:
    """RAISES PrivateSession when an ingest body must not be sent: a session it names
    (``sessions_of``), or one of ``sessions`` (ids it belongs to and does not carry, a
    transcript's file name), is in private mode, or has a marker that cannot be read.
    One session in private mode refuses the whole body, as the hooks always have."""
    unknown = ""
    named = [sid for sid in sessions if isinstance(sid, str) and sid]
    for session_id in dict.fromkeys([*named, *sessions_of(payload)]):
        found = state(settings, session_id, log)
        if found == PRIVATE:
            raise PrivateSession(session_id, sure=True)
        if found == UNKNOWN and not unknown:
            unknown = session_id
    if unknown:
        raise PrivateSession(unknown, sure=False)
