"""Failed-upload bookkeeping shared by both transcript shippers.

Two hooks ship a session transcript to /ingest past a per-file byte cursor:
``ingest_hook.py`` here (Claude Code) and ``plugin-codex/hooks/synapse_stop_hook.py``
(Codex), which imports this file from the same checkout. Each keeps its own cursor file
and its own cursor semantics. This module is the part they share, which is what happens
when an upload fails:

  * The failed file's cursor entry is marked ``pending``: the time of the first failure
    since the file last shipped, the latest attempt (``tried``), a count and a short
    reason. One ``_health`` record in the same file counts the failure streak. The next
    successful POST for the file rewrites its entry, which clears the marker, and any
    successful POST ends the streak.
  * The session-start catch-up sweep takes every lagging file modified within its
    window, plus every file owed a retry (``needs_retry``) however old, least recently
    touched first. An old file the hook never attempted (history from before install)
    is not swept; ``synapse-import`` covers it.
  * Entries owed a retry are exempt from the cursor TTL.
  * After WARN_AFTER_FAILS failed POSTs in a row, or WARN_AFTER_HOURS since the first,
    with no success since, session start prints one line saying so.

Reasons come from a short fixed vocabulary and are scrubbed of the bearer token before
they are stored or printed.

Dependency-free and config-free: the caller passes in the cursor path, token, window and
TTL, so this loads under either host's bare python3. Named with the ``synapse_`` prefix
so it can't shadow an installed package (see synapse_filelock.py).
"""

from __future__ import annotations

import http
import http.client
import json
import os
import socket
import ssl
import time
import urllib.error
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Any

from synapse_filelock import lock_exclusive

# The upload-streak record in a cursor file. Transcript keys are absolute paths, so
# this key can't collide with one.
HEALTH_KEY = "_health"
WARN_AFTER_FAILS = 3  # failed POSTs in a row with no success since → warn at session start
WARN_AFTER_HOURS = 24.0  # ...or a first failure this old with no success since

State = dict[str, Any]


# ---------------------------------------------------------------------------
# Cursor file: load, locked update with TTL pruning
# ---------------------------------------------------------------------------


def load_state(cursor_path: str | os.PathLike[str]) -> State:
    try:
        with open(cursor_path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def needs_retry(ent: Any) -> bool:
    """True if this file's upload was attempted and did not complete: a failed POST
    left a ``pending`` marker, or a chunked backlog stopped partway (``size`` -1).
    Such a file is swept regardless of age and is never TTL-pruned."""
    return isinstance(ent, dict) and (bool(ent.get("pending")) or ent.get("size") == -1)


def update_state(
    cursor_path: str | os.PathLike[str],
    path: str,
    apply: Callable[[State, float], None],
    *,
    ttl_days: float,
) -> None:
    """Load, mutate (``apply(state, now)``), prune and atomically rewrite the cursor
    file under its exclusive flock. Pruning drops entries for deleted transcripts and
    for ones idle past ``ttl_days``, except a file still owed a retry, which stays
    until it ships or disappears. ``path`` (the entry being written) is always kept."""
    cursor_path = os.fspath(cursor_path)
    os.makedirs(os.path.dirname(cursor_path) or ".", exist_ok=True)
    with open(cursor_path + ".lock", "w") as lf:
        lock_exclusive(lf)
        state = load_state(cursor_path)
        now = time.time()
        apply(state, now)
        cutoff = now - ttl_days * 86400
        state = {
            p: e
            for p, e in state.items()
            if p in (path, HEALTH_KEY)
            or (
                isinstance(e, dict)
                and (e.get("ts", 0) >= cutoff or needs_retry(e))
                and os.path.exists(p)
            )
        }
        tmp = cursor_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)
        os.replace(tmp, cursor_path)


def mark_ok(state: State, now: float) -> None:
    """A POST succeeded: end the failure streak. Call from inside an ``apply``; the
    caller rewrites the file's own entry, which clears its ``pending`` marker."""
    state[HEALTH_KEY] = {"last_ok": now}


def mark_failed(state: State, path: str, reason: str, offset: int, now: float) -> None:
    """Mark ``path`` pending after a failed POST and extend the failure streak.

    The cursor never moves here: an existing entry keeps its offset, and a file with no
    entry yet gets one at ``offset`` (where this run started) with size -1. ``pending``
    holds the time of the first failure since the file last shipped; ``tried`` moves on
    every failure, which sends a file that keeps failing to the back of the sweep queue.
    ``reason`` must already be scrubbed (``record_failure`` does that)."""
    old = state.get(path)
    ent = dict(old) if isinstance(old, dict) else {"offset": offset, "size": -1}
    ent.update(
        pending=ent.get("pending") or now,
        tried=now,
        fails=int(ent.get("fails", 0)) + 1,
        last_error=reason,
        ts=now,
    )
    state[path] = ent
    h = state.get(HEALTH_KEY)
    h = dict(h) if isinstance(h, dict) else {}
    h.update(
        fail_since=h.get("fail_since") or now,
        fails=int(h.get("fails", 0)) + 1,
        last_error=reason,
        last_fail=now,
    )
    state[HEALTH_KEY] = h


def record_failure(
    cursor_path: str | os.PathLike[str],
    path: str,
    reason: str,
    offset: int,
    *,
    token: str,
    ttl_days: float,
) -> None:
    """Scrub ``reason`` and persist ``mark_failed`` under the cursor file's lock."""
    reason = scrub(reason, token)
    update_state(
        cursor_path,
        path,
        lambda state, now: mark_failed(state, path, reason, offset, now),
        ttl_days=ttl_days,
    )


# ---------------------------------------------------------------------------
# Catch-up candidates
# ---------------------------------------------------------------------------


def sweep_candidates(
    paths: Iterable[str],
    state: State,
    now: float,
    *,
    window_days: float,
    active_grace: float,
    shipped: Callable[[dict[str, Any], os.stat_result], bool],
    skip_path: str = "",
) -> list[str]:
    """Transcripts worth sweeping, in the order to sweep them.

    A candidate is not the live session (``skip_path``), not mid-write (modified within
    ``active_grace`` seconds; a live session's own Stop hooks own it), not fully shipped
    (``shipped(entry, stat)``, the host's own cursor test), and either modified within
    ``window_days`` or owed a retry (``needs_retry``) however old.

    A file older than the window that the hook never attempted is NOT a candidate: that
    is history from before the plugin was installed, and ``synapse-import`` is the tool
    for it.

    Least recently touched first (the last failed attempt, else the mtime), so repeated
    capped sweeps make oldest-first progress and a file that keeps failing rotates to the
    back instead of holding a slot."""
    out: list[tuple[float, str]] = []
    skip_real = os.path.realpath(skip_path) if skip_path else ""
    for path in paths:
        if skip_real and os.path.realpath(path) == skip_real:
            continue
        try:
            st = os.stat(path)
        except OSError:
            continue
        ent = state.get(path)
        if st.st_mtime < now - window_days * 86400 and not needs_retry(ent):
            continue
        if st.st_mtime > now - active_grace:
            continue
        if isinstance(ent, dict) and shipped(ent, st):
            continue
        tried = float(ent.get("tried") or 0) if isinstance(ent, dict) else 0.0
        out.append((max(st.st_mtime, tried), path))
    out.sort()
    return [p for _, p in out]


# ---------------------------------------------------------------------------
# Upload-failure reasons + the session-start warning
# ---------------------------------------------------------------------------


def scrub(text: str, token: str) -> str:
    """Last line of defence for anything stored or printed: never the token, never long."""
    if token:
        text = text.replace(token, "***")
    return text[:80]


def short_reason(e: BaseException) -> str:
    """A short, token-free reason for a failed POST, from a fixed vocabulary.

    Built from the exception type and the HTTP status code only. Message text from the
    server or the library is never used, since it can echo URLs, headers or response
    bodies."""
    if isinstance(e, urllib.error.HTTPError):
        try:
            return f"HTTP {e.code} {http.HTTPStatus(e.code).phrase}"
        except ValueError:
            return f"HTTP {e.code}"
    cause = e.reason if isinstance(e, urllib.error.URLError) else e
    if isinstance(cause, TimeoutError):  # socket.timeout is an alias
        return "timed out"
    if isinstance(cause, ConnectionRefusedError):
        return "connection refused"
    if isinstance(cause, socket.gaierror):
        return "DNS lookup failed"
    if isinstance(cause, (ssl.SSLError, ssl.CertificateError)):
        return "TLS error"
    if isinstance(cause, ConnectionError):
        return "connection dropped"
    if isinstance(cause, http.client.HTTPException):
        return "bad HTTP response"
    if isinstance(cause, OSError):
        return "network error"
    if cause is not e:
        return "server unreachable"  # URLError with a non-exception reason
    return type(e).__name__


def failure_warning(state: State, now: float, token: str = "") -> str | None:
    """The session-start line for a sustained upload failure, or None.

    Sustained means WARN_AFTER_FAILS failed POSTs in a row, or a first failure
    WARN_AFTER_HOURS old, with no successful POST since. Any success resets it."""
    h = state.get(HEALTH_KEY)
    if not isinstance(h, dict) or not h.get("fail_since"):
        return None
    since = float(h["fail_since"])
    if int(h.get("fails", 0)) < WARN_AFTER_FAILS and now - since < WARN_AFTER_HOURS * 3600:
        return None
    when = datetime.fromtimestamp(since).strftime("%Y-%m-%d %H:%M")
    reason = scrub(str(h.get("last_error") or "unknown error"), token)
    return (
        f"Synapse uploads have been failing since {when} ({reason}); "
        "transcripts will be retried automatically."
    )
