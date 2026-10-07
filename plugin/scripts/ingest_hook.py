#!/usr/bin/env python3
# mypy: ignore-errors
"""Claude Code `Stop` hook → push the session transcript to Synapse /ingest.

Memory-only, Logfire-free. On every turn completion Claude Code runs this with
the hook payload on stdin ({transcript_path, session_id, cwd, ...}). We ship the
transcript records past a durable per-file CURSOR to the Synapse `/ingest`
endpoint, which parses them with the same JSONLParser the bulk backfill uses and
dedups by span_id — so every path converges idempotently.

The cursor (DATA_DIR/ingest_cursors.json) advances only after a successful POST,
so a failed POST self-heals: the next Stop hook — or the SessionStart catch-up
sweep — re-ships from the same byte offset. The jsonl on disk is the durable
store; the cursor is the only extra state. This replaces the never-deployed
"disk-sweep backstop on a timer": there is no timer, the plugin's own hook
lifecycle is the scheduler.

Cursor semantics:
  * offset = byte position of the START of the last shipped turn, never EOF —
    the last turn may still be mid-flight at Stop time, so it is re-shipped on
    the next run and lands as a span_id no-op once complete;
  * size   = file EOF at the last FULLY successful run; when the current EOF
    matches, there is nothing new and the file is skipped without a POST;
  * a cursor pointing past EOF means the transcript was truncated/rewritten —
    reset to 0 and re-ship (the server skips every already-stored turn).

SessionStart runs `--catchup`: a detached sweep over the projects dir that ships
any recently-modified transcript whose cursor lags its size — sessions that died
without a Stop hook, or turns dropped while the server was unreachable.

Failed uploads are retried however old they are. A POST that fails (server
down, 401/403 from a revoked token or a proxy/WAF, timeout) marks the file's
cursor entry `pending`, with the time and a short reason; the next successful
POST for that file clears it. The sweep takes every lagging file modified
within SYNAPSE_INGEST_CATCHUP_DAYS, plus every pending or half-shipped file
regardless of age, and pending entries are exempt from the cursor TTL. An
outage longer than the window therefore no longer loses the history it
covered. A file the hook never attempted that is older than the window
(history from before the plugin was installed) is still left alone: importing
that is `synapse-import`'s job.

The sweep visits the least recently touched file first (its last failed
attempt, else its mtime), so repeated capped sweeps make oldest-first progress
and a file that keeps failing goes to the back of the queue instead of holding
a slot.

The cursor file also keeps one `_health` record of the upload streak. After
_WARN_AFTER_FAILS failed POSTs in a row, or _WARN_AFTER_HOURS of failure, with
no success since, the SessionStart parent prints one line, shown to the user
and to the model: "Synapse uploads have been failing since <date> (<reason>);
transcripts will be retried automatically." Reasons come from a short fixed
vocabulary (HTTP status, timeout, connection refused, ...) and never carry the
token.

Design constraints:
  * NEVER block or fail the turn. Claude waits for Stop hooks to exit, so the
    actual HTTP work is done in a DETACHED child (start_new_session) and the
    parent returns instantly. The parent always exits 0.
  * No third-party deps — uses urllib so it runs under whatever Python the CLI
    environment provides.
  * POSTs are chunked at turn boundaries (TAIL_RECORDS records per chunk) so a
    large backlog never ships a fragment that would mint a bogus span_id for a
    turn a later chunk completes.

The endpoint and bearer token resolve through scripts/config.py, same as every
other hook: explicit env var → plugin userConfig → the `/plugin install` answers
persisted in settings.json → default (http://localhost:8765/ingest). The legacy
SYNAPSE_INGEST_URL / SYNAPSE_INGEST_TOKEN env vars still win when set, per
config.py's precedence. The detached child re-imports config, so the same
resolution applies there.

Env (all optional):
  SYNAPSE_INGEST_LOG           default /tmp/synapse-ingest-hook.log
  SYNAPSE_INGEST_TIMEOUT       default 30 (seconds per POST)
  SYNAPSE_INGEST_TAIL          default 400 (records per POST chunk; also the
                               first-run seed window on the Stop path)
  SYNAPSE_INGEST_CATCHUP_DAYS  default 3 (the sweep picks up any lagging file
                               modified this recently; a file whose upload
                               failed is retried regardless of age)
  SYNAPSE_INGEST_ACTIVE_GRACE  default 300 (skip files modified this recently —
                               a live session's own Stop hooks handle them)
  SYNAPSE_INGEST_CATCHUP_MAX   default 20 (files per sweep; the rest defer to
                               the next SessionStart, logged, never silent)
"""

from __future__ import annotations

import glob
import http.client
import json
import os
import socket
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from synapse_filelock import lock_exclusive

INGEST_URL = config.INGEST_URL
INGEST_TOKEN = config.INGEST_TOKEN  # bearer for hosted/central endpoints
LOG_PATH = os.environ.get("SYNAPSE_INGEST_LOG", "/tmp/synapse-ingest-hook.log")
TIMEOUT = float(os.environ.get("SYNAPSE_INGEST_TIMEOUT", "30"))
TAIL_RECORDS = int(os.environ.get("SYNAPSE_INGEST_TAIL", "400"))
CATCHUP_DAYS = float(os.environ.get("SYNAPSE_INGEST_CATCHUP_DAYS", "3"))
ACTIVE_GRACE = float(os.environ.get("SYNAPSE_INGEST_ACTIVE_GRACE", "300"))
CATCHUP_MAX_FILES = int(os.environ.get("SYNAPSE_INGEST_CATCHUP_MAX", "20"))
CURSOR_PATH = config.DATA_DIR / "ingest_cursors.json"
_CURSOR_TTL_DAYS = 45  # drop state for transcripts idle this long (or deleted); pending ones stay
# The upload-streak record in the cursor file. Transcript keys are absolute paths,
# so this key can't collide with one.
_HEALTH_KEY = "_health"
_WARN_AFTER_FAILS = 3  # failed POSTs in a row with no success since → warn at session start
_WARN_AFTER_HOURS = 24.0  # ...or a failure this old with no success since
PRIVATE_DIR = config.PRIVATE_DIR  # marker file per off-the-record session (private mode)
_PRIVATE_TTL_HOURS = float(os.environ.get("SYNAPSE_PRIVATE_TTL_HOURS", "12"))

# Mirror of ingestion.jsonl_client._MACHINERY_PREFIXES — kept inline so the hook
# stays dependency-free (it runs under the CLI's bare Python, off the repo path).
_MACHINERY_PREFIXES = (
    "<command-name>",
    "<command-message>",
    "<command-args>",
    "<local-command-stdout>",
    "<local-command-stderr>",
    "<local-command-caveat>",
    "<system-reminder>",
)


def _log(msg: str) -> None:
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now(UTC).isoformat()} {msg}\n")
    except Exception:
        pass


def _is_turn_start(rec: dict[str, Any]) -> bool:
    """True if rec is a fresh human turn — a safe place to begin a slice.

    Mirrors ingestion.jsonl_client._is_user_turn. A slice must start at a turn
    boundary; otherwise its leading records are a fragment of an already-ingested
    turn (the server skips it by span_id, but trimming keeps the POST clean).
    """
    if rec.get("type") != "user":
        return False
    content = (rec.get("message") or {}).get("content")
    if isinstance(content, str):
        t = content.strip()
        return bool(t) and not t.startswith(_MACHINERY_PREFIXES)
    if isinstance(content, list):
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                t = str(b.get("text") or b.get("content") or "").strip()
                if t and not t.startswith(_MACHINERY_PREFIXES):
                    return True
    return False


def _parse_lines(lines: list[bytes]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in lines:
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _select_tail(raw_lines: list[bytes]) -> tuple[list[dict[str, Any]], str]:
    """Pick the records for a first-encounter seed: a turn-boundary-aligned tail,
    or the full file when the tail window holds no boundary (one mega-turn).
    Returns (records, mode). Pure (no I/O) so it's unit-testable.

    Kept for the Stop path's cursor seeding: a long-lived pre-cursor session was
    already shipped turn-by-turn by earlier hook fires, so re-parsing the whole
    file wholesale on the first cursored run would be pure churn.
    """
    tail = _parse_lines(raw_lines[-TAIL_RECORDS:])
    start = next((i for i, r in enumerate(tail) if _is_turn_start(r)), None)
    if start is not None:
        return tail[start:], "tail"
    return _parse_lines(raw_lines), "full-fallback"


# ---------------------------------------------------------------------------
# Cursor planning (pure) + cursor store (flocked)
# ---------------------------------------------------------------------------


def _read_offset_records(path: str) -> tuple[list[tuple[int, dict[str, Any]]], int]:
    """Read a transcript as [(byte_offset, record), ...] plus the EOF offset.

    Offsets are line starts, so any offset this returns is a valid cursor.
    Unparseable/blank lines are skipped (their bytes still count toward EOF).
    """
    out: list[tuple[int, dict[str, Any]]] = []
    eof = 0
    with open(path, "rb") as f:
        while True:
            off = f.tell()
            line = f.readline()
            if not line:
                eof = off
                break
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except Exception:
                continue
            if isinstance(rec, dict):
                out.append((off, rec))
    return out, eof


def _plan_from_cursor(
    offset_recs: list[tuple[int, dict[str, Any]]],
    cursor: int,
    chunk_records: int,
) -> list[tuple[list[dict[str, Any]], int]]:
    """Slice (byte_offset, record) pairs at/after `cursor` into POST chunks.

    Returns [(records, cursor_after), ...]; cursor_after is safe to persist once
    that chunk's POST succeeds. Invariants:
      * chunks split only at turn boundaries — a chunk never ends mid-turn,
        which would mint a bogus span_id for a turn a later chunk completes;
      * the FINAL chunk's cursor_after is the offset of the last turn-start:
        that turn may still be growing, so it re-ships next run (span_id no-op
        once complete) instead of leaving a half-turn stranded behind EOF;
      * no turn boundary at/after the cursor → one chunk, cursor_after stays at
        `cursor`, and the mega-turn keeps re-shipping until a boundary appears
        (the server dedups every repeat).

    Pure (no I/O) so it's unit-testable without a live server.
    """
    recs = [(o, r) for o, r in offset_recs if o >= cursor]
    if not recs:
        return []
    starts = [i for i, (_, r) in enumerate(recs) if _is_turn_start(r)]
    if not starts:
        return [([r for _, r in recs], cursor)]

    plans: list[tuple[list[dict[str, Any]], int]] = []
    chunk: list[dict[str, Any]] = []
    for i, s in enumerate(starts):
        lo = 0 if i == 0 else s  # glue any pre-boundary prefix onto the first turn
        hi = starts[i + 1] if i + 1 < len(starts) else len(recs)
        seg = recs[lo:hi]
        if chunk and len(chunk) + len(seg) > chunk_records:
            plans.append((chunk, recs[s][0]))  # next chunk begins at this turn
            chunk = []
        chunk.extend(r for _, r in seg)
    plans.append((chunk, recs[starts[-1]][0]))
    return plans


def _load_state() -> dict[str, Any]:
    try:
        with open(CURSOR_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _needs_retry(ent: Any) -> bool:
    """True if this file's upload was attempted and did not complete: a failed
    POST left a `pending` marker, or a chunked backlog stopped partway (`size`
    -1). Such a file is swept regardless of age and is never TTL-pruned."""
    return isinstance(ent, dict) and (bool(ent.get("pending")) or ent.get("size") == -1)


def _update_state(path: str, apply: Callable[[dict[str, Any], float], None]) -> None:
    """Load, mutate (`apply(state, now)`), prune and atomically rewrite the cursor
    file under its exclusive flock. Pruning drops entries for deleted transcripts
    and for ones idle past _CURSOR_TTL_DAYS, except a file still owed a retry,
    which stays until it ships or disappears."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(str(CURSOR_PATH) + ".lock", "w") as lf:
        lock_exclusive(lf)
        state = _load_state()
        now = time.time()
        apply(state, now)
        cutoff = now - _CURSOR_TTL_DAYS * 86400
        state = {
            p: e
            for p, e in state.items()
            if p in (path, _HEALTH_KEY)
            or (
                isinstance(e, dict)
                and (e.get("ts", 0) >= cutoff or _needs_retry(e))
                and os.path.exists(p)
            )
        }
        tmp = str(CURSOR_PATH) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)
        os.replace(tmp, CURSOR_PATH)


def _advance_cursor(path: str, offset: int, size: int, *, force: bool = False) -> None:
    """Persist a cursor under an exclusive flock, monotonic per file unless
    `force` (truncation reset). Concurrent shippers (a Stop child racing the
    catch-up sweep) serialize here and can't rewind each other; both re-shipping
    the same span is a server-side no-op, so the worst race costs one dup POST.

    `size` is the EOF of a FULLY shipped file; pass -1 for intermediate chunks
    so a crash mid-backlog can't fake the everything-shipped skip condition.

    Only ever called after a successful POST, so it also ends the upload-failure
    streak, and rewriting the entry clears the file's `pending` marker.
    """

    def apply(state: dict[str, Any], now: float) -> None:
        state[_HEALTH_KEY] = {"last_ok": now}
        ent = state.get(path)
        if not force and isinstance(ent, dict) and int(ent.get("offset", -1)) > offset:
            return  # a racer already got further (and cleared any marker)
        state[path] = {"offset": offset, "size": size, "ts": now}

    _update_state(path, apply)


def _record_failure(path: str, reason: str, offset: int) -> None:
    """Mark `path` pending after a failed POST and extend the upload-failure
    streak. The cursor never moves here: an existing entry keeps its offset, and
    a file with no entry yet gets one at `offset` (where this run started) with
    size -1. `pending` holds the time of the first failure since the file last
    shipped; `tried` is bumped on every failure, which moves a file that keeps
    failing to the back of the sweep queue."""
    reason = _scrub(reason)

    def apply(state: dict[str, Any], now: float) -> None:
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
        h = state.get(_HEALTH_KEY)
        h = dict(h) if isinstance(h, dict) else {}
        h.update(
            fail_since=h.get("fail_since") or now,
            fails=int(h.get("fails", 0)) + 1,
            last_error=reason,
            last_fail=now,
        )
        state[_HEALTH_KEY] = h

    _update_state(path, apply)


# ---------------------------------------------------------------------------
# Upload-failure reasons + the session-start warning
# ---------------------------------------------------------------------------


def _scrub(text: str) -> str:
    """Last line of defence for anything stored or printed: never the token, never long."""
    if INGEST_TOKEN:
        text = text.replace(INGEST_TOKEN, "***")
    return text[:80]


def _short_reason(e: BaseException) -> str:
    """A short, token-free reason for a failed POST, from a fixed vocabulary.

    Built from the exception type and the HTTP status code only. Message text
    from the server or the library is never used, since it can echo URLs,
    headers or response bodies."""
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


def _failure_warning(state: dict[str, Any], now: float) -> str | None:
    """The session-start line for a sustained upload failure, or None.

    Sustained means _WARN_AFTER_FAILS failed POSTs in a row, or a first failure
    _WARN_AFTER_HOURS old, with no successful POST since. Any success resets it."""
    h = state.get(_HEALTH_KEY)
    if not isinstance(h, dict) or not h.get("fail_since"):
        return None
    since = float(h["fail_since"])
    if int(h.get("fails", 0)) < _WARN_AFTER_FAILS and now - since < _WARN_AFTER_HOURS * 3600:
        return None
    when = datetime.fromtimestamp(since).strftime("%Y-%m-%d %H:%M")
    reason = _scrub(str(h.get("last_error") or "unknown error"))
    return (
        f"Synapse uploads have been failing since {when} ({reason}); "
        "transcripts will be retried automatically."
    )


def _emit_failure_warning() -> None:
    """Print the sustained-failure line at SessionStart: `systemMessage` shows it
    to the user, `additionalContext` to the model. Reads local state only (no
    network), prints nothing while uploads are healthy, and never raises."""
    try:
        line = _failure_warning(_load_state(), time.time())
        if line:
            print(
                json.dumps(
                    {
                        "systemMessage": line,
                        "hookSpecificOutput": {
                            "hookEventName": "SessionStart",
                            "additionalContext": line,
                        },
                    }
                )
            )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Shipping
# ---------------------------------------------------------------------------


def _is_private(session_id: str) -> bool:
    """True if `session_id` is in private mode → ship NOTHING for it.

    Private mode is a marker file at PRIVATE_DIR/<session_id>, written by
    private_mode.py (which also sets the durable server-side flag). This local check
    is the fast half: it keeps the turns from ever leaving the machine, and it works
    while the server is unreachable.

    Two deliberate behaviours:
      * STALE markers (older than _PRIVATE_TTL_HOURS) are ignored AND deleted. A
        marker outlives its session — a crashed SessionEnd hook leaves one behind, and
        Claude Code reuses no session id, so an orphan would otherwise sit there
        forever. The server-side row is the durable record; the marker is a live flag.
      * UNREADABLE marker dir → treated as PRIVATE. If we can't prove a session is not
        private, silently capturing it is the unacceptable failure; a deferred turn is
        not (the cursor doesn't advance, so nothing is lost if the skip was wrong).
    """
    if not session_id:
        return False
    marker = os.path.join(str(PRIVATE_DIR), session_id)
    try:
        age = time.time() - os.stat(marker).st_mtime
    except FileNotFoundError:
        return False  # no marker (or no private dir at all) — normal case
    except OSError as e:
        _log(f"PRIVATE unreadable {session_id}: {type(e).__name__} — skipping ingestion")
        return True
    if age > _PRIVATE_TTL_HOURS * 3600:
        try:
            os.unlink(marker)
        except OSError:
            pass
        _log(f"PRIVATE stale marker removed {session_id} (age {int(age)}s)")
        return False
    return True


def _post_records(records: list[dict[str, Any]], source: str = "hook") -> str:
    """POST a batch of raw transcript records to /ingest; returns the (truncated)
    response body. Shared by the Stop-hook shipper, the catch-up sweep, and the
    bulk history import (import_history.py), so all resolve URL/token identically
    through config.py. Raises on transport/HTTP errors — callers decide how to
    fail-soft."""
    body = json.dumps({"records": records, "source": source}).encode()
    # A real User-Agent, same as config.post_json: Cloudflare in front of a hosted server
    # 403s urllib's default "Python-urllib/x" ("error code: 1010"), and this hook fails
    # soft, so without it every ingest is silently dropped.
    headers = {"Content-Type": "application/json", "User-Agent": config._UA}
    if INGEST_TOKEN:
        headers["Authorization"] = f"Bearer {INGEST_TOKEN}"
    req = urllib.request.Request(INGEST_URL, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:  # user-configured Synapse URL
        return resp.read().decode()[:200]


def _ship(transcript_path: str, *, mode: str = "stop") -> tuple[int, int]:
    """Ship everything past the file's cursor to /ingest; returns (posts, records).

    stop mode seeds a missing cursor from the bounded tail (the pre-cursor hook
    already shipped that session turn-by-turn); catchup mode ships from byte 0 —
    recovering never-shipped files is its entire point. A failed POST leaves the
    cursor at the failure point, so the next Stop or sweep resumes there.

    Private mode short-circuits everything: a session with a live marker file posts
    nothing, on either path (the catch-up sweep calls this per file, so it inherits
    the same check per session). The cursor is left alone, so if the marker later
    expires the turns are still shippable — the server's private_sessions row, not
    this cursor, is what keeps them out of memory for good.
    """
    # Cheap check first: the transcript's own name is the session id.
    session_id = os.path.splitext(os.path.basename(transcript_path))[0]
    if _is_private(session_id):
        _log(f"PRIVATE skip {mode} {os.path.basename(transcript_path)}")
        return (0, 0)
    try:
        offset_recs, eof = _read_offset_records(transcript_path)
        if not offset_recs:
            return (0, 0)

        # Belt and braces: a resumed/compacted transcript can carry records whose
        # sessionId differs from the filename, and it's the record's id that the
        # server stores (and that private_mode.py marked).
        for sid in {str(r.get("sessionId") or "") for _, r in offset_recs} - {"", session_id}:
            if _is_private(sid):
                _log(f"PRIVATE skip {mode} {os.path.basename(transcript_path)} (record sid)")
                return (0, 0)

        ent = _load_state().get(transcript_path)
        cursor: int | None = None
        force = False
        if isinstance(ent, dict):
            cursor = int(ent.get("offset", 0))
            if cursor > eof:  # transcript truncated/rewritten — start over
                cursor, force = None, True
            elif int(ent.get("size", -1)) == eof:
                return (0, 0)  # nothing new since the last fully-shipped run
        if cursor is None:
            if mode == "stop":
                tail = offset_recs[-TAIL_RECORDS:]
                i = next((j for j, (_, r) in enumerate(tail) if _is_turn_start(r)), None)
                cursor = tail[i][0] if i is not None else 0
            else:
                cursor = 0

        plans = _plan_from_cursor(offset_recs, cursor, TAIL_RECORDS)
    except Exception as e:
        _log(f"ERR {mode} {transcript_path}: {type(e).__name__}: {str(e)[:160]}")
        return (0, 0)

    posts = shipped = 0
    chunk_start = cursor
    for n, (records, cursor_after) in enumerate(plans):
        if not records:
            continue
        try:
            payload = _post_records(records, source="hook" if mode == "stop" else "sweep")
        except Exception as e:
            # Cursor stays at the failure point — the next Stop or the
            # SessionStart sweep resumes exactly there. Earlier chunks stand.
            # The pending marker is what lets the sweep retry this file after
            # it ages out of the catch-up window.
            _log(f"ERR {mode} {transcript_path}: {type(e).__name__}: {str(e)[:160]}")
            try:
                _record_failure(transcript_path, _short_reason(e), chunk_start)
            except Exception as e2:
                _log(f"ERR {mode} record-failure: {type(e2).__name__}: {str(e2)[:160]}")
            break
        try:
            final = n == len(plans) - 1
            _advance_cursor(transcript_path, cursor_after, eof if final else -1, force=force)
        except Exception as e:
            _log(f"ERR {mode} {transcript_path}: {type(e).__name__}: {str(e)[:160]}")
            break
        force = False
        chunk_start = cursor_after
        posts += 1
        shipped += len(records)
        _log(
            f"OK {os.path.basename(transcript_path)} {mode} {len(records)} recs "
            f"cur={cursor_after} -> {payload}"
        )
    return (posts, shipped)


# ---------------------------------------------------------------------------
# SessionStart catch-up sweep
# ---------------------------------------------------------------------------


def _catchup_candidates(
    projects_root: str,
    skip_path: str,
    state: dict[str, Any],
    now: float,
) -> list[str]:
    """Transcripts worth sweeping: not the live session, not mid-write
    (ACTIVE_GRACE — an active session's own Stop hooks own it), with bytes past
    their cursor, and either modified within CATCHUP_DAYS or owed a retry (a
    failed or half-finished upload, _needs_retry) however old.

    A file older than the window that the hook never attempted is NOT a
    candidate: that is history from before the plugin was installed (or from
    before failures were recorded), and `synapse-import` is the tool for it.

    Least recently touched first — the last failed attempt, else the mtime — so
    repeated capped sweeps make oldest-first progress and a file that keeps
    failing rotates to the back instead of starving the rest. Pure given
    os.stat results — patchable in tests."""
    out: list[tuple[float, str]] = []
    skip_real = os.path.realpath(skip_path) if skip_path else ""
    for path in glob.glob(os.path.join(projects_root, "*", "*.jsonl")):
        if skip_real and os.path.realpath(path) == skip_real:
            continue
        try:
            st = os.stat(path)
        except OSError:
            continue
        ent = state.get(path)
        if st.st_mtime < now - CATCHUP_DAYS * 86400 and not _needs_retry(ent):
            continue
        if st.st_mtime > now - ACTIVE_GRACE:
            continue
        if isinstance(ent, dict) and int(ent.get("size", -1)) == st.st_size:
            continue  # fully shipped and unchanged
        tried = float(ent.get("tried") or 0) if isinstance(ent, dict) else 0.0
        out.append((max(st.st_mtime, tried), path))
    out.sort()
    return [p for _, p in out]


def _catchup(projects_root: str, skip_path: str) -> None:
    """Sweep the projects dir and ship every lagging transcript. Runs detached;
    a non-blocking flock makes concurrent session starts collapse to one sweep."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.DATA_DIR / "ingest_catchup.lock", "w") as lf:
        try:
            lock_exclusive(lf, blocking=False)
        except OSError:
            _log("CATCHUP skipped: another sweep is running")
            return
        candidates = _catchup_candidates(projects_root, skip_path, _load_state(), time.time())
        deferred = max(0, len(candidates) - CATCHUP_MAX_FILES)
        files = posts = recs = 0
        for path in candidates[:CATCHUP_MAX_FILES]:
            p, r = _ship(path, mode="catchup")
            files += 1 if r else 0
            posts += p
            recs += r
        _log(
            f"CATCHUP {projects_root}: {len(candidates[:CATCHUP_MAX_FILES])} checked, "
            f"{files} shipped ({recs} recs, {posts} posts)"
            + (f", {deferred} deferred to next session start" if deferred else "")
        )


def _projects_root_from(transcript_path: str) -> str:
    """~/.claude/projects/<flattened-cwd>/<session>.jsonl → ~/.claude/projects"""
    return os.path.dirname(os.path.dirname(os.path.abspath(transcript_path)))


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


def _spawn_detached(args: list[str]) -> None:
    subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), *args],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> None:
    # Detached children do the actual work.
    if len(sys.argv) >= 3 and sys.argv[1] == "--ship":
        _ship(sys.argv[2], mode="stop")
        return
    if len(sys.argv) >= 3 and sys.argv[1] == "--catchup-run":
        _catchup(sys.argv[2], sys.argv[3] if len(sys.argv) >= 4 else "")
        return

    # Parent (invoked by Claude): read the hook payload, spawn a detached
    # worker, and return immediately so the turn/session isn't delayed.
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return
    transcript_path = payload.get("transcript_path") or ""

    if len(sys.argv) >= 2 and sys.argv[1] == "--catchup":
        # Say so when uploads keep failing. Before the sweep runs, so the line
        # reflects the streak as it stood; a sweep that succeeds ends the streak.
        _emit_failure_warning()
        root = (
            _projects_root_from(transcript_path)
            if transcript_path
            else os.path.join(
                os.path.expanduser(os.environ.get("CLAUDE_CONFIG_DIR", "~/.claude")), "projects"
            )
        )
        if not os.path.isdir(root):
            return
        try:
            _spawn_detached(["--catchup-run", root, transcript_path])
        except Exception as e:
            _log(f"SPAWN-ERR catchup: {type(e).__name__}: {str(e)[:160]}")
        return

    # Default: the Stop-hook shipper.
    if not transcript_path or not os.path.exists(transcript_path):
        return
    try:
        _spawn_detached(["--ship", transcript_path])
    except Exception as e:
        _log(f"SPAWN-ERR: {type(e).__name__}: {str(e)[:160]}")


if __name__ == "__main__":
    try:
        main()
    finally:
        # A Stop/SessionStart hook must never fail the turn.
        sys.exit(0)
