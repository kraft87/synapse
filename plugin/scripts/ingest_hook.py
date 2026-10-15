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
    the transcript can lag the hook (Claude Code may write a turn's final
    message after Stop fires), so the last turn is re-shipped on the next run.
    A turn that did not grow is a span_id no-op; one that did lands as a new
    episode under its new span_id (the turn's LAST uuid), beside the shorter
    episode it grew from;
  * size   = file EOF at the last FULLY successful run; when the current EOF
    matches, there is nothing new and the file is skipped without a POST;
  * a cursor pointing past EOF means the transcript was truncated/rewritten —
    reset to 0 and re-ship (the server skips every already-stored turn).

SessionStart runs `--catchup`: a detached sweep over the projects dir that ships
any recently-modified transcript whose cursor lags its size — sessions that died
without a Stop hook, or turns dropped while the server was unreachable. After a /clear
or an auto-compaction (SessionStart fires for those too) it only repeats the upload
warning below, which the fresh context window needs again; the sweep has already run.

Failed uploads are retried however old they are. A POST that fails (server
down, 401/403 from a revoked token or a proxy/WAF, timeout) marks the file's
cursor entry `pending`, with the time and a short reason; the next successful
POST for that file clears it. The sweep takes every lagging file modified
within the catch-up window (3 days), plus every pending or half-shipped file
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
WARN_AFTER_FAILS failed POSTs in a row, or WARN_AFTER_HOURS of failure, with
no success since, the SessionStart parent prints one line, shown to the user
and to the model: "Synapse uploads have been failing since <date> (<reason>);
transcripts will be retried automatically." Reasons come from a short fixed
vocabulary (HTTP status, timeout, connection refused, ...) and never carry the
token. That failure bookkeeping lives in synapse_ingest_retry.py, which the
Codex Stop hook (plugin-codex/hooks/synapse_stop_hook.py) shares, so both
hosts retry and warn the same way.

Web research. The server captures it from the records it receives (SYNAPSE_WEB_CAPTURE, off
by default) and says whether it does in every /ingest reply (`capture`); the hook remembers
the last answer (DATA_DIR/server_captures exists while it was yes), and only while it is yes:
  * a tool result Claude Code saved to a file because it was too large (the transcript keeps
    only "Output too large ... Full output saved to: <path>") is read back into the shipped
    record when the call looks like web research (persisted_outputs.py: capped per result
    and per POST, only files that exist under the projects directory);
  * the Stop hook and the sweep also ship the session's sub-agent transcripts
    (<session>/subagents/agent-<id>.jsonl, where most research runs), each under its own
    cursor, from byte 0 (a sub-agent file has no turns to seed a tail from).
An old server never says yes, so a disabled or older server is not sent what it would drop.

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

Log: ~/.synapse/logs/ingest-hook.log (one line per upload or failure).

The sweep window (3 days), its cap (20 files), the active-write grace (300 s), the POST
chunk (400 records, also the first-run seed window on the Stop path), the POST timeout
(30 s) and the private-marker expiry (12 h) are constants, not settings; the window, cap,
grace and expiry live in synapse_ingest_retry and are shared with the Codex hook.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _spawn_detached(args: list[str]) -> None:
    subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), *args],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _stop_hook() -> None:
    """The Stop hook proper: hand the transcript to a detached `--ship` child."""
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return
    transcript_path = payload.get("transcript_path") or ""
    if not transcript_path or not os.path.exists(transcript_path):
        return
    try:
        _spawn_detached(["--ship", transcript_path])
    except Exception as e:
        _log(f"SPAWN-ERR: {type(e).__name__}: {str(e)[:160]}")


def _log(msg: str) -> None:
    import synapse_home  # not config: the Stop parent below runs before config is imported

    synapse_home.log("ingest-hook", msg)


if __name__ == "__main__" and len(sys.argv) == 1:
    # Claude waits for this process at the end of every turn, and all it does is spawn the
    # child: answer before the imports below (config, the retry bookkeeping, urllib), which
    # cost about as much again as the spawn.
    try:
        _stop_hook()
    finally:
        sys.exit(0)

import glob  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402
from collections.abc import Callable  # noqa: E402
from typing import Any  # noqa: E402

import config  # noqa: E402
import persisted_outputs  # noqa: E402
import synapse_ingest_retry as retry  # noqa: E402
import synapse_project  # noqa: E402
from synapse_filelock import lock_exclusive  # noqa: E402

INGEST_URL = config.INGEST_URL
INGEST_TOKEN = config.INGEST_TOKEN  # bearer for hosted/central endpoints
TIMEOUT = 30.0
TAIL_RECORDS = 400
CATCHUP_DAYS = retry.CATCHUP_DAYS
ACTIVE_GRACE = retry.ACTIVE_GRACE
CATCHUP_MAX_FILES = retry.CATCHUP_MAX_FILES
CURSOR_PATH = config.DATA_DIR / "ingest_cursors.json"
_CURSOR_TTL_DAYS = 45  # drop state for transcripts idle this long (or deleted); pending ones stay
PRIVATE_DIR = config.PRIVATE_DIR  # marker file per off-the-record session (private mode)
CAPTURE_MARKER = config.DATA_DIR / "server_captures"  # exists while /ingest last said capture is on

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


# ---------------------------------------------------------------------------
# Cursor planning (pure) + cursor store (flocked)
# ---------------------------------------------------------------------------


def _read_offset_records(path: str, start: int = 0) -> tuple[list[tuple[int, dict[str, Any]]], int]:
    """Read a transcript from byte `start` as [(byte_offset, record), ...] plus the EOF
    offset. `start` must be a line start (0, or a cursor from this module): the steady
    state parses only what is new, never the whole session again.

    Offsets are line starts, so any offset this returns is a valid cursor.
    Unparseable/blank lines are skipped (their bytes still count toward EOF).
    """
    out: list[tuple[int, dict[str, Any]]] = []
    eof = 0
    with open(path, "rb") as f:
        f.seek(start)
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


def _starts_a_line(path: str, offset: int) -> bool:
    """True if `offset` is 0 or follows a newline, i.e. can be read from."""
    if offset == 0:
        return True
    with open(path, "rb") as f:
        f.seek(offset - 1)
        return f.read(1) == b"\n"


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
        that turn may still be growing, so it re-ships next run (a span_id no-op
        if it did not grow, a new episode under its new span_id if it did)
        instead of leaving a half-turn stranded behind EOF;
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
    return retry.load_state(CURSOR_PATH)


def _update_state(path: str, apply: Callable[[dict[str, Any], float], None]) -> None:
    """Locked load → `apply(state, now)` → prune → atomic rewrite of the cursor
    file. Pruning drops deleted transcripts and ones idle past _CURSOR_TTL_DAYS,
    except a file still owed a retry (synapse_ingest_retry.update_state)."""
    retry.update_state(CURSOR_PATH, path, apply, ttl_days=_CURSOR_TTL_DAYS)


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
        retry.mark_ok(state, now)
        ent = state.get(path)
        if not force and isinstance(ent, dict) and int(ent.get("offset", -1)) > offset:
            return  # a racer already got further (and cleared any marker)
        state[path] = {"offset": offset, "size": size, "ts": now}

    _update_state(path, apply)


def _record_failure(path: str, reason: str, offset: int) -> None:
    """Mark `path` pending after a failed POST and extend the upload-failure
    streak. The cursor never moves here; a file with no entry yet gets one at
    `offset` (where this run started). See synapse_ingest_retry.mark_failed."""
    retry.record_failure(
        CURSOR_PATH, path, reason, offset, token=INGEST_TOKEN, ttl_days=_CURSOR_TTL_DAYS
    )


# ---------------------------------------------------------------------------
# Upload-failure reasons + the session-start warning (synapse_ingest_retry)
# ---------------------------------------------------------------------------


def _emit_failure_warning() -> None:
    """Print the sustained-failure line at SessionStart: `systemMessage` shows it
    to the user, `additionalContext` to the model. Reads local state only (no
    network), prints nothing while uploads are healthy, and never raises."""
    try:
        line = retry.failure_warning(_load_state(), time.time(), INGEST_TOKEN)
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
    """True if `session_id` is in private mode → ship NOTHING for it
    (synapse_ingest_retry.is_private, shared with the Codex hook)."""
    return retry.is_private(session_id, PRIVATE_DIR, _log)


def _cwd_labels(records: list[dict[str, Any]]) -> dict[str, str]:
    """Each cwd the records carry, with its project label, resolved the way every client
    labels a project (synapse_project: marker file, git toplevel, folder).

    The records, not the hook payload, because the catch-up sweep and the history import
    have no payload, and a swept transcript's session ran somewhere other than this one."""
    cwds = {c for r in records if isinstance(r, dict) and isinstance(c := r.get("cwd"), str) and c}
    return {c: lb for c in cwds if (lb := synapse_project.project_label(c))}


def _server_captures() -> bool:
    """True if the server's last /ingest reply said web capture is on (see _note_capture)."""
    return CAPTURE_MARKER.exists()


def _note_capture(reply: str) -> None:
    """Remember what an /ingest reply says about web capture. A server that does not say
    (older, or a capture switch that is off or misspelled) counts as off; a reply that is not
    the server's JSON (a proxy's error page) changes nothing. Never raises: not knowing means
    not shipping the extras, which is safe."""
    try:
        on = json.loads(reply).get("capture") is True
    except (ValueError, AttributeError):
        return
    try:
        if on:
            CAPTURE_MARKER.parent.mkdir(parents=True, exist_ok=True)
            CAPTURE_MARKER.touch()
        else:
            CAPTURE_MARKER.unlink(missing_ok=True)
    except OSError:
        pass


def _payload(records: list[dict[str, Any]], source: str, *, inline: bool) -> dict[str, Any]:
    """The /ingest body for a batch of raw transcript records: the records themselves, the
    results of large tool outputs read in from their files when ``inline`` (persisted_outputs:
    only for a server that captures), and the batch's project label (_cwd_labels). Shared
    with `synapse capture`, whose dry run posts the same body to /capture/dry-run."""
    payload: dict[str, Any] = {
        "records": persisted_outputs.inline(records, config.PROJECTS_DIR) if inline else records,
        "source": source,
    }
    by_cwd = _cwd_labels(records)
    if len(set(by_cwd.values())) == 1:
        payload["project"] = next(iter(by_cwd.values()))
    elif by_cwd:
        # A session that moved between repositories: each turn is labelled from its own
        # cwd (an older server ignores this and labels each turn from the cwd's basename).
        payload["cwd_projects"] = by_cwd
    return payload


def _post_records(records: list[dict[str, Any]], source: str = "hook") -> str:
    """POST a batch of raw transcript records to /ingest; returns the (truncated)
    response body. Shared by the Stop-hook shipper, the catch-up sweep, and the
    bulk history import (import_history.py), so all resolve URL/token identically
    through config.py, and all send the batch's project label (_cwd_labels). Reads the
    server's answer on web capture from the reply (_note_capture) for the next POST.
    Raises on transport/HTTP errors — callers decide how to fail-soft."""
    body = json.dumps(_payload(records, source, inline=_server_captures())).encode()
    # A real User-Agent, same as config.post_json: Cloudflare in front of a hosted server
    # 403s urllib's default "Python-urllib/x" ("error code: 1010"), and this hook fails
    # soft, so without it every ingest is silently dropped.
    headers = {"Content-Type": "application/json", "User-Agent": config.USER_AGENT}
    if INGEST_TOKEN:
        headers["Authorization"] = f"Bearer {INGEST_TOKEN}"
    req = urllib.request.Request(INGEST_URL, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:  # user-configured Synapse URL
        reply = resp.read().decode()
    _note_capture(reply)
    return reply[:200]


def _parent_session(transcript_path: str) -> str | None:
    """The session a sub-agent transcript (<session>/subagents/agent-<id>.jsonl) belongs to;
    None for any other file. Its records carry the parent's sessionId."""
    subagents = os.path.dirname(transcript_path)
    if os.path.basename(subagents) != "subagents":
        return None
    return os.path.basename(os.path.dirname(subagents))


def _ship(transcript_path: str, *, mode: str = "stop") -> tuple[int, int]:
    """Ship everything past the file's cursor to /ingest; returns (posts, records).

    stop mode seeds a missing cursor from the bounded tail (the pre-cursor hook
    already shipped that session turn-by-turn); catchup mode ships from byte 0 —
    recovering never-shipped files is its entire point — and so does a sub-agent
    transcript in either mode: it has no turns to seed a tail from, and the server
    reads whole tool calls from it. A failed POST leaves the
    cursor at the failure point, so the next Stop or sweep resumes there.

    Private mode short-circuits everything: a session with a live marker file posts
    nothing, on either path (the catch-up sweep calls this per file, so it inherits
    the same check per session). The cursor is left alone, so if the marker later
    expires the turns are still shippable — the server's private_sessions row, not
    this cursor, is what keeps them out of memory for good.
    """
    # Cheap check first: the transcript's own name is the session id (a sub-agent's, its
    # session directory's).
    parent = _parent_session(transcript_path)
    session_id = parent or os.path.splitext(os.path.basename(transcript_path))[0]
    if _is_private(session_id):
        _log(f"PRIVATE skip {mode} {os.path.basename(transcript_path)}")
        return (0, 0)
    try:
        size = os.stat(transcript_path).st_size
        ent = _load_state().get(transcript_path)
        cursor: int | None = None
        force = False
        if isinstance(ent, dict):
            cursor = int(ent.get("offset", 0))
            if cursor > size or not _starts_a_line(transcript_path, cursor):
                cursor, force = None, True  # transcript truncated/rewritten — start over
            elif int(ent.get("size", -1)) == size:
                return (0, 0)  # nothing new since the last fully-shipped run
        offset_recs, eof = _read_offset_records(transcript_path, cursor or 0)
        if not offset_recs:
            return (0, 0)
        if cursor is None:
            if mode == "stop" and parent is None:
                tail = offset_recs[-TAIL_RECORDS:]
                i = next((j for j, (_, r) in enumerate(tail) if _is_turn_start(r)), None)
                cursor = tail[i][0] if i is not None else 0
            else:
                cursor = 0

        plans = _plan_from_cursor(offset_recs, cursor, TAIL_RECORDS)

        # Belt and braces: a resumed/compacted transcript can carry records whose
        # sessionId differs from the filename, and it's the record's id that the
        # server stores (and that private_mode.py marked). Only the records about
        # to ship matter.
        shipping = {str(r.get("sessionId") or "") for recs, _ in plans for r in recs}
        for sid in shipping - {"", session_id}:
            if _is_private(sid):
                _log(f"PRIVATE skip {mode} {os.path.basename(transcript_path)} (record sid)")
                return (0, 0)
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
                _record_failure(transcript_path, retry.short_reason(e), chunk_start)
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


def _subagent_files(transcript_path: str) -> list[str]:
    """The session's sub-agent transcripts, which Claude Code writes beside its own."""
    session_id = os.path.splitext(os.path.basename(transcript_path))[0]
    subagents = os.path.join(os.path.dirname(transcript_path), session_id, "subagents")
    return sorted(glob.glob(os.path.join(glob.escape(subagents), "*.jsonl")))


def _ship_session(transcript_path: str) -> None:
    """The Stop hook's work: the session's transcript, then (to a server that captures web
    research, which the transcript's reply has just told us) its sub-agent transcripts."""
    _ship(transcript_path, mode="stop")
    if _server_captures():
        for path in _subagent_files(transcript_path):
            _ship(path, mode="stop")


# ---------------------------------------------------------------------------
# SessionStart catch-up sweep
# ---------------------------------------------------------------------------


def _catchup_candidates(
    projects_root: str,
    skip_path: str,
    state: dict[str, Any],
    now: float,
    *,
    subagents: bool = False,
) -> list[str]:
    """Transcripts worth sweeping: not the live session, not mid-write
    (ACTIVE_GRACE — an active session's own Stop hooks own it), with bytes past
    their cursor, and either modified within CATCHUP_DAYS or owed a retry (a
    failed or half-finished upload, retry.needs_retry) however old.

    A file older than the window that the hook never attempted is NOT a
    candidate: that is history from before the plugin was installed (or from
    before failures were recorded), and `synapse-import` is the tool for it.

    Least recently touched first — the last failed attempt, else the mtime — so
    repeated capped sweeps make oldest-first progress and a file that keeps
    failing rotates to the back instead of starving the rest. Pure given
    os.stat results — patchable in tests. The rules are
    synapse_ingest_retry.sweep_candidates, shared with the Codex hook.

    ``subagents`` adds the sub-agent transcripts (<session>/subagents/agent-<id>.jsonl)
    under the same rules; the caller asks for them only while the server captures."""
    paths = glob.glob(os.path.join(projects_root, "*", "*.jsonl"))
    if subagents:
        paths += glob.glob(os.path.join(projects_root, "*", "*", "subagents", "*.jsonl"))
    return retry.sweep_candidates(
        paths,
        state,
        now,
        window_days=CATCHUP_DAYS,
        active_grace=ACTIVE_GRACE,
        # fully shipped and unchanged: `size` is the EOF of the last complete run
        shipped=lambda ent, st: int(ent.get("size", -1)) == st.st_size,
        skip_path=skip_path,
    )


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
        candidates = _catchup_candidates(
            projects_root,
            skip_path,
            _load_state(),
            time.time(),
            subagents=_server_captures(),
        )
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


def main() -> None:
    # Detached children do the actual work.
    if len(sys.argv) >= 3 and sys.argv[1] == "--ship":
        _ship_session(sys.argv[2])
        return
    if len(sys.argv) >= 3 and sys.argv[1] == "--catchup-run":
        _catchup(sys.argv[2], sys.argv[3] if len(sys.argv) >= 4 else "")
        return

    if len(sys.argv) >= 2 and sys.argv[1] == "--catchup":
        # Parent (invoked by Claude at SessionStart): read the hook payload, spawn a
        # detached sweep, and return immediately so the session isn't delayed.
        try:
            payload = json.load(sys.stdin)
        except Exception:
            return
        transcript_path = payload.get("transcript_path") or ""
        # Say so when uploads keep failing. Before the sweep runs, so the line
        # reflects the streak as it stood; a sweep that succeeds ends the streak.
        _emit_failure_warning()
        if payload.get("source") in ("clear", "compact"):
            return  # a fresh context window, not a new session: the sweep ran at its start
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

    _stop_hook()  # the default; run as a script this is answered before the imports above


if __name__ == "__main__":
    try:
        main()
    finally:
        # A Stop/SessionStart hook must never fail the turn.
        sys.exit(0)
