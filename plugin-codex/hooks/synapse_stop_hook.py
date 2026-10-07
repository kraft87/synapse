#!/usr/bin/env python3
"""Codex CLI `Stop` hook → push the session rollout tail to Synapse /ingest.

The Codex analog of plugin/scripts/ingest_hook.py. Codex fires Stop when a
turn completes, passing JSON on stdin ({session_id, transcript_path, cwd,
hook_event_name, turn_id, ...}) where transcript_path is the session's
rollout .jsonl. We ship the not-yet-shipped tail of that file to Synapse's
`/ingest` endpoint with format="codex", where CodexRolloutParser — the same
parser the disk sweep (ingestion.codex_backfill) uses — turns it into
episodes, so push and sweep converge idempotently on span_id.

Design constraints, inherited from the Claude Code hook:
  * NEVER block or fail the turn. The HTTP work runs in a DETACHED child
    (start_new_session) and the parent exits 0 immediately — belt and
    suspenders on top of Codex's async hook support.
  * No third-party deps — urllib only.
  * Ship a byte-cursor tail, not the whole file. Cursor state lives in
    ~/.synapse/codex_cursors.json keyed by rollout path and advances only
    after a successful POST, so a failed push retries next turn. A tail that
    starts mid-turn (prior failure) is trimmed forward to the next real user
    message; if no boundary exists in the tail, fall back to the full file —
    span_id dedup server-side makes the re-ship a no-op.
  * A pushed tail usually lacks the rollout's session_meta line, so the POST
    carries session_id explicitly; the server passes it to the parser as the
    identity hint.

SessionStart (hooks/session_start.py) spawns `--catchup`: a detached sweep
that ships any rollout whose cursor lags its size — sessions that died
without a Stop hook, or turns dropped while the server was unreachable.

Failed uploads are retried however old they are. A POST that fails (server
down, 401/403 from a revoked token or a proxy/WAF, timeout) marks the
rollout's cursor entry `pending`, with the time and a short reason; the next
successful POST for that rollout clears it. The sweep takes every lagging
rollout modified within SYNAPSE_CODEX_CATCHUP_DAYS, plus every pending one
regardless of age, least recently touched first, at most
SYNAPSE_CODEX_CATCHUP_MAX per session start. Pending entries are exempt from
the cursor TTL. A rollout older than the window that the hook never attempted
(history from before install) is left alone; `python -m
ingestion.codex_backfill` is the tool for that.

After 3 failed POSTs in a row, or 24 hours of failure, with no success since,
session start prints one line, shown to the user and to the model: "Synapse
uploads have been failing since <date> (<reason>); transcripts will be
retried automatically." The reason comes from a short fixed vocabulary and
never carries the token.

That failure bookkeeping is shared with the Claude Code plugin rather than
copied: plugin/scripts/synapse_ingest_retry.py, imported from this checkout
(the installer points Codex at this repo, as it does for the skills sync), so
both hosts retry and warn by the same rules. Each keeps its own cursor file
and cursor semantics.

Env:
  SYNAPSE_URL                 base URL           (default http://localhost:8765)
  SYNAPSE_INGEST_URL          override /ingest   (else derived from SYNAPSE_URL)
  SYNAPSE_INGEST_TOKEN        bearer token (an enrolled machine's saved device
                              token wins)
  SYNAPSE_CODEX_CURSORS       cursor state file  (default ~/.synapse/codex_cursors.json)
  SYNAPSE_CODEX_CATCHUP_DAYS  default 3 (the sweep picks up any lagging rollout
                              modified this recently; one whose upload failed
                              is retried regardless of age)
  SYNAPSE_CODEX_CATCHUP_MAX   default 20 (rollouts per sweep; the rest defer to
                              the next session start, logged)
  SYNAPSE_PRIVATE_DIR         private markers    (default ~/.synapse/private)
  SYNAPSE_CODEX_HOOK_LOG      log file           (default /tmp/synapse-codex-hook.log)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

# The bearer comes from the shared resolver so this hook, the board and the MCP header
# helper always agree: on an enrolled machine the saved device token beats a stale env
# SYNAPSE_INGEST_TOKEN (common._resolve_token). Stdlib-only, like this hook.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
# The failure bookkeeping is the Claude plugin's (synapse_ingest_retry), from this
# checkout. Appended, not prepended, so nothing there can shadow a module this hook
# imports.
sys.path.append(str(Path(__file__).resolve().parents[2] / "plugin" / "scripts"))
import synapse_ingest_retry as retry
from common import TOKEN
from synapse_filelock import lock_exclusive


def _claude_plugin_options() -> dict[str, str]:
    """Fallback config source: the Synapse *Claude Code* plugin persists
    SYNAPSE_URL / SYNAPSE_INGEST_TOKEN in ~/.claude/settings.json at install
    time. Codex hooks only inherit plain env, so on a machine running both
    plugins this reuses that config instead of requiring duplicate env vars.
    Env always wins here; the token is resolved in common (see above)."""
    try:
        data = json.loads(
            Path(os.path.expanduser("~/.claude/settings.json")).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return {}
    merged: dict[str, str] = {}
    for cfg_key, cfg in (data.get("pluginConfigs") or {}).items():
        if str(cfg_key).split("@", 1)[0] == "synapse":
            opts = cfg.get("options") or {}
            merged.update({k: str(v) for k, v in opts.items() if v not in (None, "")})
    return merged


_FALLBACK = _claude_plugin_options()


def _cfg(key: str, default: str = "") -> str:
    return os.environ.get(key) or _FALLBACK.get(key) or default


BASE_URL = _cfg("SYNAPSE_URL", "http://localhost:8765").rstrip("/")
INGEST_URL = _cfg("SYNAPSE_INGEST_URL") or BASE_URL + "/ingest"
TIMEOUT = float(os.environ.get("SYNAPSE_INGEST_TIMEOUT", "30"))
LOG_PATH = os.environ.get("SYNAPSE_CODEX_HOOK_LOG", "/tmp/synapse-codex-hook.log")
CURSORS_PATH = Path(
    os.path.expanduser(os.environ.get("SYNAPSE_CODEX_CURSORS", "~/.synapse/codex_cursors.json"))
)
PRIVATE_DIR = Path(os.path.expanduser(os.environ.get("SYNAPSE_PRIVATE_DIR", "~/.synapse/private")))
SESSIONS_ROOT = Path(os.path.expanduser("~/.codex/sessions"))
CATCHUP_DAYS = float(os.environ.get("SYNAPSE_CODEX_CATCHUP_DAYS", "3"))
CATCHUP_MAX_FILES = int(os.environ.get("SYNAPSE_CODEX_CATCHUP_MAX", "20"))
# The sweep skips rollouts written this recently: a live session's own Stop hook
# owns them, and a sweep that caught one mid-turn would ship half a turn.
ACTIVE_GRACE = 300.0
_CURSOR_TTL_DAYS = 45  # drop state for rollouts idle this long (or deleted); pending ones stay
_ROLLOUT_ID = re.compile(r"rollout-.*-([0-9a-f-]{36})\.jsonl$")

# Mirror of ingestion.codex_client._MACHINERY_PREFIXES — kept inline so the hook
# stays dependency-free (it runs under whatever python3 the machine provides).
_MACHINERY_PREFIXES = (
    "<environment_context>",
    "<user_instructions>",
    "<turn_context>",
    "<permissions",
    "<skills_instructions>",
    "<multi_agent",
    "<collaboration_mode>",
    "<system-reminder>",
)


def _log(msg: str) -> None:
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    except OSError:
        pass


def _is_user_boundary(rec: dict[str, Any]) -> bool:
    """Mirror of codex_client._is_user_turn: a real human message record."""
    if rec.get("type") != "response_item":
        return False
    payload = rec.get("payload") or {}
    if payload.get("type") != "message" or payload.get("role") != "user":
        return False
    parts = []
    content = payload.get("content")
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for b in content:
            if isinstance(b, dict) and b.get("type") in ("input_text", "text"):
                parts.append(str(b.get("text") or ""))
    text = "\n".join(parts).strip()
    return bool(text) and not text.lstrip().startswith(_MACHINERY_PREFIXES)


def _read_records(path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    """Read complete JSONL records from byte ``offset`` to EOF.

    Returns (records, new_offset). A trailing line without a newline is a
    write in progress — excluded, and the offset stops before it.
    """
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read()
    end = len(data)
    if data and not data.endswith(b"\n"):
        end = data.rfind(b"\n") + 1  # 0 if no complete line at all
    records: list[dict[str, Any]] = []
    for line in data[:end].splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            records.append(rec)
    return records, offset + end


# ---------------------------------------------------------------------------
# Cursor store (flocked) + failure bookkeeping (shared: synapse_ingest_retry)
# ---------------------------------------------------------------------------


def _load_state() -> dict[str, Any]:
    return retry.load_state(CURSORS_PATH)


def _load_cursor(path_key: str) -> int:
    ent = _load_state().get(path_key)
    try:
        return int(ent.get("offset", 0)) if isinstance(ent, dict) else 0
    except (TypeError, ValueError):
        return 0


def _save_cursor(path_key: str, offset: int) -> None:
    """Persist the cursor after a successful POST, under the cursor file's lock.
    A success also ends the upload-failure streak, and rewriting the entry
    clears the rollout's `pending` marker."""

    def apply(state: dict[str, Any], now: float) -> None:
        retry.mark_ok(state, now)
        state[path_key] = {"offset": offset, "ts": now}

    retry.update_state(CURSORS_PATH, path_key, apply, ttl_days=_CURSOR_TTL_DAYS)


def _record_failure(path_key: str, reason: str, offset: int) -> None:
    """Mark the rollout pending after a failed POST and extend the failure
    streak. The cursor never moves; a rollout with no entry yet gets one at
    `offset`, where this run started."""
    retry.record_failure(
        CURSORS_PATH, path_key, reason, offset, token=TOKEN, ttl_days=_CURSOR_TTL_DAYS
    )


def failure_warning(now: float | None = None) -> str | None:
    """The session-start line for a sustained upload failure, or None while
    uploads are healthy. Reads local state only (no network)."""
    return retry.failure_warning(_load_state(), time.time() if now is None else now, TOKEN)


# ---------------------------------------------------------------------------
# Shipping
# ---------------------------------------------------------------------------


def _find_rollout(session_id: str) -> Path | None:
    matches = sorted(SESSIONS_ROOT.rglob(f"rollout-*-{session_id}.jsonl"))
    return matches[-1] if matches else None


def _post(records: list[dict[str, Any]], session_id: str) -> dict[str, Any]:
    body = {
        "records": records,
        "format": "codex",
        "session_id": session_id,
        "source": "codex-hook",
    }
    headers = {"Content-Type": "application/json", "User-Agent": "synapse-codex-hook/0.1"}
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    req = urllib.request.Request(
        INGEST_URL, data=json.dumps(body).encode(), method="POST", headers=headers
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        resp: dict[str, Any] = json.loads(r.read() or b"{}")
        return resp


def _ship(payload: dict[str, Any]) -> tuple[int, int]:
    """Ship the rollout past its cursor; returns (posts, records). A failed POST
    leaves the cursor where it was and marks the rollout pending, so the next
    Stop or any later sweep retries it."""
    session_id = payload.get("session_id") or ""
    transcript = payload.get("transcript_path")
    if not session_id:
        _log("skip: no session_id in hook payload")
        return (0, 0)
    if (PRIVATE_DIR / session_id).exists():
        _log(f"skip: private session {session_id}")
        return (0, 0)
    path = Path(transcript) if transcript else _find_rollout(session_id)
    if not path or not path.exists():
        _log(f"skip: rollout not found for {session_id}")
        return (0, 0)

    path_key = str(path)
    offset = _load_cursor(path_key)
    try:
        size = path.stat().st_size
        if offset > size:
            offset = 0  # file replaced/truncated — reship, span dedup absorbs it
        records, new_offset = _read_records(path, offset)
    except OSError as e:
        _log(f"error reading {path}: {e}")
        return (0, 0)
    if not records:
        return (0, 0)

    if offset > 0 and not any(_is_user_boundary(r) for r in records):
        # Mid-turn tail with no boundary to trim to (prior failure inside one
        # mega-turn) — reship the whole file; server-side span dedup no-ops
        # everything already stored.
        try:
            records, new_offset = _read_records(path, 0)
        except OSError as e:
            _log(f"error re-reading {path}: {e}")
            return (0, 0)
    elif offset > 0:
        first = next(i for i, r in enumerate(records) if _is_user_boundary(r))
        records = records[first:]

    if not records:
        return (0, 0)
    try:
        resp = _post(records, session_id)
    except Exception as e:
        _log(f"POST failed for {session_id} ({len(records)} records): {e}")
        try:
            _record_failure(path_key, retry.short_reason(e), offset)
        except Exception as e2:
            _log(f"record-failure failed for {session_id}: {type(e2).__name__}: {e2}")
        return (0, 0)
    try:
        _save_cursor(path_key, new_offset)
    except Exception as e:
        _log(f"cursor save failed for {session_id}: {type(e).__name__}: {e}")
    _log(f"shipped {len(records)} records for {session_id} -> ingested={resp.get('ingested')}")
    return (1, len(records))


# ---------------------------------------------------------------------------
# SessionStart catch-up sweep
# ---------------------------------------------------------------------------


def _catchup_candidates(
    sessions_root: Path, skip_path: str, state: dict[str, Any], now: float
) -> list[str]:
    """Rollouts worth sweeping, in sweep order (synapse_ingest_retry.sweep_candidates):
    not the live session, not written in the last ACTIVE_GRACE seconds, with bytes
    past their cursor, and either modified within CATCHUP_DAYS or owed a retry
    however old. Least recently touched first."""
    return retry.sweep_candidates(
        (str(p) for p in sessions_root.rglob("rollout-*.jsonl") if _ROLLOUT_ID.search(p.name)),
        state,
        now,
        window_days=CATCHUP_DAYS,
        active_grace=ACTIVE_GRACE,
        # nothing past the cursor: the offset is the end of the last line shipped
        shipped=lambda ent, st: int(ent.get("offset", -1)) == st.st_size,
        skip_path=skip_path,
    )


def _catchup(skip_path: str = "") -> None:
    """Backstop sweep: ship the tail of every lagging rollout.

    Spawned detached by the SessionStart hook. Cursors make re-shipping cheap
    and server-side span dedup makes it idempotent, so overlap with the live
    Stop hook is a no-op. A non-blocking lock collapses concurrent session
    starts to one sweep.
    """
    CURSORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(str(CURSORS_PATH) + ".catchup.lock", "w") as lf:
        try:
            lock_exclusive(lf, blocking=False)
        except OSError:
            _log("catchup skipped: another sweep is running")
            return
        candidates = _catchup_candidates(SESSIONS_ROOT, skip_path, _load_state(), time.time())
        batch = candidates[:CATCHUP_MAX_FILES]
        files = recs = 0
        for path in batch:
            m = _ROLLOUT_ID.search(os.path.basename(path))
            if not m:
                continue
            _, r = _ship({"session_id": m.group(1), "transcript_path": path})
            files += 1 if r else 0
            recs += r
        deferred = len(candidates) - len(batch)
        if batch:
            _log(
                f"catchup: {len(batch)} checked, {files} shipped ({recs} records)"
                + (f", {deferred} deferred to next session start" if deferred else "")
            )


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--catchup":
        _catchup(sys.argv[2] if len(sys.argv) >= 3 else "")
        return 0
    if len(sys.argv) >= 3 and sys.argv[1] == "--ship":
        _ship(json.loads(sys.argv[2]))
        return 0

    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        return 0
    # Detach the actual work so the Stop hook returns instantly even if the
    # server is slow or down.
    try:
        subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--ship", json.dumps(payload)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as e:
        _log(f"detach failed: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
