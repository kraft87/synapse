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

Env:
  SYNAPSE_URL              base URL           (default http://localhost:8765)
  SYNAPSE_INGEST_URL       override /ingest   (else derived from SYNAPSE_URL)
  SYNAPSE_INGEST_TOKEN     bearer token
  SYNAPSE_CODEX_CURSORS    cursor state file  (default ~/.synapse/codex_cursors.json)
  SYNAPSE_PRIVATE_DIR      private markers    (default ~/.synapse/private)
  SYNAPSE_CODEX_HOOK_LOG   log file           (default /tmp/synapse-codex-hook.log)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


def _claude_plugin_options() -> dict[str, str]:
    """Fallback config source: the Synapse *Claude Code* plugin persists
    SYNAPSE_URL / SYNAPSE_INGEST_TOKEN in ~/.claude/settings.json at install
    time. Codex hooks only inherit plain env, so on a machine running both
    plugins this reuses that config instead of requiring duplicate env vars.
    Env always wins."""
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
TOKEN = _cfg("SYNAPSE_INGEST_TOKEN")
TIMEOUT = float(os.environ.get("SYNAPSE_INGEST_TIMEOUT", "30"))
LOG_PATH = os.environ.get("SYNAPSE_CODEX_HOOK_LOG", "/tmp/synapse-codex-hook.log")
CURSORS_PATH = Path(
    os.path.expanduser(os.environ.get("SYNAPSE_CODEX_CURSORS", "~/.synapse/codex_cursors.json"))
)
PRIVATE_DIR = Path(os.path.expanduser(os.environ.get("SYNAPSE_PRIVATE_DIR", "~/.synapse/private")))
SESSIONS_ROOT = Path(os.path.expanduser("~/.codex/sessions"))

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


def _load_cursor(path_key: str) -> int:
    try:
        cursors = json.loads(CURSORS_PATH.read_text(encoding="utf-8"))
        return int(cursors.get(path_key, {}).get("offset", 0))
    except (OSError, ValueError):
        return 0


def _save_cursor(path_key: str, offset: int) -> None:
    CURSORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        cursors = json.loads(CURSORS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cursors = {}
    if not isinstance(cursors, dict):
        cursors = {}
    cursors[path_key] = {"offset": offset, "ts": time.time()}
    tmp = CURSORS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cursors), encoding="utf-8")
    tmp.replace(CURSORS_PATH)


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


def _ship(payload: dict[str, Any]) -> None:
    session_id = payload.get("session_id") or ""
    transcript = payload.get("transcript_path")
    if not session_id:
        _log("skip: no session_id in hook payload")
        return
    if (PRIVATE_DIR / session_id).exists():
        _log(f"skip: private session {session_id}")
        return
    path = Path(transcript) if transcript else _find_rollout(session_id)
    if not path or not path.exists():
        _log(f"skip: rollout not found for {session_id}")
        return

    path_key = str(path)
    offset = _load_cursor(path_key)
    try:
        size = path.stat().st_size
        if offset > size:
            offset = 0  # file replaced/truncated — reship, span dedup absorbs it
        records, new_offset = _read_records(path, offset)
    except OSError as e:
        _log(f"error reading {path}: {e}")
        return
    if not records:
        return

    if offset > 0 and not any(_is_user_boundary(r) for r in records):
        # Mid-turn tail with no boundary to trim to (prior failure inside one
        # mega-turn) — reship the whole file; server-side span dedup no-ops
        # everything already stored.
        try:
            records, new_offset = _read_records(path, 0)
        except OSError as e:
            _log(f"error re-reading {path}: {e}")
            return
    elif offset > 0:
        first = next(i for i, r in enumerate(records) if _is_user_boundary(r))
        records = records[first:]

    if not records:
        return
    try:
        resp = _post(records, session_id)
    except Exception as e:
        _log(f"POST failed for {session_id} ({len(records)} records): {e}")
        return
    _save_cursor(path_key, new_offset)
    _log(f"shipped {len(records)} records for {session_id} -> ingested={resp.get('ingested')}")


def _catchup() -> None:
    """Backstop sweep: ship the tail of every recently-modified rollout.

    Spawned detached by the SessionStart hook. Cursors make re-shipping cheap
    and server-side span dedup makes it idempotent, so overlap with the live
    Stop hook is a no-op. Window: SYNAPSE_CODEX_CATCHUP_DAYS (default 3).
    """
    days = float(os.environ.get("SYNAPSE_CODEX_CATCHUP_DAYS", "3"))
    cutoff = time.time() - days * 86400
    from re import search

    for path in sorted(SESSIONS_ROOT.rglob("rollout-*.jsonl")):
        try:
            if path.stat().st_mtime < cutoff:
                continue
        except OSError:
            continue
        m = search(r"rollout-.*-([0-9a-f-]{36})\.jsonl$", path.name)
        if not m:
            continue
        _ship({"session_id": m.group(1), "transcript_path": str(path)})


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--catchup":
        _catchup()
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
