#!/usr/bin/env python3
# mypy: ignore-errors
"""Claude Code `Stop` hook → push the session transcript tail to Synapse /ingest.

Memory-only, Logfire-free. On every turn completion Claude Code runs this with
the hook payload on stdin ({transcript_path, session_id, cwd, ...}). We ship a
bounded TAIL of the transcript (raw JSONL records) to the Synapse `/ingest`
endpoint, which parses it with the same JSONLParser the disk sweep uses and
dedups by span_id — so push and sweep converge idempotently.

Design constraints:
  * NEVER block or fail the turn. Claude waits for Stop hooks to exit, so the
    actual HTTP work is done in a DETACHED child (start_new_session) and the
    parent returns instantly. The parent always exits 0.
  * No third-party deps — uses urllib so it runs under whatever Python the CLI
    environment provides.
  * Bounded tail, not the whole file. The server keys turns by span_id (a turn's
    last record uuid) and appends new ones at max(sequence)+1, so we only need to
    ship recent records, not the entire growing transcript every turn (that was a
    ~20s parse + O(turns^2) upsert per POST). We trim the tail to start at a real
    turn boundary so we never POST a leading fragment of an already-ingested turn;
    if the window holds no boundary (one mega-turn longer than the window) we fall
    back to the full file so that turn still lands complete.

The endpoint and bearer token resolve through scripts/config.py, same as every
other hook: explicit env var → plugin userConfig → the `/plugin install` answers
persisted in settings.json → default (http://localhost:8765/ingest). The legacy
SYNAPSE_INGEST_URL / SYNAPSE_INGEST_TOKEN env vars still win when set, per
config.py's precedence. The detached child re-imports config, so the same
resolution applies there.

Env (all optional):
  SYNAPSE_INGEST_LOG      default /tmp/synapse-ingest-hook.log
  SYNAPSE_INGEST_TIMEOUT  default 30 (seconds per POST)
  SYNAPSE_INGEST_TAIL     default 400 (raw records kept in the tail window)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from datetime import UTC, datetime
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

INGEST_URL = config.INGEST_URL
INGEST_TOKEN = config.INGEST_TOKEN  # bearer for hosted/central endpoints
LOG_PATH = os.environ.get("SYNAPSE_INGEST_LOG", "/tmp/synapse-ingest-hook.log")
TIMEOUT = float(os.environ.get("SYNAPSE_INGEST_TIMEOUT", "30"))
TAIL_RECORDS = int(os.environ.get("SYNAPSE_INGEST_TAIL", "400"))

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
    """True if rec is a fresh human turn — a safe place to begin a tail slice.

    Mirrors ingestion.jsonl_client._is_user_turn. A tail must start at a turn
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
    """Pick the records to POST: a turn-boundary-aligned tail, or the full file
    when the tail window holds no boundary (one mega-turn). Returns (records, mode).

    Pure (no I/O) so it's unit-testable without a live server.
    """
    tail = _parse_lines(raw_lines[-TAIL_RECORDS:])
    start = next((i for i, r in enumerate(tail) if _is_turn_start(r)), None)
    if start is not None:
        return tail[start:], "tail"
    return _parse_lines(raw_lines), "full-fallback"


def _post_records(records: list[dict[str, Any]], source: str = "hook") -> str:
    """POST a batch of raw transcript records to /ingest; returns the (truncated)
    response body. Shared by the Stop-hook shipper and the bulk history import
    (import_history.py), so both resolve URL/token identically through config.py.
    Raises on transport/HTTP errors — callers decide how to fail-soft."""
    body = json.dumps({"records": records, "source": source}).encode()
    headers = {"Content-Type": "application/json"}
    if INGEST_TOKEN:
        headers["Authorization"] = f"Bearer {INGEST_TOKEN}"
    req = urllib.request.Request(INGEST_URL, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:  # user-configured Synapse URL
        return resp.read().decode()[:200]


def _ship(transcript_path: str) -> None:
    """Read the transcript and POST a bounded tail to /ingest. Runs detached."""
    try:
        raw_lines: list[bytes] = []
        with open(transcript_path, "rb") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_lines.append(line)
        if not raw_lines:
            return
        records, mode = _select_tail(raw_lines)
        if not records:
            return
        payload = _post_records(records, source="hook")
        _log(f"OK {os.path.basename(transcript_path)} {mode} {len(records)} recs -> {payload}")
    except Exception as e:
        # The disk-sweep backstop will re-ingest anything a failed POST dropped.
        _log(f"ERR {transcript_path}: {type(e).__name__}: {str(e)[:160]}")


def main() -> None:
    # --ship: the detached child does the actual work.
    if len(sys.argv) >= 3 and sys.argv[1] == "--ship":
        _ship(sys.argv[2])
        return

    # Parent (invoked by Claude): read the hook payload, spawn a detached
    # shipper, and return immediately so the turn isn't delayed.
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return
    transcript_path = payload.get("transcript_path")
    if not transcript_path or not os.path.exists(transcript_path):
        return
    try:
        subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--ship", transcript_path],
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        _log(f"SPAWN-ERR: {type(e).__name__}: {str(e)[:160]}")


if __name__ == "__main__":
    try:
        main()
    finally:
        # A Stop hook must never fail the turn.
        sys.exit(0)
