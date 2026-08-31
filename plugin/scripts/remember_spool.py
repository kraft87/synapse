#!/usr/bin/env python3
# mypy: ignore-errors
"""Durable spool for remember() intents — memory writes survive an MCP outage.

    remember_spool.py add --hook H --body B [--type user] [--project p]
    remember_spool.py add            # same, reading {"hook","body","type","project"} JSON on stdin
    remember_spool.py list
    remember_spool.py flush
    remember_spool.py post-tool-use  # PostToolUse hook: reads the hook payload on stdin
    remember_spool.py session-start   # SessionStart hook: probe, flush, report

WHY THIS EXISTS. On 2026-08-25 the MCP transport's OAuth was down mid-session. The user
stated a memory correction, remember() could not be called, and the correction was silently
lost for two days while the poisoned note it was meant to replace kept re-injecting at every
session start. The plugin's plain-HTTP lane (bearer token, /ingest) worked the whole time —
only MCP was broken. A memory write had no durable queue, so a failed write was a lost write.

The spool closes that gap: an intent that cannot be written NOW is appended to
``DATA_DIR/remember_spool.jsonl`` (beside ingest_cursors.json) and replayed over the
machine-token ``POST /remember/spool`` route, which performs the identical write the MCP
tool does. Two capture paths, because the two outages look different from inside a session:

  * PostToolUse — remember() exists but FAILED. The hook sees the failed tool_response,
    spools ``tool_input``, and tells the model the write is queued (so it tells the user
    "queued", not "saved" and not nothing).
  * `add` CLI — remember() is not there AT ALL (MCP never connected, so no tool call fires
    and no PostToolUse hook exists to catch it). The SessionStart step below tells the model
    to route memory writes through this subcommand while the server is unreachable.

DURABILITY CONTRACT. The local jsonl is the store of record; the server confirm is the only
thing that removes a line.
  * appends are one flocked, fsynced write of a single line — a crash mid-append loses at
    most the line being written, never an earlier one;
  * flush posts each intent and removes ONLY the ids the server confirmed, rewriting the
    file under the same lock — so a flush that dies mid-way leaves every unconfirmed intent
    exactly where it was, and intents appended DURING the flush survive the rewrite;
  * a confirm that is sent but never received (crash in the window) re-posts next time; the
    route is idempotent on the client-generated intent id, so the retry lands as 'duplicate'
    rather than a second note.

FAIL-SOFT, like ingest_hook.py: every hook path swallows its errors and exits 0, and the
SessionStart flush runs on a bounded budget so it cannot outlive its hook timeout. A broken
spool must never break a turn — but it must also never quietly drop a memory write, so the
CLI paths (add/flush) DO report failure on stderr and exit nonzero.

Env (all optional):
  SYNAPSE_SPOOL_LOG            default /tmp/synapse-remember-spool.log
  SYNAPSE_SPOOL_TIMEOUT        default 10   (seconds per replay POST)
  SYNAPSE_SPOOL_PROBE_TIMEOUT  default 4    (seconds for the SessionStart reachability probe)
  SYNAPSE_SPOOL_BUDGET         default 12   (seconds a SessionStart flush may spend)
  SYNAPSE_SPOOL_MAX            default 25   (intents per flush; the rest defer, logged)
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
import uuid
from datetime import UTC, datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from filelock import lock_exclusive

SPOOL_PATH = config.DATA_DIR / "remember_spool.jsonl"
LOCK_PATH = str(SPOOL_PATH) + ".lock"
FLUSH_LOCK_PATH = str(SPOOL_PATH) + ".flush.lock"
LOG_PATH = os.environ.get("SYNAPSE_SPOOL_LOG", "/tmp/synapse-remember-spool.log")
TIMEOUT = float(os.environ.get("SYNAPSE_SPOOL_TIMEOUT", "10"))
PROBE_TIMEOUT = float(os.environ.get("SYNAPSE_SPOOL_PROBE_TIMEOUT", "4"))
BUDGET = float(os.environ.get("SYNAPSE_SPOOL_BUDGET", "12"))
MAX_PER_FLUSH = int(os.environ.get("SYNAPSE_SPOOL_MAX", "25"))

ROUTE = "/remember/spool"
_VALID_TYPES = ("user", "feedback", "project", "reference")


def _log(msg: str) -> None:
    """Same shape as ingest_hook._log: ISO timestamp + message, failures swallowed."""
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now(UTC).isoformat()} {msg}\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Spool file (append-only jsonl + flocked rewrite)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _locked(path: str = LOCK_PATH):
    """Exclusive flock around a spool mutation. Serializes concurrent hooks/CLIs."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as lf:
        lock_exclusive(lf)
        yield


def make_record(
    hook: str | None,
    body: str | None,
    *,
    type: str = "project",
    project: str | None = None,
    session_id: str | None = None,
    content: str | None = None,
    origin: str = "cli",
) -> dict:
    """One spool record. ``id`` is client-generated and is the server's idempotency key —
    it is minted ONCE here and reused by every replay attempt, forever."""
    return {
        "id": uuid.uuid4().hex,
        "ts": datetime.now(UTC).isoformat(),
        "hook": (hook or "").strip() or None,
        "body": (body or "").strip() or None,
        "type": type if type in _VALID_TYPES else "project",
        "project": project or None,
        "session_id": session_id or None,
        "content": (content or "").strip() or None,
        "origin": origin,
    }


def is_writable(rec: dict) -> bool:
    """True if the record carries enough to replay — remember()'s own two forms."""
    return bool((rec.get("hook") and rec.get("body")) or rec.get("content"))


def load() -> list[dict]:
    """Every spooled record, oldest first. Unparseable lines are skipped, not fatal."""
    out: list[dict] = []
    try:
        with open(SPOOL_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if isinstance(rec, dict) and rec.get("id"):
                    out.append(rec)
    except FileNotFoundError:
        return []
    except OSError as e:
        _log(f"ERR load: {type(e).__name__}: {str(e)[:160]}")
        return []
    return out


def append(rec: dict) -> None:
    """Durably append one record. Single line, flocked, fsynced — the whole point."""
    with _locked():
        with open(SPOOL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())


def remove_ids(ids: set[str]) -> int:
    """Drop confirmed intents, KEEPING anything appended since the flush read the file.

    Removal is by id, never by truncation/offset: a concurrent `add` between the read and
    this rewrite must survive, and it does because the rewrite re-reads under the lock."""
    if not ids:
        return 0
    with _locked():
        current = load()
        remaining = [r for r in current if r.get("id") not in ids]
        removed = len(current) - len(remaining)
        tmp = str(SPOOL_PATH) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for r in remaining:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, SPOOL_PATH)
    return removed


# ---------------------------------------------------------------------------
# Server lane
# ---------------------------------------------------------------------------


def probe() -> tuple[bool, str]:
    """Cheap "can I write memory over HTTP right now" check: reachability AND auth, via the
    replay route's no-op probe mode. Deliberately hits the SAME route the flush uses — a
    /health ping would say "up" while a stale bearer still 401s every write."""
    try:
        r = config.post_json(ROUTE, {"probe": True}, timeout=PROBE_TIMEOUT)
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:120]}"
    if r.get("status") == "ok":
        return True, "ok"
    return False, str(r.get("detail") or r)[:120]


def post_intent(rec: dict) -> dict:
    """Replay one intent. Raises on transport/HTTP error — the caller keeps the line."""
    payload = {
        "intent_id": rec["id"],
        "hook": rec.get("hook"),
        "body": rec.get("body"),
        "content": rec.get("content"),
        "type": rec.get("type") or "project",
        "project": rec.get("project"),
        "session_id": rec.get("session_id"),
    }
    return config.post_json(ROUTE, payload, timeout=TIMEOUT)


def flush(*, max_items: int = MAX_PER_FLUSH, budget: float | None = None) -> dict:
    """Replay spooled intents; dequeue ONLY what the server confirmed.

    Returns {"flushed": [rec...], "left": N, "error": str|None, "busy": bool}. Stops at the
    first transport failure (the server is down; the rest would fail too) and at `budget`
    seconds so a SessionStart flush stays inside its hook timeout. A 4xx refusal is NOT a
    transport failure: the payload is unreplayable, so the line is dropped and logged rather
    than retried at every session start forever."""
    records = load()
    if not records:
        return {"flushed": [], "left": 0, "error": None, "busy": False}

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(FLUSH_LOCK_PATH, "w") as lf:
        try:
            lock_exclusive(lf, blocking=False)
        except OSError:
            _log("FLUSH skipped: another flush is running")
            return {"flushed": [], "left": len(records), "error": None, "busy": True}

        started = time.monotonic()
        done_ids: set[str] = set()
        flushed: list[dict] = []
        error: str | None = None
        for rec in records[:max_items]:
            if budget is not None and time.monotonic() - started > budget:
                error = "budget exhausted — remaining intents defer to the next flush"
                break
            if not is_writable(rec):
                # Nothing the server could ever accept; keeping it would block the queue.
                done_ids.add(rec["id"])
                _log(f"DROP unwritable {rec.get('id')} origin={rec.get('origin')}")
                continue
            try:
                resp = post_intent(rec)
            except Exception as e:
                code = getattr(e, "code", None)
                if code is not None and 400 <= int(code) < 500 and int(code) != 401:
                    done_ids.add(rec["id"])
                    _log(f"DROP rejected {rec['id']} HTTP {code}: {str(e)[:120]}")
                    continue
                error = f"{type(e).__name__}: {str(e)[:160]}"
                _log(f"ERR flush {rec['id']}: {error}")
                break
            if resp.get("status") != "ok":
                error = str(resp.get("detail") or resp)[:160]
                _log(f"ERR flush {rec['id']}: {error}")
                break
            done_ids.add(rec["id"])
            rec = dict(rec, note_id=resp.get("note_id"), outcome=resp.get("outcome"))
            flushed.append(rec)
            _log(
                f"OK flush {rec['id']} outcome={resp.get('outcome')} note={resp.get('note_id')} "
                f"hook={(rec.get('hook') or rec.get('content') or '')[:80]!r}"
            )

        removed = remove_ids(done_ids)
        left = len(load())
        if flushed or error:
            _log(f"FLUSH {len(flushed)} written, {removed} dequeued, {left} left, err={error}")
        return {"flushed": flushed, "left": left, "error": error, "busy": False}


# ---------------------------------------------------------------------------
# Capture path 1: PostToolUse on a FAILED remember()
# ---------------------------------------------------------------------------


def _walk(obj, depth: int = 0):
    """Yield every dict nested in a hook tool_response, including dicts encoded as JSON
    inside text blocks (Claude Code wraps MCP results as content blocks of stringified JSON)."""
    if depth > 6:
        return
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v, depth + 1)
    elif isinstance(obj, list | tuple):
        for v in obj:
            yield from _walk(v, depth + 1)
    elif isinstance(obj, str):
        s = obj.strip()
        if s.startswith(("{", "[")):
            try:
                yield from _walk(json.loads(s), depth + 1)
            except Exception:
                return


def response_ok(tool_response) -> bool:
    """True only on POSITIVE confirmation that remember() actually wrote.

    Deliberately not "look for an error flag": an outage can hand the hook an empty, absent,
    or unrecognized response, and treating "I can't tell" as success is precisely how the
    2026-08-25 correction vanished. Confirmation = a note_id, or an explicit status 'ok'.
    The cost of guessing wrong the other way is one redundant replay, which the note
    reconcile absorbs as an update rather than a duplicate."""
    for d in _walk(tool_response):
        if d.get("note_id") is not None:
            return True
        if d.get("status") == "ok" and "detail" not in d:
            return True
    return False


def _record_from_tool_input(ti: dict, payload: dict) -> dict:
    return make_record(
        ti.get("hook"),
        ti.get("body"),
        type=str(ti.get("type") or "project"),
        project=ti.get("project"),
        session_id=ti.get("session_id") or payload.get("session_id"),
        content=ti.get("content"),
        origin="posttooluse",
    )


def _post_tool_use() -> int:
    """Spool the intent when remember() came back anything other than a confirmed write."""
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    if not isinstance(payload, dict):
        return 0
    if response_ok(payload.get("tool_response")):
        return 0
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return 0
    rec = _record_from_tool_input(tool_input, payload)
    if not is_writable(rec):
        return 0  # nothing recoverable — the call was malformed, not lost
    try:
        append(rec)
    except Exception as e:
        _log(f"ERR spool from PostToolUse: {type(e).__name__}: {str(e)[:160]}")
        return 0
    queued = len(load())
    _log(
        f"SPOOLED {rec['id']} origin=posttooluse queued={queued} hook={(rec.get('hook') or '')[:80]!r}"
    )
    msg = (
        f"[Synapse] remember() did NOT confirm a write, so the intent was DURABLY SPOOLED to "
        f"disk (id {rec['id'][:8]}, {queued} queued at {SPOOL_PATH}). It replays automatically "
        f"at the next session start once the server is reachable. Do NOT retry remember() for "
        f"it. Tell the user the memory write is QUEUED, not lost — and not yet saved."
    )
    print(
        json.dumps(
            {"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": msg}}
        )
    )
    return 0


# ---------------------------------------------------------------------------
# SessionStart: probe, flush, report
# ---------------------------------------------------------------------------


def _hook_summary(recs: list[dict], cap: int = 3) -> str:
    labels = [(r.get("hook") or r.get("content") or "")[:60] for r in recs]
    shown = "; ".join(x for x in labels[:cap] if x)
    if len(labels) > cap:
        shown += f"; +{len(labels) - cap} more"
    return shown


def _session_start() -> int:
    """Probe the write lane; flush what's queued; put ONE line in context when it matters.

    Silent in the healthy, empty case — a session start that has nothing to say says nothing.
    """
    with contextlib.suppress(Exception):
        json.load(sys.stdin)  # drain the payload; nothing here needs it

    queued = load()
    ok, detail = probe()

    if not ok:
        script = os.path.abspath(__file__)
        extra = f" {len(queued)} memory write(s) are already queued." if queued else ""
        _log(f"SESSIONSTART server unreachable ({detail}); {len(queued)} queued")
        print(
            f"[Synapse] The memory server is NOT reachable over HTTP ({detail}), so the "
            f"remember() MCP tool is probably down too — a memory write attempted now can be "
            f"silently lost.{extra} Until it is back, route every durable memory write through "
            f'the spool instead: `python3 {script} add --hook "<one-line hook>" --body "<full '
            f'note>" --type user|feedback|project|reference [--project <slug>]`. Spooled writes '
            f"flush automatically at the next session start. Tell the user a write is QUEUED, "
            f"never that it is saved."
        )
        return 0

    if not queued:
        return 0

    res = flush(budget=BUDGET)
    flushed = res["flushed"]
    if flushed:
        line = f"[Synapse] flushed {len(flushed)} spooled memory write(s): {_hook_summary(flushed)}"
        if res["left"]:
            line += f" ({res['left']} still queued — flushing again next session start)"
        _log(f"SESSIONSTART {line}")
        print(line)
    elif res["error"] and not res["busy"]:
        _log(f"SESSIONSTART flush failed: {res['error']}; {res['left']} queued")
        print(
            f"[Synapse] {res['left']} spooled memory write(s) could not be flushed "
            f"({res['error']}). They are safe on disk and retry next session start."
        )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_add_args(argv: list[str]) -> dict:
    """`add` args, or the same fields as JSON on stdin when no flags are given.

    stdin is the ergonomic path for a model writing a multi-line body: shell quoting a
    paragraph correctly is exactly the kind of friction that loses a memory write."""
    flags = {
        "--hook": "hook",
        "--body": "body",
        "--type": "type",
        "--project": "project",
        "--session-id": "session_id",
        "--content": "content",
    }
    out: dict = {}
    i = 0
    while i < len(argv):
        key = flags.get(argv[i])
        if key is None:
            raise ValueError(f"unknown argument {argv[i]!r}")
        if i + 1 >= len(argv):
            raise ValueError(f"{argv[i]} needs a value")
        out[key] = argv[i + 1]
        i += 2
    if not out or argv == ["-"]:
        raw = sys.stdin.read()
        try:
            data = json.loads(raw or "{}")
        except Exception as e:
            raise ValueError(f"stdin is not JSON: {e}") from e
        if not isinstance(data, dict):
            raise ValueError("stdin JSON must be an object")
        out = {k: v for k, v in data.items() if k in flags.values()}
    return out


def _add(argv: list[str]) -> int:
    try:
        fields = _parse_add_args(argv)
    except ValueError as e:
        print(f"remember_spool add: {e}", file=sys.stderr)
        return 2
    bad_type = fields.get("type") and fields["type"] not in _VALID_TYPES
    if bad_type:
        print(
            f"remember_spool add: invalid type {fields['type']!r} — expected one of {_VALID_TYPES}",
            file=sys.stderr,
        )
        return 2
    rec = make_record(
        fields.get("hook"),
        fields.get("body"),
        type=str(fields.get("type") or "project"),
        project=fields.get("project"),
        session_id=fields.get("session_id"),
        content=fields.get("content"),
        origin="cli",
    )
    if not is_writable(rec):
        print(
            "remember_spool add: need --hook AND --body (or --content) — same forms as remember()",
            file=sys.stderr,
        )
        return 2
    try:
        append(rec)
    except OSError as e:
        # The one place this tool must be loud: if the spool itself cannot be written, the
        # memory write is genuinely lost and the caller has to know.
        print(
            f"remember_spool add FAILED to spool ({e}) — the write is NOT saved.", file=sys.stderr
        )
        return 1
    _log(f"SPOOLED {rec['id']} origin=cli hook={(rec.get('hook') or '')[:80]!r}")
    print(f"spooled {rec['id']} ({len(load())} queued)")

    # Try to land it immediately: `add` is also the path used when the MCP tool is simply
    # absent while the HTTP lane is perfectly healthy, and in that case the user should get
    # a real write now, not at the next session start.
    res = flush(max_items=MAX_PER_FLUSH, budget=BUDGET)
    if any(f["id"] == rec["id"] for f in res["flushed"]):
        note = next(f for f in res["flushed"] if f["id"] == rec["id"])
        print(f"written to memory now (note {note.get('note_id')}, {note.get('outcome')})")
    else:
        print(
            f"server unavailable ({res['error'] or 'not reachable'}) — queued, flushes at next session start"
        )
    return 0


def _list() -> int:
    recs = load()
    if not recs:
        print("spool empty")
        return 0
    for r in recs:
        label = (r.get("hook") or r.get("content") or "")[:70]
        print(
            f"{r['id'][:8]}  {r.get('ts', '')}  {r.get('type', '')}  {r.get('origin', '')}  {label}"
        )
    print(f"{len(recs)} queued at {SPOOL_PATH}")
    return 0


def _flush_cli() -> int:
    res = flush()
    for f in res["flushed"]:
        print(f"wrote {f['id'][:8]} -> note {f.get('note_id')} ({f.get('outcome')})")
    if res["busy"]:
        print("another flush is running; nothing done")
        return 0
    print(f"{len(res['flushed'])} flushed, {res['left']} left")
    if res["error"]:
        print(f"stopped: {res['error']}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else ""
    if cmd == "add":
        return _add(argv[1:])
    if cmd == "list":
        return _list()
    if cmd == "flush":
        return _flush_cli()
    if cmd == "post-tool-use":
        return _post_tool_use()
    if cmd == "session-start":
        return _session_start()
    print(__doc__.strip().split("\n\n")[1], file=sys.stderr)
    return 2


if __name__ == "__main__":
    _HOOK_CMDS = ("post-tool-use", "session-start")
    try:
        _rc = main(sys.argv[1:])
    except Exception as _e:  # a hook must never fail the turn
        if len(sys.argv) > 1 and sys.argv[1] in _HOOK_CMDS:
            _log(f"ERR {sys.argv[1]}: {type(_e).__name__}: {str(_e)[:160]}")
            sys.exit(0)
        raise
    sys.exit(0 if len(sys.argv) > 1 and sys.argv[1] in _HOOK_CMDS else _rc)
