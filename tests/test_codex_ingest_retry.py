"""Codex catch-up retries failed uploads however old they are, and says so when they keep failing.

The Codex mirror of test_ingest_hook_retry.py. The Codex Stop hook's sweep used to look only
at rollouts modified within SYNAPSE_CODEX_CATCHUP_DAYS, so an outage longer than that window
(server down, a revoked token, a proxy answering 403) silently lost every rollout that aged
out before uploads recovered. The hook now shares the Claude plugin's failure bookkeeping
(plugin/scripts/synapse_ingest_retry.py): a failed POST leaves a ``pending`` marker on the
rollout's cursor entry, the sweep retries marked rollouts regardless of age, and a sustained
failure streak prints one line at session start.

Pinned here:
  * the hook imports the Claude plugin's module, not a copy;
  * a failed rollout older than the window is retried; a never-attempted one is not;
  * success clears the marker and ends the streak; failure never moves the cursor;
  * the per-sweep cap holds, a rollout that keeps failing rotates to the back, and
    concurrent sweeps collapse to one;
  * pending entries survive the cursor TTL; an old-format cursor file still loads;
  * the warning appears after a streak, disappears after a success, and session_start.py
    emits it as systemMessage + additionalContext;
  * the token never lands in the state file or the warning.

Loaded by path: the hooks run under whatever python3 Codex finds, off the repo path.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import uuid
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parents[1]
_STOP_HOOK = _REPO / "plugin-codex" / "hooks" / "synapse_stop_hook.py"
_SESSION_START = _REPO / "plugin-codex" / "hooks" / "session_start.py"
_SHARED = _REPO / "plugin" / "scripts" / "synapse_ingest_retry.py"

_ENV_VARS = (
    "SYNAPSE_URL",
    "SYNAPSE_INGEST_URL",
    "SYNAPSE_INGEST_TOKEN",
    "SYNAPSE_CODEX_CATCHUP_DAYS",
    "SYNAPSE_CODEX_CATCHUP_MAX",
    "SYNAPSE_CODEX_CATCHUP",
    "SYNAPSE_DATA_DIR",
)

DAY = 86400.0


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def env(monkeypatch, tmp_path) -> Iterator[Path]:
    """HOME, cursor file, log and private dir all under tmp_path; no inherited config.

    The hooks import plugin-codex/scripts/common.py as ``common``, which resolves the
    token at import time, so each test gets a fresh one and the caller's is put back."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SYNAPSE_CODEX_CURSORS", str(tmp_path / "state" / "codex_cursors.json"))
    monkeypatch.setenv("SYNAPSE_CODEX_HOOK_LOG", str(tmp_path / "hook.log"))
    monkeypatch.setenv("SYNAPSE_PRIVATE_DIR", str(tmp_path / "private"))
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    saved = sys.modules.pop("common", None)
    yield home
    sys.modules.pop("common", None)
    if saved is not None:
        sys.modules["common"] = saved


@pytest.fixture()
def hook(env, tmp_path, monkeypatch) -> ModuleType:
    """A fresh Codex Stop-hook module whose rollouts and state live under tmp_path."""
    mod = _load("codex_stop_hook_retry_test", _STOP_HOOK)
    sys.modules.pop("common", None)  # a later load (session_start) resolves its own
    monkeypatch.setattr(mod, "SESSIONS_ROOT", tmp_path / "sessions")
    return mod


def _user(text: str) -> dict:
    return {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        },
    }


def _assistant(text: str) -> dict:
    return {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        },
    }


def _turns3() -> list[dict]:
    return [_user("q1"), _assistant("r1"), _user("q2"), _assistant("r2"), _user("q3"), _assistant("r3")]  # fmt: skip


def _mk(sessions: Path, age_s: float, now: float) -> str:
    """A 3-turn rollout whose mtime is `age_s` seconds before `now`."""
    day = sessions / "2026" / "01" / "31"
    day.mkdir(parents=True, exist_ok=True)
    p = day / f"rollout-2026-01-31T09-12-00-{uuid.uuid4()}.jsonl"
    with open(p, "wb") as f:
        for r in _turns3():
            f.write(json.dumps(r).encode() + b"\n")
    os.utime(p, (now - age_s, now - age_s))
    return str(p)


def _sid(path: str) -> str:
    return path[-len(".jsonl") - 36 : -len(".jsonl")]


def _stop(hook: ModuleType, path: str) -> tuple[int, int]:
    """One Stop-hook ship of `path`, the way the detached child runs it."""
    result: tuple[int, int] = hook._ship({"session_id": _sid(path), "transcript_path": path})
    return result


def _fail(exc: BaseException):
    def post(records, session_id):
        raise exc

    return post


def _ok(posted: list[list[dict]] | None = None):
    def post(records, session_id):
        if posted is not None:
            posted.append(records)
        return {"ingested": len(records)}

    return post


def _http_error(
    code: int, msg: str = "Forbidden", url: str = "http://synapse.example/ingest"
) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, msg, None, None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# One shared code path
# ---------------------------------------------------------------------------


def test_codex_hook_shares_the_claude_plugins_retry_module(hook):
    """The bookkeeping is imported from the Claude plugin, not copied, so the two hosts
    can't drift apart on what is retried or when the warning shows."""
    assert Path(hook.retry.__file__).resolve() == _SHARED.resolve()
    claude_hook = (_REPO / "plugin" / "scripts" / "ingest_hook.py").read_text()
    assert "import synapse_ingest_retry as retry" in claude_hook


# ---------------------------------------------------------------------------
# Retry regardless of age
# ---------------------------------------------------------------------------


def test_failed_old_rollout_is_retried(hook, tmp_path, monkeypatch):
    """The incident: uploads 403'd for longer than the window. A rollout whose upload was
    attempted and failed stays a sweep candidate after it ages out, then ships."""
    now = time.time()
    path = _mk(hook.SESSIONS_ROOT, age_s=3600, now=now)

    monkeypatch.setattr(hook, "_post", _fail(_http_error(403)))
    assert _stop(hook, path) == (0, 0)

    # 90 days later: far outside the window, still owed a retry
    old = now - 90 * DAY
    os.utime(path, (old, old))
    assert hook._catchup_candidates(hook.SESSIONS_ROOT, "", hook._load_state(), now) == [path]

    posted: list[list[dict]] = []
    monkeypatch.setattr(hook, "_post", _ok(posted))
    hook._catchup()
    assert [r["payload"]["content"][0]["text"] for r in posted[0]] == [
        "q1", "r1", "q2", "r2", "q3", "r3"
    ]  # fmt: skip
    ent = hook._load_state()[path]
    assert ent["offset"] == os.path.getsize(path) and "pending" not in ent
    # shipped → no longer a candidate, at any age
    assert hook._catchup_candidates(hook.SESSIONS_ROOT, "", hook._load_state(), now) == []


def test_never_attempted_old_rollout_is_excluded(hook):
    """History from before the hook was installed is codex_backfill's job: an old rollout
    with no state, or one that last shipped cleanly and has no failure on record, is not
    swept just because it exists."""
    now = time.time()
    age = hook.CATCHUP_DAYS * DAY + 3600
    pre_install = _mk(hook.SESSIONS_ROOT, age_s=age, now=now)
    grew_unshipped = _mk(hook.SESSIONS_ROOT, age_s=age, now=now)
    failed = _mk(hook.SESSIONS_ROOT, age_s=age, now=now)
    state = {
        grew_unshipped: {"offset": 10, "ts": now - age},
        failed: {"offset": 10, "ts": now - age, "pending": now - age},
    }
    got = hook._catchup_candidates(hook.SESSIONS_ROOT, "", state, now)
    assert got == [failed]
    assert pre_install not in got and grew_unshipped not in got


def test_in_window_lagging_rollouts_are_still_swept(hook):
    """The window behaviour is unchanged: a recent rollout with bytes past its cursor, or
    none at all, is swept; a fully shipped one is not."""
    now = time.time()
    fresh = _mk(hook.SESSIONS_ROOT, age_s=DAY, now=now)
    lagging = _mk(hook.SESSIONS_ROOT, age_s=2 * DAY, now=now)
    shipped = _mk(hook.SESSIONS_ROOT, age_s=DAY, now=now)
    state = {
        lagging: {"offset": 10, "ts": now - 2 * DAY},
        shipped: {"offset": os.path.getsize(shipped), "ts": now - DAY},
    }
    assert hook._catchup_candidates(hook.SESSIONS_ROOT, "", state, now) == [lagging, fresh]


def test_retry_still_respects_active_grace_and_live_session(hook):
    now = time.time()
    active = _mk(hook.SESSIONS_ROOT, age_s=10, now=now)
    live = _mk(hook.SESSIONS_ROOT, age_s=3600, now=now)
    state = {p: {"offset": 0, "ts": now, "pending": now} for p in (active, live)}
    assert hook._catchup_candidates(hook.SESSIONS_ROOT, live, state, now) == []


def test_failure_on_new_rollout_records_its_start_cursor(hook, monkeypatch):
    """A Stop-path failure on a rollout with no cursor yet records where that run started
    (byte 0 for Codex, which ships a new rollout whole)."""
    path = _mk(hook.SESSIONS_ROOT, age_s=3600, now=time.time())
    monkeypatch.setattr(hook, "_post", _fail(OSError("down")))
    _stop(hook, path)
    ent = hook._load_state()[path]
    assert ent["offset"] == 0 and ent["pending"] and ent["fails"] == 1
    assert ent["last_error"] == "network error"


def test_failure_never_moves_an_existing_cursor(hook, monkeypatch):
    path = _mk(hook.SESSIONS_ROOT, age_s=3600, now=time.time())
    with open(path, "rb") as f:
        second_turn = sum(len(x) for x in f.readlines()[:2])
    hook._save_cursor(path, second_turn)
    with open(path, "ab") as f:  # a new turn arrives, then its upload fails
        f.write(json.dumps(_user("q4")).encode() + b"\n")
    monkeypatch.setattr(hook, "_post", _fail(_http_error(503, "Unavailable")))
    _stop(hook, path)
    ent = hook._load_state()[path]
    assert ent["offset"] == second_turn
    assert ent["pending"] and ent["fails"] == 1


# ---------------------------------------------------------------------------
# Success clears the marker
# ---------------------------------------------------------------------------


def test_success_clears_marker_and_streak(hook, monkeypatch):
    path = _mk(hook.SESSIONS_ROOT, age_s=3600, now=time.time())

    monkeypatch.setattr(hook, "_post", _fail(_http_error(503, "Unavailable")))
    _stop(hook, path)
    _stop(hook, path)
    state = hook._load_state()
    first_since = state[path]["pending"]
    assert state[path]["fails"] == 2
    assert state[path]["last_error"] == "HTTP 503 Service Unavailable"
    assert state[hook.retry.HEALTH_KEY]["fails"] == 2
    assert state[path]["tried"] >= first_since  # pending keeps the FIRST failure time

    monkeypatch.setattr(hook, "_post", _ok())
    assert _stop(hook, path) == (1, 6)
    state = hook._load_state()
    ent = state[path]
    for key in ("pending", "tried", "fails", "last_error"):
        assert key not in ent
    assert ent["offset"] == os.path.getsize(path)
    assert "fail_since" not in state[hook.retry.HEALTH_KEY]
    assert state[hook.retry.HEALTH_KEY]["last_ok"]


def test_pending_entries_survive_the_cursor_ttl(hook):
    """The TTL drops idle cursors; a rollout still owed a retry must not be dropped with
    them, or a long outage would quietly forget what it owes."""
    now = time.time()
    ancient = now - (hook._CURSOR_TTL_DAYS + 30) * DAY
    owed = _mk(hook.SESSIONS_ROOT, age_s=90 * DAY, now=now)
    idle = _mk(hook.SESSIONS_ROOT, age_s=90 * DAY, now=now)
    other = _mk(hook.SESSIONS_ROOT, age_s=3600, now=now)
    hook.CURSORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    hook.CURSORS_PATH.write_text(
        json.dumps(
            {
                owed: {"offset": 0, "ts": ancient, "pending": ancient},
                idle: {"offset": 5, "ts": ancient},
                str(hook.SESSIONS_ROOT / "deleted.jsonl"): {"offset": 5, "ts": now},
            }
        )
    )
    hook._save_cursor(other, 1)  # any write prunes
    state = hook._load_state()
    assert owed in state and other in state
    assert idle not in state and str(hook.SESSIONS_ROOT / "deleted.jsonl") not in state


def test_old_format_cursor_file_still_loads(hook, monkeypatch):
    """Cursor files written before this change ({path: {offset, ts}}) keep working: the
    hook resumes from the stored offset instead of re-shipping the whole rollout."""
    path = _mk(hook.SESSIONS_ROOT, age_s=3600, now=time.time())
    with open(path, "rb") as f:
        lines = f.readlines()
    third_turn = sum(len(x) for x in lines[:4])
    hook.CURSORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    hook.CURSORS_PATH.write_text(json.dumps({path: {"offset": third_turn, "ts": time.time()}}))
    posted: list[list[dict]] = []
    monkeypatch.setattr(hook, "_post", _ok(posted))
    assert _stop(hook, path) == (1, 2)
    assert [r["payload"]["content"][0]["text"] for r in posted[0]] == ["q3", "r3"]


# ---------------------------------------------------------------------------
# Cap, ordering, lock
# ---------------------------------------------------------------------------


def test_cap_holds_and_failing_rollouts_rotate(hook, monkeypatch):
    """Least recently touched first (last failed attempt, else mtime), never more than
    CATCHUP_MAX per sweep, and a rollout that just failed goes to the back: rollouts that
    keep failing can't hold the sweep's slots forever."""
    now = time.time()
    a = _mk(hook.SESSIONS_ROOT, age_s=60 * DAY, now=now)
    b = _mk(hook.SESSIONS_ROOT, age_s=50 * DAY, now=now)
    c = _mk(hook.SESSIONS_ROOT, age_s=2 * DAY, now=now)  # in-window, never attempted
    d = _mk(hook.SESSIONS_ROOT, age_s=40 * DAY, now=now)
    hook.CURSORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    hook.CURSORS_PATH.write_text(
        json.dumps(
            {
                a: {"offset": 0, "ts": now, "pending": now - 60 * DAY, "tried": now - 10 * DAY},
                b: {"offset": 0, "ts": now, "pending": now - 50 * DAY, "tried": now - 20 * DAY},
                d: {"offset": 0, "ts": now, "pending": now - 40 * DAY, "tried": now - 1 * DAY},
            }
        )
    )  # fmt: skip
    # waiting longest first: b (tried 20d ago), a (10d), c (mtime 2d), d (tried 1d)
    assert hook._catchup_candidates(hook.SESSIONS_ROOT, "", hook._load_state(), now) == [
        b, a, c, d
    ]  # fmt: skip

    monkeypatch.setattr(hook, "CATCHUP_MAX_FILES", 2)
    monkeypatch.setattr(hook, "_post", _fail(_http_error(403)))
    attempted: list[str] = []
    real_ship = hook._ship

    def ship(payload):
        attempted.append(payload["transcript_path"])
        assert payload["session_id"] == _sid(payload["transcript_path"])
        return real_ship(payload)

    monkeypatch.setattr(hook, "_ship", ship)
    hook._catchup()
    assert attempted == [b, a]  # capped at 2
    hook._catchup()
    assert attempted == [b, a, c, d]  # the two that just failed moved to the back
    assert "2 deferred to next session start" in Path(hook.LOG_PATH).read_text()


def test_concurrent_sweeps_collapse_to_one(hook, monkeypatch):
    _mk(hook.SESSIONS_ROOT, age_s=DAY, now=time.time())
    shipped: list[dict] = []
    monkeypatch.setattr(hook, "_ship", lambda payload: shipped.append(payload) or (0, 0))
    hook.CURSORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(str(hook.CURSORS_PATH) + ".catchup.lock", "w") as held:
        hook.retry.lock_exclusive(held)
        hook._catchup()  # another sweep holds the lock → skip, don't queue
    assert shipped == []
    hook._catchup()
    assert len(shipped) == 1


def test_catchup_entry_passes_the_live_rollout_to_skip(hook, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(hook, "_catchup", calls.append)
    monkeypatch.setattr(sys, "argv", ["synapse_stop_hook.py", "--catchup", "/x/live.jsonl"])
    assert hook.main() == 0
    monkeypatch.setattr(sys, "argv", ["synapse_stop_hook.py", "--catchup"])
    assert hook.main() == 0
    assert calls == ["/x/live.jsonl", ""]


# ---------------------------------------------------------------------------
# Session-start warning
# ---------------------------------------------------------------------------


def test_warning_appears_after_a_streak_and_clears_on_success(hook, monkeypatch):
    path = _mk(hook.SESSIONS_ROOT, age_s=3600, now=time.time())
    assert hook.failure_warning() is None  # nothing on record

    monkeypatch.setattr(hook, "_post", _fail(_http_error(403)))
    for _ in range(hook.retry.WARN_AFTER_FAILS - 1):
        _stop(hook, path)
    assert hook.failure_warning() is None  # a blip

    _stop(hook, path)
    line = hook.failure_warning()
    assert line is not None
    since = time.strftime(
        "%Y-%m-%d", time.localtime(hook._load_state()[hook.retry.HEALTH_KEY]["fail_since"])
    )
    assert line.startswith(f"Synapse uploads have been failing since {since} ")
    assert "(HTTP 403 Forbidden)" in line
    assert line.endswith("transcripts will be retried automatically.")
    assert "\n" not in line

    monkeypatch.setattr(hook, "_post", _ok())
    _stop(hook, path)
    assert hook.failure_warning() is None


def test_warning_after_a_long_silent_failure(hook):
    """One failure is a blip, unless nothing has succeeded for a day since."""
    now = time.time()
    hook.CURSORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    health = {"fail_since": now - 3600, "fails": 1, "last_error": "timed out"}
    hook.CURSORS_PATH.write_text(json.dumps({hook.retry.HEALTH_KEY: health}))
    assert hook.failure_warning(now) is None
    later = now + hook.retry.WARN_AFTER_HOURS * 3600
    assert "(timed out)" in (hook.failure_warning(later) or "")


@pytest.fixture()
def session_start(env, monkeypatch) -> ModuleType:
    """session_start.py loaded fresh, with its network blocks stubbed out."""
    sys.modules.pop("common", None)
    mod = _load("codex_session_start_retry_test", _SESSION_START)
    sys.modules.pop("common", None)
    monkeypatch.setattr(mod, "_prefs_text", lambda: None)
    monkeypatch.setattr(mod, "_board_text", lambda project: None)
    monkeypatch.setattr(mod, "_sync_skills", lambda: None)
    return mod


def _run_session_start(mod, monkeypatch, capsys, payload: dict) -> str:
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    mod.main()
    out: str = capsys.readouterr().out
    return out


def test_session_start_prints_warning_for_user_and_model(hook, session_start, monkeypatch, capsys):
    """systemMessage is what Codex shows the user; additionalContext is what the model
    reads. The warning goes to both, ahead of the board, and the sweep still starts."""
    path = _mk(hook.SESSIONS_ROOT, age_s=3600, now=time.time())
    monkeypatch.setattr(hook, "_post", _fail(_http_error(401, "Unauthorized")))
    for _ in range(hook.retry.WARN_AFTER_FAILS):
        _stop(hook, path)

    spawned: list[str] = []
    monkeypatch.setattr(session_start, "_spawn_catchup", spawned.append)
    monkeypatch.setattr(session_start, "_board_text", lambda project: "[board]")
    out = json.loads(
        _run_session_start(session_start, monkeypatch, capsys, {"transcript_path": path})
    )
    msg = out["systemMessage"]
    assert msg.startswith("Synapse uploads have been failing since ")
    assert "(HTTP 401 Unauthorized)" in msg
    assert out["hookSpecificOutput"] == {
        "hookEventName": "SessionStart",
        "additionalContext": f"{msg}\n\n[board]",
    }
    assert spawned == [path]  # the live rollout is passed along for the sweep to skip


def test_session_start_reads_the_warning_before_the_sweep_runs(
    hook, session_start, monkeypatch, capsys
):
    """A sweep that succeeds ends the streak, so the line is read first: otherwise a fast
    sweep could hide the warning for the very outage it reports."""
    order: list[str] = []
    real_warning = session_start._upload_warning
    monkeypatch.setattr(
        session_start, "_upload_warning", lambda: order.append("warning") or real_warning()
    )
    monkeypatch.setattr(session_start, "_spawn_catchup", lambda skip: order.append("sweep"))
    _run_session_start(session_start, monkeypatch, capsys, {})
    assert order == ["warning", "sweep"]


def test_session_start_is_silent_when_healthy(hook, session_start, monkeypatch, capsys):
    path = _mk(hook.SESSIONS_ROOT, age_s=3600, now=time.time())
    monkeypatch.setattr(hook, "_post", _ok())
    _stop(hook, path)
    monkeypatch.setattr(session_start, "_spawn_catchup", lambda skip: None)
    assert _run_session_start(session_start, monkeypatch, capsys, {}) == ""
    monkeypatch.setattr(session_start, "_board_text", lambda project: "[board]")
    out = json.loads(_run_session_start(session_start, monkeypatch, capsys, {}))
    assert "systemMessage" not in out
    assert out["hookSpecificOutput"]["additionalContext"] == "[board]"


def test_session_start_end_to_end(env, tmp_path):
    """The real hook, run the way Codex runs it: the shared module resolves from the
    checkout and the warning reaches stdout as one JSON envelope."""
    now = time.time()
    state = tmp_path / "state" / "codex_cursors.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    health = {"fail_since": now - 2 * DAY, "fails": 7, "last_error": "connection refused"}
    state.write_text(json.dumps({"_health": health}))
    run_env = {
        **os.environ,
        "SYNAPSE_URL": "http://127.0.0.1:9",
        "SYNAPSE_BOARD": "0",
        "SYNAPSE_PREFS_BLOCK": "0",
        "SYNAPSE_CODEX_CATCHUP": "0",
        "SYNAPSE_SKILLS_SYNC": "0",
    }
    proc = subprocess.run(
        [sys.executable, str(_SESSION_START)],
        input=json.dumps({"cwd": str(tmp_path)}),
        capture_output=True,
        text=True,
        env=run_env,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert "(connection refused)" in out["systemMessage"]
    assert out["hookSpecificOutput"]["additionalContext"] == out["systemMessage"]


# ---------------------------------------------------------------------------
# Never the token
# ---------------------------------------------------------------------------


def test_token_never_lands_in_state_or_warning(hook, session_start, monkeypatch, capsys):
    secret = "tok-SECRET-0123456789abcdef"
    monkeypatch.setattr(hook, "TOKEN", secret)
    path = _mk(hook.SESSIONS_ROOT, age_s=3600, now=time.time())

    leaky = [
        _http_error(403, f"bad {secret}", url=f"http://synapse.example/ingest?t={secret}"),
        urllib.error.URLError(f"refused for Bearer {secret}"),
        RuntimeError(f"Authorization: Bearer {secret}"),
    ]
    for exc in leaky:
        monkeypatch.setattr(hook, "_post", _fail(exc))
        _stop(hook, path)
    # a reason recorded directly (a future caller) is scrubbed too
    hook._record_failure(path, f"Bearer {secret}", 0)

    raw = hook.CURSORS_PATH.read_text()
    assert secret not in raw
    line = hook.failure_warning()
    assert line and secret not in line

    monkeypatch.setenv("SYNAPSE_INGEST_TOKEN", secret)  # what session_start's fresh load sees
    monkeypatch.setattr(session_start, "_spawn_catchup", lambda skip: None)
    out = _run_session_start(session_start, monkeypatch, capsys, {})
    assert out and secret not in out
