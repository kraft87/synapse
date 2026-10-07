"""Catch-up retries failed uploads however old they are, and says so when they keep failing.

The hook fails soft by design, and the sweep used to look only at files modified within
SYNAPSE_INGEST_CATCHUP_DAYS. An outage longer than that window (server down, a revoked
token, a proxy answering 403) lost every transcript that aged out before the uploads
recovered. Now a failed POST leaves a ``pending`` marker on the file's cursor entry, the
sweep retries marked files regardless of age, and a sustained failure streak prints one
line at session start.

Pinned here:
  * a failed file older than the window is retried; a never-attempted one is not
    (pre-install history is synapse-import's job);
  * success clears the marker and ends the streak;
  * the per-sweep cap holds, and a file that keeps failing rotates to the back;
  * pending entries survive the cursor TTL;
  * the warning appears after a streak, disappears after a success, and goes out
    as systemMessage + additionalContext;
  * the token never lands in the state file or the warning.

Loaded by path like test_ingest_hook_cursor.py: the hook runs under the CLI's bare Python.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import socket
import sys
import time
import urllib.error
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parents[1]
_PLUGIN_HOOK = _REPO / "plugin" / "scripts" / "ingest_hook.py"

_PLUGIN_ENV_VARS = (
    "SYNAPSE_URL",
    "SYNAPSE_INGEST_URL",
    "SYNAPSE_INGEST_TOKEN",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_URL",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_URL",
    "CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_TOKEN",
)

DAY = 86400.0


@pytest.fixture()
def hook(monkeypatch, tmp_path) -> ModuleType:
    """A fresh plugin-hook module whose config/state all live under tmp_path."""
    cfg_dir = tmp_path / "claude"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(cfg_dir))
    monkeypatch.setenv("SYNAPSE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SYNAPSE_INGEST_LOG", str(tmp_path / "hook.log"))
    monkeypatch.chdir(tmp_path)
    for var in _PLUGIN_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    sys.modules.pop("config", None)
    spec = importlib.util.spec_from_file_location("ingest_hook_retry_test", _PLUGIN_HOOK)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules.pop("config", None)
    return mod


def _u(uid: str) -> dict:
    return {"type": "user", "uuid": uid, "sessionId": "s1", "message": {"content": "q"}}


def _a(uid: str) -> dict:
    return {
        "type": "assistant",
        "uuid": uid,
        "sessionId": "s1",
        "message": {"content": [{"type": "text", "text": "r"}]},
    }


def _turns3() -> list[dict]:
    return [_u("u1"), _a("a1"), _u("u2"), _a("a2"), _u("u3"), _a("a3")]


def _mk(proj: Path, name: str, age_s: float, now: float) -> str:
    """A 3-turn transcript whose mtime is `age_s` seconds before `now`."""
    p = proj / name
    with open(p, "wb") as f:
        for r in _turns3():
            f.write(json.dumps(r).encode() + b"\n")
    os.utime(p, (now - age_s, now - age_s))
    return str(p)


def _projects(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "projects"
    proj = root / "-home-user-dev"
    proj.mkdir(parents=True)
    return root, proj


def _fail(exc: BaseException):
    def post(recs, source="hook"):
        raise exc

    return post


def _http_error(
    code: int, msg: str = "Forbidden", url: str = "http://synapse.example/ingest"
) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, msg, None, None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Retry regardless of age
# ---------------------------------------------------------------------------


def test_failed_old_file_is_retried(hook, tmp_path, monkeypatch):
    """The incident: uploads 403'd for longer than the window. A file whose upload was
    attempted and failed stays a sweep candidate after it ages out, then ships."""
    root, proj = _projects(tmp_path)
    now = time.time()
    path = _mk(proj, "failed.jsonl", age_s=3600, now=now)

    monkeypatch.setattr(hook, "_post_records", _fail(_http_error(403)))
    assert hook._ship(path, mode="stop") == (0, 0)

    # 90 days later: far outside the window, still owed a retry
    old = now - 90 * DAY
    os.utime(path, (old, old))
    assert hook._catchup_candidates(str(root), "", hook._load_state(), now) == [path]

    posted: list[list[dict]] = []
    monkeypatch.setattr(
        hook, "_post_records", lambda recs, source="hook": posted.append(recs) or "ok"
    )
    hook._catchup(str(root), "")
    assert [r["uuid"] for r in posted[0]] == ["u1", "a1", "u2", "a2", "u3", "a3"]
    ent = hook._load_state()[path]
    assert ent["size"] == os.path.getsize(path)
    # shipped → no longer a candidate, at any age
    assert hook._catchup_candidates(str(root), "", hook._load_state(), now) == []


def test_partially_shipped_old_file_is_retried(hook, tmp_path):
    """A backlog that stopped partway (size -1, e.g. the process was killed between
    chunks) was attempted and did not complete, so it is retried at any age too."""
    root, proj = _projects(tmp_path)
    now = time.time()
    path = _mk(proj, "partial.jsonl", age_s=30 * DAY, now=now)
    state = {path: {"offset": 0, "size": -1, "ts": now - 30 * DAY}}
    assert hook._catchup_candidates(str(root), "", state, now) == [path]


def test_never_attempted_old_file_is_excluded(hook, tmp_path):
    """History from before the plugin was installed is synapse-import's job: an old file
    with no state, or one that last shipped cleanly and has no failure on record, is not
    swept just because it exists."""
    root, proj = _projects(tmp_path)
    now = time.time()
    age = hook.CATCHUP_DAYS * DAY + 3600
    pre_install = _mk(proj, "pre_install.jsonl", age_s=age, now=now)
    grew_unshipped = _mk(proj, "grew.jsonl", age_s=age, now=now)
    failed = _mk(proj, "failed.jsonl", age_s=age, now=now)
    state = {
        grew_unshipped: {"offset": 0, "size": 10, "ts": now - age},
        failed: {"offset": 0, "size": 10, "ts": now - age, "pending": now - age},
    }
    got = hook._catchup_candidates(str(root), "", state, now)
    assert got == [failed]
    assert pre_install not in got and grew_unshipped not in got


def test_retry_still_respects_active_grace_and_live_session(hook, tmp_path):
    root, proj = _projects(tmp_path)
    now = time.time()
    active = _mk(proj, "active.jsonl", age_s=10, now=now)
    live = _mk(proj, "live.jsonl", age_s=3600, now=now)
    state = {p: {"offset": 0, "size": -1, "ts": now, "pending": now} for p in (active, live)}
    assert hook._catchup_candidates(str(root), live, state, now) == []


def test_stop_failure_on_new_file_records_its_seed_cursor(hook, tmp_path, monkeypatch):
    """A Stop-path failure on a file with no cursor yet records where that run started,
    so the retry resumes from the same seed rather than inventing a new one."""
    _, proj = _projects(tmp_path)
    path = _mk(proj, "s.jsonl", age_s=3600, now=time.time())
    monkeypatch.setattr(hook, "TAIL_RECORDS", 3)  # tail window [a2, u3, a3] → seed at u3
    monkeypatch.setattr(hook, "_post_records", _fail(OSError("down")))
    hook._ship(path, mode="stop")
    ent = hook._load_state()[path]
    with open(path, "rb") as f:
        lines = f.readlines()
    u3_offset = sum(len(x) for x in lines[:4])
    assert ent["offset"] == u3_offset and ent["size"] == -1 and ent["pending"]


def test_failure_never_moves_an_existing_cursor(hook, tmp_path, monkeypatch):
    _, proj = _projects(tmp_path)
    path = _mk(proj, "s.jsonl", age_s=3600, now=time.time())
    hook._advance_cursor(path, 123, 456)
    monkeypatch.setattr(hook, "_post_records", _fail(OSError("down")))
    hook._ship(path, mode="catchup")
    ent = hook._load_state()[path]
    assert (ent["offset"], ent["size"]) == (123, 456)
    assert ent["pending"] and ent["fails"] == 1


# ---------------------------------------------------------------------------
# Success clears the marker
# ---------------------------------------------------------------------------


def test_success_clears_marker_and_streak(hook, tmp_path, monkeypatch):
    _, proj = _projects(tmp_path)
    path = _mk(proj, "s.jsonl", age_s=3600, now=time.time())

    monkeypatch.setattr(hook, "_post_records", _fail(_http_error(503, "Unavailable")))
    hook._ship(path, mode="catchup")
    hook._ship(path, mode="catchup")
    state = hook._load_state()
    first_since = state[path]["pending"]
    assert state[path]["fails"] == 2
    assert state[path]["last_error"] == "HTTP 503 Service Unavailable"
    assert state[hook._HEALTH_KEY]["fails"] == 2
    # the marker keeps the FIRST failure time; `tried` moves on every failure
    assert state[path]["tried"] >= first_since

    monkeypatch.setattr(hook, "_post_records", lambda recs, source="hook": "ok")
    assert hook._ship(path, mode="catchup") == (1, 6)
    state = hook._load_state()
    ent = state[path]
    for key in ("pending", "tried", "fails", "last_error"):
        assert key not in ent
    assert ent["size"] == os.path.getsize(path)
    assert "fail_since" not in state[hook._HEALTH_KEY]
    assert state[hook._HEALTH_KEY]["last_ok"]


def test_pending_entries_survive_the_cursor_ttl(hook, tmp_path):
    """The TTL drops idle cursors; a file still owed a retry must not be dropped with
    them, or a long outage would quietly forget what it owes."""
    _, proj = _projects(tmp_path)
    now = time.time()
    ancient = now - (hook._CURSOR_TTL_DAYS + 30) * DAY
    owed = _mk(proj, "owed.jsonl", age_s=90 * DAY, now=now)
    idle = _mk(proj, "idle.jsonl", age_s=90 * DAY, now=now)
    other = _mk(proj, "other.jsonl", age_s=3600, now=now)
    hook.config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    hook.CURSOR_PATH.write_text(
        json.dumps(
            {
                owed: {"offset": 0, "size": -1, "ts": ancient, "pending": ancient},
                idle: {"offset": 0, "size": 5, "ts": ancient},
            }
        )
    )
    hook._advance_cursor(other, 0, 1)  # any write prunes
    state = hook._load_state()
    assert owed in state and idle not in state and other in state


# ---------------------------------------------------------------------------
# Cap + ordering
# ---------------------------------------------------------------------------


def test_cap_holds_and_failing_files_rotate(hook, tmp_path, monkeypatch):
    """Least recently touched first (last failed attempt, else mtime), never more than
    CATCHUP_MAX per sweep, and a file that just failed goes to the back: files that keep
    failing can't hold the sweep's slots forever."""
    root, proj = _projects(tmp_path)
    now = time.time()
    a = _mk(proj, "a.jsonl", age_s=60 * DAY, now=now)
    b = _mk(proj, "b.jsonl", age_s=50 * DAY, now=now)
    c = _mk(proj, "c.jsonl", age_s=2 * DAY, now=now)  # in-window, never attempted
    d = _mk(proj, "d.jsonl", age_s=40 * DAY, now=now)
    hook.config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    hook.CURSOR_PATH.write_text(
        json.dumps(
            {
                a: {"offset": 0, "size": -1, "ts": now, "pending": now - 60 * DAY,
                    "tried": now - 10 * DAY},
                b: {"offset": 0, "size": -1, "ts": now, "pending": now - 50 * DAY,
                    "tried": now - 20 * DAY},
                d: {"offset": 0, "size": -1, "ts": now, "pending": now - 40 * DAY,
                    "tried": now - 1 * DAY},
            }
        )
    )  # fmt: skip
    # waiting longest first: b (tried 20d ago), a (10d), c (mtime 2d), d (tried 1d)
    assert hook._catchup_candidates(str(root), "", hook._load_state(), now) == [b, a, c, d]

    monkeypatch.setattr(hook, "CATCHUP_MAX_FILES", 2)
    attempted: list[str] = []

    def ship(path, mode="stop"):
        attempted.append(path)
        hook._record_failure(path, "HTTP 403 Forbidden", 0)
        return (0, 0)

    monkeypatch.setattr(hook, "_ship", ship)
    hook._catchup(str(root), "")
    assert attempted == [b, a]  # capped at 2
    hook._catchup(str(root), "")
    assert attempted == [b, a, c, d]  # the two that just failed moved to the back


# ---------------------------------------------------------------------------
# Session-start warning
# ---------------------------------------------------------------------------


def test_warning_appears_after_a_streak_and_clears_on_success(hook, tmp_path, monkeypatch):
    _, proj = _projects(tmp_path)
    path = _mk(proj, "s.jsonl", age_s=3600, now=time.time())
    now = time.time()
    assert hook._failure_warning(hook._load_state(), now) is None  # nothing on record

    monkeypatch.setattr(hook, "_post_records", _fail(_http_error(403)))
    for _ in range(hook._WARN_AFTER_FAILS - 1):
        hook._ship(path, mode="stop")
    assert hook._failure_warning(hook._load_state(), time.time()) is None  # a blip

    hook._ship(path, mode="stop")
    state = hook._load_state()
    line = hook._failure_warning(state, time.time())
    assert line is not None
    since = time.strftime("%Y-%m-%d", time.localtime(state[hook._HEALTH_KEY]["fail_since"]))
    assert line.startswith(f"Synapse uploads have been failing since {since} ")
    assert "(HTTP 403 Forbidden)" in line
    assert line.endswith("transcripts will be retried automatically.")
    assert "\n" not in line

    monkeypatch.setattr(hook, "_post_records", lambda recs, source="hook": "ok")
    hook._ship(path, mode="stop")
    assert hook._failure_warning(hook._load_state(), time.time()) is None


def test_warning_after_a_long_silent_failure(hook):
    """One failure is a blip, unless nothing has succeeded for a day since."""
    now = time.time()
    fresh = {hook._HEALTH_KEY: {"fail_since": now - 3600, "fails": 1, "last_error": "timed out"}}
    stale = {
        hook._HEALTH_KEY: {
            "fail_since": now - hook._WARN_AFTER_HOURS * 3600 - 60,
            "fails": 1,
            "last_error": "timed out",
        }
    }
    assert hook._failure_warning(fresh, now) is None
    assert "(timed out)" in (hook._failure_warning(stale, now) or "")


def test_catchup_hook_prints_warning_for_user_and_model(hook, tmp_path, monkeypatch, capsys):
    """`--catchup` (the SessionStart entry) prints the line as systemMessage (the user
    sees it) and additionalContext (the model sees it), then still spawns the sweep."""
    root, proj = _projects(tmp_path)
    path = _mk(proj, "s.jsonl", age_s=3600, now=time.time())
    monkeypatch.setattr(hook, "_post_records", _fail(_http_error(401, "Unauthorized")))
    for _ in range(hook._WARN_AFTER_FAILS):
        hook._ship(path, mode="stop")

    spawned: list[list[str]] = []
    monkeypatch.setattr(hook, "_spawn_detached", spawned.append)
    monkeypatch.setattr(sys, "argv", ["ingest_hook.py", "--catchup"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"transcript_path": path})))
    hook.main()
    out = json.loads(capsys.readouterr().out)
    assert out["systemMessage"].startswith("Synapse uploads have been failing since ")
    assert "(HTTP 401 Unauthorized)" in out["systemMessage"]
    assert out["hookSpecificOutput"] == {
        "hookEventName": "SessionStart",
        "additionalContext": out["systemMessage"],
    }
    assert spawned and spawned[0][:2] == ["--catchup-run", str(root)]


def test_catchup_hook_is_silent_when_healthy(hook, tmp_path, monkeypatch, capsys):
    _, proj = _projects(tmp_path)
    path = _mk(proj, "s.jsonl", age_s=3600, now=time.time())
    monkeypatch.setattr(hook, "_spawn_detached", lambda args: None)
    monkeypatch.setattr(sys, "argv", ["ingest_hook.py", "--catchup"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"transcript_path": path})))
    hook.main()
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Reasons: short, fixed vocabulary, never the token
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "reason"),
    [
        (_http_error(403), "HTTP 403 Forbidden"),
        (_http_error(599, "?"), "HTTP 599"),
        (urllib.error.URLError(ConnectionRefusedError(111, "refused")), "connection refused"),
        (urllib.error.URLError(socket.gaierror(-2, "Name or service not known")), "DNS lookup failed"),
        (urllib.error.URLError(TimeoutError()), "timed out"),
        (TimeoutError("read timed out"), "timed out"),
        (ConnectionResetError(104, "reset"), "connection dropped"),
        (urllib.error.URLError("no host given"), "server unreachable"),
        (ValueError("anything at all"), "ValueError"),
    ],
)  # fmt: skip
def test_short_reason_vocabulary(hook, exc, reason):
    assert hook._short_reason(exc) == reason


def test_token_never_lands_in_state_or_warning(hook, tmp_path, monkeypatch, capsys):
    secret = "tok-SECRET-0123456789abcdef"
    monkeypatch.setattr(hook, "INGEST_TOKEN", secret)
    _, proj = _projects(tmp_path)
    path = _mk(proj, "s.jsonl", age_s=3600, now=time.time())

    leaky = [
        _http_error(403, f"bad {secret}", url=f"http://synapse.example/ingest?t={secret}"),
        urllib.error.URLError(f"refused for Bearer {secret}"),
        RuntimeError(f"Authorization: Bearer {secret}"),
    ]
    for exc in leaky:
        monkeypatch.setattr(hook, "_post_records", _fail(exc))
        hook._ship(path, mode="stop")
    # a reason recorded directly (a future caller) is scrubbed too
    hook._record_failure(path, f"Bearer {secret}", 0)

    raw = hook.CURSOR_PATH.read_text()
    assert secret not in raw
    line = hook._failure_warning(json.loads(raw), time.time())
    assert line and secret not in line

    monkeypatch.setattr(hook, "_spawn_detached", lambda args: None)
    monkeypatch.setattr(sys, "argv", ["ingest_hook.py", "--catchup"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"transcript_path": path})))
    hook.main()
    out = capsys.readouterr().out
    assert out and secret not in out
