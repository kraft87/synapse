"""Every client hook names a project the way plugin/scripts/synapse_project.py does.

Labels used to come from four places: the server's two transcript parsers, the board
hook and the Codex session hook, each taking the basename of whatever cwd it saw. A
session started in ``<repo>/sub/deeper`` was filed as ``deeper``, a linked worktree as
the worktree's folder name, and a Windows cwd parsed on the server as the whole path.
Each hook now asks the one resolver (marker file, else git toplevel with worktrees
folded into their main checkout, else the folder name) and sends or uses its answer:

  * ingest_hook._post_records: the distinct record cwds resolve to
    one label -> ``project`` in the /ingest body; none or several -> no key at all
  * Codex Stop hook: _rollout_cwd (payload cwd, else the newest session_meta or
    turn_context in the shipped records, else the rollout's first line), and _ship
    hands the resolved label to _post, which puts it in the body only when set
  * board_block and the Codex session_start hook: the board's ``project`` param
  * skills_sync.main: the ``project:<label>`` scope (the target folder is unchanged)
  * skill_review._local_dir: matches that label case-insensitively
  * config_sync._targets and git_feeder._push_repo: the same label

The repositories are real (``git init`` plus ``git worktree add`` in a tmp dir), so
the resolver is proved against the layout git writes, not a hand-made ``.git`` file.
The hooks that ship or fetch also run as real processes against a local HTTP server,
and the test asserts on the request that server received.

No real tokens, no network beyond 127.0.0.1, every config/state dir under tmp_path.
"""

from __future__ import annotations

import http.server
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tests.helpers.http import LoopbackServer

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "plugin" / "scripts"
_CODEX_HOOKS = _REPO / "plugin-codex" / "hooks"
_INGEST_HOOK = _SCRIPTS / "ingest_hook.py"
_BOARD_BLOCK = _SCRIPTS / "board_block.py"
_SKILLS_SYNC = _SCRIPTS / "skills_sync.py"
_SKILL_REVIEW = _SCRIPTS / "skill_review.py"
_CONFIG_SYNC = _SCRIPTS / "config_sync.py"
_GIT_FEEDER = _SCRIPTS / "git_feeder.py"
_STOP_HOOK = _CODEX_HOOKS / "synapse_stop_hook.py"
_SESSION_START = _CODEX_HOOKS / "session_start.py"

# A placeholder bearer for the local test server; not a credential anywhere.
_TOKEN = "test-placeholder-ingest-token"
_WIN_CWD = "C:\\Users\\dev\\src\\Acme"

_SETTINGS = (
    "SYNAPSE_URL",
    "SYNAPSE_INGEST_URL",
    "SYNAPSE_INGEST_TOKEN",
    "SYNAPSE_RECALL_URL",
    "SYNAPSE_MCP_URL",
    "SYNAPSE_BOARD",
    "SYNAPSE_PREFS_BLOCK",
    "SYNAPSE_SKILLS_SYNC",
    "SYNAPSE_CONFIG_SYNC",
    "SYNAPSE_CONFIG_PATHS",
    "SYNAPSE_TIMELINE_REPOS",
    "SYNAPSE_TIMELINE_SINCE_DAYS",
    "SYNAPSE_MACHINE_ROLE",
)
_ENV_VARS = (
    *_SETTINGS,
    *(f"CLAUDE_PLUGIN_OPTION_{k}" for k in _SETTINGS),
    "CLAUDE_PROJECT_DIR",
    "CLAUDE_PROJECTS_DIR",
    "SYNAPSE_CODEX_SKILLS_DIR",
)
# Modules the scripts import by bare name that resolve config at import time.
_ENV_MODULES = ("config", "enroll", "common", "synapse_home", "synapse_project")

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "Test Author",
    "GIT_AUTHOR_EMAIL": "author@example.com",
    "GIT_COMMITTER_NAME": "Test Author",
    "GIT_COMMITTER_EMAIL": "author@example.com",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": os.devnull,
}


# ---------------------------------------------------------------------------
# Real git layouts
# ---------------------------------------------------------------------------


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-c", "init.defaultBranch=main", *args],
        cwd=cwd,
        env={**os.environ, **_GIT_ENV},
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return proc.stdout.strip()


def _make_repo(parent: Path, name: str, *, marker: str | None = None) -> Path:
    """A committed checkout at parent/name with sub/deeper (holding project config files),
    and a .synapse-project marker at the top when `marker` is given."""
    root = parent / name
    deeper = root / "sub" / "deeper"
    (deeper / ".claude" / "rules").mkdir(parents=True)
    _git(parent, "init", "-q", str(root))
    (root / "README.md").write_text("readme\n", encoding="utf-8")
    (deeper / "CLAUDE.md").write_text("project instructions\n", encoding="utf-8")
    (deeper / ".claude" / "rules" / "style.md").write_text("style rule\n", encoding="utf-8")
    if marker is not None:
        (root / ".synapse-project").write_text(marker, encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "feat: initial commit")
    return root


@dataclass(frozen=True)
class Layout:
    root: Path  # <base>/MyRepo: a real checkout
    sub: Path  # <root>/sub/deeper
    worktree: Path  # <base>/MyRepo-feature: a linked worktree of root
    wt_sub: Path  # <worktree>/sub/deeper
    other_sub: Path  # <base>/acme/sub/deeper: an unrelated repository
    marked_sub: Path  # <base>/checkout-2/sub/deeper: a repo whose marker names "alpha"


@pytest.fixture(scope="module")
def layout(tmp_path_factory) -> Layout:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    base = tmp_path_factory.mktemp("work")
    root = _make_repo(base, "MyRepo")
    worktree = base / "MyRepo-feature"
    _git(root, "worktree", "add", "-q", str(worktree))
    other = _make_repo(base, "acme")
    marked = _make_repo(base, "checkout-2", marker="# this clone's identity\n\nalpha\n")
    # What git itself says, so the layout is the real one and not an assumption.
    assert (worktree / ".git").is_file()
    common = Path(_git(worktree, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    assert common.resolve() == (root / ".git").resolve()
    assert Path(_git(root / "sub" / "deeper", "rev-parse", "--show-toplevel")).resolve() == (
        root.resolve()
    )
    return Layout(
        root=root,
        sub=root / "sub" / "deeper",
        worktree=worktree,
        wt_sub=worktree / "sub" / "deeper",
        other_sub=other / "sub" / "deeper",
        marked_sub=marked / "sub" / "deeper",
    )


def _where(layout: Layout, name: str) -> str | None:
    return {
        "root": str(layout.root),
        "sub": str(layout.sub),
        "wt": str(layout.worktree),
        "wt_sub": str(layout.wt_sub),
        "other": str(layout.other_sub),
        "marked": str(layout.marked_sub),
        "win": _WIN_CWD,
        "none": None,
    }[name]


# ---------------------------------------------------------------------------
# Isolation, loading, the local server
# ---------------------------------------------------------------------------


def _forget() -> None:
    for name in _ENV_MODULES:
        sys.modules.pop(name, None)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path) -> Iterator[None]:
    """HOME, Synapse home, Claude config, data, cursor, log and private dirs all under
    tmp_path; no inherited Synapse settings; a cwd outside any repository; sys.path and
    the import-time-config modules restored around the test."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("SYNAPSE_HOME", str(tmp_path / "synapse-home"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("SYNAPSE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CLAUDE_SKILLS_DIR", str(home / ".claude" / "skills"))
    monkeypatch.setenv("SYNAPSE_CODEX_CURSORS", str(tmp_path / "state" / "codex_cursors.json"))
    monkeypatch.setenv("SYNAPSE_PRIVATE_DIR", str(tmp_path / "private"))
    monkeypatch.setenv("SYNAPSE_SURFACE", "test-project-wiring-host")
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", [*sys.path])
    _forget()
    yield
    _forget()


def _load(path: Path, name: str) -> ModuleType:
    """Load a hook script fresh by path (each prepends its own directory to sys.path)."""
    _forget()
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def server() -> Iterator[SimpleNamespace]:
    """A local stand-in for the Synapse server: records every request (method, path,
    query, lowercased headers, JSON body) and answers like the real routes."""
    seen: list[SimpleNamespace] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def _record(self, body) -> str:
            url = urllib.parse.urlsplit(self.path)
            seen.append(
                SimpleNamespace(
                    method=self.command,
                    path=url.path,
                    query=dict(urllib.parse.parse_qsl(url.query)),
                    headers={k.lower(): v for k, v in self.headers.items()},
                    body=body,
                )
            )
            return url.path

        def _reply(self, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            path = self._record(body)
            if path == "/ingest":
                self._reply({"status": "ok", "ingested": len(body.get("records") or [])})
            elif path == "/timeline/events":
                self._reply({"inserted": len(body.get("events") or []), "skipped": 0})
            else:
                self._reply({"status": "ok"})

        def do_GET(self) -> None:
            path = self._record(None)
            if path == "/context":
                self._reply({"status": "ok", "text": "[board]"})
            else:
                self._reply({"status": "ok", "items": []})

        def log_message(self, *args) -> None:
            pass

    server = LoopbackServer(Handler)
    try:
        yield SimpleNamespace(url=server.url, seen=seen)
    finally:
        server.close()


def _proc_env(url: str, **extra: str) -> dict[str, str]:
    """The isolated test environment for a hook process, pointed at the local server."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k.lower() not in ("http_proxy", "https_proxy", "all_proxy")
    }
    env.update(
        SYNAPSE_URL=url,
        SYNAPSE_INGEST_TOKEN=_TOKEN,
        NO_PROXY="127.0.0.1,localhost",
        no_proxy="127.0.0.1,localhost",
        **extra,
    )
    return env


def _hook_log(name: str) -> str:
    """What a hook process wrote to its log (shown when its assertion fails)."""
    path = Path(os.environ["SYNAPSE_HOME"]) / "logs" / f"{name}.log"
    return path.read_text(errors="replace") if path.exists() else "(no log)"


def _run(args: list[str], url: str, cwd: Path, stdin: str = "", **extra: str):
    proc = subprocess.run(
        [sys.executable, *args],
        input=stdin,
        env=_proc_env(url, **extra),
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc


class _Resp:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *a) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


def _capture_bodies(monkeypatch, mod: ModuleType, reply: bytes) -> list[dict]:
    """Patch urlopen on the loaded module; return the JSON bodies it would have sent."""
    bodies: list[dict] = []

    def fake_urlopen(req, timeout=None):
        bodies.append(json.loads(req.data))
        return _Resp(reply)

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    return bodies


# ---------------------------------------------------------------------------
# Claude Code ingest hook
# ---------------------------------------------------------------------------


def _claude_turn(n: int, cwd: str | None, sid: str = "s1") -> list[dict]:
    extra = {"cwd": cwd} if cwd is not None else {}
    return [
        {
            "type": "user",
            "uuid": f"u{n}",
            "sessionId": sid,
            "message": {"role": "user", "content": f"question {n}"},
            **extra,
        },
        {
            "type": "assistant",
            "uuid": f"a{n}",
            "sessionId": sid,
            "message": {"role": "assistant", "content": [{"type": "text", "text": f"reply {n}"}]},
            **extra,
        },
    ]


def _turns(layout: Layout, names: list[str]) -> list[dict]:
    out: list[dict] = []
    for n, name in enumerate(names, 1):
        out += _claude_turn(n, _where(layout, name))
    return out


@pytest.fixture()
def ingest() -> ModuleType:
    return _load(_INGEST_HOOK, "ingest_hook_project_wiring_test")


def test_post_records_survives_a_malformed_cwd(ingest, layout, monkeypatch):
    bodies = _capture_bodies(monkeypatch, ingest, b'{"status":"ok"}')
    records = [*_turns(layout, ["sub"]), {"type": "user", "uuid": "x", "cwd": ["not", "a path"]}]
    ingest._post_records(records)
    assert len(bodies) == 1
    assert bodies[0]["project"] == "MyRepo"  # the malformed cwd is ignored, not fatal


def test_post_records_skips_falsy_and_unlabelable_cwds(ingest, layout, monkeypatch):
    bodies = _capture_bodies(monkeypatch, ingest, b'{"status":"ok"}')
    records = _turns(layout, ["sub"])
    records += [{"type": "user", "cwd": ""}, {"type": "user", "cwd": "C:\\"}, "not-a-record"]
    ingest._post_records(records)
    ingest._post_records([{"type": "user", "cwd": "C:\\"}])
    assert bodies[0]["project"] == "MyRepo"
    assert "project" not in bodies[1] and "cwd_projects" not in bodies[1]


@pytest.mark.parametrize(
    ("cwds", "expected"),
    [
        pytest.param(["sub"], "MyRepo", id="repo-subfolder"),
        pytest.param(["wt_sub"], "MyRepo", id="linked-worktree"),
        pytest.param(["root", "sub", "wt", "wt_sub"], "MyRepo", id="checkout-and-worktree-one"),
        pytest.param(["none", "sub", "none"], "MyRepo", id="records-without-cwd-ignored"),
        pytest.param(["marked"], "alpha", id="marker-file"),
        pytest.param(["win"], "Acme", id="windows-path"),
        pytest.param(["none"], None, id="no-cwd"),
        pytest.param(["sub", "other"], None, id="two-repositories"),
        pytest.param(["win", "sub"], None, id="windows-and-local-differ"),
    ],
)
def test_post_records_body_carries_the_resolved_project(
    ingest, layout, monkeypatch, cwds, expected
):
    bodies = _capture_bodies(monkeypatch, ingest, b'{"status":"ok"}')
    records = _turns(layout, cwds)
    ingest._post_records(records, source="sweep")
    assert len(bodies) == 1
    body = bodies[0]
    assert body["records"] == records
    assert body["source"] == "sweep"
    if expected is None:
        # No single label: each cwd's label instead (none at all when no record has a cwd).
        assert "project" not in body
        cwds = {r["cwd"] for r in records if r.get("cwd")}
        expect_map = {c: ingest.synapse_project.project_label(c) for c in cwds}
        assert body.get("cwd_projects", {}) == expect_map
        assert len(set(expect_map.values())) != 1
    else:
        assert body["project"] == expected
        assert "cwd_projects" not in body


@pytest.mark.parametrize(
    ("where", "expected"),
    [
        pytest.param("sub", "MyRepo", id="repo-subfolder"),
        pytest.param("wt_sub", "MyRepo", id="linked-worktree"),
        pytest.param("win", "Acme", id="windows-path"),
        pytest.param("none", None, id="no-cwd"),
    ],
)
def test_ingest_hook_process_ships_the_project(layout, server, tmp_path, where, expected):
    """`ingest_hook.py --ship <transcript>` (what the detached Stop child runs), from a
    cwd outside any repository: the label comes from the records, not the process."""
    sid = str(uuid.uuid4())
    transcript = tmp_path / "claude" / "projects" / "-work-MyRepo-sub-deeper" / f"{sid}.jsonl"
    transcript.parent.mkdir(parents=True)
    records = _claude_turn(1, _where(layout, where), sid) + _claude_turn(
        2, _where(layout, where), sid
    )
    transcript.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    _run([str(_INGEST_HOOK), "--ship", str(transcript)], server.url, tmp_path)

    posts = [r for r in server.seen if r.path == "/ingest"]
    assert len(posts) == 1, _hook_log("ingest-hook")
    body = posts[0].body
    assert [r["uuid"] for r in body["records"]] == ["u1", "a1", "u2", "a2"]
    assert body["source"] == "hook"
    assert posts[0].headers.get("authorization") == f"Bearer {_TOKEN}"
    if expected is None:
        assert "project" not in body
    else:
        assert body["project"] == expected


# ---------------------------------------------------------------------------
# Codex Stop hook
# ---------------------------------------------------------------------------


def _cx_meta(cwd: str, sid: str = "sid") -> dict:
    return {"type": "session_meta", "payload": {"id": sid, "cwd": cwd, "originator": "test"}}


def _cx_ctx(cwd: str) -> dict:
    return {"type": "turn_context", "payload": {"cwd": cwd, "model": "test-model"}}


def _cx_user(text: str) -> dict:
    return {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        },
    }


def _cx_assistant(text: str) -> dict:
    return {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        },
    }


def _rollout(home: Path, records: list[dict]) -> tuple[Path, str]:
    """A rollout under ~/.codex/sessions as Codex names it; returns (path, session id)."""
    sid = str(uuid.uuid4())
    day = home / ".codex" / "sessions" / "2026" / "10" / "12"
    day.mkdir(parents=True, exist_ok=True)
    path = day / f"rollout-2026-10-12T09-00-00-{sid}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path, sid


def _park_cursor_after_first_line(cursors: Path, rollout: Path) -> None:
    """A cursor past the session_meta line: the tail shipped later no longer carries it,
    as after an earlier successful POST, or for a catch-up sweep."""
    first = len(rollout.read_bytes().split(b"\n", 1)[0]) + 1
    cursors.parent.mkdir(parents=True, exist_ok=True)
    cursors.write_text(json.dumps({str(rollout): {"offset": first, "ts": time.time()}}))


@pytest.fixture()
def codex_stop(monkeypatch, tmp_path) -> ModuleType:
    mod = _load(_STOP_HOOK, "codex_stop_hook_project_wiring_test")
    monkeypatch.setattr(mod, "SESSIONS_ROOT", tmp_path / "home" / ".codex" / "sessions")
    return mod


@pytest.fixture()
def codex_posts(codex_stop, monkeypatch) -> list[dict]:
    calls: list[dict] = []

    def fake_post(records, session_id, project=None):
        calls.append({"records": records, "session_id": session_id, "project": project})
        return {"ingested": len(records)}

    monkeypatch.setattr(codex_stop, "_post", fake_post)
    return calls


def test_rollout_cwd_prefers_the_stop_payload(codex_stop, tmp_path):
    path, _ = _rollout(tmp_path / "home", [_cx_meta("/w/file"), _cx_user("q")])
    records = [_cx_meta("/w/meta"), _cx_ctx("/w/turn"), _cx_user("q")]
    assert codex_stop._rollout_cwd({"cwd": "/w/payload"}, records, path) == "/w/payload"


def test_rollout_cwd_then_the_newest_context_record(codex_stop, tmp_path):
    path, _ = _rollout(tmp_path / "home", [_cx_meta("/w/file"), _cx_user("q")])
    records = [
        _cx_meta("/w/meta"),
        _cx_user("q1"),
        _cx_ctx("/w/turn1"),
        _cx_assistant("r1"),
        _cx_ctx("/w/turn2"),
        _cx_user("q2"),
    ]
    for payload in ({}, {"cwd": ""}, {"cwd": None}):
        assert codex_stop._rollout_cwd(payload, records, path) == "/w/turn2"
    # Newest by position, whichever of the two types it is.
    assert codex_stop._rollout_cwd({}, [_cx_ctx("/w/turn"), _cx_meta("/w/meta")], path) == (
        "/w/meta"
    )
    # A context record without a cwd is passed over, not taken as the answer.
    blank = {"type": "turn_context", "payload": {"cwd": ""}}
    assert codex_stop._rollout_cwd({}, [_cx_ctx("/w/turn1"), blank], path) == "/w/turn1"


def test_rollout_cwd_then_the_rollouts_first_line(codex_stop, tmp_path):
    path, _ = _rollout(tmp_path / "home", [_cx_meta("/w/file"), _cx_user("q"), _cx_assistant("r")])
    tail = [_cx_user("q"), _cx_assistant("r")]
    assert codex_stop._rollout_cwd({}, tail, path) == "/w/file"
    # Other record types carrying a cwd-shaped field are not consulted.
    event = {"type": "event_msg", "payload": {"type": "note", "cwd": "/w/event"}}
    assert codex_stop._rollout_cwd({}, [event, *tail], path) == "/w/file"


def test_rollout_cwd_is_none_when_nothing_names_one(codex_stop, tmp_path):
    home = tmp_path / "home"
    no_meta, _ = _rollout(home, [_cx_user("q"), _cx_assistant("r")])
    tail = [_cx_user("q")]
    assert codex_stop._rollout_cwd({}, tail, no_meta) is None
    empty, _ = _rollout(home, [])
    assert codex_stop._rollout_cwd({}, tail, empty) is None
    garbled = home / "garbled.jsonl"
    garbled.write_text("{not json\n", encoding="utf-8")
    assert codex_stop._rollout_cwd({}, tail, garbled) is None
    assert codex_stop._rollout_cwd({}, tail, home / "missing.jsonl") is None


@pytest.mark.parametrize(
    ("payload_cwd", "meta_cwd", "expected"),
    [
        pytest.param("sub", "other", "MyRepo", id="payload-cwd-wins"),
        pytest.param(None, "wt_sub", "MyRepo", id="session-meta-in-records"),
        pytest.param("marked", "marked", "alpha", id="marker-file"),
        pytest.param("win", "win", "Acme", id="windows-path"),
    ],
)
def test_codex_ship_passes_the_resolved_project(
    codex_stop, codex_posts, layout, tmp_path, payload_cwd, meta_cwd, expected
):
    path, sid = _rollout(
        tmp_path / "home",
        [_cx_meta(_where(layout, meta_cwd)), _cx_user("q"), _cx_assistant("r")],
    )
    payload = {"session_id": sid, "transcript_path": str(path)}
    if payload_cwd:
        payload["cwd"] = _where(layout, payload_cwd)
    assert codex_stop._ship(payload) == (1, 3)
    assert [c["project"] for c in codex_posts] == [expected]
    assert codex_posts[0]["session_id"] == sid


def test_codex_ship_reads_the_first_line_for_a_catchup_tail(
    codex_stop, codex_posts, layout, tmp_path
):
    """The sweep passes no cwd and the tail past the cursor holds no session_meta: the
    rollout's first line still names the session's folder."""
    path, sid = _rollout(
        tmp_path / "home",
        [_cx_meta(str(layout.wt_sub)), _cx_user("q"), _cx_assistant("r")],
    )
    _park_cursor_after_first_line(codex_stop.CURSORS_PATH, path)
    assert codex_stop._ship({"session_id": sid, "transcript_path": str(path)}) == (1, 2)
    (call,) = codex_posts
    assert [r["type"] for r in call["records"]] == ["response_item", "response_item"]
    assert call["project"] == "MyRepo"


def test_codex_ship_without_any_cwd_sends_no_project(codex_stop, codex_posts, tmp_path):
    path, sid = _rollout(tmp_path / "home", [_cx_user("q"), _cx_assistant("r")])
    assert codex_stop._ship({"session_id": sid, "transcript_path": str(path)}) == (1, 2)
    assert [c["project"] for c in codex_posts] == [None]


def test_codex_post_body_has_project_only_when_set(codex_stop, monkeypatch):
    bodies = _capture_bodies(monkeypatch, codex_stop, b'{"ingested": 1}')
    records = [_cx_user("q")]
    codex_stop._post(records, "sid-1", "MyRepo")
    codex_stop._post(records, "sid-2")
    codex_stop._post(records, "sid-3", None)
    codex_stop._post(records, "sid-4", "")
    assert bodies[0] == {
        "records": records,
        "format": "codex",
        "session_id": "sid-1",
        "source": "codex-hook",
        "project": "MyRepo",
    }
    assert ["project" in b for b in bodies] == [True, False, False, False]


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        pytest.param("stop", "MyRepo", id="stop-payload-cwd"),
        pytest.param("tail", "MyRepo", id="catchup-tail-first-line"),
        pytest.param("windows", "Acme", id="windows-path"),
    ],
)
def test_codex_stop_hook_process_ships_the_project(layout, server, tmp_path, mode, expected):
    """`synapse_stop_hook.py --ship <payload>` (what the detached Stop child runs)."""
    home = tmp_path / "home"
    meta = {"stop": str(layout.sub), "tail": str(layout.wt_sub), "windows": _WIN_CWD}[mode]
    path, sid = _rollout(home, [_cx_meta(meta), _cx_user("q"), _cx_assistant("r")])
    payload = {"session_id": sid, "transcript_path": str(path)}
    if mode == "tail":
        _park_cursor_after_first_line(tmp_path / "state" / "codex_cursors.json", path)
    else:
        payload["cwd"] = meta

    _run([str(_STOP_HOOK), "--ship", json.dumps(payload)], server.url, tmp_path)

    posts = [r for r in server.seen if r.path == "/ingest"]
    assert len(posts) == 1, _hook_log("codex-hook")
    body = posts[0].body
    assert body["project"] == expected
    assert body["format"] == "codex"
    assert body["session_id"] == sid
    assert posts[0].headers.get("authorization") == f"Bearer {_TOKEN}"
    if mode == "tail":
        assert all(r["type"] != "session_meta" for r in body["records"])


# ---------------------------------------------------------------------------
# Board: Claude board_block and Codex session_start
# ---------------------------------------------------------------------------


def _run_board(monkeypatch, mod: ModuleType, stdin: str) -> list[tuple[str, dict]]:
    """main() with the hook payload on stdin and get_json mocked (test_board_block._run)."""
    calls: list[tuple[str, dict]] = []

    def fake_get_json(path, params=None, timeout=30.0):
        calls.append((path, dict(params or {})))
        return {"status": "ok", "text": "[board]"}

    monkeypatch.setattr(mod, "get_json", fake_get_json)
    monkeypatch.setattr(mod.enroll, "is_enrolled", lambda: True)
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    mod.main()
    return calls


@pytest.mark.parametrize(
    ("where", "expected"),
    [
        pytest.param("sub", "MyRepo", id="repo-subfolder"),
        pytest.param("wt_sub", "MyRepo", id="linked-worktree"),
        pytest.param("marked", "alpha", id="marker-file"),
        pytest.param("win", "Acme", id="windows-path"),
    ],
)
def test_board_block_scopes_the_board_to_the_resolved_project(
    monkeypatch, layout, capsys, where, expected
):
    mod = _load(_BOARD_BLOCK, "board_block_project_wiring_test")
    calls = _run_board(monkeypatch, mod, json.dumps({"cwd": _where(layout, where)}))
    assert calls == [("/context", {"project": expected})]
    out = json.loads(capsys.readouterr().out)
    assert out["hookSpecificOutput"]["additionalContext"] == "[board]"


def test_board_block_falls_back_to_the_process_cwd(monkeypatch, layout):
    mod = _load(_BOARD_BLOCK, "board_block_project_wiring_test")
    monkeypatch.chdir(layout.wt_sub)
    assert _run_board(monkeypatch, mod, "{}") == [("/context", {"project": "MyRepo"})]


def test_board_block_process_sends_the_resolved_project(layout, server, tmp_path):
    proc = _run([str(_BOARD_BLOCK)], server.url, tmp_path, json.dumps({"cwd": str(layout.sub)}))
    gets = [r for r in server.seen if r.path == "/context"]
    assert [g.query for g in gets] == [{"project": "MyRepo"}]
    assert json.loads(proc.stdout)["hookSpecificOutput"]["additionalContext"] == "[board]"


@pytest.fixture()
def codex_start(monkeypatch) -> SimpleNamespace:
    """session_start.py loaded fresh with every network or process side effect stubbed;
    `boards` collects the project each board fetch was asked for."""
    mod = _load(_SESSION_START, "codex_session_start_project_wiring_test")
    boards: list[str | None] = []
    monkeypatch.setattr(mod, "_board_text", lambda project: boards.append(project) or "[board]")
    monkeypatch.setattr(mod, "_prefs_text", lambda *a, **k: None)
    monkeypatch.setattr(mod, "_sync_skills", lambda *a, **k: None)
    monkeypatch.setattr(mod, "_spawn_catchup", lambda *a, **k: None)
    monkeypatch.setattr(mod, "_upload_warning", lambda *a, **k: None)
    return SimpleNamespace(mod=mod, boards=boards)


@pytest.mark.parametrize(
    ("where", "expected"),
    [
        pytest.param("sub", "MyRepo", id="repo-subfolder"),
        pytest.param("wt_sub", "MyRepo", id="linked-worktree"),
        pytest.param("marked", "alpha", id="marker-file"),
        pytest.param("win", "Acme", id="windows-path"),
    ],
)
def test_codex_session_start_board_uses_the_resolved_project(
    codex_start, monkeypatch, layout, capsys, where, expected
):
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"cwd": _where(layout, where)})))
    codex_start.mod.main()
    assert codex_start.boards == [expected]
    out = json.loads(capsys.readouterr().out)
    assert out["hookSpecificOutput"]["additionalContext"] == "[board]"


def test_codex_session_start_falls_back_to_the_process_cwd(codex_start, monkeypatch, layout):
    monkeypatch.chdir(layout.sub)
    monkeypatch.setattr(sys, "stdin", io.StringIO("{}"))
    codex_start.mod.main()
    assert codex_start.boards == ["MyRepo"]


def test_codex_session_start_process_sends_the_resolved_project(layout, server, tmp_path):
    proc = _run(
        [str(_SESSION_START)],
        server.url,
        tmp_path,
        json.dumps({"cwd": str(layout.wt_sub)}),
        SYNAPSE_PREFS_BLOCK="0",
        HOME=str(tmp_path),  # the sweep it spawns finds no rollouts
        SYNAPSE_SKILLS_SYNC="0",
    )
    gets = [r for r in server.seen if r.path == "/context"]
    assert [g.query for g in gets] == [{"project": "MyRepo"}]
    assert json.loads(proc.stdout)["hookSpecificOutput"]["additionalContext"] == "[board]"


# ---------------------------------------------------------------------------
# skills_sync, skill_review, config_sync, git_feeder
# ---------------------------------------------------------------------------


def _skills_sync_passes(monkeypatch, project_dir: Path) -> tuple[ModuleType, list]:
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(project_dir))
    monkeypatch.setenv("SYNAPSE_SKILLS_SYNC", "1")
    mod = _load(_SKILLS_SYNC, "skills_sync_project_wiring_test")
    calls: list[tuple[str, Path]] = []

    def fake_sync(scope, target_dir, *a, **k):
        calls.append((scope, target_dir))
        return (0, 0, 0)

    monkeypatch.setattr(mod, "_sync", fake_sync)
    mod.main()
    return mod, calls


@pytest.mark.parametrize(
    ("where", "scope", "root"),
    [
        pytest.param("sub", "project:MyRepo", "root", id="repo-subfolder"),
        pytest.param("wt", "project:MyRepo", "wt", id="linked-worktree"),
        pytest.param("marked", "project:alpha", "marked-root", id="marker-file"),
    ],
)
def test_skills_sync_project_scope_is_the_resolved_label(monkeypatch, layout, where, scope, root):
    proj = Path(_where(layout, where))
    mod, calls = _skills_sync_passes(monkeypatch, proj)
    # The scope names the project, and the folder is the project ROOT's .claude/skills:
    # Claude Code reads it from any subfolder (it walks up to the repository root), and a
    # linked worktree keeps its own (Claude Code stops at the worktree root).
    folder = {
        "root": layout.root,
        "wt": layout.worktree,
        "marked-root": layout.marked_sub.parent.parent,
    }[root]
    assert calls == [("global", mod.config.SKILLS_DIR), (scope, folder / ".claude" / "skills")]


def test_skills_sync_home_project_still_skips_the_project_pass(monkeypatch, tmp_path):
    mod, calls = _skills_sync_passes(monkeypatch, tmp_path / "home")
    assert calls == [("global", mod.config.SKILLS_DIR)]


def test_skill_review_local_dir_matches_the_label_in_any_case(monkeypatch, layout):
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(layout.sub))
    mod = _load(_SKILL_REVIEW, "skill_review_project_wiring_test")
    target = layout.root / ".claude" / "skills"  # the project root, as skills_sync.main
    for scope in ("project:MYREPO", "project:myrepo", "project:MyRepo"):
        assert mod._local_dir(scope) == target, scope
    # The folder's own name is no longer the label, and another project is not this one.
    for scope in ("project:deeper", "project:sub", "project:acme", "project:"):
        assert mod._local_dir(scope) is None, scope
    assert mod._local_dir("global") == mod.config.SKILLS_DIR

    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(layout.worktree))
    assert mod._local_dir("project:myrepo") == layout.worktree / ".claude" / "skills"
    assert mod._local_dir("project:MyRepo-feature") is None


def test_skill_review_local_dir_never_the_global_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path / "home"))
    mod = _load(_SKILL_REVIEW, "skill_review_project_wiring_test")
    assert mod._local_dir("project:HOME") is None


@pytest.mark.parametrize(("where", "root"), [("sub", "root"), ("wt_sub", "wt")])
def test_config_sync_targets_label_project_files_with_the_resolved_name(
    monkeypatch, layout, where, root
):
    """The project ROOT's files, under the project's label, whichever subfolder the session
    started in: a subfolder's own CLAUDE.md would otherwise share the root's (surface,
    scope, file_key) row and the two would overwrite each other."""
    proj = Path(_where(layout, where))
    top = Path(_where(layout, root))
    for base, text in ((top, "root"), (proj, "subfolder")):
        (base / ".claude" / "rules").mkdir(parents=True, exist_ok=True)
        (base / "CLAUDE.md").write_text(text)
        (base / ".claude" / "rules" / "style.md").write_text(text)
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(proj))
    mod = _load(_CONFIG_SYNC, "config_sync_project_wiring_test")
    project = sorted((scope, key, p) for scope, key, p in mod._targets() if scope != "global")
    assert project == [
        ("project:MyRepo", ".claude/rules/style.md", top / ".claude" / "rules" / "style.md"),
        ("project:MyRepo", "CLAUDE.md", top / "CLAUDE.md"),
    ]


def _feeder(monkeypatch) -> tuple[ModuleType, list[dict]]:
    mod = _load(_GIT_FEEDER, "git_feeder_project_wiring_test")
    events: list[dict] = []

    def post_json(path, payload, timeout=30.0):
        assert path == "/timeline/events"
        events.extend(payload["events"])
        return {"inserted": len(payload["events"]), "skipped": 0}

    monkeypatch.setattr(mod, "post_json", post_json)
    return mod, events


@pytest.mark.parametrize(
    ("where", "expected"),
    [
        pytest.param("root", "MyRepo", id="checkout"),
        pytest.param("wt", "MyRepo", id="linked-worktree"),
        pytest.param("marked", "alpha", id="marker-file"),
    ],
)
def test_git_feeder_files_commits_under_the_resolved_label(monkeypatch, layout, where, expected):
    mod, events = _feeder(monkeypatch)
    commit = {
        "sha": "0123abc",
        "t_valid": "2026-10-01T10:00:00+00:00",
        "is_merge": False,
        "subject": "fix: tidy",
        "churn": 3,
    }
    monkeypatch.setattr(mod, "_read_commits", lambda repo, since_days: [commit])
    assert mod._push_repo(Path(_where(layout, where)), 30) == (1, 0)
    (ev,) = events
    assert ev["project"] == expected
    # the dedup key stays the folder's name, so a new label never re-inserts old commits
    assert ev["source"] == f"git:{Path(_where(layout, where)).name}"
    assert ev["fact"] == f"committed to {expected}: fix: tidy"


def test_git_feeder_reads_a_worktree_with_real_git(monkeypatch, layout):
    """No stubbed git: the worktree's real log, filed under its main checkout's name."""
    mod, events = _feeder(monkeypatch)
    inserted, _ = mod._push_repo(layout.worktree, 30)
    assert inserted == len(events) >= 1
    assert {e["project"] for e in events} == {"MyRepo"}
    assert events[0]["source_ref"] == _git(layout.worktree, "rev-parse", "HEAD")


def test_git_feeder_process_files_worktree_commits_under_the_checkout(layout, server, tmp_path):
    _run(
        [str(_GIT_FEEDER)],
        server.url,
        tmp_path,
        SYNAPSE_TIMELINE_REPOS=str(layout.worktree),
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
    )
    posts = [r for r in server.seen if r.path == "/timeline/events"]
    assert len(posts) == 1
    assert {e["project"] for e in posts[0].body["events"]} == {"MyRepo"}
    assert {e["source"] for e in posts[0].body["events"]} == {f"git:{layout.worktree.name}"}
