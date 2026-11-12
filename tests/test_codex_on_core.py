"""The Codex hooks on the client core (plugin-codex over plugin/synapse_client).

Every hook is loaded by path the way Codex runs it, with HOME, SYNAPSE_HOME and every
other file a hook reads at import under a temp dir (scratch_user), and talks to a local
stdlib server on 127.0.0.1. No test reaches a configured Synapse server or reads a real
credential.
"""

from __future__ import annotations

import ast
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.helpers.client_core import REPO, scratch_user, use_core

_CODEX = REPO / "plugin-codex"
_TOKEN = "tkCODEXcore7Rq2Wm9"
_SURFACE_ID = "dev-codex-test"


# ---------------------------------------------------------------------------
# A local Synapse stand-in
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def log_message(self, format: str, *args: Any) -> None:  # silence
        pass

    def _handle(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        path = self.path.split("?", 1)[0]
        self.server.seen.append(
            {
                "method": self.command,
                "path": self.path,
                "headers": {k.lower(): v for k, v in self.headers.items()},
                "body": json.loads(body) if body else None,
            }
        )
        status, reply, delay = self.server.routes.get(path, (200, {}, 0.0))
        if delay:
            self.server.release.wait(delay)
        raw = json.dumps(reply).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        except OSError:  # the client gave up waiting
            pass

    do_GET = do_POST = do_PUT = do_DELETE = _handle


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.seen: list[dict[str, Any]] = []
        #: path -> (status, JSON reply, seconds to stall before answering)
        self.routes: dict[str, tuple[int, Any, float]] = {}
        self.release = threading.Event()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def answer(self, path: str, reply: Any, status: int = 200, delay: float = 0.0) -> None:
        self.routes[path] = (status, reply, delay)


@pytest.fixture()
def server() -> Iterator[_Server]:
    srv = _Server()
    threading.Thread(target=srv.serve_forever, args=(0.02,), daemon=True).start()
    yield srv
    srv.release.set()
    srv.shutdown()
    srv.server_close()


@pytest.fixture()
def scratch(monkeypatch, tmp_path) -> Iterator[Path]:
    """A scratch user, and sys.path / sys.modules put back after: the hooks extend the
    path and leave ``common`` imported."""
    home = scratch_user(monkeypatch, tmp_path)
    monkeypatch.setenv("SYNAPSE_CODEX_HOOK_LOG", str(tmp_path / "codex-hook.log"))
    monkeypatch.setenv("SYNAPSE_CODEX_CURSORS", str(tmp_path / "codex_cursors.json"))
    monkeypatch.setenv("SYNAPSE_CODEX_CATCHUP", "0")  # no detached sweep from a test
    monkeypatch.setattr(sys, "path", list(sys.path))
    saved = sys.modules.get("common")
    yield home
    sys.modules.pop("common", None)
    if saved is not None:
        sys.modules["common"] = saved


def _load(name: str, path: Path) -> ModuleType:
    """``path`` freshly imported as a hook process would: ``common`` re-read too."""
    sys.modules.pop("common", None)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _common() -> ModuleType:
    common = _load("common", _CODEX / "scripts" / "common.py")
    sys.modules["common"] = common
    return common


def _write_config(home: Path, data: dict[str, Any]) -> None:
    path = home / ".synapse" / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _user_agent() -> str:
    use_core()
    from synapse_client import __version__

    return f"synapse-client/{__version__} codex"


# ---------------------------------------------------------------------------
# common.py: a thin layer over Settings and Transport
# ---------------------------------------------------------------------------


def test_common_has_no_http_or_token_resolution_of_its_own(scratch, monkeypatch):
    tree = ast.parse((_CODEX / "scripts" / "common.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    words: set[str] = set()  # identifiers and string literals, docstrings aside
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.FunctionDef)
        and node.body
        and isinstance(node.body[0], ast.Expr)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Name):
            words.add(node.id)
        elif isinstance(node, ast.Attribute):
            words.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                words.add(node.value)
    assert not imported & {"urllib.request", "urllib.parse", "http.client", "hashlib"}
    assert not words & {"Authorization", "sha256", "hexdigest", "token_sha256", "urlopen"}
    assert not any(word.startswith("Bearer") for word in words)

    monkeypatch.setenv("SYNAPSE_URL", "https://syn.example.net/mcp")
    monkeypatch.setenv("SYNAPSE_INGEST_TOKEN", _TOKEN)
    common = _common()
    use_core()
    from synapse_client.settings import Settings

    assert isinstance(common.SETTINGS, Settings)
    assert common.BASE_URL == common.SETTINGS.base_url == "https://syn.example.net"
    assert common.INGEST_URL == common.SETTINGS.ingest_url == "https://syn.example.net/ingest"
    assert common.TOKEN == common.SETTINGS.token == _TOKEN


def test_common_defaults_to_a_local_server_when_no_url_is_configured(scratch):
    common = _common()
    assert common.BASE_URL == "http://localhost:8765"
    assert common.INGEST_URL == "http://localhost:8765/ingest"


def test_the_device_token_is_not_sent_to_the_default_url(scratch):
    """An enrolled machine with no URL configured: requests would go to the localhost
    default, which did not mint the device token, so none is sent (as before the move)."""
    use_core()
    from synapse_client.credentials import token_sha256

    record = {
        "surface_id": _SURFACE_ID,
        "token_sha256": token_sha256(_TOKEN),
        "server": "https://syn.example.net",
    }
    _write_config(scratch, {"token": _TOKEN, "device": record})
    common = _common()
    assert common.BASE_URL == "http://localhost:8765"
    assert common.TOKEN == ""
    notice = common.token_override_notice()
    assert notice.startswith("[Synapse] http://localhost:8765 is not the server this machine")
    assert _TOKEN not in notice


def test_common_requests_carry_the_client_user_agent(scratch, monkeypatch, server):
    monkeypatch.setenv("SYNAPSE_URL", server.url)
    monkeypatch.setenv("SYNAPSE_INGEST_TOKEN", _TOKEN)
    common = _common()
    server.answer("/context", {"status": "ok"})
    assert common.get_json("/context", {"project": "p"}) == {"status": "ok"}
    common.request_json("PUT", "/private-sessions/s-1", timeout=5)
    for req in server.seen:
        assert req["headers"]["user-agent"] == _user_agent()
        assert req["headers"]["authorization"] == f"Bearer {_TOKEN}"
    assert [r["path"] for r in server.seen] == ["/context?project=p", "/private-sessions/s-1"]


def test_common_keeps_the_bearer_off_another_origin(scratch, monkeypatch, server):
    """Intended with the move to the core's Transport: an absolute URL at an origin that
    is neither the server's nor the ingest URL's is sent without the bearer."""
    monkeypatch.setenv("SYNAPSE_URL", "http://localhost:9")
    monkeypatch.setenv("SYNAPSE_INGEST_TOKEN", _TOKEN)
    common = _common()
    common.post_json(server.url + "/elsewhere", {})
    assert server.seen and "authorization" not in server.seen[0]["headers"]


def test_the_stop_hook_upload_carries_the_client_user_agent(scratch, monkeypatch, server):
    monkeypatch.setenv("SYNAPSE_URL", server.url)
    monkeypatch.setenv("SYNAPSE_INGEST_TOKEN", _TOKEN)
    hook = _load("codex_stop_hook_on_core", _CODEX / "hooks" / "synapse_stop_hook.py")
    server.answer("/ingest", {"status": "ok", "inserted": 1})
    assert hook._post([{"type": "x"}], "s-1", project="proj") == {"status": "ok", "inserted": 1}
    (req,) = server.seen
    assert (req["method"], req["path"]) == ("POST", "/ingest")
    assert req["headers"]["user-agent"] == _user_agent()
    assert req["headers"]["authorization"] == f"Bearer {_TOKEN}"
    assert req["body"] == {
        "records": [{"type": "x"}],
        "format": "codex",
        "session_id": "s-1",
        "source": "codex-hook",
        "project": "proj",
    }


# ---------------------------------------------------------------------------
# install.py: the hook and MCP blocks Codex trusts by hash, unchanged
# ---------------------------------------------------------------------------

# What install.py wrote into an empty config.toml at the branch point of the move onto
# the core (40fdc3f). Codex trusts a hook by a hash of its command text: a byte that
# moves here makes every user re-trust every hook.
_INSTALLED_CHECKOUT = r"""
# Synapse hooks (installed by synapse plugin-codex/install.py)
[[hooks.Stop]]
[[hooks.Stop.hooks]]
type = "command"
command = "python3 \"/home/me/src/synapse/plugin-codex/hooks/synapse_stop_hook.py\""
timeout = 30

[[hooks.SessionStart]]
[[hooks.SessionStart.hooks]]
type = "command"
command = "python3 \"/home/me/src/synapse/plugin-codex/hooks/session_start.py\""
timeout = 20

[[hooks.UserPromptSubmit]]
[[hooks.UserPromptSubmit.hooks]]
type = "command"
command = "python3 \"/home/me/src/synapse/plugin-codex/hooks/user_prompt_submit.py\""
timeout = 5

[[hooks.PreToolUse]]
matcher = "mcp__.*__(recall|recall_full_turns|recall_feedback|fetch_session|remember)$"
[[hooks.PreToolUse.hooks]]
type = "command"
command = "python3 \"/home/me/src/synapse/plugin-codex/hooks/pre_tool_use.py\""
timeout = 5

[[hooks.PostToolUse]]
matcher = "mcp__.*__(recall|recall_full_turns)$"
[[hooks.PostToolUse.hooks]]
type = "command"
command = "python3 \"/home/me/src/synapse/plugin-codex/hooks/post_tool_use.py\""
timeout = 5

[[hooks.SessionEnd]]
[[hooks.SessionEnd.hooks]]
type = "command"
command = "python3 \"/home/me/src/synapse/plugin-codex/scripts/private_mode.py\" --session-end"
timeout = 3

[mcp_servers.synapse]
url = "https://synapse.example.net/mcp"
http_headers_helper = "python3 \"/home/me/src/synapse/plugin/scripts/mcp_headers.py\" --url \"https://synapse.example.net/mcp\""
"""

_INSTALLED_MANAGED = r"""
# Synapse hooks (installed by synapse plugin-codex/install.py)
[[hooks.Stop]]
[[hooks.Stop.hooks]]
type = "command"
command = "python3 \"/home/me/.synapse/bin/synapse_run.py\" plugin-codex/hooks/synapse_stop_hook.py"
timeout = 30

[[hooks.SessionStart]]
[[hooks.SessionStart.hooks]]
type = "command"
command = "python3 \"/home/me/.synapse/bin/synapse_run.py\" plugin-codex/hooks/session_start.py"
timeout = 20

[[hooks.UserPromptSubmit]]
[[hooks.UserPromptSubmit.hooks]]
type = "command"
command = "python3 \"/home/me/.synapse/bin/synapse_run.py\" plugin-codex/hooks/user_prompt_submit.py"
timeout = 5

[[hooks.PreToolUse]]
matcher = "mcp__.*__(recall|recall_full_turns|recall_feedback|fetch_session|remember)$"
[[hooks.PreToolUse.hooks]]
type = "command"
command = "python3 \"/home/me/.synapse/bin/synapse_run.py\" plugin-codex/hooks/pre_tool_use.py"
timeout = 5

[[hooks.PostToolUse]]
matcher = "mcp__.*__(recall|recall_full_turns)$"
[[hooks.PostToolUse.hooks]]
type = "command"
command = "python3 \"/home/me/.synapse/bin/synapse_run.py\" plugin-codex/hooks/post_tool_use.py"
timeout = 5

[[hooks.SessionEnd]]
[[hooks.SessionEnd.hooks]]
type = "command"
command = "python3 \"/home/me/.synapse/bin/synapse_run.py\" plugin-codex/scripts/private_mode.py --session-end"
timeout = 3

[mcp_servers.synapse]
url = "https://synapse.example.net/mcp"
http_headers_helper = "python3 \"/home/me/.synapse/bin/synapse_run.py\" plugin/scripts/mcp_headers.py --url \"https://synapse.example.net/mcp\""
"""


@pytest.mark.parametrize("layout", ["checkout", "managed"])
def test_install_writes_the_branch_point_blocks_byte_for_byte(layout, tmp_path, capsys):
    installer = _load("synapse_installer_on_core", _CODEX / "install.py")
    installer.CONFIG_PATH = tmp_path / "config.toml"
    launcher = {
        "checkout": installer.Launcher("python3", root=Path("/home/me/src/synapse")),
        "managed": installer.Launcher(
            "python3", runner=Path("/home/me/.synapse/bin/synapse_run.py")
        ),
    }[layout]
    installer.install("https://synapse.example.net", launcher=launcher)
    expected = {"checkout": _INSTALLED_CHECKOUT, "managed": _INSTALLED_MANAGED}[layout]
    assert installer.CONFIG_PATH.read_bytes() == expected.encode("utf-8")
    capsys.readouterr()


# ---------------------------------------------------------------------------
# session_start.py: the core's session block, fetched side by side under a deadline
# ---------------------------------------------------------------------------

_SESSION_START = _CODEX / "hooks" / "session_start.py"
_STALE = "tkSTALEenvXq81Lm"
_WARNING = "[Synapse] uploads have been failing since yesterday (connection refused)."
_BOARD = {"status": "ok", "text": "[Synapse board]\n- a note"}
_PREFS = {"items": [{"pref": "short answers", "polarity": "like"}]}
_PREFS_TEXT = "[Synapse preferences]\n  - (likes) short answers"


def _enroll(home: Path, server_url: str) -> None:
    """config.json as `synapse login` leaves it: URL, device token, enrollment record."""
    use_core()
    from synapse_client.credentials import token_sha256

    record = {
        "surface_id": _SURFACE_ID,
        "token_sha256": token_sha256(_TOKEN),
        "server": server_url,
    }
    _write_config(home, {"url": server_url, "token": _TOKEN, "device": record})


def _start(monkeypatch, cwd: Path, *, warning: str | None = None) -> ModuleType:
    hook = _load("codex_session_start_on_core", _SESSION_START)
    monkeypatch.setattr(hook, "_upload_warning", lambda: warning)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"cwd": str(cwd)})))
    return hook


def _printed(capsys) -> dict[str, Any] | None:
    out = capsys.readouterr().out
    return json.loads(out) if out.strip() else None


def test_session_start_order_is_notice_warning_board_preferences(
    scratch, monkeypatch, tmp_path, server, capsys
):
    _enroll(scratch, server.url)
    monkeypatch.setenv("SYNAPSE_INGEST_TOKEN", _STALE)  # overridden: the notice leads
    server.answer("/context", _BOARD)
    server.answer("/preferences/top", _PREFS)
    hook = _start(monkeypatch, tmp_path, warning=_WARNING)
    hook.main()
    out = _printed(capsys)
    notice = (
        f"[Synapse] SYNAPSE_INGEST_TOKEN from your environment is ignored: this machine is "
        f"enrolled as {_SURFACE_ID}. Remove it from your shell profile or wherever Codex "
        "inherits it."
    )
    assert out == {
        "systemMessage": _WARNING,
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": "\n\n".join((notice, _WARNING, _BOARD["text"], _PREFS_TEXT)),
        },
    }
    assert sorted(r["path"].split("?")[0] for r in server.seen) == ["/context", "/preferences/top"]
    for req in server.seen:
        assert req["headers"]["authorization"] == f"Bearer {_TOKEN}"
        assert req["headers"]["user-agent"] == _user_agent()


def test_session_start_explains_a_machine_that_is_not_enrolled(
    scratch, monkeypatch, tmp_path, server, capsys
):
    monkeypatch.setenv("SYNAPSE_URL", server.url)
    server.answer("/context", {"detail": "unauthorized"}, status=401)
    server.answer("/preferences/top", {"detail": "unauthorized"}, status=401)
    hook = _start(monkeypatch, tmp_path)
    hook.main()
    text = _printed(capsys)["hookSpecificOutput"]["additionalContext"]
    assert text == hook.context.not_enrolled_block(hook.context.DEFAULT_LOGIN_HINT)
    assert text.startswith("[Synapse — this machine is not enrolled]")
    assert "Run `synapse login` in a terminal and approve the sign-in" in text


def test_session_start_explains_a_restricted_caller(scratch, monkeypatch, tmp_path, server, capsys):
    monkeypatch.setenv("SYNAPSE_URL", server.url)
    server.answer("/context", {"status": "ok", "trust": "restricted", "text": "[empty board]"})
    server.answer("/preferences/top", {"items": []})
    hook = _start(monkeypatch, tmp_path)
    hook.main()
    text = _printed(capsys)["hookSpecificOutput"]["additionalContext"]
    restricted = hook.context.restricted_block(hook.context.DEFAULT_LOGIN_HINT)
    assert text == f"{restricted}\n\n[empty board]"
    assert "Server with a sign-in configured: run `synapse login` in a terminal." in text


@pytest.mark.parametrize("switch", ["SYNAPSE_BOARD", "SYNAPSE_PREFS_BLOCK"])
def test_session_start_kill_switches_skip_the_request(
    scratch, monkeypatch, tmp_path, server, capsys, switch
):
    monkeypatch.setenv("SYNAPSE_URL", server.url)
    server.answer("/context", _BOARD)
    server.answer("/preferences/top", _PREFS)
    hook = _start(monkeypatch, tmp_path)
    monkeypatch.setenv(switch, "0")  # read when the hook runs, as before
    hook.main()
    text = _printed(capsys)["hookSpecificOutput"]["additionalContext"]
    paths = [r["path"].split("?")[0] for r in server.seen]
    if switch == "SYNAPSE_BOARD":
        assert (text, paths) == (_PREFS_TEXT, ["/preferences/top"])
    else:
        assert (text, paths) == (_BOARD["text"], ["/context"])


@pytest.mark.parametrize("value", ["0", "1"])
def test_session_start_catchup_kill_switch(scratch, monkeypatch, tmp_path, capsys, value):
    monkeypatch.setenv("SYNAPSE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("SYNAPSE_CODEX_CATCHUP", value)
    hook = _start(monkeypatch, tmp_path)
    spawned: list[list[str]] = []
    monkeypatch.setattr(hook.subprocess, "Popen", lambda cmd, **kw: spawned.append(cmd))
    hook.main()
    assert [cmd[-2] for cmd in spawned] == ([] if value == "0" else ["--catchup"])


def test_session_start_prints_nothing_with_nothing_to_say(scratch, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SYNAPSE_URL", "http://127.0.0.1:9")  # nothing listens
    hook = _start(monkeypatch, tmp_path)
    hook.main()
    assert capsys.readouterr().out == ""


def test_session_start_envelope_without_a_warning_has_no_system_message(
    scratch, monkeypatch, tmp_path, server, capsys
):
    monkeypatch.setenv("SYNAPSE_URL", server.url)
    server.answer("/context", _BOARD)
    server.answer("/preferences/top", {"items": []})
    hook = _start(monkeypatch, tmp_path)
    hook.main()
    assert _printed(capsys) == {
        "hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": _BOARD["text"]}
    }


def test_session_start_prints_what_arrived_when_one_request_fails(
    scratch, monkeypatch, tmp_path, server, capsys
):
    monkeypatch.setenv("SYNAPSE_URL", server.url)
    server.answer("/context", _BOARD)
    server.answer("/preferences/top", {"detail": "boom"}, status=500)
    hook = _start(monkeypatch, tmp_path, warning=_WARNING)
    hook.main()
    out = _printed(capsys)
    assert out["hookSpecificOutput"]["additionalContext"] == f"{_WARNING}\n\n{_BOARD['text']}"


def test_session_start_stops_waiting_at_its_deadline(
    scratch, monkeypatch, tmp_path, server, capsys
):
    """The preferences request hangs past the deadline; the board answered in time and
    is printed, and the hook does not wait out the hung request."""
    monkeypatch.setenv("SYNAPSE_URL", server.url)
    server.answer("/context", _BOARD)
    server.answer("/preferences/top", _PREFS, delay=30)
    hook = _start(monkeypatch, tmp_path, warning=_WARNING)
    monkeypatch.setattr(hook, "_FETCH_TIMEOUT", 30.0)  # only the deadline can stop it
    monkeypatch.setattr(hook, "_DEADLINE_S", 1.0)
    monkeypatch.setattr(hook, "_STARTED", time.monotonic())
    began = time.monotonic()
    hook.main()
    assert time.monotonic() - began < 3
    out = _printed(capsys)
    assert out["hookSpecificOutput"]["additionalContext"] == f"{_WARNING}\n\n{_BOARD['text']}"


def test_session_start_skills_sync_shares_the_deadline(scratch, monkeypatch, tmp_path):
    monkeypatch.setenv("SYNAPSE_SKILLS_SYNC", "1")
    hook = _start(monkeypatch, tmp_path)
    seen: list[float] = []
    monkeypatch.setattr(hook.subprocess, "run", lambda cmd, **kw: seen.append(kw["timeout"]))
    monkeypatch.setattr(hook, "_STARTED", time.monotonic() - 10)  # 5 s left of 15
    hook._sync_skills(str(tmp_path))
    assert 4 < seen[0] <= 5
    monkeypatch.setattr(hook, "_STARTED", time.monotonic() - 20)  # past the deadline
    hook._sync_skills(str(tmp_path))
    assert len(seen) == 1


_HUNG_RUN = """
import importlib.util, io, sys, time
spec = importlib.util.spec_from_file_location("codex_session_start", sys.argv[1])
hook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hook)
hook._FETCH_TIMEOUT = 60.0
hook._DEADLINE_S = 1.0
hook._STARTED = time.monotonic()
sys.stdin = io.StringIO("{}")
hook.main()
"""


def test_a_hung_server_does_not_hold_the_hook_process(scratch, tmp_path, server):
    """Both requests hang: the process prints and exits at the deadline, without waiting
    for the request threads (they are daemons)."""
    server.answer("/context", _BOARD, delay=60)
    server.answer("/preferences/top", _PREFS, delay=60)
    env = {**os.environ, "SYNAPSE_URL": server.url}
    began = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-c", _HUNG_RUN, str(_SESSION_START)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert time.monotonic() - began < 10
    assert proc.stdout == ""


# ---------------------------------------------------------------------------
# user_prompt_submit.py: the core's prompt nudge, with Codex's tool names
# ---------------------------------------------------------------------------

_PROMPT_HOOK = _CODEX / "hooks" / "user_prompt_submit.py"
# The directive as Codex printed it before the move: the wording is unchanged.
_CODEX_NUDGE = (
    "[Synapse] IMPORTANT: the user's cross-session history is searchable — assume "
    "it has context on any past work, project, device, or person they name. BEFORE "
    "answering anything that references the past: synapse recall (skip only if the "
    "answer is already in this conversation). About to say 'noted' or 'I'll "
    "remember'? Call synapse remember FIRST, then reply."
)
_STAMP = re.compile(r"^\[\w{3} \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} .*\]$")


def _prompt_context(capsys) -> str | None:
    hook = _load("codex_prompt_hook_on_core", _PROMPT_HOOK)
    hook.main()
    out = _printed(capsys)
    if out is None:
        return None
    assert out["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    return out["hookSpecificOutput"]["additionalContext"]


def test_prompt_nudge_leads_with_the_time(scratch, capsys):
    stamp, nudge = _prompt_context(capsys).split("\n")
    assert _STAMP.match(stamp), stamp
    assert nudge == _CODEX_NUDGE


def test_prompt_nudge_without_the_time_is_the_old_output(scratch, monkeypatch, capsys):
    monkeypatch.setenv("SYNAPSE_PROMPT_TIMESTAMP", "0")
    assert _prompt_context(capsys) == _CODEX_NUDGE


def test_prompt_nudge_off_leaves_the_time(scratch, monkeypatch, capsys):
    monkeypatch.setenv("SYNAPSE_RECALL_NUDGE", "0")
    assert _STAMP.match(_prompt_context(capsys))
    monkeypatch.setenv("SYNAPSE_PROMPT_TIMESTAMP", "0")
    assert _prompt_context(capsys) is None


def test_prompt_hook_as_codex_runs_it(scratch):
    proc = subprocess.run(
        [sys.executable, str(_PROMPT_HOOK)],
        input="{}",
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    text = json.loads(proc.stdout)["hookSpecificOutput"]["additionalContext"]
    assert _STAMP.match(text.split("\n")[0]) and text.endswith(_CODEX_NUDGE)


# ---------------------------------------------------------------------------
# The recall-feedback nudge: one text (the core's), OFF unless the setting is "1"
# ---------------------------------------------------------------------------

_CLAUDE_FEEDBACK_HOOK = REPO / "plugin" / "scripts" / "recall_feedback_nudge.py"
_CODEX_FEEDBACK_HOOK = _CODEX / "hooks" / "post_tool_use.py"
# What both hooks printed with the nudge on, before the text moved into the core: the
# Claude hook's output must not change by a byte.
_FEEDBACK_OUT = (
    b'{"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": '
    b'"[Synapse] recall() returned. If you used any of the served ids (e:/n:/f:/t:/w:) '
    b"in your answer, call recall_feedback(query=<verbatim query>, helpful=[...], "
    b'noise=[...], comment=\\"...\\") ONCE to close the retrieval-quality loop: '
    b"helpful=load-bearing ids, noise=irrelevant ids, comment=free text for anything "
    b"else (too much served, buried hit, bad ordering). ONLY IF you needed specific "
    b"content memory likely holds and it was not served, add missing=what it was + "
    b"found_via=where you got it instead (full_turns/filesystem/user/nowhere...) "
    b"\\u2014 omit both otherwise. Skip the call only if you used none of the "
    b'results."}}\n'
)


def _feedback_hook(path: Path, setting: str | None, tool: str = "mcp__synapse__recall") -> bytes:
    env = {k: v for k, v in os.environ.items() if k != "SYNAPSE_RECALL_FEEDBACK_NUDGE"}
    if setting is not None:
        env["SYNAPSE_RECALL_FEEDBACK_NUDGE"] = setting
    proc = subprocess.run(
        [sys.executable, str(path)],
        input=json.dumps({"tool_name": tool}).encode(),
        capture_output=True,
        env=env,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


@pytest.mark.parametrize(
    "hook", [_CLAUDE_FEEDBACK_HOOK, _CODEX_FEEDBACK_HOOK], ids=["claude", "codex"]
)
@pytest.mark.parametrize("setting", [None, "0", "", "true", "yes", "2"])
def test_feedback_nudge_is_off_unless_the_setting_is_1(scratch, hook, setting):
    assert _feedback_hook(hook, setting) == b""


@pytest.mark.parametrize(
    "hook", [_CLAUDE_FEEDBACK_HOOK, _CODEX_FEEDBACK_HOOK], ids=["claude", "codex"]
)
def test_feedback_nudge_on_with_1_prints_the_pinned_bytes(scratch, hook):
    assert _feedback_hook(hook, "1") == _FEEDBACK_OUT


@pytest.mark.parametrize("tool", ["mcp__synapse__recall_full_turns", "mcp__synapse__recall"])
def test_codex_feedback_nudge_follows_rateable_tools(scratch, tool):
    assert _feedback_hook(_CODEX_FEEDBACK_HOOK, "1", tool) == _FEEDBACK_OUT


@pytest.mark.parametrize("tool", ["mcp__synapse__recall_feedback", "mcp__synapse__fetch", ""])
def test_codex_feedback_nudge_keeps_its_tool_name_guard(scratch, tool):
    assert _feedback_hook(_CODEX_FEEDBACK_HOOK, "1", tool) == b""


def test_feedback_nudge_text_takes_the_tool_name():
    use_core()
    from synapse_client.context import recall_feedback_nudge

    text = recall_feedback_nudge()
    assert json.dumps(text).encode() in _FEEDBACK_OUT
    named = recall_feedback_nudge(recall_feedback="synapse:recall_feedback")
    assert named == text.replace("call recall_feedback(", "call synapse:recall_feedback(")


# ---------------------------------------------------------------------------
# synapse_stop_hook.py: private mode is the core's rule
# ---------------------------------------------------------------------------

_SESSION = "0b6c2f1e-4d3a-4c55-9a7e-2f1d8b9c0a11"


def _rollout(path: Path) -> Path:
    def item(role: str, kind: str, text: str) -> dict[str, Any]:
        content = [{"type": kind, "text": text}]
        payload = {"type": "message", "role": role, "content": content}
        return {"type": "response_item", "payload": payload}

    recs = [item("user", "input_text", "q"), item("assistant", "output_text", "r")]
    path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    return path


def _ship(monkeypatch, tmp_path: Path, session_id: str) -> tuple[ModuleType, list[str]]:
    """The Stop hook shipping one rollout for ``session_id``; returns the hook and the
    session ids it posted."""
    hook = _load("codex_stop_hook_private_on_core", _CODEX / "hooks" / "synapse_stop_hook.py")
    posted: list[str] = []
    monkeypatch.setattr(
        hook, "_post", lambda recs, sid, project=None: posted.append(sid) or {"ingested": 1}
    )
    rollout = _rollout(tmp_path / f"rollout-2026-11-12T09-00-00-{_SESSION}.jsonl")
    hook._ship({"session_id": session_id, "transcript_path": str(rollout)})
    return hook, posted


def _hook_log(tmp_path: Path) -> str:
    log = tmp_path / "codex-hook.log"
    return log.read_text(encoding="utf-8") if log.exists() else ""


def test_stop_hook_honours_a_fresh_private_marker(scratch, monkeypatch, tmp_path):
    marker = Path(os.environ["SYNAPSE_PRIVATE_DIR"]) / _SESSION
    marker.parent.mkdir(parents=True)
    marker.write_text("", encoding="utf-8")
    _hook, posted = _ship(monkeypatch, tmp_path, _SESSION)
    assert posted == [] and marker.exists()
    assert f"skip: private session {_SESSION}" in _hook_log(tmp_path)


@pytest.mark.parametrize("ttl", [None, "1"])
def test_stop_hook_removes_a_stale_private_marker_and_ships(scratch, monkeypatch, tmp_path, ttl):
    if ttl:
        monkeypatch.setenv("SYNAPSE_PRIVATE_TTL_HOURS", ttl)
    marker = Path(os.environ["SYNAPSE_PRIVATE_DIR"]) / _SESSION
    marker.parent.mkdir(parents=True)
    marker.write_text("", encoding="utf-8")
    age = (2 if ttl else 13) * 3600
    os.utime(marker, (time.time() - age, time.time() - age))
    _hook, posted = _ship(monkeypatch, tmp_path, _SESSION)
    assert posted == [_SESSION] and not marker.exists()
    assert f"PRIVATE stale marker removed {_SESSION}" in _hook_log(tmp_path)


def test_stop_hook_treats_an_unreadable_marker_as_private(scratch, monkeypatch, tmp_path):
    """The private dir is a file, so the marker cannot even be looked up (the stat fails
    with NotADirectoryError, root or not): the session is not shipped."""
    private_dir = Path(os.environ["SYNAPSE_PRIVATE_DIR"])
    private_dir.write_text("not a directory", encoding="utf-8")
    _hook, posted = _ship(monkeypatch, tmp_path, _SESSION)
    assert posted == []
    log = _hook_log(tmp_path)
    assert f"PRIVATE unreadable {_SESSION}: NotADirectoryError" in log
    assert f"skip: private session {_SESSION}" in log


@pytest.mark.parametrize("session_id", ["..", "../victim", "a/b", "..\\victim"])
def test_stop_hook_rejects_a_session_id_that_is_not_a_file_name(
    scratch, monkeypatch, tmp_path, session_id
):
    """No marker can be named that: not private, and nothing outside the private dir is
    looked at or removed, however old."""
    private_dir = Path(os.environ["SYNAPSE_PRIVATE_DIR"])
    private_dir.mkdir(parents=True)
    victim = tmp_path / "victim"
    victim.write_text("keep", encoding="utf-8")
    os.utime(victim, (time.time() - 48 * 3600, time.time() - 48 * 3600))
    _hook, posted = _ship(monkeypatch, tmp_path, session_id)
    assert posted == [session_id]
    assert victim.read_text(encoding="utf-8") == "keep"
    assert "PRIVATE ignored a session id that is not a file name" in _hook_log(tmp_path)
