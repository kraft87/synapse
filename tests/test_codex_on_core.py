"""The Codex hooks on the client core (plugin-codex over plugin/synapse_client).

Every hook is loaded by path the way Codex runs it, with HOME, SYNAPSE_HOME and every
other file a hook reads at import under a temp dir (scratch_user), and talks to a local
stdlib server on 127.0.0.1. No test reaches a configured Synapse server or reads a real
credential.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import threading
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
