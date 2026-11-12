"""Every hook that sends a credential keeps it off a redirect to another origin.

urllib's stock redirect handler copies the Authorization header to wherever a server
points; the client core's (synapse_client.http.urlopen) does not. Each sender below is
driven at a real stdlib server on 127.0.0.1 that answers 302 to a second one on another
port: the second must see no Authorization header, and a redirect within the first
server must still carry it. The rule itself is unit-tested in test_client_core_http.py.
No test reaches a configured Synapse server; every file a hook reads at import is under
a temp dir (scratch_user).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.helpers.client_core import REPO, drop_scripts, load_script, scratch_user

_TOKEN = "tkREDIR4c9Wq2Zp7Hd"
_CODEX = REPO / "plugin-codex"
# One reply every sender accepts: JSON for the helpers, and an MCP result that carries a
# token for synapse_login's issue_machine_token call.
_REPLY = json.dumps(
    {"jsonrpc": "2.0", "id": 2, "result": {"structuredContent": {"token": "minted"}}}
).encode()


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def log_message(self, format: str, *args: Any) -> None:  # silence
        pass

    def _handle(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        self.server.seen.append(
            {
                "method": self.command,
                "path": self.path,
                "headers": {k.lower(): v for k, v in self.headers.items()},
            }
        )
        location = self.server.redirects.get(self.path)
        if location is not None:
            self.send_response(302)
            self.send_header("Location", location)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(_REPLY)))
        self.end_headers()
        self.wfile.write(_REPLY)

    do_GET = do_POST = _handle


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.seen: list[dict[str, Any]] = []
        self.redirects: dict[str, str] = {}

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


@pytest.fixture()
def servers() -> Iterator[tuple[_Server, _Server]]:
    """``home`` (the configured server) and ``away`` (another port). /hop on home
    redirects to away, /stay redirects within home; /land answers on both."""
    pair = (_Server(), _Server())
    for srv in pair:
        threading.Thread(target=srv.serve_forever, args=(0.02,), daemon=True).start()
    home, away = pair
    home.redirects["/hop"] = away.url + "/land"
    home.redirects["/stay"] = "/land"
    yield pair
    for srv in pair:
        srv.shutdown()
        srv.server_close()


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _codex_common(mp: pytest.MonkeyPatch, url: str) -> ModuleType:
    """common.py configured for the server ``url`` is on: it sends through the core's
    Transport, which gives the bearer to the configured server's origin only."""
    sys.modules.pop("common", None)
    common = _load("common", _CODEX / "scripts" / "common.py")
    mp.setattr(common, "BASE_URL", url.rsplit("/", 1)[0])
    mp.setattr(common, "TOKEN", _TOKEN)
    return common


def _config(mp: pytest.MonkeyPatch) -> ModuleType:
    config = load_script("config")
    mp.setattr(config, "INGEST_TOKEN", _TOKEN)
    return config


def _ingest_hook(mp: pytest.MonkeyPatch, url: str) -> None:
    hook = load_script("ingest_hook")
    mp.setattr(hook, "INGEST_URL", url)
    mp.setattr(hook, "INGEST_TOKEN", _TOKEN)
    hook._post_records([])


def _codex_stop_hook(mp: pytest.MonkeyPatch, url: str) -> None:
    sys.modules.pop("common", None)
    hook = _load("codex_stop_hook_redirect_test", _CODEX / "hooks" / "synapse_stop_hook.py")
    mp.setattr(hook, "INGEST_URL", url)
    mp.setattr(hook, "TOKEN", _TOKEN)
    hook._post([], "s-1")


def _login(mp: pytest.MonkeyPatch, url: str) -> None:
    sys.modules.pop("synapse_login", None)
    login = load_script("synapse_login")
    mp.setattr(login, "MCP_URL", url)
    assert login._mcp_call_token(_TOKEN) == "minted"


#: name -> send(monkeypatch, url): one request (or two, for the login's MCP handshake)
#: carrying ``Bearer _TOKEN`` to ``url``.
SENDERS: dict[str, Callable[[pytest.MonkeyPatch, str], Any]] = {
    "config.post_json": lambda mp, url: _config(mp).post_json(url, {}),
    "config.get_json": lambda mp, url: _config(mp).get_json(url),
    "config.request_json": lambda mp, url: _config(mp).request_json("GET", url),
    "codex common.post_json": lambda mp, url: _codex_common(mp, url).post_json(url, {}),
    "codex common.get_json": lambda mp, url: _codex_common(mp, url).get_json(url),
    "ingest_hook._post_records": _ingest_hook,
    "codex stop hook _post": _codex_stop_hook,
    "synapse_login._mcp_call_token": _login,
}


@pytest.fixture(params=list(SENDERS))
def send(request, monkeypatch, tmp_path) -> Iterator[Callable[[str], Any]]:
    scratch_user(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))  # the hooks extend it; put it back
    saved = {name: sys.modules.get(name) for name in ("common", "synapse_login")}
    yield lambda url: SENDERS[request.param](monkeypatch, url)
    drop_scripts()
    for name, mod in saved.items():
        sys.modules.pop(name, None)
        if mod is not None:
            sys.modules[name] = mod


def test_a_redirect_to_another_port_gets_no_bearer(send, servers):
    home, away = servers
    send(home.url + "/hop")
    assert home.seen and home.seen[0]["headers"]["authorization"] == f"Bearer {_TOKEN}"
    assert away.seen, "the redirect was not followed"
    assert all("authorization" not in req["headers"] for req in away.seen)


def test_a_redirect_within_the_server_keeps_the_bearer(send, servers):
    home, away = servers
    send(home.url + "/stay")
    landed = [req for req in home.seen if req["path"] == "/land"]
    assert landed, "the redirect was not followed"
    assert all(req["headers"]["authorization"] == f"Bearer {_TOKEN}" for req in landed)
    assert not away.seen
