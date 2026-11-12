"""The core's tool surface (plugin/synapse_client/tools.py and tool_schemas.json).

No database: the drift test runs scripts/gen_client_tool_schemas.py in a subprocess,
which imports the server with telemetry off and no .env, and compares what it writes
with the committed snapshot. ``inject_session`` is checked against the PreToolUse hooks
it ports, run as the harness runs them. ``call`` talks to a stdlib server on 127.0.0.1.
"""

from __future__ import annotations

import email.message
import json
import os
import socket
import subprocess
import sys
import threading
import urllib.error
from collections.abc import Iterator, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from tests.helpers.client_core import PLUGIN, REPO, use_core

use_core()
from synapse_client import tools  # noqa: E402
from synapse_client.api import Client  # noqa: E402
from synapse_client.http import ToolError  # noqa: E402
from synapse_client.settings import Settings  # noqa: E402

_SNAPSHOT = PLUGIN / "synapse_client" / "tool_schemas.json"
_GENERATOR = REPO / "scripts" / "gen_client_tool_schemas.py"
_SERVER_ORDER = [
    "recall",
    "recall_full_turns",
    "fetch",
    "fetch_session",
    "remember",
    "recall_feedback",
]


def _raw() -> list[dict[str, Any]]:
    return json.loads(_SNAPSHOT.read_text(encoding="utf-8"))["tools"]


def _raw_props(name: str) -> dict[str, Any]:
    (tool,) = [t for t in _raw() if t["name"] == name]
    return tool["inputSchema"]["properties"]


# ---------------------------------------------------------------------------
# The snapshot matches the server
# ---------------------------------------------------------------------------


def test_the_snapshot_is_what_the_generator_makes_now(tmp_path):
    """Fails when a tool, its description or its parameters change on the server and the
    snapshot was not regenerated: run scripts/gen_client_tool_schemas.py."""
    out = tmp_path / "tool_schemas.json"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SYNAPSE_", "LOGFIRE_"))}
    proc = subprocess.run(
        [sys.executable, str(_GENERATOR), "--out", str(out)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout == ""  # nothing (telemetry included) printed over the run
    assert out.read_bytes() == _SNAPSHOT.read_bytes(), "regenerate tool_schemas.json"


def test_the_snapshot_keeps_the_server_order_and_only_what_tools_list_shows():
    tools_ = _raw()
    assert [t["name"] for t in tools_] == _SERVER_ORDER
    assert "issue_machine_token" not in {t["name"] for t in tools_}  # hidden from tools/list
    for tool in tools_:
        assert set(tool) == {"name", "description", "inputSchema"}
        assert tool["description"].strip()
        assert tool["inputSchema"]["type"] == "object"


def test_every_session_parameter_the_core_fills_exists_on_the_server():
    """A renamed server parameter would otherwise be filled under its old name and
    rejected (the server forbids unknown arguments)."""
    for name, field in tools.SESSION_FIELD.items():
        assert field in _raw_props(name), (name, field)


def test_no_hidden_parameter_is_required():
    """Hiding a required parameter would leave the model unable to make a valid call."""
    for tool in _raw():
        required = set(tool["inputSchema"].get("required", ()))
        assert not required & tools.hidden_parameters(tool["name"]), tool["name"]


# ---------------------------------------------------------------------------
# schemas(): what the model is shown
# ---------------------------------------------------------------------------


def _params(name: str) -> dict[str, Any]:
    (tool,) = [t for t in tools.schemas() if t["name"] == name]
    return tool["parameters"]


def test_schemas_are_name_description_parameters_in_server_order():
    out = tools.schemas()
    assert [t["name"] for t in out] == _SERVER_ORDER
    raw = {t["name"]: t for t in _raw()}
    for tool in out:
        assert set(tool) == {"name", "description", "parameters"}
        assert tool["description"] == raw[tool["name"]]["description"]


def test_surface_is_hidden_everywhere():
    assert any("surface" in _raw_props(t["name"]) for t in _raw())  # there is one to hide
    for tool in tools.schemas():
        assert "surface" not in tool["parameters"]["properties"], tool["name"]


@pytest.mark.parametrize("name", ["recall", "recall_full_turns", "fetch_session"])
def test_self_session_is_hidden_where_the_core_fills_it(name):
    assert "self_session" in _raw_props(name)
    assert "self_session" not in _params(name)["properties"]


@pytest.mark.parametrize("name", ["recall_feedback", "remember"])
def test_session_id_is_hidden_where_the_core_fills_it(name):
    assert "session_id" in _raw_props(name)
    assert "session_id" not in _params(name)["properties"]


@pytest.mark.parametrize("name", ["recall_full_turns", "fetch_session"])
def test_the_session_to_read_stays_visible(name):
    """recall_full_turns' and fetch_session's session_id names the conversation to read
    (or "self"): the model sets it, so it is not hidden."""
    assert _params(name)["properties"]["session_id"] == _raw_props(name)["session_id"]


def test_nothing_else_is_hidden():
    for tool in _raw():
        name = tool["name"]
        kept = set(_params(name)["properties"])
        assert kept == set(_raw_props(name)) - tools.hidden_parameters(name)
        assert _params(name).get("required", []) == tool["inputSchema"].get("required", [])
        rest = {k: v for k, v in tool["inputSchema"].items() if k != "properties"}
        assert {k: v for k, v in _params(name).items() if k != "properties"} == rest


def test_schemas_are_fresh_objects_each_call():
    first = tools.schemas()
    first[0]["parameters"]["properties"].clear()
    first[0]["description"] = "changed"
    second = tools.schemas()
    assert second[0]["parameters"]["properties"] and second[0]["description"] != "changed"


def test_schemas_raise_when_the_snapshot_is_missing(monkeypatch):
    monkeypatch.setattr(tools, "SNAPSHOT", "no-such-snapshot.json")
    with pytest.raises(OSError):
        tools.schemas()


# ---------------------------------------------------------------------------
# inject_session: the hooks' rule
# ---------------------------------------------------------------------------

_SID = "s-1"
_CC_HOOK = REPO / "plugin" / "scripts" / "self_session_inject.py"
_CODEX_HOOK = REPO / "plugin-codex" / "hooks" / "pre_tool_use.py"
_MCP = "mcp__plugin_synapse_synapse__"

#: (tool, arguments): the cases tests/test_surface_inject.py runs the hooks on, plus
#: the empty-value and other-tool cases.
_CASES = [
    *((tool, {"query": "q"}) for tool in _SERVER_ORDER),
    *((tool, {}) for tool in ("recall", "recall_full_turns", "fetch_session")),
    ("fetch", {"ids": []}),
    ("recall", {"self_session": "s-orig"}),
    ("remember", {"session_id": "s-orig", "content": "x"}),
    ("recall", {"self_session": ""}),
    ("recall_feedback", {"session_id": None, "query": "q"}),
    ("recall_full_turns", {"session_id": "self", "query": "q"}),
    ("fetch_session", {"session_id": "abc"}),
    ("Bash", {"command": "ls"}),
    ("issue_machine_token", {}),
]


def _ids(case: tuple[str, dict[str, Any]]) -> str:
    return f"{case[0]}-{json.dumps(case[1], sort_keys=True)}"


def _hook(hook: Path, tool: str, arguments: dict[str, Any], session_id: str) -> dict[str, Any]:
    """What the hook makes of the call: its updatedInput, or the input unchanged when it
    emits nothing. Run as the harness runs it, with no Synapse configuration at all."""
    payload = {"tool_name": _MCP + tool, "session_id": session_id, "tool_input": arguments}
    proc = subprocess.run(
        [sys.executable, str(hook)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"},
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout.strip()
    return json.loads(out)["hookSpecificOutput"]["updatedInput"] if out else arguments


@pytest.mark.parametrize("hook", [_CC_HOOK, _CODEX_HOOK], ids=["claude-code", "codex"])
@pytest.mark.parametrize("case", _CASES, ids=[_ids(c) for c in _CASES])
def test_inject_session_matches_the_hooks(hook, case):
    tool, arguments = case
    expected = _hook(hook, tool, dict(arguments), _SID)
    assert tools.inject_session(tool, arguments, _SID) == expected


@pytest.mark.parametrize("tool", ["recall", "recall_full_turns", "fetch_session"])
def test_the_recall_family_gets_self_session(tool):
    assert tools.inject_session(tool, {"query": "q"}, _SID) == {"query": "q", "self_session": _SID}


@pytest.mark.parametrize("tool", ["recall_feedback", "remember"])
def test_session_id_tools_never_get_self_session(tool):
    out = tools.inject_session(tool, {"query": "q"}, _SID)
    assert out == {"query": "q", "session_id": _SID}


@pytest.mark.parametrize("tool", [*_SERVER_ORDER, "Bash"])
def test_no_surface_is_ever_added(tool):
    assert "surface" not in tools.inject_session(tool, {"query": "q"}, _SID)


def test_fetch_and_unknown_tools_are_left_alone():
    assert tools.inject_session("fetch", {"ids": ["e:1"]}, _SID) == {"ids": ["e:1"]}
    assert tools.inject_session("something_new", {"a": 1}, _SID) == {"a": 1}


def test_a_value_already_set_is_never_overwritten():
    assert tools.inject_session("recall", {"self_session": "s-orig"}, _SID) == {
        "self_session": "s-orig"
    }
    assert tools.inject_session("remember", {"session_id": "s-orig"}, _SID) == {
        "session_id": "s-orig"
    }


@pytest.mark.parametrize("session_id", [None, ""])
def test_no_session_id_injects_nothing(session_id):
    assert tools.inject_session("recall", {"query": "q"}, session_id) == {"query": "q"}


def test_inject_session_returns_a_new_dict():
    arguments = {"query": "q"}
    out = tools.inject_session("recall", arguments, _SID)
    assert out is not arguments and arguments == {"query": "q"}
    unchanged = tools.inject_session("fetch", arguments, _SID)
    assert unchanged is not arguments and unchanged == arguments
    assert tools.inject_session("recall", None, _SID) == {"self_session": _SID}


@pytest.mark.parametrize("arguments", ["not-a-dict", ["q"], 3])
def test_inject_session_refuses_arguments_that_are_not_a_mapping(arguments):
    with pytest.raises(TypeError):
        tools.inject_session("recall", arguments, _SID)


# ---------------------------------------------------------------------------
# call(): never raises
# ---------------------------------------------------------------------------


class _FakeClient:
    def __init__(self, result: str | BaseException = "ok") -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, Any], float]] = []

    def call_tool(self, name: str, arguments: Mapping[str, Any], timeout: float = 60) -> str:
        self.calls.append((name, dict(arguments), timeout))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def test_call_injects_the_session_and_returns_the_tools_text():
    client = _FakeClient("three results")
    out = tools.call(client, "recall", {"query": "q"}, session_id=_SID, timeout=5)
    assert out == "three results"
    assert client.calls == [("recall", {"query": "q", "self_session": _SID}, 5)]


def test_call_defaults_to_the_api_timeout():
    client = _FakeClient()
    tools.call(client, "fetch", {"ids": []})
    assert client.calls == [("fetch", {"ids": []}, tools.DEFAULT_TIMEOUT)]


def _http(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://s/mcp", code, "x", email.message.Message(), None)


@pytest.mark.parametrize(
    ("err", "expected"),
    [
        (ToolError("recall", "query: field required"), {"error": "query: field required"}),
        (
            ToolError("recall", "Unknown tool", code=-32602),
            {"error": "Unknown tool", "code": -32602},
        ),
        (ToolError("recall", ""), {"error": "the tool reported an error"}),
        (_http(401), {"error": "Synapse could not answer: HTTP 401 Unauthorized"}),
        (_http(503), {"error": "Synapse could not answer: HTTP 503 Service Unavailable"}),
        (
            urllib.error.URLError(ConnectionRefusedError()),
            {"error": "Synapse could not answer: connection refused"},
        ),
        (TimeoutError(), {"error": "Synapse could not answer: timed out"}),
        (ValueError("a reply with no answer"), {"error": "Synapse could not answer: ValueError"}),
        (RuntimeError("boom"), {"error": "Synapse could not answer: RuntimeError"}),
    ],
    ids=lambda v: type(v).__name__ if isinstance(v, BaseException) else "",
)
def test_call_turns_every_failure_into_an_error_object(err, expected):
    out = tools.call(_FakeClient(err), "recall", {"query": "q"}, session_id=_SID)
    assert json.loads(out) == {**expected, "tool": "recall"}


def test_a_transport_error_never_echoes_its_message():
    secret = "https://s/mcp?token=tkLEAK123"
    out = tools.call(_FakeClient(urllib.error.URLError(secret)), "recall", {})
    assert "tkLEAK123" not in out and json.loads(out)["error"]


def test_call_refuses_arguments_that_are_not_an_object_without_calling():
    client = _FakeClient()
    out = tools.call(client, "recall", "q", session_id=_SID)
    assert json.loads(out) == {"error": "the arguments must be a JSON object", "tool": "recall"}
    assert client.calls == []


# ---------------------------------------------------------------------------
# call(): end to end over HTTP, against a stdlib MCP stand-in on 127.0.0.1
# ---------------------------------------------------------------------------


class _McpHandler(BaseHTTPRequestHandler):
    server: _McpServer

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        self.server.seen.append({"headers": dict(self.headers.items()), "body": body})
        params = body["params"]
        if params["name"] == "fails":
            result = {"content": [{"type": "text", "text": "it broke"}], "isError": True}
        else:
            text = json.dumps(params["arguments"], sort_keys=True)
            result = {"content": [{"type": "text", "text": text}], "isError": False}
        data = json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": result}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class _McpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _McpHandler)
        self.seen: list[dict[str, Any]] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


@pytest.fixture()
def mcp() -> Iterator[_McpServer]:
    server = _McpServer()
    thread = threading.Thread(target=server.serve_forever, args=(0.02,), daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def _client(url: str) -> Client:
    env = {"SYNAPSE_URL": url, "SYNAPSE_INGEST_TOKEN": "tkTOOLS4v8"}
    return Client(Settings.load(env=env, home="/nonexistent-synapse-home"), "synapse-test/1.0")


def test_call_end_to_end_sends_the_injected_session(mcp):
    out = tools.call(_client(mcp.url), "remember", {"content": "x"}, session_id=_SID)
    assert json.loads(out) == {"content": "x", "session_id": _SID}
    (seen,) = mcp.seen
    assert seen["body"]["method"] == "tools/call"
    assert seen["headers"]["Authorization"] == "Bearer tkTOOLS4v8"


def test_call_end_to_end_reports_a_tool_error(mcp):
    out = tools.call(_client(mcp.url), "fails", {}, session_id=_SID)
    assert json.loads(out) == {"error": "it broke", "tool": "fails"}


def test_call_end_to_end_when_nothing_listens():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    out = tools.call(_client(f"http://127.0.0.1:{port}"), "recall", {"query": "q"}, timeout=5)
    assert json.loads(out) == {
        "error": "Synapse could not answer: connection refused",
        "tool": "recall",
    }
