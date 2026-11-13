"""The Hermes memory provider (plugin/synapse_client/harnesses/hermes/provider.py).

The provider is tested as Hermes loads it: ``HermesHarness.install`` builds the plugin
folder in a temp Hermes home, the folder is imported under a package name of its own,
and its ``register(ctx)`` hands over a new provider. Hermes itself is not imported:
``agent.memory_provider`` is a stub with the same base class and
``spawn_context_thread``. The server is a local fake on an ephemeral port, and
SYNAPSE_HOME a temp dir.
"""

from __future__ import annotations

import importlib.util
import json
import socket
import sys
import threading
import time
import types
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from tests.helpers.client_core import PLUGIN, scratch_user, use_core

use_core()
from synapse_client import __version__  # noqa: E402
from synapse_client.harnesses.hermes import install as hi  # noqa: E402
from synapse_client.harnesses.hermes import messages as hm  # noqa: E402
from synapse_client.outbox import Outbox  # noqa: E402

_TOKEN = "tkHERMES-provider-7c1"
_SID = "20261112_101500_ab12cd"

# ---------------------------------------------------------------------------
# Hermes's side: the base class, and the loader
# ---------------------------------------------------------------------------

_STUB = """
import abc
import threading

SPAWNED = []


def spawn_context_thread(target, *, name, daemon=True, args=(), kwargs=None):
    SPAWNED.append(name)
    return threading.Thread(target=target, args=args, kwargs=kwargs, name=name, daemon=daemon)


class MemoryProvider(abc.ABC):
    pre_compress_checkpoint_api_version = 1

    @property
    @abc.abstractmethod
    def name(self): ...

    @abc.abstractmethod
    def is_available(self): ...

    @abc.abstractmethod
    def initialize(self, session_id, **kwargs): ...

    @abc.abstractmethod
    def get_tool_schemas(self): ...

    def unavailable_reason(self):
        return ""

    def system_prompt_block(self):
        return ""

    def prefetch(self, query, *, session_id=""):
        return ""

    def queue_prefetch(self, query, *, session_id=""):
        pass

    def recall_status(self):
        return None

    def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None,
                  turn_author=None):
        pass

    def handle_tool_call(self, tool_name, args, **kwargs):
        raise NotImplementedError

    def shutdown(self):
        pass

    def on_turn_start(self, turn_number, message, **kwargs):
        pass

    def identity_signature(self):
        return {}

    def on_session_end(self, messages):
        pass

    def on_session_switch(self, new_session_id, *, parent_session_id="", reset=False,
                          rewound=False, **kwargs):
        pass

    def on_pre_compress(self, messages):
        return ""

    def on_delegation(self, task, result, *, child_session_id="", **kwargs):
        pass

    def get_config_schema(self):
        return []

    def save_config(self, values, hermes_home):
        pass

    def on_memory_write(self, action, target, content, metadata=None):
        pass

    def backup_paths(self):
        return []
"""


@pytest.fixture()
def hermes_stub(monkeypatch) -> types.ModuleType:
    package = types.ModuleType("agent")
    package.__path__ = []
    module = types.ModuleType("agent.memory_provider")
    exec(compile(_STUB, "agent/memory_provider.py", "exec"), module.__dict__)
    package.memory_provider = module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agent", package)
    monkeypatch.setitem(sys.modules, "agent.memory_provider", module)
    return module


class _Ctx:
    def __init__(self) -> None:
        self.providers: list[Any] = []

    def register_memory_provider(self, provider: Any) -> None:
        self.providers.append(provider)


def _load(folder: Path) -> types.ModuleType:
    """Import the plugin folder the way Hermes does: a package under its own name."""
    name = f"_hermes_user_memory_test.synapse__{uuid.uuid4().hex[:16]}"
    sys.modules.setdefault("_hermes_user_memory_test", types.ModuleType("_hermes_user_memory_test"))
    spec = importlib.util.spec_from_file_location(
        name, folder / "__init__.py", submodule_search_locations=[str(folder)]
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def user(monkeypatch, tmp_path) -> Path:
    home = scratch_user(monkeypatch, tmp_path)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    return home


@pytest.fixture()
def plugin(hermes_stub, user, tmp_path) -> Iterator[types.ModuleType]:
    home = tmp_path / "hermes-a"
    hi.HermesHarness(which=lambda name: None, env={}).install("http://unused", hermes_home=home)
    before = set(sys.modules)
    module = _load(home / "plugins" / "synapse")
    try:
        yield module
    finally:
        for name in set(sys.modules) - before:
            if name.startswith("_hermes_user_memory_test"):
                del sys.modules[name]
        sys.modules.pop("_hermes_user_memory_test", None)


def _new(module: types.ModuleType) -> Any:
    ctx = _Ctx()
    module.register(ctx)
    (provider,) = ctx.providers
    return provider


# ---------------------------------------------------------------------------
# The fake server
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def log_message(self, *args: Any) -> None:
        pass

    def _send(self, status: int, body: Any) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _record(self, body: Any = None) -> None:
        self.server.seen.append(
            {"method": self.command, "path": self.path, "headers": dict(self.headers), "body": body}
        )
        if self.server.hang.is_set():
            self.server.release.wait(30)

    def do_GET(self) -> None:
        self._record()
        if self.path.startswith("/context"):
            reply = {"status": "ok", "text": "BOARD-TEXT", "trust": "full"}
            if not self.server.older:
                reply["preferences"] = "[Synapse preferences]\n  - (likes) Kyle likes small commits"
            self._send(200, reply)
        elif self.path.startswith("/preferences/top") and self.server.older:
            self._send(200, {"items": [{"pref": "Kyle likes small commits", "polarity": "like"}]})
        else:
            self._send(404, {})

    def do_PUT(self) -> None:
        self._record()
        if self.path.startswith("/private-sessions/") and self.server.private_status == 200:
            self._send(200, {"private": True})
        else:
            self._send(self.server.private_status, {"error": "down"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"null")
        self._record(body)
        if self.path == "/ingest":
            if self.server.ingest_status != 200:
                self._send(self.server.ingest_status, {"error": "down"})
            else:
                self._send(200, {"status": "ok", "episodes": 1})
        elif self.path == "/mcp":
            if self.server.tool_error:
                reply = {
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "error": {"code": -32602, "message": "bad"},
                }
            else:
                text = self.server.tool_text or json.dumps(body["params"]["arguments"])
                reply = {
                    "jsonrpc": "2.0",
                    "id": body["id"],
                    "result": {"content": [{"type": "text", "text": text}]},
                }
            self._send(200, reply)
        else:
            self._send(404, {})


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.seen: list[dict[str, Any]] = []
        self.hang = threading.Event()
        self.release = threading.Event()
        self.ingest_status = 200
        self.private_status = 200
        #: A server from before /context carried the preferences block.
        self.older = False
        self.tool_error = False
        self.tool_text = ""

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def posts(self, path: str = "/ingest") -> list[Any]:
        return [r["body"] for r in self.seen if r["method"] == "POST" and r["path"] == path]

    def puts(self) -> list[str]:
        return [r["path"] for r in self.seen if r["method"] == "PUT"]


@pytest.fixture()
def server(user) -> Iterator[_Server]:
    srv = _Server()
    thread = threading.Thread(target=srv.serve_forever, args=(0.02,), daemon=True)
    thread.start()
    _configure(user, srv.url)
    try:
        yield srv
    finally:
        srv.release.set()
        srv.shutdown()
        srv.server_close()


def _configure(user: Path, url: str) -> None:
    config = user / ".synapse" / "config.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps({"url": url, "token": _TOKEN}), encoding="utf-8")


def _closed_port() -> str:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return f"http://127.0.0.1:{port}"


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


def _turn(n: int, *, tools: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {"role": "user", "content": f"question {n}", "message_uid": f"u{n}", "timestamp": 1.0 + n}
    ]
    if tools:
        rows += [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": '{"cmd": "ls"}'},
                    }
                ],
                "message_uid": f"c{n}",
            },
            {
                "role": "tool",
                "content": "a.txt b.txt",
                "tool_call_id": "c1",
                "name": "terminal",
                "tool_name": "terminal",
                "message_uid": f"t{n}",
            },
        ]
    rows.append(
        {
            "role": "assistant",
            "content": f"answer {n}",
            "message_uid": f"a{n}",
            "api_content": "SECRET-API-CONTENT",
            "reasoning": "thinking",
        }
    )
    return rows


def _conversation(count: int, *, tools: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [{"role": "system", "content": "You are Hermes."}]
    for n in range(1, count + 1):
        rows += _turn(n, tools=tools)
    return rows


def _users(records: list[dict[str, Any]]) -> list[str]:
    return [r["content"] for r in records if r["role"] == "user"]


def _start(provider: Any, home: Path, session_id: str = _SID, **kwargs: Any) -> Any:
    kwargs.setdefault("platform", "cli")
    kwargs.setdefault("agent_context", "primary")
    provider.initialize(session_id, hermes_home=str(home), **kwargs)
    return provider


# ---------------------------------------------------------------------------
# Session start and per prompt
# ---------------------------------------------------------------------------


def test_the_session_start_block_and_the_nudge(plugin, server, tmp_path, hermes_stub):
    work = tmp_path / "my-project"
    work.mkdir()
    provider = _start(_new(plugin), tmp_path / "hermes-a", cwd=str(work))
    block = provider.system_prompt_block()
    assert block.endswith(
        "BOARD-TEXT\n\n[Synapse preferences]\n  - (likes) Kyle likes small commits"
    )
    # One request for the board and the preferences, on one thread.
    (ctx_request,) = [r for r in server.seen if r["method"] == "GET"]
    assert ctx_request["path"] == "/context?project=my-project&parts=board%2Cpreferences"
    headers = {k.lower(): v for k, v in ctx_request["headers"].items()}
    assert headers["user-agent"] == f"synapse-client/{__version__} hermes"
    assert headers["authorization"] == f"Bearer {_TOKEN}"
    assert {"synapse-session-start", "synapse-outbox"} <= set(hermes_stub.SPAWNED)
    assert not {"synapse-board", "synapse-preferences"} & set(hermes_stub.SPAWNED)

    seen = len(server.seen)
    nudge = provider.prefetch("what did we decide about the cache?", session_id=_SID)
    assert "Call remember FIRST" in nudge and ": recall (skip only" in nudge
    assert len(server.seen) == seen  # no request
    assert provider.recall_status() is None
    assert provider.queue_prefetch("x") is None
    assert provider.on_pre_compress([]) == ""
    assert provider.pre_compress_checkpoint_api_version == 1
    assert provider.name == "synapse"


def test_a_server_from_before_the_preferences_block_is_asked_for_its_top_preferences(
    plugin, server, tmp_path
):
    server.older = True
    provider = _start(_new(plugin), tmp_path / "hermes-a")
    block = provider.system_prompt_block()
    assert block.endswith(
        "BOARD-TEXT\n\n[Synapse preferences]\n  - (likes) Kyle likes small commits"
    )
    gets = [r["path"].split("?")[0] for r in server.seen if r["method"] == "GET"]
    assert gets == ["/context", "/preferences/top"]


def test_the_session_start_keeps_to_its_budget(plugin, server, tmp_path):
    home = tmp_path / "hermes-a"
    outbox = Outbox(home / "synapse" / "outbox")
    for _ in range(3):
        outbox.put({"format": "messages", "records": []}, reason="HTTP 503")
    server.hang.set()
    provider = _new(plugin)
    started = time.monotonic()
    _start(provider, home)
    elapsed = time.monotonic() - started
    assert elapsed < plugin_budget(plugin) + 1.5
    block = provider.system_prompt_block()
    # The local line leads; the board and preferences did not come back in time.
    assert block.startswith("Synapse uploads have been failing since")
    assert "BOARD-TEXT" not in block


def plugin_budget(plugin: types.ModuleType) -> float:
    return sys.modules[plugin.__name__ + ".synapse_client.harnesses.hermes.provider"].START_BUDGET


# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------


def test_a_finished_turn_is_uploaded(plugin, server, tmp_path):
    work = tmp_path / "proj"
    work.mkdir()
    provider = _start(_new(plugin), tmp_path / "hermes-a", cwd=str(work))
    provider.on_turn_start(1, "question 1", model="anthropic/claude-x", platform="cli")
    messages = _conversation(1, tools=True)
    snapshot = json.dumps(messages, sort_keys=True)
    provider.sync_turn("question 1", "answer 1", session_id=_SID, messages=messages)
    assert json.dumps(messages, sort_keys=True) == snapshot
    (body,) = server.posts()
    assert body == {
        "format": "messages",
        "client": "hermes",
        "session_id": _SID,
        "project": "proj",
        "cwd": str(work),
        "model": "anthropic/claude-x",
        "source": "hermes-sync",
        "records": hm.to_records(messages),
    }
    assert "SECRET-API-CONTENT" not in json.dumps(body)
    assert [r["role"] for r in body["records"]] == ["user", "assistant", "tool", "assistant"]


def test_an_unfinished_turn_waits(plugin, server, tmp_path):
    provider = _start(_new(plugin), tmp_path / "hermes-a")
    running = _conversation(1, tools=True)[:-1]  # ends in the tool result
    provider.sync_turn("question 1", "", session_id=_SID, messages=running)
    provider.sync_turn("question 1", "", session_id=_SID, messages=running[:-1])  # tool call
    assert server.posts() == []
    provider.sync_turn(
        "question 1", "answer 1", session_id=_SID, messages=_conversation(1, tools=True)
    )
    assert len(server.posts()) == 1


def test_no_turn_is_sent_twice(plugin, server, tmp_path):
    provider = _start(_new(plugin), tmp_path / "hermes-a")
    provider.sync_turn("q", "a", session_id=_SID, messages=_conversation(2))
    provider.sync_turn("q", "a", session_id=_SID, messages=_conversation(2))
    provider.sync_turn("q", "a", session_id=_SID, messages=_conversation(3))
    first, second = server.posts()
    assert _users(first["records"]) == ["question 1", "question 2"]
    # The turn before leads the batch as context.
    assert _users(second["records"]) == ["question 2", "question 3"]


def test_messages_none_sends_the_two_strings(plugin, server, tmp_path):
    provider = _start(_new(plugin), tmp_path / "hermes-a")
    provider.sync_turn("what is up", "not much", session_id=_SID)
    provider.sync_turn("what is up", "not much", session_id=_SID)
    (body,) = server.posts()
    assert body["records"] == [
        {"role": "user", "content": "what is up"},
        {"role": "assistant", "content": "not much"},
    ]


def test_a_resumed_session_looks_back_ten_turns(plugin, server, tmp_path):
    provider = _start(_new(plugin), tmp_path / "hermes-a", resumed=True, message_count=60)
    provider.sync_turn("q", "a", session_id=_SID, messages=_conversation(15))
    provider.sync_turn("q", "a", session_id=_SID, messages=_conversation(16))
    first, second = server.posts()
    assert _users(first["records"]) == [f"question {n}" for n in range(5, 16)]
    assert _users(second["records"]) == ["question 15", "question 16"]


def test_a_new_session_sends_its_whole_history(plugin, server, tmp_path):
    provider = _start(_new(plugin), tmp_path / "hermes-a")
    provider.sync_turn("q", "a", session_id=_SID, messages=_conversation(15))
    (body,) = server.posts()
    assert len(_users(body["records"])) == 15


def test_a_failed_upload_waits_in_the_outbox_and_drains_after_a_success(plugin, server, tmp_path):
    home = tmp_path / "hermes-a"
    provider = _start(_new(plugin), home)
    server.ingest_status = 503
    provider.sync_turn("q", "a", session_id=_SID, messages=_conversation(1))
    outbox = Outbox(home / "synapse" / "outbox")
    assert outbox.pending() == 1
    assert outbox.last_error() == "HTTP 503 Service Unavailable"
    provider.sync_turn("q", "a", session_id=_SID, messages=_conversation(1))
    assert len(server.posts()) == 1  # handled: not tried again

    server.ingest_status = 200
    provider.sync_turn("q", "a", session_id=_SID, messages=_conversation(2))
    failed, fresh, drained = server.posts()
    assert drained == failed
    assert _users(fresh["records"]) == ["question 1", "question 2"]
    assert outbox.pending() == 0


def test_a_private_session_sends_nothing_and_keeps_its_turns(plugin, server, tmp_path):
    provider = _start(_new(plugin), tmp_path / "hermes-a")
    marker = tmp_path / "private" / _SID
    marker.parent.mkdir(parents=True)
    marker.write_text("", encoding="utf-8")
    provider.sync_turn("q", "a", session_id=_SID, messages=_conversation(1))
    provider.on_session_end(_conversation(1))
    assert server.posts() == []
    marker.unlink()
    provider.sync_turn("q", "a", session_id=_SID, messages=_conversation(2))
    (body,) = server.posts()
    assert _users(body["records"]) == ["question 1", "question 2"]


def test_the_provider_does_not_drain_what_went_private_while_it_waited(plugin, server, tmp_path):
    """An upload that failed waits in the outbox; its session is then taken off the
    record. The provider's drain drops it instead of sending it."""
    home = tmp_path / "hermes-a"
    provider = _start(_new(plugin), home)
    server.ingest_status = 503
    provider.sync_turn("q", "a", session_id=_SID, messages=_conversation(1))
    outbox = Outbox(home / "synapse" / "outbox")
    assert outbox.pending() == 1 and len(server.posts()) == 1
    marker = tmp_path / "private" / _SID
    marker.parent.mkdir(parents=True)
    marker.write_text("", encoding="utf-8")
    server.ingest_status = 200
    provider._drain(5.0)
    assert len(server.posts()) == 1 and outbox.pending() == 0


def test_compression_and_session_end_upload_what_is_left(plugin, server, tmp_path):
    provider = _start(_new(plugin), tmp_path / "hermes-a")
    assert provider.on_pre_compress(_conversation(2)) == ""
    provider.on_session_end(_conversation(3))
    first, second = server.posts()
    assert _users(first["records"]) == ["question 1", "question 2"]
    assert _users(second["records"]) == ["question 2", "question 3"]


def test_a_session_switch_rebinds(plugin, server, tmp_path):
    provider = _start(_new(plugin), tmp_path / "hermes-a")
    provider.sync_turn("q", "a", session_id=_SID, messages=_conversation(2))
    provider.on_session_switch(_SID, rewound=True)
    provider.sync_turn("q", "a", messages=_conversation(2))
    assert len(server.posts()) == 1

    provider.on_session_switch("20261112_120000_ffffff", parent_session_id=_SID, reason="branch")
    provider.sync_turn("q", "a", messages=_conversation(2))
    second = server.posts()[1]
    assert second["session_id"] == "20261112_120000_ffffff"
    assert _users(second["records"]) == ["question 1", "question 2"]

    provider.on_session_switch("20261112_120000_ffffff", reset=True)
    provider.sync_turn("q", "a", messages=_conversation(2))
    assert len(server.posts()) == 3


_NEXT = "20261112_120000_ffffff"


def _private(tmp_path: Path, session_id: str) -> Path:
    marker = tmp_path / "private" / session_id
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("", encoding="utf-8")
    return marker


@pytest.mark.parametrize("reason", ["compression", "branch", None])
def test_a_conversation_that_goes_on_under_a_new_id_stays_private(plugin, server, tmp_path, reason):
    """Hermes gives a conversation it compresses or branches a new session id. Private
    mode is kept per id, so the turns spoken off the record went up under the new one."""
    provider = _start(_new(plugin), tmp_path / "hermes-a")
    _private(tmp_path, _SID)
    provider.sync_turn("q", "a", session_id=_SID, messages=_conversation(2))
    provider.on_session_switch(
        _NEXT, parent_session_id=_SID, **({"reason": reason} if reason else {})
    )
    assert (tmp_path / "private" / _NEXT).is_file()
    assert server.puts() == [f"/private-sessions/{_NEXT}"]
    provider.sync_turn("q", "a", messages=_conversation(3))
    provider.on_pre_compress(_conversation(3))
    provider.on_session_end(_conversation(3))
    assert server.posts() == []


def test_a_private_parent_is_enough_to_stay_private(plugin, server, tmp_path):
    provider = _start(_new(plugin), tmp_path / "hermes-a")
    _private(tmp_path, "20261111_090000_aaaaaa")
    provider.on_session_switch(_NEXT, parent_session_id="20261111_090000_aaaaaa", reason="branch")
    provider.sync_turn("q", "a", messages=_conversation(1))
    assert (tmp_path / "private" / _NEXT).is_file() and server.posts() == []


@pytest.mark.parametrize(
    "switch",
    [
        {"parent_session_id": _SID, "reason": "resume"},  # /resume: another conversation
        {"reset": True, "reason": "new_session"},  # /new
    ],
)
def test_another_conversation_does_not_inherit_private_mode(plugin, server, tmp_path, switch):
    provider = _start(_new(plugin), tmp_path / "hermes-a")
    _private(tmp_path, _SID)
    provider.on_session_switch(_NEXT, **switch)
    assert not (tmp_path / "private" / _NEXT).exists() and server.puts() == []
    provider.sync_turn("q", "a", messages=_conversation(1))
    (body,) = server.posts()
    assert body["session_id"] == _NEXT


def test_turns_spoken_off_the_record_are_not_looked_back_at(plugin, server, tmp_path):
    """The new id's marker goes (`private off`, or its twelve hours): what was said
    before stays unsent, where any other switch looks back LOOKBACK turns."""
    provider = _start(_new(plugin), tmp_path / "hermes-a")
    _private(tmp_path, _SID)
    provider.on_session_switch(_NEXT, parent_session_id=_SID, reason="compression")
    provider.sync_turn("q", "a", messages=_conversation(2))
    (tmp_path / "private" / _NEXT).unlink()
    provider.sync_turn("q", "a", messages=_conversation(3))
    assert server.posts() == []
    provider.sync_turn("q", "a", messages=_conversation(4))
    (body,) = server.posts()
    assert body["session_id"] == _NEXT
    assert _users(body["records"]) == ["question 3", "question 4"]


def test_a_server_down_at_the_switch_is_told_at_the_next_turn(plugin, server, tmp_path):
    provider = _start(_new(plugin), tmp_path / "hermes-a")
    _private(tmp_path, _SID)
    server.private_status = 503
    provider.on_session_switch(_NEXT, parent_session_id=_SID, reason="compression")
    told = len(server.puts())
    assert told >= 1 and (tmp_path / "private" / _NEXT).is_file()
    (tmp_path / "private" / _NEXT).unlink()  # the marker alone would not have lasted
    provider.sync_turn("q", "a", messages=_conversation(1))
    assert len(server.puts()) > told and server.posts() == []
    server.private_status = 200
    told = len(server.puts())
    provider.sync_turn("q", "a", messages=_conversation(2))
    assert len(server.puts()) == told + 1 and server.posts() == []
    provider.sync_turn("q", "a", messages=_conversation(3))
    assert len(server.puts()) == told + 1  # told once: not asked again


def test_a_session_that_is_off_carries_nothing(plugin, server, tmp_path):
    """A provider that uploads nothing writes nothing either: no marker, no request."""
    provider = _start(_new(plugin), tmp_path / "hermes-a", agent_context="subagent")
    _private(tmp_path, _SID)
    provider.on_session_switch(_NEXT, parent_session_id=_SID, reason="compression")
    assert not (tmp_path / "private" / _NEXT).exists() and server.puts() == []


# ---------------------------------------------------------------------------
# Where the provider works
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"platform": "cli", "agent_context": "primary"}, "active"),
        ({}, "active"),
        ({"platform": "", "agent_context": ""}, "active"),
        ({"platform": "tui", "user_id": "kyle@example.net"}, "active"),
        ({"platform": "desktop", "agent_context": "primary"}, "active"),
        ({"platform": "acp"}, "active"),
        ({"platform": "cron", "agent_context": "cron"}, "inert"),
        ({"platform": "cli", "agent_context": "cron"}, "inert"),
        ({"platform": "cli", "agent_context": "subagent"}, "inert"),
        ({"platform": "subagent", "agent_context": "subagent"}, "inert"),
        ({"platform": "cli", "agent_context": "flush"}, "inert"),
        ({"platform": "api_server"}, "inert"),
        ({"platform": "telegram", "chat_id": "123", "user_id": "9"}, "inert"),
        ({"platform": "discord", "gateway_session_key": "agent:main:discord:dm:1"}, "inert"),
        ({"platform": "cli", "chat_id": "123"}, "inert"),
        ({"platform": "whatsapp"}, "inert"),
    ],
)
def test_scope(plugin, kwargs, expected):
    module = sys.modules[plugin.__name__ + ".synapse_client.harnesses.hermes.provider"]
    assert module.scope(kwargs) == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"platform": "telegram", "chat_id": "123", "user_id": "9", "user_name": "someone"},
        {"platform": "api_server"},
        {"platform": "cli", "agent_context": "subagent", "parent_session_id": _SID},
        {"platform": "cli", "agent_context": "flush"},
        {"platform": "cron", "agent_context": "cron"},
    ],
)
def test_an_inert_session_shows_nothing_sends_nothing_and_refuses_tools(
    plugin, server, tmp_path, kwargs
):
    provider = _start(_new(plugin), tmp_path / "hermes-a", **kwargs)
    assert provider.system_prompt_block() == ""
    assert provider.prefetch("anything") == ""
    provider.sync_turn("q", "a", session_id=_SID, messages=_conversation(1))
    provider.on_pre_compress(_conversation(1))
    provider.on_session_end(_conversation(1))
    out = json.loads(provider.handle_tool_call("recall", {"query": "x"}))
    assert "local sessions only" in out["error"]
    assert server.seen == []


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def test_before_initialize_the_schemas_are_served_and_a_call_is_refused(plugin, server):
    provider = _new(plugin)
    schemas = provider.get_tool_schemas()
    assert [s["name"] for s in schemas][:2] == ["recall", "recall_full_turns"]
    assert all(set(s) == {"name", "description", "parameters"} for s in schemas)
    out = json.loads(provider.handle_tool_call("recall", {"query": "cache"}))
    assert "local sessions only" in out["error"]
    assert server.seen == []


def test_a_tool_call_reaches_the_server_with_its_arguments(plugin, server, tmp_path):
    provider = _start(_new(plugin), tmp_path / "hermes-a")
    out = provider.handle_tool_call("recall", {"query": "cache"})
    assert json.loads(out) == {"query": "cache", "self_session": _SID}  # the fake echoes
    (call,) = server.posts("/mcp")
    assert call["params"]["name"] == "recall"


def test_a_tool_call_carries_the_session_id(plugin, server, tmp_path):
    provider = _start(_new(plugin), tmp_path / "hermes-a")
    server.tool_text = "Found 2 memories."
    out = provider.handle_tool_call("recall", {"query": "cache"}, hermes_home="x", caller_gsk="y")
    assert json.loads(out) == {"result": "Found 2 memories."}
    provider.handle_tool_call("remember", {"content": "Kyle uses fish"})
    recall, remember = server.posts("/mcp")
    assert recall["params"]["arguments"] == {"query": "cache", "self_session": _SID}
    assert remember["params"]["arguments"] == {"content": "Kyle uses fish", "session_id": _SID}
    server.tool_text = '{"ok": true, "id": "n:12"}'
    assert json.loads(provider.handle_tool_call("remember", {"content": "x"})) == {
        "ok": True,
        "id": "n:12",
    }


def test_a_failed_tool_call_is_a_json_error(plugin, server, tmp_path, user):
    provider = _start(_new(plugin), tmp_path / "hermes-a")
    server.tool_error = True
    out = json.loads(provider.handle_tool_call("recall", {"query": "x"}))
    assert out == {"error": "bad", "tool": "recall", "code": -32602}
    out = json.loads(provider.handle_tool_call("recall", "not an object"))  # type: ignore[arg-type]
    assert out["error"] == "the arguments must be a JSON object"


# ---------------------------------------------------------------------------
# The compression cut
# ---------------------------------------------------------------------------

_CUT = "2026-11-12T15:00:00.250000+00:00"
_OTHER = "20261112_120000_ffffff"


def _module(plugin: types.ModuleType) -> types.ModuleType:
    return sys.modules[plugin.__name__ + ".synapse_client.harnesses.hermes.provider"]


def _compress(provider: Any) -> None:
    """An in-place compression as Hermes runs it: the upload, the commit, the switch."""
    provider.on_pre_compress(_conversation(2))
    provider.on_session_end(_conversation(2))
    provider.on_session_switch(_SID, parent_session_id=_SID, reset=False, reason="compression")


def _sent(provider: Any, server: _Server, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    """The arguments ``tool`` reached the server with."""
    provider.handle_tool_call(tool, args)
    return server.posts("/mcp")[-1]["params"]["arguments"]


def test_no_cut_before_a_compression(plugin, server, tmp_path):
    provider = _start(_new(plugin), tmp_path / "hermes-a")
    assert _sent(provider, server, "recall", {"query": "x"}) == {"query": "x", "self_session": _SID}
    provider.on_session_switch(_SID, rewound=True)
    provider.on_session_switch(_SID, parent_session_id=_SID, reason="resume")
    assert "self_since" not in _sent(provider, server, "recall_full_turns", {"query": "x"})


def test_an_in_place_compression_sets_the_cut_on_the_two_recall_tools(
    plugin, server, tmp_path, monkeypatch
):
    monkeypatch.setattr(_module(plugin), "_clock", lambda: _CUT)
    provider = _start(_new(plugin), tmp_path / "hermes-a")
    _compress(provider)
    own = {"self_session": _SID, "self_since": _CUT}
    assert _sent(provider, server, "recall", {"query": "x"}) == {"query": "x", **own}
    assert _sent(provider, server, "recall_full_turns", {"query": "x"}) == {"query": "x", **own}
    assert _sent(provider, server, "fetch_session", {"session_id": "self"}) == {
        "session_id": "self",
        "self_session": _SID,
    }
    assert _sent(provider, server, "recall_feedback", {"query": "x"}) == {
        "query": "x",
        "session_id": _SID,
    }
    assert _sent(provider, server, "remember", {"content": "x"}) == {
        "content": "x",
        "session_id": _SID,
    }
    assert _sent(provider, server, "fetch", {"ids": ["e:1"]}) == {"ids": ["e:1"]}
    # Never over a value the call carries, never with someone else's session.
    given = {"query": "x", "self_since": "2026-11-01T00:00:00Z"}
    assert _sent(provider, server, "recall", given)["self_since"] == "2026-11-01T00:00:00Z"
    assert "self_since" not in _sent(
        provider, server, "recall", {"query": "x", "self_session": "s"}
    )
    provider.on_session_switch(_SID, rewound=True)  # an undo keeps the id, and the cut
    assert _sent(provider, server, "recall", {"query": "x"})["self_since"] == _CUT


def test_the_cut_is_the_client_clock_at_the_commit(plugin, server, tmp_path):
    """Hermes stamps each message with time.time(), and the server stores the turn under
    that stamp: the cut is the same clock, read when the compression commits."""
    provider = _start(_new(plugin), tmp_path / "hermes-a")
    before = time.time()
    _compress(provider)
    after = time.time()
    cut = datetime.fromisoformat(_sent(provider, server, "recall", {"query": "x"})["self_since"])
    assert cut.utcoffset() == timedelta(0)
    assert before - 1e-6 <= cut.timestamp() <= after + 1e-6


@pytest.mark.parametrize(
    "switch",
    [
        {"new_session_id": _OTHER, "parent_session_id": _SID, "reason": "compression"},
        {"new_session_id": _OTHER, "parent_session_id": _SID, "reason": "branch"},
        {"new_session_id": _SID, "reset": True, "reason": "new_session"},
    ],
    ids=["compressed-into-a-new-id", "new-id", "reset"],
)
def test_a_new_id_or_a_reset_drops_the_cut(plugin, server, tmp_path, monkeypatch, switch):
    monkeypatch.setattr(_module(plugin), "_clock", lambda: _CUT)
    provider = _start(_new(plugin), tmp_path / "hermes-a")
    _compress(provider)
    provider.on_session_switch(**switch)
    assert _sent(provider, server, "recall", {"query": "x"}) == {
        "query": "x",
        "self_session": switch["new_session_id"],
    }


def test_a_restart_forgets_the_cut(plugin, server, tmp_path, monkeypatch):
    monkeypatch.setattr(_module(plugin), "_clock", lambda: _CUT)
    provider = _start(_new(plugin), tmp_path / "hermes-a")
    _compress(provider)
    _start(provider, tmp_path / "hermes-a", resumed=True)
    assert "self_since" not in _sent(provider, server, "recall", {"query": "x"})
    fresh = _start(_new(plugin), tmp_path / "hermes-a", resumed=True)
    assert "self_since" not in _sent(fresh, server, "recall", {"query": "x"})


def test_a_cut_that_cannot_be_taken_never_raises(plugin, server, tmp_path, monkeypatch):
    def broken() -> str:
        raise OSError("no clock")

    monkeypatch.setattr(_module(plugin), "_clock", broken)
    provider = _start(_new(plugin), tmp_path / "hermes-a")
    _compress(provider)
    assert _sent(provider, server, "recall", {"query": "x"}) == {"query": "x", "self_session": _SID}


# ---------------------------------------------------------------------------
# Isolation and failure
# ---------------------------------------------------------------------------


def test_two_instances_share_nothing(plugin, server, tmp_path):
    a = _start(_new(plugin), tmp_path / "hermes-a", session_id="sess-a")
    b = _start(_new(plugin), tmp_path / "hermes-b", session_id="sess-b")
    server.ingest_status = 503
    a.sync_turn("q", "a", messages=_conversation(1))
    assert Outbox(tmp_path / "hermes-a" / "synapse" / "outbox").pending() == 1
    assert Outbox(tmp_path / "hermes-b" / "synapse" / "outbox").pending() == 0
    server.ingest_status = 200
    b.sync_turn("q", "a", messages=_conversation(1))
    assert [p["session_id"] for p in server.posts()] == ["sess-a", "sess-b"]
    out = b.handle_tool_call("remember", {"content": "x"})
    assert json.loads(out)["session_id"] == "sess-b"


def test_no_hook_raises_with_the_server_down(plugin, user, tmp_path):
    _configure(user, _closed_port())
    provider = _new(plugin)
    assert provider.is_available()
    assert json.loads(provider.handle_tool_call("recall", {"query": "x"}))["error"]
    _start(provider, tmp_path / "hermes-a", cwd=str(tmp_path))
    assert provider.system_prompt_block() == ""
    assert provider.prefetch("x")
    provider.on_turn_start(1, "x", model=None)
    provider.sync_turn("q", "a", session_id=_SID, messages=_conversation(2))
    provider.sync_turn("q", "a", session_id=_SID, messages="garbage")  # type: ignore[arg-type]
    provider.sync_turn("q", "a", session_id=_SID, messages=[None, 3, {"role": "user"}])
    assert provider.on_pre_compress(_conversation(3)) == ""
    provider.on_session_end(_conversation(3))
    provider.on_session_switch("", reset=True)
    assert json.loads(provider.handle_tool_call("recall", {"query": "x"}))["error"]
    provider.on_memory_write("add", "memory", "x")
    provider.on_delegation("task", "result", child_session_id="c")
    assert provider.identity_signature() == {}
    assert provider.backup_paths() == []
    assert provider.get_config_schema() == []
    provider.save_config({}, str(tmp_path))
    provider.shutdown()
    assert Outbox(tmp_path / "hermes-a" / "synapse" / "outbox").pending() >= 1


def test_availability_reads_config_only(plugin, user):
    provider = _new(plugin)
    assert not provider.is_available()
    assert "synapse login" in provider.unavailable_reason()
    _configure(user, _closed_port())
    assert provider.is_available()


def test_every_thread_goes_through_hermes():
    source = (PLUGIN / "synapse_client" / "harnesses" / "hermes" / "provider.py").read_text(
        encoding="utf-8"
    )
    assert "Thread(" not in source and "spawn_context_thread(" in source
