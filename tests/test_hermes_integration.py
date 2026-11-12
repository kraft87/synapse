"""Synapse inside a real Hermes: the installed plugin folder, loaded by Hermes's own
loader and driven through Hermes's own MemoryManager.

Skipped unless ``SYNAPSE_HERMES_CHECKOUT`` names a Hermes checkout with its virtualenv
at ``.venv``:

    SYNAPSE_HERMES_CHECKOUT=~/workspace/hermes-agent python -m pytest tests/test_hermes_integration.py

The test installs with ``HermesHarness.install`` into a temp Hermes home (no ``hermes``
program on the harness's PATH, so the installer writes config.yaml itself), then runs a
driver script under the checkout's interpreter with a temp HERMES_HOME, SYNAPSE_HOME and
HOME. The driver calls ``plugins.memory.load_memory_provider("synapse")`` and puts the
provider through a CLI session: initialize, system prompt, turn start, prefetch, tool
schemas, a tool call, two ``sync_all`` turns with realistic rows, ``on_pre_compress``,
``on_session_end`` and shutdown. The server is a local fake; a ``sitecustomize`` on the
driver's PYTHONPATH (inherited by Hermes's plugin host) refuses any connection that is
not to the loopback address. It runs twice: plugins in-process, and with
``plugins.isolation: host``, where the provider's code runs in Hermes's plugin host and
the agent holds a proxy.

The checkout is only read: the driver runs with PYTHONDONTWRITEBYTECODE, and the
``hermes`` program is never started.
"""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from tests.helpers.client_core import use_core

use_core()
from synapse_client import __version__  # noqa: E402
from synapse_client.harnesses.hermes import install as hi  # noqa: E402
from synapse_client.harnesses.hermes import messages as hm  # noqa: E402
from synapse_client.harnesses.hermes import sessions as hs  # noqa: E402

from ingestion.messages_client import MessagesParser  # noqa: E402

_CHECKOUT = os.environ.get("SYNAPSE_HERMES_CHECKOUT", "")
pytestmark = pytest.mark.skipif(
    not _CHECKOUT, reason="set SYNAPSE_HERMES_CHECKOUT to a Hermes checkout to run"
)

_TOKEN = "tkHERMES-integration-5d2"
_SID = "20261112_140000_c0ffee"
_MODEL = "test-model-x"
_SECRET = "sk-testFAKEkey1234567890abcdefghij"
#: Small: a home Hermes had set up for a first launch would hold its toolchain.
_HOME_LIMIT = 5 * 1024 * 1024

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
        with self.server.lock:
            self.server.seen.append(
                {
                    "method": self.command,
                    "path": self.path,
                    "headers": dict(self.headers),
                    "body": body,
                }
            )

    def do_GET(self) -> None:
        self._record()
        if self.path.startswith("/context"):
            self._send(200, {"status": "ok", "text": "BOARD-TEXT", "trust": "full"})
        elif self.path.startswith("/preferences/top"):
            self._send(200, {"items": [{"pref": "Kyle likes small commits", "polarity": "like"}]})
        else:
            self._send(404, {})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"null")
        self._record(body)
        if self.path == "/ingest":
            self._send(200, {"status": "ok", "episodes": 1})
        elif self.path == "/mcp":
            text = json.dumps({"served": body["params"]["arguments"]})
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
        self.lock = threading.Lock()
        self.seen: list[dict[str, Any]] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def requests(self, method: str, path: str) -> list[dict[str, Any]]:
        return [r for r in self.seen if r["method"] == method and r["path"].startswith(path)]


@pytest.fixture()
def server() -> Iterator[_Server]:
    srv = _Server()
    thread = threading.Thread(target=srv.serve_forever, args=(0.02,), daemon=True)
    thread.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()


# ---------------------------------------------------------------------------
# What runs under Hermes's interpreter
# ---------------------------------------------------------------------------

#: Installed as sitecustomize in every Python process the driver starts, the plugin
#: host included: a connection anywhere but the loopback address is refused and noted.
_GUARD = textwrap.dedent(
    """
    import os
    import socket

    _LOCAL = ("127.0.0.1", "::1", "localhost")
    _connect = socket.socket.connect
    _resolve = socket.getaddrinfo


    def _note(where):
        path = os.environ.get("SYNAPSE_TEST_NETWORK_LOG")
        if path:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(repr(where) + "\\n")


    def connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6) and address[0] not in _LOCAL:
            _note(address)
            raise OSError("the integration test allows loopback connections only")
        return _connect(self, address)


    def getaddrinfo(host, *args, **kwargs):
        if host not in _LOCAL and host is not None:
            _note(host)
            raise socket.gaierror("the integration test resolves no names")
        return _resolve(host, *args, **kwargs)


    socket.socket.connect = connect
    socket.getaddrinfo = getaddrinfo
    """
)

_DRIVER = textwrap.dedent(
    """
    import json
    import os
    import sys

    from agent.memory_manager import MemoryManager
    from plugins.memory import load_memory_provider

    sid, model, secret, cwd = sys.argv[1:5]
    home = os.environ["HERMES_HOME"]
    out = {"warnings": [], "statuses": []}


    def turn(n, *, tool=False):
        rows = [{"role": "user", "content": f"question {n}", "timestamp": 1000.0 + n}]
        if tool:
            rows += [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call_{n}",
                            "type": "function",
                            "function": {"name": "terminal", "arguments": '{"command": "env"}'},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": f"call_{n}",
                    "name": "terminal",
                    "content": f"PATH=/usr/bin\\nOPENAI_API_KEY={secret}",
                },
            ]
        rows.append({"role": "assistant", "content": f"answer {n}"})
        return rows


    provider = load_memory_provider("synapse")
    out["loaded"] = provider is not None and provider.name == "synapse"
    out["available"] = bool(provider and provider.is_available())

    manager = MemoryManager()
    manager.add_provider(provider)
    manager.initialize_all(
        session_id=sid,
        platform="cli",
        hermes_home=home,
        agent_context="primary",
        warning_callback=out["warnings"].append,
        status_callback=out["statuses"].append,
        resumed=False,
        message_count=0,
        cwd=cwd,
        agent_identity="default",
        agent_workspace="hermes",
    )
    out["system_prompt"] = manager.build_system_prompt()
    manager.on_turn_start(
        1, "question 1", model=model, author_id=None, author_name=None,
        author_is_bot=False, previous_message="",
    )
    out["prefetch"] = manager.prefetch_all("question 1", session_id=sid)
    out["schemas"] = sorted(s["name"] for s in manager.get_all_tool_schemas())
    out["tool"] = manager.handle_tool_call("recall", {"query": "board cache"})

    messages = [{"role": "system", "content": "You are Hermes."}] + turn(1, tool=True)
    manager.sync_all("question 1", "answer 1", session_id=sid, messages=messages)
    out["flushed_1"] = manager.flush_pending(timeout=60)

    messages += turn(2)
    manager.sync_all("question 2", "answer 2", session_id=sid, messages=messages)
    out["flushed_2"] = manager.flush_pending(timeout=60)

    messages += [
        {
            "role": "user",
            "content": "[Your active task list was preserved across context compression]\\n- ship",
            "_todo_snapshot_synthetic": True,
        }
    ] + turn(3)
    out["pre_compress"] = manager.on_pre_compress(messages)

    messages = [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": "[CONTEXT COMPACTION] earlier turns", "_compressed_summary": True},
    ] + messages[-2:] + turn(4)
    manager.on_session_end(messages)
    manager.shutdown_all()

    try:
        from hermes_cli.plugins import get_plugin_manager

        host = get_plugin_manager()._running_plugin_host()
        out["host_pid"] = host.pid if host is not None else None
        if host is not None:
            host.shutdown()
    except Exception as exc:
        out["host_error"] = repr(exc)

    from agent import context_compressor, conversation_compression_scaffolding, prompt_builder

    out["markers"] = {
        "flags": list(conversation_compression_scaffolding._SYNTHETIC_USER_FLAGS),
        "prefixes": list(conversation_compression_scaffolding._SYNTHETIC_USER_PREFIXES)
        + list(context_compressor._SYNTHETIC_USER_ROW_PREFIXES)
        + ["[" + o.replace("/?", "") for o in prompt_builder.CONTROL_FRAME_OPENERS]
        + ["[/" + o.replace("/?", "") for o in prompt_builder.CONTROL_FRAME_OPENERS if o.startswith("/?")],
        "summary_key": context_compressor.COMPRESSED_SUMMARY_METADATA_KEY,
        "steer_kind": prompt_builder.STEER_DISPLAY_KIND,
    }

    from agent import turn_failure_copy as failures

    notices = [failures.FAILED_TURN_NOTICE, failures.PARTIAL_FAILED_TURN_NOTICE,
               failures.AUTOMATION_INTERRUPTED_NOTICE]
    out["markers"].update(
        failed_kind=failures.FAILED_TURN_DISPLAY_KIND,
        failed_notices=notices,
        untyped=[failures.untyped_failed_turn_display_kind("assistant", n) for n in notices],
    )

    # The gateway's names for a local session, as the checkout makes them.
    try:
        from gateway.config import Platform
        from gateway.session import SessionSource, build_session_key
        from gateway.session_policy import SURFACES, STORED_AS_LOCAL
        from hermes_state_keys import local_identity
        from hermes_state_local import API_BINDING_PREFIX, POLICY_PREFIX

        owner = "uid:%d" % os.getuid()
        chat = local_identity(os.environ["HERMES_HOME"], owner, "req-1")
        local = SessionSource(platform=Platform.LOCAL, chat_id=chat, chat_type="dm", user_id=owner)
        out["gateway"] = {
            "chat": chat,
            "route": build_session_key(local),
            "policy_prefix": POLICY_PREFIX,
            "api_prefix": API_BINDING_PREFIX,
            "surfaces": sorted(SURFACES),
            "stored_as_local": sorted(STORED_AS_LOCAL),
        }
    except Exception as exc:
        out["gateway_error"] = repr(exc)
    print("RESULT " + json.dumps(out))
    """
)


def _checkout() -> tuple[Path, Path]:
    checkout = Path(os.path.expanduser(_CHECKOUT)).resolve()
    python = checkout / ".venv" / "bin" / "python"
    if not (checkout / "plugins" / "memory" / "__init__.py").is_file() or not python.exists():
        pytest.fail(f"{checkout} is not a Hermes checkout with a .venv")
    return checkout, python


def _size(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink())


def _run(tmp_path: Path, server: _Server, isolation: str) -> dict[str, Any]:
    checkout, python = _checkout()
    user = tmp_path / "home"
    hermes_home = tmp_path / "hermes"
    synapse_home = user / ".synapse"
    synapse_home.mkdir(parents=True)
    (synapse_home / "config.json").write_text(
        json.dumps({"url": server.url, "token": _TOKEN}), encoding="utf-8"
    )
    hermes_home.mkdir()
    report = hi.HermesHarness(which=lambda name: None, env={}).install(
        server.url, hermes_home=hermes_home
    )
    assert report.endswith(f"selected in {hermes_home / 'config.yaml'}"), report
    if isolation != "in_process":
        with (hermes_home / "config.yaml").open("a", encoding="utf-8") as fh:
            fh.write(f"plugins:\n  isolation: {isolation}\n")

    guard = tmp_path / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(_GUARD, encoding="utf-8")
    driver = tmp_path / "driver.py"
    driver.write_text(_DRIVER, encoding="utf-8")
    network_log = tmp_path / "network.log"
    project = tmp_path / "work" / "atlas"
    project.mkdir(parents=True)
    env = {
        "HOME": str(user),
        "HERMES_HOME": str(hermes_home),
        "SYNAPSE_HOME": str(synapse_home),
        "PATH": os.defpath,
        "PYTHONPATH": os.pathsep.join((str(guard), str(checkout))),
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(tmp_path),
        "LANG": "C.UTF-8",
        "SYNAPSE_TEST_NETWORK_LOG": str(network_log),
    }
    done = subprocess.run(
        [str(python), str(driver), _SID, _MODEL, _SECRET, str(project)],
        cwd=str(checkout),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    lines = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")]
    assert done.returncode == 0 and lines, f"driver failed:\n{done.stdout}\n{done.stderr}"
    out: dict[str, Any] = json.loads(lines[-1][len("RESULT ") :])
    out["stderr"] = done.stderr
    out["network"] = network_log.read_text() if network_log.exists() else ""
    out["home_size"] = _size(hermes_home)
    out["project"] = str(project)
    return out


def _stored(posts: list[dict[str, Any]]) -> list[Any]:
    """The episodes the server keeps from ``posts``: each parsed by the route's own
    parser, and one per span_id, which is what the route dedups on."""
    episodes: dict[str, Any] = {}
    for body in posts:
        for episode in MessagesParser().parse_records(
            body["records"],
            body["source"],
            body["project"],
            session_id=body["session_id"],
            client=body["client"],
            cwd=body["cwd"],
            model=body["model"],
        ):
            episodes.setdefault(str(episode.span_id), episode)
    return list(episodes.values())


@pytest.mark.parametrize("isolation", ["in_process", "host"])
def test_synapse_runs_inside_hermes(tmp_path: Path, server: _Server, isolation: str) -> None:
    out = _run(tmp_path, server, isolation)
    print(
        f"{isolation}: Hermes home {out['home_size']} bytes, plugin host pid {out.get('host_pid')}"
    )

    # Nothing left the machine, and the home stayed a home.
    assert out["network"] == ""
    assert out["home_size"] < _HOME_LIMIT
    if isolation == "host":
        assert out.get("host_pid"), "the provider did not run in Hermes's plugin host"
    else:
        assert out.get("host_pid") is None

    # Loaded, initialized, and its session-start block in the system prompt.
    assert out["loaded"] and out["available"]
    assert out["warnings"] == []
    assert "BOARD-TEXT" in out["system_prompt"]
    assert "Kyle likes small commits" in out["system_prompt"]
    assert out["prefetch"]
    assert {"recall", "remember"} <= set(out["schemas"])
    for request in server.requests("GET", "/"):
        assert request["headers"]["User-Agent"] == f"synapse-client/{__version__} hermes"
        assert request["headers"]["Authorization"] == f"Bearer {_TOKEN}"
    assert server.requests("GET", "/context") and server.requests("GET", "/preferences/top")

    # The tool call reached /mcp with its arguments and came back as Hermes's JSON.
    (call,) = server.requests("POST", "/mcp")
    assert call["body"]["params"]["name"] == "recall"
    assert call["body"]["params"]["arguments"]["query"] == "board cache"
    assert json.loads(out["tool"])["served"]["query"] == "board cache"

    # Every batch names the session, model and project; each turn is stored once, in
    # order (a batch leads with the turn before its first new one, as context).
    posts = [r["body"] for r in server.requests("POST", "/ingest")]
    assert posts
    for body in posts:
        assert body["format"] == "messages"
        assert body["client"] == "hermes"
        assert body["session_id"] == _SID
        assert body["model"] == _MODEL
        assert body["cwd"] == out["project"]
        assert body["project"] == "atlas"
        assert body["source"] == "hermes-sync"
    stored = _stored(posts)
    assert [e.human_turn for e in stored] == [f"question {n}" for n in (1, 2, 3, 4)]
    assert [e.assistant_turn for e in stored] == [f"answer {n}" for n in (1, 2, 3, 4)]
    assert {e.model for e in stored} == {_MODEL}
    assert {e.project for e in stored} == {"atlas"}
    assert "[tool:terminal]" in stored[0].content
    assert "[result] PATH=/usr/bin" in stored[0].content

    # Hermes scrubs what it hands a memory provider, so the key never left; the system
    # prompt and the compaction summary are not sent, and the task-list row goes marked
    # synthetic, which the server skips.
    everything = json.dumps(posts)
    assert _SECRET not in everything
    assert "You are Hermes." not in everything
    assert "CONTEXT COMPACTION" not in everything
    records = [rec for body in posts for rec in body["records"]]
    flagged = [rec for rec in records if "task list was preserved" in rec["content"]]
    assert flagged and all(rec.get("synthetic") is True for rec in flagged)
    assert not any("task list" in e.content for e in stored)

    # The rows this Hermes writes into the user role for itself are the rows the
    # normaliser drops: every flag, prefix and the summary key, as the checkout has them.
    markers = out["markers"]
    assert markers["summary_key"] == hm._SUMMARY_KEY
    for flag in markers["flags"]:
        assert hm._synthetic({flag: True}, "typed text"), flag
    for prefix in markers["prefixes"]:
        assert hm._synthetic({}, prefix + " body"), prefix
    assert hm._synthetic({"display_kind": markers["steer_kind"]}, "steer text")

    # This Hermes's failed-turn boundary, typed or as bare text, is never a reply.
    assert markers["failed_kind"] == hm._FAILED_TURN_KIND
    assert sorted(markers["failed_notices"]) == sorted(hm._FAILED_TURN_NOTICES)
    assert markers["untyped"] == [hm._FAILED_TURN_KIND] * 3
    for notice in markers["failed_notices"]:
        assert hm.to_records([{"role": "assistant", "content": notice}]) == []
        typed = {"role": "assistant", "content": notice, "display_kind": markers["failed_kind"]}
        assert hm.to_records([typed]) == []

    # The gateway's local-session names are the ones sessions.py recognises.
    assert "gateway_error" not in out, out.get("gateway_error")
    names = out["gateway"]
    assert hs._LOCAL_CHAT.fullmatch(names["chat"])
    route = hs._LOCAL_ROUTE.fullmatch(names["route"])
    assert route is not None and route.group(1) == names["chat"], names["route"]
    assert (names["policy_prefix"], names["api_prefix"]) == (hs._POLICY_KEY, hs._API_KEY)
    assert hs.OWNER_SOURCES <= set(names["surfaces"])
    assert set(names["stored_as_local"]) - {"tool"} <= hs.OWNER_SOURCES
