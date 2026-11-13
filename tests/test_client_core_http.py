"""The core's transport and API client (plugin/synapse_client/http.py, api.py).

Real servers on 127.0.0.1 ephemeral ports, nothing else: a stdlib HTTP server for the
JSON helpers, redirects and the SSE edge cases, and a real FastMCP app served by uvicorn
(both reply framings) for ``mcp_call``. No test reaches a configured Synapse server.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from tests.helpers.client_core import use_core

use_core()
from synapse_client import http as core_http  # noqa: E402
from synapse_client.api import Client  # noqa: E402
from synapse_client.http import ToolError, Transport  # noqa: E402
from synapse_client.settings import Settings  # noqa: E402

_TOKEN = "tkHTTP7c2Vb9Qx4Ln"
_UA = "synapse-test/1.0"
_UNREADABLE = "http://[bad"  # urlsplit raises ValueError on it


# ---------------------------------------------------------------------------
# A stdlib server that records every request and answers from a route table
# ---------------------------------------------------------------------------

Route = Callable[["_Handler"], None]


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def log_message(self, format: str, *args: Any) -> None:  # silence
        pass

    def _handle(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        split = urllib.parse.urlsplit(self.path)
        self.server.seen.append(
            {
                "method": self.command,
                "path": split.path,
                "query": dict(urllib.parse.parse_qsl(split.query)),
                "headers": {k.lower(): v for k, v in self.headers.items()},
                "body": body,
            }
        )
        route = self.server.routes.get(split.path)
        if route is None:
            self.reply(404, {"detail": "no such route"})
        else:
            route(self)

    do_GET = do_POST = do_PUT = do_DELETE = _handle

    def reply(self, code: int, payload: Any = None, *, raw: bytes | None = None) -> None:
        data = raw if raw is not None else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def redirect(self, location: str, code: int = 302) -> None:
        self.send_response(code)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.seen: list[dict[str, Any]] = []
        self.routes: dict[str, Route] = {}

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


@pytest.fixture()
def serve() -> Iterator[Callable[[], _Server]]:
    servers: list[_Server] = []

    def start() -> _Server:
        srv = _Server()
        threading.Thread(target=srv.serve_forever, args=(0.02,), daemon=True).start()
        servers.append(srv)
        return srv

    yield start
    for srv in servers:
        srv.shutdown()
        srv.server_close()


@pytest.fixture()
def srv(serve) -> _Server:
    return serve()


def _echo(handler: _Handler) -> None:
    handler.reply(200, {"status": "ok", "path": handler.path})


def _closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# ---------------------------------------------------------------------------
# Transport: the JSON helpers
# ---------------------------------------------------------------------------


def test_get_json_sends_the_user_agent_the_bearer_and_the_params(srv):
    srv.routes["/context"] = _echo
    t = Transport(srv.url + "/", _TOKEN, _UA)
    assert t.get_json("/context", {"project": "a b", "x": 1})["status"] == "ok"
    req = srv.seen[-1]
    assert req["method"] == "GET" and req["query"] == {"project": "a b", "x": "1"}
    assert req["headers"]["user-agent"] == _UA
    assert req["headers"]["authorization"] == f"Bearer {_TOKEN}"
    assert "content-type" not in req["headers"]


def test_post_json_sends_a_json_body(srv):
    srv.routes["/ingest"] = _echo
    Transport(srv.url, _TOKEN, _UA).post_json("/ingest", {"records": [1], "s": "é"})
    req = srv.seen[-1]
    assert req["method"] == "POST" and json.loads(req["body"]) == {"records": [1], "s": "é"}
    assert req["headers"]["content-type"] == "application/json"


@pytest.mark.parametrize("method", ["PUT", "DELETE", "GET"])
def test_request_json_reaches_any_method_without_a_body(srv, method):
    srv.routes["/private-sessions/s1"] = _echo
    Transport(srv.url, _TOKEN, _UA).request_json(method, "/private-sessions/s1")
    req = srv.seen[-1]
    assert req["method"] == method and req["body"] == b""
    assert "content-type" not in req["headers"]


def test_an_empty_body_reads_as_an_empty_object(srv):
    srv.routes["/empty"] = lambda h: h.reply(200, raw=b"")
    assert Transport(srv.url, "", _UA).get_json("/empty") == {}


def test_no_token_sends_no_authorization(srv):
    srv.routes["/context"] = _echo
    Transport(srv.url, "", _UA).get_json("/context")
    assert "authorization" not in srv.seen[-1]["headers"]


@pytest.mark.parametrize("code", [401, 403, 404, 500])
def test_http_errors_raise_with_their_status(srv, code):
    srv.routes["/x"] = lambda h: h.reply(code, {"detail": "no"})
    with pytest.raises(urllib.error.HTTPError) as err:
        Transport(srv.url, _TOKEN, _UA).get_json("/x")
    assert err.value.code == code


def test_an_unreachable_server_raises(srv):
    with pytest.raises(urllib.error.URLError):
        Transport(f"http://127.0.0.1:{_closed_port()}", _TOKEN, _UA).get_json("/x", timeout=2)


def test_a_relative_path_with_no_server_url_raises():
    with pytest.raises(ValueError, match="no Synapse server URL"):
        Transport("", _TOKEN, _UA).get_json("/context")


def test_invalid_json_raises_value_error(srv):
    srv.routes["/bad"] = lambda h: h.reply(200, raw=b"<html>")
    with pytest.raises(ValueError):
        Transport(srv.url, _TOKEN, _UA).get_json("/bad")


def test_open_returns_the_raw_reply_and_sends_the_body_as_given(srv):
    srv.routes["/ingest"] = lambda h: h.reply(200, raw=b"not json, 202 accepted")
    t = Transport(srv.url + "/", _TOKEN, _UA, 7)
    headers = {"Content-Type": "application/json"}
    with t.open("POST", "/ingest", b'{"records": []}', headers) as resp:
        assert resp.read() == b"not json, 202 accepted"
    req = srv.seen[-1]
    assert (req["method"], req["path"], req["body"]) == ("POST", "/ingest", b'{"records": []}')
    assert req["headers"]["user-agent"] == _UA
    assert req["headers"]["authorization"] == f"Bearer {_TOKEN}"
    assert req["headers"]["content-type"] == "application/json"


def test_open_keeps_the_bearer_rule_and_raises_on_http_errors(serve):
    home, other = serve(), serve()
    other.routes["/x"] = lambda h: h.reply(500, raw=b"boom")
    with pytest.raises(urllib.error.HTTPError) as err:
        Transport(home.url, _TOKEN, _UA).open("GET", other.url + "/x")
    assert err.value.code == 500
    assert "authorization" not in other.seen[-1]["headers"]
    assert other.seen[-1]["body"] == b""


def test_the_repr_leaves_the_token_out():
    assert _TOKEN not in repr(Transport("https://s.example", _TOKEN, _UA))


def test_a_server_url_that_cannot_be_read_fails_the_request_not_the_transport(serve):
    # A hook builds its Transport before it knows whether it will make a request (the
    # private toggle's --session-end makes none): a URL urlsplit rejects must not stop it.
    other = serve()
    other.routes["/x"] = _echo
    t = Transport(_UNREADABLE, _TOKEN, _UA)
    with pytest.raises(ValueError):
        t.get_json("/context")
    t.post_json(other.url + "/x", {})
    assert "authorization" not in other.seen[-1]["headers"]


def test_an_unreadable_token_origin_names_no_server_and_leaves_the_others(serve):
    home, other = serve(), serve()
    other.routes["/x"] = _echo
    t = Transport(home.url, _TOKEN, _UA, token_origins=[_UNREADABLE, other.url + "/x"])
    t.post_json(other.url + "/x", {})
    assert other.seen[-1]["headers"]["authorization"] == f"Bearer {_TOKEN}"


# ---------------------------------------------------------------------------
# Transport: the bearer stays with its server
# ---------------------------------------------------------------------------


def test_an_absolute_url_elsewhere_gets_no_bearer_unless_named(serve):
    home, other = serve(), serve()
    other.routes["/ingest"] = _echo
    Transport(home.url, _TOKEN, _UA).post_json(other.url + "/ingest", {})
    assert "authorization" not in other.seen[-1]["headers"]
    Transport(home.url, _TOKEN, _UA, token_origins=[other.url + "/ingest"]).post_json(
        other.url + "/ingest", {}
    )
    assert other.seen[-1]["headers"]["authorization"] == f"Bearer {_TOKEN}"


def test_a_same_origin_redirect_keeps_the_bearer(srv):
    srv.routes["/old"] = lambda h: h.redirect("/context")
    srv.routes["/context"] = _echo
    Transport(srv.url, _TOKEN, _UA).get_json("/old")
    assert [r["path"] for r in srv.seen] == ["/old", "/context"]
    assert srv.seen[-1]["headers"]["authorization"] == f"Bearer {_TOKEN}"


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_a_cross_origin_redirect_drops_the_bearer(serve, code):
    home, away = serve(), serve()
    home.routes["/context"] = lambda h: h.redirect(away.url + "/context", code)
    away.routes["/context"] = _echo
    assert Transport(home.url, _TOKEN, _UA).get_json("/context")["status"] == "ok"
    assert home.seen[-1]["headers"]["authorization"] == f"Bearer {_TOKEN}"
    assert "authorization" not in away.seen[-1]["headers"]
    assert away.seen[-1]["headers"]["user-agent"] == _UA


def test_stock_urllib_forwards_the_bearer_across_origins(serve):
    """Why the core has its own redirect handler: plain urllib.request.urlopen sends the
    Authorization header to wherever the server redirects. The hook scripts that build
    their own requests open them through ``core_http.urlopen`` instead."""
    home, away = serve(), serve()
    home.routes["/context"] = lambda h: h.redirect(away.url + "/context")
    away.routes["/context"] = _echo
    req = urllib.request.Request(
        home.url + "/context", headers={"Authorization": f"Bearer {_TOKEN}"}
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        resp.read()
    assert away.seen[-1]["headers"].get("authorization") == f"Bearer {_TOKEN}"


def test_the_core_urlopen_drops_the_bearer_where_stock_urllib_forwards_it(serve):
    home, away = serve(), serve()
    home.routes["/context"] = lambda h: h.redirect(away.url + "/context")
    home.routes["/old"] = lambda h: h.redirect("/context")
    away.routes["/context"] = _echo
    for path in ("/context", "/old"):
        req = urllib.request.Request(home.url + path, headers={"Authorization": f"Bearer {_TOKEN}"})
        with core_http.urlopen(req, timeout=5) as resp:
            resp.read()
    assert "authorization" not in away.seen[-1]["headers"]
    # the same-origin hop still carries it: /old -> /context on home
    assert home.seen[-1]["path"] == "/context"
    assert home.seen[-1]["headers"]["authorization"] == f"Bearer {_TOKEN}"


@pytest.mark.parametrize(
    ("old", "new", "keeps"),
    [
        ("http://s.example/a", "http://s.example/b", True),
        ("http://s.example/a", "http://s.example:80/b", True),
        ("https://s.example:443/a", "https://s.example/b", True),
        ("http://s.example:8080/a", "http://s.example:8080/b", True),
        ("http://S.Example/a", "http://s.example/b", True),
        # the upgrade to TLS on the default ports, the one change of scheme allowed
        ("http://s.example/a", "https://s.example/b", True),
        ("http://s.example:80/a", "https://s.example:443/b", True),
        # any other change of scheme, host or port drops it
        ("http://s.example:8080/a", "https://s.example:8443/b", False),
        ("http://s.example:8080/a", "https://s.example/b", False),
        ("http://s.example/a", "https://s.example:8443/b", False),
        ("https://s.example/a", "http://s.example/b", False),
        ("http://s.example/a", "http://other.example/b", False),
        ("http://s.example/a", "http://s.example.evil.example/b", False),
        ("http://s.example/a", "https://other.example/b", False),
        ("http://s.example:8080/a", "http://s.example:8081/b", False),
        ("http://s.example/a", "http://s.example:8080/b", False),
        ("http://s.example/a", "ftp://s.example/b", False),
        ("http://s.example/a", "http://s.example:bad/b", False),
    ],
)
def test_which_redirects_keep_the_bearer(old, new, keeps):
    assert core_http._keeps_auth(old, new) is keeps
    req = urllib.request.Request(old, headers={"Authorization": f"Bearer {_TOKEN}"})
    out = core_http._SameOriginAuth().redirect_request(req, None, 302, "Found", {}, new)
    assert out is not None
    assert (out.get_header("Authorization") == f"Bearer {_TOKEN}") is keeps
    assert out.get_header("Authorization") in (None, f"Bearer {_TOKEN}")


# ---------------------------------------------------------------------------
# Client: one thin method per route
# ---------------------------------------------------------------------------


def _client(url: str, *, ingest: str = "", token: str = _TOKEN) -> Client:
    env = {"SYNAPSE_URL": url, "SYNAPSE_INGEST_TOKEN": token}
    if ingest:
        env["SYNAPSE_INGEST_URL"] = ingest
    return Client(Settings.load(env=env, home="/nonexistent-synapse-home"), _UA)


def test_context_sends_the_project_only_when_there_is_one(srv):
    srv.routes["/context"] = lambda h: h.reply(200, {"status": "ok", "text": "board"})
    client = _client(srv.url)
    assert client.context("synapse")["text"] == "board"
    assert srv.seen[-1]["query"] == {"project": "synapse"}
    client.context()
    assert srv.seen[-1]["query"] == {}
    assert srv.seen[-1]["headers"]["authorization"] == f"Bearer {_TOKEN}"


def test_context_raises_on_401(srv):
    srv.routes["/context"] = lambda h: h.reply(401, {"detail": "no"})
    with pytest.raises(urllib.error.HTTPError):
        _client(srv.url).context()


def test_preferences_returns_the_items(srv):
    items = [{"pref": "a", "polarity": "like"}, "junk", {"pref": "b"}]
    srv.routes["/preferences/top"] = lambda h: h.reply(200, {"status": "ok", "items": items})
    assert _client(srv.url).preferences(limit=3) == [items[0], items[2]]
    assert srv.seen[-1]["query"] == {"limit": "3"}
    srv.routes["/preferences/top"] = lambda h: h.reply(200, {"status": "ok"})
    assert _client(srv.url).preferences() == []
    srv.routes["/preferences/top"] = lambda h: h.reply(200, [1])
    with pytest.raises(ValueError):
        _client(srv.url).preferences()


def test_ingest_posts_to_the_ingest_url(serve):
    home, legacy = serve(), serve()
    home.routes["/ingest"] = lambda h: h.reply(200, {"status": "ok", "ingested": 2})
    assert _client(home.url).ingest({"format": "codex", "records": []})["ingested"] == 2
    assert home.seen[-1]["path"] == "/ingest"
    # A legacy SYNAPSE_INGEST_URL at its own origin: the token was resolved for it too.
    legacy.routes["/in"] = _echo
    _client(home.url, ingest=legacy.url + "/in").ingest({"records": []})
    assert legacy.seen[-1]["headers"]["authorization"] == f"Bearer {_TOKEN}"


@pytest.mark.parametrize(
    ("call", "method"),
    [("private_get", "GET"), ("private_put", "PUT"), ("private_delete", "DELETE")],
)
def test_private_routes_quote_the_session_id(srv, call, method):
    srv.routes["/private-sessions/a/b c"] = _echo  # the server's unquoted view
    with pytest.raises(urllib.error.HTTPError):
        getattr(_client(srv.url), call)("a/b c")  # quoted: no such route, 404
    assert srv.seen[-1]["method"] == method
    assert srv.seen[-1]["path"] == "/private-sessions/a%2Fb%20c"
    srv.routes["/private-sessions/s-1"] = lambda h: h.reply(200, {"private": True})
    assert getattr(_client(srv.url), call)("s-1") == {"private": True}


# ---------------------------------------------------------------------------
# tool_text and the SSE reader, on their own
# ---------------------------------------------------------------------------


def test_tool_text_joins_text_items_and_falls_back_to_structured_content():
    reply = {
        "id": 1,
        "result": {
            "content": [
                {"type": "text", "text": "a"},
                {"type": "image", "data": "x"},
                {"type": "text", "text": "b"},
            ]
        },
    }
    assert core_http.tool_text("t", reply) == "a\nb"
    reply = {"id": 1, "result": {"content": [], "structuredContent": {"k": 1}}}
    assert json.loads(core_http.tool_text("t", reply)) == {"k": 1}


def test_tool_text_raises_for_tool_and_protocol_errors():
    with pytest.raises(ToolError) as err:
        core_http.tool_text(
            "t", {"result": {"isError": True, "content": [{"type": "text", "text": "bad"}]}}
        )
    assert (str(err.value), err.value.tool, err.value.code) == ("bad", "t", None)
    with pytest.raises(ToolError) as err:
        core_http.tool_text("t", {"error": {"code": -32602, "message": "Invalid params"}})
    assert (str(err.value), err.value.code) == ("Invalid params", -32602)
    with pytest.raises(ValueError):
        core_http.tool_text("t", {"id": 1})


@pytest.mark.parametrize(
    ("raw", "events"),
    [
        (b"event: message\r\ndata: {}\r\n\r\n", ["{}"]),
        (b"data: a\n\ndata: b\n\n", ["a", "b"]),
        (b"data: a\r\rdata: b\r\r", ["a", "b"]),
        (b": ping\n\ndata: x\ndata: y\n\n", ["x\ny"]),
        (b"data:nospace\n", ["nospace"]),
        (b"id: 4\nretry: 10\n\n", []),
    ],
)
def test_sse_events(raw, events):
    lines = raw.splitlines(keepends=True)
    assert list(core_http.sse_events(lines)) == events


# ---------------------------------------------------------------------------
# mcp_call against a stdlib server: framing edge cases
# ---------------------------------------------------------------------------


def _sse(handler: _Handler, chunks: list[bytes], hold: float = 0.0) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
    handler.end_headers()
    for chunk in chunks:
        handler.wfile.write(chunk)
        handler.wfile.flush()
    time.sleep(hold)


def _msg(payload: Any) -> bytes:
    return b"event: message\r\ndata: " + json.dumps(payload).encode() + b"\r\n\r\n"


_OK = {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "hi"}]}}


def test_mcp_call_skips_notifications_and_stops_at_the_answer(srv):
    note = {"jsonrpc": "2.0", "method": "notifications/message", "params": {"data": "x"}}
    srv.routes["/mcp"] = lambda h: _sse(h, [b": ping\r\n\r\n", _msg(note), _msg(_OK)], hold=3)
    started = time.monotonic()
    assert Transport(srv.url, _TOKEN, _UA).mcp_call("recall", {"query": "q"}, timeout=10) == "hi"
    assert time.monotonic() - started < 2.5  # did not wait for the stream to close


def test_mcp_call_sends_one_tools_call_with_the_headers_the_server_needs(srv):
    srv.routes["/mcp"] = lambda h: h.reply(200, _OK)
    Transport(srv.url, _TOKEN, _UA).mcp_call("fetch", {"ids": ["e:1"]})
    assert len(srv.seen) == 1
    req = srv.seen[0]
    assert json.loads(req["body"]) == {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "fetch", "arguments": {"ids": ["e:1"]}},
    }
    headers = req["headers"]
    assert headers["accept"] == "application/json, text/event-stream"
    assert headers["mcp-protocol-version"] == core_http.PROTOCOL_VERSION
    assert headers["authorization"] == f"Bearer {_TOKEN}"


def test_mcp_call_reads_a_batch_reply(srv):
    srv.routes["/mcp"] = lambda h: h.reply(200, [{"jsonrpc": "2.0", "id": 9}, _OK])
    assert Transport(srv.url, _TOKEN, _UA).mcp_call("recall", {}) == "hi"


def test_mcp_call_raises_when_nothing_answers_the_call(srv):
    srv.routes["/mcp"] = lambda h: _sse(h, [_msg({"jsonrpc": "2.0", "id": 2, "result": {}})])
    with pytest.raises(ValueError, match="no answer"):
        Transport(srv.url, _TOKEN, _UA).mcp_call("recall", {})


def test_mcp_call_raises_http_errors(srv):
    srv.routes["/mcp"] = lambda h: h.reply(401, {"error": "invalid_token"})
    with pytest.raises(urllib.error.HTTPError) as err:
        Transport(srv.url, "wrong", _UA).mcp_call("recall", {})
    assert err.value.code == 401


# ---------------------------------------------------------------------------
# mcp_call against a real FastMCP app, both framings
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", params=[False, True], ids=["sse", "json"])
def mcp_url(request) -> Iterator[str]:
    uvicorn = pytest.importorskip("uvicorn")
    fastmcp = pytest.importorskip("fastmcp")
    from fastmcp.exceptions import ToolError as ServerToolError
    from fastmcp.server.dependencies import get_http_headers

    app = fastmcp.FastMCP("client-core-test")

    @app.tool
    def echo(text: str, times: int = 1) -> str:
        return text * times

    @app.tool
    def refuse(why: str) -> str:
        raise ServerToolError(f"refused: {why}")

    @app.tool
    def crash() -> str:
        raise RuntimeError("internal detail")

    @app.tool
    def structured() -> dict:
        return {"a": 1}

    @app.tool
    def headers() -> dict:
        seen = get_http_headers(include_all=True)
        keep = ("authorization", "user-agent", "mcp-protocol-version", "mcp-session-id")
        return {k: v for k, v in seen.items() if k in keep}

    asgi = app.http_app(path="/mcp", stateless_http=True, json_response=request.param)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(asgi, log_level="warning", lifespan="on", ws="none"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started
    yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    server.should_exit = True
    thread.join(timeout=5)
    sock.close()


def test_a_cold_tools_call_works_without_initialize(mcp_url):
    t = Transport(mcp_url, _TOKEN, _UA)
    assert t.mcp_call("echo", {"text": "ab", "times": 2}) == "abab"


def test_the_bearer_and_protocol_version_reach_the_tool(mcp_url):
    seen = json.loads(Transport(mcp_url, _TOKEN, _UA).mcp_call("headers", {}))
    assert seen["authorization"] == f"Bearer {_TOKEN}"
    assert seen["user-agent"] == _UA
    assert seen["mcp-protocol-version"] == core_http.PROTOCOL_VERSION
    assert "mcp-session-id" not in seen


def test_structured_content_comes_back_as_json_text(mcp_url):
    assert json.loads(Transport(mcp_url, "", _UA).mcp_call("structured", {})) == {"a": 1}


@pytest.mark.parametrize(
    ("tool", "args", "message"),
    [
        ("refuse", {"why": "x"}, "refused: x"),
        ("crash", {}, "internal detail"),
        ("missing", {}, "Unknown tool"),
        ("echo", {"times": 2}, "text"),
    ],
)
def test_a_tool_that_raises_is_a_tool_error(mcp_url, tool, args, message):
    with pytest.raises(ToolError) as err:
        Transport(mcp_url, "", _UA).mcp_call(tool, args)
    assert message in str(err.value)
    assert err.value.tool == tool and err.value.code is None


def test_client_call_tool_goes_through_the_same_call(mcp_url):
    assert _client(mcp_url).call_tool("echo", {"text": "z"}) == "z"
