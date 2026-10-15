"""The shared route helper (mcp_server.http_helpers.route) and the routes built on it.

The helper owns the gate, the JSON-object body, the threadpool boundary and the error
envelope, so these tests pin that contract once: the gate is consulted once and before the
body is read, a body that is not a JSON object is a 400 envelope (not a 500), a handler's
refusal and its unexpected failure map to the envelope, and a ``None`` result is the 404.
The second half drives the real body-taking routes with a bad body: none of them touches the
database before it answers, so no database is needed.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import pytest
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

from mcp_server.http_helpers import RouteError, route

_TOKEN = "route-test-token"
_H = {"Authorization": f"Bearer {_TOKEN}"}


class _Gate:
    """A bearer gate that counts how often a request asks it."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, request: Request) -> bool:
        self.calls += 1
        return request.headers.get("authorization", "") == f"Bearer {_TOKEN}"


def _client(gate: _Gate, register) -> TestClient:
    mcp = FastMCP("test-route")
    register(mcp, gate)
    return TestClient(mcp.http_app())


def _routes(mcp: Any, gate: Any) -> None:
    @mcp.custom_route("/echo", methods=["POST"])
    @route(gate, body=True)
    def echo(request: Request, body: dict[str, Any]) -> dict[str, Any]:
        return {"got": body}

    @mcp.custom_route("/refuse", methods=["GET"])
    @route(gate)
    def refuse(request: Request) -> dict[str, Any]:
        raise RouteError("no thanks", 403)

    @mcp.custom_route("/default-refusal", methods=["GET"])
    @route(gate)
    def default_refusal(request: Request) -> dict[str, Any]:
        raise RouteError("bad input")

    @mcp.custom_route("/boom", methods=["GET"])
    @route(gate)
    def boom(request: Request) -> dict[str, Any]:
        raise RuntimeError("x" * 500)

    @mcp.custom_route("/maybe", methods=["GET"])
    @route(gate, not_found="thing not found")
    def maybe(request: Request) -> dict[str, Any] | None:
        return {} if request.query_params.get("have") else None

    @mcp.custom_route("/ready-made", methods=["GET"])
    @route(gate)
    def ready_made(request: Request) -> JSONResponse:
        return JSONResponse({"status": "conflict"}, status_code=409)

    @mcp.custom_route("/where-sync", methods=["GET"])
    @route(gate)
    def where_sync(request: Request) -> dict[str, Any]:
        return {"thread": threading.get_ident()}

    @mcp.custom_route("/where-async", methods=["GET"])
    @route(gate)
    async def where_async(request: Request) -> dict[str, Any]:
        return {"thread": threading.get_ident()}

    @mcp.custom_route("/open", methods=["GET"])
    @route(None)
    def open_route(request: Request) -> dict[str, Any]:
        return {"ok": True}


def test_the_gate_is_asked_once_and_before_the_body_is_read():
    gate = _Gate()
    with _client(gate, _routes) as client:
        r = client.post("/echo", content=b"not json")
        assert (r.status_code, r.json()) == (401, {"status": "error", "detail": "unauthorized"})
        assert gate.calls == 1
        gate.calls = 0
        assert client.post("/echo", json={"a": 1}, headers=_H).json() == {"got": {"a": 1}}
        assert gate.calls == 1


def test_a_body_that_is_not_a_json_object_is_a_400_envelope():
    with _client(_Gate(), _routes) as client:
        for content in (b"not json", b"", b"[1, 2]", b'"text"', b"7", b"null"):
            r = client.post("/echo", content=content, headers=_H)
            assert r.status_code == 400, content
            assert r.json()["status"] == "error"
        assert client.post("/echo", content=b"nope", headers=_H).json()["detail"] == (
            "invalid JSON body"
        )
        assert client.post("/echo", content=b"[1]", headers=_H).json()["detail"] == (
            "body must be a JSON object"
        )


def test_a_route_error_answers_its_own_status_and_detail():
    with _client(_Gate(), _routes) as client:
        r = client.get("/refuse", headers=_H)
        assert (r.status_code, r.json()) == (403, {"status": "error", "detail": "no thanks"})
        r = client.get("/default-refusal", headers=_H)
        assert (r.status_code, r.json()["detail"]) == (400, "bad input")


def test_an_unexpected_failure_is_a_logged_500_with_a_bounded_detail(caplog):
    with _client(_Gate(), _routes) as client, caplog.at_level(logging.ERROR):
        r = client.get("/boom", headers=_H)
    assert r.status_code == 500
    assert r.json() == {"status": "error", "detail": "x" * 200}
    assert any("boom failed" in rec.getMessage() for rec in caplog.records)


def test_none_is_the_404_only_where_the_route_names_one():
    with _client(_Gate(), _routes) as client:
        r = client.get("/maybe", headers=_H)
        assert (r.status_code, r.json()["detail"]) == (404, "thing not found")
        # an empty result is a result
        r = client.get("/maybe?have=1", headers=_H)
        assert (r.status_code, r.json()) == (200, {})


def test_a_ready_made_response_passes_through():
    with _client(_Gate(), _routes) as client:
        r = client.get("/ready-made", headers=_H)
    assert (r.status_code, r.json()) == (409, {"status": "conflict"})


def test_plain_handlers_run_off_the_event_loop_and_async_ones_on_it():
    with _client(_Gate(), _routes) as client:
        sync_thread = client.get("/where-sync", headers=_H).json()["thread"]
        async_thread = client.get("/where-async", headers=_H).json()["thread"]
    assert sync_thread != async_thread


def test_a_route_without_a_gate_answers_anyone():
    gate = _Gate()
    with _client(gate, _routes) as client:
        assert client.get("/open").json() == {"ok": True}
    assert gate.calls == 0


# ---------------------------------------------------------------------------
# The real routes: a bad body is answered before any database is touched
# ---------------------------------------------------------------------------

_UNUSED_DSN = "postgresql://unused.invalid/none"

#: (method, path) of every route that takes a JSON object body.
_BODY_ROUTES = [
    ("POST", "/dash/api/flag"),
    ("POST", "/dash/api/proposals/skill:1/decision"),
    ("POST", "/timeline/events"),
    ("POST", "/remember/spool"),
    ("POST", "/surfaces/mint"),
    ("PUT", "/surfaces/oauth:someone"),
    ("POST", "/ingest"),
    ("POST", "/capture/dry-run"),
    ("POST", "/feedback"),
    ("POST", "/recall"),
]


def _everything(gate: _Gate) -> TestClient:
    from mcp_server import (
        dash,
        feedback_tools,
        ingest_route,
        recall_route,
        remember_routes,
        surface_routes,
        timeline_routes,
    )

    mcp = FastMCP("test-route-all")
    dash.register(mcp, _UNUSED_DSN, gate)
    timeline_routes.register(mcp, _UNUSED_DSN, gate, "")
    remember_routes.register(mcp, _UNUSED_DSN, gate, remember_fn=lambda **kw: None)
    surface_routes.register(mcp, _UNUSED_DSN, gate)
    ingest_route.register(mcp, lambda: _UNUSED_DSN, gate)
    feedback_tools.register(mcp, lambda: _UNUSED_DSN, gate)
    recall_route.register(mcp, lambda: None, gate, lambda r: None, lambda r: True)
    return TestClient(mcp.http_app())


@pytest.mark.parametrize(("method", "path"), _BODY_ROUTES)
def test_every_body_route_answers_a_bad_body_with_the_envelope(method, path):
    gate = _Gate()
    with _everything(gate) as client:
        unauth = client.request(method, path, content=b"[1]")
        assert (unauth.status_code, unauth.json()["detail"]) == (401, "unauthorized"), path
        for content, detail in (
            (b"[1]", "body must be a JSON object"),
            (b'"text"', "body must be a JSON object"),
            (b"not json", "invalid JSON body"),
        ):
            r = client.request(method, path, content=content, headers=_H)
            assert (r.status_code, r.json()) == (
                400,
                {"status": "error", "detail": detail},
            ), (path, content)


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("GET", "/dash/api/episode/abc", None),
        ("GET", "/dash/api/episode/abc/derived", None),
        ("GET", "/dash/api/graph/neighborhood", None),
        ("GET", "/dash/api/behavior/file", None),
        ("GET", "/dash/api/proposals/nope", None),
        ("POST", "/dash/api/proposals/nope/decision", {"action": "approve"}),
        ("DELETE", "/dash/api/episodes/abc", None),
        ("GET", "/dash/api/timeline?type=nonsense", None),
    ],
)
def test_a_dashboard_route_asks_the_gate_once_even_when_it_refuses_the_request(method, path, body):
    gate = _Gate()
    with _everything(gate) as client:
        r = client.request(method, path, json=body, headers=_H)
    assert r.status_code == 400, (path, r.text)
    assert gate.calls == 1, path
