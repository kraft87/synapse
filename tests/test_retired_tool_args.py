"""The recall tool no longer takes ``group_id``, and a client that still sends it is served.

Older clients, and tool schemas a client cached, pass ``group_id`` on every recall. FastMCP
rejects an argument a tool does not declare, so ``RetiredToolArgs`` drops it before the call
is validated: the tool list stays clean and the call is answered as if it were not sent.
The same goes for the plain-HTTP recall route, which reads its body field by field.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import MiddlewareContext
from mcp import types as mt
from starlette.testclient import TestClient

from ingestion.surfaces import FULL_TRUST
from mcp_server import recall_route, server
from mcp_server.hidden_tools import RETIRED_ARGS, RetiredToolArgs

_LOGGER = "mcp_server.hidden_tools"


class _Engine:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def recall(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(kw)
        return {"query": kw["query"], "facts": []}


@pytest.fixture()
def engine(monkeypatch):
    stub = _Engine()
    monkeypatch.setattr(server, "_recall_engine", stub)
    return stub


def _call(tool: str, args: dict[str, Any]):
    async def _run():
        async with Client(server.mcp) as c:
            return await c.call_tool(tool, args)

    return asyncio.run(_run())


def _retired_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == _LOGGER]


def test_a_client_that_still_sends_group_id_is_served(engine, caplog):
    caplog.set_level(logging.INFO, logger=_LOGGER)
    res = _call("recall", {"query": "what did we decide", "group_id": "personal"})
    assert res.data == {"query": "what did we decide", "facts": []}
    (call,) = engine.calls
    assert call["query"] == "what did we decide"
    assert "group_id" not in call
    assert _retired_lines(caplog) == ["retired arg group_id ignored on recall"]


def test_a_call_without_it_logs_nothing(engine, caplog):
    caplog.set_level(logging.INFO, logger=_LOGGER)
    _call("recall", {"query": "q", "project": "p"})
    assert engine.calls[0]["project"] == "p"
    assert _retired_lines(caplog) == []


def test_tools_list_has_no_group_id():
    async def _list():
        async with Client(server.mcp) as c:
            return await c.list_tools()

    tools = asyncio.run(_list())
    recall = next(t for t in tools if t.name == "recall")
    assert "group_id" not in recall.inputSchema["properties"]
    assert "group_id" not in (recall.description or "")
    for t in tools:
        assert "group_id" not in t.inputSchema.get("properties", {}), t.name


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("recall", {"query": "q", "bogus": 1}),
        ("fetch", {"ids": ["e:1"], "group_id": "personal"}),
    ],
    ids=["another argument to recall", "group_id to another tool"],
)
def test_only_the_retired_argument_of_the_retired_tool_is_accepted(engine, tool, args):
    with pytest.raises(ToolError):
        _call(tool, args)
    assert engine.calls == []


def test_the_middleware_drops_only_what_is_retired_and_leaves_the_request_alone():
    assert RETIRED_ARGS == {"recall": frozenset({"group_id"})}
    sent = {"query": "q", "group_id": "personal", "project": "p"}
    message = mt.CallToolRequestParams(name="recall", arguments=dict(sent))
    seen: list[MiddlewareContext] = []

    async def _next(ctx):
        seen.append(ctx)
        return "served"

    ctx = MiddlewareContext(message=message, method="tools/call")
    assert asyncio.run(RetiredToolArgs().on_call_tool(ctx, _next)) == "served"
    assert seen[0].message.arguments == {"query": "q", "project": "p"}
    assert message.arguments == sent, "the caller's request is not edited in place"


def test_the_http_recall_route_ignores_group_id_in_the_body():
    eng = _Engine()
    m = FastMCP("retired-args-http")
    recall_route.register(
        m, lambda: eng, lambda request: True, lambda request: FULL_TRUST, lambda r: False
    )
    with TestClient(m.http_app()) as client:
        res = client.post("/recall", json={"query": "q", "group_id": "personal"})
    assert res.status_code == 200
    (call,) = eng.calls
    assert call["query"] == "q" and "group_id" not in call
