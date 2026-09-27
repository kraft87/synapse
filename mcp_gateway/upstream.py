"""Upstream MCP servers as FastMCP providers, with identity-safe caching.

Built on FastMCP 3.4.2's proxy COMPONENTS (``ProxyTool`` & co.), which forward calls and
pass upstream results — schemas, structured content, ``isError`` — through unchanged. It
deliberately does NOT use ``ProxyProvider``/``ProxyClient``/``create_proxy`` as shipped:

* ``ProxyClient`` turns on ``forward_incoming_headers``, relaying the caller's
  ``Authorization`` (and most other inbound headers) to the upstream, handing a
  third-party server the caller's Synapse device token.
* ``ProxyProvider`` keeps one component cache for every session and identity, and its
  ``_get_*`` read that shared cache right after a refresh another identity may have won.

So each upstream here gets an explicit client factory (auth chosen per upstream, header
forwarding off) and a cache keyed by an identity function: the caller's token hash for
Synapse, a constant for configured upstreams, which only ever see the deployment's own
credential.

Discovery is bounded: every listing runs under ``discovery_timeout``; a failed listing
falls back to the last good snapshot (at most an hour old), and otherwise the upstream
sits out for ``failure_backoff`` seconds so a dead upstream cannot slow every request.
Failures surface as :class:`UpstreamUnavailable`, which FastMCP's aggregate provider logs and skips —
memory and skills keep working while a configured upstream is down.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import anyio
import httpx
import mcp.types
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.exceptions import FastMCPError, PromptError, ResourceError, ToolError
from fastmcp.prompts import Prompt, PromptResult
from fastmcp.resources import Resource, ResourceTemplate
from fastmcp.resources.base import ResourceResult
from fastmcp.server.context import Context
from fastmcp.server.providers.base import Provider
from fastmcp.server.providers.proxy import ProxyPrompt, ProxyResource, ProxyTemplate, ProxyTool
from fastmcp.tools.base import Tool, ToolResult
from fastmcp.utilities.components import FastMCPComponent
from fastmcp.utilities.versions import VersionSpec, version_sort_key
from mcp.shared.exceptions import McpError

logger = logging.getLogger(__name__)

ClientFactory = Callable[[], Client[Any]]

_MAX_IDENTITIES = 512
#: A failed refresh may fall back to a snapshot at most this old (seconds).
_MAX_STALE = 3600.0
_KINDS = ("tools", "resources", "templates", "prompts")


class UpstreamUnavailable(Exception):
    """An upstream could not be listed. The message never carries a URL or credential."""


def _describe(label: str, e: BaseException) -> str:
    """A URL-free description, looking through wrappers (FastMCP's Client re-raises a
    refused connection as RuntimeError) for the error that actually explains the failure."""
    chain: list[BaseException] = []
    cur: BaseException | None = e
    while cur is not None and all(cur is not c for c in chain):
        chain.append(cur)
        cur = cur.__cause__ or cur.__context__
    for c in chain:
        if isinstance(c, httpx.HTTPStatusError):
            return f"{label} upstream returned HTTP {c.response.status_code}"
        if isinstance(c, TimeoutError | httpx.TimeoutException):
            return f"{label} upstream timed out"
    cause = next((c for c in chain if isinstance(c, httpx.HTTPError | OSError)), e)
    return f"{label} upstream request failed ({type(cause).__name__})"


async def _guard[T](label: str, op: Awaitable[T], error: type[FastMCPError]) -> T:
    """Run an upstream operation, keeping upstream errors and hiding transport details.

    A JSON-RPC error from the upstream is its own answer and passes through verbatim.
    Anything else (httpx, anyio, timeouts) could quote the keyed upstream URL, so it is
    replaced by a generic message — raised ``from None`` so no traceback carries it either.
    """
    try:
        return await op
    except FastMCPError:
        raise
    except McpError as e:
        raise error(e.error.message) from None
    except Exception as e:
        raise error(_describe(label, e)) from None


class GatewayTool(ProxyTool):
    _gw_label: str = "upstream"

    async def run(self, arguments: dict[str, Any], context: Context | None = None) -> ToolResult:
        return await _guard(self._gw_label, super().run(arguments, context), ToolError)


class GatewayResource(ProxyResource):
    _gw_label: str = "upstream"

    async def read(self) -> ResourceResult:
        return await _guard(self._gw_label, super().read(), ResourceError)


class GatewayTemplate(ProxyTemplate):
    _gw_label: str = "upstream"

    async def create_resource(
        self, uri: str, params: dict[str, Any], context: Context | None = None
    ) -> ProxyResource:
        # ProxyTemplate reads eagerly here. It percent-quotes each parameter (a skill file
        # path travels as "scripts%2Frun.py"); the upstream's template matcher unquotes it.
        return await _guard(
            self._gw_label, super().create_resource(uri, params, context), ResourceError
        )


class GatewayPrompt(ProxyPrompt):
    _gw_label: str = "upstream"

    async def render(self, arguments: dict[str, Any]) -> PromptResult:  # type: ignore[override]
        return await _guard(self._gw_label, super().render(arguments), PromptError)


class _Snapshot:
    __slots__ = ("items", "stamp")

    def __init__(self, items: Sequence[Any], stamp: float) -> None:
        self.items = items
        self.stamp = stamp


class UpstreamProvider(Provider):
    """Expose one upstream MCP server's tools, resources, templates and prompts.

    Args:
        label: Short upstream name used in logs and errors (never a URL).
        client_factory: Returns a NEW, unconnected Client for the current request.
        identity: Cache partition for the current request (raise PermissionError when
            there is no caller; the provider then serves nothing).
        allow: ``allow(kind)`` — may the current caller see this upstream's ``kind``
            ("tools", "resources", "templates", "prompts")? Checked before any upstream
            I/O on every list and lookup, so the gateway can withhold a component class
            (e.g. skill resources) even when the upstream itself would serve it.
    """

    def __init__(
        self,
        label: str,
        client_factory: ClientFactory,
        *,
        identity: Callable[[], str],
        allow: Callable[[str], bool],
        cache_ttl: float,
        discovery_timeout: float,
        failure_backoff: float,
    ) -> None:
        super().__init__()
        self.label = label
        self._client_factory = client_factory
        self._identity = identity
        self._allow = allow
        self._ttl = cache_ttl
        self._discovery_timeout = discovery_timeout
        self._backoff = failure_backoff
        self._snapshots: OrderedDict[tuple[str, str], _Snapshot] = OrderedDict()
        self._failed_at: OrderedDict[str, float] = OrderedDict()

    def __repr__(self) -> str:
        return f"UpstreamProvider({self.label!r})"

    # ------------------------------------------------------------------ discovery

    async def _fetch(self, kind: str) -> Sequence[Any]:
        factory = self._client_factory
        client = factory()
        try:
            async with client:
                if kind == "tools":
                    out: list[Any] = [
                        GatewayTool.from_mcp_tool(factory, t) for t in await client.list_tools()
                    ]
                elif kind == "resources":
                    out = [
                        GatewayResource.from_mcp_resource(factory, r)
                        for r in await client.list_resources()
                    ]
                elif kind == "templates":
                    out = [
                        GatewayTemplate.from_mcp_template(factory, t)
                        for t in await client.list_resource_templates()
                    ]
                else:
                    out = [
                        GatewayPrompt.from_mcp_prompt(factory, p)
                        for p in await client.list_prompts()
                    ]
        except McpError as e:
            if e.error.code == mcp.types.METHOD_NOT_FOUND:
                return []
            raise
        for item in out:
            item._gw_label = self.label
        return out

    async def _components(self, kind: str) -> Sequence[Any]:
        if not self._allow(kind):
            return []
        try:
            ident = self._identity()
        except PermissionError:
            return []
        now = time.monotonic()
        key = (ident, kind)
        snap = self._snapshots.get(key)
        if snap is not None and now - snap.stamp < self._ttl:
            return snap.items
        if snap is not None and now - snap.stamp >= _MAX_STALE:
            self._snapshots.pop(key, None)
            snap = None
        failed = self._failed_at.get(ident)
        if failed is not None and now - failed < self._backoff:
            if snap is not None:
                return snap.items
            raise UpstreamUnavailable(f"{self.label} upstream unavailable (backing off)")
        try:
            with anyio.fail_after(self._discovery_timeout):
                items = await self._fetch(kind)
        except Exception as e:
            self._remember(self._failed_at, ident, time.monotonic())
            message = (
                _describe(self.label, e)
                if not isinstance(e, McpError)
                else (f"{self.label} upstream error listing {kind}")
            )
            logger.warning("%s (%s discovery)", message, kind)
            if snap is not None:
                return snap.items
            raise UpstreamUnavailable(message) from None
        self._failed_at.pop(ident, None)
        if self._ttl > 0:
            self._remember(self._snapshots, key, _Snapshot(items, time.monotonic()))
        return items

    @staticmethod
    def _remember(store: OrderedDict[Any, Any], key: Any, value: Any) -> None:
        store[key] = value
        store.move_to_end(key)
        while len(store) > _MAX_IDENTITIES * len(_KINDS):
            store.popitem(last=False)

    @staticmethod
    def _pick[C: FastMCPComponent](matching: list[C], version: VersionSpec | None) -> C | None:
        if version:
            matching = [c for c in matching if version.matches(c.version)]
        return max(matching, key=version_sort_key) if matching else None

    # ------------------------------------------------------------------ provider API

    async def _list_tools(self) -> Sequence[Tool]:
        return await self._components("tools")

    async def _get_tool(self, name: str, version: VersionSpec | None = None) -> Tool | None:
        tools: Sequence[Tool] = await self._components("tools")
        return self._pick([t for t in tools if t.name == name], version)

    async def _list_resources(self) -> Sequence[Resource]:
        return await self._components("resources")

    async def _get_resource(self, uri: str, version: VersionSpec | None = None) -> Resource | None:
        resources: Sequence[Resource] = await self._components("resources")
        return self._pick([r for r in resources if str(r.uri) == uri], version)

    async def _list_resource_templates(self) -> Sequence[ResourceTemplate]:
        return await self._components("templates")

    async def _get_resource_template(
        self, uri: str, version: VersionSpec | None = None
    ) -> ResourceTemplate | None:
        templates: Sequence[ResourceTemplate] = await self._components("templates")
        return self._pick([t for t in templates if t.matches(uri) is not None], version)

    async def _list_prompts(self) -> Sequence[Prompt]:
        return await self._components("prompts")

    async def _get_prompt(self, name: str, version: VersionSpec | None = None) -> Prompt | None:
        prompts: Sequence[Prompt] = await self._components("prompts")
        return self._pick([p for p in prompts if p.name == name], version)

    async def get_tasks(self) -> Sequence[Any]:
        # The default lists every component at server startup, outside any request — there
        # is no caller then, and proxied components cannot run as tasks anyway.
        return []


class GatedProvider(Provider):
    """Serve an inner provider's components only when ``allow()`` holds for the caller."""

    def __init__(self, inner: Provider, allow: Callable[[], bool], label: str) -> None:
        super().__init__()
        self._inner = inner
        self._allow = allow
        self._label = label

    def __repr__(self) -> str:
        return f"GatedProvider({self._label!r})"

    async def _list_tools(self) -> Sequence[Tool]:
        return await self._inner.list_tools() if self._allow() else []

    async def _get_tool(self, name: str, version: VersionSpec | None = None) -> Tool | None:
        return await self._inner.get_tool(name, version) if self._allow() else None

    async def _list_resources(self) -> Sequence[Resource]:
        return await self._inner.list_resources() if self._allow() else []

    async def _get_resource(self, uri: str, version: VersionSpec | None = None) -> Resource | None:
        return await self._inner.get_resource(uri, version) if self._allow() else None

    async def _list_resource_templates(self) -> Sequence[ResourceTemplate]:
        return await self._inner.list_resource_templates() if self._allow() else []

    async def _get_resource_template(
        self, uri: str, version: VersionSpec | None = None
    ) -> ResourceTemplate | None:
        return await self._inner.get_resource_template(uri, version) if self._allow() else None

    async def _list_prompts(self) -> Sequence[Prompt]:
        return await self._inner.list_prompts() if self._allow() else []

    async def _get_prompt(self, name: str, version: VersionSpec | None = None) -> Prompt | None:
        return await self._inner.get_prompt(name, version) if self._allow() else None

    async def get_tasks(self) -> Sequence[Any]:
        return []


def http_client_factory(
    url: str,
    *,
    headers: Callable[[], dict[str, str]],
    timeout: float,
    init_timeout: float,
) -> ClientFactory:
    """A factory for fresh upstream clients that send ONLY the headers given here.

    ``forward_incoming_headers`` stays off (the plain-transport default), so nothing from
    the inbound request — authorization, cookies, forwarding headers — reaches the upstream
    unless ``headers()`` puts it there explicitly. A new client (and httpx pool) per
    operation means no connection, cookie or session state is shared between callers.
    """

    def factory() -> Client[Any]:
        transport = StreamableHttpTransport(url, headers=headers())
        transport.forward_incoming_headers = False
        return Client(transport, timeout=timeout, init_timeout=init_timeout)

    return factory
