"""HTTP to one Synapse server: JSON helpers and a single MCP tool call.

``Transport`` is the only place a request is built. It sends the bearer to its own
server and never anywhere else: not to an absolute URL at another origin (unless the
caller names that origin), and not across a redirect to another origin. urllib's stock
redirect handler forwards every header, Authorization included, wherever a server
redirects; this module's handler does not, and ``urlopen`` lends it to the hook scripts
that still build their own requests (plugin/scripts/config.py and the rest).

Everything here RAISES: urllib.error.HTTPError for an HTTP error status, URLError or
OSError (TimeoutError included) when the server cannot be reached, ValueError for a
reply that is not what the protocol promises, and ``ToolError`` when the server answered
an MCP call with an error. Callers decide whether to fail open (a hook) or surface it (a
CLI or a tool handler).

MCP: the server runs streamable HTTP in stateless mode (mcp_server/server.py), where
every request is served by a fresh session that starts out initialized. A ``tools/call``
needs no ``initialize`` round trip, and the server reads nothing from one (the caller is
the bearer). Replies come as one JSON object or as server-sent events, depending on the
server's ``json_response`` setting (SSE by default); both are handled.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator, Mapping
from http.client import HTTPMessage
from typing import IO, Any

from .credentials import Origin, origin

#: The protocol revision this client speaks. Sent as MCP-Protocol-Version: with no
#: initialize to negotiate it, the header is how the server learns it.
PROTOCOL_VERSION = "2025-06-18"
_MCP_ACCEPT = "application/json, text/event-stream"


class ToolError(Exception):
    """The server answered an MCP call, and the answer is an error: the tool raised or
    rejected its arguments (``code`` is None), or the JSON-RPC layer refused the request
    (``code`` is the JSON-RPC error code). ``str(err)`` is the server's message."""

    def __init__(self, tool: str, message: str, code: int | None = None) -> None:
        super().__init__(message)
        self.tool = tool
        self.code = code


def _keeps_auth(old_url: str, new_url: str) -> bool:
    """Whether a redirect from ``old_url`` to ``new_url`` may carry the Authorization
    header: the same origin (scheme, host and port, a default port written out or not),
    or the upgrade from http to https on the same host and the default ports (80 to 443).
    Any other change of scheme, host or port drops it. The upgrade is the one exception
    the ``requests`` library makes too: a server URL configured as http:// behind a TLS
    redirect would otherwise lose its bearer and answer 401."""
    old, new = urllib.parse.urlsplit(old_url), urllib.parse.urlsplit(new_url)
    try:
        old_port, new_port = old.port, new.port
    except ValueError:  # a port that is not a number: nothing to compare
        return False
    if old.hostname != new.hostname:
        return False
    old_scheme, new_scheme = old.scheme.lower(), new.scheme.lower()
    default_ports = {"http": 80, "https": 443}
    if old_port is None:
        old_port = default_ports.get(old_scheme)
    if new_port is None:
        new_port = default_ports.get(new_scheme)
    if old_scheme == new_scheme:
        return old_port == new_port
    return (old_scheme, old_port, new_scheme, new_port) == ("http", 80, "https", 443)


class _SameOriginAuth(urllib.request.HTTPRedirectHandler):
    """Follow redirects as urllib does, minus the Authorization header when the redirect
    leaves the origin the request was sent to (``_keeps_auth``)."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and not _keeps_auth(req.full_url, newurl):
            for name in list(new.headers):
                if name.lower() == "authorization":
                    del new.headers[name]
        return new


def urlopen(req: urllib.request.Request, timeout: float | None = None) -> Any:
    """``urllib.request.urlopen(req, timeout=timeout)`` with this module's redirect
    handler: a redirect is followed as urllib follows it, but the Authorization header
    goes along only where ``_keeps_auth`` allows. For the hook scripts that build their
    own requests (plugin/scripts/config.py, plugin-codex/scripts/common.py), so that the
    rule exists once. Raises as urlopen does."""
    opener = urllib.request.build_opener(_SameOriginAuth())
    if timeout is None:
        return opener.open(req)
    return opener.open(req, timeout=timeout)


def _decode(raw: bytes) -> Any:
    """A JSON reply body; an empty one reads as ``{}`` (the hooks' helpers agree)."""
    return json.loads(raw or b"{}")


def sse_events(lines: Iterable[bytes]) -> Iterator[str]:
    """The ``data`` of each server-sent event in ``lines`` (raw lines, any of CRLF, LF or
    CR endings), multi-line data joined with newlines as the SSE spec says. Comments and
    other fields are skipped; an event without data yields nothing."""
    data: list[str] = []
    for raw in lines:
        for line in raw.decode("utf-8").splitlines() or [""]:
            if not line:
                if data:
                    yield "\n".join(data)
                data = []
            elif line.startswith("data:"):
                value = line[5:]
                data.append(value[1:] if value.startswith(" ") else value)
    if data:
        yield "\n".join(data)


def _messages(payload: Any) -> list[Any]:
    """A JSON-RPC payload as a list of messages (a batch reply is already one)."""
    return payload if isinstance(payload, list) else [payload]


def _reply(messages: Iterable[Any], request_id: int) -> Mapping[str, Any] | None:
    for msg in messages:
        if (
            isinstance(msg, dict)
            and msg.get("id") == request_id
            and ("result" in msg or "error" in msg)
        ):
            return msg
    return None


def tool_text(tool: str, reply: Mapping[str, Any]) -> str:
    """The text of a ``tools/call`` reply: its text content items joined by newlines, or
    the structured content as JSON when there is no text. Raises ToolError when the reply
    is an error, ValueError when it has no result."""
    if "error" in reply:
        err = reply["error"] if isinstance(reply["error"], dict) else {}
        code = err.get("code")
        raise ToolError(
            tool,
            str(err.get("message") or "JSON-RPC error"),
            code if isinstance(code, int) else None,
        )
    result = reply.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"the reply to tools/call {tool} carries no result")
    content = result.get("content") or []
    text = "\n".join(
        str(item.get("text", ""))
        for item in content
        if isinstance(item, dict) and item.get("type") == "text"
    )
    if not text and result.get("structuredContent") is not None:
        text = json.dumps(result["structuredContent"])
    if result.get("isError"):
        raise ToolError(tool, text or "the tool reported an error")
    return text


class Transport:
    """JSON over HTTP to the Synapse server at ``base_url``, with ``token`` as the bearer.

    Paths are endpoint-relative (``/preferences/top``) or absolute URLs. The bearer goes
    only to ``base_url``'s origin and to ``token_origins``; an absolute URL elsewhere is
    sent without it. A Transport holds no connection and no mutable state; build one per
    settings value and share it freely.
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        user_agent: str,
        default_timeout: float = 30.0,
        *,
        token_origins: Iterable[str] = (),
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._token = token
        self.user_agent = user_agent
        self.default_timeout = default_timeout
        self._origins: frozenset[Origin] = frozenset(
            origin(url) for url in (self.base_url, *token_origins) if url
        )
        self._opener = urllib.request.build_opener(_SameOriginAuth())

    def __repr__(self) -> str:
        return f"Transport({self.base_url!r}, user_agent={self.user_agent!r})"

    def url(self, path: str) -> str:
        """The absolute URL for ``path``. Raises ValueError for a relative path when no
        server URL is configured."""
        if path.startswith(("http://", "https://")):
            return path
        if not self.base_url:
            raise ValueError("no Synapse server URL is configured")
        return self.base_url + path

    def _headers(self, url: str, extra: Mapping[str, str]) -> dict[str, str]:
        headers = {"User-Agent": self.user_agent, **extra}
        if self._token and origin(url) in self._origins:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _open(
        self,
        method: str,
        path: str,
        body: bytes | None,
        headers: Mapping[str, str],
        timeout: float | None,
    ) -> Any:
        url = self.url(path)
        req = urllib.request.Request(
            url, data=body, method=method.upper(), headers=self._headers(url, headers)
        )
        return self._opener.open(req, timeout=self.default_timeout if timeout is None else timeout)

    def request_json(
        self,
        method: str,
        path: str,
        payload: Any = None,
        *,
        params: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Send ``payload`` (as JSON, when not None) with any HTTP method and return the
        parsed JSON reply (``{}`` for an empty body). ``params`` are urlencoded onto the
        query string. Raises (see the module docstring)."""
        if params:
            path += ("&" if "?" in path else "?") + urllib.parse.urlencode(params)
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"} if body is not None else {}
        with self._open(method, path, body, headers, timeout) as resp:
            return _decode(resp.read())

    def get_json(
        self, path: str, params: Mapping[str, Any] | None = None, timeout: float | None = None
    ) -> Any:
        """GET ``path`` and return the parsed JSON reply. Raises."""
        return self.request_json("GET", path, params=params, timeout=timeout)

    def post_json(self, path: str, payload: Any, timeout: float | None = None) -> Any:
        """POST ``payload`` as JSON to ``path`` and return the parsed JSON reply. Raises."""
        return self.request_json("POST", path, payload, timeout=timeout)

    def mcp_call(
        self,
        tool: str,
        arguments: Mapping[str, Any],
        timeout: float | None = None,
        *,
        path: str = "/mcp",
    ) -> str:
        """Call one MCP tool and return its text (``tool_text``). One request, no
        ``initialize`` (see the module docstring).

        Raises ToolError when the tool reports an error or the server refuses the call;
        HTTPError for an HTTP error status (401 when the token is not accepted); URLError
        or OSError when the server cannot be reached; ValueError when the reply holds no
        answer to this call."""
        request_id = 1
        body = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {"name": tool, "arguments": dict(arguments)},
            }
        ).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": _MCP_ACCEPT,
            "MCP-Protocol-Version": PROTOCOL_VERSION,
        }
        with self._open("POST", path, body, headers, timeout) as resp:
            kind = (resp.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
            if kind == "text/event-stream":
                # Stop at the answer: a server may keep the stream open after it.
                reply = None
                for data in sse_events(resp):
                    reply = _reply(_messages(json.loads(data)), request_id)
                    if reply is not None:
                        break
            else:
                reply = _reply(_messages(_decode(resp.read())), request_id)
        if reply is None:
            raise ValueError(f"the server's reply holds no answer to tools/call {tool}")
        return tool_text(tool, reply)
