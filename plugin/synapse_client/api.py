"""The Synapse server's client routes, one thin method each.

Every method RAISES what the transport raises (http.py's module docstring): an HTTP
error, an unreachable server, a malformed reply, a tool error. None is swallowed here;
the callers that must fail open (session-start blocks, the ingest hook) catch around
the call, and context.py's pieces do exactly that.
"""

from __future__ import annotations

import urllib.parse
from collections.abc import Mapping
from typing import Any

from .http import Transport
from .settings import Settings


def _object(reply: Any, route: str) -> dict[str, Any]:
    if not isinstance(reply, dict):
        raise ValueError(f"{route} answered with {type(reply).__name__}, not a JSON object")
    return reply


class Client:
    """One server, one credential: ``settings.base_url`` with ``settings.token``.

    ``user_agent`` names the harness and version (``synapse-client/0.24.0 hermes``). Send a real
    one: Cloudflare in front of a hosted server refuses urllib's default.
    """

    def __init__(self, settings: Settings, user_agent: str) -> None:
        self.settings = settings
        # The legacy SYNAPSE_INGEST_URL may name its own origin; settings resolved the
        # token for it too, so the bearer may go there.
        self.transport = Transport(
            settings.base_url,
            settings.token,
            user_agent,
            token_origins=(settings.ingest_url,),
        )

    def context(self, project: str | None = None, timeout: float = 10) -> dict[str, Any]:
        """``GET /context``: the rendered board for ``project`` (``status``, ``text``,
        ``trust``, ...). Raises; a 401 is how an unenrolled machine is told no."""
        params = {"project": project} if project else {}
        return _object(self.transport.get_json("/context", params, timeout=timeout), "/context")

    def preferences(self, limit: int = 8, timeout: float = 10) -> list[dict[str, Any]]:
        """``GET /preferences/top``: the top preference items (``pref``, ``polarity``,
        ...), ``[]`` when the server has none. Raises."""
        reply = _object(
            self.transport.get_json("/preferences/top", {"limit": limit}, timeout=timeout),
            "/preferences/top",
        )
        items = reply.get("items") or []
        return [item for item in items if isinstance(item, dict)]

    def ingest(self, payload: Mapping[str, Any], timeout: float = 30) -> Any:
        """``POST`` ``payload`` to the ingest URL and return the parsed reply. The payload
        is the ingest route's body (``format``, ``records``, ``source``, ``project``, ...);
        the server answers 400 for a format it does not know. Raises."""
        return self.transport.post_json(self.settings.ingest_url, dict(payload), timeout=timeout)

    def _private(self, method: str, session_id: str, timeout: float) -> dict[str, Any]:
        path = f"/private-sessions/{urllib.parse.quote(session_id, safe='')}"
        return _object(self.transport.request_json(method, path, timeout=timeout), path)

    def private_get(self, session_id: str, timeout: float = 15) -> dict[str, Any]:
        """``GET /private-sessions/<id>``: whether the server holds the session off the
        record (``{"private": bool, ...}``). Raises."""
        return self._private("GET", session_id, timeout)

    def private_put(self, session_id: str, timeout: float = 15) -> dict[str, Any]:
        """``PUT /private-sessions/<id>``: take the session off the record, durably.
        Raises: a caller must not tell the user a session is private unless this
        returned."""
        return self._private("PUT", session_id, timeout)

    def private_delete(self, session_id: str, timeout: float = 15) -> dict[str, Any]:
        """``DELETE /private-sessions/<id>``: put the session back on the record. Raises."""
        return self._private("DELETE", session_id, timeout)

    def call_tool(self, name: str, arguments: Mapping[str, Any], timeout: float = 60) -> str:
        """Call the MCP tool ``name`` and return its text. Raises ToolError when the tool
        reports an error, and whatever the transport raises (see http.Transport.mcp_call).
        tools.call is the never-raising wrapper for tool handlers."""
        return self.transport.mcp_call(name, arguments, timeout=timeout)
