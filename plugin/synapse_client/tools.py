"""The server's MCP tools for a harness that registers them itself.

A harness with no MCP client of its own (or one that wants its own tool loop) offers the
model ``schemas()`` and answers each call with ``call()``. The definitions come from
``tool_schemas.json`` beside this file, a snapshot of the server's tools/list made by
scripts/gen_client_tool_schemas.py; tests/test_client_core_tools.py fails when the two
drift apart.

The session id travels the way the Claude and Codex PreToolUse hooks send it
(plugin/scripts/self_session_inject.py): ``self_session`` on recall, recall_full_turns
and fetch_session, so serving leaves out the calling session's own turns, which are
already in the model's context; ``session_id`` on recall_feedback and remember, which
accept nothing else. The model never sees those parameters: ``schemas()`` hides them,
and ``surface`` (deprecated and ignored by the server) everywhere. The user-facing
``session_id`` of recall_full_turns and fetch_session, the conversation to read, stays.

``schemas()`` RAISES when the snapshot is missing or unreadable (a broken install, to be
seen at startup); ``inject_session`` raises TypeError for arguments that are not a
mapping; ``call()`` never raises.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .api import Client
from .http import ToolError
from .outbox import short_reason

SNAPSHOT = "tool_schemas.json"
#: Tools that carry the caller's session id, and the parameter each takes it in.
SESSION_FIELD = MappingProxyType(
    {
        "recall": "self_session",
        "recall_full_turns": "self_session",
        "fetch_session": "self_session",
        "recall_feedback": "session_id",
        "remember": "session_id",
    }
)
#: Parameters no tool shows the model: deprecated, and ignored by the server.
HIDDEN_EVERYWHERE = frozenset({"surface"})
DEFAULT_TIMEOUT = 60.0


def hidden_parameters(tool: str) -> frozenset[str]:
    """The parameters of ``tool`` that ``schemas()`` leaves out: ``surface``, and the
    session parameter ``inject_session`` fills."""
    field = SESSION_FIELD.get(tool)
    return HIDDEN_EVERYWHERE | {field} if field else HIDDEN_EVERYWHERE


def _snapshot() -> list[dict[str, Any]]:
    raw = json.loads((Path(__file__).parent / SNAPSHOT).read_text(encoding="utf-8"))
    tools = raw.get("tools") if isinstance(raw, dict) else None
    if not isinstance(tools, list):
        raise ValueError(f"{SNAPSHOT} holds no tool list")
    return [tool for tool in tools if isinstance(tool, dict)]


def _without(schema: Mapping[str, Any], hidden: frozenset[str]) -> dict[str, Any]:
    out = copy.deepcopy(dict(schema))
    props = out.get("properties")
    if isinstance(props, dict):
        out["properties"] = {k: v for k, v in props.items() if k not in hidden}
    required = out.get("required")
    if isinstance(required, list):
        out["required"] = [name for name in required if name not in hidden]
    return out


def schemas() -> list[dict[str, Any]]:
    """The tools to offer a model, in the server's order: ``{"name", "description",
    "parameters"}`` each, ``parameters`` being a JSON Schema object with the hidden
    parameters left out. Read from the snapshot on every call, as fresh objects the
    caller may change. RAISES OSError or ValueError when the snapshot is missing or
    malformed."""
    out = []
    for tool in _snapshot():
        name = str(tool.get("name") or "")
        if not name:
            continue
        params = tool.get("inputSchema")
        out.append(
            {
                "name": name,
                "description": str(tool.get("description") or ""),
                "parameters": _without(
                    params if isinstance(params, dict) else {"type": "object"},
                    hidden_parameters(name),
                ),
            }
        )
    return out


def inject_session(
    tool: str, arguments: Mapping[str, Any] | None, session_id: str | None
) -> dict[str, Any]:
    """A copy of ``arguments`` with the caller's session id in the parameter ``tool``
    takes it in (``SESSION_FIELD``), as the PreToolUse hooks add it. A value already
    there is never overwritten (a retry, a nested call); an empty one counts as unset,
    as it does for the hooks. No session id, or a tool without the parameter (fetch, or
    one this module does not know): the copy, unchanged. RAISES TypeError when
    ``arguments`` is not a mapping."""
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, Mapping):
        raise TypeError(f"{tool} arguments must be a mapping, not {type(arguments).__name__}")
    out = dict(arguments)
    field = SESSION_FIELD.get(tool)
    if field and session_id and not out.get(field):
        out[field] = session_id
    return out


def _failure(tool: str, message: str, code: int | None = None) -> str:
    body: dict[str, Any] = {"error": message, "tool": tool}
    if code is not None:
        body["code"] = code
    return json.dumps(body, ensure_ascii=False)


def call(
    client: Client,
    tool: str,
    arguments: Mapping[str, Any] | None,
    session_id: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> str:
    """Call ``tool`` on the server with the session id injected (``inject_session``)
    and return its text. Never raises: any failure comes back as a JSON object with an
    ``error`` field (and ``tool``), for the model to read like any other result. A tool
    error carries the server's message, and ``code`` when the JSON-RPC layer refused the
    request; a transport failure carries a short reason built from the error's type and
    status only (``outbox.short_reason``), never from text that could echo a URL or a
    header."""
    try:
        args = inject_session(tool, arguments, session_id)
    except TypeError:
        return _failure(tool, "the arguments must be a JSON object")
    try:
        return client.call_tool(tool, args, timeout=timeout)
    except ToolError as err:
        return _failure(tool, str(err) or "the tool reported an error", err.code)
    except Exception as err:
        return _failure(tool, f"Synapse could not answer: {short_reason(err)}")
