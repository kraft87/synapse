"""Tool calls out of transcript records: each ``tool_use`` paired with its ``tool_result``.

The records are what the Stop hooks POST to ``/ingest`` (Claude Code transcript lines,
Codex rollout lines). A call with no result yet (cut off by the end of a tail) is skipped;
the shipment that carries its result starts at it.

Two readers feed the same :class:`ToolCall`:

* :func:`episode_tool_calls`: the calls of a stored turn, read from the records the parser
  built the turn from (``Episode.records``), so a call carries its turn's session, project
  and span;
* :func:`sidechain_tool_calls`: the calls of sub-agent records. ``JSONLParser`` drops those
  from the stored conversation, but a sub-agent is where much web research happens; they
  belong to no stored turn.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ingestion.models import Episode
from ingestion.turns import parse_iso


@dataclass
class ToolCall:
    """One tool call and what it returned."""

    tool_use_id: str
    tool_name: str
    tool_input: dict[str, Any]
    result_text: str
    is_error: bool
    session_id: str
    project: str | None
    timestamp: datetime | None
    #: span_id of the stored turn the call ran in; None for a sub-agent record
    span_id: str | None = None
    #: that turn's episode id, when the writer just stored it; looked up from the span otherwise
    episode_id: int | None = None
    #: the surface the credential of the POST resolves to (``episodes.surface_id``)
    surface_id: str | None = None


def result_text(content: Any) -> str:
    """The text of a tool result: a string, or a list of ``{"text": ...}`` blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(result_text(c) for c in content if c)
    if isinstance(content, dict):
        text = content.get("text") or content.get("content") or content.get("output")
        return result_text(text) if text else ""
    return "" if content is None else str(content)


# ---------- Claude Code ----------


def _claude_pairs(
    records: list[dict[str, Any]],
) -> Iterator[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """(use record, use block, result block) for each paired call, in result order."""
    pending: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for rec in records:
        content = (rec.get("message") or {}).get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and isinstance(block.get("id"), str):
                pending[block["id"]] = (rec, block)
            elif block.get("type") == "tool_result" and isinstance(block.get("tool_use_id"), str):
                use = pending.pop(block["tool_use_id"], None)
                if use:
                    yield use[0], use[1], block


def _claude_call(use_rec: dict[str, Any], use: dict[str, Any], res: dict[str, Any]) -> ToolCall:
    inp = use.get("input")
    return ToolCall(
        tool_use_id=use["id"],
        tool_name=str(use.get("name") or ""),
        tool_input=inp if isinstance(inp, dict) else {},
        result_text=result_text(res.get("content")),
        is_error=bool(res.get("is_error")),
        session_id="",
        project=None,
        timestamp=parse_iso(use_rec.get("timestamp")),
    )


# ---------- Codex ----------

_CODEX_CALLS = ("function_call", "custom_tool_call", "local_shell_call")
_CODEX_OUTPUTS = ("function_call_output", "custom_tool_call_output")


def _codex_input(payload: dict[str, Any]) -> dict[str, Any]:
    """A Codex call's arguments as an input object: ``function_call`` carries a JSON string,
    ``custom_tool_call`` a free-form one, ``local_shell_call`` an action with a command."""
    if payload.get("type") == "local_shell_call":
        action = payload.get("action")
        return {"command": (action or {}).get("command")} if isinstance(action, dict) else {}
    raw = (
        payload.get("arguments") if payload.get("type") == "function_call" else payload.get("input")
    )
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"input": raw}
        return parsed if isinstance(parsed, dict) else {"input": raw}
    return raw if isinstance(raw, dict) else {}


def _codex_calls(records: list[dict[str, Any]]) -> Iterator[ToolCall]:
    pending: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for rec in records:
        payload = rec.get("payload")
        if not isinstance(payload, dict):
            continue
        ptype, call_id = payload.get("type"), payload.get("call_id")
        if not isinstance(call_id, str):
            continue
        if ptype in _CODEX_CALLS:
            pending[call_id] = (rec, payload)
        elif ptype in _CODEX_OUTPUTS and call_id in pending:
            use_rec, use = pending.pop(call_id)
            yield ToolCall(
                tool_use_id=call_id,
                tool_name="shell"
                if use.get("type") == "local_shell_call"
                else str(use.get("name") or ""),
                tool_input=_codex_input(use),
                result_text=result_text(payload.get("output")),
                is_error=False,
                session_id="",
                project=None,
                timestamp=parse_iso(use_rec.get("timestamp")),
            )


# ---------- Entry points ----------


def episode_tool_calls(ep: Episode, project_default: str | None = None) -> list[ToolCall]:
    """The tool calls of the turn ``ep`` was parsed from, stamped with its session, project,
    span and surface. Empty for a platform whose records the writer does not read."""
    if ep.platform == "codex":
        calls = list(_codex_calls(ep.records))
    elif ep.platform == "claude_code":
        calls = [_claude_call(*pair) for pair in _claude_pairs(ep.records)]
    else:
        return []
    for call in calls:
        call.session_id = ep.session_id
        call.project = ep.project or project_default
        call.span_id = ep.span_id
        call.surface_id = ep.surface_id
    return calls


def sidechain_tool_calls(
    records: list[dict[str, Any]],
    project_of: Callable[[str | None], str | None],
    surface_id: str | None,
) -> list[ToolCall]:
    """The calls of the sub-agent (``isSidechain``) records among Claude Code ``records``.
    ``project_of`` labels a record's ``cwd``."""
    side = [r for r in records if isinstance(r, dict) and r.get("isSidechain")]
    calls = []
    for use_rec, use, res in _claude_pairs(side):
        call = _claude_call(use_rec, use, res)
        session_id = use_rec.get("sessionId")
        if not isinstance(session_id, str) or not session_id:
            continue  # nothing to stamp the artifact with
        call.session_id = session_id
        call.project = project_of(use_rec.get("cwd"))
        call.surface_id = surface_id
        calls.append(call)
    return calls
