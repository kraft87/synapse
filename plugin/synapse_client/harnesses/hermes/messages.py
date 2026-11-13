"""Hermes chat messages -> ``/ingest`` ``messages`` records.

Hermes hands a memory provider its whole conversation on every turn, as the live
message dicts of its own transcript (OpenAI-style, plus Hermes's bookkeeping keys).
:func:`to_records` keeps only what the server's ``messages`` parser reads and builds new
dicts; the input is never modified, since those dicts are Hermes's own.

What a record keeps: ``role`` (user, assistant and tool only), the text content (a
string, or the text parts of a list joined by newlines), an assistant row's tool calls
as ``{"name", "arguments"}``, a tool row's tool ``name``, ``ts`` from ``timestamp`` and
``id`` from ``message_uid``. Nothing else leaves: not ``api_content`` (the text Hermes
actually sent, with the memory context it injected, our own nudge included), not
``display_metadata``, reasoning, or provider-specific blobs.

- A row carrying ``_compressed_summary`` is a compaction summary or marker, not
  something anyone said: dropped.
- Every ``<memory-context>...</memory-context>`` block is removed from the text.
- A user row is ``synthetic`` when Hermes injected it in the user's place. The server
  skips such a row, so it neither opens a turn nor renders. The markers mirror Hermes
  without importing it: the producer flags of ``agent/conversation_compression_scaffolding``
  (``_SYNTHETIC_USER_FLAGS``), any ``display_kind`` (a hidden row, or a ``/steer``
  message, which lands mid-turn and must not split it), and the opening text of Hermes's
  own control frames (``agent/prompt_builder.CONTROL_FRAME_OPENERS``,
  ``agent/context_compressor._SYNTHETIC_USER_ROW_PREFIXES``,
  ``agent/conversation_compression_scaffolding._SYNTHETIC_USER_PREFIXES``). The flags are
  not persisted, so a resumed session's rows are recognised by their text.
- An assistant row is a reply only when it shows what the user got. Hermes types the
  two kinds of assistant row that are not (agent/turn_failure_copy.py and its writers):
  ``display_kind`` "failed_turn", the boundary Hermes appends when a turn fails ("This
  turn did not complete...", agent/conversation_loop.py, gateway/run_turn_persistence.py,
  gateway/session_results.py), is dropped: failure text is never stored as something
  the assistant said. "hidden", a held turn's interim commentary or rejected answer
  (agent/stream_delivery.py ``withhold_held_row``) or the empty interrupt placeholder,
  keeps only its tool calls, if any. Any other ``display_kind`` on an assistant row is
  treated as "hidden". A row persisted before Hermes typed its boundaries is recognised,
  as Hermes recognises it (``untyped_failed_turn_display_kind``), by text that is
  exactly one of the three notices; a reply that quotes one stays a reply. Every other
  ``display_kind`` Hermes writes is on a user row (above).
- Tool results are capped at 2000 characters and tool-call arguments at 1000; the
  server renders less of either. User and assistant text are sent whole.

The records are cut into turns, and the finished ones uploaded, by the core
(``synapse_client/messages.py``, ``turn_upload.TurnUploader``). Two of the rules above exist
for that cut. Synthetic rows do not count towards a turn's key: Hermes adds them after a
finished turn (a task-list snapshot after compression), and that must neither reopen the
turn nor make it look new. And a failed turn never completes, so its user message is not
sent either: it got no reply, Hermes asks the user to send it again, and the server,
which groups rows at each user row, would otherwise file it with no answer or as the
next turn's context.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

from ...messages import Record

#: Caps on what a record carries; the server renders at most 500 / 300 of these.
TOOL_RESULT_CAP = 2000
TOOL_ARGUMENTS_CAP = 1000

_ROLES = ("user", "assistant", "tool")
_TEXT_PARTS = ("text", "input_text", "output_text")
_MEMORY_CONTEXT = re.compile(r"<memory-context>.*?</memory-context>", re.DOTALL)
_SUMMARY_KEY = "_compressed_summary"
_SYNTHETIC_FLAGS = (
    "_todo_snapshot_synthetic",
    "_empty_recovery_synthetic",
    "_verification_stop_synthetic",
    "_pre_verify_synthetic",
    "_dropped_toolcall_nudge",
)
#: Hermes's failed-turn boundary (agent/turn_failure_copy.py ``FAILED_TURN_DISPLAY_KIND``)
#: and the three notices it writes as its text.
_FAILED_TURN_KIND = "failed_turn"
_FAILED_TURN_NOTICES = (
    "Your request was not processed. Send it again if you still want me to carry it out.",
    "This turn did not complete. Some actions may already have run; verify their effects "
    "before resending.",
    "An automated delivery was interrupted and was not run again. Some actions may already "
    "have run; verify their effects.",
)
_SYNTHETIC_PREFIXES = (
    "[System:",
    "[System note:",
    "[SYSTEM]",
    "[Runtime note:",
    "[CONTEXT",
    "[PRIOR CONTEXT",
    "[IMPORTANT:",
    "[Your active task list",
    "[Planning state preserved",
    "[ASYNC DELEGATION",
    "[OUT-OF-BAND",
    "[/OUT-OF-BAND",
    "Cronjob Response:",
)


def _text(content: Any) -> str:
    """A message's text: the string itself, or its text parts joined by newlines."""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = [
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") in _TEXT_PARTS
        ]
        text = "\n".join(p for p in parts if p.strip())
    else:
        return ""
    return _MEMORY_CONTEXT.sub("", text).strip()


def _arguments(value: Any) -> str:
    """Tool-call arguments as capped text: a string as sent, an object as JSON."""
    if isinstance(value, str):
        text = value
    elif value in (None, "", {}, []):
        text = ""
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except (TypeError, ValueError):
            text = str(value)
    return text[:TOOL_ARGUMENTS_CAP]


def _tool_calls(value: Any) -> list[dict[str, str]]:
    """``{"name", "arguments"}`` for each call, from OpenAI's nested shape or a flat one."""
    if not isinstance(value, list):
        return []
    calls = []
    for call in value:
        if not isinstance(call, Mapping):
            continue
        fn = call.get("function")
        spec = fn if isinstance(fn, Mapping) else call
        name = spec.get("name")
        calls.append(
            {
                "name": name if isinstance(name, str) and name else "tool",
                "arguments": _arguments(spec.get("arguments")),
            }
        )
    return calls


def _synthetic(message: Mapping[str, Any], text: str) -> bool:
    """Whether a user-role row was put there by Hermes rather than typed by the user."""
    if any(message.get(flag) for flag in _SYNTHETIC_FLAGS) or message.get("display_kind"):
        return True
    return text.startswith(_SYNTHETIC_PREFIXES)


def _record(message: Any) -> Record | None:
    """The record for one message, or None for a message that is not sent."""
    if not isinstance(message, Mapping) or message.get(_SUMMARY_KEY):
        return None
    role = message.get("role")
    if role not in _ROLES:
        return None
    text = _text(message.get("content"))
    record: Record = {"role": role, "content": text}
    if role == "user":
        if not text:
            return None
        if _synthetic(message, text):
            record["synthetic"] = True
    elif role == "assistant":
        calls = _tool_calls(message.get("tool_calls"))
        kind = message.get("display_kind")
        if kind == _FAILED_TURN_KIND or (not kind and text in _FAILED_TURN_NOTICES):
            return None
        if kind:
            if not calls:
                return None
            record["content"] = ""
        if calls:
            record["tool_calls"] = calls
    else:
        record["content"] = text[:TOOL_RESULT_CAP]
        name = message.get("tool_name") or message.get("name")
        if isinstance(name, str) and name:
            record["name"] = name
    ts = message.get("timestamp")
    if isinstance(ts, int | float | str) and not isinstance(ts, bool) and ts != "":
        record["ts"] = ts
    uid = message.get("message_uid")
    if isinstance(uid, str) and uid:
        record["id"] = uid
    return record


def to_records(messages: Iterable[Any] | None) -> list[Record]:
    """The ``/ingest`` records for Hermes's ``messages``, in order. Never modifies them,
    never raises over a malformed entry (it is skipped)."""
    if not messages:
        return []
    return [rec for message in messages if (rec := _record(message)) is not None]
