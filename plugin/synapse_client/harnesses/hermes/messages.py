"""Hermes chat messages -> ``/ingest`` ``messages`` records, and the turns among them.

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
- Tool results are capped at 2000 characters and tool-call arguments at 1000; the
  server renders less of either. User and assistant text are sent whole.

A turn starts at a real user row (not synthetic, not empty) and runs to the next one.
It is complete when its last record that is not synthetic is an assistant row with text
and no tool calls: an assistant row that calls tools is followed by their results, so a
turn ending in one is still running (Hermes's list is live and may be read mid-turn).
Only complete turns are uploaded. Synthetic rows do not count towards a turn's key
either: Hermes adds them after a finished turn (a task-list snapshot after compression),
and that must neither reopen the turn nor make it look new.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any

#: Caps on what a record carries; the server renders at most 500 / 300 of these.
TOOL_RESULT_CAP = 2000
TOOL_ARGUMENTS_CAP = 1000

Record = dict[str, Any]
Turn = list[Record]

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


def _opens_turn(record: Record) -> bool:
    return record["role"] == "user" and not record.get("synthetic") and bool(record["content"])


def turns(records: Sequence[Record]) -> list[Turn]:
    """``records`` cut into turns, each starting at a real user row. Records before the
    first real user row belong to no turn and are left out."""
    out: list[Turn] = []
    for record in records:
        if _opens_turn(record):
            out.append([record])
        elif out:
            out[-1].append(record)
    return out


def _real(turn: Sequence[Record]) -> list[Record]:
    return [record for record in turn if not record.get("synthetic")]


def complete(turn: Sequence[Record]) -> bool:
    """Whether ``turn`` has finished: its last record that is not synthetic is an
    assistant row with text and no tool calls."""
    real = _real(turn)
    if not real:
        return False
    last = real[-1]
    return last["role"] == "assistant" and bool(last["content"]) and not last.get("tool_calls")


def turn_key(turn: Sequence[Record]) -> str:
    """A stable key for ``turn``: the ``id`` of its last record that is not synthetic,
    else a hash of those records."""
    real = _real(turn)
    last_id = real[-1].get("id") if real else None
    if isinstance(last_id, str) and last_id:
        return last_id
    blob = json.dumps(real, ensure_ascii=False, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8", "surrogatepass")).hexdigest()[:32]


def pending(
    all_turns: Sequence[Turn],
    sent: Collection[str],
    *,
    lookback: int | None = None,
) -> list[tuple[Turn | None, Turn]]:
    """The complete turns whose key is not in ``sent``, oldest first, each paired with
    the complete turn before it (its context, or None). With ``lookback``, only the
    last ``lookback`` complete turns are considered."""
    done = [turn for turn in all_turns if complete(turn)]
    start = max(0, len(done) - lookback) if lookback is not None else 0
    return [
        (done[i - 1] if i > 0 else None, done[i])
        for i in range(start, len(done))
        if turn_key(done[i]) not in sent
    ]


def batches(
    selected: Sequence[tuple[Turn | None, Turn]],
    *,
    max_turns: int = 20,
    max_bytes: int = 256_000,
) -> list[tuple[list[Record], list[Turn]]]:
    """Group ``(context, turn)`` pairs into upload batches of at most ``max_turns`` new
    turns and roughly ``max_bytes`` of records. ``context`` is the turn before ``turn``
    (or None): it leads each batch so the server's ``[context]`` line is right, and is
    not counted. Returns ``(records, new_turns)`` per batch. A single turn larger than
    ``max_bytes`` still goes, alone."""
    out: list[tuple[list[Record], list[Turn]]] = []
    records: list[Record] = []
    new: list[Turn] = []
    size = 0
    for context, turn in selected:
        turn_size = _size(turn)
        if new and (len(new) >= max_turns or size + turn_size > max_bytes):
            out.append((records, new))
            records, new, size = [], [], 0
        if not new and context:
            records.extend(context)
            size += _size(context)
        records.extend(turn)
        new.append(turn)
        size += turn_size
    if new:
        out.append((records, new))
    return out


def _size(turn: Sequence[Record]) -> int:
    return len(json.dumps(list(turn), ensure_ascii=False, default=str).encode("utf-8", "replace"))
