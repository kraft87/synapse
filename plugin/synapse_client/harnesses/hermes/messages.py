"""Hermes chat messages -> ``/ingest`` ``messages`` records.

Hermes hands a memory provider its whole conversation on every turn, as the live
message dicts of its own transcript (OpenAI-style, plus Hermes's bookkeeping keys).
:func:`to_records` keeps only what the server's ``messages`` parser reads and builds new
dicts; the input is never modified, since those dicts are Hermes's own.

The live provider converts the list Hermes hands it, which Hermes scrubbed of secrets
for providers; it takes only each row's ``display_kind`` from the session store, by uid
(``kinds``). The importer converts the dicts it rebuilds from the store (store.py).

What a record keeps: ``role`` (user, assistant and tool only), the text content (a
string, or the text parts of a list joined by newlines; on a user row each image is a
``[screenshot]`` too, as Hermes stores the row), an assistant row's tool calls
as ``{"name", "arguments"}``, a tool row's tool ``name``, ``ts`` from ``timestamp`` and
``id`` from ``message_uid``. Nothing else leaves: not ``api_content`` (the text Hermes
actually sent, with the memory context it injected, our own nudge included), not
``display_metadata``, reasoning, or provider-specific blobs.

- A row carrying ``_compressed_summary`` is a compaction summary or marker, or a row
  Hermes folded one into (agent/context_compressor.py ``_merge_summary_into_tail_row``):
  its text is not sent. A user row of that kind stays, as a ``synthetic`` row with no
  text; any other is dropped.
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
- A user row with no text (nothing once the memory block is gone) is sent as a
  ``synthetic`` row as well. It opens no turn, so its reply lands on the turn before;
  the row stays in the records to show that this reply answers something later than
  that turn (the core's ``messages.joined``). No user row is left out, whatever it
  holds. A picture is not such a row: it reads ``[screenshot]``, as in the store
  (agent/session_persistence.py ``_durable_content``), unless it has no text and a
  message before it still waits for its reply (a run that failed): it is then an empty
  ``synthetic`` row, as the released client sent it, so that message goes with the
  reply.
- A user row whose ``display_kind`` is one Hermes stamps on the row that opens a run it
  starts itself (_RUN_KINDS: a background task's report, a gateway event, an
  auto-continue, a hidden Desktop send, a skill run) is ``synthetic`` and ``begins``,
  with its ``kind``: the core makes it and its reply a turn of its own, so the turn
  before is not sent again. Its text is not sent: a report holds the task's output,
  which can quote files, web pages or other people. Every other kind is written inside
  a run (``steer``) or between runs and answered by none (``model_switch``,
  ``personality_switch``): synthetic, in the turn it lands in.
- A user row Hermes folded the next message into (agent/agent_runtime_helpers.py
  ``_merge_consecutive_users``: a marker, a note or a message that got no reply, then
  the next message, joined by a blank line, the first one's text kept as
  ``_merged_turn_prefix``) is sent as the two rows it was (``_split``) when the part in
  front of the message is Hermes's alone (``_hermes_wrote``): that part as an empty
  synthetic row under the row's own uid, its text left out, and the user's message under
  the last uid Hermes noted it absorbed. Otherwise the row goes whole, as the released
  client sent it: a message the user typed is never dropped where the released client
  stored it (a marker's text then goes beside it).
  Any other row that holds more than one message (rows a compression merged, which
  leave only ``_absorbed_message_uids``) is sent as it is, followed by a ``synthetic``
  row with no text, which says so to the core: ``messages.joined`` cannot read it off
  one row.
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
- A row Hermes never stores (an ephemeral scaffolding flag: the re-prompt pair after a
  dropped tool call, a recovery nudge; agent/session_persistence.py
  ``_EPHEMERAL_SCAFFOLDING_FLAGS``) is not sent either, so the live list and the store
  give the same records.
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
#: The image parts Hermes stores as ``[screenshot]`` (agent/session_persistence.py
#: ``_IMAGE_PART_TYPES``, ``_image_marker``).
_IMAGE_PARTS = ("image", "image_url", "input_image")
_IMAGE_DETAILS = ("low", "high", "auto")
#: The user-row kinds Hermes stamps on the row that opens a run it starts itself
#: (``persist_user_display_kind``, agent/turn_context.py ``_stage_turn_user_message``):
#: a background task's report (tools/process_registry_notifications.py), a gateway event
#: or heartbeat (gateway/response_filters.py ``display_kind_for_event``), an auto-continue
#: (tui_gateway/session_auto_continue.py), a hidden Desktop send (gateway/session_display.py),
#: a skill run.
_RUN_KINDS = frozenset(
    {
        "async_delegation_complete",
        "process_complete",
        "internal_notification",
        "auto_continue",
        "hidden",
        "skill_invocation",
    }
)
_MEMORY_CONTEXT = re.compile(r"<memory-context>.*?</memory-context>", re.DOTALL)
_SUMMARY_KEY = "_compressed_summary"
_PREFIX_KEY = "_merged_turn_prefix"
_MERGED_KEYS = (_PREFIX_KEY, "_absorbed_message_uids")
_SYNTHETIC_FLAGS = (
    "_todo_snapshot_synthetic",
    "_empty_recovery_synthetic",
    "_verification_stop_synthetic",
    "_pre_verify_synthetic",
    "_dropped_toolcall_nudge",
)
#: Hermes's failed-turn boundary (agent/turn_failure_copy.py ``FAILED_TURN_DISPLAY_KIND``)
#: and the three notices it writes as its text.
#: Hermes never stores a row carrying one of these (agent/session_persistence.py
#: ``_EPHEMERAL_SCAFFOLDING_FLAGS``), so neither is it sent.
_EPHEMERAL_FLAGS = (
    "_empty_recovery_synthetic",
    "_empty_terminal_sentinel",
    "_thinking_prefill",
    "_verification_stop_synthetic",
    "_pre_verify_synthetic",
    "_kanban_stop_synthetic",
    "_dropped_toolcall_nudge",
)
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


def _image_marker(part: Mapping[str, Any]) -> str:
    url = part.get("image_url")
    detail = part.get("detail") or (url.get("detail") if isinstance(url, Mapping) else None)
    return f"[screenshot detail={detail}]" if detail in _IMAGE_DETAILS else "[screenshot]"


def _user_text(content: Any) -> str:
    """A user message's text as Hermes stores it (agent/session_persistence.py
    ``_durable_content``): a list's text parts and a ``[screenshot]`` for each image,
    joined by newlines, so the live list and the store read the same row."""
    if isinstance(content, list):
        content = "\n".join(
            str(part.get("text", "")) if part.get("type") == "text" else _image_marker(part)
            for part in content
            if isinstance(part, Mapping)
            and (part.get("type") == "text" or part.get("type") in _IMAGE_PARTS)
        )
    return _text(content)


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
    if not isinstance(message, Mapping):
        return None
    role = message.get("role")
    summary = bool(message.get(_SUMMARY_KEY))
    if role not in _ROLES or (summary and role != "user"):
        return None
    if any(message.get(flag) for flag in _EPHEMERAL_FLAGS):
        return None
    if summary:
        text = ""
    elif role == "user":
        text = _user_text(message.get("content"))
    else:
        text = _text(message.get("content"))
    record: Record = {"role": role, "content": text}
    if role == "user":
        kind = message.get("display_kind")
        if not text or _synthetic(message, text):
            record["synthetic"] = True
        if kind in _RUN_KINDS and not summary:
            record["synthetic"] = True
            record["begins"] = True
            record["kind"] = kind
            record["content"] = ""
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


def _kind(message: Any, kinds: Mapping[str, str] | None) -> Any:
    """``message`` with the ``display_kind`` the store holds for its uid, when its live
    dict has none."""
    if not kinds or not isinstance(message, Mapping) or message.get("display_kind"):
        return message
    kind = kinds.get(str(message.get("message_uid") or ""))
    return {**message, "display_kind": kind} if kind else message


def _hermes_wrote(head: Mapping[str, Any], row: Mapping[str, Any], kinds: Any) -> bool:
    """Whether the part of a folded row in front of the message is Hermes's alone: its
    text reads as Hermes's own (the released client sent such a row as synthetic, the
    message in it included, so splitting it loses nothing it stored), or the store holds
    a ``display_kind`` for the row's uid and the row absorbed only the message (the
    part in front is that one row). A part that may hold a message the user typed (a
    longer fold whose text does not read as Hermes's) is not split: the row goes whole,
    as the released client sent it."""
    record = _record(head)
    if record is not None and record.get("synthetic"):
        return True
    absorbed = row.get("_absorbed_message_uids")
    alone = isinstance(absorbed, list) and len(absorbed) == 1
    uid = row.get("message_uid")
    return alone and bool(kinds) and isinstance(uid, str) and bool(kinds.get(uid))


def _split(message: Mapping[str, Any]) -> list[Mapping[str, Any]] | None:
    """A user row Hermes folded the next message into, as the two rows it was: the text
    that got no reply (``_merged_turn_prefix``, under the row's own uid) and the message
    after it (under the uid Hermes noted it absorbed, when it noted one). Hermes joins
    them with a blank line and splits them the same way (agent/agent_runtime_helpers.py
    ``_merge_consecutive_users``, agent/session_persistence.py
    ``_content_with_turn_override``). None when the row is not such a row or its text no
    longer starts with the prefix: it is then sent as it is."""
    prefix = message.get(_PREFIX_KEY)
    content = message.get("content")
    if message.get("role") != "user" or not isinstance(prefix, str) or not prefix:
        return None
    if not isinstance(content, str) or not content.startswith(prefix + "\n\n"):
        return None
    head = {k: v for k, v in message.items() if k not in _MERGED_KEYS}
    head["content"] = prefix
    rest = {k: v for k, v in head.items() if k not in ("message_uid", "display_kind")}
    rest["content"] = content[len(prefix) + 2 :]
    absorbed = message.get("_absorbed_message_uids")
    if isinstance(absorbed, list) and absorbed and isinstance(absorbed[-1], str):
        rest["message_uid"] = absorbed[-1]
    return [head, rest]


_PICTURE = re.compile(r"\[screenshot(?: detail=\w+)?\]")


def _only_pictures(record: Record) -> bool:
    """A real user row whose text is pictures only (``[screenshot]`` lines)."""
    lines = str(record.get("content") or "").split("\n")
    return (
        record["role"] == "user"
        and not record.get("synthetic")
        and any(lines)
        and all(_PICTURE.fullmatch(line) for line in lines)
    )


def to_records(
    messages: Iterable[Any] | None, kinds: Mapping[str, str] | None = None
) -> list[Record]:
    """The ``/ingest`` records for Hermes's ``messages``, in order. ``kinds``: the
    ``display_kind`` the session store holds for a row, by ``message_uid``, for a row
    whose live dict has lost it (Hermes rebuilt the agent, or merged the row). Never
    modifies the messages, never raises over a malformed entry (it is skipped)."""
    records: list[Record] = []
    asking = False  # a real user row is waiting for its reply
    for message in messages or ():
        parts = _split(message) if isinstance(message, Mapping) else None
        if parts is not None and (asking or not _hermes_wrote(parts[0], message, kinds)):
            # The row goes whole, as the released client sent it: the part in front may
            # be the user's, or a message before it is still waiting for its reply (a
            # run that failed), which the released client stored with this reply.
            parts = None
        for index, one in enumerate(parts or (message,)):
            one = _kind(one, kinds)
            record = _record(one)
            if record is None:
                continue
            if parts is not None and index == 0:
                # What Hermes folded in front of the message (a marker, a note, and any
                # message that got no reply before them) is not sent: the row stays,
                # empty, to say a message landed here.
                record = {k: v for k, v in record.items() if k not in ("begins", "kind")}
                record.update(content="", synthetic=True)
            if asking and _only_pictures(record):
                # A picture with no text after a message still waiting for its reply (a
                # run that failed): the released client sent it as an empty synthetic
                # row, so the message went with the reply to the picture. So here.
                record = {**record, "content": "", "synthetic": True}
            records.append(record)
            if record["role"] == "user" and not record.get("synthetic") and record["content"]:
                asking = True
            elif (
                record["role"] == "assistant" and record["content"] and not record.get("tool_calls")
            ):
                asking = False
            if (
                parts is None
                and record["role"] == "user"
                and not record.get("synthetic")
                and any(one.get(key) for key in _MERGED_KEYS)
            ):
                records.append({"role": "user", "content": "", "synthetic": True})
    return records
