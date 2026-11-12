"""Parse a client's list of chat messages into Episodes (``/ingest`` format "messages").

Claude Code and Codex CLI write a transcript file per session, and each has a parser
for its own record shapes. A client that holds its conversation in memory as
OpenAI-style chat messages (the Hermes agent, and any later one) posts that list
instead, and the body names what a message list does not carry itself:

    {"format": "messages", "client": "hermes", "session_id": "...",
     "project": optional label, "cwd": optional path, "model": optional default,
     "source": optional cosmetic label, "records": [<message>, ...]}

``client`` becomes ``Episode.platform`` and the span-id prefix. Each message is a dict:
- "role"        — "user", "assistant" or "tool". Any other role (system, developer,
  ...) is skipped.
- "content"     — a string, or a list of parts of which only "text", "input_text"
  and "output_text" parts are read (their "text"); images and the like are skipped.
- "tool_calls"  — assistant rows: OpenAI's {"function": {"name", "arguments"}} or a
  flat {"name", "arguments"}, arguments a string or a JSON-able object.
- "name" / "tool_name" — tool rows. Accepted, not rendered: [result] carries no
  tool name in any parser.
- "id"          — optional stable per-message id.
- "ts"          — optional epoch seconds or ISO-8601 string.
- "model"       — optional; overrides the batch default for its turn.
- "synthetic"   — optional; a row the harness injected in the user's place. Skipped
  outright: it neither opens a turn nor renders.

User text loses every <memory-context>...</memory-context> block before anything
else: that is how a client injects recalled memory into the prompt, and it must
never be stored as something the user said. A user row empty without it is dropped.

Records are grouped into user-turn exchanges and rendered exactly as
CodexRolloutParser does (a new group at each user row; the same [context] / [user] /
[tool:name] / [result] / [assistant] lines and truncation), so the three parsers'
episodes read alike. A batch that begins mid-turn yields a leading group with no
[user] line, as it does there.

Identity is the span_id, which the route dedups on: "<client>:<id>" of the group's
last message carrying an id, so overlapping pushes converge. A client without ids
gets "<client>:<session_id>:<hash>" over the group's own rendered lines, the
[context] line excluded, so a turn hashes the same in whichever batch carries it.
That only converges when every batch starts at a user row: a batch that begins
mid-turn renders a fragment, which hashes as a turn of its own.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import UTC, datetime
from typing import Any, NamedTuple

from ingestion.models import Episode
from ingestion.project_label import label_from_path

logger = logging.getLogger(__name__)


#: A client name. Matched with ``fullmatch``: ``$`` alone also accepts a trailing newline.
CLIENT_PATTERN = r"^[a-z][a-z0-9_]{0,31}$"
_CLIENT_RE = re.compile(CLIENT_PATTERN)

#: Names a messages client may not take: the platforms the other parsers stamp. Platform
#: is provenance other lanes select on (the skills lane mines ``platform = 'claude_code'``),
#: and every parser's span ids start with its platform (schema 068), so a client by one of
#: these names could also claim span ids that parser will write.
RESERVED_CLIENTS: frozenset[str] = frozenset(
    {"claude_code", "codex", "cursor", "chatgpt", "claude_ai"}
)

_ROLES = ("user", "assistant", "tool")
_TEXT_PARTS = ("text", "input_text", "output_text")
_MEMORY_CONTEXT = re.compile(r"<memory-context>.*?</memory-context>", re.DOTALL)


class _Row(NamedTuple):
    """One kept message, reduced to what grouping and rendering read."""

    role: str
    text: str
    tool_lines: list[str]
    msg_id: str | None
    ts: datetime | None
    model: str | None


def valid_client(name: object) -> bool:
    """True if ``name`` may label a messages-format client: it matches
    :data:`CLIENT_PATTERN` and is not in :data:`RESERVED_CLIENTS`."""
    return (
        isinstance(name, str)
        and _CLIENT_RE.fullmatch(name) is not None
        and name not in RESERVED_CLIENTS
    )


def _strip_nulls(text: str) -> str:
    """Postgres TEXT columns reject NUL bytes — strip them defensively."""
    return text.replace("\x00", "") if "\x00" in text else text


def _content_text(content: Any) -> str:
    """A message's text: the string itself, or its text parts joined by newlines."""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") not in _TEXT_PARTS:
            continue
        t = str(part.get("text") or "").strip()
        if t:
            parts.append(t)
    return "\n".join(parts).strip()


def _arguments_text(args: Any) -> str:
    """Tool-call arguments as text: a string as sent, an object as sorted-key JSON."""
    if isinstance(args, str):
        return args.strip()
    if not args:
        return ""
    try:
        return json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):  # mixed-type keys, a cycle
        return str(args)


def _tool_call_lines(tool_calls: Any) -> list[str]:
    """Compact [tool:name] lines for an assistant row's tool calls, either shape."""
    if not isinstance(tool_calls, list):
        return []
    lines = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function")
        spec = fn if isinstance(fn, dict) else call
        name = spec.get("name")
        if not isinstance(name, str) or not name:
            name = "tool"
        detail = _arguments_text(spec.get("arguments"))
        if detail:
            lines.append(f"[tool:{name}] {detail[:300]}")
    return lines


def _parse_ts(value: Any) -> datetime | None:
    """A message's ``ts`` as a datetime, or None when absent or unusable.

    Epoch seconds become a timezone-aware UTC datetime; a string is read as ISO-8601
    the way the other parsers read theirs. A bad value is ignored, never an error."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        try:
            return datetime.fromtimestamp(value, tz=UTC)
        except (OverflowError, OSError, ValueError):  # NaN, inf, out of range
            return None
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _row(rec: Any) -> _Row | None:
    """The kept form of one record, or None for a record that is skipped."""
    if not isinstance(rec, dict) or rec.get("synthetic"):
        return None
    role = rec.get("role")
    if role not in _ROLES:
        return None
    text = _content_text(rec.get("content"))
    tool_lines: list[str] = []
    if role == "user":
        text = _MEMORY_CONTEXT.sub("", text).strip()
        if not text:
            return None
    elif role == "assistant":
        tool_lines = _tool_call_lines(rec.get("tool_calls"))
    msg_id = rec.get("id")
    model = rec.get("model")
    return _Row(
        role=role,
        text=text,
        tool_lines=tool_lines,
        msg_id=msg_id if isinstance(msg_id, str) and msg_id else None,
        ts=_parse_ts(rec.get("ts")),
        model=model if isinstance(model, str) and model else None,
    )


class MessagesParser:
    """Turns one session's chat messages into a list of Episodes."""

    def parse_records(
        self,
        records: list[dict[str, Any]],
        source_label: str,
        project_override: str | None = None,
        *,
        session_id: str | None,
        client: str,
        cwd: str | None = None,
        model: str | None = None,
    ) -> list[Episode]:
        """The /ingest seam for ``format: "messages"``.

        ``client`` must pass :func:`valid_client` (the route answers 400 before it gets
        here); anything else is a caller bug and raises ``ValueError``. ``cwd`` labels
        the turns when ``project_override`` is None and is kept in metadata; ``model``
        is the default for turns whose messages name none. ``source_label`` is cosmetic.
        """
        if not valid_client(client):
            raise ValueError(f"invalid messages client {client!r}")
        cwd = cwd if isinstance(cwd, str) and cwd else None
        model = model if isinstance(model, str) and model else None
        if not session_id:
            return []

        rows = [row for rec in records if (row := _row(rec)) is not None]

        groups: list[list[_Row]] = []
        current: list[_Row] = []
        for row in rows:
            if row.role == "user" and current:
                groups.append(current)
                current = [row]
            else:
                current.append(row)
        if current:
            groups.append(current)

        episodes: list[Episode] = []
        prev_assistant: str | None = None
        seq = 0
        seen_span_ids: set[str] = set()

        for group in groups:
            seq += 1

            parts: list[str] = []  # this group's own lines; [context] is added below
            human_turn: str | None = None
            assistant_turn: str | None = None
            last_id: str | None = None
            created_at: datetime | None = None
            turn_model: str | None = None

            for row in group:
                last_id = row.msg_id or last_id
                created_at = created_at or row.ts
                turn_model = row.model or turn_model
                if row.role == "user":
                    if human_turn is None:
                        human_turn = row.text[:3000]
                        parts.append(f"[user] {human_turn}")
                elif row.role == "assistant":
                    if row.text:
                        assistant_turn = row.text
                        parts.append(f"[assistant] {row.text[:3000]}")
                    parts.extend(row.tool_lines)
                elif len(row.text) > 20:
                    parts.append(f"[result] {row.text[:500]}")

            if not parts:
                continue

            if last_id:
                span_id = f"{client}:{last_id}"
            else:
                rendered = "\n\n".join(parts).encode("utf-8", "surrogatepass")
                span_id = f"{client}:{session_id}:{hashlib.sha256(rendered).hexdigest()[:24]}"
            if span_id in seen_span_ids:
                continue  # replayed turn — already emitted
            seen_span_ids.add(span_id)

            if prev_assistant:
                parts.insert(0, f"[context] {prev_assistant.strip()[:300]}")

            episodes.append(
                Episode(
                    session_id=session_id,
                    sequence=seq,
                    project=project_override or label_from_path(cwd),
                    platform=client,
                    model=turn_model or model,
                    human_turn=_strip_nulls(human_turn) if human_turn else None,
                    assistant_turn=_strip_nulls(assistant_turn) if assistant_turn else None,
                    content=_strip_nulls("\n\n".join(parts)),
                    span_id=span_id,
                    source="messages",
                    metadata={
                        "source_label": source_label,
                        "ts": created_at.isoformat() if created_at else None,
                        "cwd": cwd,
                        "client": client,
                    },
                    created_at=created_at,
                )
            )
            prev_assistant = assistant_turn

        return episodes
