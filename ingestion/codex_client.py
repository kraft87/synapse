"""Parse OpenAI Codex CLI rollout files into Episodes.

Codex CLI writes one JSONL per session at
~/.codex/sessions/YYYY/MM/DD/rollout-<timestamp>-<session_uuid>.jsonl

Every line is {"timestamp", "ordinal", "type", "payload"}. Record types:
- "session_meta"  — once, first line: session_id, cwd, cli_version, originator.
- "response_item" — the conversation: payload.type in "message" (role
  developer/user/assistant, content blocks input_text/output_text),
  "reasoning" (encrypted, unusable), "custom_tool_call",
  "custom_tool_call_output".
- "turn_context"  — per user turn: turn_id, cwd, approval/sandbox policy.
- "event_msg" / "world_state" — lifecycle telemetry, skipped.

Records are grouped into user-turn exchanges the same way the Claude Code
parser does: a new group starts at each real user message. Codex also carries
an explicit turn id (payload.internal_chat_message_metadata_passthrough
.turn_id), but the user-boundary rule matches it and also covers records
where the passthrough is absent.

Role labels need care: "developer" messages are injected instructions, and
the first "user" message of a session is an <environment_context> XML block,
not the human. Both are machinery, filtered like Claude Code's
<system-reminder> family.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import orjson

from ingestion.models import Episode

logger = logging.getLogger(__name__)


# User-role message prefixes that are Codex harness machinery, not human input.
_MACHINERY_PREFIXES = (
    "<environment_context>",
    "<user_instructions>",
    "<turn_context>",
    "<permissions",
    "<skills_instructions>",
    "<multi_agent",
    "<collaboration_mode>",
    "<system-reminder>",
)

# rollout-2026-08-13T20-49-40-<uuid>.jsonl — the trailing uuid is the session id.
_ROLLOUT_NAME = re.compile(
    r"rollout-.*-([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\.jsonl$"
)

# response_item payload types that carry conversation content. "reasoning" is
# deliberately absent: Codex persists it as encrypted_content ciphertext.
_ITEM_TYPES = (
    "message",
    "custom_tool_call",
    "custom_tool_call_output",
    "function_call",
    "function_call_output",
    "local_shell_call",
    "web_search_call",
)


def _is_machinery_text(text: str) -> bool:
    return text.lstrip().startswith(_MACHINERY_PREFIXES)


def _strip_nulls(text: str) -> str:
    """Postgres TEXT columns reject NUL bytes — strip them defensively."""
    return text.replace("\x00", "") if "\x00" in text else text


def _content_text(content: Any) -> str:
    """Join the text of a Codex content-block list (input_text/output_text)."""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") in ("input_text", "output_text", "text"):
            t = str(block.get("text") or "").strip()
            if t:
                parts.append(t)
    return "\n".join(parts).strip()


def _tool_call_line(payload: dict[str, Any]) -> str | None:
    """Render a tool-call response_item as a compact [tool:name] line."""
    ptype = payload.get("type")
    if ptype == "custom_tool_call":
        name = payload.get("name", "tool")
        detail = str(payload.get("input") or "")
    elif ptype == "function_call":
        name = payload.get("name", "function")
        detail = str(payload.get("arguments") or "")
    elif ptype == "local_shell_call":
        name = "shell"
        action = payload.get("action") or {}
        cmd = action.get("command")
        detail = " ".join(cmd) if isinstance(cmd, list) else str(cmd or "")
    elif ptype == "web_search_call":
        name = "web_search"
        action = payload.get("action") or {}
        detail = str(action.get("query") or "")
    else:
        return None
    detail = detail.strip()
    if not detail:
        return None
    return f"[tool:{name}] {detail[:300]}"


def _tool_output_text(payload: dict[str, Any]) -> str:
    return _content_text(payload.get("output"))


def _is_user_turn(record: dict[str, Any]) -> bool:
    """True if this response_item is a fresh human input (not machinery,
    not an injected developer message)."""
    payload = record.get("payload") or {}
    if payload.get("type") != "message" or payload.get("role") != "user":
        return False
    text = _content_text(payload.get("content"))
    return bool(text) and not _is_machinery_text(text)


def _cwd_to_project(cwd: str | None) -> str | None:
    if not cwd:
        return None
    return cwd.rstrip("/").rsplit("/", 1)[-1] or None


def session_id_from_path(path: Path | str) -> str | None:
    m = _ROLLOUT_NAME.search(str(path))
    return m.group(1) if m else None


class CodexRolloutParser:
    """Turns a Codex CLI rollout .jsonl into a list of Episodes."""

    def parse_file(self, path: Path, project_override: str | None = None) -> list[Episode]:
        raw: list[dict[str, Any]] = []
        with open(path, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                if isinstance(rec, dict):
                    raw.append(rec)
        return self.parse_records(
            raw,
            str(path),
            project_override,
            session_id_hint=session_id_from_path(path),
        )

    def parse_records(
        self,
        records: list[dict[str, Any]],
        source_label: str,
        project_override: str | None = None,
        session_id_hint: str | None = None,
    ) -> list[Episode]:
        """Shared seam for the disk sweep and the /ingest push, mirroring
        JSONLParser.parse_records: same records -> same span_ids, so both
        paths converge idempotently.

        A pushed TAIL usually lacks the session_meta line, so push callers
        must supply ``session_id_hint`` (the hook knows it from the rollout
        filename); session_meta wins when present.
        """
        session_id = session_id_hint
        cwd: str | None = None
        model: str | None = None
        cli_version: str | None = None
        originator: str | None = None

        items: list[dict[str, Any]] = []
        for rec in records:
            if not isinstance(rec, dict):
                continue
            rtype = rec.get("type")
            payload = rec.get("payload")
            if not isinstance(payload, dict):
                continue
            if rtype == "session_meta":
                session_id = payload.get("session_id") or payload.get("id") or session_id
                cwd = payload.get("cwd") or cwd
                cli_version = payload.get("cli_version")
                originator = payload.get("originator")
                continue
            if rtype == "turn_context":
                cwd = payload.get("cwd") or cwd
                model = payload.get("model") or model
                continue
            if rtype != "response_item":
                continue
            ptype = payload.get("type")
            if ptype not in _ITEM_TYPES:
                continue  # reasoning (encrypted) and unknown types
            if ptype == "message":
                role = payload.get("role")
                if role == "developer":
                    continue  # injected instructions
                if role == "user":
                    text = _content_text(payload.get("content"))
                    if not text or _is_machinery_text(text):
                        continue  # <environment_context> et al.
            items.append(rec)

        if not session_id or not items:
            return []

        groups: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for rec in items:
            if _is_user_turn(rec) and current:
                groups.append(current)
                current = [rec]
            else:
                current.append(rec)
        if current:
            groups.append(current)

        episodes: list[Episode] = []
        prev_assistant: str | None = None
        seq = 0
        seen_span_ids: set[str] = set()

        for group in groups:
            if not group:
                continue
            seq += 1

            content_parts: list[str] = []
            human_turn: str | None = None
            assistant_turn: str | None = None
            last_item_id: str | None = None
            ts: str | None = None

            if prev_assistant:
                content_parts.append(f"[context] {prev_assistant.strip()[:300]}")

            for rec in group:
                ts = ts or rec.get("timestamp")
                payload = rec.get("payload") or {}
                last_item_id = payload.get("id") or last_item_id
                ptype = payload.get("type")

                if ptype == "message":
                    text = _content_text(payload.get("content"))
                    if not text:
                        continue
                    if payload.get("role") == "user":
                        if human_turn is None:
                            human_turn = text[:3000]
                            content_parts.append(f"[user] {human_turn}")
                    else:
                        assistant_turn = text
                        content_parts.append(f"[assistant] {text[:3000]}")
                elif ptype in ("custom_tool_call_output", "function_call_output"):
                    txt = _tool_output_text(payload)
                    if len(txt) > 20:
                        content_parts.append(f"[result] {txt[:500]}")
                else:
                    line = _tool_call_line(payload)
                    if line:
                        content_parts.append(line)

            if not content_parts:
                continue

            span_id = f"codex:{last_item_id}" if last_item_id else None
            if span_id is not None:
                if span_id in seen_span_ids:
                    continue  # replayed item (resume/fork re-dump) — already emitted
                seen_span_ids.add(span_id)

            try:
                created_at = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None
            except (ValueError, TypeError):
                created_at = None

            episodes.append(
                Episode(
                    session_id=session_id,
                    sequence=seq,
                    project=project_override or _cwd_to_project(cwd),
                    platform="codex",
                    model=model,
                    human_turn=_strip_nulls(human_turn) if human_turn else None,
                    assistant_turn=_strip_nulls(assistant_turn) if assistant_turn else None,
                    content=_strip_nulls("\n\n".join(content_parts)),
                    span_id=span_id,
                    source="codex",
                    metadata={
                        "rollout_path": source_label,
                        "ts": ts,
                        "cwd": cwd,
                        "cli_version": cli_version,
                        "originator": originator,
                    },
                    created_at=created_at,
                )
            )
            prev_assistant = assistant_turn

        return episodes
