"""Serialization, provenance, and identifier parsing for recall responses."""

from __future__ import annotations

import math
import re
from typing import Any

_ROLE_MARKER_RE = re.compile(
    r"(?m)^\[(user|attachments|assistant|context|result|title|tool:[^\]\n]{0,80})\]"
)
_USER_MARKERS = {"user", "attachments"}
_FETCH_MAX = 20


def to_web_recall_item(row: dict[str, Any]) -> dict[str, Any]:
    """Serialize a web chunk in the compact, feedback-safe recall shape."""
    out: dict[str, Any] = {}
    if (record_id := row.get("id")) is not None:
        # Web rows already carry the served "w:N" form (assigned at query time in
        # _search_web); re-wrapping would yield "w:w:N" and fail recall_feedback's validator.
        out["id"] = str(record_id) if str(record_id).startswith("w:") else f"w:{record_id}"
    if context := (row.get("context_prefix") or "").strip():
        out["context"] = context
    elif excerpt := (row.get("content") or "")[:200].strip():
        out["excerpt"] = excerpt
    if url := row.get("url"):
        out["url"] = url
    if title := row.get("title"):
        out["title"] = title
    if (created_at := row.get("created_at")) is not None:
        out["date"] = str(created_at)[:10]
    return out


def to_recall_item(row: dict[str, Any]) -> dict[str, Any]:
    """Serialize an episode row into the minimal caller-facing recall shape."""
    out: dict[str, Any] = {}
    if (record_id := row.get("id")) is not None:
        out["id"] = record_id
    out["content"] = row.get("content", "")
    if (project := row.get("project")) is not None:
        out["project"] = project
    if (created_at := row.get("created_at")) is not None:
        out["date"] = str(created_at)[:10]
    if (session_id := row.get("session_id")) is not None:
        out["session"] = session_id
    return out


def role_spans(content: str) -> list[tuple[int, str]]:
    """Return role-marker spans for user and assistant provenance attribution."""
    spans: list[tuple[int, str]] = []
    for match in _ROLE_MARKER_RE.finditer(content):
        tag = match.group(1)
        side = "" if tag == "title" else ("user" if tag in _USER_MARKERS else "assistant")
        spans.append((match.start(), side))
    return spans


def passage_role(spans: list[tuple[int, str]], start: int, end: int) -> str | None:
    """Return the source role for a passage, or ``mixed`` when it spans both roles."""
    if not spans or end <= start:
        return None
    seen: set[str] = set()
    for index, (offset, side) in enumerate(spans):
        next_offset = spans[index + 1][0] if index + 1 < len(spans) else math.inf
        if offset < end and next_offset > start and side:
            seen.add(side)
    if not seen:
        return None
    return seen.pop() if len(seen) == 1 else "mixed"


def parse_episode_ids(ids: list[Any]) -> list[int]:
    """Parse and cap the episode identifiers accepted by fetch()."""
    parsed: list[int] = []
    seen: set[int] = set()
    for value in ids:
        identifier: int | None = None
        if isinstance(value, int) and not isinstance(value, bool):
            identifier = value
        elif isinstance(value, str):
            candidate = value.split(":", 1)[1] if value.startswith("e:") else value
            if candidate.isdigit():
                identifier = int(candidate)
        if identifier is not None and identifier not in seen:
            seen.add(identifier)
            parsed.append(identifier)
    return parsed[:_FETCH_MAX]


def parse_fetch_ids(ids: list[Any]) -> tuple[list[int], list[int], list[str], list[str]]:
    """Parse mixed episode/note fetch identifiers, preserving request order."""
    episodes: list[int] = []
    notes: list[int] = []
    skipped: list[str] = []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in ids:
        kind: str | None = None
        identifier: int | None = None
        if isinstance(value, int) and not isinstance(value, bool):
            kind, identifier = "e", value
        elif isinstance(value, str):
            raw = value.strip()
            prefix, separator, rest = raw.partition(":")
            if not separator:
                prefix, rest = "e", raw
            if prefix in ("e", "n") and rest.isdigit():
                kind, identifier = prefix, int(rest)
        if kind is None or identifier is None:
            skipped.append(str(value))
            continue
        key = f"{kind}:{identifier}"
        if key in seen or len(normalized) >= _FETCH_MAX:
            continue
        seen.add(key)
        normalized.append(key)
        (episodes if kind == "e" else notes).append(identifier)
    return episodes, notes, skipped, normalized


def apply_supersessions(
    items: list[dict[str, Any]], supersessions: dict[int, list[str]], served_facts: set[str]
) -> None:
    """Attach unserved current facts that supersede claims in served episodes."""
    for item in items:
        record_id = item.get("id")
        if not isinstance(record_id, str) or not record_id.startswith("e:"):
            continue
        raw_id = record_id.split(":", 1)[1]
        if not raw_id.isdigit():
            continue
        successors = [
            fact for fact in supersessions.get(int(raw_id), []) if fact and fact not in served_facts
        ]
        if successors:
            item["superseded_by"] = successors
