"""Serialization, provenance, and identifier parsing for recall responses."""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime, time
from typing import Any

from ingestion.localtime import local_date
from mcp_server.recall_query import bm25_best_window, bm25_tokenize

_ROLE_MARKER_RE = re.compile(
    r"(?m)^\[(user|attachments|assistant|context|result|title|tool:[^\]\n]{0,80})\]"
)
_USER_MARKERS = {"user", "attachments"}
_FETCH_MAX = 20

#: Characters of whole turns one recall_full_turns / fetch call serves. Claude Code cuts an
#: MCP result off past MAX_MCP_OUTPUT_TOKENS (25K tokens by default) and a tool-heavy turn
#: runs to 100K+ characters (the median is 4-5K), so a few of them unbudgeted came back
#: truncated at a point the client chose, or filled the caller's context with tool traces.
#: About 10K tokens: a default call of five ordinary turns is untouched.
FULL_TURN_BUDGET = 40_000
#: A turn that does not fit is served as its first _EXCERPT_HEAD characters plus the
#: _EXCERPT_WINDOW characters that best match the query.
_EXCERPT_HEAD = 1_000
_EXCERPT_WINDOW = 2_000


def fact_day(ts: Any) -> str:
    """The calendar day a fact's ``t_valid`` / ``last_supported_at`` is served under.

    An instant is the user's day (``local_date``): a fact said at 20:30 EDT is today's, not
    tomorrow's. A value at exactly 00:00:00 UTC is a date the extractor read with no time of
    day (written that way before date-only values were stored at noon UTC, and until
    scripts/restamp_date_only_dates.py has run), so it keeps its own date: moved into the
    user's zone it would show the day before."""
    if isinstance(ts, str) and len(ts) > 10:
        try:
            ts = datetime.fromisoformat(ts)
        except ValueError:
            return ts[:10]
    if isinstance(ts, datetime):
        utc = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)
        if utc.time() == time(0):
            return utc.date().isoformat()
    return local_date(ts)


def _excerpt(content: str, query_tokens: list[str]) -> str:
    """The opening of a turn plus, when the query matches further in, the stretch around the match."""
    head = content[:_EXCERPT_HEAD]
    if not query_tokens:
        return head
    window = bm25_best_window(content, query_tokens, _EXCERPT_WINDOW)
    start = content.find(window)
    if start <= len(head):  # the match sits in or against the opening: one contiguous stretch
        return content[: start + len(window)]
    return f"{head}\n…\n{window}"


def fit_turns_to_budget(
    items: list[dict[str, Any]],
    query: str | None,
    *,
    budget: int = FULL_TURN_BUDGET,
    keep_first: bool = False,
) -> int:
    """Cap the whole turns of one call at ``budget`` characters; returns how many were cut.

    Turns are served whole in rank order while they fit. One that does not is replaced, in
    place, by an excerpt (see ``_excerpt``; just the opening when there is no ``query``) and
    gains ``full_chars``, the length ``fetch`` would return it at. A later, smaller turn can
    still fit what is left. ``keep_first`` never cuts the first item: fetch(ids) is the only
    way to read a turn whole, so the one asked for first comes back whole whatever its size."""
    tokens = bm25_tokenize(query) if query else []
    used = 0
    n_cut = 0
    for index, item in enumerate(items):
        content = item.get("content") or ""
        if (keep_first and index == 0) or used + len(content) <= budget:
            used += len(content)
            continue
        item["content"] = _excerpt(content, tokens)
        item["full_chars"] = len(content)
        used += len(item["content"])
        n_cut += 1
    return n_cut


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
        out["date"] = local_date(created_at)  # the user's day, not the UTC one
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


# Episode-validity overlay: a successor attaches to a served passage only when the passage
# states the retired claim, i.e. contains at least this share of the claim's content words.
# A fact extracted from a multi-turn chunk cites every episode in the chunk (4-31 of them),
# so the citation alone put unrelated successors on unrelated passages: a piano fact on a
# restic setup turn. Measured on prod 2026-10-09: the passage best matching a single-source
# claim scored 0.25-1.0 (most >= 0.4); a third of chunk-citation attachments scored < 0.3.
OVERLAY_MIN_SUPPORT = 0.3

_CLAIM_STOP = frozenset(
    "a an the and or of to in on for with by at from as is are was were be been being it its "
    "this that these those has have had not no but via into over under than then so if when "
    "while also only just about after before during kyle user kyle's user's meaning".split()
)
_CLAIM_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.\-/@:]*[a-z0-9]|[a-z0-9]")
_CLAIM_PART_RE = re.compile(r"[/._\-:@]+")
_CLAIM_MEANING_RE = re.compile(r"\(meaning [^)]*\)")
_CLAIM_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")


def _claim_norm(text: str) -> str:
    return _CLAIM_THOUSANDS_RE.sub("", _CLAIM_MEANING_RE.sub(" ", text.lower()))


def claim_support(claim: str, text: str) -> float:
    """Share of ``claim``'s content words (3+ chars, not stopwords) that ``text`` contains.

    A dotted/slashed token (a path, host or file name) also counts when all of its parts
    appear. Extractor annotations like "(meaning 2026-10-09)" and thousands separators are
    ignored. A claim with no content words returns 1.0: nothing to check against."""
    want = {
        t
        for t in _CLAIM_TOKEN_RE.findall(_claim_norm(claim))
        if len(t) >= 3 and t not in _CLAIM_STOP
    }
    if not want:
        return 1.0
    have: set[str] = set()
    for t in _CLAIM_TOKEN_RE.findall(_claim_norm(text)):
        have.add(t)
        have.update(part for part in _CLAIM_PART_RE.split(t) if part)
    hits = sum(
        1
        for t in want
        if t in have or all(part in have for part in _CLAIM_PART_RE.split(t) if part)
    )
    return hits / len(want)


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
