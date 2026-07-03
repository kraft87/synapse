"""
Write web_extractor outputs into Postgres web_artifacts.

The writer is the persistence half of the web-capture pipeline; the parser
half lives in ``ingestion.web_extractors``. Together they walk a JSONL,
locate tool_use/tool_result pairs for web tools, and produce one
``web_artifacts`` row per pair.

Idempotency is by ``tool_use_id`` (UNIQUE in the schema). Re-runs over the
same JSONL produce no duplicates and no errors.

A checkpoint per JSONL is recorded in ``ingestion_state`` with
``source = 'web:<absolute path>'`` so re-runs skip files whose mtime has not
advanced since last successful pass.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import psycopg

from ingestion.web_extractors import (
    WEB_TOOLS_ALL,
    ExtractError,
    ExtractResult,
    ResearchJobRef,
    SearchResultSet,
    WebScrape,
    extract,
)

# ---------- URL canonicalization ----------


def canonicalize_url(url: str) -> str | None:
    """Lowercase scheme/host, drop fragment, normalize trailing slash.

    Intentionally conservative — does NOT strip tracking params (some sites
    route on them). Returns None for empty/invalid input.
    """
    if not url or not isinstance(url, str):
        return None
    try:
        s = urlsplit(url.strip())
    except Exception:
        return None
    if not s.scheme or not s.netloc:
        return None
    scheme = s.scheme.lower()
    netloc = s.netloc.lower()
    path = s.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    return urlunsplit((scheme, netloc, path, s.query, ""))


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


# ---------- Walking JSONLs ----------


@dataclass
class WebToolEvent:
    """One tool_use/tool_result pair extracted from a JSONL."""

    tool_use_id: str
    tool_name: str
    tool_input: dict[str, Any] | None
    result_text: str
    timestamp: datetime | None
    session_id: str | None
    jsonl_path: str


def _coerce_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for c in content:
            if isinstance(c, dict):
                t = c.get("text") or c.get("content")
                if isinstance(t, str):
                    parts.append(t)
                elif isinstance(t, list):
                    parts.append(_coerce_text(t))
            elif isinstance(c, str):
                parts.append(c)
        return "\n".join(parts)
    if isinstance(content, dict):
        return _coerce_text(content.get("content"))
    return str(content)


def _parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def iter_web_events(jsonl_path: Path) -> Iterator[WebToolEvent]:
    """Yield matched tool_use/tool_result pairs for web tools in one JSONL.

    Tool calls without a matched result (mid-stream or interrupted) are
    skipped.
    """
    pending: dict[str, dict[str, Any]] = {}
    session_id: str | None = None
    try:
        fh = jsonl_path.open("r", encoding="utf-8", errors="replace")
    except OSError:
        return
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if session_id is None:
                session_id = rec.get("sessionId") or rec.get("session_id")
            ts = _parse_ts(rec.get("timestamp"))
            msg = rec.get("message") or {}
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "tool_use":
                    name = block.get("name")
                    if name not in WEB_TOOLS_ALL:
                        continue
                    tu_id = block.get("id")
                    if not tu_id:
                        continue
                    pending[tu_id] = {
                        "name": name,
                        "input": block.get("input"),
                        "ts": ts,
                    }
                elif btype == "tool_result":
                    tu_id_raw = block.get("tool_use_id")
                    if not isinstance(tu_id_raw, str):
                        continue
                    tu = pending.pop(tu_id_raw, None)
                    if not tu:
                        continue
                    yield WebToolEvent(
                        tool_use_id=tu_id_raw,
                        tool_name=tu["name"],
                        tool_input=tu["input"] if isinstance(tu["input"], dict) else None,
                        result_text=_coerce_text(block.get("content")),
                        timestamp=tu["ts"] or ts,
                        session_id=session_id,
                        jsonl_path=str(jsonl_path),
                    )


# ---------- Row construction ----------


def _build_row(event: WebToolEvent, parsed: ExtractResult) -> dict[str, Any] | None:
    """Translate a (event, parsed extractor output) into a web_artifacts row.

    Returns None if `parsed` is an ExtractError or otherwise non-storeable
    (caller increments the appropriate counter).
    """
    if isinstance(parsed, ExtractError):
        return None

    fetched_at = event.timestamp or datetime.now(UTC)

    base: dict[str, Any] = {
        "kind": parsed.kind,
        "tool_name": event.tool_name,
        "tool_use_id": event.tool_use_id,
        "session_id": event.session_id,
        "jsonl_path": event.jsonl_path,
        "fetched_at": fetched_at,
        "raw_chars": getattr(parsed, "raw_chars", None) or 0,
        "persisted_output_path": getattr(parsed, "persisted_output_path", None),
    }

    if isinstance(parsed, WebScrape):
        url_canonical = canonicalize_url(parsed.url) if parsed.url else None
        content_hash = _sha256_hex(parsed.content_markdown) if parsed.content_markdown else None
        base.update(
            {
                "url": parsed.url or None,
                "url_canonical": url_canonical,
                "content_hash": content_hash,
                "title": parsed.title,
                "content_markdown": parsed.content_markdown,
                "synthesized": parsed.synthesized,
                "prompt": parsed.prompt,
                "author": parsed.author,
                "published_at": parsed.published_at,
            }
        )
    elif isinstance(parsed, SearchResultSet):
        # Serialize items as a JSON array; preserve typed shape from extractor.
        items_json = [it.model_dump(mode="json") for it in parsed.items]
        base.update(
            {
                "query": parsed.query,
                "items": json.dumps(items_json),
                "item_count": len(parsed.items),
            }
        )
    elif isinstance(parsed, ResearchJobRef):
        base.update(
            {
                "research_id": parsed.research_id,
                "research_instructions": parsed.instructions,
                "research_model": parsed.model,
            }
        )

    return base


# ---------- Writer ----------


_INSERT_SQL = """
INSERT INTO web_artifacts (
    kind, tool_name, tool_use_id,
    url, url_canonical, content_hash, title, content_markdown,
    synthesized, prompt, author, published_at,
    query, items, item_count,
    research_id, research_instructions, research_model,
    session_id, parent_episode_id, jsonl_path,
    persisted_output_path, raw_chars, fetched_at, metadata
) VALUES (
    %(kind)s, %(tool_name)s, %(tool_use_id)s,
    %(url)s, %(url_canonical)s, %(content_hash)s, %(title)s, %(content_markdown)s,
    %(synthesized)s, %(prompt)s, %(author)s, %(published_at)s,
    %(query)s, %(items)s, %(item_count)s,
    %(research_id)s, %(research_instructions)s, %(research_model)s,
    %(session_id)s, %(parent_episode_id)s, %(jsonl_path)s,
    %(persisted_output_path)s, %(raw_chars)s, %(fetched_at)s, %(metadata)s
)
ON CONFLICT (tool_use_id) DO NOTHING
RETURNING id
"""

_ALL_COLS = (
    "kind",
    "tool_name",
    "tool_use_id",
    "url",
    "url_canonical",
    "content_hash",
    "title",
    "content_markdown",
    "synthesized",
    "prompt",
    "author",
    "published_at",
    "query",
    "items",
    "item_count",
    "research_id",
    "research_instructions",
    "research_model",
    "session_id",
    "parent_episode_id",
    "jsonl_path",
    "persisted_output_path",
    "raw_chars",
    "fetched_at",
    "metadata",
)


def _fill_defaults(row: dict[str, Any]) -> dict[str, Any]:
    """Ensure every column key is present (psycopg params barf otherwise)."""
    return {col: row.get(col) for col in _ALL_COLS}


@dataclass
class IngestStats:
    files_scanned: int = 0
    files_skipped_unchanged: int = 0
    events_seen: int = 0
    inserted: int = 0
    skipped_duplicate: int = 0
    extract_errors: int = 0
    db_errors: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)
    by_tool: dict[str, int] = field(default_factory=dict)


class WebArtifactsIngester:
    """Walk JSONLs, extract web tool_results, write to web_artifacts.

    Usage:
        ing = WebArtifactsIngester(conn)
        stats = ing.ingest_paths([Path("...jsonl"), ...])
    """

    def __init__(self, conn: psycopg.Connection[Any]) -> None:
        self._conn = conn

    def _checkpoint_get(self, source: str) -> datetime | None:
        row = self._conn.execute(
            "SELECT last_ingested_at FROM ingestion_state WHERE source = %s",
            (source,),
        ).fetchone()
        if not row:
            return None
        # dict_row vs tuple-row
        val = row["last_ingested_at"] if isinstance(row, dict) else row[0]
        if not isinstance(val, datetime):
            return None
        if val.tzinfo is None:
            val = val.replace(tzinfo=UTC)
        return val

    def _checkpoint_set(self, source: str, ts: datetime) -> None:
        self._conn.execute(
            """
            INSERT INTO ingestion_state (source, last_ingested_at)
            VALUES (%s, %s)
            ON CONFLICT (source) DO UPDATE SET last_ingested_at = EXCLUDED.last_ingested_at
            """,
            (source, ts),
        )

    def _insert(self, row: dict[str, Any]) -> bool:
        """Return True if a new row was inserted, False on conflict."""
        try:
            cur = self._conn.execute(_INSERT_SQL, _fill_defaults(row))
            r = cur.fetchone()
            return r is not None
        except psycopg.Error:
            self._conn.rollback()
            raise

    def ingest_one(self, jsonl_path: Path, stats: IngestStats | None = None) -> IngestStats:
        stats = stats or IngestStats()
        source_key = f"web:{jsonl_path}"
        try:
            mtime = datetime.fromtimestamp(jsonl_path.stat().st_mtime, tz=UTC)
        except OSError:
            return stats
        last = self._checkpoint_get(source_key)
        if last and mtime <= last:
            stats.files_skipped_unchanged += 1
            return stats
        stats.files_scanned += 1

        rows_in_file = 0
        for event in iter_web_events(jsonl_path):
            stats.events_seen += 1
            parsed = extract(event.tool_name, event.tool_input, event.result_text)
            row = _build_row(event, parsed)
            if row is None:
                stats.extract_errors += 1
                continue
            try:
                inserted = self._insert(row)
            except psycopg.Error:
                stats.db_errors += 1
                continue
            if inserted:
                stats.inserted += 1
                rows_in_file += 1
                stats.by_kind[row["kind"]] = stats.by_kind.get(row["kind"], 0) + 1
                stats.by_tool[event.tool_name] = stats.by_tool.get(event.tool_name, 0) + 1
            else:
                stats.skipped_duplicate += 1

        # Commit + checkpoint per file so partial runs are durable.
        self._checkpoint_set(source_key, mtime)
        self._conn.commit()
        return stats

    def ingest_paths(self, paths: Iterable[Path]) -> IngestStats:
        stats = IngestStats()
        for p in paths:
            self.ingest_one(p, stats)
        return stats
