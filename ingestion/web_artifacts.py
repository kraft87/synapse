"""
Write captured web research into Postgres web_artifacts.

The persistence half of the web-capture pipeline; the parser half lives in
``ingestion.web_extractors`` and the decision of what to capture in ``ingestion.capture``,
which calls this module for every call it stores. One ``web_artifacts`` row per
tool_use/tool_result pair.

Idempotency is by ``tool_use_id`` (UNIQUE in the schema): shipping the same records again
inserts nothing and raises nothing. A page or result list that was already stored (same
canonical URL and content hash) from another call is not stored twice either.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import psycopg

from ingestion.secret_redact import redact_secrets
from ingestion.textsafe import strip_nul
from ingestion.web_extractors import ExtractResult, ResearchJobRef, SearchResultSet, WebScrape

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


def _clean(value: Any) -> Any:
    """NUL bytes out (Postgres text rejects them; escaped as \\u0000 they still fail the jsonb
    cast) and credentials redacted, in every string of any nesting."""
    if isinstance(value, str):
        return redact_secrets(strip_nul(value))
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


# ---------- Row construction ----------


def build_row(
    parsed: ExtractResult,
    *,
    tool_name: str,
    tool_use_id: str,
    session_id: str,
    fetched_at: datetime,
    parent_episode_id: int | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Translate a parsed capture into a web_artifacts row. Credentials are redacted from
    every text field, so what is stored, hashed and served never carries one."""
    base: dict[str, Any] = {
        "kind": parsed.kind,
        "tool_name": tool_name,
        "tool_use_id": tool_use_id,
        "session_id": session_id,
        "parent_episode_id": parent_episode_id,
        "fetched_at": fetched_at,
        "raw_chars": getattr(parsed, "raw_chars", None) or 0,
    }

    if isinstance(parsed, WebScrape):
        url = _clean(parsed.url) if parsed.url else None
        markdown = _clean(parsed.content_markdown)
        base.update(
            {
                "url": url,
                "url_canonical": canonicalize_url(url) if url else None,
                "content_hash": _sha256_hex(markdown) if markdown else None,
                "title": _clean(parsed.title),
                "content_markdown": markdown,
                "synthesized": parsed.synthesized,
                "prompt": _clean(parsed.prompt),
                "author": _clean(parsed.author),
                "published_at": parsed.published_at,
            }
        )
    elif isinstance(parsed, SearchResultSet):
        items = _clean([it.model_dump(mode="json") for it in parsed.items])
        query = _clean(parsed.query)
        urls = sorted(canonicalize_url(it["url"]) or it["url"] for it in items)
        base.update(
            {
                "query": query,
                "items": json.dumps(items),
                "item_count": len(items),
                # The same results for the same query are one stored copy.
                "content_hash": _sha256_hex(json.dumps([query, urls])),
            }
        )
    elif isinstance(parsed, ResearchJobRef):
        base.update(
            {
                "research_id": _clean(parsed.research_id),
                "research_instructions": _clean(parsed.instructions),
                "research_model": parsed.model,
            }
        )

    base["metadata"] = json.dumps(_clean(metadata))
    return base


# ---------- Writer ----------


_INSERT_SQL = """
INSERT INTO web_artifacts (
    kind, tool_name, tool_use_id,
    url, url_canonical, content_hash, title, content_markdown,
    synthesized, prompt, author, published_at,
    query, items, item_count,
    research_id, research_instructions, research_model,
    session_id, parent_episode_id,
    raw_chars, fetched_at, metadata
) VALUES (
    %(kind)s, %(tool_name)s, %(tool_use_id)s,
    %(url)s, %(url_canonical)s, %(content_hash)s, %(title)s, %(content_markdown)s,
    %(synthesized)s, %(prompt)s, %(author)s, %(published_at)s,
    %(query)s, %(items)s, %(item_count)s,
    %(research_id)s, %(research_instructions)s, %(research_model)s,
    %(session_id)s, %(parent_episode_id)s,
    %(raw_chars)s, %(fetched_at)s, %(metadata)s
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
    "raw_chars",
    "fetched_at",
    "metadata",
)


def insert_artifact(conn: psycopg.Connection[Any], row: dict[str, Any]) -> bool:
    """Insert ``row``; True if it landed, False when its tool_use_id is already stored.

    A failed insert aborts the caller's transaction: run it in a savepoint to carry on."""
    params = {col: row.get(col) for col in _ALL_COLS}  # psycopg params barf on a missing key
    return conn.execute(_INSERT_SQL, params).fetchone() is not None


def artifact_stored(
    conn: psycopg.Connection[Any],
    tool_use_id: str,
    url_canonical: str | None,
    content_hash: str | None,
) -> bool:
    """True if this call was already captured, or the same content under the same URL
    (a result list has no URL: the same results for the same query) was captured from
    another call."""
    row = conn.execute(
        """
        SELECT 1 FROM web_artifacts
        WHERE tool_use_id = %s
           OR (content_hash = %s AND url_canonical IS NOT DISTINCT FROM %s)
        LIMIT 1
        """,
        (tool_use_id, content_hash, url_canonical),
    ).fetchone()
    return row is not None


def count_fetched_between(conn: psycopg.Connection[Any], start: datetime, end: datetime) -> int:
    """Artifacts fetched in [start, end)."""
    row = conn.execute(
        "SELECT count(*) AS n FROM web_artifacts WHERE fetched_at >= %s AND fetched_at < %s",
        (start, end),
    ).fetchone()
    return int(row["n"]) if row else 0
