"""
Parsers for web-research tool_results in Claude Code JSONLs.

Each tool's tool_result has a distinct shape (text-block-format, JSON, markdown).
This module produces a normalized typed output regardless of source.

Three output shapes:
  - WebScrape: single-page content with URL + body. `synthesized=True` for WebFetch
    (LLM-mediated answer, not raw page); False for firecrawl_scrape/crawling_exa.
  - SearchResultSet: list of (title, url, snippet?, published_at?) items.
  - ResearchJobRef: deep_researcher_start returns a job id, not results.

`ExtractError` is returned for non-content responses (MCP validation failures,
redirect-detection, captcha pages, etc.) so callers can count them rather than
treating them as missing data.

Persisted-output redirects ("Output too large (N KB). Full output saved to:
/path/to/tool-results/<id>.{txt,json}") are detected; if the file exists,
content is loaded from disk and re-parsed. Without this, ~10-15% of the
largest tool_results would be lost to ~2KB previews.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

# ---------- Output models ----------


class SearchResultItem(BaseModel):
    url: str
    title: str | None = None
    snippet: str | None = None
    published_at: datetime | None = None
    author: str | None = None
    position: int | None = None


class SearchResultSet(BaseModel):
    kind: Literal["search_result_set"] = "search_result_set"
    tool_name: str
    query: str | None = None
    items: list[SearchResultItem] = Field(default_factory=list)
    raw_chars: int = 0
    persisted_output_path: str | None = None


class WebScrape(BaseModel):
    kind: Literal["web_scrape"] = "web_scrape"
    tool_name: str
    url: str
    title: str | None = None
    content_markdown: str = ""
    synthesized: bool = False
    prompt: str | None = None
    published_at: datetime | None = None
    author: str | None = None
    raw_chars: int = 0
    persisted_output_path: str | None = None


class ResearchJobRef(BaseModel):
    kind: Literal["research_job_ref"] = "research_job_ref"
    tool_name: str
    research_id: str
    instructions: str
    model: str | None = None


class ExtractError(BaseModel):
    kind: Literal["error"] = "error"
    tool_name: str
    reason: str
    detail: str | None = None
    raw_chars: int = 0


ExtractResult = Annotated[
    SearchResultSet | WebScrape | ResearchJobRef | ExtractError,
    Field(discriminator="kind"),
]


# ---------- Constants ----------

WEB_TOOLS_SCRAPE = {
    "WebFetch",
    "mcp__claude_ai_firecrawl__firecrawl_scrape",
    "mcp__claude_ai_Exa_search__crawling_exa",
}

WEB_TOOLS_SEARCH = {
    "WebSearch",
    "mcp__claude_ai_firecrawl__firecrawl_search",
    "mcp__claude_ai_Exa_search__web_search_exa",
    "mcp__claude_ai_Exa_search__web_search_advanced_exa",
    "mcp__claude_ai_Exa_search__deep_search_exa",
    "mcp__claude_ai_Exa_search__people_search_exa",
    "mcp__claude_ai_Exa_search__company_research_exa",
}

WEB_TOOLS_RESEARCH = {
    "mcp__claude_ai_Exa_search__deep_researcher_start",
    "mcp__claude_ai_Exa_search__deep_researcher_check",
}

WEB_TOOLS_ALL = WEB_TOOLS_SCRAPE | WEB_TOOLS_SEARCH | WEB_TOOLS_RESEARCH


# ---------- Helpers ----------

_PERSISTED_HEADER_RE = re.compile(
    r"(?:<persisted-output>|Error: result \([\d,]+(?: characters)?[^)]*\))",
    re.IGNORECASE,
)
_PERSISTED_PATH_RE = re.compile(
    r"(?:saved to:?|saved to)\s*(/[^\s\"'<>]+\.(?:json|txt))",
    re.IGNORECASE,
)


def _detect_persisted_output(text: str) -> str | None:
    """Return the on-disk path for a persisted-output response, or None."""
    if not text:
        return None
    if not _PERSISTED_HEADER_RE.search(text[:500]):
        return None
    m = _PERSISTED_PATH_RE.search(text[:2000])
    return m.group(1) if m else None


def _load_persisted(path: str) -> str | None:
    """Load a persisted-output file. Handles .txt and .json (array-of-text-blocks)."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        raw = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if path.endswith(".json"):
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                parts = []
                for item in data:
                    if isinstance(item, dict):
                        t = item.get("text") or item.get("content")
                        if isinstance(t, str):
                            parts.append(t)
                        elif isinstance(t, list):
                            for sub in t:
                                if isinstance(sub, dict) and isinstance(sub.get("text"), str):
                                    parts.append(sub["text"])
                    elif isinstance(item, str):
                        parts.append(item)
                return "\n".join(parts)
        except json.JSONDecodeError:
            pass
    return raw


_ERROR_PRELUDES = (
    "MCP error -32602",
    "execution failed:",
    "validation failed",
    "Invalid request body",
    "REDIRECT DETECTED",
    "error (400):",
    "error (404):",
    "error (429):",
    "error (500):",
)


def _is_pure_error(text: str) -> str | None:
    """Return a brief reason if `text` is clearly a tool-error response, else None."""
    if not text:
        return "empty"
    head = text[:400]
    for pre in _ERROR_PRELUDES:
        if pre in head:
            return pre.rstrip(":").lower().replace(" ", "_")
    return None


def _parse_datetime(s: str | None) -> datetime | None:
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if s in ("", "N/A", "n/a", "None", "null"):
        return None
    # Try ISO 8601 first
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except ValueError:
        pass
    # YYYY-MM-DD fallback
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError:
        return None


# ---------- WebFetch ----------

_REDIRECT_RE = re.compile(r"REDIRECT DETECTED", re.IGNORECASE)


def parse_webfetch(tool_input: dict[str, Any], result_text: str) -> ExtractResult:
    raw_chars = len(result_text or "")
    if not result_text:
        return ExtractError(tool_name="WebFetch", reason="empty", raw_chars=0)
    if _REDIRECT_RE.search(result_text[:200]):
        return ExtractError(
            tool_name="WebFetch",
            reason="redirect_detected",
            detail=result_text[:400],
            raw_chars=raw_chars,
        )
    err = _is_pure_error(result_text)
    if err:
        return ExtractError(
            tool_name="WebFetch", reason=err, detail=result_text[:400], raw_chars=raw_chars
        )
    url = (tool_input or {}).get("url") or ""
    prompt = (tool_input or {}).get("prompt")
    return WebScrape(
        tool_name="WebFetch",
        url=url,
        content_markdown=result_text.strip(),
        synthesized=True,
        prompt=prompt,
        raw_chars=raw_chars,
    )


# ---------- firecrawl_scrape ----------

_FIRECRAWL_BOILERPLATE = [
    re.compile(r"^\s*\[Skip to content\]\([^)]+\)\s*", re.MULTILINE),
    # Whole "You signed in / out / switched accounts ... Dismiss alert" run —
    # match lazily across the multi-Reload block.
    re.compile(
        r"You signed (?:in|out) with another tab.*?Dismiss alert",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"\{\{\s*message\s*\}\}", re.IGNORECASE),
    re.compile(r"\[Reload\]\([^)]+\)(?:\s+to refresh your session\.?)?", re.IGNORECASE),
]


def _strip_firecrawl_boilerplate(md: str) -> str:
    for pat in _FIRECRAWL_BOILERPLATE:
        md = pat.sub("", md)
    return re.sub(r"\n{3,}", "\n\n", md).strip()


def parse_firecrawl_scrape(tool_input: dict[str, Any], result_text: str) -> ExtractResult:
    raw_chars = len(result_text or "")
    err = _is_pure_error(result_text)
    if err:
        return ExtractError(
            tool_name="mcp__claude_ai_firecrawl__firecrawl_scrape",
            reason=err,
            detail=(result_text or "")[:400],
            raw_chars=raw_chars,
        )
    url = (tool_input or {}).get("url") or ""
    text = result_text or ""
    md = text
    # firecrawl returns {"markdown": "..."} OR sometimes raw markdown
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                md = obj.get("markdown") or obj.get("content") or text
                if not url:
                    url = obj.get("url") or url
        except json.JSONDecodeError:
            pass
    md = _strip_firecrawl_boilerplate(md)
    return WebScrape(
        tool_name="mcp__claude_ai_firecrawl__firecrawl_scrape",
        url=url,
        content_markdown=md,
        synthesized=False,
        raw_chars=raw_chars,
    )


# ---------- crawling_exa ----------

_EXA_HEADER_RE = re.compile(
    r"^#\s*(?P<title>.+?)\s*\nURL:\s*(?P<url>https?://\S+)\s*\n(?:Author:\s*(?P<author>.+?)\s*\n)?(?:Published:\s*(?P<published>[^\n]+)\s*\n)?",
    re.MULTILINE,
)


def parse_crawling_exa(tool_input: dict[str, Any], result_text: str) -> ExtractResult:
    raw_chars = len(result_text or "")
    err = _is_pure_error(result_text)
    if err:
        return ExtractError(
            tool_name="mcp__claude_ai_Exa_search__crawling_exa",
            reason=err,
            detail=(result_text or "")[:400],
            raw_chars=raw_chars,
        )
    text = result_text or ""
    m = _EXA_HEADER_RE.search(text[:1000])
    url = ""
    title = None
    author = None
    published = None
    if m:
        title = m.group("title").strip()
        url = m.group("url").strip()
        author = (m.group("author") or "").strip() or None
        published = _parse_datetime(m.group("published"))
    # Fall back to input urls
    if not url:
        inp_url = (tool_input or {}).get("urls") or (tool_input or {}).get("url")
        if isinstance(inp_url, str):
            # crawling_exa sometimes JSON-encodes: '["http..."]'
            s = inp_url.strip()
            if s.startswith("["):
                try:
                    arr = json.loads(s)
                    if isinstance(arr, list) and arr:
                        url = str(arr[0])
                except json.JSONDecodeError:
                    url = s
            else:
                url = s
    return WebScrape(
        tool_name="mcp__claude_ai_Exa_search__crawling_exa",
        url=url,
        title=title,
        content_markdown=text.strip(),
        synthesized=False,
        author=author,
        published_at=published,
        raw_chars=raw_chars,
    )


# ---------- WebSearch ----------

_WEBSEARCH_LINKS_RE = re.compile(r"Links:\s*(\[.+?\])\s*(?:\n|$)", re.DOTALL)


def parse_websearch(tool_input: dict[str, Any], result_text: str) -> ExtractResult:
    raw_chars = len(result_text or "")
    err = _is_pure_error(result_text)
    if err:
        return ExtractError(
            tool_name="WebSearch", reason=err, detail=(result_text or "")[:400], raw_chars=raw_chars
        )
    query = (tool_input or {}).get("query")
    items: list[SearchResultItem] = []
    m = _WEBSEARCH_LINKS_RE.search(result_text or "")
    if m:
        try:
            arr = json.loads(m.group(1))
            for i, it in enumerate(arr):
                if isinstance(it, dict) and it.get("url"):
                    items.append(
                        SearchResultItem(
                            url=it["url"],
                            title=it.get("title"),
                            snippet=it.get("snippet") or it.get("description"),
                            position=i + 1,
                        )
                    )
        except json.JSONDecodeError:
            pass
    return SearchResultSet(tool_name="WebSearch", query=query, items=items, raw_chars=raw_chars)


# ---------- firecrawl_search ----------


def parse_firecrawl_search(tool_input: dict[str, Any], result_text: str) -> ExtractResult:
    raw_chars = len(result_text or "")
    err = _is_pure_error(result_text)
    if err:
        return ExtractError(
            tool_name="mcp__claude_ai_firecrawl__firecrawl_search",
            reason=err,
            detail=(result_text or "")[:400],
            raw_chars=raw_chars,
        )
    query = (tool_input or {}).get("query")
    items: list[SearchResultItem] = []
    text = (result_text or "").strip()
    if text.startswith("{"):
        try:
            obj = json.loads(text)
            for key in ("web", "results", "data"):
                arr = obj.get(key)
                if isinstance(arr, list):
                    for i, it in enumerate(arr):
                        if not isinstance(it, dict) or not it.get("url"):
                            continue
                        items.append(
                            SearchResultItem(
                                url=it["url"],
                                title=it.get("title"),
                                snippet=it.get("description") or it.get("snippet"),
                                published_at=_parse_datetime(
                                    it.get("publishedDate") or it.get("date")
                                ),
                                author=it.get("author"),
                                position=it.get("position") or (i + 1),
                            )
                        )
                    break
        except json.JSONDecodeError:
            pass
    return SearchResultSet(
        tool_name="mcp__claude_ai_firecrawl__firecrawl_search",
        query=query,
        items=items,
        raw_chars=raw_chars,
    )


# ---------- Exa text-block format (web_search_exa, people_search_exa, company_research_exa) ----------

# A block is a run of lines starting with Title: ... URL: ... etc., separated
# by blank lines or '---'. Highlights spans subsequent lines until the next block.
_EXA_BLOCK_SPLIT_RE = re.compile(r"\n\s*\n(?=Title:\s)", re.MULTILINE)
_EXA_FIELD_RE = re.compile(
    r"^(?P<key>Title|URL|Published|Author|Highlights?):\s*(?P<val>.*?)(?=\n(?:Title|URL|Published|Author|Highlights?):|\Z)",
    re.DOTALL | re.MULTILINE,
)


def _parse_exa_block(block: str) -> SearchResultItem | None:
    fields: dict[str, str] = {}
    for m in _EXA_FIELD_RE.finditer(block):
        key = m.group("key").lower().rstrip("s")  # Highlights -> highlight
        if key == "highlight":
            key = "snippet"
        elif key == "published":
            key = "published"
        elif key == "url":
            key = "url"
        elif key == "title":
            key = "title"
        elif key == "author":
            key = "author"
        fields[key] = m.group("val").strip()
    if not fields.get("url"):
        return None
    return SearchResultItem(
        url=fields["url"],
        title=fields.get("title"),
        snippet=fields.get("snippet"),
        published_at=_parse_datetime(fields.get("published")),
        author=fields.get("author") if fields.get("author") not in (None, "", "N/A") else None,
    )


def parse_exa_text_blocks(
    tool_name: str, tool_input: dict[str, Any], result_text: str
) -> ExtractResult:
    raw_chars = len(result_text or "")
    err = _is_pure_error(result_text)
    if err:
        return ExtractError(
            tool_name=tool_name, reason=err, detail=(result_text or "")[:400], raw_chars=raw_chars
        )
    query = (tool_input or {}).get("query")
    text = result_text or ""
    items: list[SearchResultItem] = []
    # Find the start of the first Title: header
    start = text.find("Title:")
    body = text[start:] if start >= 0 else text
    blocks = _EXA_BLOCK_SPLIT_RE.split(body) if body else []
    for i, blk in enumerate(blocks):
        if not blk.strip():
            continue
        item = _parse_exa_block(blk)
        if item:
            item.position = i + 1
            items.append(item)
    return SearchResultSet(tool_name=tool_name, query=query, items=items, raw_chars=raw_chars)


# ---------- web_search_advanced_exa (JSON envelope) ----------


def parse_exa_advanced(tool_input: dict[str, Any], result_text: str) -> ExtractResult:
    raw_chars = len(result_text or "")
    err = _is_pure_error(result_text)
    if err:
        return ExtractError(
            tool_name="mcp__claude_ai_Exa_search__web_search_advanced_exa",
            reason=err,
            detail=(result_text or "")[:400],
            raw_chars=raw_chars,
        )
    query = (tool_input or {}).get("query")
    items: list[SearchResultItem] = []
    text = (result_text or "").strip()
    if text.startswith("{"):
        try:
            obj = json.loads(text)
            results = obj.get("results") or []
            for i, it in enumerate(results):
                if not isinstance(it, dict):
                    continue
                url = it.get("url") or it.get("id")
                if not url:
                    continue
                items.append(
                    SearchResultItem(
                        url=url,
                        title=it.get("title"),
                        snippet=(it.get("text") or it.get("highlights") or "")[:500] or None,
                        published_at=_parse_datetime(it.get("publishedDate")),
                        author=it.get("author") or None,
                        position=i + 1,
                    )
                )
        except json.JSONDecodeError:
            pass
    return SearchResultSet(
        tool_name="mcp__claude_ai_Exa_search__web_search_advanced_exa",
        query=query,
        items=items,
        raw_chars=raw_chars,
    )


# ---------- deep_search_exa (markdown ### N. Title \n **URL:** ...) ----------

_DEEP_SEARCH_ITEM_RE = re.compile(
    r"###\s+(?P<n>\d+)\.\s+(?P<title>[^\n|]+?)(?:\s*\|\s*[^\n]+)?\s*\n"
    r"(?:\*\*URL:\*\*|URL:)\s*(?P<url>https?://\S+)"
    r"(?:(?:.|\n)*?(?=###\s+\d+\.|\Z))",
    re.MULTILINE,
)


def parse_deep_search(tool_input: dict[str, Any], result_text: str) -> ExtractResult:
    raw_chars = len(result_text or "")
    err = _is_pure_error(result_text)
    if err:
        return ExtractError(
            tool_name="mcp__claude_ai_Exa_search__deep_search_exa",
            reason=err,
            detail=(result_text or "")[:400],
            raw_chars=raw_chars,
        )
    query = (tool_input or {}).get("objective") or (tool_input or {}).get("query")
    items: list[SearchResultItem] = []
    text = result_text or ""
    for m in _DEEP_SEARCH_ITEM_RE.finditer(text):
        url = m.group("url").rstrip(".,;:!?)\"'")
        items.append(
            SearchResultItem(
                url=url,
                title=m.group("title").strip(),
                position=int(m.group("n")),
            )
        )
    return SearchResultSet(
        tool_name="mcp__claude_ai_Exa_search__deep_search_exa",
        query=query,
        items=items,
        raw_chars=raw_chars,
    )


# ---------- deep_researcher_start ----------


def parse_deep_researcher_start(tool_input: dict[str, Any], result_text: str) -> ExtractResult:
    raw_chars = len(result_text or "")
    try:
        obj = json.loads((result_text or "").strip())
    except json.JSONDecodeError:
        return ExtractError(
            tool_name="mcp__claude_ai_Exa_search__deep_researcher_start",
            reason="non_json",
            raw_chars=raw_chars,
        )
    if not obj.get("success") or not obj.get("researchId"):
        return ExtractError(
            tool_name="mcp__claude_ai_Exa_search__deep_researcher_start",
            reason="no_research_id",
            raw_chars=raw_chars,
        )
    return ResearchJobRef(
        tool_name="mcp__claude_ai_Exa_search__deep_researcher_start",
        research_id=obj["researchId"],
        instructions=obj.get("instructions") or (tool_input or {}).get("instructions") or "",
        model=obj.get("model"),
    )


# ---------- deep_researcher_check ----------


def parse_deep_researcher_check(tool_input: dict[str, Any], result_text: str) -> ExtractResult:
    """Status polls return null/skip; only the completed report is content."""
    raw_chars = len(result_text or "")
    try:
        obj = json.loads((result_text or "").strip())
    except json.JSONDecodeError:
        return ExtractError(
            tool_name="mcp__claude_ai_Exa_search__deep_researcher_check",
            reason="non_json",
            raw_chars=raw_chars,
        )
    status = obj.get("status")
    if status != "completed":
        return ExtractError(
            tool_name="mcp__claude_ai_Exa_search__deep_researcher_check",
            reason=f"status_{status}",
            raw_chars=raw_chars,
        )
    report = obj.get("report") or obj.get("output") or ""
    return WebScrape(
        tool_name="mcp__claude_ai_Exa_search__deep_researcher_check",
        url=f"exa://research/{obj.get('researchId', '')}",
        content_markdown=str(report),
        synthesized=True,  # LLM-generated research report
        raw_chars=raw_chars,
    )


# ---------- Dispatch ----------

_DISPATCH = {
    "WebFetch": parse_webfetch,
    "WebSearch": parse_websearch,
    "mcp__claude_ai_firecrawl__firecrawl_scrape": parse_firecrawl_scrape,
    "mcp__claude_ai_firecrawl__firecrawl_search": parse_firecrawl_search,
    "mcp__claude_ai_Exa_search__crawling_exa": parse_crawling_exa,
    "mcp__claude_ai_Exa_search__web_search_advanced_exa": parse_exa_advanced,
    "mcp__claude_ai_Exa_search__deep_search_exa": parse_deep_search,
    "mcp__claude_ai_Exa_search__deep_researcher_start": parse_deep_researcher_start,
    "mcp__claude_ai_Exa_search__deep_researcher_check": parse_deep_researcher_check,
}

# Exa text-block-format tools share a parser, dispatched by name
_EXA_TEXT_BLOCK_TOOLS = {
    "mcp__claude_ai_Exa_search__web_search_exa",
    "mcp__claude_ai_Exa_search__people_search_exa",
    "mcp__claude_ai_Exa_search__company_research_exa",
}


def extract(
    tool_name: str,
    tool_input: dict[str, Any] | None,
    result_text: str,
) -> ExtractResult:
    """
    Parse a (tool_name, tool_input, result_text) triple into a typed shape.
    If `result_text` is a persisted-output redirect, the on-disk file is loaded
    and re-parsed; the path is recorded on the returned object.
    """
    persisted_path = _detect_persisted_output(result_text or "")
    effective_text = result_text
    if persisted_path:
        loaded = _load_persisted(persisted_path)
        if loaded:
            effective_text = loaded

    if tool_name in _EXA_TEXT_BLOCK_TOOLS:
        out = parse_exa_text_blocks(tool_name, tool_input or {}, effective_text or "")
    elif tool_name in _DISPATCH:
        out = _DISPATCH[tool_name](tool_input or {}, effective_text or "")
    else:
        return ExtractError(
            tool_name=tool_name, reason="unknown_tool", raw_chars=len(result_text or "")
        )

    if persisted_path and hasattr(out, "persisted_output_path"):
        out.persisted_output_path = persisted_path
    return out
