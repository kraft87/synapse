"""
Operation-specific parsers for web-research tool_results.

Each tool's tool_result has a distinct shape (text-block-format, JSON, markdown).
This module produces a normalized typed output regardless of source. They are
refinements: ``ingestion.capture`` detects web research by shape and calls ``refine`` for
the operations listed in ``_DISPATCH``, for their better title / author / date / item
fields. An operation is matched by its own name whatever the connector is called
(``*__crawling_exa``, ``*__firecrawl_scrape``, ``WebFetch``), so a renamed server loses
nothing.

Three output shapes:
  - WebScrape: single-page content with URL + body. `synthesized=True` for WebFetch
    (LLM-mediated answer, not raw page); False for firecrawl_scrape/crawling_exa.
  - SearchResultSet: list of (title, url, snippet?, published_at?) items.
  - ResearchJobRef: deep_researcher_start returns a job id, not results.

`ExtractError` is returned for non-content responses (MCP validation failures,
redirect-detection, captcha pages, etc.) so callers can count them rather than
treating them as missing data.

Parsers read only the text they are given. A persisted-output redirect ("Output too large
(N KB). Full output saved to: <path>") names a file on the client machine; the server cannot
read it, so the Stop hook inlines it before shipping the record (or the capture skips the
call).
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
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


# ---------- Operation names ----------


def tool_id(tool_name: str) -> str:
    """The operation a tool name calls, whatever the connector prefix: ``mcp__<server>__<tool>``
    (the server is whatever the user named it: ``claude_ai_firecrawl`` for the claude.ai
    connector, ``firecrawl`` or ``Exa_search`` for a local one) and ``<server>__<tool>``
    both give ``<tool>``; built-in names (WebFetch) pass through."""
    return tool_name.rsplit("__", 1)[-1]


# ---------- Helpers ----------

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


# A page that is a wall rather than content: captcha, login, rate limit, script shell. Real
# pages are longer than these, so the phrases only count in a short body.
_BLOCK_PAGE_RE = re.compile(
    r"captcha|verify (?:that )?you(?: are|'re) (?:a )?human|are you a robot|just a moment"
    r"|attention required|enable javascript|access denied|403 forbidden|404 not found"
    r"|page not found|(?:sign|log) ?in to (?:continue|view|access)|please (?:sign|log) ?in"
    r"|login required|unusual traffic|too many requests|rate limit exceeded",
    re.IGNORECASE,
)
_BLOCK_PAGE_MAX_CHARS = 3000


def error_reason(text: str) -> str | None:
    """Return a brief reason if `text` is clearly a tool-error, redirect, captcha or login
    response rather than page content, else None."""
    if not text:
        return "empty"
    head = text[:400]
    for pre in _ERROR_PRELUDES:
        if pre in head:
            return pre.rstrip(":").lower().replace(" ", "_")
    if len(text) < _BLOCK_PAGE_MAX_CHARS and _BLOCK_PAGE_RE.search(text[:600]):
        return "blocked_page"
    return None


def _error_guard(tool_name: str, result_text: str) -> ExtractError | None:
    """Shared pure-error guard for the text-format parsers. Returns an
    ``ExtractError`` when `result_text` is a tool-error response, else None."""
    err = error_reason(result_text)
    if err:
        return ExtractError(
            tool_name=tool_name,
            reason=err,
            detail=(result_text or "")[:400],
            raw_chars=len(result_text or ""),
        )
    return None


def parse_datetime(s: str | None) -> datetime | None:
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
    err = error_reason(result_text)
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

_BOILERPLATE = [
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


def strip_boilerplate(md: str) -> str:
    for pat in _BOILERPLATE:
        md = pat.sub("", md)
    return re.sub(r"\n{3,}", "\n\n", md).strip()


def parse_firecrawl_scrape(tool_input: dict[str, Any], result_text: str) -> ExtractResult:
    raw_chars = len(result_text or "")
    if e := _error_guard("mcp__claude_ai_firecrawl__firecrawl_scrape", result_text):
        return e
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
    md = strip_boilerplate(md)
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
    if e := _error_guard("mcp__claude_ai_Exa_search__crawling_exa", result_text):
        return e
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
        published = parse_datetime(m.group("published"))
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
    if e := _error_guard("WebSearch", result_text):
        return e
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
    if e := _error_guard("mcp__claude_ai_firecrawl__firecrawl_search", result_text):
        return e
    query = (tool_input or {}).get("query")
    items: list[SearchResultItem] = []
    text = (result_text or "").strip()
    if text.startswith("{"):
        try:
            obj = json.loads(text)
            # Current Firecrawl servers wrap the lists: {"success": true, "data": {"web": [...]}}.
            if isinstance(obj.get("data"), dict):
                obj = obj["data"]
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
                                published_at=parse_datetime(
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
        fields[key] = m.group("val").strip()
    if not fields.get("url"):
        return None
    return SearchResultItem(
        url=fields["url"],
        title=fields.get("title"),
        snippet=fields.get("snippet"),
        published_at=parse_datetime(fields.get("published")),
        author=fields.get("author") if fields.get("author") not in (None, "", "N/A") else None,
    )


def parse_exa_text_blocks(
    tool_name: str, tool_input: dict[str, Any], result_text: str
) -> ExtractResult:
    raw_chars = len(result_text or "")
    if e := _error_guard(tool_name, result_text):
        return e
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
    if e := _error_guard("mcp__claude_ai_Exa_search__web_search_advanced_exa", result_text):
        return e
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
                        published_at=parse_datetime(it.get("publishedDate")),
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
    if e := _error_guard("mcp__claude_ai_Exa_search__deep_search_exa", result_text):
        return e
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
    "firecrawl_scrape": parse_firecrawl_scrape,
    "firecrawl_search": parse_firecrawl_search,
    "crawling_exa": parse_crawling_exa,
    "web_search_advanced_exa": parse_exa_advanced,
    "deep_search_exa": parse_deep_search,
    "deep_researcher_start": parse_deep_researcher_start,
    "deep_researcher_check": parse_deep_researcher_check,
}

# Exa text-block-format tools share a parser, dispatched by name
_EXA_TEXT_BLOCK_TOOLS = {"web_search_exa", "people_search_exa", "company_research_exa"}


def refine(
    tool_name: str,
    tool_input: dict[str, Any] | None,
    result_text: str,
) -> ExtractResult | None:
    """Parse a (tool_name, tool_input, result_text) triple with the parser registered for
    the operation, or None when this operation has none (the generic shape detector in
    ``ingestion.capture`` takes it)."""
    key = tool_id(tool_name)
    if key in _EXA_TEXT_BLOCK_TOOLS:
        out = parse_exa_text_blocks(tool_name, tool_input or {}, result_text or "")
    elif key in _DISPATCH:
        out = _DISPATCH[key](tool_input or {}, result_text or "")
    else:
        return None
    out.tool_name = tool_name  # the name it was called under, whatever the server
    return out
