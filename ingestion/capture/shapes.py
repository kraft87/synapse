"""What web research looks like, whatever tool did it.

A call's INPUT says what it asked for (a URL, or a search query); its OUTPUT says what came
back (a page, or a list of results). Capture needs both: URLs in an output alone are not
research (a note, a mail, a log can all be full of links).

Output shapes, tried on text the tool returned:

* page: the text itself (or a ``markdown`` / ``content`` / ``text`` / ``body`` field of a JSON
  envelope), for a call that asked for exactly one URL;
* result list: JSON objects with a ``url`` and a ``title`` or ``snippet``; ``Title:`` / ``URL:``
  blocks (also headings followed by a ``URL:`` line); or markdown links that start a line.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ingestion.web_extractors import SearchResultItem, parse_datetime

# Input fields that carry what the call asked for.
_URL_KEYS = ("url", "urls", "uri", "link", "links", "href")
_QUERY_KEYS = ("query", "queries", "q", "objective", "search_query", "search_term", "keywords")

_SNIPPET_CHARS = 500

URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")


def input_urls(tool_input: dict[str, Any]) -> list[str]:
    """Every URL string the input carries, in order, once (a JSON-encoded list included)."""
    found: list[str] = []
    for key in _URL_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value.lstrip().startswith("["):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
        for v in value if isinstance(value, list) else [value]:
            if isinstance(v, str) and v.strip() and v.strip() not in found:
                found.append(v.strip())
    return found


def input_queries(tool_input: dict[str, Any]) -> list[str]:
    found: list[str] = []
    for key in _QUERY_KEYS:
        value = tool_input.get(key)
        for v in value if isinstance(value, list) else [value]:
            if isinstance(v, str) and v.strip():
                found.append(v.strip())
    return found


# ---------- Pages ----------

_BODY_FIELDS = ("markdown", "content", "text", "body")
_H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_TITLE_LINE_RE = re.compile(r"^Title:\s*(.+)$", re.MULTILINE | re.IGNORECASE)


def page_body(text: str) -> tuple[str, str | None]:
    """The page text of a one-URL result and its title when it has one: the body field of a
    JSON envelope, else the text itself."""
    body, title = text, None
    if text.lstrip().startswith("{"):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            for node in (obj, obj.get("data")):
                if not isinstance(node, dict):
                    continue
                field = next((f for f in _BODY_FIELDS if isinstance(node.get(f), str)), None)
                if field and node[field].strip():
                    meta = node.get("metadata")
                    title = node.get("title") or (
                        meta.get("title") if isinstance(meta, dict) else None
                    )
                    body = node[field]
                    break
    if not isinstance(title, str) or not title.strip():
        m = _H1_RE.search(body[:1000]) or _TITLE_LINE_RE.search(body[:1000])
        title = m.group(1) if m else None
    return body, (title.strip()[:300] or None) if isinstance(title, str) else None


# ---------- Result lists ----------

_ITEM_URL = ("url", "link", "href", "uri")
_ITEM_TITLE = ("title", "name", "headline")
_ITEM_SNIPPET = ("snippet", "description", "summary", "text", "content", "excerpt", "highlights")
_ITEM_DATE = ("publishedDate", "published_date", "published", "date")


def _first_str(obj: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = obj.get(key)
        if isinstance(value, list):
            value = " ".join(str(v) for v in value if isinstance(v, str))
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _json_lists(node: Any, depth: int = 0) -> list[list[dict[str, Any]]]:
    """Every list of objects within three levels of a parsed JSON value."""
    found: list[list[dict[str, Any]]] = []
    if isinstance(node, list):
        found.append([d for d in node if isinstance(d, dict)])
    if depth < 3:
        children = (
            node.values() if isinstance(node, dict) else node if isinstance(node, list) else ()
        )
        for child in children:
            found.extend(_json_lists(child, depth + 1))
    return found


def _json_items(text: str) -> list[SearchResultItem]:
    """The JSON the text is, or the first array of objects inside it (``Links: [{...}]``)."""
    stripped = text.lstrip()
    if stripped.startswith(("[", "{")):
        data, start = stripped, 0
    else:
        data, start = text, text.find("[{")
        if start < 0:
            return []
    try:
        lists = _json_lists(json.JSONDecoder().raw_decode(data, start)[0])
    except json.JSONDecodeError:
        return []
    best: list[SearchResultItem] = []
    for objs in lists:
        items = []
        for o in objs:
            url = _first_str(o, _ITEM_URL)
            if url:
                snippet = _first_str(o, _ITEM_SNIPPET)
                items.append(
                    SearchResultItem(
                        url=url,
                        title=_first_str(o, _ITEM_TITLE),
                        snippet=snippet[:_SNIPPET_CHARS] if snippet else None,
                        published_at=parse_datetime(_first_str(o, _ITEM_DATE)),
                        author=_first_str(o, ("author",)),
                    )
                )
        if len(items) > len(best):
            best = items
    return best


_FIELD_LINE_RE = re.compile(
    r"^[ \t>*#-]*\**(?P<key>title|url|link|source|published|author|highlights?|snippet|"
    r"description|summary|text|content)\**\s*:\**\s*(?P<val>.*)$",
    re.IGNORECASE,
)
_HEADING_RE = re.compile(r"^#{1,6}\s*(?:\d+[.)]\s*)?(?P<title>[^|\n]+?)(?:\s*\|.*)?$")


def _block_items(text: str) -> list[SearchResultItem]:
    """``Title:`` / ``URL:`` blocks, and headings with a ``URL:`` line under them."""
    items: list[SearchResultItem] = []
    block: dict[str, Any] = {}
    snippet: list[str] = []

    def flush() -> None:
        if block.get("url"):
            body = " ".join(snippet).strip()
            items.append(
                SearchResultItem(
                    url=block["url"],
                    title=block.get("title"),
                    snippet=body[:_SNIPPET_CHARS] or None,
                    published_at=parse_datetime(block.get("published")),
                    author=block.get("author"),
                )
            )
        block.clear()
        snippet.clear()

    for line in text.splitlines():
        field = _FIELD_LINE_RE.match(line)
        heading = _HEADING_RE.match(line)
        if field:
            key, val = field["key"].lower(), field["val"].strip().strip("*").strip()
            if key == "title":
                if block.get("url") or block.get("title"):
                    flush()
                block["title"] = val or None
            elif key in ("url", "link", "source"):
                m = URL_RE.search(val)
                if m and "url" not in block:
                    block["url"] = m.group(0).rstrip(".,;:!?")
            elif key in ("published", "author"):
                if val:
                    block[key] = val
            elif val and block:
                snippet.append(val)  # "Highlights: ..." and its kin label the snippet
        elif heading:
            if block.get("url"):
                flush()
            block.setdefault("title", heading["title"].strip().strip("*"))
        elif line.strip() and block:
            snippet.append(line.strip())
    flush()
    return items


_LINK_LINE_RE = re.compile(
    r"^\s*(?:[-*+•]|\d+[.)])?\s*\**\[(?P<title>[^\]]+)\]\((?P<url>https?://[^)\s]+)\)\**(?P<rest>.*)$"
)


def _link_items(text: str) -> list[SearchResultItem]:
    """Markdown links that start a line (list entries): a prose answer with two links in
    its sentences is not a list of results."""
    items = []
    for line in text.splitlines():
        m = _LINK_LINE_RE.match(line)
        if m:
            rest = m["rest"].strip(" \t-:|")
            items.append(
                SearchResultItem(
                    url=m["url"], title=m["title"].strip(), snippet=rest[:_SNIPPET_CHARS] or None
                )
            )
    return items


def result_items(text: str, minimum: int) -> list[SearchResultItem]:
    """The result list ``text`` holds, tried as JSON, then blocks, then markdown links; the
    first reading with at least ``minimum`` items wins. Empty when none does."""
    for read in (_json_items, _block_items, _link_items):
        items = read(text)
        if len(items) >= minimum:
            return items
    return []
