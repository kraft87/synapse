"""Decide whether a tool call is web research worth storing, by the SHAPE of the call.

Never by tool name: a call is captured when its input carries a URL or a search query AND
its output is page-shaped (one source URL plus a substantial body) or result-list-shaped
(two or more items, each a public URL with a title or snippet). The parsers in
``ingestion.web_extractors`` are optional refinements for the operations they know, picked by
operation name whatever the connector is called; the rest goes through the generic shapes.

:func:`decide` returns what to store (:class:`Capture`) or why not (:class:`Skip`), and does
nothing else, so the dry run reports exactly what a live capture would do. Skip reasons:

* ``own_tool`` ``subagent`` ``browser`` ``denied_family`` ``shell``: the tool is never captured
  (``ingestion.capture.guards``);
* ``no_source`` the input carries neither a URL nor a query; ``ambiguous_source`` it carries
  several URLs, so no page is the source of the output;
* ``non_public_url`` a source URL is not on the public internet;
* ``empty`` ``error_page`` ``blocked_page`` ``redirect`` ``not_ready`` ``persisted_output``
  ``too_large``: the output is not content;
* ``too_short`` the page body is under :data:`MIN_BODY_CHARS`; ``no_shape`` the output is
  neither a page nor a list of at least :data:`MIN_LIST_ITEMS` results.

The store step adds ``private``, ``duplicate`` and the caps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ingestion.capture.calls import ToolCall
from ingestion.capture.guards import is_public_url, tool_exclusion
from ingestion.capture.shapes import URL_RE, input_queries, input_urls, page_body, result_items
from ingestion.web_artifacts import canonicalize_url
from ingestion.web_extractors import (
    ExtractError,
    ResearchJobRef,
    SearchResultItem,
    SearchResultSet,
    WebScrape,
    error_reason,
    refine,
    strip_boilerplate,
    tool_id,
)

#: A page body shorter than this (links and URLs not counted) is a stub, a redirect notice or
#: an answer the episode's own result preview already holds in full.
MIN_BODY_CHARS = 400
MIN_LIST_ITEMS = 2
MAX_LIST_ITEMS = 50
#: Per stored body, in UTF-8 bytes; a longer page keeps its start.
MAX_BODY_BYTES = 200_000
#: A tool result longer than this is not parsed at all.
MAX_RESULT_CHARS = 2_000_000

# Exa's deep research reports have no page: they are named by a synthetic address.
_SYNTHETIC_URL_PREFIX = "exa://research/"

# "Output too large (N KB). Full output saved to: <path>": a file on the client's machine.
_PERSISTED_REDIRECT = re.compile(r"<persisted-output>|Output too large|Error: result \(", re.I)


@dataclass(frozen=True)
class Skip:
    reason: str


@dataclass(frozen=True)
class Capture:
    parsed: WebScrape | SearchResultSet | ResearchJobRef
    #: "generic", or the name of the operation whose parser refined it
    source: str


def decide(call: ToolCall) -> Capture | Skip:
    """What to store for ``call``, or why nothing."""
    if reason := tool_exclusion(call.tool_name, call.tool_input):
        return Skip(reason)
    text = call.result_text
    if call.is_error:
        return Skip("error_page")
    if not text.strip():
        return Skip("empty")
    if len(text) > MAX_RESULT_CHARS:
        return Skip("too_large")
    if _PERSISTED_REDIRECT.search(text[:500]):
        return Skip("persisted_output")

    refined = refine(call.tool_name, call.tool_input, text)
    if isinstance(refined, ExtractError):
        return Skip(_error_skip(refined.reason))
    if refined is not None and not (isinstance(refined, SearchResultSet) and not refined.items):
        return _vet(refined, tool_id(call.tool_name))
    # No parser for this operation, or its format changed: the shapes decide.
    parsed = _generic(call)
    return parsed if isinstance(parsed, Skip) else _vet(parsed, "generic")


def _error_skip(reason: str) -> str:
    if reason == "redirect_detected":
        return "redirect"
    if reason in ("empty", "blocked_page"):
        return reason
    return "not_ready" if reason.startswith("status_") else "error_page"


def _generic(call: ToolCall) -> WebScrape | SearchResultSet | Skip:
    text = call.result_text
    urls = input_urls(call.tool_input)
    if urls:
        if len(urls) > 1:
            return Skip("ambiguous_source")
        body, title = page_body(text)
        return WebScrape(
            tool_name=call.tool_name,
            url=urls[0],
            title=title,
            content_markdown=body,
            raw_chars=len(text),
        )
    queries = input_queries(call.tool_input)
    if not queries:
        return Skip("no_source")
    items = result_items(text, MIN_LIST_ITEMS)
    if not items:
        return Skip("no_shape")
    return SearchResultSet(
        tool_name=call.tool_name, query="; ".join(queries)[:500], items=items, raw_chars=len(text)
    )


def _vet(parsed: WebScrape | SearchResultSet | ResearchJobRef, source: str) -> Capture | Skip:
    """The guards and quality floors every parse meets, refined or generic."""
    if isinstance(parsed, ResearchJobRef):
        return Capture(parsed, source)
    if isinstance(parsed, WebScrape):
        if not parsed.url:
            return Skip("no_source")
        if not (is_public_url(parsed.url) or parsed.url.startswith(_SYNTHETIC_URL_PREFIX)):
            return Skip("non_public_url")
        body = strip_boilerplate(parsed.content_markdown)
        if reason := error_reason(body):
            return Skip(_error_skip(reason))
        if len(URL_RE.sub("", body)) < MIN_BODY_CHARS:
            return Skip("too_short")
        body = body.encode()[:MAX_BODY_BYTES].decode(errors="ignore")
        return Capture(parsed.model_copy(update={"content_markdown": body}), source)

    items: list[SearchResultItem] = []
    seen: set[str] = set()
    private = 0
    for item in parsed.items:
        if not is_public_url(item.url):
            private += 1
            continue
        key = canonicalize_url(item.url)
        if key and key not in seen and (item.title or item.snippet):
            seen.add(key)
            items.append(item)
    if len(items) < MIN_LIST_ITEMS:
        return Skip("non_public_url" if private else "no_shape")
    return Capture(parsed.model_copy(update={"items": items[:MAX_LIST_ITEMS]}), source)
