"""ingestion.capture.detect: web research recognised by the shape of the call, not its name."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingestion.capture import ToolCall, decide
from ingestion.capture import detect as detect_mod
from ingestion.capture.detect import Capture, Skip
from ingestion.web_extractors import SearchResultSet, WebScrape

FIXTURES = Path(__file__).parent / "fixtures" / "web_extractors"

PAGE = "# A Real Article\n\n" + "The measured latency was low and the design held up well. " * 12
LIST_JSON = json.dumps(
    {
        "results": [
            {"title": "One", "url": "https://one.com/a", "snippet": "first hit"},
            {"title": "Two", "url": "https://two.com/b", "description": "second hit"},
            {"title": "Three", "url": "https://three.com/c"},
        ]
    }
)


def call(name: str, tool_input: dict, text: str, **kw) -> ToolCall:
    fields = dict(
        tool_use_id="toolu_1",
        tool_name=name,
        tool_input=tool_input,
        result_text=text,
        is_error=False,
        session_id="s1",
        project="proj",
        timestamp=None,
    )
    return ToolCall(**{**fields, **kw})


def skipped(c: ToolCall) -> str:
    d = decide(c)
    assert isinstance(d, Skip), f"captured: {d}"
    return d.reason


def captured(c: ToolCall) -> Capture:
    d = decide(c)
    assert isinstance(d, Capture), f"skipped: {d}"
    return d


# ---------- by shape, under any tool name ----------


class TestPages:
    def test_a_url_in_and_a_page_out_is_captured_under_any_name(self):
        c = captured(call("mcp__acme__read_site", {"url": "https://example.com/post"}, PAGE))
        assert c.source == "generic"
        assert isinstance(c.parsed, WebScrape)
        assert c.parsed.url == "https://example.com/post"
        assert c.parsed.title == "A Real Article"
        assert not c.parsed.synthesized

    def test_a_json_envelope_is_unwrapped(self):
        text = json.dumps({"data": {"markdown": PAGE, "metadata": {"title": "From metadata"}}})
        c = captured(call("mcp__acme__scrape", {"urls": ["https://example.com/p"]}, text))
        assert c.parsed.content_markdown.startswith("# A Real Article")
        assert c.parsed.title == "From metadata"

    def test_a_json_encoded_url_list_of_one_is_a_source(self):
        c = captured(call("mcp__acme__crawl", {"urls": '["https://example.com/p"]'}, PAGE))
        assert c.parsed.url == "https://example.com/p"

    def test_urls_in_the_output_alone_never_qualify(self):
        text = PAGE + "\nsee https://example.com/a and https://example.com/b"
        assert skipped(call("mcp__acme__note", {"text": "x"}, text)) == "no_source"

    def test_several_input_urls_have_no_one_source_page(self):
        inp = {"urls": ["https://example.com/a", "https://example.com/b"]}
        assert skipped(call("mcp__acme__crawl", inp, PAGE)) == "ambiguous_source"

    def test_a_stub_is_too_short_and_urls_do_not_pad_it(self):
        assert (
            skipped(call("mcp__acme__fetch", {"url": "https://example.com"}, "ok")) == "too_short"
        )
        urls = "\n".join(f"https://example.com/page/{i}" for i in range(60))
        assert skipped(call("mcp__acme__map", {"url": "https://example.com"}, urls)) == "too_short"

    def test_the_body_keeps_its_start_when_over_the_byte_cap(self, monkeypatch):
        monkeypatch.setattr(detect_mod, "MAX_BODY_BYTES", 1000)
        text = "é" * 900 + PAGE  # multi-byte: the cut must not split a character
        c = captured(call("mcp__acme__fetch", {"url": "https://example.com"}, text))
        assert len(c.parsed.content_markdown.encode()) <= 1000
        assert c.parsed.content_markdown.startswith("éé")


class TestResultLists:
    def test_a_query_in_and_a_json_list_out_is_captured(self):
        c = captured(call("mcp__acme__find", {"query": "voyage"}, LIST_JSON))
        assert isinstance(c.parsed, SearchResultSet)
        assert c.parsed.query == "voyage"
        assert [i.url for i in c.parsed.items] == [
            "https://one.com/a",
            "https://two.com/b",
            "https://three.com/c",
        ]

    def test_title_url_blocks(self):
        text = (
            "Title: One\nURL: https://one.com/a\nPublished: 2026-01-02\nHighlights:\nfirst\n\n"
            "Title: Two\nURL: https://two.com/b\nAuthor: Ann\nHighlights:\nsecond\n"
        )
        c = captured(call("mcp__acme__find", {"q": "x"}, text))
        one, two = c.parsed.items
        assert (one.title, one.snippet) == ("One", "first")
        assert one.published_at is not None
        assert (two.title, two.author) == ("Two", "Ann")

    def test_headings_with_a_url_line(self):
        text = "### 1. First | site.org\n**URL:** https://a.com/x\ntext\n### 2. Second\nURL: https://b.com/y\n"
        c = captured(call("mcp__acme__find", {"objective": "x"}, text))
        assert [(i.title, i.url) for i in c.parsed.items] == [
            ("First", "https://a.com/x"),
            ("Second", "https://b.com/y"),
        ]

    def test_markdown_links_that_start_a_line(self):
        text = "1. [One](https://one.com/a) - the first\n2. [Two](https://two.com/b)\n"
        c = captured(call("mcp__acme__find", {"query": "x"}, text))
        assert [i.title for i in c.parsed.items] == ["One", "Two"]
        assert c.parsed.items[0].snippet == "the first"

    def test_links_inside_prose_are_not_a_list(self):
        text = "See [the docs](https://a.com/d) and [the blog](https://b.com/e) for more."
        assert skipped(call("mcp__acme__ask", {"query": "x"}, text)) == "no_shape"

    def test_a_single_result_is_not_a_list(self):
        text = json.dumps([{"title": "Only", "url": "https://one.com/a"}])
        assert skipped(call("mcp__acme__find", {"query": "x"}, text)) == "no_shape"

    def test_items_with_neither_title_nor_snippet_do_not_count(self):
        text = json.dumps([{"url": "https://a.com"}, {"url": "https://b.com"}])
        assert skipped(call("mcp__acme__find", {"query": "x"}, text)) == "no_shape"

    def test_private_items_are_dropped_and_too_few_left_skips(self):
        mixed = json.dumps(
            [
                {"title": "A", "url": "https://a.com"},
                {"title": "Wiki", "url": "http://wiki.internal/x"},
                {"title": "B", "url": "https://b.com"},
                {"title": "Me", "url": "http://localhost:8000/"},
            ]
        )
        c = captured(call("mcp__acme__find", {"query": "x"}, mixed))
        assert [i.title for i in c.parsed.items] == ["A", "B"]
        private = json.dumps(
            [
                {"title": "A", "url": "https://a.com"},
                {"title": "Wiki", "url": "http://wiki.internal/x"},
            ]
        )
        assert skipped(call("mcp__acme__find", {"query": "x"}, private)) == "non_public_url"

    def test_duplicate_urls_collapse(self):
        text = json.dumps(
            [{"title": f"t{i}", "url": "https://a.com/x/"} for i in range(3)]
            + [{"title": "b", "url": "https://b.com"}]
        )
        c = captured(call("mcp__acme__find", {"query": "x"}, text))
        assert [i.url for i in c.parsed.items] == ["https://a.com/x/", "https://b.com"]


# ---------- the guardrails ----------


class TestGuardrails:
    @pytest.mark.parametrize(
        "name,reason",
        [
            ("mcp__Synapse__recall", "own_tool"),
            ("Task", "subagent"),
            ("mcp__playwright__browser_navigate", "browser"),
            ("mcp__claude_ai_Gmail__get_thread", "denied_family"),
            ("mcp__github__get_file_contents", "denied_family"),
            ("Bash", "shell"),
        ],
    )
    def test_excluded_tools_are_never_captured_even_with_a_page_shape(self, name, reason):
        assert skipped(call(name, {"url": "https://example.com/a", "query": "q"}, PAGE)) == reason

    def test_a_shell_command_is_not_captured_by_the_generic_rule(self):
        out = "curl output\n" + PAGE
        assert skipped(call("Bash", {"command": "curl -s https://example.com/a"}, out)) == "shell"

    def test_a_private_source_url_is_not_captured(self):
        for url in (
            "http://localhost:3000/",
            "http://192.168.1.9/x",
            "http://nas.lan/",
            "http://wiki/",
        ):
            assert skipped(call("mcp__acme__fetch", {"url": url}, PAGE)) == "non_public_url"

    def test_the_private_url_check_applies_to_the_known_parsers_too(self):
        assert (
            skipped(call("WebFetch", {"url": "http://localhost:3000/"}, PAGE)) == "non_public_url"
        )

    @pytest.mark.parametrize(
        "text,reason",
        [
            ("", "empty"),
            ("   \n", "empty"),
            ("REDIRECT DETECTED: the URL redirects to https://other.example", "redirect"),
            ("Just a moment...\nVerify you are human", "blocked_page"),
            ("Please sign in to continue", "blocked_page"),
            ("MCP error -32602: invalid arguments", "error_page"),
            (
                "<persisted-output>\nOutput too large (61 KB). Full output saved to: /home/u/x.txt\n",
                "persisted_output",
            ),
        ],
    )
    def test_error_captcha_login_and_redirect_pages_are_not_content(self, text, reason):
        assert skipped(call("mcp__acme__fetch", {"url": "https://example.com"}, text)) == reason

    def test_a_tool_error_flag_is_an_error(self):
        c = call("mcp__acme__fetch", {"url": "https://example.com"}, PAGE, is_error=True)
        assert skipped(c) == "error_page"

    def test_an_enormous_result_is_not_parsed(self, monkeypatch):
        monkeypatch.setattr(detect_mod, "MAX_RESULT_CHARS", 100)
        assert (
            skipped(call("mcp__acme__fetch", {"url": "https://example.com"}, PAGE)) == "too_large"
        )


# ---------- the known parsers as refinements ----------


class TestRefinements:
    def test_a_known_operation_is_refined_whatever_the_connector_is_called(self):
        text = "# Page Title\nURL: https://target.com\nAuthor: Someone\n\n" + "body text " * 80
        for name in ("mcp__claude_ai_Exa_search__crawling_exa", "mcp__renamed__crawling_exa"):
            c = captured(call(name, {"urls": '["https://target.com"]'}, text))
            assert c.source == "crawling_exa"
            assert c.parsed.author == "Someone"

    def test_webfetch_is_marked_synthesized(self):
        c = captured(call("WebFetch", {"url": "https://example.com", "prompt": "sum"}, PAGE))
        assert c.source == "WebFetch" and c.parsed.synthesized

    def test_a_refined_parser_whose_format_drifted_falls_back_to_the_shapes(self):
        # firecrawl_search now answers with a shape its parser does not know.
        drifted = LIST_JSON.replace('"results"', '"hits"')
        c = captured(call("mcp__firecrawl__firecrawl_search", {"query": "q"}, drifted))
        assert c.source == "generic" and len(c.parsed.items) == 3

    def test_a_deep_research_report_has_no_page_but_is_kept(self):
        report = json.dumps({"status": "completed", "researchId": "r_1", "report": PAGE})
        c = captured(call("mcp__Exa_search__deep_researcher_check", {"researchId": "r_1"}, report))
        assert c.parsed.url == "exa://research/r_1" and c.parsed.synthesized

    def test_a_deep_research_poll_that_is_not_done_is_not_ready(self):
        running = json.dumps({"status": "running", "researchId": "r_1"})
        assert skipped(call("mcp__Exa_search__deep_researcher_check", {}, running)) == "not_ready"

    def test_a_job_ref_is_kept(self):
        out = json.dumps({"success": True, "researchId": "r_1", "instructions": "look"})
        c = captured(call("mcp__Exa_search__deep_researcher_start", {"instructions": "look"}, out))
        assert c.parsed.kind == "research_job_ref"


# ---------- real tool output ----------


def _samples(slug: str, key: str = "happy_samples") -> list[dict]:
    return json.loads((FIXTURES / f"{slug}.json").read_text())[key]


@pytest.mark.parametrize(
    "slug",
    [
        "firecrawl_firecrawl_search",
        "Exa_search_web_search_exa",
        "Exa_search_web_search_advanced_exa",
        "Exa_search_deep_search_exa",
        "Exa_search_people_search_exa",
        "WebSearch",
    ],
)
def test_the_shapes_find_what_each_search_parser_finds_under_a_renamed_tool(slug):
    """A tool nobody listed returns the same result lists; the generic reader gets the same URLs."""
    for s in _samples(slug):
        via_name = captured(call(f"mcp__x__{slug}", s["input"], s["result_text"]))
        via_shape = captured(call("mcp__unlisted__lookup", s["input"], s["result_text"]))
        assert via_shape.source == "generic"
        assert [i.url for i in via_shape.parsed.items] == [i.url for i in via_name.parsed.items]


def test_the_shapes_capture_scraped_pages_under_a_renamed_tool():
    for slug in ("firecrawl_firecrawl_scrape", "WebFetch"):
        for s in _samples(slug):
            d = decide(call("mcp__unlisted__read_page", s["input"], s["result_text"]))
            # A real 404 page that a scraper returned as content is a wall, not research.
            assert isinstance(d, Capture) or d.reason in ("blocked_page", "too_short"), d


def test_real_failures_are_skipped():
    for s in _samples("WebFetch", "error_samples") + _samples(
        "Exa_search_crawling_exa", "error_samples"
    ):
        assert isinstance(
            decide(call("mcp__unlisted__read_page", s["input"], s["result_text"])), Skip
        )
