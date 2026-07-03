"""Tests for ingestion.web_extractors against real JSONL fixtures."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from ingestion.web_extractors import (
    ExtractError,
    ResearchJobRef,
    SearchResultSet,
    WebScrape,
    _parse_datetime,
    _strip_firecrawl_boilerplate,
    extract,
    parse_crawling_exa,
    parse_deep_researcher_check,
    parse_deep_researcher_start,
    parse_deep_search,
    parse_exa_advanced,
    parse_exa_text_blocks,
    parse_firecrawl_scrape,
    parse_firecrawl_search,
    parse_webfetch,
    parse_websearch,
)

FIXTURES = Path(__file__).parent / "fixtures" / "web_extractors"


def _load(slug: str) -> dict:
    return json.loads((FIXTURES / f"{slug}.json").read_text())


# ---------- Datetime helper ----------


class TestParseDatetime:
    def test_iso_z(self):
        dt = _parse_datetime("2026-04-23T03:00:00.000Z")
        assert dt is not None and dt.year == 2026 and dt.month == 4

    def test_date_only(self):
        dt = _parse_datetime("2012-04-14")
        assert dt is not None and dt.year == 2012

    def test_na_returns_none(self):
        assert _parse_datetime("N/A") is None
        assert _parse_datetime("") is None
        assert _parse_datetime(None) is None


# ---------- Firecrawl boilerplate stripper ----------


class TestFirecrawlBoilerplate:
    def test_strips_github_skip_to_content(self):
        md = "[Skip to content](https://example.com#start)\n\nReal content here"
        assert "[Skip to content]" not in _strip_firecrawl_boilerplate(md)
        assert "Real content here" in _strip_firecrawl_boilerplate(md)

    def test_strips_signed_in_chatter(self):
        # Real firecrawl-on-GitHub output: each "Reload" is a markdown link with a URL.
        md = (
            "You signed in with another tab or window. [Reload](https://github.com/x) to refresh your session."
            "You signed out in another tab or window. [Reload](https://github.com/x) to refresh your session."
            "You switched accounts on another tab or window. [Reload](https://github.com/x) to refresh your session."
            "Dismiss alert\n\nReal content"
        )
        out = _strip_firecrawl_boilerplate(md)
        assert "You signed in" not in out
        assert "Dismiss alert" not in out
        assert "Real content" in out


# ---------- WebFetch ----------


class TestWebFetch:
    def test_redirect_is_error(self):
        text = "REDIRECT DETECTED: The URL redirects to a different host.\n\nOriginal URL: https://a\nRedirect URL: https://b"
        out = parse_webfetch({"url": "https://a"}, text)
        assert isinstance(out, ExtractError)
        assert out.reason == "redirect_detected"

    def test_happy_path_marks_synthesized(self):
        text = "# Some answer\n\nThe documentation says X about Y."
        out = parse_webfetch({"url": "https://example.com", "prompt": "summarize"}, text)
        assert isinstance(out, WebScrape)
        assert out.synthesized is True
        assert out.url == "https://example.com"
        assert out.prompt == "summarize"

    def test_real_fixtures(self):
        # Fixture buckets are heuristic; we test per-sample correctness:
        # samples containing "REDIRECT DETECTED" parse to ExtractError,
        # the rest parse to WebScrape.
        f = _load("WebFetch")
        for s in f["happy_samples"] + f["error_samples"]:
            out = parse_webfetch(s["input"], s["result_text"])
            text = s["result_text"]
            if "REDIRECT DETECTED" in text[:200]:
                assert isinstance(out, ExtractError)
                assert out.reason == "redirect_detected"
            else:
                assert isinstance(out, WebScrape), f"unexpected error: {out!r}"
                assert out.url
                assert out.synthesized is True


# ---------- firecrawl_scrape ----------


class TestFirecrawlScrape:
    def test_unwraps_markdown_envelope(self):
        text = json.dumps({"markdown": "# Hello\n\nWorld"})
        out = parse_firecrawl_scrape({"url": "https://e.com"}, text)
        assert isinstance(out, WebScrape)
        assert out.synthesized is False
        assert "Hello" in out.content_markdown
        assert "World" in out.content_markdown

    def test_raw_markdown_passthrough(self):
        text = "# Just markdown\n\nContent"
        out = parse_firecrawl_scrape({"url": "https://e.com"}, text)
        assert isinstance(out, WebScrape)
        assert "Just markdown" in out.content_markdown

    def test_real_fixtures(self):
        f = _load("firecrawl_firecrawl_scrape")
        for s in f["happy_samples"]:
            out = parse_firecrawl_scrape(s["input"], s["result_text"])
            assert isinstance(out, WebScrape), f"happy failed: {s['result_text'][:80]}"
            assert out.synthesized is False
            assert out.url
            assert len(out.content_markdown) > 20
        for s in f["error_samples"]:
            out = parse_firecrawl_scrape(s["input"], s["result_text"])
            # "Error: result (...) exceeds maximum allowed tokens" — persisted output
            # is detected at the dispatcher level; raw parser sees the error preamble.
            assert isinstance(out, ExtractError | WebScrape)


# ---------- crawling_exa ----------


class TestCrawlingExa:
    def test_real_fixtures(self):
        # Errors here include MCP-validation failures (ExtractError) AND
        # persisted-output redirects (treated as content; dispatch loads from
        # disk). Test per-sample.
        f = _load("Exa_search_crawling_exa")
        for s in f["happy_samples"]:
            out = parse_crawling_exa(s["input"], s["result_text"])
            assert isinstance(out, WebScrape)
            assert out.synthesized is False
            assert out.url
        for s in f["error_samples"]:
            text = s["result_text"]
            out = parse_crawling_exa(s["input"], text)
            if "MCP error" in text[:200] or "validation failed" in text[:200]:
                assert isinstance(out, ExtractError)
            elif "<persisted-output>" in text[:200]:
                assert isinstance(out, WebScrape)  # raw parser; dispatch loads file
            else:
                # Some other format — at minimum doesn't crash
                assert out is not None

    def test_url_from_json_input(self):
        text = "# Page Title\nURL: https://target.com\nAuthor: Someone\n\nbody body body"
        out = parse_crawling_exa({"urls": '["https://target.com"]'}, text)
        assert isinstance(out, WebScrape)
        assert out.url == "https://target.com"
        assert out.title == "Page Title"


# ---------- WebSearch ----------


class TestWebSearch:
    def test_parses_links_json(self):
        text = (
            'Web search results for query: "x"\n\n'
            'Links: [{"title":"A","url":"https://a"},{"title":"B","url":"https://b"}]'
        )
        out = parse_websearch({"query": "x"}, text)
        assert isinstance(out, SearchResultSet)
        assert len(out.items) == 2
        assert out.items[0].url == "https://a"
        assert out.items[0].position == 1
        assert out.items[1].url == "https://b"

    def test_real_fixtures(self):
        f = _load("WebSearch")
        for s in f["happy_samples"]:
            out = parse_websearch(s["input"], s["result_text"])
            assert isinstance(out, SearchResultSet)
            assert out.query == s["input"]["query"]
            assert len(out.items) > 0
            for it in out.items:
                assert it.url.startswith("http")


# ---------- firecrawl_search ----------


class TestFirecrawlSearch:
    def test_parses_web_array(self):
        text = json.dumps(
            {
                "web": [
                    {"url": "https://a", "title": "A", "description": "Adesc", "position": 1},
                    {"url": "https://b", "title": "B"},
                ]
            }
        )
        out = parse_firecrawl_search({"query": "x"}, text)
        assert isinstance(out, SearchResultSet)
        assert len(out.items) == 2
        assert out.items[0].snippet == "Adesc"
        assert out.items[0].position == 1

    def test_real_fixtures(self):
        f = _load("firecrawl_firecrawl_search")
        for s in f["happy_samples"]:
            out = parse_firecrawl_search(s["input"], s["result_text"])
            assert isinstance(out, SearchResultSet)
            # Skip degenerately small results ("[]" etc) — they're not really happy
            if s["result_chars"] >= 100:
                assert len(out.items) > 0, (
                    f"no items from {s['result_chars']} chars: {s['result_text'][:120]}"
                )


# ---------- Exa text blocks (web_search_exa, people_search_exa) ----------


class TestExaTextBlocks:
    def test_parses_block_format(self):
        text = (
            "Title: A title\n"
            "URL: https://a.example\n"
            "Published: 2026-04-23T03:00:00.000Z\n"
            "Author: Someone\n"
            "Highlights:\nbody of highlight\n\n"
            "Title: B title\n"
            "URL: https://b.example\n"
            "Published: N/A\n"
            "Author: N/A\n"
            "Highlights:\nmore body\n"
        )
        out = parse_exa_text_blocks(
            "mcp__claude_ai_Exa_search__web_search_exa", {"query": "q"}, text
        )
        assert isinstance(out, SearchResultSet)
        assert len(out.items) == 2
        assert out.items[0].url == "https://a.example"
        assert out.items[0].title == "A title"
        assert out.items[0].author == "Someone"
        assert isinstance(out.items[0].published_at, datetime)
        assert out.items[1].author is None  # N/A filtered

    def test_real_fixtures_web_search(self):
        f = _load("Exa_search_web_search_exa")
        for s in f["happy_samples"]:
            out = parse_exa_text_blocks(
                "mcp__claude_ai_Exa_search__web_search_exa", s["input"], s["result_text"]
            )
            assert isinstance(out, SearchResultSet)
            assert len(out.items) > 0, f"no items parsed from {s['result_text'][:120]}"

    def test_real_fixtures_people_search(self):
        f = _load("Exa_search_people_search_exa")
        for s in f["happy_samples"]:
            out = parse_exa_text_blocks(
                "mcp__claude_ai_Exa_search__people_search_exa", s["input"], s["result_text"]
            )
            assert isinstance(out, SearchResultSet)
            assert len(out.items) > 0


# ---------- Exa advanced (JSON envelope) ----------


class TestExaAdvanced:
    def test_real_fixtures(self):
        f = _load("Exa_search_web_search_advanced_exa")
        for s in f["happy_samples"]:
            out = parse_exa_advanced(s["input"], s["result_text"])
            assert isinstance(out, SearchResultSet)
            assert len(out.items) > 0
            for it in out.items:
                assert it.url.startswith("http")


# ---------- Deep search ----------


class TestDeepSearch:
    def test_real_fixtures(self):
        f = _load("Exa_search_deep_search_exa")
        for s in f["happy_samples"]:
            out = parse_deep_search(s["input"], s["result_text"])
            assert isinstance(out, SearchResultSet)
            assert len(out.items) > 0, f"no items in {s['result_text'][:200]}"


# ---------- Deep researcher ----------


class TestDeepResearcher:
    def test_start_returns_job_ref(self):
        f = _load("Exa_search_deep_researcher_start")
        for s in f["happy_samples"]:
            out = parse_deep_researcher_start(s["input"], s["result_text"])
            assert isinstance(out, ResearchJobRef)
            assert out.research_id.startswith("r_")

    def test_check_polling_is_error(self):
        text = json.dumps(
            {
                "success": True,
                "status": "running",
                "researchId": "r_abc",
                "message": "Research in progress.",
            }
        )
        out = parse_deep_researcher_check({}, text)
        assert isinstance(out, ExtractError)
        assert out.reason == "status_running"

    def test_check_completed_is_scrape(self):
        text = json.dumps(
            {
                "success": True,
                "status": "completed",
                "researchId": "r_abc",
                "report": "# Final\n\nresult",
            }
        )
        out = parse_deep_researcher_check({}, text)
        assert isinstance(out, WebScrape)
        assert "Final" in out.content_markdown
        assert out.synthesized is True


# ---------- Dispatch + persisted output ----------


class TestDispatch:
    def test_unknown_tool_returns_error(self):
        out = extract("NotARealTool", {}, "some content")
        assert isinstance(out, ExtractError)
        assert out.reason == "unknown_tool"

    def test_dispatch_picks_correct_parser(self):
        out = extract(
            "WebSearch",
            {"query": "q"},
            'Web search results for query: "q"\n\nLinks: [{"title":"A","url":"https://a"}]',
        )
        assert isinstance(out, SearchResultSet)
        assert out.items[0].url == "https://a"

    def test_persisted_output_detection_records_path(self, tmp_path):
        # Build a fake persisted file with a real Exa-style result
        persisted = tmp_path / "tool-results" / "abc.txt"
        persisted.parent.mkdir(parents=True, exist_ok=True)
        persisted.write_text(
            "Title: From disk\nURL: https://disk.example\nPublished: 2026-01-01\nAuthor: N/A\nHighlights:\nfrom-disk body\n"
        )
        preamble = (
            f"<persisted-output>\nOutput too large (61.4KB). Full output saved to: {persisted}\n\n"
            'Preview (first 2KB):\n[\n  {\n    "type": "text"\n  }\n]'
        )
        out = extract("mcp__claude_ai_Exa_search__web_search_exa", {"query": "q"}, preamble)
        assert isinstance(out, SearchResultSet)
        assert out.persisted_output_path == str(persisted)
        assert any(it.url == "https://disk.example" for it in out.items)

    def test_persisted_output_missing_file_falls_back_to_preview(self, tmp_path):
        preamble = (
            f"<persisted-output>\nOutput too large. Full output saved to: {tmp_path}/missing.txt\n\n"
            "Preview (first 2KB):\nTitle: Inline\nURL: https://inline.example\nPublished: N/A\nAuthor: N/A\nHighlights:\nstub\n"
        )
        out = extract("mcp__claude_ai_Exa_search__web_search_exa", {"query": "q"}, preamble)
        # Even if the file's missing, we should fall back to parsing the preview
        # and record the attempted path.
        assert isinstance(out, SearchResultSet)
        assert out.persisted_output_path is not None


# ---------- End-to-end coverage smoke test ----------


class TestCoverageSmoke:
    """Every happy fixture should produce a non-error parsed result via dispatch."""

    @pytest.mark.parametrize(
        "slug,tool",
        [
            ("WebFetch", "WebFetch"),
            ("WebSearch", "WebSearch"),
            ("firecrawl_firecrawl_scrape", "mcp__claude_ai_firecrawl__firecrawl_scrape"),
            ("firecrawl_firecrawl_search", "mcp__claude_ai_firecrawl__firecrawl_search"),
            ("Exa_search_web_search_exa", "mcp__claude_ai_Exa_search__web_search_exa"),
            (
                "Exa_search_web_search_advanced_exa",
                "mcp__claude_ai_Exa_search__web_search_advanced_exa",
            ),
            ("Exa_search_deep_search_exa", "mcp__claude_ai_Exa_search__deep_search_exa"),
            ("Exa_search_people_search_exa", "mcp__claude_ai_Exa_search__people_search_exa"),
            ("Exa_search_crawling_exa", "mcp__claude_ai_Exa_search__crawling_exa"),
            (
                "Exa_search_deep_researcher_start",
                "mcp__claude_ai_Exa_search__deep_researcher_start",
            ),
        ],
    )
    def test_all_happy_fixtures_dispatch_cleanly(self, slug, tool):
        f = _load(slug)
        for s in f["happy_samples"]:
            out = extract(tool, s["input"], s["result_text"])
            assert not isinstance(out, ExtractError), (
                f"{tool} happy sample produced ExtractError: {out!r}\n"
                f"input={s['input']}\nresult preview: {s['result_text'][:200]}"
            )
