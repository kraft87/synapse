"""ingestion.capture.report: the dry run's account, built from the outcomes of a capture."""

from __future__ import annotations

from ingestion.capture import Capture, CaptureResult, Skip, ToolCall, report
from ingestion.capture.report import SAMPLES_PER_TOOL
from ingestion.web_artifacts import ArchiveEntry
from ingestion.web_extractors import SearchResultItem, SearchResultSet, WebScrape


def call(name: str) -> ToolCall:
    return ToolCall(
        tool_use_id="t",
        tool_name=name,
        tool_input={},
        result_text="",
        is_error=False,
        session_id="s",
        project=None,
        timestamp=None,
    )


def page(n: int) -> Capture:
    return Capture(WebScrape(tool_name="w", url=f"https://e.com/{n}", title=f"T{n}"), "generic")


def test_outcomes_are_grouped_by_tool_with_skips_by_reason_and_a_few_samples():
    results = SearchResultSet(
        tool_name="s",
        query="what is x",
        items=[SearchResultItem(url="https://e.com/1", title="t")],
    )
    archive = ArchiveEntry(
        "x_thread", "thread:x:1", "https://x.com/u/status/1", "A thread", None, "c"
    )
    outcomes = [(call("fetch"), page(i)) for i in range(5)]
    outcomes += [
        (call("fetch"), Skip("too_short")),
        (call("fetch"), Skip("too_short")),
        (call("search"), Capture(results, "generic")),
        (call("Bash"), Capture(archive, "x_thread")),
        (call("Bash"), Skip("shell")),
        # never research: counted, not listed
        (call("Read"), Skip("no_source")),
        (call("mcp__synapse__recall"), Skip("own_tool")),
        (call("Task"), Skip("subagent")),
    ]
    out = report(CaptureResult(outcomes=outcomes, errors=2))

    assert (out["calls"], out["ignored"], out["errors"]) == (13 + 2, 3, 2)
    assert set(out["tools"]) == {"fetch", "search", "Bash"}
    fetch = out["tools"]["fetch"]
    assert (fetch["captured"], fetch["skipped"]) == (5, {"too_short": 2})
    assert len(fetch["samples"]) == SAMPLES_PER_TOOL
    assert fetch["samples"][0] == {
        "url": "https://e.com/0",
        "title": "T0",
        "kind": "web_scrape",
        "destination": "searchable",
        "source": "generic",
    }
    # a result list has no page of its own: its query stands for it
    assert out["tools"]["search"]["samples"][0]["url"] is None
    assert out["tools"]["search"]["samples"][0]["title"] == "what is x"
    sample = out["tools"]["Bash"]["samples"][0]
    assert (sample["kind"], sample["destination"], sample["source"]) == (
        "x_thread",
        "archive",
        "x_thread",
    )
    assert out["tools"]["Bash"]["skipped"] == {"shell": 1}
