"""The dry run's report: what a live capture would do with a batch of tool calls, by tool.

Built from the :class:`~ingestion.capture.store.CaptureResult` of ``capture_calls(...,
dry_run=True)``, so it is the live decision, not a second opinion. Per tool: how many calls
would be captured (with a few samples: URL, title, kind, where it would go and by which
parser or rule), and how many would be skipped, by reason (``ingestion.capture.detect`` lists
them). Calls with nothing to capture (no URL or query in the input, Synapse's own tools,
sub-agent launches: most of a transcript) are counted in ``ignored`` and not listed.
"""

from __future__ import annotations

from typing import Any

from ingestion.capture.detect import Capture, Skip
from ingestion.capture.store import CaptureResult

#: Skip reasons of a call that was never research.
NOT_RESEARCH = frozenset({"no_source", "own_tool", "subagent"})
#: Captured calls shown per tool.
SAMPLES_PER_TOOL = 3


def _sample(capture: Capture) -> dict[str, Any]:
    parsed = capture.parsed
    # A result list has no page of its own: the query stands for it.
    url = getattr(parsed, "url", None)
    title = getattr(parsed, "title", None) or getattr(parsed, "query", None)
    return {
        "url": url,
        "title": title,
        "kind": parsed.kind,
        "destination": capture.destination,
        "source": capture.source,
    }


def report(result: CaptureResult) -> dict[str, Any]:
    """``{"calls", "ignored", "errors", "tools": {name: {"captured", "skipped", "samples"}}}``."""
    tools: dict[str, dict[str, Any]] = {}
    ignored = 0
    for call, outcome in result.outcomes:
        if isinstance(outcome, Skip) and outcome.reason in NOT_RESEARCH:
            ignored += 1
            continue
        entry = tools.setdefault(call.tool_name, {"captured": 0, "skipped": {}, "samples": []})
        if isinstance(outcome, Skip):
            entry["skipped"][outcome.reason] = entry["skipped"].get(outcome.reason, 0) + 1
            continue
        entry["captured"] += 1
        if len(entry["samples"]) < SAMPLES_PER_TOOL:
            entry["samples"].append(_sample(outcome))
    return {
        "calls": len(result.outcomes) + result.errors,
        "ignored": ignored,
        "errors": result.errors,
        "tools": tools,
    }
