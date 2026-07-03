"""The temporal-intent gate for recall()'s timeline leg — pure regex tests.
The leg itself reuses TimelineRecall (integration-covered); what needs pinning is
WHEN it fires: temporal phrasing yes, ordinary topical queries no."""

from __future__ import annotations

from mcp_server.recall import _TEMPORAL_RE

FIRES = [
    "when did we ship the dating fix?",
    "how long between the MoMA visit and the exhibit?",
    "what did I do last week",
    "how many days ago did the login work land",
    "what did we build recently? show the timeline",
    "When was the N=6 change deployed?",
]
QUIET = [
    "what's the state of the login work",
    "how does the extractor dedup facts",
    "show me the browser-free login design",
    "why did recall miss the Kobo research",
]


def test_temporal_queries_fire():
    for q in FIRES:
        assert _TEMPORAL_RE.search(q), q


def test_topical_queries_stay_quiet():
    for q in QUIET:
        assert not _TEMPORAL_RE.search(q), q
