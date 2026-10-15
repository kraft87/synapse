"""The big extraction prompts put their fixed rules first and the per-call content last.

A provider that caches a shared prompt prefix (OpenRouter's automatic prefix caching, an
API key with Anthropic caching) can only reuse what precedes the first thing that differs
between calls. Stage 3's rules are ~15K chars; with the session date, entities and summary
ahead of them, two calls shared ~700 chars. The check is the same for every prompt: render
it for two different inputs and what differs is only a short tail.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

from ingestion.extraction_llm import LLMExtractor
from ingestion.extraction_policy import build_batch_resolution_prompt
from ingestion.models import ExtractedEntity
from ingestion.prompts import dedupe_nodes, extract_edge_dates

# What a call may add after the shared rules: a handful of short items.
_TAIL = 600


def _differing_tail(a: str, b: str) -> int:
    return min(len(a), len(b)) - len(os.path.commonprefix([a, b]))


def _stage3(summary: str, date: str, entities: list[ExtractedEntity]) -> str:
    msg = MagicMock()
    msg.content = [MagicMock(text='{"entities": [], "facts": []}')]
    client = MagicMock()
    client.messages.create.return_value = msg
    LLMExtractor(llm_client=client).extract(summary, entities, session_date=date)
    return client.messages.create.call_args.kwargs["messages"][0]["content"]


def test_stage3_rules_are_the_shared_prefix():
    a = _stage3(
        "We moved the repo to Forgejo.",
        "2026-01-15",
        [ExtractedEntity(name="Forgejo", type="Tool", summary="")],
    )
    b = _stage3("The dentist moved my appointment.", "2026-03-02", [])
    assert len(a) > 10_000
    assert _differing_tail(a, b) < _TAIL


def _web(content: str, kind: str, title: str) -> str:
    msg = MagicMock()
    msg.content = [MagicMock(text='{"entities": [], "facts": []}')]
    client = MagicMock()
    client.messages.create.return_value = msg
    LLMExtractor(llm_client=client).extract_web(content, [], {"kind": kind, "title": title})
    return client.messages.create.call_args.kwargs["messages"][0]["content"]


def test_web_rules_are_the_shared_prefix():
    a = _web("Trafilatura gets F1 0.93.", "research_brief", "A brief")
    b = _web("Mem0 dropped its graph backend.", "web_page", "A blog post")
    assert _differing_tail(a, b) < _TAIL


def test_stage_6b_rules_are_the_shared_prefix():
    def render(fact: str) -> str:
        item = {"id": 0, "new_fact": fact, "existing_pool": [], "candidate_pool": []}
        return build_batch_resolution_prompt([item])[0]

    assert _differing_tail(render("X uses A"), render("Y uses a different thing entirely")) < _TAIL


def test_node_dedup_rules_are_the_shared_prefix():
    def render(name: str) -> str:
        item = {"id": 0, "name": name, "summary": "", "candidates": []}
        return "\n".join(m["content"] for m in dedupe_nodes.build_batch_prompt([item]))

    assert _differing_tail(render("NYC"), render("Postgres 17")) < _TAIL


def test_edge_date_rules_are_the_shared_prefix():
    def render(fact: str, ref: str) -> str:
        item = {"id": 0, "fact": fact}
        return "\n".join(m["content"] for m in extract_edge_dates.build_batch_prompt([item], ref))

    a = render("Alex joined Acme in 2020", "2026-05-17T10:00:00-04:00")
    b = render("The audit moved to May 3", "2026-06-01T09:00:00-04:00")
    assert _differing_tail(a, b) < _TAIL
