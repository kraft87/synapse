"""ingestion.web_context: context prefixes for web chunks, the poller's per-cycle step.

A fake database and a scripted LLM: the retry loop is parse_with_retry, a batch the model
keeps answering badly is stored without context (not retried at every cycle's cost), and an
LLM outage or exhausted quota stops the cycle with the pages still waiting.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

from ingestion.llm_client import LLMUnavailableError, UsageLimitError
from ingestion.web_context import contextualize_artifact, contextualize_pending, parse_contexts


class _Db:
    """The slice of Database the step uses: pages, their chunks, and what it stored."""

    def __init__(self, chunks_by_page: dict[int, int]) -> None:
        self.chunks = {
            page: [
                {"id": page * 100 + i, "idx": i, "content": f"chunk {page}.{i}"} for i in range(n)
            ]
            for page, n in chunks_by_page.items()
        }
        self.stored: dict[int, str] = {}

    def get_web_chunks(self, artifact_id):
        return self.chunks[artifact_id]

    def set_web_chunk_contexts(self, rows):
        self.stored.update(dict(rows))

    def artifacts_to_contextualize(self, limit):
        pending = [
            p for p, cs in self.chunks.items() if any(c["id"] not in self.stored for c in cs)
        ]
        return [{"id": p, "content_markdown": f"page {p}"} for p in pending][:limit]


class _LLM:
    """Replies with the scripted texts in order; an Exception in the script is raised."""

    def __init__(self, script):
        self._script = list(script)
        self.prompts: list[str] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, *, messages, **kwargs):
        self.prompts.append(messages[0]["content"])
        reply = self._script.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(content=[SimpleNamespace(text=reply)])


def _reply(*pairs):
    return json.dumps({"contexts": [{"idx": i, "context": c} for i, c in pairs]})


def test_each_chunk_gets_the_context_the_model_wrote_for_it():
    db = _Db({1: 2})
    llm = _LLM([_reply((0, "about a"), (1, "about b"))])
    assert contextualize_artifact(db, llm, 1, "doc", "m") == 2
    assert db.stored == {100: "about a", 101: "about b"}
    assert "chunk 1.0" in llm.prompts[0] and "<document>\ndoc" in llm.prompts[0]


def test_a_single_chunk_page_needs_no_model():
    db, llm = _Db({1: 1}), _LLM([])
    assert contextualize_artifact(db, llm, 1, "doc", "m") == 1
    assert db.stored == {100: ""} and llm.prompts == []


def test_invalid_reply_is_fed_back_and_retried():
    db = _Db({1: 2})
    llm = _LLM(["```not json```", _reply((0, "about a"), (1, "about b"))])
    assert contextualize_artifact(db, llm, 1, "doc", "m") == 2
    assert db.stored == {100: "about a", 101: "about b"}
    assert len(llm.prompts) == 2 and "not valid JSON" in llm.prompts[1]


def test_unusable_replies_after_the_retries_store_the_chunks_without_context(caplog):
    db = _Db({1: 2})
    llm = _LLM(["not json"] * 3)
    with caplog.at_level(logging.WARNING, logger="ingestion.web_context"):
        assert contextualize_artifact(db, llm, 1, "doc", "m") == 2
    assert len(llm.prompts) == 3
    # '' is "no context", not "not yet": the page is not asked about again
    assert db.stored == {100: "", 101: ""}
    assert db.artifacts_to_contextualize(10) == []
    assert "web artifact 1" in caplog.text and "without one" in caplog.text


def test_a_page_is_sent_in_batches_and_a_bad_batch_costs_only_its_own_chunks():
    db = _Db({1: 31})
    first = _reply(*[(i, f"ctx {i}") for i in range(30)])
    llm = _LLM([first, "garbage", "garbage", "garbage"])
    assert contextualize_artifact(db, llm, 1, "doc", "m") == 31
    assert db.stored[100] == "ctx 0" and db.stored[129] == "ctx 29"
    assert db.stored[130] == ""  # the second batch's one chunk


@pytest.mark.parametrize("exc", [UsageLimitError("quota"), LLMUnavailableError("down")])
def test_transport_failure_propagates_and_stores_nothing(exc):
    db = _Db({1: 2})
    llm = _LLM([exc])
    with pytest.raises(type(exc)):
        contextualize_artifact(db, llm, 1, "doc", "m")
    assert len(llm.prompts) == 1 and db.stored == {}


def test_a_cycle_stops_at_an_outage_and_leaves_the_pages_for_the_next(caplog):
    db = _Db({1: 2, 2: 2, 3: 2})
    llm = _LLM([_reply((0, "a"), (1, "b")), LLMUnavailableError("down")])
    with caplog.at_level(logging.WARNING, logger="ingestion.web_context"):
        assert contextualize_pending(db, llm, "m") == 2
    assert set(db.stored) == {100, 101}
    assert [p["id"] for p in db.artifacts_to_contextualize(10)] == [2, 3]
    assert "LLM unavailable" in caplog.text


def test_a_cycle_takes_at_most_its_cap_of_pages_and_its_time():
    db = _Db({p: 1 for p in range(1, 6)})
    assert contextualize_pending(db, _LLM([]), "m", max_artifacts=2) == 2
    assert [p["id"] for p in db.artifacts_to_contextualize(10)] == [3, 4, 5]
    assert contextualize_pending(db, _LLM([]), "m", budget_s=0) == 0  # out of time before page 3


def test_parse_contexts_pads_one_dropped_entry_and_refuses_a_mostly_empty_reply():
    ok, reason = parse_contexts(_reply((0, "a"), (1, "b"), (2, "c")), 4)
    assert ok == ["a", "b", "c", ""] and reason is None
    bad, reason = parse_contexts(_reply((0, "a")), 4)
    assert bad is None and reason and "1/4" in reason
