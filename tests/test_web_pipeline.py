"""Captured pages reach recall without an outside cron: the poller's maintenance stages
chunk them, give the chunks their context, embed them, and queue the trusted ones for
extraction. Against the scratch DB, with a scripted LLM and embedder."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from ingestion.db import Database
from ingestion.llm_client import LLMUnavailableError
from ingestion.poller import Maintenance

FETCHED = datetime(2026, 9, 1, 12, tzinfo=UTC)
SECTION = "Contextual retrieval prepends a short situating sentence to each chunk. " * 14
PAGE = "\n\n".join(f"## Part {i}\n\n{SECTION}" for i in range(4))


class _Embedder:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed(self, texts: list[str], task: str) -> list[list[float]]:
        self.texts += texts
        return [[0.25] * 2048 for _ in texts]


class _LLM:
    """Writes 'about chunk i' for every chunk the prompt asks about; or raises ``fail``."""

    def __init__(self, fail: Exception | None = None) -> None:
        self.fail = fail
        self.calls = 0
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, *, messages, **kwargs):
        self.calls += 1
        if self.fail:
            raise self.fail
        n = int(re.search(r"Below are (\d+) chunks", messages[0]["content"]).group(1))
        reply = {"contexts": [{"idx": i, "context": f"about part {i}"} for i in range(n)]}
        return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(reply))])


@pytest.fixture
def maintenance(db_url, conn, clean_tables):
    conn.execute("TRUNCATE web_artifacts, extraction_queue RESTART IDENTITY CASCADE")
    db = Database(db_url)
    embedder = _Embedder()
    m = Maintenance(db=db, pipeline=SimpleNamespace(_embedder=embedder))  # type: ignore[arg-type]
    m._llm = _LLM()
    m.embedder = embedder  # type: ignore[attr-defined]
    yield m
    db.close()
    conn.execute("TRUNCATE web_artifacts, extraction_queue RESTART IDENTITY CASCADE")


def page(conn, tid: str, *, kind="web_scrape", body: str | None = PAGE, meta: str | None = None):
    return conn.execute(
        "INSERT INTO web_artifacts (kind, tool_name, tool_use_id, url, content_markdown, "
        "session_id, fetched_at, metadata) VALUES (%s, 'fetch', %s, 'https://e.com/x', %s, 's1', "
        "%s, %s) RETURNING id",
        (kind, tid, body, FETCHED, meta),
    ).fetchone()[0]


def chunks(conn, artifact_id: int) -> list[tuple]:
    return conn.execute(
        "SELECT idx, context_prefix, is_embedded, content_ts, embedding IS NOT NULL "
        "FROM web_chunks WHERE web_artifact_id = %s ORDER BY idx",
        (artifact_id,),
    ).fetchall()


def test_one_cycle_takes_a_captured_page_from_text_to_vectors(maintenance, conn):
    aid = page(conn, "t1")
    maintenance.cycle(daily=False)
    rows = chunks(conn, aid)
    assert len(rows) >= 3
    for idx, prefix, embedded, content_ts, has_vec in rows:
        assert (prefix, embedded, content_ts, has_vec) == (f"about part {idx}", True, FETCHED, True)
    # a vector is of the prefix and the chunk together
    texts = maintenance.embedder.texts  # newest chunk first
    assert any(t.startswith("about part 0\n\n## Part 0") for t in texts)
    assert len(texts) == len(rows)
    # the chunks cover the page: the offsets slice it back
    for content, start, end in conn.execute(
        "SELECT content, char_start, char_end FROM web_chunks ORDER BY idx"
    ).fetchall():
        assert PAGE[start:end] == content


def test_chunks_wait_for_their_context_before_they_are_embedded(maintenance, conn):
    aid = page(conn, "t1")
    built = maintenance.chunk_web_artifacts()
    assert built >= 3 and maintenance.chunk_web_artifacts() == 0  # once
    assert maintenance.embed_pending() == 0 and maintenance.embedder.texts == []
    assert {r[1] for r in chunks(conn, aid)} == {None}
    assert maintenance.contextualize_web_chunks() == built
    assert maintenance.embed_pending() == built


def test_an_llm_outage_leaves_the_chunks_waiting_not_embedded_bare(maintenance, conn):
    aid = page(conn, "t1")
    maintenance.chunk_web_artifacts()
    maintenance._llm = _LLM(fail=LLMUnavailableError("down"))
    assert maintenance.contextualize_web_chunks() == 0
    assert maintenance.embed_pending() == 0
    assert {(r[1], r[2]) for r in chunks(conn, aid)} == {(None, False)}
    maintenance._llm = _LLM()  # back up: the next cycle finishes the job
    maintenance.cycle(daily=False)
    assert {r[2] for r in chunks(conn, aid)} == {True}


def test_a_short_page_is_one_chunk_with_no_context_and_no_llm_call(maintenance, conn):
    aid = page(conn, "t1", body="A short but complete note about chunking.")
    maintenance.cycle(daily=False)
    assert [(r[0], r[1], r[2]) for r in chunks(conn, aid)] == [(0, "", True)]
    assert maintenance._llm.calls == 0
    assert maintenance.embedder.texts == ["A short but complete note about chunking."]


def test_only_pages_with_a_body_are_chunked(maintenance, conn):
    result_list = page(conn, "t1", kind="search_result_set", body=None)
    empty = page(conn, "t2", body="")
    maintenance.cycle(daily=False)
    assert chunks(conn, result_list) == [] and chunks(conn, empty) == []


def test_a_chunk_embedded_before_its_context_is_embedded_again_with_it(maintenance, conn):
    # The corpus from before this stage: chunks embedded bare, no prefix yet.
    aid = page(conn, "t1", body="A short but complete note about chunking.")
    conn.execute(
        "INSERT INTO web_chunks (web_artifact_id, idx, content, char_start, char_end, "
        "is_embedded, embedding) VALUES (%s, 0, 'A short but complete note about chunking.', "
        "0, 41, true, %s::vector)",
        (aid, "[" + ",".join(["0.1"] * 2048) + "]"),
    )
    maintenance.cycle(daily=False)
    assert [(r[1], r[2]) for r in chunks(conn, aid)] == [("", True)]
    assert len(maintenance.embedder.texts) == 1  # embedded again, once, now that it has a prefix


def test_only_pages_marked_for_the_graph_are_queued_for_extraction(maintenance, conn):
    generic = page(conn, "t1", meta='{"capture": "generic", "kg_extract": false}')
    trusted = page(conn, "t2", meta='{"capture": "docs_site", "kg_extract": true}')
    unmarked = page(conn, "t3")
    maintenance.cycle(daily=False)
    queued = [
        r[0]
        for r in conn.execute(
            "SELECT c.web_artifact_id FROM extraction_queue q "
            "JOIN web_chunks c ON c.id = q.web_chunk_id ORDER BY q.id"
        ).fetchall()
    ]
    assert queued and set(queued) == {trusted} and generic not in queued and unmarked not in queued
    assert maintenance.enqueue_web_extraction() == 0  # once
