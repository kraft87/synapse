"""Tests for ingestion.web_enqueue: gates, idempotent enqueue, near-dup collapse."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ingestion.web_enqueue import (
    GATE_MAX_LINK_DENSITY,
    GATE_MIN_CHARS,
    WEB_EXTRACTION_PRIORITY,
    enqueue_web_chunks,
    gate_failures,
    link_density,
)

# ---------- pure gates ----------


class TestLinkDensity:
    def test_empty(self):
        assert link_density("") == 0.0

    def test_prose_is_low(self):
        assert link_density("Trafilatura achieves F1 0.93 on boilerplate removal.") == 0.0

    def test_nav_block_is_high(self):
        nav = "[Home](/) [Docs](/docs) [Blog](/blog) [Pricing](/pricing) [About](/about)"
        assert link_density(nav) > GATE_MAX_LINK_DENSITY

    def test_inline_citation_passes(self):
        text = (
            "Anthropic's contextual retrieval post ([source](https://anthropic.com/x)) "
            "reports a 49 percent reduction in retrieval failures when chunk context "
            "prefixes are combined with BM25 indexing, and 67 percent with reranking. "
            "The approach costs roughly thirty times more at index time, which is why "
            "it only makes sense on bounded corpora with prompt caching enabled."
        )
        assert link_density(text) < GATE_MAX_LINK_DENSITY

    def test_image_links_count(self):
        img = "![logo](https://cdn.example.com/a-very-long-asset-path/logo.png)"
        assert link_density(img) == 1.0


class TestGateFailures:
    def test_clean_content(self):
        assert gate_failures("x" * GATE_MIN_CHARS) == []

    def test_too_small(self):
        assert "too_small" in gate_failures("short")

    def test_link_density(self):
        nav = "[a](/x) " * 60
        assert "link_density" in gate_failures(nav)


# ---------- DB enqueue (shared test Postgres) ----------


PROSE = (
    "Trafilatura is the consensus extraction library for HTML to text conversion, "
    "achieving F1 scores between 0.88 and 0.95 across benchmark corpora. Mozilla "
    "Readability ties it on accuracy but only supports English layouts well. "
    "Production pipelines extract to Markdown because headings provide free chunk "
    "boundaries and the conversion drops roughly two thirds of the raw token count."
)


@pytest.fixture()
def web_tables(conn):
    present = conn.execute("SELECT to_regclass('public.web_chunks')").fetchone()
    if not (present[0] if not isinstance(present, dict) else present["to_regclass"]):
        pytest.skip("web tables not present in test DB (migrations 011-013 unapplied)")
    col = conn.execute(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name = 'extraction_queue' AND column_name = 'web_chunk_id'"
    ).fetchone()
    if not col:
        pytest.skip("extraction_queue.web_chunk_id missing (migration 018 unapplied)")
    conn.execute("TRUNCATE web_artifacts, web_chunks, extraction_queue RESTART IDENTITY CASCADE")
    yield


def _mk_artifact(conn, *, tool_use_id: str, content_hash: str | None = None, kind="web_scrape"):
    row = conn.execute(
        """
        INSERT INTO web_artifacts (kind, tool_name, tool_use_id, url, content_hash,
                                   content_markdown, synthesized, session_id, fetched_at)
        VALUES (%s, 'test', %s, 'https://e.com/x', %s, %s, false, 'sess-1', %s)
        RETURNING id
        """,
        (kind, tool_use_id, content_hash, PROSE, datetime(2026, 6, 1, tzinfo=UTC)),
    ).fetchone()
    return row["id"] if isinstance(row, dict) else row[0]


def _mk_chunk(conn, artifact_id: int, idx: int, content: str, *, prefix="ctx: web ingestion"):
    row = conn.execute(
        """
        INSERT INTO web_chunks (web_artifact_id, idx, content, char_start, char_end,
                                is_embedded, context_prefix)
        VALUES (%s, %s, %s, 0, %s, true, %s)
        RETURNING id
        """,
        (artifact_id, idx, content, len(content), prefix),
    ).fetchone()
    return row["id"] if isinstance(row, dict) else row[0]


@pytest.mark.db
class TestEnqueue:
    def test_enqueues_clean_chunk_with_prefix(self, conn, web_tables):
        aid = _mk_artifact(conn, tool_use_id="t1")
        cid = _mk_chunk(conn, aid, 0, PROSE)
        stats = enqueue_web_chunks(conn)
        assert stats.enqueued == 1
        q = conn.execute(
            "SELECT content, content_type, priority, web_chunk_id, session_id FROM extraction_queue"
        ).fetchone()
        content, ctype, prio, web_chunk_id, session_id = (
            (q["content"], q["content_type"], q["priority"], q["web_chunk_id"], q["session_id"])
            if isinstance(q, dict)
            else q
        )
        assert ctype == "web_chunk"
        assert prio == WEB_EXTRACTION_PRIORITY
        assert web_chunk_id == cid
        assert session_id == "sess-1"
        assert content.startswith("ctx: web ingestion\n\n")

    def test_idempotent_rerun(self, conn, web_tables):
        aid = _mk_artifact(conn, tool_use_id="t1")
        _mk_chunk(conn, aid, 0, PROSE)
        assert enqueue_web_chunks(conn).enqueued == 1
        rerun = enqueue_web_chunks(conn)
        assert rerun.enqueued == 0
        n = conn.execute("SELECT count(*) FROM extraction_queue").fetchone()
        assert (n["count"] if isinstance(n, dict) else n[0]) == 1

    def test_gates_drop_nav_and_tiny(self, conn, web_tables):
        aid = _mk_artifact(conn, tool_use_id="t1")
        _mk_chunk(conn, aid, 0, "[Home](/) [Docs](/docs) " * 30)
        _mk_chunk(conn, aid, 1, "too short")
        stats = enqueue_web_chunks(conn)
        assert stats.enqueued == 0
        assert stats.skipped_link_density == 1
        assert stats.skipped_too_small == 1

    def test_skips_unready_chunks(self, conn, web_tables):
        aid = _mk_artifact(conn, tool_use_id="t1")
        conn.execute(
            "INSERT INTO web_chunks (web_artifact_id, idx, content, char_start, char_end, "
            "is_embedded, context_prefix) VALUES (%s, 0, %s, 0, 1, false, NULL)",
            (aid, PROSE),
        )
        assert enqueue_web_chunks(conn).candidates == 0

    def test_near_dup_collapses_to_lowest_artifact(self, conn, web_tables):
        a1 = _mk_artifact(conn, tool_use_id="t1", content_hash="h1")
        a2 = _mk_artifact(conn, tool_use_id="t2", content_hash="h1")
        _mk_chunk(conn, a1, 0, PROSE)
        _mk_chunk(conn, a2, 0, PROSE)
        stats = enqueue_web_chunks(conn)
        assert stats.enqueued == 1
        q = conn.execute("SELECT web_chunk_id FROM extraction_queue").fetchall()
        assert len(q) == 1

    def test_research_brief_kind_enqueues(self, conn, web_tables):
        aid = _mk_artifact(conn, tool_use_id="t1", kind="research_brief")
        _mk_chunk(conn, aid, 0, PROSE)
        assert enqueue_web_chunks(conn).enqueued == 1

    def test_since_excludes_older_artifacts(self, conn, web_tables):
        # fixture artifacts are fetched_at 2026-06-01
        aid = _mk_artifact(conn, tool_use_id="t1")
        _mk_chunk(conn, aid, 0, PROSE)
        cutoff = datetime(2026, 6, 5, tzinfo=UTC)
        assert enqueue_web_chunks(conn, since=cutoff).candidates == 0
        assert enqueue_web_chunks(conn).enqueued == 1
