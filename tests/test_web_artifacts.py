"""Tests for ingestion.web_artifacts: URL canonicalization, JSONL walking, writer idempotency."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import psycopg
import pytest

from ingestion.web_artifacts import (
    WebArtifactsIngester,
    _build_row,
    canonicalize_url,
    iter_web_events,
)
from ingestion.web_extractors import ExtractError, SearchResultSet, WebScrape

# ---------- URL canonicalization ----------


class TestCanonicalize:
    def test_lowercases_host(self):
        assert canonicalize_url("HTTPS://Example.COM/Path") == "https://example.com/Path"

    def test_drops_fragment(self):
        assert canonicalize_url("https://e.com/x#frag") == "https://e.com/x"

    def test_preserves_query(self):
        assert canonicalize_url("https://e.com/x?a=1&b=2") == "https://e.com/x?a=1&b=2"

    def test_strips_trailing_slash(self):
        assert canonicalize_url("https://e.com/x/") == "https://e.com/x"

    def test_preserves_root_slash(self):
        assert canonicalize_url("https://e.com/") == "https://e.com/"

    def test_rejects_no_scheme(self):
        assert canonicalize_url("example.com/x") is None

    def test_rejects_empty(self):
        assert canonicalize_url("") is None
        assert canonicalize_url(None) is None


# ---------- _build_row ----------


def _make_event(**kw):
    from ingestion.web_artifacts import WebToolEvent

    defaults = dict(
        tool_use_id="toolu_test",
        tool_name="WebFetch",
        tool_input={"url": "https://e.com/x"},
        result_text="content",
        timestamp=datetime(2026, 5, 14, tzinfo=UTC),
        session_id="sess-1",
        jsonl_path="/tmp/test.jsonl",
    )
    defaults.update(kw)
    return WebToolEvent(**defaults)


class TestBuildRow:
    def test_scrape_row(self):
        event = _make_event()
        parsed = WebScrape(
            tool_name="WebFetch",
            url="https://e.com/page",
            title="A title",
            content_markdown="hello",
            synthesized=True,
            prompt="summarize",
            raw_chars=12,
        )
        row = _build_row(event, parsed)
        assert row is not None
        assert row["kind"] == "web_scrape"
        assert row["url"] == "https://e.com/page"
        assert row["url_canonical"] == "https://e.com/page"
        assert row["title"] == "A title"
        assert row["synthesized"] is True
        assert row["prompt"] == "summarize"
        # SHA-256 of "hello"
        assert row["content_hash"] == (
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        )
        assert row["fetched_at"] == datetime(2026, 5, 14, tzinfo=UTC)

    def test_search_row_serializes_items_as_json(self):
        from ingestion.web_extractors import SearchResultItem

        event = _make_event(tool_name="WebSearch")
        parsed = SearchResultSet(
            tool_name="WebSearch",
            query="cats",
            items=[
                SearchResultItem(url="https://a", title="A", position=1),
                SearchResultItem(url="https://b", title="B", position=2),
            ],
            raw_chars=99,
        )
        row = _build_row(event, parsed)
        assert row is not None
        assert row["kind"] == "search_result_set"
        assert row["query"] == "cats"
        assert row["item_count"] == 2
        # items is a JSON-serialized array
        items = json.loads(row["items"])
        assert len(items) == 2
        assert items[0]["url"] == "https://a"
        assert items[0]["position"] == 1

    def test_extract_error_returns_none(self):
        event = _make_event()
        err = ExtractError(tool_name="WebFetch", reason="redirect_detected", raw_chars=0)
        assert _build_row(event, err) is None

    def test_fetched_at_falls_back_to_now(self):
        event = _make_event(timestamp=None)
        parsed = WebScrape(
            tool_name="WebFetch", url="https://e.com", content_markdown="x", synthesized=True
        )
        row = _build_row(event, parsed)
        assert row is not None
        assert isinstance(row["fetched_at"], datetime)


# ---------- iter_web_events ----------


def _jsonl_record(role: str, blocks: list[dict], session_id: str, ts: str) -> str:
    return json.dumps(
        {
            "sessionId": session_id,
            "timestamp": ts,
            "message": {"role": role, "content": blocks},
        }
    )


class TestIterWebEvents:
    def test_walks_tool_use_and_result_pair(self, tmp_path: Path):
        jsonl = tmp_path / "session.jsonl"
        jsonl.write_text(
            _jsonl_record(
                "assistant",
                [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "WebFetch",
                        "input": {"url": "https://e.com/x", "prompt": "p"},
                    }
                ],
                session_id="S",
                ts="2026-05-14T01:00:00Z",
            )
            + "\n"
            + _jsonl_record(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "body of result",
                    }
                ],
                session_id="S",
                ts="2026-05-14T01:00:01Z",
            )
            + "\n"
        )
        events = list(iter_web_events(jsonl))
        assert len(events) == 1
        ev = events[0]
        assert ev.tool_use_id == "toolu_1"
        assert ev.tool_name == "WebFetch"
        assert ev.tool_input == {"url": "https://e.com/x", "prompt": "p"}
        assert ev.result_text == "body of result"
        assert ev.session_id == "S"

    def test_skips_non_web_tools(self, tmp_path: Path):
        jsonl = tmp_path / "session.jsonl"
        jsonl.write_text(
            _jsonl_record(
                "assistant",
                [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Bash",
                        "input": {"command": "ls"},
                    }
                ],
                session_id="S",
                ts="2026-05-14T01:00:00Z",
            )
            + "\n"
            + _jsonl_record(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "files",
                    }
                ],
                session_id="S",
                ts="2026-05-14T01:00:01Z",
            )
            + "\n"
        )
        assert list(iter_web_events(jsonl)) == []

    def test_tool_result_without_matching_use_is_skipped(self, tmp_path: Path):
        jsonl = tmp_path / "session.jsonl"
        jsonl.write_text(
            _jsonl_record(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "orphan",
                        "content": "body",
                    }
                ],
                session_id="S",
                ts="2026-05-14T01:00:01Z",
            )
            + "\n"
        )
        assert list(iter_web_events(jsonl)) == []


# ---------- Writer idempotency (requires SYNAPSE_TEST_URL) ----------


@pytest.fixture
def db_test_url():
    import os

    return os.environ.get(
        "SYNAPSE_TEST_URL",
        "postgresql://synapse:synapse@127.0.0.1:5432/synapse_test",
    )


@pytest.fixture
def test_conn(db_test_url):
    try:
        conn = psycopg.connect(db_test_url, autocommit=False)
    except psycopg.OperationalError:
        pytest.skip("test DB unreachable")
    yield conn
    conn.rollback()
    conn.close()


@pytest.fixture
def clean_web_artifacts(test_conn):
    # Skip cleanly if the test DB hasn't had migration 011 applied yet
    # (e.g. CI workflow only runs 001_initial.sql via `make migrate`).
    exists = test_conn.execute("SELECT to_regclass('public.web_artifacts')").fetchone()
    test_conn.commit()
    if exists is None or exists[0] is None:
        pytest.skip("web_artifacts table not present in test DB (migration 011 unapplied)")
    test_conn.execute("TRUNCATE web_artifacts, ingestion_state RESTART IDENTITY CASCADE")
    test_conn.commit()
    yield
    test_conn.execute("TRUNCATE web_artifacts, ingestion_state RESTART IDENTITY CASCADE")
    test_conn.commit()


class TestWriterIdempotency:
    def test_writer_inserts_and_dedupes_by_tool_use_id(
        self, tmp_path: Path, test_conn, clean_web_artifacts
    ):
        jsonl = tmp_path / "s.jsonl"
        jsonl.write_text(
            _jsonl_record(
                "assistant",
                [
                    {
                        "type": "tool_use",
                        "id": "toolu_idempotent",
                        "name": "WebSearch",
                        "input": {"query": "test query"},
                    }
                ],
                "S",
                "2026-05-14T02:00:00Z",
            )
            + "\n"
            + _jsonl_record(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_idempotent",
                        "content": (
                            'Web search results for query: "test query"\n\n'
                            'Links: [{"title":"A","url":"https://a.example"}]'
                        ),
                    }
                ],
                "S",
                "2026-05-14T02:00:01Z",
            )
            + "\n"
        )
        ing = WebArtifactsIngester(test_conn)
        stats1 = ing.ingest_one(jsonl)
        assert stats1.inserted == 1
        assert stats1.by_kind == {"search_result_set": 1}

        # Bump mtime so the checkpoint doesn't short-circuit the second run.
        import os

        os.utime(jsonl, None)

        stats2 = ing.ingest_one(jsonl)
        assert stats2.inserted == 0
        assert stats2.skipped_duplicate == 1

        # Confirm exactly one row landed.
        row = test_conn.execute(
            "SELECT kind, query, item_count FROM web_artifacts WHERE tool_use_id = %s",
            ("toolu_idempotent",),
        ).fetchone()
        assert row is not None
        # tuple-row default
        kind, query, item_count = row
        assert kind == "search_result_set"
        assert query == "test query"
        assert item_count == 1

    def test_writer_skips_unchanged_files_via_checkpoint(
        self, tmp_path: Path, test_conn, clean_web_artifacts
    ):
        jsonl = tmp_path / "s2.jsonl"
        jsonl.write_text(
            _jsonl_record(
                "assistant",
                [
                    {
                        "type": "tool_use",
                        "id": "toolu_checkpt",
                        "name": "WebSearch",
                        "input": {"query": "q"},
                    }
                ],
                "S",
                "2026-05-14T02:00:00Z",
            )
            + "\n"
            + _jsonl_record(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_checkpt",
                        "content": 'Web search results for query: "q"\n\nLinks: [{"title":"A","url":"https://a"}]',
                    }
                ],
                "S",
                "2026-05-14T02:00:01Z",
            )
            + "\n"
        )
        ing = WebArtifactsIngester(test_conn)
        s1 = ing.ingest_one(jsonl)
        assert s1.files_scanned == 1

        # Second run without modifying the file should skip via checkpoint.
        s2 = ing.ingest_one(jsonl)
        assert s2.files_scanned == 0
        assert s2.files_skipped_unchanged == 1
