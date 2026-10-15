"""Tests for ingestion.web_artifacts: URL canonicalization, row building, the writer."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import psycopg
import pytest
from psycopg.rows import dict_row

from ingestion.web_artifacts import (
    artifact_stored,
    build_row,
    canonicalize_url,
    count_fetched_between,
    insert_artifact,
)
from ingestion.web_extractors import ResearchJobRef, SearchResultItem, SearchResultSet, WebScrape

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


# ---------- build_row ----------

FETCHED = datetime(2026, 5, 14, tzinfo=UTC)


def _row(parsed, **kw):
    args = dict(
        tool_name="WebFetch",
        tool_use_id="toolu_test",
        session_id="sess-1",
        fetched_at=FETCHED,
        parent_episode_id=None,
        metadata={"capture": "generic"},
    )
    return build_row(parsed, **{**args, **kw})


class TestBuildRow:
    def test_scrape_row(self):
        parsed = WebScrape(
            tool_name="WebFetch",
            url="https://e.com/page",
            title="A title",
            content_markdown="hello",
            synthesized=True,
            prompt="summarize",
            raw_chars=12,
        )
        row = _row(parsed, parent_episode_id=7)
        assert row["kind"] == "web_scrape"
        assert row["url"] == "https://e.com/page"
        assert row["url_canonical"] == "https://e.com/page"
        assert row["title"] == "A title"
        assert row["synthesized"] is True
        assert row["prompt"] == "summarize"
        assert row["parent_episode_id"] == 7
        assert row["session_id"] == "sess-1"
        # SHA-256 of "hello"
        assert row["content_hash"] == (
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        )
        assert row["fetched_at"] == FETCHED
        assert json.loads(row["metadata"]) == {"capture": "generic"}

    def test_search_row_serializes_items_as_json(self):
        parsed = SearchResultSet(
            tool_name="WebSearch",
            query="cats",
            items=[
                SearchResultItem(url="https://a", title="A", position=1),
                SearchResultItem(url="https://b", title="B", position=2),
            ],
            raw_chars=99,
        )
        row = _row(parsed)
        assert row["kind"] == "search_result_set"
        assert row["query"] == "cats"
        assert row["item_count"] == 2
        items = json.loads(row["items"])
        assert [it["url"] for it in items] == ["https://a", "https://b"]
        assert items[0]["position"] == 1

    def test_the_same_results_for_the_same_query_hash_alike(self):
        def hashed(query, urls):
            items = [SearchResultItem(url=u, title="t") for u in urls]
            return _row(SearchResultSet(tool_name="x", query=query, items=items))["content_hash"]

        assert hashed("q", ["https://a/", "https://b"]) == hashed("q", ["https://b", "https://a"])
        assert hashed("q", ["https://a", "https://b"]) != hashed("z", ["https://a", "https://b"])

    def test_research_job_row(self):
        row = _row(ResearchJobRef(tool_name="x", research_id="r1", instructions="go", model="m"))
        assert (row["kind"], row["research_id"], row.get("content_hash")) == (
            "research_job_ref",
            "r1",
            None,
        )

    def test_credentials_are_redacted_from_every_text_field(self):
        key = "sk-ant-api03-" + "A1b2C3d4E5f6" * 4
        parsed = WebScrape(
            tool_name="WebFetch",
            url=f"https://e.com/p?api_key={key}",
            title=f"title {key}",
            content_markdown=f"export ANTHROPIC_API_KEY={key}\nand a long enough body",
        )
        row = _row(parsed, metadata={"project": f"p {key}"})
        for field in ("url", "url_canonical", "title", "content_markdown", "metadata"):
            assert key not in row[field], field

    def test_nul_bytes_are_stripped(self):
        row = _row(WebScrape(tool_name="x", url="https://e.com", content_markdown="a\x00b"))
        assert row["content_markdown"] == "ab"
        items = [SearchResultItem(url="https://a", title="t", snippet="s\x00nip")]
        row = _row(SearchResultSet(tool_name="x", query="q\x00r", items=items))
        assert row["query"] == "qr" and "\\u0000" not in row["items"]


# ---------- Writer (requires SYNAPSE_TEST_URL) ----------


@pytest.fixture
def wconn(db_url):
    with psycopg.connect(db_url, autocommit=True, row_factory=dict_row) as c:
        c.execute("TRUNCATE web_artifacts RESTART IDENTITY CASCADE")
        yield c
        c.execute("TRUNCATE web_artifacts RESTART IDENTITY CASCADE")


def _scrape_row(tool_use_id="toolu_1", url="https://e.com/a", body="hello", **kw):
    parsed = WebScrape(tool_name="WebFetch", url=url, content_markdown=body)
    return _row(parsed, tool_use_id=tool_use_id, **kw)


class TestWriter:
    def test_insert_is_idempotent_by_tool_use_id(self, wconn):
        row = _scrape_row()
        assert insert_artifact(wconn, row) is True
        assert insert_artifact(wconn, row) is False
        assert wconn.execute("SELECT count(*) AS n FROM web_artifacts").fetchone()["n"] == 1

    def test_stored_by_id_or_by_url_and_content(self, wconn):
        row = _scrape_row()
        insert_artifact(wconn, row)
        key = (row["url_canonical"], row["content_hash"])
        assert artifact_stored(wconn, "toolu_1", None, None)  # same call
        assert artifact_stored(wconn, "toolu_2", *key)  # same page, another call
        assert not artifact_stored(wconn, "toolu_2", key[0], "other-hash")  # page changed
        assert not artifact_stored(wconn, "toolu_2", "https://e.com/b", key[1])  # other URL
        assert not artifact_stored(wconn, "toolu_2", None, None)

    def test_a_failed_insert_in_a_savepoint_keeps_the_others(self, wconn):
        with wconn.transaction():
            insert_artifact(wconn, _scrape_row("toolu_first", body="one"))
            with pytest.raises(psycopg.Error), wconn.transaction():
                insert_artifact(wconn, {**_scrape_row("toolu_bad"), "kind": "nonsense"})
            insert_artifact(wconn, _scrape_row("toolu_last", url="https://e.com/c", body="two"))
        rows = wconn.execute("SELECT tool_use_id FROM web_artifacts ORDER BY id").fetchall()
        assert [r["tool_use_id"] for r in rows] == ["toolu_first", "toolu_last"]

    def test_count_fetched_between(self, wconn):
        late = datetime(2026, 5, 14, 23, 59, tzinfo=UTC)
        insert_artifact(wconn, _scrape_row("a", fetched_at=late))
        insert_artifact(
            wconn,
            _scrape_row("b", url="https://e.com/b", fetched_at=datetime(2026, 5, 15, tzinfo=UTC)),
        )
        day = datetime(2026, 5, 14, tzinfo=UTC)
        assert count_fetched_between(wconn, day, datetime(2026, 5, 15, tzinfo=UTC)) == 1
        assert count_fetched_between(wconn, day, datetime(2026, 5, 16, tzinfo=UTC)) == 2
