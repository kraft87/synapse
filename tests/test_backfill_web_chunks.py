"""Tests for scripts/backfill_web_chunks.py — the web lane's chunk + embed step.

Against the shared Postgres test DB (skipped cleanly when it's down):
  * the chunk phase takes research briefs only by default; other kinds need --kinds
  * --reset deletes only the selected kinds' chunks
  * --dry-run lists what it would chunk, writes nothing, needs no embedding key,
    and runs on a read-only connection
  * the embed phase builds its embedder through ingestion.embedding.create_embedder
    (the same factory, and so the same model and width, as recall's query embedder)
"""

from __future__ import annotations

import importlib.util
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "backfill_web_chunks.py"

_spec = importlib.util.spec_from_file_location("backfill_web_chunks_script", _SCRIPT)
assert _spec is not None and _spec.loader is not None
bwc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bwc)

from tests.helpers.embed import onehot  # noqa: E402

_DB_URL = os.environ["SYNAPSE_DB_URL"]  # conftest pins this to the test database

BRIEF = (
    "# Drip coffee machines\n\n"
    + "The Fellow Aiden brews to SCA Gold Cup temperature and holds it steady. " * 30
    + "\n\n## Ratio Six\n\n"
    + "The Ratio Six uses a single showerhead and a glass carafe with no hot plate. " * 30
)


@pytest.fixture()
def web_tables(conn):
    conn.execute("TRUNCATE web_artifacts, web_chunks RESTART IDENTITY CASCADE")
    yield
    conn.execute("TRUNCATE web_artifacts, web_chunks RESTART IDENTITY CASCADE")


def _mk_artifact(conn, *, tool_use_id: str, kind: str, content: str = BRIEF) -> int:
    row = conn.execute(
        """
        INSERT INTO web_artifacts (kind, tool_name, tool_use_id, url, content_markdown,
                                   synthesized, fetched_at)
        VALUES (%s, 'test', %s, 'research://x', %s, true, %s)
        RETURNING id
        """,
        (kind, tool_use_id, content, datetime(2026, 9, 1, tzinfo=UTC)),
    ).fetchone()
    return int(row[0])


def _chunk_counts(conn) -> dict[int, int]:
    rows = conn.execute(
        "SELECT web_artifact_id, count(*) FROM web_chunks GROUP BY web_artifact_id"
    ).fetchall()
    return {int(a): int(n) for a, n in rows}


class _StubEmbedder:
    model_name = "stub"
    dimensions = 0

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str]] = []

    def embed(self, texts: list[str], task: str = "document") -> list[list[float]]:
        self.calls.append((list(texts), task))
        return [onehot(i) for i in range(len(texts))]


@pytest.mark.db
class TestChunkPhase:
    def test_default_chunks_research_briefs_only(self, conn, web_tables):
        brief = _mk_artifact(conn, tool_use_id="brief:coffee:2026-09-01", kind="research_brief")
        scrape = _mk_artifact(conn, tool_use_id="toolu_1", kind="web_scrape")

        with psycopg.connect(_DB_URL) as c:
            arts, chunks = bwc.phase_chunk(c)

        counts = _chunk_counts(conn)
        assert arts == 1
        assert counts.get(brief, 0) == chunks > 1
        assert scrape not in counts

    def test_kinds_widens_the_selection(self, conn, web_tables):
        scrape = _mk_artifact(conn, tool_use_id="toolu_1", kind="web_scrape")

        with psycopg.connect(_DB_URL) as c:
            arts, _ = bwc.phase_chunk(c, kinds=("web_scrape",))

        assert arts == 1
        assert _chunk_counts(conn).get(scrape, 0) > 0

    def test_reset_deletes_only_the_selected_kinds(self, conn, web_tables):
        brief = _mk_artifact(conn, tool_use_id="brief:coffee:2026-09-01", kind="research_brief")
        scrape = _mk_artifact(conn, tool_use_id="toolu_1", kind="web_scrape")
        with psycopg.connect(_DB_URL) as c:
            bwc.phase_chunk(c, kinds=("research_brief", "web_scrape"))
        before = _chunk_counts(conn)

        assert bwc.main(["--reset", "--chunk-only"]) == 0

        # the brief is re-chunked; the web_scrape's chunks, outside --kinds, survive
        assert _chunk_counts(conn) == before
        assert before[brief] > 0 and before[scrape] > 0


@pytest.mark.db
class TestDryRun:
    def test_lists_candidates_and_writes_nothing(self, conn, web_tables, monkeypatch, capsys):
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        brief = _mk_artifact(conn, tool_use_id="brief:coffee:2026-09-01", kind="research_brief")
        _mk_artifact(conn, tool_use_id="toolu_1", kind="web_scrape")

        def _no_embedder(**_kw: Any) -> None:
            raise AssertionError("dry run must not build an embedder")

        monkeypatch.setattr(bwc, "create_embedder", _no_embedder)
        read_only: list[bool] = []
        real_dry_run = bwc.dry_run

        def _spy(c: psycopg.Connection[Any], kinds: Any) -> tuple[int, int]:
            read_only.append(c.read_only)
            return real_dry_run(c, kinds)

        monkeypatch.setattr(bwc, "dry_run", _spy)

        assert bwc.main(["--dry-run"]) == 0

        assert read_only == [True]  # main runs it on the server-enforced read-only connection
        out = capsys.readouterr().out
        assert "brief:coffee:2026-09-01" in out
        assert "toolu_1" not in out
        assert "would chunk 1 artifacts" in out
        assert _chunk_counts(conn) == {}
        assert brief  # the brief is still there, still unchunked

    def test_dry_run_connection_rejects_writes(self, web_tables):
        with bwc.connect(_DB_URL, read_only=True) as c:
            with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
                c.execute(
                    "INSERT INTO web_artifacts (kind, tool_name, tool_use_id, url) "
                    "VALUES ('research_brief', 'test', 'x', 'research://x')"
                )


@pytest.mark.db
class TestEmbedPhase:
    def test_embeds_through_create_embedder_with_context_prefix(
        self, conn, web_tables, monkeypatch
    ):
        aid = _mk_artifact(conn, tool_use_id="brief:coffee:2026-09-01", kind="research_brief")
        conn.execute(
            """
            INSERT INTO web_chunks (web_artifact_id, idx, content, char_start, char_end,
                                    context_prefix)
            VALUES (%s, 0, 'chunk zero', 0, 10, 'ctx: coffee'), (%s, 1, 'chunk one', 10, 19, NULL)
            """,
            (aid, aid),
        )
        stub = _StubEmbedder()
        seen: dict[str, Any] = {}

        def _factory(**kw: Any) -> _StubEmbedder:
            seen.update(kw)
            return stub

        monkeypatch.setattr(bwc, "create_embedder", _factory)
        monkeypatch.setenv("VOYAGE_API_KEY", "test-key")

        assert bwc.main(["--embed-only"]) == 0

        assert seen.get("db_url") == _DB_URL  # dims/model validated against synapse_meta
        assert stub.calls == [(["ctx: coffee\n\nchunk zero", "chunk one"], "document")]
        rows = conn.execute(
            "SELECT is_embedded, embedding IS NOT NULL FROM web_chunks ORDER BY idx"
        ).fetchall()
        assert rows == [(True, True), (True, True)]
