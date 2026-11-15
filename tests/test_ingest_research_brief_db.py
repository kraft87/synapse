"""scripts/ingest_research_brief.py against the shared Postgres test DB.

A brief is keyed by 'brief:<slug>:<mtime day>'. The invariants:
  * a first run inserts the row; the web lane then chunks it
  * a re-run with the same content is a no-op and leaves the derived rows alone
  * a re-run with changed content (a corrected brief) updates the row and deletes its
    web_chunks in the same transaction, so the lane chunks, contextualizes and embeds the
    new text and no chunk of the old text stays served; the extraction_queue rows of the
    old chunks go with them (ON DELETE CASCADE)
  * a changed brief loses an audience label decided on the old text, unless --audience
    is given, which wins on every path as before
"""

from __future__ import annotations

import importlib.util
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.helpers.embed import onehot

_REPO = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


brief_script = _load("ingest_research_brief_script", "ingest_research_brief.py")
bwc = _load("backfill_web_chunks_for_brief_tests", "backfill_web_chunks.py")

_DB_URL = os.environ["SYNAPSE_DB_URL"]  # conftest pins this to the test database

# A fixed mtime keeps the key's date stable however long the test takes.
_MTIME = datetime(2026, 11, 15, 12, 0, tzinfo=UTC).timestamp()
_KEY = "brief:ai-search-apis:2026-11-15"

DRAFT = (
    "# AI search APIs\n\n"
    + "Verdict: Exa is the default pick for semantic search over the open web. " * 30
    + "\n\n## Tavily\n\n"
    + "Verdict: Tavily is not worth adding next to Exa for this workload. " * 30
)
CORRECTED = (
    "# AI search APIs\n\n"
    + "Verdict: Firecrawl search is the default pick when the page body is needed. " * 30
    + "\n\n## Tavily\n\n"
    + "Verdict: Tavily is worth a trial for its extract endpoint latency. " * 30
)


@pytest.fixture()
def web_tables(conn):
    conn.execute("TRUNCATE web_artifacts, web_chunks, extraction_queue RESTART IDENTITY CASCADE")
    yield conn
    conn.execute("TRUNCATE web_artifacts, web_chunks, extraction_queue RESTART IDENTITY CASCADE")


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    os.utime(path, (_MTIME, _MTIME))
    return path


def _run(monkeypatch, capsys, path: Path, *extra: str) -> str:
    monkeypatch.setenv("SYNAPSE_DB_URL", _DB_URL)
    monkeypatch.setattr(
        sys, "argv", ["ingest", "--file", str(path), "--slug", "ai-search-apis", *extra]
    )
    assert brief_script.main() == 0
    return capsys.readouterr().out


def _artifact(conn):
    rows = conn.execute(
        "SELECT id, content_markdown, content_hash, raw_chars, title, audience "
        "FROM web_artifacts WHERE tool_use_id = %s",
        (_KEY,),
    ).fetchall()
    assert len(rows) == 1
    return rows[0]


def _process(conn) -> list[int]:
    """What the lane cron does to the brief: chunk, contextualize, embed, plus an
    extraction_queue row per chunk (the retired web->KG lane's shape). Returns chunk ids."""
    bwc.phase_chunk(conn)
    ids = [int(r[0]) for r in conn.execute("SELECT id FROM web_chunks ORDER BY id").fetchall()]
    for i, cid in enumerate(ids):
        conn.execute(
            "UPDATE web_chunks SET context_prefix = 'ctx', embedding = %s::vector, "
            "is_embedded = true WHERE id = %s",
            (str(onehot(i)), cid),
        )
        conn.execute(
            "INSERT INTO extraction_queue (web_chunk_id, content, content_type, priority) "
            "VALUES (%s, 'x', 'web_chunk', 5)",
            (cid,),
        )
    return ids


def _chunks(conn):
    return conn.execute(
        "SELECT id, content, context_prefix, is_embedded FROM web_chunks ORDER BY id"
    ).fetchall()


def _queued(conn) -> int:
    row = conn.execute("SELECT count(*) FROM extraction_queue WHERE web_chunk_id IS NOT NULL")
    return int(row.fetchone()[0])


def test_first_ingest_inserts_the_row(web_tables, tmp_path, monkeypatch, capsys):
    out = _run(monkeypatch, capsys, _write(tmp_path / "b.md", DRAFT))
    assert out.startswith("ingested research_brief id=1 (brief:ai-search-apis:2026-11-15,")
    _id, content, content_hash, raw_chars, title, audience = _artifact(web_tables)
    assert content == DRAFT.strip()
    assert raw_chars == len(DRAFT.strip())
    assert title == "AI search APIs"
    assert audience is None
    assert content_hash


def test_identical_rerun_is_a_noop_and_keeps_derived_rows(
    web_tables, tmp_path, monkeypatch, capsys
):
    doc = _write(tmp_path / "b.md", DRAFT)
    _run(monkeypatch, capsys, doc)
    ids = _process(web_tables)
    assert ids
    before = (_artifact(web_tables), _chunks(web_tables), _queued(web_tables))

    out = _run(monkeypatch, capsys, doc)

    assert out.strip() == f"already ingested ({_KEY}); no-op"
    assert (_artifact(web_tables), _chunks(web_tables), _queued(web_tables)) == before


def test_changed_rerun_updates_the_row_and_drops_stale_derived_rows(
    web_tables, tmp_path, monkeypatch, capsys
):
    doc = _write(tmp_path / "b.md", DRAFT)
    _run(monkeypatch, capsys, doc)
    old_ids = _process(web_tables)
    assert _queued(web_tables) == len(old_ids)
    old_hash = _artifact(web_tables)[2]

    out = _run(monkeypatch, capsys, _write(doc, CORRECTED))

    assert out.startswith(f"updated research_brief id=1 ({_KEY}, {len(CORRECTED.strip())} chars;")
    assert f"{len(old_ids)} stale web_chunks deleted" in out
    art_id, content, content_hash, raw_chars, _title, _aud = _artifact(web_tables)
    assert art_id == 1
    assert content == CORRECTED.strip()
    assert raw_chars == len(CORRECTED.strip())
    assert content_hash != old_hash
    # Nothing derived from the old text is left to serve or extract.
    assert _chunks(web_tables) == []
    assert _queued(web_tables) == 0

    # The lane's own "needs chunking" test now selects the brief again...
    assert [r[0] for r in bwc._pending(web_tables, bwc.DEFAULT_KINDS)] == [1]
    bwc.phase_chunk(web_tables)
    new = _chunks(web_tables)
    # ...and it re-chunks the new text only, into fresh ids awaiting context + embedding.
    assert new
    assert not {r[0] for r in new} & set(old_ids)
    assert all(r[2] is None and r[3] is False for r in new)
    text = " ".join(r[1] for r in new)
    assert "Firecrawl search is the default pick" in text
    assert "Exa is the default pick" not in text
    assert "not worth adding" not in text

    # A third run with the corrected text is a no-op again.
    assert _run(monkeypatch, capsys, doc).strip() == f"already ingested ({_KEY}); no-op"
    assert _chunks(web_tables) == new


def test_changed_rerun_clears_a_label_decided_on_the_old_text(
    web_tables, tmp_path, monkeypatch, capsys
):
    doc = _write(tmp_path / "b.md", DRAFT)
    _run(monkeypatch, capsys, doc)
    web_tables.execute("UPDATE web_artifacts SET audience = 'work-safe'")  # classifier verdict

    out = _run(monkeypatch, capsys, _write(doc, CORRECTED))

    assert "audience label cleared" in out
    assert _artifact(web_tables)[5] is None


def test_audience_flag_wins_on_insert_noop_and_update(web_tables, tmp_path, monkeypatch, capsys):
    doc = _write(tmp_path / "b.md", DRAFT)

    out = _run(monkeypatch, capsys, doc, "--audience", "work-safe")
    assert out.startswith("ingested research_brief")
    assert "audience=work-safe" in out
    assert _artifact(web_tables)[5] == "work-safe"
    ids = _process(web_tables)

    # Same content: still a no-op for the content, the flag relabels as before.
    out = _run(monkeypatch, capsys, doc, "--audience", "personal")
    assert f"already ingested ({_KEY}); no-op" in out
    assert "audience=personal" in out
    assert _artifact(web_tables)[5] == "personal"
    assert [r[0] for r in _chunks(web_tables)] == ids

    # Changed content with the flag: the explicit label is kept, not cleared.
    out = _run(monkeypatch, capsys, _write(doc, CORRECTED), "--audience", "work-safe")
    assert out.startswith("updated research_brief id=1")
    assert "audience label cleared" not in out
    assert "audience=work-safe" in out
    assert _artifact(web_tables)[5] == "work-safe"
    assert _chunks(web_tables) == []
