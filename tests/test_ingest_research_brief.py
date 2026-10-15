"""scripts/ingest_research_brief.py: a brief is a page the poller carries to recall, and the
graph reads it (kg_extract), since someone wrote it on purpose."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "ingest_research_brief.py"
_spec = importlib.util.spec_from_file_location("ingest_research_brief", _SCRIPT)
assert _spec and _spec.loader
brief = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(brief)


@pytest.fixture(autouse=True)
def _clean(conn):
    conn.execute("TRUNCATE web_artifacts RESTART IDENTITY CASCADE")
    yield
    conn.execute("TRUNCATE web_artifacts RESTART IDENTITY CASCADE")


def test_a_brief_is_stored_marked_for_extraction_and_once(conn, db_url, tmp_path, monkeypatch):
    path = tmp_path / "brief.md"
    path.write_text("# How RAG ingests pages\n\n" + "A synthesized finding. " * 30)
    monkeypatch.setenv("SYNAPSE_DB_URL", db_url)
    monkeypatch.setattr(sys, "argv", ["x", "--file", str(path), "--slug", "rag-ingest"])
    assert brief.main() == 0
    assert brief.main() == 0  # idempotent by tool_use_id
    ((kind, title, meta),) = conn.execute(
        "SELECT kind, title, metadata FROM web_artifacts"
    ).fetchall()
    assert (kind, title) == ("research_brief", "How RAG ingests pages")
    assert meta == {"capture": "research_brief", "kg_extract": True}
