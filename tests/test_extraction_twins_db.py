"""Same-batch twins through Stage 6 + 7 into the real kg_relationships table.

The LLM stages are stubbed and the graph starts empty, so Stage 6 finds no candidates
and every surviving fact is created. Pinned: a fanned-out restatement lands as ONE row
carrying the batch's episodes, mention_count 1 and the user support of the copy that was
dropped; facts that differ only in a number still land as two rows.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ingestion.extractor import ExtractionPipeline
from ingestion.kg_client import KGClient
from ingestion.models import ExtractedFact
from tests.helpers.embed import GROUP, onehot

SEG = "2026-09-01T12:00:00+00:00"


@pytest.fixture()
def pipe(conn, monkeypatch, db_url):
    monkeypatch.setenv("SYNAPSE_DB_URL", db_url)
    conn.execute("TRUNCATE kg_entities, kg_relationships RESTART IDENTITY CASCADE")
    p = ExtractionPipeline.__new__(ExtractionPipeline)
    p._kg = KGClient()
    # Release-notes twins share one vector, the two PR facts another: cosine alone would
    # merge both pairs.
    p._embedder = MagicMock()
    p._embedder.embed.side_effect = lambda texts, task=None: [
        onehot(1 if "PR" in t else 0) for t in texts
    ]
    p._edge_date_extractor = MagicMock()
    p._edge_date_extractor.extract_batch.side_effect = lambda facts, reference_time=None: [
        (None, None) for _ in facts
    ]
    p._contradiction_detector = MagicMock()
    p._contradiction_detector.detect_contradictions_batch.side_effect = lambda facts, *a, **k: [
        [] for _ in facts
    ]
    return p


def _fact(text: str, source: str, target: str, **over) -> ExtractedFact:
    return ExtractedFact(source=source, target=target, relationship="R", fact=text, **over)


def test_twin_lands_once_with_batch_provenance(pipe, conn):
    facts = [
        _fact(
            "Alice and Bob reviewed the release notes", "Alice", "Notes", attribution="assistant"
        ),
        _fact("Bob and Alice reviewed the release notes", "Bob", "Notes", attribution="user"),
        _fact("PR #12 was merged into main", "PR #12", "main"),
        _fact("PR #13 was merged into main", "PR #13", "main"),
    ]
    uuid_map = {n: f"new:e-{n}" for f in facts for n in (f.source, f.target)}

    pipe._process_facts_for_group(facts, uuid_map, [41, 42], GROUP, default_valid_at=SEG)

    rows = conn.execute(
        "SELECT fact, src_uuid, episodes, mention_count, last_supported_at IS NOT NULL, "
        "last_supported_by FROM kg_relationships ORDER BY fact"
    ).fetchall()
    assert [r[0] for r in rows] == [
        "Alice and Bob reviewed the release notes",
        "PR #12 was merged into main",
        "PR #13 was merged into main",
    ]
    _, src, episodes, mention_count, supported, supported_by = rows[0]
    assert src == "e-Alice"
    assert episodes == [41, 42]
    assert mention_count == 1
    assert supported and supported_by == 42  # the dropped copy was the user-stated one
