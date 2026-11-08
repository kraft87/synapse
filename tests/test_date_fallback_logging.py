"""The silent now() valid-from fallback is loud (logging only; no behavior change).

A fact edge's t_valid is ``in-text date or default_valid_at or now()``. Before 2026-07 a
failing source-episode lookup, an empty chunk lookup, or a missing segment date quietly
took the ``now()`` branch and mis-dated a large batch of facts. Retrieval and the writes
are unchanged here; what is pinned is that each way into that branch now leaves a log
line, with ids and counts only (never fact text or conversation content), and that the
rows written are exactly the ones written before.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from ingestion.db_documents import DocumentStore
from ingestion.extractor import ExtractionPipeline
from ingestion.models import ExtractedEntity, ExtractedFact, ExtractionResult

SEG = "2026-09-01T12:00:00+00:00"  # the segment (conversation) timestamp
IN_TEXT = "2026-03-15T00:00:00Z"  # a date the edge-date extractor read out of a fact


def _fact(i: int, **over) -> ExtractedFact:
    kw = {
        "source": f"S{i}",
        "target": f"T{i}",
        "relationship": "DOES",
        "fact": f"private-text-{i}",
    }
    kw.update(over)
    return ExtractedFact(**kw)


def _uuid_map(facts: list[ExtractedFact]) -> dict[str, str]:
    m: dict[str, str] = {}
    for f in facts:
        m[f.source] = f"u-{f.source}"
        m[f.target] = f"u-{f.target}"
    return m


def _pipe(n_facts: int, dates: list[tuple[str | None, str | None]] | None = None):
    pipe = ExtractionPipeline.__new__(ExtractionPipeline)
    pipe._kg = MagicMock()
    pipe._edge_date_extractor = MagicMock()
    pipe._edge_date_extractor.extract_batch.return_value = dates or [(None, None)] * n_facts
    pipe._contradiction_detector = MagicMock()
    pipe._contradiction_detector.detect_contradictions_batch.return_value = [
        [] for _ in range(n_facts)
    ]
    return pipe


def _rows(pipe) -> list[dict]:
    calls = pipe._kg.create_edges_batch.call_args_list
    return calls[0].args[0] if calls else []


def _fallback_records(caplog) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.name == "ingestion.extraction_edges" and "t_valid=now()" in r.getMessage()
    ]


# ---------------------------------------------------------------------------
# Stage 7: conversation facts written with the write clock as t_valid
# ---------------------------------------------------------------------------


class TestNowFallbackWarning:
    def test_one_warning_per_batch_with_count_and_ids(self, caplog):
        facts = [_fact(0), _fact(1), _fact(2)]
        # fact 1 carries an in-text date, so only facts 0 and 2 take the now() branch
        pipe = _pipe(3, dates=[(None, None), (IN_TEXT, None), (None, None)])
        with caplog.at_level(logging.WARNING, logger="ingestion.extraction_edges"):
            pipe._stage7_write_edges(
                facts, _uuid_map(facts), [41, 42], "g-tech", set(), {}, queue_id=1234
            )
        (rec,) = _fallback_records(caplog)
        assert rec.levelno == logging.WARNING
        msg = rec.getMessage()
        assert msg.startswith("stage7: 2 of 3 fact(s)")
        assert "group=g-tech" in msg
        assert "queue_id=1234" in msg
        assert "source_episodes=2" in msg

    def test_warning_carries_no_fact_text(self, caplog):
        facts = [_fact(0), _fact(1)]
        pipe = _pipe(2)
        with caplog.at_level(logging.DEBUG):
            pipe._stage7_write_edges(facts, _uuid_map(facts), [], "g", set(), {}, queue_id=7)
        assert _fallback_records(caplog)
        assert "private-text" not in caplog.text

    def test_the_failed_lookup_case_is_the_one_it_flags(self, caplog):
        # episode_ids == [] and no default_valid_at is exactly what a swallowed lookup
        # failure leaves behind; the count is what makes the empty list visible.
        facts = [_fact(0)]
        pipe = _pipe(1)
        with caplog.at_level(logging.WARNING, logger="ingestion.extraction_edges"):
            pipe._stage7_write_edges(facts, _uuid_map(facts), [], "g", set(), {})
        (rec,) = _fallback_records(caplog)
        assert "source_episodes=0" in rec.getMessage()
        assert "queue_id=None" in rec.getMessage()

    def test_silent_when_the_segment_date_is_known(self, caplog):
        facts = [_fact(0), _fact(1)]
        pipe = _pipe(2)
        with caplog.at_level(logging.DEBUG):
            pipe._stage7_write_edges(
                facts, _uuid_map(facts), [1], "g", set(), {}, default_valid_at=SEG
            )
        assert _fallback_records(caplog) == []

    def test_silent_when_every_fact_has_its_own_date(self, caplog):
        facts = [_fact(0)]
        pipe = _pipe(1, dates=[(IN_TEXT, None)])
        with caplog.at_level(logging.DEBUG):
            pipe._stage7_write_edges(facts, _uuid_map(facts), [], "g", set(), {})
        assert _fallback_records(caplog) == []

    def test_silent_on_the_web_lane(self, caplog):
        # A web fact is dated by its page; this warning is about conversation facts.
        facts = [_fact(0)]
        pipe = _pipe(1)
        with caplog.at_level(logging.DEBUG):
            pipe._stage7_write_edges(facts, _uuid_map(facts), [], "g", set(), {}, web_artifact_id=9)
        assert _fallback_records(caplog) == []

    def test_facts_that_are_not_written_are_not_counted(self, caplog):
        # fact 0 is a skipped duplicate, fact 1 has unresolved endpoints: only fact 2 lands.
        facts = [_fact(0), _fact(1), _fact(2)]
        pipe = _pipe(3)
        umap = _uuid_map(facts)
        del umap["T1"]
        with caplog.at_level(logging.WARNING, logger="ingestion.extraction_edges"):
            pipe._stage7_write_edges(facts, umap, [], "g", {0}, {})
        (rec,) = _fallback_records(caplog)
        assert rec.getMessage().startswith("stage7: 1 of 3 fact(s)")

    def test_rows_written_are_unchanged(self, caplog):
        facts = [_fact(0), _fact(1)]
        pipe = _pipe(2, dates=[(None, None), (IN_TEXT, None)])
        with caplog.at_level(logging.WARNING, logger="ingestion.extraction_edges"):
            pipe._stage7_write_edges(facts, _uuid_map(facts), [], "g", set(), {})
        r0, r1 = _rows(pipe)
        assert r0["t_valid"] == r0["valid_at"] == r0["t_created"]  # the write clock, as before
        assert r1["t_valid"] == IN_TEXT

    def test_count_is_exposed_on_a_span_attribute(self, monkeypatch):
        import logfire

        seen: dict[str, object] = {}

        class _Span:
            def set_attribute(self, key, value):
                seen[key] = value

        @contextmanager
        def _span(name, **attrs):
            seen["span"] = name
            yield _Span()

        monkeypatch.setattr(logfire, "span", _span)
        facts = [_fact(0), _fact(1)]
        pipe = _pipe(2, dates=[(None, None), (IN_TEXT, None)])
        pipe._stage7_write_edges(facts, _uuid_map(facts), [], "g", set(), {})
        assert seen["span"] == "stage7_resolve_valid_at {n}"
        assert seen["now_fallback"] == 1

    def test_queue_id_reaches_the_warning_through_the_group_path(self, caplog):
        # process_item hands the queue id to _process_facts_for_group, which must pass it
        # on to Stage 7; the unit tests above call Stage 7 directly and would not notice.
        facts = [_fact(0)]
        pipe = _pipe(1)
        pipe._embedder = MagicMock()
        pipe._embedder.embed.side_effect = lambda texts, task=None: [[0.0, 0.0, 1.0] for _ in texts]
        pipe._stage6a_embedding_filter = MagicMock(return_value={})
        pipe._stage6b_batch_confirm = MagicMock(return_value=(set(), {}, {}, True))
        pipe._kg.nearest_live_edges.side_effect = lambda embs, *a, **k: [[] for _ in embs]
        with caplog.at_level(logging.WARNING, logger="ingestion.extraction_edges"):
            pipe._process_facts_for_group(facts, _uuid_map(facts), [], "g", queue_id=99)
        (rec,) = _fallback_records(caplog)
        assert "queue_id=99" in rec.getMessage()
        assert len(_rows(pipe)) == 1


# ---------------------------------------------------------------------------
# process_item: a source-episode lookup that raises is logged, then ignored as before
# ---------------------------------------------------------------------------


def _item_pipeline(db) -> ExtractionPipeline:
    embedder = MagicMock()
    embedder.embed.side_effect = lambda names, task=None: [[0.0, 0.0, 0.0, 0.0] for _ in names]
    kg = MagicMock()
    kg.find_similar_nodes.return_value = []
    pipe = ExtractionPipeline(db=db, llm_client=MagicMock(), embedder=embedder, kg_client=kg)
    entities = [
        ExtractedEntity(name="User", type="Person", summary=""),
        ExtractedEntity(name="Zwift", type="Tool", summary=""),
    ]
    facts = [_fact(0, source="User", target="Zwift")]
    pipe._stage2_deterministic = MagicMock(return_value=[])
    pipe._stage3_llm = MagicMock(return_value=ExtractionResult(entities=entities, facts=facts))
    pipe._stage4_resolve = MagicMock(
        side_effect=lambda ents, group_id, deduper=None: {e.name: f"new:{e.name}" for e in ents}
    )
    pipe._stage5_write_nodes = MagicMock()
    pipe._process_facts_for_group = MagicMock()
    pipe._deduper_for = MagicMock(return_value=MagicMock())
    return pipe


@pytest.mark.parametrize(
    ("content_type", "lookup"),
    [
        ("chunk", "get_chunk_episode_ids"),
        ("summary", "get_synth_document_source_ids"),
    ],
)
def test_failed_episode_lookup_is_logged_and_extraction_continues(caplog, content_type, lookup):
    db = MagicMock()
    getattr(db, lookup).side_effect = RuntimeError("connection reset")
    pipe = _item_pipeline(db)
    item = {
        "id": 555,
        "content_type": content_type,
        "content": "private-conversation-text",
        "project": None,
        "session_id": "sess-1",
    }
    with caplog.at_level(logging.WARNING, logger="ingestion.extractor"):
        pipe.process_item(item)

    (rec,) = [r for r in caplog.records if r.name == "ingestion.extractor"]
    assert rec.levelno == logging.WARNING
    msg = rec.getMessage()
    assert lookup in msg
    assert "queue_id=555" in msg
    assert f"content_type={content_type}" in msg
    assert "RuntimeError: connection reset" in msg
    assert "private-conversation-text" not in caplog.text

    # Same behavior as the bare except it replaces: no source episodes, no segment date,
    # and the facts are still processed (with the queue id for the stage-7 warning).
    db.get_episodes_valid_at.assert_not_called()
    kwargs = pipe._process_facts_for_group.call_args.kwargs
    assert pipe._process_facts_for_group.call_args.args[2] == []  # episode_ids
    assert kwargs["default_valid_at"] is None
    assert kwargs["reference_time"] is None
    assert kwargs["queue_id"] == 555


def test_successful_lookup_logs_nothing(caplog):
    db = MagicMock()
    db.get_chunk_episode_ids.return_value = [1, 2]
    db.get_episodes_valid_at.return_value = SEG
    pipe = _item_pipeline(db)
    with caplog.at_level(logging.DEBUG, logger="ingestion.extractor"):
        pipe.process_item(
            {"id": 1, "content_type": "chunk", "content": "x", "session_id": "sess-1"}
        )
    assert [r for r in caplog.records if r.name == "ingestion.extractor"] == []
    assert pipe._process_facts_for_group.call_args.kwargs["default_valid_at"] == SEG


# ---------------------------------------------------------------------------
# db_documents: an empty chunk lookup is logged and still returns []
# ---------------------------------------------------------------------------


class _FakeConn:
    def __init__(self, row):
        self._row = row

    def execute(self, *_a, **_kw):
        return self

    def fetchone(self):
        return self._row


def _store(row) -> DocumentStore:
    store = DocumentStore("postgresql://unused/unused")

    @contextmanager
    def _conn():
        yield _FakeConn(row)

    store._conn = _conn  # type: ignore[method-assign]
    return store


def _lookup_records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == "ingestion.db_documents"]


@pytest.mark.parametrize(
    ("row", "reason"),
    [
        (None, "no chunk matches"),
        ({"episode_ids": None}, "chunk has no episode_ids"),
    ],
)
def test_empty_chunk_lookup_is_logged_and_returns_empty(caplog, row, reason):
    store = _store(row)
    with caplog.at_level(logging.WARNING, logger="ingestion.db_documents"):
        assert store.get_chunk_episode_ids("sess-9", "private-chunk-text") == []
    (rec,) = _lookup_records(caplog)
    assert rec.levelno == logging.WARNING
    msg = rec.getMessage()
    assert reason in msg
    assert "session_id=sess-9" in msg
    assert f"content_len={len('private-chunk-text')}" in msg
    assert "private-chunk-text" not in caplog.text


@pytest.mark.parametrize("raw", [[3, 4], "[3, 4]"])
def test_found_chunk_lookup_logs_nothing(caplog, raw):
    store = _store({"episode_ids": raw})
    with caplog.at_level(logging.DEBUG, logger="ingestion.db_documents"):
        assert store.get_chunk_episode_ids("sess-9", "x") == [3, 4]
    assert _lookup_records(caplog) == []
