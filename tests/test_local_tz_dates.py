"""Dates resolve on the user's local calendar day, not the UTC one (prod, 2026-10-11).

Episode 242826 was written at 2026-10-11 00:30:44 UTC, which is 20:30 on 2026-10-10 in
Mississauga (America/Toronto). The user said "Got my first ride in finally." and the
extractor wrote "User got their first ride in on Zwift on 2026-10-11 (meaning 2026-10-11)",
and recall served the turn dated 2026-10-11. Every date was the UTC calendar day of
episodes.created_at (``str(ts)[:10]`` of a timestamp the UTC-session Postgres returns
in UTC), so any turn after 20:00 EDT / 19:00 EST was anchored to the next day.

Only symbols that predate the fix are imported, so this file also runs against the
pre-fix code, where every reproduction fails on its assertion.
"""

from __future__ import annotations

from contextlib import nullcontext
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from ingestion.models import ExtractedEntity, ExtractedFact, ExtractionResult
from mcp_server.recall import Recall, _to_recall_item

# Episode 242826's created_at as the DB layer hands it over (get_episodes_valid_at).
EVENING = "2026-10-11T00:30:44.296000+00:00"  # 20:30 EDT on 2026-10-10
EVENING_DT = datetime(2026, 10, 11, 0, 30, 44, 296000, tzinfo=UTC)
WINTER_EVENING = "2026-01-15T00:30:00+00:00"  # 19:30 EST on 2026-01-14
DAYTIME = "2026-10-10T14:31:00+00:00"  # 10:31 EDT, same day in both zones


@pytest.fixture(autouse=True)
def _toronto(monkeypatch):
    """The deployment's setting; individual tests override it."""
    monkeypatch.setenv("SYNAPSE_TIMEZONE", "America/Toronto")


# ---------------------------------------------------------------------------
# Extraction: the session date handed to the prompt + the edge-date reference time
# ---------------------------------------------------------------------------


def _pipeline(valid_at: str):
    from ingestion.extractor import ExtractionPipeline

    embedder = MagicMock()
    embedder.embed.side_effect = lambda names, task=None: [[0.0, 0.0, 0.0, 0.0] for _ in names]
    db = MagicMock()
    db.get_chunk_episode_ids.return_value = [242823, 242824, 242825, 242826]
    db.get_episodes_valid_at.return_value = valid_at
    kg = MagicMock()
    kg.find_similar_nodes.return_value = []
    pipe = ExtractionPipeline(db=db, llm_client=MagicMock(), embedder=embedder, kg_client=kg)
    entities = [
        ExtractedEntity(name="User", type="Person", summary=""),
        ExtractedEntity(name="Zwift", type="Tool", summary=""),
    ]
    facts = [
        ExtractedFact(
            source="User",
            target="Zwift",
            relationship="RODE_ON",
            fact="User got their first ride in on Zwift",
        )
    ]
    pipe._stage2_deterministic = MagicMock(return_value=[])
    pipe._stage3_llm = MagicMock(return_value=ExtractionResult(entities=entities, facts=facts))
    pipe._stage4_resolve = MagicMock(
        side_effect=lambda ents, group_id, deduper=None: {e.name: f"new:{e.name}" for e in ents}
    )
    pipe._stage5_write_nodes = MagicMock()
    pipe._process_facts_for_group = MagicMock()
    pipe._deduper_for = MagicMock(return_value=MagicMock())
    return pipe


def _run_chunk(valid_at: str):
    pipe = _pipeline(valid_at)
    pipe.process_item(
        {
            "content_type": "chunk",
            "content": "[user] Got my first ride in finally. Works like a charm.",
            "project": None,
            "session_id": "98d71966-312e-4c84-a93d-b2aca1474b30",
        }
    )
    session_date = pipe._stage3_llm.call_args.args[2]
    kwargs = pipe._process_facts_for_group.call_args.kwargs
    return session_date, kwargs


def test_evening_turn_session_date_is_the_local_day():
    session_date, _ = _run_chunk(EVENING)
    assert session_date == "2026-10-10"


def test_session_date_defaults_to_toronto_when_unset(monkeypatch):
    monkeypatch.delenv("SYNAPSE_TIMEZONE", raising=False)
    session_date, _ = _run_chunk(EVENING)
    assert session_date == "2026-10-10"


def test_winter_evening_turn_uses_standard_time():
    session_date, _ = _run_chunk(WINTER_EVENING)
    assert session_date == "2026-01-14"


def test_daytime_turn_is_the_same_day_either_way():
    session_date, _ = _run_chunk(DAYTIME)
    assert session_date == "2026-10-10"


def test_timezone_setting_is_honored(monkeypatch):
    monkeypatch.setenv("SYNAPSE_TIMEZONE", "Asia/Tokyo")
    session_date, _ = _run_chunk(DAYTIME)  # 23:31 JST
    assert session_date == "2026-10-10"
    session_date, _ = _run_chunk("2026-10-10T16:00:00+00:00")  # 01:00 JST next day
    assert session_date == "2026-10-11"


def test_edge_date_reference_time_is_the_local_wall_clock():
    """The edge-date extractor resolves "yesterday" etc. against reference_time, so it
    must carry the local wall clock. It is the same instant, and the fact's default
    t_valid stays that same instant."""
    _, kwargs = _run_chunk(EVENING)
    ref = kwargs["reference_time"]
    assert ref.startswith("2026-10-10T20:30:44")
    assert datetime.fromisoformat(ref) == EVENING_DT
    assert datetime.fromisoformat(kwargs["default_valid_at"]) == EVENING_DT


# ---------------------------------------------------------------------------
# Timeline gate: "This turn happened on <date>"
# ---------------------------------------------------------------------------


class _GateDb:
    def __init__(self, valid_at: str) -> None:
        self._valid_at = valid_at
        self.inserted: list[dict] = []

    def get_episodes_valid_at(self, ids):
        return self._valid_at

    def timeline_ident_exists(self, *a, **k):
        return False

    def insert_timeline_event(self, **kw):
        self.inserted.append(kw)
        return 1


class _StubEmb:
    model_name = "stub"

    def embed(self, texts, task=None):
        return [[0.1, 0.2, 0.3] for _ in texts]


def _gate(monkeypatch, valid_at: str, date: str | None = None):
    import ingestion.timeline_gate as tg
    from ingestion.llm_schemas import TimelineGateEvents

    monkeypatch.setenv("SYNAPSE_TIMELINE_GATE", "1")
    monkeypatch.setenv("SYNAPSE_TIMELINE_DEDUP", "0")
    prompts: list[str] = []

    def _call(*a, **k):
        prompts.append(k["base_prompt"])
        return TimelineGateEvents.model_validate(
            {
                "events": [
                    {
                        "event": "completed first ride in the Docker-based Zwift setup",
                        "salience": 1,
                        "event_type": "milestone",
                        "date": date,
                    }
                ]
            }
        )

    monkeypatch.setattr(tg, "structured_call", _call)
    db = _GateDb(valid_at)
    g = tg.TimelineGate(db=db, llm_client=object(), embedder=_StubEmb())
    g.process({"id": 1, "episode_id": 242826, "content": "x" * 500, "project": None})
    return prompts[0], db.inserted


def test_timeline_gate_turn_date_is_the_local_day(monkeypatch):
    prompt, inserted = _gate(monkeypatch, EVENING)
    assert "This turn happened on 2026-10-10." in prompt
    # A same-day event keeps the precise turn instant.
    assert datetime.fromisoformat(str(inserted[0]["t_valid"])) == EVENING_DT


def test_timeline_gate_same_local_day_date_keeps_turn_instant(monkeypatch):
    """The gate naming the local turn day is "this turn", not a past day: the event
    keeps the turn's own timestamp instead of a noon stamp."""
    _, inserted = _gate(monkeypatch, EVENING, date="2026-10-10")
    assert datetime.fromisoformat(str(inserted[0]["t_valid"])) == EVENING_DT


def test_timeline_gate_past_date_is_judged_against_the_local_day(monkeypatch):
    _, inserted = _gate(monkeypatch, EVENING, date="2026-10-09")
    assert inserted[0]["t_valid"] == "2026-10-09T12:00:00+00:00"


# ---------------------------------------------------------------------------
# Recall: the date served on episodes
# ---------------------------------------------------------------------------


def test_recall_item_date_is_the_local_day():
    item = _to_recall_item({"id": "e:242826", "content": "c", "created_at": EVENING_DT})
    assert item["date"] == "2026-10-10"


def test_recall_item_daytime_date_unchanged():
    item = _to_recall_item(
        {"id": "e:1", "content": "c", "created_at": datetime(2026, 10, 10, 14, 31, tzinfo=UTC)}
    )
    assert item["date"] == "2026-10-10"


def test_recall_item_accepts_iso_strings():
    assert _to_recall_item({"id": "e:1", "content": "c", "created_at": EVENING})["date"] == (
        "2026-10-10"
    )


def test_passage_date_is_the_local_day():
    long = "".join(f"\n## Section {i}\nlorem ipsum dolor sit amet " * 6 for i in range(24))

    class _Emb:
        def rerank_scored(self, query, documents, top_k=None):
            return [(i, 1.0 - 0.01 * i) for i in range(len(documents))][: top_k or None]

    r = Recall("", "")
    r._reranker = _Emb()
    out = r._compact_to_passages(
        "q", [{"id": "e:242826", "content": long, "created_at": EVENING_DT}], n=1
    )
    assert out[0]["date"] == "2026-10-10"


def test_fetch_session_dates_are_the_local_day(monkeypatch):
    import mcp_server.recall as recall_mod
    from ingestion.surfaces import FULL_TRUST

    monkeypatch.setattr(recall_mod, "lookup_surface", lambda _u, _s: FULL_TRUST)
    rows = [
        {
            "id": 242826,
            "sequence": 1,
            "content": "Got my first ride in finally.",
            "created_at": EVENING_DT,
            "has_h": True,
            "has_a": True,
        }
    ]

    class _Rows:
        def __init__(self, rows):
            self._rows = rows

        def fetchall(self):
            return self._rows

        def fetchone(self):
            return self._rows[0]

    class _PG:
        def execute(self, sql, params=()):
            if "count(*)" in sql:
                return _Rows([{"n": 1, "first": EVENING_DT, "last": EVENING_DT, "project": None}])
            return _Rows(rows)

    r = object.__new__(Recall)
    r._db_url = "postgresql://stub"
    r._pg = lambda: nullcontext(_PG())
    r._record_metrics = lambda *a, **k: None
    r._increment_retrieval_counts = lambda ids: None
    out = r.fetch_session("98d71966-312e-4c84-a93d-b2aca1474b30")
    assert out["first_date"] == "2026-10-10"
    assert out["last_date"] == "2026-10-10"
    assert out["turns"][0]["date"] == "2026-10-10"


# ---------------------------------------------------------------------------
# End to end on the real DB: an episode row at 2026-10-11T00:30:44Z
# ---------------------------------------------------------------------------


@pytest.mark.db
def test_db_episode_after_8pm_edt_gets_local_session_date(db_url, clean_tables):
    from ingestion.db import Database
    from ingestion.extractor import ExtractionPipeline
    from ingestion.models import Episode

    db = Database(db_url)
    try:
        ep_id = db.upsert_episode(
            Episode(
                session_id="tz-repro",
                sequence=1,
                content="[user] Got my first ride in finally.",
                created_at=EVENING_DT,
            )
        )
        pipe = ExtractionPipeline(
            db=db, llm_client=MagicMock(), embedder=MagicMock(), kg_client=MagicMock()
        )
        pipe._stage2_deterministic = MagicMock(return_value=[])
        pipe._stage3_llm = MagicMock(return_value=ExtractionResult(entities=[], facts=[]))
        pipe.process_item(
            {
                "content_type": "summary",
                "content": "Got my first ride in finally.",
                "project": None,
                "session_id": "tz-repro",
                "episode_id": ep_id,
            }
        )
        assert pipe._stage3_llm.call_args.args[2] == "2026-10-10"
    finally:
        db.close()
