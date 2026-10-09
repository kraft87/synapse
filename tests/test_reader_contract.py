"""Reader contract: a served fact supports exactly what it states.

Two production incidents motivated this (2026-10): a habit extracted once was served as a
present-tense fact for a year with nothing marking it unconfirmed, and a fact that a
PROFILE listed an activity was read as the user doing that activity, then written up as a
first-person anecdote. The store was right in the second case; the reader over-read it.

No schema change stops a model from over-reading an honest fact, so the contract lives in
the always-loaded surfaces (server instructions + the recall tool description) and the
serving shape carries the currency signals the contract refers to. These tests pin both,
driven by SYNTHETIC fixtures (tests/fixtures/reader_contract.json) — every case there is
invented. The fixtures double as an eval set for any model-in-the-loop check: ``licensed``
is what a reader may say from the served fact, ``forbidden`` what it may not.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingestion.surfaces import SurfaceTrust
from mcp_server.recall import Recall

_FIXTURE = Path(__file__).parent / "fixtures" / "reader_contract.json"
_NO_DB = "postgresql://synapse@127.0.0.1:1/nonexistent"
_SERVED_KEYS = {"fact", "id", "date", "ongoing", "supported"}
_FULL = SurfaceTrust(surface_id="dev-t", trust="full", known=True, credential_bound=True)


def _cases() -> list[dict]:
    return json.loads(_FIXTURE.read_text())["cases"]


class _FakeEmbedder:
    def embed(self, texts, task="query"):
        return [[0.0] * 4 for _ in texts]


class _FakeTimeline:
    def recall_timeline(self, **kwargs):
        return {"items": []}


def _wired(kg_facts: list[dict]) -> Recall:
    """A Recall with every leg stubbed except the KG fact leg, which serves ``kg_facts``
    in the internal shape (_uuid/_date/_ongoing/_supported)."""
    r = Recall(_NO_DB, "")
    r._ensure_embedder = lambda: _FakeEmbedder()
    r._ensure_timeline = lambda: _FakeTimeline()
    r._search_bm25_episodes = lambda q, proj, limit, sid=None, allowed=None, own=None: []
    r._search_vector_episodes = lambda emb, proj, limit, sid=None, allowed=None, own=None: []
    r._search_vector_web = lambda emb, n: []
    r._search_bm25_web = lambda q, n: []
    r._search_kg = lambda *a, **k: (list(kg_facts), [])
    r._search_notes = lambda *a, **k: []
    r._fetch_history_pairs_pg = lambda gid, uuids, cap: []
    r._surface_supersessions = lambda *a, **k: []
    r._episode_supersessions = lambda *a, **k: {}
    r._compact_to_passages = lambda q, eps, n: []
    r._increment_fact_retrieval_counts = lambda *a, **k: None
    r._increment_retrieval_counts = lambda ids: None
    r._rerank_pool_scored = lambda q, pl: []
    r._record_metrics = lambda row: None
    return r


def _internal(served: dict, uid: str) -> dict:
    """Fixture 'served' shape -> the KG leg's internal row."""
    row = {"fact": served["fact"], "_uuid": uid, "_date": served.get("date")}
    row["_ongoing"] = served.get("ongoing")
    row["_supported"] = served.get("supported")
    return row


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["id"])
def test_fixture_is_well_formed_and_synthetic(case):
    assert case["licensed"] and case["forbidden"] and case["expected_reader_move"]
    assert set(case["served"]) <= _SERVED_KEYS
    assert "User" in case["served"]["fact"]  # the extractor's placeholder subject, never a name


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["id"])
def test_served_fact_carries_its_currency_signals(case):
    r = _wired([_internal(case["served"], "u-1")])
    out = r.recall(query="q", project=None, session_focus=[], group_id="technical", trust=_FULL)
    (fact,) = out["facts"]
    assert fact["id"] == "f:u-1"
    assert fact["fact"] == case["served"]["fact"]
    assert fact.get("date") == case["served"].get("date")
    # `ongoing` appears only when true; `supported` only when the user ever asserted it
    assert fact.get("ongoing") == case["served"].get("ongoing")
    assert fact.get("supported") == case["served"].get("supported")
    assert set(fact) <= _SERVED_KEYS


def test_pre_058_facts_serve_exactly_as_before():
    r = _wired(
        [{"fact": "User uses Postgres", "_uuid": "u-2", "_date": "2026-06-10T00:00:00+00:00"}]
    )
    out = r.recall(query="q", project=None, session_focus=[], group_id="technical", trust=_FULL)
    assert out["facts"] == [{"fact": "User uses Postgres", "id": "f:u-2", "date": "2026-06-10"}]


def test_contract_is_on_the_always_loaded_surfaces():
    import mcp_server.server as server_mod
    from mcp_server.retrieval_tools import register as _register  # noqa: F401

    instr = server_mod._INSTRUCTIONS
    assert "supports exactly what it states" in instr
    assert "describes the document" in instr
    assert "user's voice" in instr
    doc = " ".join((server_mod.recall.__doc__ or "").split())  # fold docstring line breaks
    for needle in (
        "`ongoing: true`",
        "`supported`",
        "not current truth",
        "describes that document",
    ):
        assert needle in doc, needle
