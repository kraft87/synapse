"""Notes leg (_search_notes): hook-KNN over the live notes store -> cross-encoder floor
on "hook — body" text -> top notes_limit served as n:-id items. Pure-logic — the KNN
query is stubbed at mcp_server.recall_sources, the reranker on the instance. The floor
self-gates like web (all-subfloor serves []); any DB failure serves [] (fail-soft)."""

from __future__ import annotations

from contextlib import nullcontext
from typing import ClassVar

import mcp_server.recall_sources as sources_mod
from mcp_server.recall import Recall
from mcp_server.recall_config import RecallConfig


def _bare(notes_floor: float = 0.60) -> Recall:
    r = object.__new__(Recall)
    r._db_url = "postgresql://stub"
    r._config = RecallConfig(notes_floor=notes_floor)
    r._pg = lambda: nullcontext(object())  # type: ignore[method-assign]
    return r


class _RR:
    def __init__(self, scored):
        self._scored = scored

    def rerank_scored(self, query, docs):
        if self._scored == "raise":
            raise RuntimeError("voyage down")
        return self._scored


def _note(i: int, body: str = "", project: str | None = None) -> dict:
    return {
        "id": i,
        "hook": f"hook{i}",
        "body": body or f"body{i}",
        "type": "project" if project else "user",
        "project": project,
        "sim": 0.9,
    }


class _DB:
    rows: ClassVar[list[dict]] = []
    fail = False
    last_kwargs: ClassVar[dict] = {}


def _search_live_notes(conn, owner_id, project, embedding, limit=24, audience=None):
    if _DB.fail:
        raise RuntimeError("db down")
    _DB.last_kwargs = {
        "owner_id": owner_id,
        "project": project,
        "limit": limit,
        "audience": audience,
    }
    return list(_DB.rows)


def _wire(monkeypatch, rows, scored, floor=0.60):
    _DB.rows = rows
    _DB.fail = False
    monkeypatch.setattr(sources_mod, "search_live_notes", _search_live_notes)
    r = _bare(notes_floor=floor)
    r._reranker = _RR(scored)
    return r


def test_notes_served_in_relevance_order_with_ids(monkeypatch):
    rows = [_note(1), _note(2, project="synapse"), _note(3)]
    # rerank promotes note 3 over note 1; note 2 subfloor
    r = _wire(monkeypatch, rows, [(2, 0.9), (0, 0.7), (1, 0.4)])
    out = r._search_notes("q", [0.0], None)
    assert [it["id"] for it in out] == ["n:3", "n:1"]
    assert out[0]["hook"] == "hook3" and out[0]["note"] == "body3"
    assert "project" not in out[0]  # user note carries no project key


def test_notes_project_key_and_scope_passthrough(monkeypatch):
    rows = [_note(7, project="synapse")]
    r = _wire(monkeypatch, rows, [(0, 0.9)])
    out = r._search_notes("q", [0.0], "synapse")
    assert out[0]["project"] == "synapse"
    assert _DB.last_kwargs["project"] == "synapse"
    assert _DB.last_kwargs["limit"] == r._config.notes_fetch


def test_notes_all_subfloor_serves_nothing(monkeypatch):
    r = _wire(monkeypatch, [_note(1), _note(2)], [(0, 0.3), (1, 0.2)])
    assert r._search_notes("q", [0.0], None) == []


def test_notes_limit_caps_served(monkeypatch):
    rows = [_note(i) for i in range(1, 7)]
    scored = [(i, 0.9 - i * 0.01) for i in range(6)]
    r = _wire(monkeypatch, rows, scored)
    out = r._search_notes("q", [0.0], None)
    assert len(out) == r._config.notes_limit


def test_notes_body_capped_full_via_fetch(monkeypatch):
    r = _wire(monkeypatch, [], [(0, 0.9)])
    cap = r._config.notes_body_cap
    _DB.rows = [_note(1, body="x" * (cap + 50))]
    out = r._search_notes("q", [0.0], None)
    assert len(out[0]["note"]) == cap + 1  # cap + ellipsis
    assert out[0]["note"].endswith("…")


def test_notes_db_failure_serves_nothing(monkeypatch):
    r = _wire(monkeypatch, [], [(0, 0.9)])
    _DB.fail = True
    assert r._search_notes("q", [0.0], None) == []


def test_notes_reranker_outage_degrades_to_knn_order(monkeypatch):
    # _floor_by_rerank keeps all on rerank failure — leg serves KNN order, never fails
    rows = [_note(1), _note(2), _note(3), _note(4)]
    r = _wire(monkeypatch, rows, "raise")
    out = r._search_notes("q", [0.0], None)
    assert [it["id"] for it in out] == ["n:1", "n:2", "n:3"]


def test_notes_floor_disabled_serves_knn_order(monkeypatch):
    rows = [_note(1), _note(2)]
    r = _wire(monkeypatch, rows, [(0, 0.9), (1, 0.8)], floor=0.0)
    out = r._search_notes("q", [0.0], None)
    assert [it["id"] for it in out] == ["n:1", "n:2"]


def test_notes_audience_filter_reaches_the_store(monkeypatch):
    """The restricted tier filter is pushed into SQL, not applied after the KNN.

    Filtering post-hoc would spend the fetch budget on rows the caller can never see,
    so a restricted surface's notes bucket would thin out for the wrong reason."""
    r = _wire(monkeypatch, [_note(1)], [(0, 0.9)])
    r._search_notes("q", [0.0], None, audience="work-safe")
    assert _DB.last_kwargs["audience"] == "work-safe"
    # Full trust passes None — no filter at all, not "both tiers" spelled out.
    r._search_notes("q", [0.0], None)
    assert _DB.last_kwargs["audience"] is None
