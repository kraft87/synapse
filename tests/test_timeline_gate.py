"""Unit tests for the timeline chat gate (ingestion/timeline_gate.py). Pure — the
LLM/db/embedder are stubs; the live Haiku + PG path was validated separately."""

from __future__ import annotations

import pytest

from ingestion.llm_client import MalformedResponseError
from ingestion.timeline_gate import TimelineGate, _parse_gate


def test_parse_gate_null_is_skip():
    assert _parse_gate('{"event": null}') is None


def test_parse_gate_event_and_salience():
    d = _parse_gate(
        'noise {"event": "fixed the dating bug", "salience": 2, "event_type": "action"} trailing'
    )
    assert d == {"event": "fixed the dating bug", "salience": 2, "event_type": "action"}


def test_parse_gate_bad_event_type_nulls():
    d = _parse_gate('{"event": "did a thing", "salience": 1, "event_type": "vibe"}')
    assert d["event_type"] is None


def test_parse_gate_bad_salience_clamps_to_med():
    d = _parse_gate('{"event": "ran the benchmark", "salience": 9}')
    assert d["salience"] == 1


def test_parse_gate_malformed_raises():
    with pytest.raises(MalformedResponseError):
        _parse_gate("no json here at all")


class _Boom:
    def __getattr__(self, _):  # any use of llm/db/embedder would explode
        raise AssertionError("should not be touched")


def _gate(enabled=True, monkeypatch=None):
    if monkeypatch:
        monkeypatch.setenv("SYNAPSE_TIMELINE_GATE", "1" if enabled else "0")
    return TimelineGate(db=_Boom(), llm_client=_Boom(), embedder=_Boom())


def test_disabled_gate_is_noop(monkeypatch):
    g = _gate(enabled=False, monkeypatch=monkeypatch)
    g.process({"id": 1, "episode_id": 5, "content": "x" * 500})  # would explode if it ran


def test_short_content_skips_llm(monkeypatch):
    g = _gate(enabled=True, monkeypatch=monkeypatch)
    g.process({"id": 1, "episode_id": 5, "content": "ok thanks"})  # under _MIN_CONTENT


def test_missing_episode_id_skips(monkeypatch):
    g = _gate(enabled=True, monkeypatch=monkeypatch)
    g.process({"id": 1, "content": "x" * 500})


def test_errors_are_swallowed(monkeypatch):
    # llm blows up -> process() logs and returns, never raises (KG work must not break)
    g = _gate(enabled=True, monkeypatch=monkeypatch)
    g.process({"id": 1, "episode_id": 5, "content": "x" * 500})


# ---- identifier dedup (write-time) ----


def test_extract_idents():
    from ingestion.timeline_gate import extract_idents

    f = "shipped PR #193 (commit 068074a8) and fixed #99"
    assert extract_idents(f) == ["#193", "068074a8", "#99"]
    assert extract_idents("decided to use halfvec everywhere") == []


class _Rec:
    """Records calls; configurable ident_exists answer."""

    def __init__(self, exists):
        self._exists = exists
        self.inserted = []

    def get_episodes_valid_at(self, ids):
        return "2026-07-02T10:00:00+00:00"

    def timeline_ident_exists(self, idents, project, t_valid, window_hours):
        return self._exists

    def insert_timeline_event(self, **kw):
        self.inserted.append(kw)
        return 1


class _StubEmb:
    def embed(self, texts, task):
        return [[0.0] * 4 for _ in texts]


def _gated(monkeypatch, event, exists):
    import ingestion.timeline_gate as tg

    monkeypatch.setenv("SYNAPSE_TIMELINE_GATE", "1")
    db = _Rec(exists)
    g = tg.TimelineGate(db=db, llm_client=object(), embedder=_StubEmb())
    monkeypatch.setattr(
        tg, "parse_with_retry", lambda *a, **k: {"event": event, "salience": 1, "event_type": None}
    )
    g.process({"id": 1, "episode_id": 5, "content": "x" * 500, "project": "synapse"})
    return db.inserted


def test_commit_announcement_with_known_ident_skips(monkeypatch):
    assert _gated(monkeypatch, "committed PR #103 to main", exists=True) == []


def test_deploy_with_known_ident_still_writes(monkeypatch):
    assert len(_gated(monkeypatch, "deployed PR #103 to production", exists=True)) == 1


def test_commit_announcement_with_fresh_ident_writes(monkeypatch):
    assert len(_gated(monkeypatch, "committed PR #103 to main", exists=False)) == 1
