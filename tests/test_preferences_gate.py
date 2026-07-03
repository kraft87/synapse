"""Unit tests for the preferences chat gate (ingestion/preferences_gate.py). Pure — the
LLM/db/embedder are stubs; the live Haiku + PG path is validated separately."""

from __future__ import annotations

import pytest

from ingestion.llm_client import MalformedResponseError
from ingestion.preferences_gate import (
    PreferencesGate,
    _parse_prefs,
    decide_pref_action,
)

# ---- JSON parser ----


def test_parse_prefs_empty_list():
    assert _parse_prefs('{"preferences": []}') == []


def test_parse_prefs_missing_key_is_empty():
    # A model that answers "nothing here" without the key parses to no preferences.
    assert _parse_prefs('{"note": "no preferences"}') == []


def test_parse_prefs_valid():
    d = _parse_prefs(
        'noise {"preferences": [{"pref": "User prefers bullet lists over tables", '
        '"polarity": "like"}, {"pref": "User dislikes em-dashes", "polarity": "dislike"}]} tail'
    )
    assert d == [
        {"pref": "User prefers bullet lists over tables", "polarity": "like"},
        {"pref": "User dislikes em-dashes", "polarity": "dislike"},
    ]


def test_parse_prefs_drops_bad_entries():
    d = _parse_prefs(
        '{"preferences": ['
        '{"pref": "  User likes concise answers  ", "polarity": "like"},'  # trimmed, kept
        '{"pref": "User wants X", "polarity": "vibe"},'  # bad polarity -> dropped
        '{"pref": "", "polarity": "like"},'  # empty pref -> dropped
        '{"polarity": "rule"},'  # missing pref -> dropped
        '"not-an-object"'  # wrong type -> dropped
        "]}"
    )
    assert d == [{"pref": "User likes concise answers", "polarity": "like"}]


def test_parse_prefs_non_list_raises():
    with pytest.raises(MalformedResponseError):
        _parse_prefs('{"preferences": "a string, not a list"}')


def test_parse_prefs_no_json_raises():
    with pytest.raises(MalformedResponseError):
        _parse_prefs("no json here at all")


# ---- dedup / supersession decision (pure) ----


def _cand(id, polarity, sim):
    return {"id": id, "polarity": polarity, "sim": sim}


def test_decide_no_candidates_inserts():
    assert decide_pref_action("like", []) == {"action": "insert", "target_id": None}


def test_decide_near_duplicate_reasserts():
    # >= 0.90 to the top live pref -> restatement, regardless of polarity.
    assert decide_pref_action("like", [_cand(7, "like", 0.94)]) == {
        "action": "reassert",
        "target_id": 7,
    }


def test_decide_reassert_wins_even_on_polarity_match_above_reassert_band():
    # A very-high sim is a restatement even if it's technically a different polarity label.
    assert decide_pref_action("rule", [_cand(3, "like", 0.97)])["action"] == "reassert"


def test_decide_stance_flip_in_band_supersedes():
    # 0.78..0.90 AND opposite polarity (like<->dislike) = the user flipped -> supersede.
    assert decide_pref_action("dislike", [_cand(5, "like", 0.84)]) == {
        "action": "supersede",
        "target_id": 5,
    }


def test_decide_same_polarity_in_band_inserts():
    # Similar topic, same stance = a distinct-but-adjacent preference -> new row.
    assert decide_pref_action("like", [_cand(5, "like", 0.84)]) == {
        "action": "insert",
        "target_id": None,
    }


def test_decide_rule_change_in_band_is_not_a_contradiction():
    # rule vs like is not a detectable polarity flip -> conservatively insert.
    assert decide_pref_action("rule", [_cand(5, "like", 0.84)])["action"] == "insert"


def test_decide_below_supersede_band_inserts():
    assert decide_pref_action("dislike", [_cand(5, "like", 0.60)]) == {
        "action": "insert",
        "target_id": None,
    }


def test_decide_uses_top_candidate_only():
    # Candidates are descending by sim; the nearest one drives the decision.
    cands = [_cand(1, "like", 0.95), _cand(2, "dislike", 0.99)]
    assert decide_pref_action("like", cands) == {"action": "reassert", "target_id": 1}


# ---- gate guards (fail-soft, env, short-content) ----


class _Boom:
    def __getattr__(self, _):  # any use of llm/db/embedder would explode
        raise AssertionError("should not be touched")


def _gate(enabled=True, monkeypatch=None):
    if monkeypatch:
        monkeypatch.setenv("SYNAPSE_PREFS_GATE", "1" if enabled else "0")
    return PreferencesGate(db=_Boom(), llm_client=_Boom(), embedder=_Boom())


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


# ---- full write flow (stubbed db + embedder + gate return) ----


class _RecDB:
    """Records reconciliation calls; returns configurable KNN candidates."""

    def __init__(self, candidates):
        self._candidates = candidates
        self.inserted: list[dict] = []
        self.reasserted: list[int] = []
        self.superseded: list[tuple[int, int]] = []
        self._next_id = 100

    def find_live_preferences(self, owner_id, group_id, embedding, limit=5):
        return self._candidates

    def insert_preference(self, **kw):
        self.inserted.append(kw)
        self._next_id += 1
        return self._next_id

    def reassert_preference(self, pref_id):
        self.reasserted.append(pref_id)

    def supersede_preference(self, old_id, new_id):
        self.superseded.append((old_id, new_id))


class _StubEmb:
    model_name = "voyage-4-large"

    def embed(self, texts, task):
        return [[0.0] * 4 for _ in texts]


def _run_flow(monkeypatch, prefs, candidates):
    import ingestion.preferences_gate as pg

    monkeypatch.setenv("SYNAPSE_PREFS_GATE", "1")
    monkeypatch.setattr(pg, "_group_for", lambda project: "technical")
    monkeypatch.setattr(pg, "parse_with_retry", lambda *a, **k: prefs)
    db = _RecDB(candidates)
    g = pg.PreferencesGate(db=db, llm_client=object(), embedder=_StubEmb())
    g.process({"id": 1, "episode_id": 5, "content": "x" * 500, "project": "synapse"})
    return db


def test_flow_insert_new(monkeypatch):
    db = _run_flow(
        monkeypatch, [{"pref": "User prefers dark mode", "polarity": "like"}], candidates=[]
    )
    assert len(db.inserted) == 1
    assert db.inserted[0]["pref"] == "User prefers dark mode"
    assert db.inserted[0]["polarity"] == "like"
    assert db.inserted[0]["source_ref"] == "ep:5"
    assert db.reasserted == [] and db.superseded == []


def test_flow_reassert(monkeypatch):
    db = _run_flow(
        monkeypatch,
        [{"pref": "User prefers dark mode", "polarity": "like"}],
        candidates=[{"id": 42, "polarity": "like", "sim": 0.95}],
    )
    assert db.reasserted == [42]
    assert db.inserted == [] and db.superseded == []


def test_flow_supersede_inserts_then_links(monkeypatch):
    db = _run_flow(
        monkeypatch,
        [{"pref": "User now prefers light mode", "polarity": "dislike"}],
        candidates=[{"id": 42, "polarity": "like", "sim": 0.84}],
    )
    assert len(db.inserted) == 1
    new_id = db.inserted[0] and db.superseded[0][1]
    assert db.superseded == [(42, new_id)]  # old retired, pointed at the freshly-inserted row
    assert db.reasserted == []


def test_flow_empty_gate_is_noop(monkeypatch):
    db = _run_flow(monkeypatch, [], candidates=[])
    assert db.inserted == [] and db.reasserted == [] and db.superseded == []
