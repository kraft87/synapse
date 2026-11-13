"""The compaction cut's second lane: the caller's earlier turns are added, never swapped in.

With ``self_session`` and a usable ``self_since``, what both tools serve from other
sessions is exactly what ``self_session`` alone serves: same turns, same order, same
scores. The caller's own turns from before its last compaction ride in the same rerank
call as the others but never enter their selection; at most
``_RECALL_SELF_PRIOR_LIMIT`` of them are then added, each only when it scores at least
as well as the weakest other turn served (ranks above it when the order is the fused
one), and the merged list is ordered by score.

Pure logic: every leg but the rerank is stubbed, and the reranker scores each document
by its text alone, so adding documents to a call never moves the others' scores.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

import mcp_server.recall as recall_mod
from ingestion.surfaces import FULL_TRUST
from mcp_server.recall import Recall

_NO_DB = "postgresql://synapse@127.0.0.1:1/nonexistent"  # an unstubbed leg fails loudly
_ME = "MY-SESSION"
_CUT = datetime(2026, 11, 12, 14, 0, tzinfo=UTC)
_SINCE = "2026-11-12T14:00:00Z"
_BEFORE = _CUT - timedelta(hours=2)
_AFTER = _CUT + timedelta(minutes=30)


class _Reranker:
    """Scores a document by its first word, so a call's other documents never move it.

    ``fail`` lists the 1-based calls that raise, to put one stage on the fused order."""

    def __init__(self, scores: dict[str, float], fail: tuple[int, ...] = ()) -> None:
        self.scores = scores
        self.fail = fail
        self.calls: list[list[str]] = []

    def rerank_scored(self, query, documents, top_k=None):
        self.calls.append(list(documents))
        if len(self.calls) in self.fail:
            raise RuntimeError("rerank backend down")
        out = [(i, self.scores[d.split()[0]]) for i, d in enumerate(documents)]
        return sorted(out, key=lambda x: x[1], reverse=True)


class _Embedder:
    def embed(self, texts, task="query"):
        return [[0.0] * 4 for _ in texts]


def _row(n: int, name: str, session: str, at: datetime) -> dict[str, Any]:
    return {
        "id": f"e:{n}",
        "content": f"{name} turn about the pool settings",
        "doc_type": "episode",
        "project": "synapse",
        "created_at": at,
        "session_id": session,
        "retrieval_count": 0,
    }


def _pool(spec: list[tuple[str, str, datetime]]) -> list[dict[str, Any]]:
    """Rows in fused order from (name, session, created_at)."""
    return [_row(i + 1, name, sess, at) for i, (name, sess, at) in enumerate(spec)]


# Crowding: six own turns from before the cut (more than the passage window and the
# limit hold) and one from after it, all ranked above every other turn, fused and scored.
_CROWD = _pool(
    [(f"p{i}", _ME, _BEFORE) for i in range(1, 7)]
    + [("a1", _ME, _AFTER)]
    + [(f"o{i}", f"S{i}", _BEFORE) for i in range(1, 7)]
)
_CROWD_SCORES = {
    **{f"p{i}": 0.96 - 0.01 * i for i in range(1, 7)},
    "a1": 0.99,
    **dict(zip([f"o{i}" for i in range(1, 7)], [0.80, 0.75, 0.70, 0.65, 0.30, 0.20], strict=True)),
}


@pytest.fixture(autouse=True)
def _knobs(monkeypatch):
    monkeypatch.setattr(recall_mod, "_RERANK_RECENCY", False)
    monkeypatch.setattr(recall_mod, "_RECALL_FLOOR", 0.0)
    monkeypatch.setattr(recall_mod, "_EPISODE_CUTOFF_TAU", 0.0)
    monkeypatch.setattr(recall_mod, "_RECALL_SELF_PRIOR_LIMIT", 2)
    monkeypatch.setattr(recall_mod, "_RECALL_SELF_PRIOR_CANDIDATES", 5)
    # A narrow passage window: three source turns, two passages.
    monkeypatch.setattr(recall_mod, "_RECALL_PASSAGE_SRC_K", 3)
    monkeypatch.setattr(recall_mod, "_RECALL_PASSAGE_N", 2)


def _engine(pool, reranker) -> tuple[Recall, list[dict], list[list[int]]]:
    """A Recall with every leg but the rerank stubbed: (engine, metric rows, overlay asks)."""
    r = Recall(_NO_DB, "")
    r._reranker = reranker
    r._ensure_embedder = lambda: _Embedder()
    r._search_bm25_episodes = lambda *a, **k: [dict(e) for e in pool]
    r._search_vector_episodes = lambda *a, **k: []
    r._episode_pool = lambda *a, **k: [dict(e) for e in pool]
    r._search_web_reranked = lambda *a, **k: []
    r._search_kg = lambda *a, **k: ([], [])
    r._search_notes = lambda *a, **k: []
    r._fetch_superseded_pairs_pg = lambda *a, **k: []
    r._surface_supersessions = lambda *a, **k: []
    overlay: list[list[int]] = []
    r._episode_supersessions = lambda ids, **k: overlay.append(list(ids)) or {}
    r._increment_fact_retrieval_counts = lambda *a, **k: None
    r._increment_retrieval_counts = lambda ids: None
    metrics: list[dict] = []
    r._record_metrics = metrics.append
    return r, metrics, overlay


def _overview(pool, scores, self_since=None, fail=()):
    r, metrics, overlay = _engine(pool, _Reranker(scores, fail))
    out = r.recall("q", self_session=_ME, self_since=self_since, trust=FULL_TRUST)
    return out.get("episodes", []), metrics[-1], overlay


def _turns(pool, scores, self_since=None, fail=()):
    reranker = _Reranker(scores, fail)
    r, metrics, _ = _engine(pool, reranker)
    out = r.recall_episodes("q", self_session=_ME, self_since=self_since, trust=FULL_TRUST)
    return out["episodes"], metrics[-1], reranker


def _others(items):
    return [it for it in items if it.get("session") != _ME]


def _own(items):
    return [it["id"] for it in items if it.get("session") == _ME]


def _ids(items):
    return [it["id"] for it in items]


_E = {name: f"e:{i + 1}" for i, name in enumerate([e["content"].split()[0] for e in _CROWD])}


# ---------------------------------------------------------------------------
# Rule 1: what is served from other sessions does not move
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fail", [(), (1,)], ids=["reranked", "episode-rerank-degraded"])
def test_overview_serves_the_same_others_with_the_cut(fail):
    base, base_m, base_overlay = _overview(_CROWD, _CROWD_SCORES, fail=fail)
    lane, lane_m, lane_overlay = _overview(_CROWD, _CROWD_SCORES, _SINCE, fail=fail)
    assert base and _own(base) == []
    assert _others(lane) == base
    assert lane_m["rerank_top_score"] == base_m["rerank_top_score"]
    assert lane_m["served_ids"]["n_echo_suppressed"] == base_m["served_ids"]["n_echo_suppressed"]
    assert lane_m["served_ids"]["n_dup_passages"] == base_m["served_ids"]["n_dup_passages"]
    # The overlay looks up the others exactly as before; the own turns get their own ask.
    assert lane_overlay[0] == base_overlay[0]
    assert lane_overlay[1:] == [[int(i.split(":")[1]) for i in _own(lane)]]


def test_overview_keeps_the_others_when_their_passages_fit(monkeypatch):
    # The others' passages fit without a rerank (three of four), the own ones would not:
    # the others still serve unscored, in the episode order (the fused one here).
    monkeypatch.setattr(recall_mod, "_RECALL_PASSAGE_SRC_K", 20)
    monkeypatch.setattr(recall_mod, "_RECALL_PASSAGE_N", 4)
    pool = _pool(
        [(f"p{i}", _ME, _BEFORE) for i in range(1, 4)]
        + [(f"o{i}", f"S{i}", _BEFORE) for i in range(1, 4)]
    )
    scores = {"p1": 0.9, "p2": 0.9, "p3": 0.9, "o1": 0.5, "o2": 0.7, "o3": 0.8}
    base, _, _ = _overview(pool, scores, fail=(1,))
    lane, _, _ = _overview(pool, scores, _SINCE, fail=(1,))
    assert _ids(base) == ["e:4", "e:5", "e:6"]
    assert _others(lane) == base
    assert _ids(lane) == ["e:1", "e:2", "e:4", "e:5", "e:6"]


@pytest.mark.parametrize(
    ("tau", "fail"),
    [(0.0, ()), (0.0, (1,)), (0.5, ())],
    ids=["reranked", "degraded", "score-cutoff"],
)
def test_full_turns_serve_the_same_others_with_the_cut(monkeypatch, tau, fail):
    monkeypatch.setattr(recall_mod, "_EPISODE_CUTOFF_TAU", tau)
    monkeypatch.setattr(recall_mod, "_EPISODE_CUTOFF_MIN_K", 1)
    base, base_m, _ = _turns(_CROWD, _CROWD_SCORES, fail=fail)
    lane, lane_m, _ = _turns(_CROWD, _CROWD_SCORES, _SINCE, fail=fail)
    assert base and _own(base) == []
    assert _others(lane) == base
    assert lane_m["rerank_top_score"] == base_m["rerank_top_score"]


def test_full_turns_cutoff_is_computed_from_the_others_alone(monkeypatch):
    # Others top 0.80: tau 0.5 keeps the four scoring >= 0.40, whatever the own turns score.
    monkeypatch.setattr(recall_mod, "_EPISODE_CUTOFF_TAU", 0.5)
    monkeypatch.setattr(recall_mod, "_EPISODE_CUTOFF_MIN_K", 1)
    lane, _, _ = _turns(_CROWD, _CROWD_SCORES, _SINCE)
    assert _ids(_others(lane)) == [_E["o1"], _E["o2"], _E["o3"], _E["o4"]]


def test_one_other_turn_reports_the_score_it_reports_alone():
    pool = _pool([("p1", _ME, _BEFORE), ("o1", "S1", _BEFORE)])
    scores = {"p1": 0.9, "o1": 0.5}
    _, base_m, _ = _turns(pool, scores)
    lane, lane_m, _ = _turns(pool, scores, _SINCE)
    assert _ids(lane) == ["e:1", "e:2"]
    assert lane_m["rerank_top_score"] == base_m["rerank_top_score"] == 1.0


# ---------------------------------------------------------------------------
# Rule 2: a second, small lane
# ---------------------------------------------------------------------------


def test_never_more_own_turns_than_the_limit():
    lane, _, _ = _overview(_CROWD, _CROWD_SCORES, _SINCE)
    assert _own(lane) == [_E["p1"], _E["p2"]]
    turns, _, _ = _turns(_CROWD, _CROWD_SCORES, _SINCE)
    assert _own(turns) == [_E["p1"], _E["p2"]]


def test_the_turns_after_the_cut_never_serve():
    lane, _, _ = _overview(_CROWD, _CROWD_SCORES, _SINCE)
    turns, _, _ = _turns(_CROWD, _CROWD_SCORES, _SINCE)
    assert _E["a1"] not in _ids(lane) + _ids(turns)


def test_only_the_first_candidates_in_fused_order_compete(monkeypatch):
    # p6 scores best of the own turns but sits sixth in the fused order, past the five.
    scores = {**_CROWD_SCORES, "p6": 0.98}
    turns, _, reranker = _turns(_CROWD, scores, _SINCE)
    assert _own(turns) == [_E["p1"], _E["p2"]]
    assert not any(d.startswith("p6") for d in reranker.calls[0])
    lane, _, _ = _overview(_CROWD, scores, _SINCE)
    assert _own(lane) == [_E["p1"], _E["p2"]]


_MIXED = _pool(
    [("p1", _ME, _BEFORE), ("p2", _ME, _BEFORE)]
    + [(f"o{i}", f"S{i}", _BEFORE) for i in range(1, 7)]
)


def test_an_own_turn_below_the_weakest_served_other_is_not_added():
    # Served others: o1..o5 (full turns, limit 5) down to 0.65; o1, o2 (passages) down to 0.75.
    scores = {"p1": 0.60, "p2": 0.74, "o1": 0.80, "o2": 0.75, "o3": 0.70, "o4": 0.66, "o5": 0.65}
    scores["o6"] = 0.10
    turns, _, _ = _turns(_MIXED, scores, _SINCE)
    assert _own(turns) == ["e:2"]
    lane, _, _ = _overview(_MIXED, scores, _SINCE)
    assert _own(lane) == []


def test_a_tie_with_the_weakest_served_other_is_added():
    scores = {"p1": 0.75, "p2": 0.65, "o1": 0.80, "o2": 0.75, "o3": 0.70, "o4": 0.66, "o5": 0.65}
    scores["o6"] = 0.10
    turns, _, _ = _turns(_MIXED, scores, _SINCE)
    assert _own(turns) == ["e:1", "e:2"]
    lane, _, _ = _overview(_MIXED, scores, _SINCE)
    assert _own(lane) == ["e:1"]


def test_degraded_full_turns_add_only_own_turns_ranked_above_the_last_served_other():
    # Fused order: o1, p1, o2..o5 (the five served), p2 after the last served other.
    pool = _pool(
        [("o1", "S1", _BEFORE), ("p1", _ME, _BEFORE)]
        + [(f"o{i}", f"S{i}", _BEFORE) for i in range(2, 6)]
        + [("p2", _ME, _BEFORE), ("o6", "S6", _BEFORE)]
    )
    scores = {"p1": 0.1, "p2": 0.99, **{f"o{i}": 0.5 for i in range(1, 7)}}
    turns, _, _ = _turns(pool, scores, _SINCE, fail=(1,))
    assert _ids(turns) == ["e:1", "e:2", "e:3", "e:4", "e:5", "e:6"]
    assert _own(turns) == ["e:2"]


def test_limit_zero_turns_the_lane_off(monkeypatch):
    monkeypatch.setattr(recall_mod, "_RECALL_SELF_PRIOR_LIMIT", 0)
    base, _, _ = _turns(_CROWD, _CROWD_SCORES)
    turns, turns_m, reranker = _turns(_CROWD, _CROWD_SCORES, _SINCE)
    assert turns == base
    assert turns_m["served_ids"]["n_self_served"] == 0
    assert not any(d.split()[0].startswith(("p", "a")) for d in reranker.calls[0])
    lane, _, _ = _overview(_CROWD, _CROWD_SCORES, _SINCE)
    assert _own(lane) == []


def test_with_no_other_turn_the_own_turns_serve_best_first():
    pool = _pool(
        [("a1", _ME, _AFTER), ("p1", _ME, _BEFORE), ("p2", _ME, _BEFORE), ("p3", _ME, _BEFORE)]
    )
    scores = {"a1": 0.99, "p1": 0.40, "p2": 0.70, "p3": 0.60}
    base, _, _ = _turns(pool, scores)
    turns, turns_m, _ = _turns(pool, scores, _SINCE)
    assert base == []
    assert _ids(turns) == ["e:3", "e:4"]
    assert turns_m["served_ids"]["n_self_served"] == 2
    lane, _, _ = _overview(pool, scores, _SINCE)
    assert _own(lane) and set(_own(lane)) <= {"e:3", "e:4"}


def test_with_no_other_turn_the_overview_floor_still_holds(monkeypatch):
    # The raw pool top (0.5, the caller's own turn after the cut) is under the floor:
    # no turn can be served, so no own turn is either.
    monkeypatch.setattr(recall_mod, "_RECALL_FLOOR", 0.58)
    pool = _pool([("a1", _ME, _AFTER), ("p1", _ME, _BEFORE)])
    lane, _, _ = _overview(pool, {"a1": 0.5, "p1": 0.4}, _SINCE)
    assert lane == []


def test_an_enforced_floor_adds_no_own_turn(monkeypatch):
    monkeypatch.setattr(recall_mod, "_RECALL_FLOOR", 0.58)
    monkeypatch.setattr(recall_mod, "_RECALL_FLOOR_KEEP_MIN", 1)
    scores = {**{k: v * 0.5 for k, v in _CROWD_SCORES.items()}}
    base, _, _ = _overview(_CROWD, scores)
    lane, _, _ = _overview(_CROWD, scores, _SINCE)
    assert len(base) == 1 and lane == base


def test_an_own_turn_that_quotes_the_query_is_not_added():
    query = "what did we settle about the halfvec index on the embeddings table"
    pool = _pool([("p1", _ME, _BEFORE), ("p2", _ME, _BEFORE), ("o1", "S1", _BEFORE)])
    pool[0]["content"] = f"p1 {query}"
    scores = {"p1": 0.9, "p2": 0.85, "o1": 0.5}
    r, _, _ = _engine(pool, _Reranker(scores))
    out = r.recall_episodes(query, self_session=_ME, self_since=_SINCE, trust=FULL_TRUST)
    assert _ids(out["episodes"]) == ["e:2", "e:3"]
    r, _, _ = _engine(pool, _Reranker(scores))
    out = r.recall(query, self_session=_ME, self_since=_SINCE, trust=FULL_TRUST)
    assert "e:1" not in _ids(out.get("episodes", []))


# ---------------------------------------------------------------------------
# Rule 3: the merged list is ordered by score
# ---------------------------------------------------------------------------


def test_own_turns_sit_where_their_score_puts_them():
    scores = {"p1": 0.78, "p2": 0.68, "o1": 0.80, "o2": 0.75, "o3": 0.70, "o4": 0.66, "o5": 0.65}
    scores["o6"] = 0.10
    turns, turns_m, _ = _turns(_MIXED, scores, _SINCE)
    assert _ids(turns) == ["e:3", "e:1", "e:4", "e:5", "e:2", "e:6", "e:7"]
    assert turns_m["served_ids"]["n_self_served"] == 2
    lane, lane_m, _ = _overview(_MIXED, scores, _SINCE)
    assert _ids(lane) == ["e:3", "e:1", "e:4"]
    assert lane_m["served_ids"]["n_self_served"] == 1


def test_passages_that_fit_without_a_rerank_keep_the_episode_order(monkeypatch):
    # Every passage fits (no passage rerank call): an own turn ranked above the last
    # other source serves at its rank, one ranked below it does not.
    monkeypatch.setattr(recall_mod, "_RECALL_PASSAGE_SRC_K", 20)
    monkeypatch.setattr(recall_mod, "_RECALL_PASSAGE_N", 6)
    pool = _pool(
        [
            ("p1", _ME, _BEFORE),
            ("o1", "S1", _BEFORE),
            ("o2", "S2", _BEFORE),
            ("p2", _ME, _BEFORE),
        ]
    )
    scores = {"p1": 0.6, "o1": 0.9, "o2": 0.5, "p2": 0.4}
    reranker = _Reranker(scores)
    r, _, _ = _engine(pool, reranker)
    out = r.recall("q", self_session=_ME, self_since=_SINCE, trust=FULL_TRUST)
    assert _ids(out["episodes"]) == ["e:2", "e:1", "e:3"]
    assert len(reranker.calls) == 1  # the episode rerank only


def test_own_turns_keep_room_when_the_others_fill_the_passage_cap(monkeypatch):
    # The others' passages fill the rerank input cap; the own ones are chunked under a cap
    # of their own, after it, so they can still be added.
    monkeypatch.setattr(recall_mod, "_RECALL_PASSAGE_CAND", 2)
    monkeypatch.setattr(recall_mod, "_RECALL_PASSAGE_N", 1)
    pool = _pool([(f"o{i}", f"S{i}", _BEFORE) for i in range(1, 4)] + [("p1", _ME, _BEFORE)])
    scores = {"o1": 0.8, "o2": 0.7, "o3": 0.6, "p1": 0.9}
    base, _, _ = _overview(pool, scores)
    lane, _, _ = _overview(pool, scores, _SINCE)
    assert _ids(base) == ["e:1"]
    assert _others(lane) == base
    assert _ids(lane) == ["e:4", "e:1"]


def test_the_lane_settings_are_read_live(monkeypatch):
    # Both tools take their snapshot per call, so a bench run can set them on the module.
    monkeypatch.setattr(recall_mod, "_RECALL_SELF_PRIOR_LIMIT", 7)
    monkeypatch.setattr(recall_mod, "_RECALL_SELF_PRIOR_CANDIDATES", 9)
    settings = Recall(_NO_DB, "")._settings()
    assert (settings._RECALL_SELF_PRIOR_LIMIT, settings._RECALL_SELF_PRIOR_CANDIDATES) == (7, 9)
