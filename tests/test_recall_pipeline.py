"""recall()'s fan-out and the ranking contract it shares with recall_full_turns.

Pure-logic: every leg is stubbed (no DB, no Voyage). What is pinned is the ORDER work
happens in, because that is what latency and relevance ride on:

  * the episode rerank starts as soon as the episode pool exists and does not wait for the
    web and KG legs, which it does not depend on;
  * the supersession surface is a wave-1 leg beside the KG, and the calling session's
    facts are looked up once for facts and successors together;
  * the calling session's turns and facts leave BEFORE the rerank and the fact cut, so
    their slots backfill instead of shrinking the buckets;
  * the abstention floor judges the top score of what survives echo suppression;
  * recall_full_turns starts BM25 before the query embedding, as recall() does;
  * interactive recall gives up on a Voyage 429 within seconds.
"""

from __future__ import annotations

import threading
from contextlib import nullcontext
from typing import Any

import pytest

import mcp_server.recall as recall_mod
from ingestion.surfaces import FULL_TRUST
from mcp_server.recall import Recall
from mcp_server.recall_config import RecallConfig
from mcp_server.recall_sources import KG_SCOPE_PROBE

# Poison DSN: an unstubbed leg fails fast and everywhere instead of finding a local DB.
_NO_DB = "postgresql://synapse@127.0.0.1:1/nonexistent"
_WAIT_S = 3.0


class _Emb:
    def embed(self, texts, task="query"):
        return [[0.0] * 4 for _ in texts]


@pytest.fixture(autouse=True)
def _fresh_kg_scope_probe():
    KG_SCOPE_PROBE.reset()
    yield
    KG_SCOPE_PROBE.reset()


def _pool(n: int, session: str | None = None, start: int = 0) -> list[dict[str, Any]]:
    return [
        {
            "id": f"e:{i}",
            "content": f"pool doc {i}",
            "doc_type": "episode",
            "project": None,
            "created_at": None,
            "retrieval_count": 0,
            "session_id": session if session is not None else f"s{i}",
        }
        for i in range(start, start + n)
    ]


def _fact(uuid: str, fact_id: int) -> dict[str, Any]:
    return {"fact": f"fact {uuid}", "_uuid": uuid, "_id": fact_id, "_date": None}


def _engine(
    pool: list[dict[str, Any]] | None = None,
    scores: list[float] | None = None,
    floor: float = 0.0,
):
    """A Recall with every leg stubbed; returns (recall, captured telemetry rows)."""
    r = Recall(_NO_DB, "", config=RecallConfig(recall_floor=floor))
    p = _pool(4) if pool is None else pool
    r._ensure_embedder = lambda: _Emb()
    r._search_bm25_episodes = lambda q, proj, limit, sid=None, allowed=None, own=None: list(p)
    r._search_vector_episodes = lambda emb, proj, limit, sid=None, allowed=None, own=None: []
    r._search_web_reranked = lambda q, emb, **_k: []
    r._search_kg = lambda *a, **k: ([], [])
    r._search_notes = lambda q, emb, proj, audience=None: []
    r._fetch_superseded_pairs_pg = lambda gid, uuids, cap: []
    r._surface_supersessions = lambda *a, **k: []
    r._episode_supersessions = lambda *a, **k: {}
    r._compact_to_passages = lambda q, eps, n, **_k: [
        {"id": e["id"], "content": f"passage {e['id']}"} for e in eps[:n]
    ]
    r._increment_fact_retrieval_counts = lambda *a, **k: None
    r._increment_retrieval_counts = lambda ids: None
    r._rerank_pool_scored = lambda q, pl: [
        (i, scores[i] if scores else 0.9 - 0.01 * i) for i in range(len(pl))
    ]
    captured: list[dict[str, Any]] = []
    r._record_metrics = captured.append
    return r, captured


# --- recall-03: the rerank does not wait for the web and KG legs --------------------


def test_episode_rerank_starts_before_the_web_and_kg_legs_finish():
    rerank_started = threading.Event()
    saw: dict[str, bool] = {}
    r, _ = _engine()

    def _web(q, emb):
        saw["web"] = rerank_started.wait(_WAIT_S)
        return []

    def _kg(*a, **k):
        saw["kg"] = rerank_started.wait(_WAIT_S)
        return [], []

    def _rerank(q, pl):
        rerank_started.set()
        return [(i, 0.9 - 0.01 * i) for i in range(len(pl))]

    r._search_web_reranked = _web
    r._search_kg = _kg
    r._rerank_pool_scored = _rerank
    r.recall("where is the gizmo", trust=FULL_TRUST)
    assert saw == {"web": True, "kg": True}


def test_supersession_surface_runs_beside_the_kg_leg():
    sup_started = threading.Event()
    saw: dict[str, bool] = {}
    r, _ = _engine()

    def _kg(*a, **k):
        saw["kg"] = sup_started.wait(_WAIT_S)
        return [], []

    def _sup(*a, **k):
        sup_started.set()
        return []

    r._search_kg = _kg
    r._surface_supersessions = _sup
    r.recall("where is the gizmo", trust=FULL_TRUST)
    assert saw == {"kg": True}


def test_supersession_successors_are_deduped_against_served_facts_and_capped():
    r, _ = _engine()
    r._search_kg = lambda *a, **k: ([_fact("served-1", 1)], [])
    r._surface_supersessions = lambda *a, **k: [
        _fact(u, i) for i, u in enumerate(("served-1", "n-1", "n-2", "n-3", "n-4"), start=1)
    ]
    out = r.recall("where is the gizmo", trust=FULL_TRUST)
    assert [f["id"] for f in out["facts"]] == ["f:1", "f:2", "f:3", "f:4"]


# --- recall-11: self-session turns and facts leave before the cut --------------------


def test_self_session_turns_leave_the_pool_before_the_rerank():
    # 10 of the calling session's turns rank first among 115 fused candidates. They are
    # dropped before the 100-deep cut, so the rerank sees 100 servable turns and no slot
    # is spent on a turn that is always dropped.
    pool = _pool(10, session="me") + _pool(105, start=10)
    seen: list[list[str]] = []
    r, captured = _engine(pool)

    def _rerank(q, pl):
        seen.append([e["id"] for e in pl])
        return [(i, 0.9 - 0.001 * i) for i in range(len(pl))]

    r._rerank_pool_scored = _rerank
    out = r.recall("where is the gizmo", trust=FULL_TRUST, self_session="me")
    assert len(seen[0]) == 100
    assert len(seen) == 1
    assert not {f"e:{i}" for i in range(10)} & set(seen[0])
    assert "e:10" in [it["id"] for it in out["episodes"]]
    assert captured[0]["served_ids"]["n_self_excluded"] == 10


def test_self_session_facts_backfill_the_facts_bucket():
    r, captured = _engine()
    limits: list[int] = []
    facts = [
        {"fact": f"fact {i}", "_uuid": f"own-{i}" if i < 3 else f"u{i}", "_id": 100 + i}
        for i in range(20)
    ]

    def _kg(q, emb, gid, focus, fact_limit, **kw):
        limits.append(fact_limit)
        return [dict(f) for f in facts[:fact_limit]], []

    r._search_kg = _kg
    r._self_session_facts = lambda uuids, session: {u for u in uuids if u.startswith("own-")}
    out = r.recall("where is the gizmo", trust=FULL_TRUST, self_session="me")
    assert limits[0] > r._config.fact_limit  # asked for a few extra
    assert len(out["facts"]) == r._config.fact_limit  # and the bucket is still full
    assert not {"f:100", "f:101", "f:102"} & {f["id"] for f in out["facts"]}
    assert captured[0]["served_ids"]["n_self_facts_excluded"] == 3

    limits.clear()
    r.recall("where is the gizmo", trust=FULL_TRUST)  # no hook, no headroom
    assert limits == [r._config.fact_limit]


def test_facts_and_successors_share_one_provenance_lookup():
    r, _ = _engine()
    lookups: list[list[str]] = []
    r._search_kg = lambda *a, **k: ([{"fact": "a", "_uuid": "u1"}], [])
    r._surface_supersessions = lambda *a, **k: [{"fact": "b", "_uuid": "u2"}]

    def _own(uuids, session):
        lookups.append(list(uuids))
        return set()

    r._self_session_facts = _own
    r.recall("where is the gizmo", trust=FULL_TRUST, self_session="me")
    assert lookups == [["u1", "u2"]]


# --- recall-12: the floor judges what survives echo suppression ----------------------

_ECHO_QUERY = "what is the exact configuration of the gizmo in the staging cluster"


def test_a_query_echo_cannot_hold_the_abstention_floor_open():
    pool = _pool(4)
    pool[0]["content"] = _ECHO_QUERY  # the prompt quoting itself
    r, captured = _engine(pool, scores=[0.95, 0.40, 0.35, 0.30], floor=0.58)
    out = r.recall(_ECHO_QUERY, trust=FULL_TRUST)
    assert "episodes" not in out  # every survivor is weak: the bucket abstains
    served = captured[0]["served_ids"]
    assert captured[0]["rerank_top_score"] == 0.4
    assert served["rerank_top_all"] == 0.95  # kept for recalibrating the floor
    assert served["would_abstain"] is True and served["n_echo_suppressed"] == 1


def test_the_floor_is_unchanged_when_nothing_is_filtered():
    r, captured = _engine(scores=[0.95, 0.40, 0.35, 0.30], floor=0.58)
    out = r.recall("where is the gizmo", trust=FULL_TRUST)
    assert out["episodes"]
    assert captured[0]["rerank_top_score"] == 0.95
    assert "rerank_top_all" not in captured[0]["served_ids"]


def test_select_episodes_reports_the_top_of_the_survivors():
    pool = _pool(4)
    pool[0]["content"] = _ECHO_QUERY
    r, _ = _engine(pool, scores=[0.95, 0.40, 0.35, 0.30])
    served, n_echo, top = r._select_episodes(_ECHO_QUERY, pool, limit=2)
    assert [e["id"] for e in served] == ["e:1", "e:2"]
    assert n_echo == 1 and top == 0.4


# --- recall-18: recall_full_turns overlaps BM25 with the embedding -------------------


def test_recall_full_turns_starts_bm25_before_the_query_embedding():
    bm25_started = threading.Event()
    saw: dict[str, bool] = {}
    r, _ = _engine()

    class _SlowEmb:
        def embed(self, texts, task="query"):
            saw["bm25_first"] = bm25_started.wait(_WAIT_S)
            return [[0.0] * 4 for _ in texts]

    def _bm25(q, proj, limit, sid=None, allowed=None, own=None):
        bm25_started.set()
        return _pool(4)

    r._ensure_embedder = lambda: _SlowEmb()
    r._search_bm25_episodes = _bm25
    out = r.recall_episodes("where is the gizmo", trust=FULL_TRUST)
    assert saw == {"bm25_first": True}
    assert len(out["episodes"]) == 4


def test_recall_full_turns_backfills_past_the_callers_own_turns():
    pool = _pool(3, session="me") + _pool(3, start=3)
    r, captured = _engine(pool)
    out = r.recall_episodes("where is the gizmo", trust=FULL_TRUST, self_session="me", limit=3)
    assert [it["id"] for it in out["episodes"]] == ["e:3", "e:4", "e:5"]
    assert captured[0]["served_ids"]["n_self_excluded"] == 3


def test_recall_full_turns_stays_inside_the_per_call_budget():
    from mcp_server.recall_presentation import FULL_TURN_BUDGET

    pool = _pool(6)
    for ep in pool:
        ep["content"] = f"turn {ep['id']} " + "tool trace line " * 3_000  # ~48K chars each
    r, captured = _engine(pool)
    out = r.recall_episodes("where is the gizmo", trust=FULL_TRUST, limit=6)
    served = out["episodes"]
    assert [it["id"] for it in served] == [f"e:{i}" for i in range(6)]  # same turns, same order
    assert all(it["full_chars"] > 40_000 for it in served)  # each alone passes the budget
    assert sum(len(it["content"]) for it in served) <= 6 * 3_100
    assert sum(len(it["content"]) for it in served) < FULL_TURN_BUDGET
    assert captured[0]["served_ids"]["n_excerpted"] == 6
    query = "where is the gizmo"
    assert captured[0]["chars"] == len(query) + sum(len(it["content"]) for it in served)


def test_recall_full_turns_serves_ordinary_turns_whole():
    r, captured = _engine(_pool(5))
    out = r.recall_episodes("where is the gizmo", trust=FULL_TRUST)
    assert [it["content"] for it in out["episodes"]] == [f"pool doc {i}" for i in range(5)]
    assert all("full_chars" not in it for it in out["episodes"])
    assert "n_excerpted" not in captured[0]["served_ids"]


def test_fetch_reports_fact_ids_as_skipped_in_either_form():
    """Facts are rateable, not fetchable: "f:<id>" and the older "f:<uuid>" are told apart
    from nothing else, and both come back under skipped."""
    from mcp_server.recall_presentation import parse_fetch_ids

    old_style = "f:0b3c1d2e-4f56-7890-abcd-ef0123456789"
    episodes, notes, skipped, _normalized = parse_fetch_ids(["e:5", "f:41", old_style, "n:2"])
    assert (episodes, notes) == ([5], [2])
    assert skipped == ["f:41", old_style]


# --- recall-19: interactive recall does not ride out a Voyage 429 --------------------


def test_recall_builds_interactive_embedder_and_reranker(monkeypatch):
    built: dict[str, dict[str, Any]] = {}

    def _create(kind: str):
        def _factory(**kwargs: Any) -> object:
            built[kind] = kwargs
            return object()

        return _factory

    monkeypatch.setattr(recall_mod._embedding, "create_embedder", _create("embedder"))
    monkeypatch.setattr(recall_mod._embedding, "create_reranker", _create("reranker"))
    r = Recall(_NO_DB, "key")
    r._ensure_embedder()
    r._ensure_reranker()
    assert built["embedder"]["interactive"] is True
    assert built["reranker"]["interactive"] is True


# --- the schema-056 probe, with the KG-derived legs now running side by side ----------


def test_a_leg_arriving_mid_probe_waits_for_the_answer():
    # The KG leg and the supersession surface both gate on the probe. The second must not
    # read "probed a moment ago" as "column absent" and serve a restricted caller nothing.
    inside_probe, release = threading.Event(), threading.Event()
    probes: list[int] = []

    class _Pg:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def transaction(self):
            return nullcontext()

        def execute(self, sql):
            probes.append(1)
            inside_probe.set()
            release.wait(_WAIT_S)
            return self

        def fetchone(self):
            return (1,)

    r = Recall(_NO_DB, "")
    r._pg = lambda: _Pg()
    answers: list[bool] = []
    first = threading.Thread(target=lambda: answers.append(r._kg_scope_ready()))
    first.start()
    assert inside_probe.wait(_WAIT_S)
    second = threading.Thread(target=lambda: answers.append(r._kg_scope_ready()))
    second.start()
    release.set()
    first.join(_WAIT_S)
    second.join(_WAIT_S)
    assert answers == [True, True] and len(probes) == 1


def test_select_episodes_serves_the_top_limit_and_degrades_to_rrf_order():
    pool = _pool(4)
    r, _ = _engine(pool, scores=[0.0, 0.0, 0.0, 0.0])  # the rerank degraded: no real scores
    served, n_echo, top = r._select_episodes("where is the gizmo", pool, limit=2)
    assert [e["id"] for e in served] == ["e:0", "e:1"] and n_echo == 0 and top == 0.0
    assert r._select_episodes("where is the gizmo", [], limit=2) == ([], 0, 0.0)
