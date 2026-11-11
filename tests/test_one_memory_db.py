"""Recall reads one memory per owner: no read looks at a row's legacy ``group_id``.

Rows written before schema 067 carry ``technical`` or ``personal``; after it every row says
``technical``. Production holds both states during the rollout, so every test seeds both
values side by side and pins what no longer depends on them:

  * full trust: the vector, BM25 and hop legs each serve both values, in ONE ranked list that
    ends at the fact limit, with no extras after it and no cross-group marker in the telemetry
  * one fact-floor request over that list, and the retrieval-count bump reaches both values
  * a superseded pair, the supersession surface and the episode overlay link rows across
    the two values
  * a restricted device is served no ``personal``-audience fact from any search leg, whichever
    value the row carries: vector, BM25, hop, superseded pairs, surface, overlay, and no
    seed entities
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

import mcp_server.recall as recall_mod
from ingestion.surfaces import FULL_TRUST, SurfaceTrust
from mcp_server import kg_pg
from mcp_server.recall import Recall
from tests.helpers.embed import onehot

WORK = "work-proj"
TECH, PERS = "technical", "personal"
LEGACY = (TECH, PERS)
_RESTRICTED = SurfaceTrust(trust="restricted", allowed_projects=(WORK,), known=True)
_INVALID = "2026-02-01T00:00:00+00:00"


@pytest.fixture()
def kg(conn):
    def _wipe() -> None:
        conn.execute("TRUNCATE kg_relationships, kg_entities RESTART IDENTITY CASCADE")
        conn.execute("TRUNCATE episodes RESTART IDENTITY CASCADE")

    _wipe()
    yield conn
    _wipe()


def _ep(conn, project: str = WORK) -> int:
    return conn.execute(
        "INSERT INTO episodes (session_id, sequence, project, content) "
        "VALUES (%s, 1, %s, 'a turn') RETURNING id",
        (f"one-mem-{uuid.uuid4().hex[:10]}", project),
    ).fetchone()[0]


def _fact(
    conn,
    uid: str,
    group: str,
    fact: str,
    ep: int,
    *,
    emb: int | None = None,
    src: str = "ent-x",
    tgt: str = "ent-y",
    audience: str | None = "work-safe",
    t_invalid: str | None = None,
    invalidated_by: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO kg_relationships (uuid, owner_id, group_id, src_uuid, tgt_uuid, name, "
        "  fact, fact_embedding, episodes, t_valid, t_invalid, invalidated_by, audience) "
        "VALUES (%s, 'default', %s, %s, %s, 'RELATES', %s, %s::vector, %s::jsonb, "
        "        '2026-01-01T00:00:00+00:00', %s, %s, %s)",
        (
            uid,
            group,
            src,
            tgt,
            fact,
            None if emb is None else kg_pg._vec_literal(onehot(emb)),
            f"[{ep}]",
            t_invalid,
            invalidated_by,
            audience,
        ),
    )


def _entity(conn, uid: str, group: str, name: str, emb: int) -> None:
    conn.execute(
        "INSERT INTO kg_entities (uuid, owner_id, group_id, name, normalized_name, summary, "
        "  embedding) VALUES (%s, 'default', %s, %s, %s, 's', %s::vector)",
        (uid, group, name, name.lower(), kg_pg._vec_literal(onehot(emb))),
    )


class _Emb:
    def embed(self, texts, task):
        return [onehot(0) for _ in texts]


class _Now:
    """An executor that runs a submitted job before submit returns."""

    def submit(self, fn, *a, **k):
        fn(*a, **k)


class _Rerank:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[list[str]] = []

    def rerank_scored(self, query, docs):
        self.calls.append(list(docs))
        ranked = sorted(enumerate(docs), key=lambda x: -self.scores[x[1]])
        return [(i, self.scores[d]) for i, d in ranked]


def _engine(db_url: str, *, bump: bool = False, metrics: list | None = None) -> Recall:
    """Real KG legs; anything that would reach a network or rank episodes is stubbed."""
    r = Recall(db_url, "")
    r._ensure_embedder = lambda: _Emb()
    r._rerank_pool_scored = lambda q, pool: [(i, 0.9) for i in range(len(pool))]
    r._compact_to_passages = lambda q, eps, n, **_k: [{"id": e["id"], "text": "t"} for e in eps]
    r._search_web_reranked = lambda q, emb, **_k: []
    r._search_notes = lambda *a, **k: []
    r._increment_retrieval_counts = lambda ids: None
    r._record_metrics = (lambda m: None) if metrics is None else metrics.append
    if bump:
        r._async_executor = _Now()
    else:
        r._increment_fact_retrieval_counts = lambda *a, **k: None
    return r


def _served(out: dict[str, Any], bucket: str = "facts") -> set[str]:
    return {f["id"].removeprefix("f:") for f in out.get(bucket) or []}


def _uuids(facts: list[dict[str, Any]]) -> set[str]:
    return {f["_uuid"] for f in facts}


def _seed_legs(conn) -> None:
    """In each legacy value, per leg only that leg can serve, a work-safe fact and a
    personal-audience one: ``v`` by vector (its text misses the query), ``b`` by BM25 (no
    embedding), ``h`` by the hop from the value's seed entity (neither). Every fact sits
    inside the allowlist, so for a restricted caller only its label decides."""
    ep = _ep(conn)
    for g in LEGACY:
        _entity(conn, f"seed-{g}", g, f"Seed {g}", 0)
        for kind, audience in (("safe", "work-safe"), ("personal", "personal")):
            _fact(conn, f"v-{kind}-{g}", g, f"alpha {kind} {g}", ep, emb=0, audience=audience)
            _fact(conn, f"b-{kind}-{g}", g, f"widget {kind} {g}", ep, audience=audience)
            _fact(
                conn,
                f"h-{kind}-{g}",
                g,
                f"epsilon {kind} {g}",
                ep,
                src=f"seed-{g}",
                audience=audience,
            )


_SAFE = {f"{leg}-safe-{g}" for leg in "vbh" for g in LEGACY}
_EVERY = _SAFE | {f"{leg}-personal-{g}" for leg in "vbh" for g in LEGACY}


# ---------------------------------------------------------------------------
# Full trust: one ranked list
# ---------------------------------------------------------------------------


def test_full_trust_serves_both_legacy_values_on_every_leg(kg, db_url):
    _seed_legs(kg)
    metrics: list[dict[str, Any]] = []
    out = _engine(db_url, metrics=metrics).recall("widget", trust=FULL_TRUST)
    assert _served(out) == _EVERY
    served_ids = metrics[0]["served_ids"]
    assert "facts_cross_group" not in served_ids
    assert sorted(served_ids["facts"]) == sorted(_EVERY)
    assert "group_id" not in metrics[0]


def test_the_facts_are_ranked_across_both_values_not_one_value_then_the_other(kg, db_url):
    """The best matches are personal-pile facts (found by vector AND BM25) and the technical
    pile has many weaker BM25-only ones: a ranking per pile would put the technical facts
    first and append a few personal ones after the limit."""
    ep = _ep(kg)
    for i in range(3):
        _fact(kg, f"p{i}", PERS, f"widget {i}", ep, emb=0)
    for i in range(12):
        _fact(kg, f"t{i:02d}", TECH, "widget " + "filler " * (i + 4), ep)
    facts, _ = Recall(db_url, "")._search_kg("widget", onehot(0), [], 5)
    assert len(facts) == 5
    assert _uuids(facts[:3]) == {"p0", "p1", "p2"}
    assert not any("_cross" in f for f in facts)


def test_recall_serves_the_fact_limit_over_both_values_and_no_extras(kg, db_url):
    ep = _ep(kg)
    for i in range(9):
        _fact(kg, f"t{i}", TECH, f"widget tech {i}", ep, emb=0)
        _fact(kg, f"p{i}", PERS, f"widget personal {i}", ep, emb=0)
    metrics: list[dict[str, Any]] = []
    out = _engine(db_url, metrics=metrics).recall("widget", trust=FULL_TRUST)
    limit = recall_mod._FACT_LIMIT
    assert len(out["facts"]) == limit
    assert len(metrics[0]["served_ids"]["facts"]) == limit
    # both values are in it, not one value's twelve with the other's after
    assert {u[0] for u in _served(out)} == {"t", "p"}


def test_a_recall_makes_one_fact_floor_request_over_both_values(kg, db_url, monkeypatch):
    monkeypatch.setattr(recall_mod, "_RECALL_FACT_FLOOR", 0.5)
    ep = _ep(kg)
    _entity(kg, "seed-t", TECH, "Seed Tech", 0)
    _fact(kg, "v-t", TECH, "alpha tech", ep, emb=0)
    _fact(kg, "b-t", TECH, "widget tech", ep)
    _fact(kg, "h-t", TECH, "epsilon tech", ep, src="seed-t")
    _fact(kg, "v-p", PERS, "alpha personal", ep, emb=0)
    _fact(kg, "b-p", PERS, "widget personal", ep)
    r = _engine(db_url)
    scores = {"alpha tech": 0.9, "widget tech": 0.9, "epsilon tech": 0.9}
    r._reranker = _Rerank(scores | {"alpha personal": 0.9, "widget personal": 0.1})
    out = r.recall("widget", trust=FULL_TRUST)
    assert len(r._reranker.calls) == 1
    assert sorted(r._reranker.calls[0]) == sorted(
        ["alpha tech", "widget tech", "epsilon tech", "alpha personal", "widget personal"]
    )
    assert _served(out) == {"v-t", "b-t", "h-t", "v-p"}, "the sub-floor personal-pile fact drops"


def test_the_retrieval_count_bump_reaches_both_values(kg, db_url):
    _seed_legs(kg)
    out = _engine(db_url, bump=True).recall("widget", trust=FULL_TRUST)
    assert _served(out) == _EVERY
    counts = dict(kg.execute("SELECT uuid, retrieval_count FROM kg_relationships").fetchall())
    assert {u: counts[u] for u in _EVERY} == dict.fromkeys(_EVERY, 1)


# ---------------------------------------------------------------------------
# Superseded pairs, the supersession surface and the overlay link across values
# ---------------------------------------------------------------------------


def _supersession(
    conn, now_group: str, old_group: str, *, now_audience: str | None, old_audience: str | None
) -> int:
    """A live fact and the retired one it superseded, in the given legacy values; returns
    the episode that cites the retired one. The live fact has no embedding and its text
    misses "widget", so a search for "widget" reaches it only through the retired one."""
    ep = _ep(conn)
    _fact(
        conn,
        "now",
        now_group,
        "gadget runs on postgres",
        ep,
        src="s",
        tgt="t",
        audience=now_audience,
    )
    _fact(
        conn,
        "old",
        old_group,
        "widget runs on sqlite",
        ep,
        emb=0,
        src="s",
        tgt="t",
        t_invalid=_INVALID,
        invalidated_by="now",
        audience=old_audience,
    )
    return ep


def _extras(r: Recall, ep: int, **scope: Any) -> tuple[list[str], set[str], dict[int, list[str]]]:
    pairs = r._fetch_superseded_pairs_pg(["now"], 5, **scope)
    surface = r._surface_supersessions(onehot(0), set(), **scope)
    overlay = r._episode_supersessions([ep], **scope)
    return [p["id"] for p in pairs], _uuids(surface), overlay


@pytest.mark.parametrize(("now_group", "old_group"), [(a, b) for a in LEGACY for b in LEGACY])
def test_full_trust_pairs_surface_and_overlay_ignore_the_legacy_value(
    kg, db_url, now_group, old_group
):
    ep = _supersession(kg, now_group, old_group, now_audience="personal", old_audience="personal")
    pairs, surface, overlay = _extras(Recall(db_url, ""), ep)
    assert pairs == ["f:old"]
    assert surface == {"now"}
    assert overlay == {ep: ["gadget runs on postgres"]}


# ---------------------------------------------------------------------------
# A restricted device is served no personal-audience fact, from any leg
# ---------------------------------------------------------------------------


def test_restricted_search_serves_no_personal_fact_from_either_value_on_any_leg(kg, db_url):
    _seed_legs(kg)
    facts, seeds = Recall(db_url, "")._search_kg(
        "widget", onehot(0), [], 40, allowed_projects=[WORK]
    )
    assert _uuids(facts) == _SAFE, "vector, BM25 and hop: the work-safe facts of both values only"
    assert seeds == [], "no seed entity: its name and summary come from all its facts"
    # the same rows are all there for full trust, so the filter is what removed them
    full, full_seeds = Recall(db_url, "")._search_kg("widget", onehot(0), [], 40)
    assert _uuids(full) == _EVERY
    assert {s["uuid"] for s in full_seeds} == {f"seed-{g}" for g in LEGACY}


def test_a_restricted_recall_serves_no_personal_fact_from_either_value(kg, db_url):
    _seed_legs(kg)
    r = _engine(db_url)
    assert _served(r.recall("widget", trust=_RESTRICTED)) == _SAFE


@pytest.mark.parametrize(("now_group", "old_group"), [(TECH, PERS), (PERS, TECH)])
@pytest.mark.parametrize(
    ("now_audience", "old_audience", "served"),
    [
        ("work-safe", "work-safe", True),
        ("work-safe", "personal", False),
        ("personal", "work-safe", False),
        ("personal", "personal", False),
    ],
)
def test_a_restricted_pair_surface_and_overlay_need_both_sides_work_safe(
    kg, db_url, now_group, old_group, now_audience, old_audience, served
):
    ep = _supersession(
        kg, now_group, old_group, now_audience=now_audience, old_audience=old_audience
    )
    pairs, surface, overlay = _extras(Recall(db_url, ""), ep, allowed_projects=[WORK])
    if served:
        assert pairs == ["f:old"]
        assert surface == {"now"}
        assert overlay == {ep: ["gadget runs on postgres"]}
    else:
        assert (pairs, surface, overlay) == ([], set(), {})


@pytest.mark.parametrize(("old_audience", "served"), [("work-safe", {"now"}), ("personal", set())])
def test_a_restricted_recall_is_pulled_no_successor_of_a_personal_fact(
    kg, db_url, old_audience, served
):
    """End to end: the successor is reachable only through the supersession surface, and the
    surface needs the retired fact to pass the label rule too. Both rows are inside the
    allowlist and in different legacy values, so only the label decides."""
    _supersession(kg, TECH, PERS, now_audience="work-safe", old_audience=old_audience)
    assert _served(_engine(db_url).recall("widget", trust=_RESTRICTED)) == served
    assert _served(_engine(db_url).recall("widget", trust=FULL_TRUST)) == {"now"}
