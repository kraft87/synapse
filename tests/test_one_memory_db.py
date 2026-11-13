"""Recall reads one memory per owner, as one ranked list.

* full trust: the vector, BM25 and hop legs each serve, in ONE ranked list that ends at the
  fact limit, with no extras after it and no cross-group marker in the telemetry
* one fact-floor request over that list, and the retrieval-count bump reaches every fact
  it served
* a superseded pair, the supersession surface and the episode overlay link a live fact to
  the one it retired
* a restricted device is served no ``personal``-audience fact from any search leg: vector,
  BM25, hop, superseded pairs, surface, overlay, and no seed entities
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

import mcp_server.recall as recall_mod
from ingestion.surfaces import FULL_TRUST, SurfaceTrust
from mcp_server import kg_pg
from mcp_server.recall import Recall
from mcp_server.recall_settings import _SUP_LIMIT
from tests.helpers.embed import onehot

WORK = "work-proj"
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
        "INSERT INTO kg_relationships (uuid, owner_id, src_uuid, tgt_uuid, name, "
        "  fact, fact_embedding, episodes, t_valid, t_invalid, invalidated_by, audience) "
        "VALUES (%s, 'default', %s, %s, 'RELATES', %s, %s::vector, %s::jsonb, "
        "        '2026-01-01T00:00:00+00:00', %s, %s, %s)",
        (
            uid,
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


def _entity(conn, uid: str, name: str, emb: int) -> None:
    conn.execute(
        "INSERT INTO kg_entities (uuid, owner_id, name, normalized_name, summary, embedding) "
        "VALUES (%s, 'default', %s, %s, 's', %s::vector)",
        (uid, name, name.lower(), kg_pg._vec_literal(onehot(emb))),
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
    """Per leg only that leg can serve, a work-safe fact and a personal-audience one: ``v`` by
    vector (its text misses the query), ``b`` by BM25 (no embedding), ``h`` by the hop from the
    seed entity (neither). Every fact sits inside the allowlist, so for a restricted caller only
    its label decides."""
    ep = _ep(conn)
    _entity(conn, "seed", "Seed", 0)
    for kind, audience in (("safe", "work-safe"), ("personal", "personal")):
        _fact(conn, f"v-{kind}", f"alpha {kind}", ep, emb=0, audience=audience)
        _fact(conn, f"b-{kind}", f"widget {kind}", ep, audience=audience)
        _fact(conn, f"h-{kind}", f"epsilon {kind}", ep, src="seed", audience=audience)


_SAFE = {f"{leg}-safe" for leg in "vbh"}
_EVERY = _SAFE | {f"{leg}-personal" for leg in "vbh"}


# ---------------------------------------------------------------------------
# Full trust: one ranked list
# ---------------------------------------------------------------------------


def test_full_trust_serves_every_leg_in_one_list(kg, db_url):
    _seed_legs(kg)
    metrics: list[dict[str, Any]] = []
    out = _engine(db_url, metrics=metrics).recall("widget", trust=FULL_TRUST)
    assert _served(out) == _EVERY
    served_ids = metrics[0]["served_ids"]
    assert "facts_cross_group" not in served_ids
    assert sorted(served_ids["facts"]) == sorted(_EVERY)
    assert "group_id" not in metrics[0]


def test_the_best_matches_lead_the_one_ranked_list(kg, db_url):
    """Three facts are found by vector AND BM25, twelve weaker ones by BM25 only: the three
    lead the list, and nothing is appended after the limit."""
    ep = _ep(kg)
    for i in range(3):
        _fact(kg, f"p{i}", f"widget {i}", ep, emb=0)
    for i in range(12):
        _fact(kg, f"t{i:02d}", "widget " + "filler " * (i + 4), ep)
    facts, _ = Recall(db_url, "")._search_kg("widget", onehot(0), [], 5)
    assert len(facts) == 5
    assert _uuids(facts[:3]) == {"p0", "p1", "p2"}
    assert not any("_cross" in f for f in facts)


def test_recall_serves_the_fact_limit_and_no_extras(kg, db_url):
    ep = _ep(kg)
    limit = recall_mod._FACT_LIMIT
    for i in range(limit * 3 // 2):
        _fact(kg, f"f{i}", f"widget {i}", ep, emb=0)
    metrics: list[dict[str, Any]] = []
    out = _engine(db_url, metrics=metrics).recall("widget", trust=FULL_TRUST)
    assert len(out["facts"]) == limit
    assert len(metrics[0]["served_ids"]["facts"]) == limit


def test_a_recall_makes_one_fact_floor_request_over_every_leg(kg, db_url, monkeypatch):
    monkeypatch.setattr(recall_mod, "_RECALL_FACT_FLOOR", 0.5)
    ep = _ep(kg)
    _entity(kg, "seed", "Seed", 0)
    _fact(kg, "v1", "alpha one", ep, emb=0)
    _fact(kg, "b1", "widget one", ep)
    _fact(kg, "h1", "epsilon one", ep, src="seed")
    _fact(kg, "v2", "alpha two", ep, emb=0)
    _fact(kg, "b2", "widget two", ep)
    r = _engine(db_url)
    scores = {"alpha one": 0.9, "widget one": 0.9, "epsilon one": 0.9, "alpha two": 0.9}
    r._reranker = _Rerank(scores | {"widget two": 0.1})
    out = r.recall("widget", trust=FULL_TRUST)
    assert len(r._reranker.calls) == 1
    assert sorted(r._reranker.calls[0]) == sorted([*scores, "widget two"])
    assert _served(out) == {"v1", "b1", "h1", "v2"}, "the sub-floor fact drops"


def test_the_retrieval_count_bump_reaches_every_served_fact(kg, db_url):
    _seed_legs(kg)
    out = _engine(db_url, bump=True).recall("widget", trust=FULL_TRUST)
    assert _served(out) == _EVERY
    counts = dict(kg.execute("SELECT uuid, retrieval_count FROM kg_relationships").fetchall())
    assert {u: counts[u] for u in _EVERY} == dict.fromkeys(_EVERY, 1)


# ---------------------------------------------------------------------------
# Superseded pairs, the supersession surface and the overlay
# ---------------------------------------------------------------------------


def _supersession(conn, *, now_audience: str | None, old_audience: str | None) -> int:
    """A live fact and the retired one it superseded; returns the episode that cites the
    retired one. The live fact has no embedding and its text misses "widget", so a search for
    "widget" reaches it only through the retired one."""
    ep = _ep(conn)
    _fact(
        conn,
        "now",
        "gadget runs on postgres",
        ep,
        src="s",
        tgt="t",
        audience=now_audience,
    )
    _fact(
        conn,
        "old",
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


def test_full_trust_pairs_surface_and_overlay_serve_a_personal_pair(kg, db_url):
    ep = _supersession(kg, now_audience="personal", old_audience="personal")
    pairs, surface, overlay = _extras(Recall(db_url, ""), ep)
    assert pairs == ["f:old"]
    assert surface == {"now"}
    assert overlay == {ep: ["gadget runs on postgres"]}


def test_a_successor_repeating_a_served_fact_text_is_served_once(kg, db_url):
    """The ranked list serves a text once however many rows carry it. The same sentence is
    live on two pairs here: one is reached by the ranked list, the other as the successor of
    a retired fact, and the surface must not serve it a second time."""
    ep = _ep(kg)
    _fact(kg, "live-1", "widget runs on postgres", ep, emb=0, src="s1", tgt="t1")
    _fact(kg, "live-2", "widget runs on postgres", ep, src="s2", tgt="t2")
    _fact(
        kg,
        "old-2",
        "widget runs on sqlite",
        ep,
        emb=0,
        src="s2",
        tgt="t2",
        t_invalid=_INVALID,
        invalidated_by="live-2",
    )
    out = _engine(db_url).recall("widget", trust=FULL_TRUST)
    assert [f["fact"] for f in out["facts"]] == ["widget runs on postgres"]
    assert _served(out) == {"live-1"}


def test_successors_repeating_each_other_take_one_surface_slot(kg, db_url):
    """Two retired facts point at successors with the same text: the text is served once and
    the freed slot goes to the next successor in line, not to the repeat."""
    ep = _ep(kg)
    texts = ["gadget runs on postgres", "gadget runs on postgres"]
    texts += [f"gizmo {i} runs on mysql" for i in range(_SUP_LIMIT)]
    for i, text in enumerate(texts):
        # nearer the query for a smaller i, all inside the surface's distance gate
        w = 1.0 - 0.05 * i
        _fact(kg, f"now-{i}", text, ep, src=f"s{i}", tgt="t")
        _fact(
            kg,
            f"old-{i}",
            f"widget {i} runs on sqlite",
            ep,
            src=f"s{i}",
            tgt="t",
            t_invalid=_INVALID,
            invalidated_by=f"now-{i}",
        )
        tilted = onehot(0)
        tilted[0], tilted[1] = w, (1.0 - w * w) ** 0.5
        kg.execute(
            "UPDATE kg_relationships SET fact_embedding = %s::vector WHERE uuid = %s",
            (kg_pg._vec_literal(tilted), f"old-{i}"),
        )
    out = _engine(db_url).recall("widget", trust=FULL_TRUST)
    assert [f["fact"] for f in out["facts"]] == [texts[0], *texts[2 : _SUP_LIMIT + 1]]


def _many_retired(conn, ep: int, n: int, personal_from: int | None = None) -> None:
    """``n`` retired facts citing one episode, each with its live successor. Retired in index
    order, the oldest first, so reading the rows in storage order keeps the wrong ones. From
    ``personal_from`` on, both sides are personal."""
    for i in range(n):
        audience = "personal" if personal_from is not None and i >= personal_from else "work-safe"
        _fact(
            conn,
            f"now-{i}",
            f"gadget {i} runs on postgres",
            ep,
            src=f"s{i}",
            audience=audience,
        )
        _fact(
            conn,
            f"old-{i}",
            f"widget {i} runs on sqlite",
            ep,
            src=f"s{i}",
            t_invalid=f"2026-02-{i + 1:02d}T00:00:00+00:00",
            invalidated_by=f"now-{i}",
            audience=audience,
        )


def _gadgets(*indexes: int) -> list[str]:
    return [f"gadget {i} runs on postgres" for i in indexes]


@pytest.mark.parametrize("with_served_text", [False, True])
def test_the_overlay_keeps_the_most_recently_retired_links_under_its_limit(
    kg, db_url, with_served_text
):
    """More facts cite the episode than the lookup fetches: the rows that survive the fetch
    limit are the most recently retired, not whichever the plan reads."""
    ep = _ep(kg)
    cap = 2
    n = cap * (4 if with_served_text else 1) + 4
    _many_retired(kg, ep, n)
    served_text = {ep: " ".join(f"widget {i} runs on sqlite" for i in range(n))}
    overlay = Recall(db_url, "")._episode_supersessions(
        [ep], cap=cap, served_text=served_text if with_served_text else None
    )
    assert overlay == {ep: _gadgets(n - 1, n - 2)}


def test_a_restricted_overlay_keeps_the_most_recent_links_that_pass_the_label_rule(kg, db_url):
    """The label rule filters before the order and the limit: the newest links here are
    personal, so a restricted read keeps the newest work-safe ones and a full read the newest."""
    ep = _ep(kg)
    n = 12
    _many_retired(kg, ep, n, personal_from=n - 3)
    r = Recall(db_url, "")
    assert r._episode_supersessions([ep], cap=2) == {ep: _gadgets(n - 1, n - 2)}
    assert r._episode_supersessions([ep], cap=2, allowed_projects=[WORK]) == {
        ep: _gadgets(n - 4, n - 5)
    }


# ---------------------------------------------------------------------------
# A restricted device is served no personal-audience fact, from any leg
# ---------------------------------------------------------------------------


def test_restricted_search_serves_no_personal_fact_on_any_leg(kg, db_url):
    _seed_legs(kg)
    facts, seeds = Recall(db_url, "")._search_kg(
        "widget", onehot(0), [], 40, allowed_projects=[WORK]
    )
    assert _uuids(facts) == _SAFE, "vector, BM25 and hop: the work-safe facts only"
    assert seeds == [], "no seed entity: its name and summary come from all its facts"
    # the same rows are all there for full trust, so the filter is what removed them
    full, full_seeds = Recall(db_url, "")._search_kg("widget", onehot(0), [], 40)
    assert _uuids(full) == _EVERY
    assert {s["uuid"] for s in full_seeds} == {"seed"}


def test_a_restricted_recall_serves_no_personal_fact(kg, db_url):
    _seed_legs(kg)
    r = _engine(db_url)
    assert _served(r.recall("widget", trust=_RESTRICTED)) == _SAFE


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
    kg, db_url, now_audience, old_audience, served
):
    ep = _supersession(kg, now_audience=now_audience, old_audience=old_audience)
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
    allowlist, so only the label decides."""
    _supersession(kg, now_audience="work-safe", old_audience=old_audience)
    assert _served(_engine(db_url).recall("widget", trust=_RESTRICTED)) == served
    assert _served(_engine(db_url).recall("widget", trust=FULL_TRUST)) == {"now"}
