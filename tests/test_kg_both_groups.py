"""Full-trust recall reads both KG groups, as a superset; restricted reads the one asked for.

The extractor routes each fact into the ``technical`` or ``personal`` graph by a name
regex, so the owner's facts on one topic land in both, and a recall that read only the
requested group missed the other half. The group is not a trust boundary: a restricted
surface is scoped by fact provenance (schema 056). Full trust keeps the requested group's
serve exactly as a one-group read and ADDS what the other group has: up to
``CROSS_GROUP_EXTRAS`` vector/BM25 facts (no seed/hop leg) and the superseded pairs of the
extras it serves; the supersession surface and overlay stay on the requested group. Sections:
  * the group list a recall reads, per trust and per personal-scope switch
  * full trust: the requested group's facts unchanged, the other group's added after
  * superseded pairs and supersession calls, the fact floor and the retrieval-count bump
  * restricted callers and single-group reads send the statements they always did
"""

from __future__ import annotations

import uuid
from contextlib import nullcontext
from typing import Any

import psycopg
import pytest

import mcp_server.recall as recall_mod
from ingestion.surfaces import FULL_TRUST, SurfaceTrust
from mcp_server import kg_pg
from mcp_server.recall import Recall
from tests.helpers.embed import onehot

WORK = "work-proj"
TECH, PERS = "technical", "personal"


@pytest.fixture()
def kg(conn):
    def _wipe() -> None:
        conn.execute("TRUNCATE kg_relationships, kg_entities RESTART IDENTITY CASCADE")
        conn.execute("TRUNCATE episodes RESTART IDENTITY CASCADE")

    _wipe()
    yield conn
    _wipe()


def _ep(conn, project: str | None = WORK) -> int:
    return conn.execute(
        "INSERT INTO episodes (session_id, sequence, project, content) "
        "VALUES (%s, 1, %s, 'a turn') RETURNING id",
        (f"kbg-{uuid.uuid4().hex[:10]}", project),
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
    t_invalid: str | None = None,
    invalidated_by: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO kg_relationships (uuid, owner_id, group_id, src_uuid, tgt_uuid, name, "
        "  fact, fact_embedding, episodes, t_valid, t_invalid, invalidated_by) "
        "VALUES (%s, 'default', %s, %s, %s, 'RELATES', %s, %s::vector, %s::jsonb, "
        "        '2026-01-01T00:00:00+00:00', %s, %s)",
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
        ),
    )


def _entity(conn, uid: str, group: str, name: str, emb: int) -> None:
    conn.execute(
        "INSERT INTO kg_entities (uuid, owner_id, group_id, name, normalized_name, summary, "
        "  embedding) VALUES (%s, 'default', %s, %s, %s, 's', %s::vector)",
        (uid, group, name, name.lower(), kg_pg._vec_literal(onehot(emb))),
    )


def _seed_both(conn) -> None:
    """One fact per leg in each group: vector (embedding slot 0), BM25 ("widget"), hop."""
    ep = _ep(conn)
    _entity(conn, "seed-t", TECH, "Seed Tech", 0)
    _entity(conn, "seed-p", PERS, "Seed Personal", 0)
    _fact(conn, "v-t", TECH, "alpha beta", ep, emb=0)
    _fact(conn, "v-p", PERS, "gamma delta", ep, emb=0)
    _fact(conn, "b-t", TECH, "widget gizmo", ep)
    _fact(conn, "b-p", PERS, "widget thing", ep)
    _fact(conn, "h-t", TECH, "epsilon", ep, src="seed-t")
    _fact(conn, "h-p", PERS, "zeta", ep, src="seed-p")


_TECH_FACTS = {"v-t", "b-t", "h-t"}
_PERS_FACTS = {"v-p", "b-p", "h-p"}


class _Emb:
    def embed(self, texts, task):
        return [onehot(0) for _ in texts]


class _Now:
    """An executor that runs a submitted job before submit returns."""

    def submit(self, fn, *a, **k):
        fn(*a, **k)


def _engine(db_url: str, *, bump: bool = False) -> Recall:
    """Real KG legs; anything that would reach a network or rank episodes is stubbed."""
    r = Recall(db_url, "")
    r._ensure_embedder = lambda: _Emb()
    r._rerank_pool_scored = lambda q, pool: [(i, 0.9) for i in range(len(pool))]
    r._compact_to_passages = lambda q, eps, n, **_k: [{"id": e["id"], "text": "t"} for e in eps]
    r._search_web_reranked = lambda q, emb: []
    r._search_notes = lambda *a, **k: []
    r._increment_retrieval_counts = lambda ids: None
    r._record_metrics = lambda m: None
    if bump:
        r._async_executor = _Now()
    else:
        r._increment_fact_retrieval_counts = lambda *a, **k: None
    return r


def _served(out: dict[str, Any], bucket: str = "facts") -> set[str]:
    return {f["id"].removeprefix("f:") for f in out.get(bucket) or []}


_RESTRICTED = SurfaceTrust(trust="restricted", allowed_projects=(WORK,), known=True)


# ---------------------------------------------------------------------------
# The group list
# ---------------------------------------------------------------------------


def test_read_groups_per_trust_and_scope(monkeypatch):
    monkeypatch.delenv("SYNAPSE_PERSONAL_SCOPE", raising=False)
    assert kg_pg.read_groups(TECH, restricted=False) == [TECH, PERS]
    assert kg_pg.read_groups(PERS, restricted=False) == [PERS, TECH]
    assert kg_pg.read_groups(PERS, restricted=True) == PERS
    # Scope off: one graph exists, so full trust reads it the way it always did.
    monkeypatch.setenv("SYNAPSE_PERSONAL_SCOPE", "0")
    assert kg_pg.read_groups(TECH, restricted=False) == TECH


def test_group_sql_keeps_single_group_equality():
    assert kg_pg.group_sql(TECH) == "group_id = %s"
    assert kg_pg.group_sql([TECH, PERS], "a.group_id") == "a.group_id = ANY(%s)"


# ---------------------------------------------------------------------------
# Full trust: both groups, every leg
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("asked", "own", "added"),
    [(TECH, _TECH_FACTS, {"v-p", "b-p"}), (PERS, _PERS_FACTS, {"v-t", "b-t"})],
)
def test_full_trust_recall_adds_the_other_groups_matches(kg, db_url, asked, own, added):
    """Every leg for the requested group; vector and BM25 only for the other (its hop
    fact never comes back: no seed or hop leg runs there)."""
    _seed_both(kg)
    out = _engine(db_url).recall("widget", group_id=asked, trust=FULL_TRUST)
    assert _served(out) == own | added


def _many(conn) -> None:
    """More requested-group matches than the limit, plus other-group matches."""
    ep = _ep(conn)
    _entity(conn, "seed-t", TECH, "Seed Tech", 0)
    for i in range(15):
        _fact(conn, f"t{i:02d}", TECH, f"widget number {i}", ep, emb=0 if i % 2 else None)
    _fact(conn, "t-hop", TECH, "epsilon", ep, src="seed-t")
    for i in range(6):
        _fact(conn, f"p{i}", PERS, f"widget widget {i}", ep, emb=0)


def test_the_requested_groups_facts_are_exactly_a_one_group_read(kg, db_url):
    _many(kg)
    with psycopg.connect(db_url) as c, c.cursor() as cur:
        one = kg_pg.search_kg(cur, "widget", onehot(0), "default", TECH, [], 12)
        both = kg_pg.search_kg(cur, "widget", onehot(0), "default", [TECH, PERS], [], 12)
    assert len(one[0]) == 12
    assert both[0][:12] == one[0], "same facts, same order"
    assert both[1] == one[1], "same seeds"
    extras = both[0][12:]
    assert len(extras) == kg_pg.CROSS_GROUP_EXTRAS
    assert all(f["_cross"] and f["_uuid"].startswith("p") for f in extras)
    assert not any("_cross" in f for f in one[0])


@pytest.mark.parametrize(("asked", "facts"), [(TECH, _TECH_FACTS), (PERS, _PERS_FACTS)])
def test_restricted_recall_reads_only_the_requested_group(kg, db_url, asked, facts):
    _seed_both(kg)
    out = _engine(db_url).recall("widget", group_id=asked, trust=_RESTRICTED)
    assert _served(out) == facts


def _supersession_in(conn, group: str) -> int:
    """A retired fact and its live successor, both in ``group``; returns the episode."""
    ep = _ep(conn)
    _fact(conn, "now", group, "widget runs on postgres", ep, emb=None, src="s", tgt="t")
    _fact(
        conn,
        "old",
        group,
        "widget runs on sqlite",
        ep,
        emb=0,
        src="s",
        tgt="t",
        t_invalid="2026-02-01T00:00:00+00:00",
        invalidated_by="now",
    )
    return ep


def test_full_trust_supersession_extras_reach_the_other_group(kg, db_url):
    ep = _supersession_in(kg, PERS)
    r = Recall(db_url, "")
    groups = kg_pg.read_groups(TECH, restricted=False)
    assert [p["id"] for p in r._fetch_superseded_pairs_pg(groups, ["now"], 5)] == ["f:old"]
    assert {f["_uuid"] for f in r._surface_supersessions(onehot(0), groups, set())} == {"now"}
    assert r._episode_supersessions([ep], groups) == {ep: ["widget runs on postgres"]}
    # A restricted caller asking for the other group still gets none of them.
    scoped = {"allowed_projects": [WORK]}
    assert r._fetch_superseded_pairs_pg(TECH, ["now"], 5, **scoped) == []
    assert r._surface_supersessions(onehot(0), TECH, set(), **scoped) == []
    assert r._episode_supersessions([ep], TECH, **scoped) == {}


def test_full_trust_recall_pairs_a_fact_served_from_the_other_group(kg, db_url):
    _supersession_in(kg, PERS)
    out = _engine(db_url).recall("widget", group_id=TECH, trust=FULL_TRUST)
    assert "now" in _served(out)
    assert _served(out, "superseded_facts") == {"old"}


def _spied_engine(db_url: str, surface: list, overlay: list, metrics: list) -> Recall:
    """Spies on the supersession surface/overlay calls; one served episode for the overlay."""
    r = _engine(db_url)
    r._search_bm25_episodes = lambda *a, **k: [{"id": "e:7", "doc_type": "episode"}]
    r._compact_to_passages = lambda q, eps, n, **_k: [{"id": "e:7", "content": "t"}]

    def _surface(emb, g, served, cap=3, **k):
        surface.append((g, cap, k))
        return [{"fact": f"{g} successor", "_uuid": f"sup-{len(surface)}"}]

    def _overlay(ids, g, **k):
        overlay.append((g, k))
        return {7: [f"successor {len(overlay)}"]}

    r._surface_supersessions = _surface
    r._episode_supersessions = _overlay
    r._record_metrics = metrics.append
    return r


def test_full_trust_supersession_surface_and_overlay_stay_on_the_requested_group(kg, db_url):
    """The other group's surface is a serial scan over its retired edges (0.4-2s on a
    real graph) and its overlay attached off-topic successors on the bench, so full trust
    makes exactly the one-group calls; their facts are not cross-group extras."""
    surface: list = []
    overlay: list = []
    metrics: list = []
    out = _spied_engine(db_url, surface, overlay, metrics).recall(
        "widget", group_id=PERS, trust=FULL_TRUST
    )
    assert surface == [(PERS, 3, {})]
    assert [g for g, _ in overlay] == [PERS]
    assert out["episodes"][0]["superseded_by"] == ["successor 1"]
    assert [f["id"] for f in out["facts"]] == ["f:sup-1"]
    assert "facts_cross_group" not in metrics[0]["served_ids"]


@pytest.mark.parametrize(("extra_score", "paired"), [(0.9, True), (0.1, False)])
def test_a_cross_group_extra_the_floor_drops_takes_its_superseded_pair_with_it(
    kg, db_url, monkeypatch, extra_score, paired
):
    monkeypatch.setattr(recall_mod, "_RECALL_FACT_FLOOR", 0.5)
    _supersession_in(kg, PERS)  # the other group's live "widget runs on postgres" + its predecessor
    _fact(kg, "own-1", TECH, "widget alpha", _ep(kg), emb=0)
    _fact(kg, "own-2", TECH, "widget beta", _ep(kg), emb=0)
    r = _engine(db_url)
    r._reranker = _Rerank(
        {"widget alpha": 0.9, "widget beta": 0.9, "widget runs on postgres": extra_score}
    )
    out = r.recall("widget", group_id=TECH, trust=FULL_TRUST)
    assert ("now" in _served(out)) is paired
    assert _served(out, "superseded_facts") == ({"old"} if paired else set())


def test_restricted_recall_makes_no_other_group_call(kg, db_url):
    surface: list = []
    overlay: list = []
    metrics: list = []
    _spied_engine(db_url, surface, overlay, metrics).recall(
        "widget", group_id=PERS, trust=_RESTRICTED
    )
    assert surface == [(PERS, 3, {"allowed_projects": [WORK]})]
    assert [g for g, _ in overlay] == [PERS]
    assert "facts_cross_group" not in metrics[0]["served_ids"]


class _Rerank:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[list[str]] = []

    def rerank_scored(self, query, docs):
        self.calls.append(list(docs))
        ranked = sorted(enumerate(docs), key=lambda x: -self.scores[x[1]])
        return [(i, self.scores[d]) for i, d in ranked]


def test_floor_keeps_the_requested_groups_backstop_and_floors_extras_alone(monkeypatch):
    monkeypatch.setattr(recall_mod, "_RECALL_FACT_FLOOR", 0.5)
    r = object.__new__(Recall)
    r._reranker = _Rerank({"a": 0.2, "b": 0.3, "x": 0.9, "y": 0.1})
    facts = [
        {"fact": "a", "_uuid": "a"},
        {"fact": "b", "_uuid": "b"},
        {"fact": "x", "_uuid": "x", "_cross": True},
        {"fact": "y", "_uuid": "y", "_cross": True},
    ]
    # Own facts all sub-floor: the keep-one backstop keeps the best OWN fact, as a
    # one-group read did, even though an extra outscores it; the sub-floor extra drops.
    assert [f["fact"] for f in r._floor_facts("q", facts)] == ["b", "x"]
    assert r._reranker.calls == [["a", "b"], ["x", "y"]], "own facts reranked as before"
    # A lone own fact is not reranked (recall never floored one); extras still are.
    r._reranker.calls.clear()
    assert [f["fact"] for f in r._floor_facts("q", [facts[0], facts[3]])] == ["a"]
    assert r._reranker.calls == [["y"]]


# ---------------------------------------------------------------------------
# The retrieval-count bump
# ---------------------------------------------------------------------------


def _count(conn, uid: str) -> int:
    return conn.execute(
        "SELECT retrieval_count FROM kg_relationships WHERE uuid = %s", (uid,)
    ).fetchone()[0]


def test_bump_reaches_a_fact_served_from_the_other_group(kg, db_url):
    _seed_both(kg)
    out = _engine(db_url, bump=True).recall("widget", group_id=TECH, trust=FULL_TRUST)
    assert "b-p" in _served(out)
    assert _count(kg, "b-p") == 1
    assert _count(kg, "b-t") == 1


# ---------------------------------------------------------------------------
# Single-group statements are unchanged
# ---------------------------------------------------------------------------


class _RecordingConn:
    def __init__(self) -> None:
        self.sql: list[str] = []
        self.params: list[Any] = []

    def execute(self, sql, params=None):
        self.sql.append(sql)
        self.params.append(params)
        return self

    def fetchall(self):
        return []

    def fetchone(self):
        return None


def _statements(group: str | list[str], allowed: list[str] | None) -> _RecordingConn:
    cur = _RecordingConn()
    kg_pg.search_kg(cur, "widget q", onehot(0), "default", group, [], 5, allowed)
    r = Recall("postgresql://unused/none", "")
    r._pg = lambda: nullcontext(cur)
    r._kg_scope_ready = lambda: True
    scope = {} if allowed is None else {"allowed_projects": allowed}
    r._fetch_superseded_pairs_pg(group, ["u1"], 2, **scope)
    r._surface_supersessions(onehot(0), group, set(), **scope)
    r._episode_supersessions([1], group, **scope)
    return cur


@pytest.mark.parametrize("allowed", [None, [WORK]])
def test_one_group_sends_plain_equality_with_a_scalar(allowed):
    cur = _statements(TECH, allowed)
    assert len(cur.sql) == 6
    assert not any("group_id = ANY" in s for s in cur.sql)
    assert all(TECH in p for p in cur.params)
    assert not any(isinstance(v, list) and TECH in v for p in cur.params for v in p)


def test_full_trust_list_reads_the_requested_group_with_its_one_group_statements():
    one = _statements(TECH, None)
    both = _statements([TECH, PERS], None)
    # vector, BM25, seed/degree for the requested group: text and parameters unchanged
    assert both.sql[:3] == one.sql[:3]
    assert both.params[:3] == one.params[:3]
    # then the other group's vector and BM25, and the KG extras over both groups
    assert len(both.sql) == 8
    assert all("group_id = ANY(%s)" in s for s in both.sql[3:])
    assert [PERS] in both.params[3] and [PERS] in both.params[4]
    assert all([TECH, PERS] in p for p in both.params[5:])
