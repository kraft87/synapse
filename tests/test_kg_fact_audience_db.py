"""The fact-audience rule and the owner-seed exclusion against a real database (schema 066).

A restricted surface is served a KG fact only when its provenance is inside the allowlist
(056) AND it is labelled work-safe, or unlabelled and older than 066 (the legacy clause).
The statements are in test_kg_fact_audience_sql.py; what they do to rows is here:

  * every restricted leg: vector, BM25, the seed degree gate, the hop, the superseded
    pairs, the supersession surface and the episode overlay
  * a fact regrouped out of the personal pile keeps its personal v0 label, whatever provenance says
  * a missing cutover stamp fails closed; the BM25 TopK pushdown survives the clause
  * the owner's flagged entity is never a hop seed, for any caller, and its facts are
    still served by the other legs
  * a database without 066 serves a restricted read nothing, on every leg, and full trust
    exactly what it did before, with no errors and no warnings; the schema applied under a
    running engine takes effect without a restart
"""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import Any

import psycopg
import pytest
from psycopg.rows import dict_row

from ingestion.surfaces import FACT_AUDIENCE_PROBE, SurfaceTrust
from mcp_server import kg_pg
from mcp_server import recall_fact_audience as recall_fa_mod
from mcp_server.recall import Recall
from mcp_server.recall_warnings import warn_sink
from tests.helpers.embed import GROUP, onehot
from tests.helpers.surfaces import clear_surfaces, register_full, register_restricted

WORK = "work-proj"
PERS = "personal"  # the legacy second pile: rows written before 067 still carry it, no read looks
LEGS = ("v", "b", "h")  # vector, BM25, hop


@pytest.fixture()
def kg(conn):
    """A clean episodes + KG slate (KG first: truncating episodes fires 056's trigger)."""

    def _wipe() -> None:
        conn.execute("TRUNCATE kg_relationships, kg_entities RESTART IDENTITY CASCADE")
        conn.execute("TRUNCATE episodes RESTART IDENTITY CASCADE")
        clear_surfaces(conn)

    _wipe()
    yield conn
    _wipe()


@pytest.fixture()
def cutover(conn):
    """``set(n)`` moves the legacy cutover (``None`` removes the stamp); the stamp the
    schema wrote is put back afterwards."""
    key = kg_pg.FACT_AUDIENCE_CUTOVER_KEY
    original = conn.execute("SELECT value FROM synapse_meta WHERE key = %s", (key,)).fetchone()

    def _set(n: int | None) -> None:
        if n is None:
            conn.execute("DELETE FROM synapse_meta WHERE key = %s", (key,))
        else:
            conn.execute(
                "INSERT INTO synapse_meta (key, value) VALUES (%s, %s) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                (key, str(n)),
            )

    yield _set
    if original is None:
        _set(None)
    else:
        _set(int(original[0]))


def _ep(conn, project: str = WORK) -> int:
    return conn.execute(
        "INSERT INTO episodes (session_id, sequence, project, content) "
        "VALUES (%s, 1, %s, 'a turn') RETURNING id",
        (f"fa-{uuid.uuid4().hex[:10]}", project),
    ).fetchone()[0]


def _fact(
    conn,
    uid: str,
    fact: str,
    ep: int,
    *,
    audience: str | None,
    emb: int | None = None,
    src: str = "ent-x",
    tgt: str = "ent-y",
    group: str = GROUP,
    t_invalid: str | None = None,
    invalidated_by: str | None = None,
    label_version: int | None = None,
) -> None:
    conn.execute(
        "INSERT INTO kg_relationships (uuid, owner_id, group_id, src_uuid, tgt_uuid, name, "
        "  fact, fact_embedding, episodes, t_valid, t_invalid, invalidated_by, audience, "
        "  label_version) "
        "VALUES (%s, 'default', %s, %s, %s, 'RELATES', %s, %s::vector, %s::jsonb, "
        "        '2026-01-01T00:00:00+00:00', %s, %s, %s, %s)",
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
            label_version,
        ),
    )


def _entity(conn, uid: str, name: str, emb: int, group: str = GROUP) -> None:
    conn.execute(
        "INSERT INTO kg_entities (uuid, owner_id, group_id, name, normalized_name, summary, "
        "  embedding) VALUES (%s, 'default', %s, %s, %s, 's', %s::vector)",
        (uid, group, name, name.lower(), kg_pg._vec_literal(onehot(emb))),
    )


def _max_id(conn) -> int:
    return conn.execute("SELECT COALESCE(max(id), 0) FROM kg_relationships").fetchone()[0]


def _uuids(facts: list[dict[str, Any]]) -> set[str]:
    return {f["_uuid"] for f in facts}


class _Emb:
    def embed(self, texts, task):
        return [onehot(0) for _ in texts]


def _recall_engine(db_url: str) -> Recall:
    """Real KG legs; everything that would reach a network or rank episodes is stubbed."""
    r = Recall(db_url, "")
    r._ensure_embedder = lambda: _Emb()
    r._rerank_pool_scored = lambda q, pool: [(i, 0.9) for i in range(len(pool))]
    r._compact_to_passages = lambda q, eps, n, **_k: [{"id": e["id"], "text": "t"} for e in eps]
    r._search_web_reranked = lambda q, emb, **_k: []
    r._search_notes = lambda *a, **k: []
    r._increment_retrieval_counts = lambda ids: None
    r._increment_fact_retrieval_counts = lambda *a, **k: None
    r._record_metrics = lambda m: None
    return r


def _engine_on(tx) -> Recall:
    """recall() on ONE connection: a single leg worker keeps every statement serial on the
    transaction that hides the columns."""
    r = _recall_engine("postgresql://unused/none")
    r._pg = lambda: nullcontext(tx)
    r._leg_executor = ThreadPoolExecutor(max_workers=1)
    return r


def _restricted_kg(r: Recall, **kw: Any) -> set[str]:
    facts, seeds = r._search_kg("widget", onehot(0), [], 20, allowed_projects=[WORK], **kw)
    assert seeds == []
    return _uuids(facts)


# ---------------------------------------------------------------------------
# The label rule on every fact leg
# ---------------------------------------------------------------------------


def _leg_fact(conn, leg: str, kind: str, audience: str | None, ep: int) -> None:
    """One fact that only ``leg`` can serve: ``v`` by vector (its text misses the query), ``b``
    by BM25 (no embedding), ``h`` by the hop from the seed entity (neither)."""
    text = {"v": f"alpha {kind}", "b": f"widget {kind}", "h": f"epsilon {kind}"}[leg]
    _fact(
        conn,
        f"{leg}-{kind}",
        text,
        ep,
        audience=audience,
        emb=0 if leg == "v" else None,
        src="seed-ent" if leg == "h" else "ent-x",
    )


def _seed_label_matrix(conn, cutover) -> int:
    """Per leg: a work-safe fact, a personal one, a legacy NULL one (written before the
    cutover) and a new NULL one (after it). All four sit inside the allowlist."""
    ep = _ep(conn)
    _entity(conn, "seed-ent", "Seed", 0)
    for leg in LEGS:
        _leg_fact(conn, leg, "legacy", None, ep)
    cutover(_max_id(conn))
    for leg in LEGS:
        _leg_fact(conn, leg, "safe", "work-safe", ep)
        _leg_fact(conn, leg, "personal", "personal", ep)
        _leg_fact(conn, leg, "new", None, ep)
    return ep


SERVED = {f"{leg}-{kind}" for leg in LEGS for kind in ("safe", "legacy")}
EVERY = {f"{leg}-{kind}" for leg in LEGS for kind in ("safe", "legacy", "personal", "new")}


def test_a_restricted_read_serves_work_safe_and_legacy_unlabelled_facts_on_every_leg(
    kg, cutover, db_url
):
    _seed_label_matrix(kg, cutover)
    r = Recall(db_url, "")
    assert _restricted_kg(r) == SERVED
    # one allowed project more changes nothing: only the label decides here
    facts, _ = r._search_kg("widget", onehot(0), [], 20, allowed_projects=[WORK, "other"])
    assert _uuids(facts) == SERVED


def test_full_trust_serves_every_label(kg, cutover, db_url):
    _seed_label_matrix(kg, cutover)
    facts, seeds = Recall(db_url, "")._search_kg("widget", onehot(0), [], 20)
    assert _uuids(facts) == EVERY
    assert [s["uuid"] for s in seeds] == ["seed-ent"]


def test_restricted_recall_serves_the_same_facts(kg, cutover, db_url):
    _seed_label_matrix(kg, cutover)
    restricted = register_restricted(kg, [WORK])
    full = register_full(kg)
    sink: list[str] = []
    with warn_sink(sink):
        out = _recall_engine(db_url).recall("widget", surface=restricted)
    assert {f["id"].removeprefix("f:") for f in out["facts"]} == SERVED
    assert "warnings" not in out and sink == []
    out_full = _recall_engine(db_url).recall("widget", surface=full)
    assert {f["id"].removeprefix("f:") for f in out_full["facts"]} == EVERY


def test_the_cutover_is_a_database_value_read_on_every_statement(kg, cutover, db_url):
    """Moving the stamp moves the legacy window: nothing in the process caches it."""
    _seed_label_matrix(kg, cutover)
    r = Recall(db_url, "")
    cutover(0)
    assert _restricted_kg(r) == {f"{leg}-safe" for leg in LEGS}
    cutover(_max_id(kg))
    assert _restricted_kg(r) == {f"{leg}-{k}" for leg in LEGS for k in ("safe", "legacy", "new")}


def test_a_missing_cutover_stamp_serves_work_safe_facts_only(kg, cutover, db_url):
    _seed_label_matrix(kg, cutover)
    cutover(None)
    assert _restricted_kg(Recall(db_url, "")) == {f"{leg}-safe" for leg in LEGS}


@pytest.mark.parametrize(
    ("kind", "is_seed"),
    [("work-safe", True), ("legacy", True), ("personal", False), ("new", False)],
)
def test_a_seed_counts_only_the_edges_a_restricted_caller_may_be_served(
    kg, cutover, db_url, kind, is_seed
):
    """The degree gate sits on the same rule: an entity whose only reachable edges are
    withheld has no servable hop, so it takes no seed slot, and the hop query is skipped."""
    cutover(0)
    ep = _ep(kg)
    _entity(kg, "lonely", "Lonely", 0)
    _fact(
        kg,
        "lonely-edge",
        "omega",
        ep,
        audience={"work-safe": "work-safe", "personal": "personal"}.get(kind),
        src="lonely",
    )
    if kind == "legacy":
        cutover(_max_id(kg))
    executed: list[str] = []

    class _Rec:
        def __init__(self, cur):
            self._cur = cur

        def execute(self, sql, params=None):
            executed.append(sql)
            self._cur.execute(sql, params)
            return self

        def fetchall(self):
            return self._cur.fetchall()

    def _hops(allowed: list[str] | None) -> int:
        executed.clear()
        with psycopg.connect(db_url) as c, c.cursor() as cur:
            kg_pg.search_kg_postgres(_Rec(cur), "q", onehot(0), "default", [], 5, allowed)
        return sum("src_uuid = %s OR tgt_uuid = %s" in s for s in executed)

    assert _hops([WORK]) == (1 if is_seed else 0)
    assert _hops(None) == 1, "full trust seeds it whatever the label"


# ---------------------------------------------------------------------------
# The superseded-pair legs: BOTH sides must pass
# ---------------------------------------------------------------------------

_INV = "2026-02-01T00:00:00+00:00"


def _pair(conn, name: str, *, now: str | None, old: str | None, ep: int) -> None:
    """A live edge ``n-<name>`` and the retired one it superseded, ``o-<name>``."""
    _fact(conn, f"n-{name}", f"now {name}", ep, audience=now, src=f"s-{name}", tgt=f"t-{name}")
    _fact(
        conn,
        f"o-{name}",
        f"old {name}",
        ep,
        audience=old,
        emb=1,
        src=f"s-{name}",
        tgt=f"t-{name}",
        t_invalid=_INV,
        invalidated_by=f"n-{name}",
    )


def _seed_pairs(conn, cutover) -> int:
    """ok: both work-safe. displaced: the old side is personal. legacy: the old side is
    unlabelled and predates the cutover. served-new: the current side is unlabelled and
    newer. The legacy old side is written first, so the cutover falls after it."""
    ep = _ep(conn)
    _fact(
        conn, "o-legacy", "old legacy", ep, audience=None, emb=1, src="s-legacy", tgt="t-legacy",
        t_invalid=_INV, invalidated_by="n-legacy",
    )  # fmt: skip
    cutover(_max_id(conn))
    _fact(conn, "n-legacy", "now legacy", ep, audience="work-safe", src="s-legacy", tgt="t-legacy")
    _pair(conn, "ok", now="work-safe", old="work-safe", ep=ep)
    _pair(conn, "displaced", now="work-safe", old="personal", ep=ep)
    _pair(conn, "served-new", now=None, old="work-safe", ep=ep)
    return ep


NOW = ["n-ok", "n-displaced", "n-legacy", "n-served-new"]


def test_superseded_pairs_need_both_sides_to_pass_the_label_rule(kg, cutover, db_url):
    _seed_pairs(kg, cutover)
    r = Recall(db_url, "")
    scoped = r._fetch_superseded_pairs_pg(NOW, 10, allowed_projects=[WORK])
    assert [p["id"] for p in scoped] == ["f:o-ok", "f:o-legacy"]
    full = r._fetch_superseded_pairs_pg(NOW, 10)
    assert [p["id"] for p in full] == ["f:o-ok", "f:o-displaced", "f:o-legacy", "f:o-served-new"]


def test_the_supersession_surface_needs_both_edges_to_pass_the_label_rule(kg, cutover, db_url):
    _seed_pairs(kg, cutover)
    r = Recall(db_url, "")
    scoped = r._surface_supersessions(onehot(1), set(), cap=10, allowed_projects=[WORK])
    assert _uuids(scoped) == {"n-ok", "n-legacy"}
    full = r._surface_supersessions(onehot(1), set(), cap=10)
    assert _uuids(full) == set(NOW)


def test_the_episode_overlay_needs_both_edges_to_pass_the_label_rule(kg, cutover, db_url):
    ep = _seed_pairs(kg, cutover)
    r = Recall(db_url, "")
    scoped = r._episode_supersessions([ep], allowed_projects=[WORK])
    assert sorted(scoped[ep]) == ["now legacy", "now ok"]
    full = r._episode_supersessions([ep])
    assert sorted(full[ep]) == ["now displaced", "now legacy", "now ok", "now served-new"]


# ---------------------------------------------------------------------------
# Facts regrouped out of the personal pile
# ---------------------------------------------------------------------------


def test_a_regrouped_fact_keeps_personal_v0_and_is_served_a_restricted_request_nothing(kg, db_url):
    """067 moves every personal-pile fact into the one group and keeps its 'personal' label (it
    was stamped on insert until then), so naming no pile at all is still no way round the
    label, whatever the fact's provenance says."""
    ep = _ep(kg)
    _entity(kg, "p-seed", "PSeed", 0)
    _fact(kg, "p-vec", "alpha", ep, audience="personal", label_version=0, emb=0)
    _fact(kg, "p-bm25", "widget thing", ep, audience="personal", label_version=0)
    _fact(kg, "p-hop", "epsilon", ep, audience="personal", label_version=0, src="p-seed")
    assert kg.execute(
        "SELECT DISTINCT group_id, audience, label_version FROM kg_relationships"
    ).fetchall() == [(GROUP, "personal", 0)]
    r = Recall(db_url, "")
    assert _restricted_kg(r) == set()
    facts, _ = r._search_kg("widget", onehot(0), [], 20)
    assert _uuids(facts) == {"p-vec", "p-bm25", "p-hop"}


# ---------------------------------------------------------------------------
# The BM25 pushdown
# ---------------------------------------------------------------------------


def test_the_restricted_bm25_statement_still_runs_as_a_paradedb_topk_scan(kg, cutover, db_url):
    """The cutover rides in as an InitPlan, so the ParadeDB TopN pushdown survives the
    label rule: it becomes a filter inside the Tantivy query and the scan stays TopK.
    The 056-only statement is the control, so a pg_search build that prints plans
    differently skips here rather than passing on nothing."""
    _seed_label_matrix(kg, cutover)
    ep = _ep(kg)
    for i in range(40):
        _fact(kg, f"bulk-{i}", f"widget bulk {i}", ep, audience="work-safe")
    seen: list[tuple[str, Any]] = []

    class _Rec:
        def __init__(self, cur):
            self._cur = cur

        def execute(self, sql, params=None):
            seen.append((sql, params))
            self._cur.execute(sql, params)
            return self

        def fetchall(self):
            return self._cur.fetchall()

    def _plan(audience: bool) -> str:
        seen.clear()
        with psycopg.connect(db_url, autocommit=True) as c, c.cursor() as cur:
            cur.execute("SET enable_seqscan = off")
            kg_pg.search_kg_postgres(
                _Rec(cur),
                "widget",
                onehot(0),
                "default",
                [],
                5,
                [WORK],
                fact_audience=audience,
            )
            (sql, params), *_ = [s for s in seen if "paradedb.match" in s[0]]
            return "\n".join(r[0] for r in c.execute("EXPLAIN (COSTS OFF) " + sql, params))

    control = _plan(False)
    if "kg_rel_bm25" not in control or "TopK" not in control:
        pytest.skip("this pg_search build does not print a TopK ParadeDB scan for the control")
    plan = _plan(True)
    assert "kg_rel_bm25" in plan and "TopK" in plan, plan
    assert "InitPlan" in plan, "the cutover is read once per statement"


def _bm25_leg(db_url: str, *, audience: bool, limit: int) -> tuple[list[str], str]:
    """The restricted BM25 statement search_kg_postgres sends, run under the planner settings
    recall_sources gives it: its fact uuids best first, and its plan with the Tantivy query."""
    seen: list[tuple[str, Any]] = []

    class _Rec:
        def __init__(self, cur):
            self._cur = cur

        def execute(self, sql, params=None):
            seen.append((sql, params))
            self._cur.execute(sql, params)
            return self

        def fetchall(self):
            return self._cur.fetchall()

    with psycopg.connect(db_url, autocommit=True) as c, c.cursor() as cur:
        cur.execute("SET enable_seqscan = off")
        cur.execute("SET max_parallel_workers_per_gather = 0")
        kg_pg.search_kg_postgres(
            _Rec(cur),
            "widget",
            onehot(0),
            "default",
            [],
            limit,
            [WORK],
            fact_audience=audience,
        )
        (sql, params), *_ = [s for s in seen if "paradedb.match" in s[0]]
        uuids = [r[0] for r in c.execute(sql, params).fetchall()]
        plan = "\n".join(r[0] for r in c.execute("EXPLAIN (VERBOSE, COSTS OFF) " + sql, params))
    return uuids, plan


def test_a_label_rule_every_fact_passes_leaves_the_bm25_ranking_as_it_was(kg, cutover, db_url):
    """When every matching fact is work-safe or legacy, the label rule must change nothing:
    the same facts in the same order as the 056-only statement. (The leg only ranks; RRF
    never reads a BM25 score.)

    pg_search turns a WHERE-level OR into a Tantivy boolean of "match everything"
    sub-queries, one per clause of the scan; on prod (0.23.2, 2026-11-09) that read 38x the
    pages (441ms against 32ms) and scored the unlabelled branch higher, so legacy facts
    outranked work-safe ones. One expression stays a single filter on the match's hits."""
    ep = _ep(kg)
    for i in range(30):  # legacy: unlabelled, written before the cutover; one length each
        _fact(kg, f"old-{i}", "widget" + " gear" * i, ep, audience=None)
    cutover(_max_id(kg))
    for i in range(30):
        _fact(kg, f"safe-{i}", "widget widget" + " cog" * i, ep, audience="work-safe")
    control, control_plan = _bm25_leg(db_url, audience=False, limit=5)
    rows, plan = _bm25_leg(db_url, audience=True, limit=5)
    assert len(control) == 15
    assert rows == control
    if "kg_rel_bm25" not in control_plan or "TopK" not in control_plan:
        pytest.skip("this pg_search build does not print a TopK ParadeDB scan for the control")
    assert "kg_rel_bm25" in plan and "TopK" in plan, plan
    assert '"indexed_query":"all"' not in plan.replace(" ", ""), plan


def test_the_bm25_leg_drops_personal_and_new_unlabelled_facts_before_the_top_k(kg, cutover, db_url):
    """The rule filters inside the scan, so a personal or new unlabelled fact that would rank
    first does not take a top-K slot from a servable one."""
    ep = _ep(kg)
    _fact(kg, "legacy", "widget legacy", ep, audience=None)
    cutover(_max_id(kg))
    for i in range(10):
        _fact(kg, f"pers-{i}", "widget widget widget widget personal", ep, audience="personal")
        _fact(kg, f"new-{i}", "widget widget widget widget fresh", ep, audience=None)
    _fact(kg, "safe", "widget safe", ep, audience="work-safe")
    rows, _ = _bm25_leg(db_url, audience=True, limit=1)
    assert set(rows) == {"legacy", "safe"}


def test_a_legacy_fact_relabelled_personal_drops_out_of_every_leg(kg, cutover, db_url):
    """Prod labels a legacy fact by UPDATE (the poller stage, and 066's own seeding), not at
    insert: the rule must read the row's current label, on every leg."""
    _seed_label_matrix(kg, cutover)
    before = _restricted_kg(_recall_engine(db_url))
    kg.execute("UPDATE kg_relationships SET audience = 'personal' WHERE uuid LIKE '%-legacy'")
    after = _restricted_kg(_recall_engine(db_url))
    assert {f"{leg}-legacy" for leg in LEGS} <= before
    assert after == before - {f"{leg}-legacy" for leg in LEGS}


# ---------------------------------------------------------------------------
# The owner's entity is never a hop seed
# ---------------------------------------------------------------------------


def _owner_world(conn, group: str = GROUP) -> None:
    """Two entities at the query: the owner's and another's. Each has one fact only the
    hop reaches; the owner also has a vector fact and a BM25 fact."""
    ep = _ep(conn)
    own, other = f"own-{group}", f"oth-{group}"
    _entity(conn, own, "Owner", 0, group)
    _entity(conn, other, "Other", 0, group)
    kw: dict[str, Any] = {"audience": "work-safe", "group": group}
    _fact(conn, f"hop-own-{group}", f"zeta {group}", ep, src=own, **kw)
    _fact(conn, f"hop-oth-{group}", f"eta {group}", ep, src=other, **kw)
    _fact(conn, f"vec-own-{group}", f"alpha beta {group}", ep, emb=0, src=own, **kw)
    _fact(conn, f"bm-own-{group}", f"widget gizmo {group}", ep, src=own, **kw)


def _flag(conn, uid: str) -> None:
    conn.execute("UPDATE kg_entities SET is_self = true WHERE uuid = %s", (uid,))


def _read(r: Recall, allowed: list[str] | None = None) -> set[str]:
    facts, _ = r._search_kg("widget", onehot(0), [], 20, allowed_projects=allowed)
    return _uuids(facts)


@pytest.mark.parametrize("allowed", [None, [WORK]], ids=["full trust", "restricted"])
def test_the_flagged_owner_is_never_a_hop_seed_but_its_facts_still_serve(kg, db_url, allowed):
    _owner_world(kg)
    r = Recall(db_url, "")
    everything = {f"hop-own-{GROUP}", f"vec-own-{GROUP}", f"bm-own-{GROUP}", f"hop-oth-{GROUP}"}
    assert _read(r, allowed=allowed) == everything
    _flag(kg, f"own-{GROUP}")
    r = Recall(db_url, "")  # a new engine: nothing cached
    assert _read(r, allowed=allowed) == everything - {f"hop-own-{GROUP}"}


def test_a_cleared_flag_leaves_the_read_as_it_was(kg, db_url):
    _owner_world(kg)
    before = _read(Recall(db_url, ""))
    _flag(kg, f"own-{GROUP}")
    kg.execute("UPDATE kg_entities SET is_self = false")
    assert _read(Recall(db_url, "")) == before


def test_every_flagged_owner_row_is_excluded_as_a_seed(kg, db_url):
    """Rows from before 067 can leave the owner a node in each legacy group, both flagged:
    neither is a hop seed, and the other entities' facts in either group still serve."""
    _owner_world(kg, GROUP)
    _owner_world(kg, PERS)
    _flag(kg, f"own-{GROUP}")
    served = _read(Recall(db_url, ""))
    assert f"hop-own-{GROUP}" not in served
    assert f"hop-own-{PERS}" in served
    _flag(kg, f"own-{PERS}")
    served = _read(Recall(db_url, ""))
    assert f"hop-own-{GROUP}" not in served and f"hop-own-{PERS}" not in served
    assert {f"hop-oth-{GROUP}", f"hop-oth-{PERS}"} <= served


def test_the_flagged_set_is_cached_for_its_ttl(kg, db_url, monkeypatch):
    _owner_world(kg)
    monkeypatch.setattr(recall_fa_mod, "_OWNER_SEEDS_TTL_S", 3600.0)
    r = Recall(db_url, "")
    assert f"hop-own-{GROUP}" in _read(r)
    _flag(kg, f"own-{GROUP}")
    assert f"hop-own-{GROUP}" in _read(r), "inside the TTL the engine keeps the set it read"
    monkeypatch.setattr(recall_fa_mod, "_OWNER_SEEDS_TTL_S", 0.0)
    assert f"hop-own-{GROUP}" not in _read(r)
    kg.execute("UPDATE kg_entities SET is_self = false")
    assert f"hop-own-{GROUP}" in _read(r), "a cleared flag is honoured on the next read"


def test_a_failed_owner_lookup_excludes_nothing_and_is_not_cached():
    class _Broken:
        def transaction(self):
            raise psycopg.OperationalError("down")

    r = Recall("postgresql://unused/none", "")
    assert r._owner_seed_uuids(_Broken()) == []
    assert r._owner_seeds_at is None


# ---------------------------------------------------------------------------
# A database without schema 066
# ---------------------------------------------------------------------------


@pytest.fixture()
def no_066(kg, cutover, db_url):
    """A connection on which 066's columns do not exist: they are dropped inside a
    transaction that is rolled back afterwards (CASCADE takes the CHECK-less trigger and the
    partial index with them), so the shared test database is never left altered. The
    data, labels and all, is seeded first through the committed fixture connection."""
    _seed_label_matrix(kg, cutover)
    ep = _ep(kg)
    _pair(kg, "ok", now="work-safe", old="work-safe", ep=ep)
    _pair(kg, "pers", now="personal", old="personal", ep=ep)
    _pair(kg, "new", now=None, old=None, ep=ep)
    tx = psycopg.connect(db_url, row_factory=dict_row)
    try:
        for col in ("audience", "label_version", "label_model"):
            tx.execute(f"ALTER TABLE kg_relationships DROP COLUMN {col} CASCADE")
        tx.execute("ALTER TABLE kg_entities DROP COLUMN is_self CASCADE")
        yield tx, ep
    finally:
        tx.rollback()
        tx.close()


PRE_066_PAIRS = ["n-ok", "n-pers", "n-new"]
NOTHING: dict[str, Any] = {"kg": set(), "pairs": [], "surface": set(), "overlay": {}}


def _every_path(r: Recall, ep: int) -> dict[str, Any]:
    allowed = [WORK]
    return {
        "kg": _restricted_kg(r),
        "pairs": [p["id"] for p in r._fetch_superseded_pairs_pg(PRE_066_PAIRS, 10, allowed)],
        "surface": _uuids(r._surface_supersessions(onehot(1), set(), 10, allowed)),
        "overlay": r._episode_supersessions([ep], allowed_projects=allowed),
    }


def test_before_066_restricted_reads_serve_nothing(no_066, db_url, caplog):
    """Without the label the only rule left is provenance, which cannot tell a personal fact
    from a work one, so a restricted caller is served no fact on any path. No error, no
    response warning, one log line."""
    tx, ep = no_066
    r = _engine_on(tx)
    sink: list[str] = []
    with caplog.at_level(logging.INFO), warn_sink(sink):
        got = _every_path(r, ep)
        again = _every_path(r, ep)
    assert got == again == NOTHING
    assert sink == []
    lines = [m for m in caplog.messages if "schema 066" in m]
    assert len(lines) == 1, lines
    assert all(rec.levelno <= logging.WARNING for rec in caplog.records)


def test_before_066_the_label_rule_serves_what_the_missing_columns_withhold(no_066):
    """The control for the test above: with the columns the same data is served by the
    label rule, so it is the missing columns that withhold it."""
    tx, ep = no_066
    tx.rollback()
    got = _every_path(_engine_on(tx), ep)
    assert got["kg"] == SERVED
    assert got["pairs"] == ["f:o-ok"]


def test_before_066_restricted_recall_serves_no_facts(no_066):
    tx, _ep_id = no_066
    restricted = SurfaceTrust(
        surface_id="dev-restricted", trust="restricted", allowed_projects=(WORK,), known=True
    )
    sink: list[str] = []
    with warn_sink(sink):
        out = _engine_on(tx).recall("widget", trust=restricted)
    assert out["facts"] == []
    assert "warnings" not in out and sink == []


def test_before_066_full_trust_is_unchanged(no_066):
    tx, _ep_id = no_066
    r = _engine_on(tx)
    facts, seeds = r._search_kg("widget", onehot(0), [], 20)
    assert _uuids(facts) == EVERY
    assert [s["uuid"] for s in seeds] == ["seed-ent"]
    assert len(r._fetch_superseded_pairs_pg(PRE_066_PAIRS, 10)) == 3
    assert FACT_AUDIENCE_PROBE.cached(r._db_url) is False


def test_the_schema_rolled_back_under_a_positive_probe_fails_closed_once(no_066, monkeypatch):
    """The probe said yes, then the columns went away: that call serves nothing (the
    statement names a missing column), the probe flips to missing, and every later call
    serves nothing too. A re-probe still finds them gone."""
    tx, _ep_id = no_066
    r = _engine_on(tx)
    FACT_AUDIENCE_PROBE._ok[r._db_url] = True
    sink: list[str] = []
    with warn_sink(sink):
        assert _restricted_kg(r) == set()
        assert FACT_AUDIENCE_PROBE.cached(r._db_url) is False
        assert _restricted_kg(r) == set()
        assert r._fetch_superseded_pairs_pg(PRE_066_PAIRS, 10, [WORK]) == []
        monkeypatch.setattr(FACT_AUDIENCE_PROBE, "reprobe_s", 0.0)
        assert _restricted_kg(r) == set()
    assert sink == []


def test_applying_066_under_a_running_engine_takes_effect_without_a_restart(no_066, monkeypatch):
    tx, _ep_id = no_066
    r = _engine_on(tx)
    assert _restricted_kg(r) == set()
    tx.rollback()  # the migration lands: the columns are there again
    assert _restricted_kg(r) == set(), "a negative probe is trusted until its TTL lapses"
    monkeypatch.setattr(FACT_AUDIENCE_PROBE, "reprobe_s", 0.0)
    assert _restricted_kg(r) == SERVED
    assert FACT_AUDIENCE_PROBE.cached(r._db_url) is True


def test_a_056_loss_is_told_apart_from_a_066_loss(db_url, caplog):
    """The restricted handler routes an UndefinedColumn by the column it names."""
    r = Recall(db_url, "")
    allowed: list[str] | None = [WORK]
    e056 = psycopg.errors.UndefinedColumn('column "source_projects" does not exist')
    e066 = psycopg.errors.UndefinedColumn("column o.audience does not exist")
    with caplog.at_level(logging.WARNING):
        assert r._kg_scope_error(allowed, e066)
        assert FACT_AUDIENCE_PROBE.cached(r._db_url) is False
        assert r._kg_scope_ok is False and r._kg_scope_probed_at is None
        assert r._kg_scope_error(allowed, e056)
        assert r._kg_scope_probed_at is not None
        assert not r._kg_scope_error(None, e066), "full trust never names these columns"
        assert not r._kg_scope_error(allowed, ValueError("x"))


@pytest.fixture()
def broken_probe(monkeypatch):
    """The 066 probe raises (a stale pooled connection, a statement timeout) on a database
    that HAS the columns. The returned ``heal()`` puts the real statement back."""
    real = FACT_AUDIENCE_PROBE._sql
    monkeypatch.setattr(FACT_AUDIENCE_PROBE, "_sql", "SELECT 1 FROM a_table_that_is_not_there")
    return lambda: monkeypatch.setattr(FACT_AUDIENCE_PROBE, "_sql", real)


def test_a_probe_error_on_a_066_database_serves_a_restricted_read_nothing_then_recovers(
    kg, cutover, db_url, broken_probe
):
    """The probe could not tell, which is not "the columns are absent": falling back to the
    provenance-only rule would serve a personal fact and a post-cutover unlabelled one on
    every restricted path. Each fails closed, full trust is untouched, and the failure is
    not cached, so the next read (no TTL wait) is on the label rule again."""
    _seed_label_matrix(kg, cutover)
    ep = _ep(kg)
    _pair(kg, "ok", now="work-safe", old="work-safe", ep=ep)
    _pair(kg, "pers", now="personal", old="personal", ep=ep)
    _pair(kg, "new", now=None, old=None, ep=ep)
    r = Recall(db_url, "")
    sink: list[str] = []
    with warn_sink(sink):
        blind = _every_path(r, ep)
        facts, _ = r._search_kg("widget", onehot(0), [], 20)
        assert blind == NOTHING
        assert _uuids(facts) == EVERY, "full trust never asks the label question"
        assert FACT_AUDIENCE_PROBE.cached(r._db_url) is None, "a failed probe is not cached"
        broken_probe()
        healed = _every_path(r, ep)
    assert healed["kg"] == SERVED
    assert healed["pairs"] == ["f:o-ok"]
    assert healed["surface"] == {"n-ok"}
    assert healed["overlay"] == {ep: ["now ok"]}
    assert sink == []


def test_a_probe_that_says_no_serves_a_restricted_read_nothing_and_full_trust_everything(
    kg, cutover, db_url
):
    """The columns are there, but the probe's answer is "not applied": a restricted read
    must not fall back to the provenance-only rule, which would serve the personal and the
    post-cutover unlabelled facts too. Full trust never asks, so it is served as before."""
    _seed_label_matrix(kg, cutover)
    ep = _ep(kg)
    _pair(kg, "ok", now="work-safe", old="work-safe", ep=ep)
    _pair(kg, "pers", now="personal", old="personal", ep=ep)
    r = _recall_engine(db_url)
    FACT_AUDIENCE_PROBE.missing(r._db_url)
    restricted = register_restricted(kg, [WORK])
    full = register_full(kg)
    sink: list[str] = []
    with warn_sink(sink):
        assert _every_path(r, ep) == NOTHING
        out = r.recall("widget", surface=restricted)
        facts, seeds = r._search_kg("widget", onehot(0), [], 20)
        out_full = r.recall("widget", surface=full)
    assert FACT_AUDIENCE_PROBE.cached(r._db_url) is False
    assert out["facts"] == []
    assert sink == []
    assert _uuids(facts) == EVERY
    assert [s["uuid"] for s in seeds] == ["seed-ent"]
    assert EVERY <= {f["id"].removeprefix("f:") for f in out_full["facts"]}


def test_an_unrelated_missing_column_fails_the_call_closed_and_leaves_the_probes_alone(
    db_url,
):
    """The routing reads the column the error names, not the statement it quotes: this
    statement mentions ``audience`` but the missing column is another."""
    with psycopg.connect(db_url) as c:
        with pytest.raises(psycopg.errors.UndefinedColumn) as raised:
            c.execute("SELECT audience, shoe_size FROM kg_relationships")
        c.rollback()
    e = raised.value
    assert "audience" in str(e) and "shoe_size" in (e.diag.message_primary or "")
    r = Recall(db_url, "")
    assert r._kg_scope_error([WORK], e), "that statement is failed closed all the same"
    assert FACT_AUDIENCE_PROBE.cached(r._db_url) is None
    assert r._kg_scope_probed_at is None


def test_the_probe_sees_all_four_columns(db_url):
    with psycopg.connect(db_url) as c:
        assert FACT_AUDIENCE_PROBE.ready(c, "real") is True
        c.execute("ALTER TABLE kg_entities DROP COLUMN is_self CASCADE")
        FACT_AUDIENCE_PROBE.reset()
        assert FACT_AUDIENCE_PROBE.ready(c, "real") is False, "three of four is not applied"
        c.rollback()
