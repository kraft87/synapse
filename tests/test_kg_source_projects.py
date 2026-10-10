"""Provenance-scoped KG facts for restricted surfaces (schema 056).

A restricted surface may be served a KG fact only when the fact's source-project set is
KNOWN, NON-EMPTY, and a SUBSET of the surface's allowlist. ``source_projects`` caches that
set, maintained by triggers so every writer is covered; NULL means unknown and is never
served restricted.

Sections:
  * the trigger computes the set on every write path, including the real extractor
    writer, and keeps it honest when a source episode is relabelled or deleted
  * the backfill computes existing rows and is a no-op on a re-run
  * restricted recall serves only fully covered facts, on every KG leg and extra
  * full trust is unchanged, down to the SQL text
  * deploy order: with the column missing (code shipped, 056 not applied yet) the
    restricted KG paths serve nothing and full trust keeps working
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import psycopg
import pytest
from psycopg.rows import dict_row

from ingestion import schema_check
from ingestion.kg_pg_write import KGPostgresWriter
from ingestion.surfaces import FULL_TRUST, SurfaceTrust
from mcp_server import kg_pg
from mcp_server import recall_sources as recall_sources_mod
from mcp_server.recall import Recall
from mcp_server.recall_warnings import warn_sink
from tests.helpers.embed import GROUP, onehot
from tests.helpers.surfaces import clear_surfaces, register_full, register_restricted

_SCHEMA_056 = Path(__file__).resolve().parent.parent / "schema" / "056_kg_source_projects.sql"

WORK = "work-proj"
WORK2 = "work-proj-2"
PERSONAL = "personal-proj"


@pytest.fixture()
def kg(conn):
    """A clean episodes + KG slate. KG first: truncating episodes fires 056's truncate
    trigger, which only has work to do while facts still exist."""

    def _wipe() -> None:
        conn.execute("TRUNCATE kg_relationships, kg_entities RESTART IDENTITY CASCADE")
        conn.execute("TRUNCATE episodes RESTART IDENTITY CASCADE")
        clear_surfaces(conn)

    _wipe()
    yield conn
    _wipe()


def _ep(conn, project: str | None) -> int:
    return conn.execute(
        "INSERT INTO episodes (session_id, sequence, project, content) "
        "VALUES (%s, 1, %s, 'a turn') RETURNING id",
        (f"ksp-{uuid.uuid4().hex[:10]}", project),
    ).fetchone()[0]


def _fact(
    conn,
    uid: str,
    fact: str,
    episodes: Any,
    *,
    emb: int | None = 0,
    src: str = "ent-x",
    tgt: str = "ent-y",
    t_invalid: str | None = None,
    invalidated_by: str | None = None,
) -> None:
    """Insert one edge. ``episodes`` is passed as raw JSON text (or None) so malformed
    shapes can be written exactly as a careless writer would."""
    conn.execute(
        "INSERT INTO kg_relationships (uuid, owner_id, group_id, src_uuid, tgt_uuid, name, "
        "  fact, fact_embedding, episodes, t_valid, t_invalid, invalidated_by) "
        "VALUES (%s, 'default', %s, %s, %s, 'RELATES', %s, %s::vector, %s::jsonb, "
        "        '2026-01-01T00:00:00+00:00', %s, %s)",
        (
            uid,
            GROUP,
            src,
            tgt,
            fact,
            None if emb is None else kg_pg._vec_literal(onehot(emb)),
            episodes,
            t_invalid,
            invalidated_by,
        ),
    )


def _sp(conn, uid: str) -> list[str] | None:
    return conn.execute(
        "SELECT source_projects FROM kg_relationships WHERE uuid = %s", (uid,)
    ).fetchone()[0]


def _psql_apply_056(db_url: str) -> None:
    if shutil.which("psql") is None:  # pragma: no cover - CI installs the client
        pytest.skip("psql not installed")
    subprocess.run(
        ["psql", db_url, "-q", "-v", "ON_ERROR_STOP=1", "-f", str(_SCHEMA_056)],
        check=True,
        capture_output=True,
    )


# ---------------------------------------------------------------------------
# The trigger: every write path computes the set
# ---------------------------------------------------------------------------


def test_insert_computes_a_sorted_distinct_project_set(kg):
    w1, w2, w3, p = _ep(kg, WORK), _ep(kg, WORK), _ep(kg, WORK2), _ep(kg, PERSONAL)
    _fact(kg, "one", "f", f"[{w1}]")
    _fact(kg, "same", "f", f"[{w1}, {w2}]")
    _fact(kg, "two", "f", f"[{p}, {w3}, {w1}]")
    assert _sp(kg, "one") == [WORK]
    assert _sp(kg, "same") == [WORK]
    assert _sp(kg, "two") == sorted([WORK, WORK2, PERSONAL])


def test_string_episode_ids_resolve_like_integers(kg):
    """Both id shapes exist in stored provenance (the dashboard's delete route matches
    either), so both must resolve."""
    w = _ep(kg, WORK)
    _fact(kg, "str", "f", f'["{w}"]')
    assert _sp(kg, "str") == [WORK]


def test_extractor_insert_then_reinforce_with_a_foreign_source_goes_mixed(kg):
    """The real write path: KGPostgresWriter.create_edges inserts, reinforce_edges unions
    a new source episode into ``episodes``. A second project arriving makes the fact
    mixed, and a work-only allowlist must stop seeing it."""
    w, p = _ep(kg, WORK), _ep(kg, PERSONAL)
    writer = KGPostgresWriter()
    try:
        writer.create_edges(
            [
                {
                    "edge_uuid": "ext-1",
                    "src": "ent-x",
                    "tgt": "ent-y",
                    "name": "USES",
                    "fact": "the widget uses a queue",
                    "episodes": [w],
                    "created_at": "2026-01-01T00:00:00+00:00",
                }
            ],
            GROUP,
        )
        assert _sp(kg, "ext-1") == [WORK]
        writer.reinforce_edges([("ext-1", [p])], GROUP)
    finally:
        writer._reset()
    assert _sp(kg, "ext-1") == sorted([WORK, PERSONAL])
    servable = kg.execute(
        f"SELECT count(*) FROM kg_relationships WHERE uuid = 'ext-1' AND {kg_pg.scope_predicate()}",
        ([WORK],),
    ).fetchone()[0]
    assert servable == 0


def test_a_missing_source_episode_is_unknown(kg):
    w = _ep(kg, WORK)
    _fact(kg, "gone", "f", f"[{w}, 987654321]")
    assert _sp(kg, "gone") is None


def test_a_source_episode_without_a_project_is_unknown(kg):
    w, n = _ep(kg, WORK), _ep(kg, None)
    _fact(kg, "nullproj", "f", f"[{w}, {n}]")
    assert _sp(kg, "nullproj") is None


@pytest.mark.parametrize(
    "episodes",
    [None, "[]", '["not-an-id"]', "[1.5]", "[-3]", '{"id": 1}', "7", "[null]"],
    ids=["null", "empty", "junk-string", "float", "negative", "object", "scalar", "null-elem"],
)
def test_empty_null_or_malformed_provenance_is_unknown(kg, episodes):
    """Web-artifact facts carry NULL episodes; anything that does not name episode ids
    names nothing provable."""
    _ep(kg, WORK)  # id 1 exists, so a lenient parser would have something to match
    _fact(kg, "odd", "f", episodes)
    assert _sp(kg, "odd") is None


def test_source_projects_cannot_be_written_directly(kg):
    """Any UPDATE naming the column recomputes it, so a stray write cannot widen what a
    restricted surface is served."""
    _fact(kg, "web", "f", None)
    kg.execute("UPDATE kg_relationships SET source_projects = %s WHERE uuid = 'web'", ([WORK],))
    assert _sp(kg, "web") is None


def test_relabelling_a_source_episode_recomputes_the_fact(kg):
    w = _ep(kg, WORK)
    _fact(kg, "relabel", "f", f"[{w}]")
    kg.execute("UPDATE episodes SET project = %s WHERE id = %s", (PERSONAL, w))
    assert _sp(kg, "relabel") == [PERSONAL]
    kg.execute("UPDATE episodes SET project = NULL WHERE id = %s", (w,))
    assert _sp(kg, "relabel") is None


def test_the_ingest_upsert_relabel_path_recomputes(kg, db_url):
    """upsert_episode COALESCEs a new project onto an existing turn on re-ingest."""
    from ingestion.db import Database
    from ingestion.models import Episode

    db = Database(db_url)
    try:
        ep = Episode(session_id="ksp-upsert", sequence=1, project=WORK, content="x")
        eid = db.upsert_episode(ep)
        _fact(kg, "upsert", "f", f"[{eid}]")
        assert _sp(kg, "upsert") == [WORK]
        db.upsert_episode(Episode(session_id="ksp-upsert", sequence=1, project=WORK, content="y"))
        assert _sp(kg, "upsert") == [WORK]
        db.upsert_episode(
            Episode(session_id="ksp-upsert", sequence=1, project=PERSONAL, content="z")
        )
        assert _sp(kg, "upsert") == [PERSONAL]
    finally:
        db.close()


def test_deleting_a_source_episode_makes_the_fact_unknown(kg):
    w, w2 = _ep(kg, WORK), _ep(kg, WORK)
    _fact(kg, "del", "f", f"[{w}, {w2}]")
    _fact(kg, "untouched", "f", f"[{w2}]")
    kg.execute("DELETE FROM episodes WHERE id = %s", (w,))
    assert _sp(kg, "del") is None
    assert _sp(kg, "untouched") == [WORK]


def test_dashboard_episode_delete_unlinks_and_recomputes(kg, db_url):
    """The dashboard's hard-delete rewrites a shared fact's episodes to the survivors
    before deleting the turn; the trigger recomputes from what is left."""
    from mcp_server.dashboard_routes import _delete_episode

    w, p = _ep(kg, WORK), _ep(kg, PERSONAL)
    _fact(kg, "shared", "f", f"[{w}, {p}]")
    assert _sp(kg, "shared") == sorted([WORK, PERSONAL])
    assert _delete_episode(db_url, p) is not None
    assert _sp(kg, "shared") == [WORK]


# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------


def test_backfill_computes_existing_rows_and_reruns_as_a_no_op(kg, db_url):
    """Rows written before 056 have no value. Re-applying the migration file (exactly
    what apply_schema.sh runs) computes them, and a second run rewrites nothing."""
    w, p = _ep(kg, WORK), _ep(kg, PERSONAL)
    kg.execute("ALTER TABLE kg_relationships DISABLE TRIGGER kg_rel_source_projects")
    try:
        _fact(kg, "pre-work", "f", f"[{w}]")
        _fact(kg, "pre-mixed", "f", f"[{w}, {p}]")
        _fact(kg, "pre-gone", "f", "[987654321]")
        _fact(kg, "pre-web", "f", None)
        # a stale value a pre-trigger writer could have left behind
        kg.execute(
            "UPDATE kg_relationships SET source_projects = %s WHERE uuid = 'pre-gone'", ([WORK],)
        )
        assert _sp(kg, "pre-work") is None
    finally:
        kg.execute("ALTER TABLE kg_relationships ENABLE TRIGGER kg_rel_source_projects")

    _psql_apply_056(db_url)
    assert _sp(kg, "pre-work") == [WORK]
    assert _sp(kg, "pre-mixed") == sorted([WORK, PERSONAL])
    assert _sp(kg, "pre-gone") is None
    assert _sp(kg, "pre-web") is None

    def _versions() -> dict[str, str]:
        rows = kg.execute("SELECT uuid, xmin::text FROM kg_relationships").fetchall()
        return {r[0]: r[1] for r in rows}

    before = _versions()
    _psql_apply_056(db_url)
    assert _versions() == before, "a re-run must not rewrite any row"


# ---------------------------------------------------------------------------
# Restricted serving: only fully covered facts, on every leg
# ---------------------------------------------------------------------------


def _seed_kg(conn) -> dict[str, int]:
    """One fact per (leg, verdict). Facts with NULL embeddings never reach the vector
    leg; only hop facts touch the seed entity; only BM25 facts say "widget"."""
    w, w2, p = _ep(conn, WORK), _ep(conn, WORK2), _ep(conn, PERSONAL)
    conn.execute(
        "INSERT INTO kg_entities (uuid, owner_id, group_id, name, normalized_name, summary, "
        "  embedding) VALUES ('seed-ent', 'default', %s, 'Seed', 'seed', 'seed summary', "
        "  %s::vector)",
        (GROUP, kg_pg._vec_literal(onehot(0))),
    )
    # vector leg
    _fact(conn, "v-ok", "alpha beta", f"[{w}]", emb=0)
    _fact(conn, "v-mixed", "gamma delta", f"[{w}, {p}]", emb=0)
    _fact(conn, "v-web", "eta theta", None, emb=0)
    # BM25 leg
    _fact(conn, "b-ok", "widget gizmo", f"[{w}]", emb=None)
    _fact(conn, "b-personal", "widget thing", f"[{p}]", emb=None)
    _fact(conn, "b-two", "widget sprocket", f"[{w}, {w2}]", emb=None)
    # hop leg (from the seed entity)
    _fact(conn, "h-ok", "epsilon", f"[{w}]", emb=None, src="seed-ent")
    _fact(conn, "h-gone", "zeta", f"[{w}, 987654321]", emb=None, src="seed-ent")
    return {"w": w, "w2": w2, "p": p}


_ALL = {"v-ok", "v-mixed", "v-web", "b-ok", "b-personal", "b-two", "h-ok", "h-gone"}


def _uuids(facts: list[dict[str, Any]]) -> set[str]:
    return {f["_uuid"] for f in facts}


def test_restricted_kg_leg_serves_only_fully_covered_facts(kg, db_url):
    _seed_kg(kg)
    r = Recall(db_url, "")
    sink: list[str] = []
    with warn_sink(sink):
        facts, seeds = r._search_kg("widget", onehot(0), GROUP, [], 20, allowed_projects=[WORK])
    assert _uuids(facts) == {"v-ok", "b-ok", "h-ok"}
    assert seeds == [], "entity summaries are not provably in scope"
    assert sink == []

    facts2, _ = r._search_kg(
        "widget", onehot(0), GROUP, [], 20, allowed_projects=[WORK, WORK2, "unused-proj"]
    )
    assert _uuids(facts2) == {"v-ok", "b-ok", "h-ok", "b-two"}


def test_full_trust_kg_leg_still_serves_everything(kg, db_url):
    _seed_kg(kg)
    facts, seeds = Recall(db_url, "")._search_kg("widget", onehot(0), GROUP, [], 20)
    assert _uuids(facts) == _ALL
    assert [s["uuid"] for s in seeds] == ["seed-ent"]


def test_restricted_seed_degree_counts_only_servable_edges(kg, db_url):
    """A seed whose only live edges are out of scope must not consume a seed slot."""
    p = _ep(kg, PERSONAL)
    kg.execute(
        "INSERT INTO kg_entities (uuid, owner_id, group_id, name, normalized_name, embedding) "
        "VALUES ('lonely', 'default', %s, 'Lonely', 'lonely', %s::vector)",
        (GROUP, kg_pg._vec_literal(onehot(0))),
    )
    _fact(kg, "lonely-edge", "omega", f"[{p}]", emb=None, src="lonely")
    executed: list[tuple[str, Any]] = []

    class _Rec:
        def __init__(self, cur):
            self._cur = cur

        def execute(self, sql, params=None):
            executed.append((sql, params))
            return self._cur.execute(sql, params)

        def fetchall(self):
            return self._cur.fetchall()

    with psycopg.connect(db_url) as c, c.cursor() as cur:
        kg_pg.search_kg_postgres(_Rec(cur), "q", onehot(0), "default", GROUP, [], 5, [WORK])
    hop_queries = [s for s, _ in executed if "src_uuid = %s OR tgt_uuid = %s" in s]
    assert hop_queries == [], "no in-scope edge, so no seed, so no hop query"


def _supersession_fixture(conn) -> dict[str, int]:
    w, p = _ep(conn, WORK), _ep(conn, PERSONAL)
    inv = "2026-02-01T00:00:00+00:00"
    # pair 1: both in scope
    _fact(conn, "n-ok", "now ok", f"[{w}]", emb=None, src="s1", tgt="t1")
    _fact(
        conn,
        "o-ok",
        "old ok",
        f"[{w}]",
        emb=1,
        src="s1",
        tgt="t1",
        t_invalid=inv,
        invalidated_by="n-ok",
    )
    # pair 2: the displaced side is mixed
    _fact(conn, "n-2", "now two", f"[{w}]", emb=None, src="s2", tgt="t2")
    _fact(
        conn,
        "o-2",
        "old two",
        f"[{w}, {p}]",
        emb=1,
        src="s2",
        tgt="t2",
        t_invalid=inv,
        invalidated_by="n-2",
    )
    # pair 3: the current side is out of scope
    _fact(conn, "n-3", "now three", f"[{p}]", emb=None, src="s3", tgt="t3")
    _fact(
        conn,
        "o-3",
        "old three",
        f"[{w}]",
        emb=1,
        src="s3",
        tgt="t3",
        t_invalid=inv,
        invalidated_by="n-3",
    )
    return {"w": w, "p": p}


def test_superseded_pairs_need_both_sides_in_scope(kg, db_url):
    _supersession_fixture(kg)
    r = Recall(db_url, "")
    uuids = ["n-ok", "n-2", "n-3"]
    scoped = r._fetch_superseded_pairs_pg(GROUP, uuids, 10, allowed_projects=[WORK])
    assert [x["id"] for x in scoped] == ["f:o-ok"]
    full = r._fetch_superseded_pairs_pg(GROUP, uuids, 10)
    assert [x["id"] for x in full] == ["f:o-ok", "f:o-2", "f:o-3"]


def test_supersession_surface_needs_both_edges_in_scope(kg, db_url):
    _supersession_fixture(kg)
    r = Recall(db_url, "")
    scoped = r._surface_supersessions(onehot(1), GROUP, set(), cap=10, allowed_projects=[WORK])
    assert _uuids(scoped) == {"n-ok"}
    assert _uuids(r._surface_supersessions(onehot(1), GROUP, set(), cap=10)) == {
        "n-ok",
        "n-2",
        "n-3",
    }


def test_episode_overlay_needs_both_edges_in_scope(kg, db_url):
    ids = _supersession_fixture(kg)
    r = Recall(db_url, "")
    scoped = r._episode_supersessions([ids["w"]], GROUP, allowed_projects=[WORK])
    assert scoped == {ids["w"]: ["now ok"]}
    full = r._episode_supersessions([ids["w"]], GROUP)
    assert sorted(full[ids["w"]]) == ["now ok", "now three", "now two"]


class _Emb:
    def embed(self, texts, task):
        return [onehot(0) for _ in texts]


def _recall_engine(db_url: str) -> Recall:
    """Real KG legs; everything that would reach a network or rank episodes is stubbed."""
    r = Recall(db_url, "")
    r._ensure_embedder = lambda: _Emb()
    r._rerank_pool_scored = lambda q, pool: [(i, 0.9) for i in range(len(pool))]
    r._compact_to_passages = lambda q, eps, n, **_k: [{"id": e["id"], "text": "t"} for e in eps]
    r._search_web_reranked = lambda q, emb: []
    r._search_notes = lambda *a, **k: []
    r._increment_retrieval_counts = lambda ids: None
    r._increment_fact_retrieval_counts = lambda *a, **k: None
    r._record_metrics = lambda m: None
    return r


def test_restricted_recall_serves_only_fully_covered_facts(kg, db_url):
    _seed_kg(kg)
    restricted = register_restricted(kg, [WORK])
    full = register_full(kg)

    out = _recall_engine(db_url).recall("widget", surface=restricted)
    served = {f["id"].removeprefix("f:") for f in out["facts"]}
    assert served == {"v-ok", "b-ok", "h-ok"}
    assert "warnings" not in out

    out_full = _recall_engine(db_url).recall("widget", surface=full)
    assert {f["id"].removeprefix("f:") for f in out_full["facts"]} == _ALL


def test_restricted_recall_scopes_the_supersession_extras(kg, db_url):
    """The supersession surface (query near a retired fact pulls in its successor) and
    the episode overlay (a served turn is annotated with the fact that superseded it)
    both serve fact text, so recall() must hand them the allowlist too. Two retired
    facts sit on the query axis and cite a served work turn; one successor is work-only,
    the other comes from a personal turn."""
    w = _ep(kg, WORK)
    kg.execute("UPDATE episodes SET content = 'widget design review' WHERE id = %s", (w,))
    p = _ep(kg, PERSONAL)
    inv = "2026-02-01T00:00:00+00:00"
    _fact(kg, "succ-work", "the current work answer", f"[{w}]", emb=None, src="a1")
    _fact(kg, "succ-personal", "the current personal answer", f"[{p}]", emb=None, src="a2")
    for i, succ in enumerate(("succ-work", "succ-personal")):
        _fact(kg, f"old-{i}", f"retired {i}", f"[{w}]", emb=0, t_invalid=inv, invalidated_by=succ)
    restricted = register_restricted(kg, [WORK])
    full = register_full(kg)

    out = _recall_engine(db_url).recall("widget", surface=restricted)
    assert out.get("episodes"), "the work turn is served, so the overlay runs"
    assert "succ-work" in {f["id"].removeprefix("f:") for f in out["facts"]}
    assert "current personal answer" not in str(out)

    out_full = _recall_engine(db_url).recall("widget", surface=full)
    assert "current personal answer" in str(out_full)


# ---------------------------------------------------------------------------
# Full trust: unchanged, down to the statements it sends
# ---------------------------------------------------------------------------


class _RecordingConn:
    """Records every statement; answers with no rows."""

    def __init__(self) -> None:
        self.sql: list[str] = []

    def execute(self, sql, params=None):
        self.sql.append(sql)
        return self

    def fetchall(self):
        return []

    def fetchone(self):
        return None


def test_full_trust_sql_never_mentions_the_provenance_column():
    """Full trust must not depend on schema 056 at all: no statement it sends reads the
    column, and the readiness probe never runs."""
    cur = _RecordingConn()
    kg_pg.search_kg_postgres(cur, "widget q", onehot(0), "default", GROUP, ["x"], 5)
    r = Recall("postgresql://unused/none", "")
    conn = _RecordingConn()
    r._ensure_pg = lambda: conn
    r._fetch_superseded_pairs_pg(GROUP, ["u1"], 2)
    r._surface_supersessions(onehot(0), GROUP, set())
    r._episode_supersessions([1, 2], GROUP)
    statements = cur.sql + conn.sql
    assert len(statements) >= 6
    assert not any("source_projects" in s or "pg_attribute" in s for s in statements)


def test_restricted_sql_carries_the_rule_on_every_fact_read():
    cur = _RecordingConn()
    kg_pg.search_kg_postgres(cur, "widget q", onehot(0), "default", GROUP, [], 5, [WORK])
    # vector, BM25, seed/degree (no seeds come back from the recorder, so no hop query)
    assert len(cur.sql) == 3
    assert all("source_projects <@ %s::text[]" in s for s in cur.sql)


# ---------------------------------------------------------------------------
# Deploy order: code live, schema 056 not applied yet
# ---------------------------------------------------------------------------


@pytest.fixture()
def no_column(kg, db_url):
    """A connection on which kg_relationships has NO source_projects column: the column
    (and with it the 056 trigger and index) is dropped inside a transaction that is
    rolled back afterwards, so the shared test database is never left altered. Data is
    seeded first, through the committed fixture connection."""
    ids = _seed_kg(kg)
    _supersession_fixture(kg)
    tx = psycopg.connect(db_url, row_factory=dict_row)
    try:
        tx.execute("ALTER TABLE kg_relationships DROP COLUMN source_projects CASCADE")
        yield tx, ids
    finally:
        tx.rollback()
        tx.close()


def _engine_on(tx) -> Recall:
    """recall() on ONE connection: a single leg worker keeps every statement serial on
    the transaction that hides the column."""
    r = _recall_engine("postgresql://unused/none")
    r._ensure_pg = lambda: tx
    r._leg_executor = ThreadPoolExecutor(max_workers=1)
    return r


def test_restricted_kg_paths_serve_nothing_without_the_column(no_column, caplog):
    tx, ids = no_column
    r = _engine_on(tx)
    allowed = [WORK]
    sink: list[str] = []
    with caplog.at_level(logging.WARNING), warn_sink(sink):
        assert r._search_kg("widget", onehot(0), GROUP, [], 20, allowed_projects=allowed) == (
            [],
            [],
        )
        assert r._fetch_superseded_pairs_pg(GROUP, ["n-ok"], 5, allowed_projects=allowed) == []
        assert r._surface_supersessions(onehot(1), GROUP, set(), allowed_projects=allowed) == []
        assert r._episode_supersessions([ids["w"]], GROUP, allowed_projects=allowed) == {}
    assert sink == [], "the pre-056 restricted skip added no response warning"
    missing = [m for m in caplog.messages if "source_projects is missing" in m]
    assert len(missing) == 1, "one log warning, not one per call"
    assert all(rec.levelno <= logging.WARNING for rec in caplog.records)


def test_a_column_lost_after_the_probe_cached_it_still_fails_closed(no_column, caplog):
    """Belt and braces: if the probe already said yes, the UndefinedColumn itself is
    caught on the restricted path."""
    tx, _ids = no_column
    r = _engine_on(tx)
    r._kg_scope_ok = True
    sink: list[str] = []
    with caplog.at_level(logging.WARNING), warn_sink(sink):
        out = r._search_kg("widget", onehot(0), GROUP, [], 20, allowed_projects=[WORK])
    assert out == ([], [])
    assert sink == []
    assert r._kg_scope_ok is False


def test_full_trust_keeps_working_without_the_column(no_column):
    tx, _ids = no_column
    r = _engine_on(tx)
    facts, _ = r._search_kg("widget", onehot(0), GROUP, [], 20)
    assert _uuids(facts) == _ALL
    pairs = r._fetch_superseded_pairs_pg(GROUP, ["n-ok", "n-2", "n-3"], 10)
    assert len(pairs) == 3


def test_recall_without_the_column_matches_the_pre_056_restricted_serve(no_column):
    """End to end on the column-less database: a restricted recall serves no facts and
    no superseded pairs, with no warnings key and no error; full trust serves facts."""
    tx, _ids = no_column
    restricted = SurfaceTrust(
        surface_id="dev-restricted", trust="restricted", allowed_projects=(WORK,), known=True
    )
    out = _engine_on(tx).recall("widget", trust=restricted)
    assert out["facts"] == []
    assert "superseded_facts" not in out
    assert "warnings" not in out

    out_full = _engine_on(tx).recall("widget", trust=FULL_TRUST)
    assert {f["id"].removeprefix("f:") for f in out_full["facts"]} >= _ALL


def test_applying_056_under_a_running_engine_takes_effect_without_a_restart(
    kg, db_url, monkeypatch
):
    """A negative probe is re-probed after the TTL, so once the column exists the same
    engine starts serving restricted facts."""
    _seed_kg(kg)
    r = Recall(db_url, "")
    r._kg_scope_missing()  # as if a probe had just found no column
    assert r._kg_scope_ready() is False  # inside the TTL: no re-probe yet
    monkeypatch.setattr(recall_sources_mod, "_KG_SCOPE_REPROBE_S", 0.0)
    assert r._kg_scope_ready() is True
    facts, _ = r._search_kg("widget", onehot(0), GROUP, [], 20, allowed_projects=[WORK])
    assert _uuids(facts) == {"v-ok", "b-ok", "h-ok"}


def test_schema_056_is_registered_for_apply_schema():
    script = (Path(__file__).resolve().parent.parent / "scripts" / "apply_schema.sh").read_text()
    assert "056_kg_source_projects.sql" in script
    assert os.path.exists(_SCHEMA_056)


def test_schema_056_does_not_block_boot_before_it_is_applied():
    """The image lands (watchtower) before 056 is applied by hand. The boot guard must
    let a database stamped 055 start, or the deploy-order fallback above never runs."""
    assert schema_check.is_optional_migration(_SCHEMA_056)
