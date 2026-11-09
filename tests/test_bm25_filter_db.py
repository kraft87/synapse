"""Every BM25 statement hands pg_search its filter as ONE heap filter (ingestion.bm25_sql).

pg_search re-runs the match and checks the heap once per top-level WHERE conjunct the index
does not cover, and turns a WHERE-level OR into "match everything" sub-queries. Each
statement below is captured as the code sends it, over a seeded corpus whose rows differ on
every filter, and checked two ways:

* its plan has exactly one heap filter and no "match everything" sub-query;
* it serves the rows, in the order, that the same filters as separate conjuncts serve (the
  statement before the fold), and it serves some.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest

from ingestion.bm25_sql import bm25_filter
from ingestion.kg_pg_read import KGPostgresReader
from mcp_server import dashboard_routes, kg_pg
from mcp_server.recall import Recall
from mcp_server.recall_warnings import warn_sink
from tests.helpers.embed import GROUP, onehot
from tests.helpers.surfaces import clear_surfaces

#: One per heap filter: the clause's SQL text is a string value under this key. Plans are
#: compared with spaces removed.
_CLAUSE = '"heap_filter":"'
_MATCH_ALL = '"indexed_query":"all"'
_FOLD, _FOLD_END = " AND CASE WHEN (", ") THEN 1 END = 1"


@pytest.fixture()
def corpus(conn):
    """Episodes, facts and timeline events that all match "widget", each a different length
    (so no two share a score) and spread over every value the filters below test."""

    def _wipe() -> None:
        conn.execute("TRUNCATE kg_relationships, kg_entities RESTART IDENTITY CASCADE")
        conn.execute("TRUNCATE episodes, timeline_events RESTART IDENTITY CASCADE")
        clear_surfaces(conn)

    _wipe()
    eps = []
    for i in range(30):
        eps.append(
            conn.execute(
                "INSERT INTO episodes (session_id, sequence, project, source, surface_id, "
                "  content) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                (
                    ["s", "t"][i % 2],
                    i,
                    ["p", "work", "other", "", None][i % 5],
                    ["s", "chat", None, ""][i % 4],
                    ["dev-a", None, "dev-b"][i % 3],
                    "widget" + " gear" * i,
                ),
            ).fetchone()[0]
        )
    work, other = eps[1], eps[2]  # projects 'work' and 'other'
    for i in range(24):
        conn.execute(
            "INSERT INTO kg_relationships (uuid, owner_id, group_id, src_uuid, tgt_uuid, name, "
            "  fact, episodes, t_valid, t_invalid, audience) VALUES (%s, %s, %s, 'ent-x', "
            "  'ent-y', 'RELATES', %s, %s::jsonb, '2026-01-01T00:00:00+00:00', %s, %s)",
            (
                f"f-{i}",
                ["default", "default", "tenant2"][i % 3],
                [GROUP, "personal"][i % 2],
                "widget" + " cog" * i,
                f"[{[work, work, work, work, other, work, other][i % 7]}"
                + (f", {other}]" if i % 7 == 5 else "]"),
                [None, None, None, "2026-02-01T00:00:00+00:00"][i % 4],
                ["work-safe", "personal", None, "work-safe", None][i % 5],
            ),
        )
    now = datetime(2026, 1, 1, tzinfo=UTC)
    for i in range(16):
        conn.execute(
            "INSERT INTO timeline_events (t_valid, fact, source, source_ref, project, domain) "
            "VALUES (%s, %s, 'chat', %s, %s, %s)",
            (
                now,
                "widget" + " tick" * i,
                f"bm25-{uuid.uuid4().hex[:8]}",
                ["p", "", None, "q"][i % 4],
                [GROUP, "personal"][i % 2],
            ),
        )
    yield conn
    _wipe()


@pytest.fixture()
def sent(monkeypatch) -> list[tuple[str, Any]]:
    """Every BM25 statement (sql, params) any cursor sends while the test runs."""
    seen: list[tuple[str, Any]] = []
    real = psycopg.Cursor.execute

    def execute(self, query, params=None, **kw):
        if "paradedb.match" in str(query):
            seen.append((str(query), params))
        return real(self, query, params, **kw)

    monkeypatch.setattr(psycopg.Cursor, "execute", execute)
    return seen


def _session(c: psycopg.Connection) -> None:
    """Recall's planner settings, which pin the BM25 scan on these small tables."""
    c.execute("SET enable_seqscan = off")
    c.execute("SET max_parallel_workers_per_gather = 0")


def _tantivy_query(c: psycopg.Connection, sql: str, params: Any) -> str:
    plan = "\n".join(r[0] for r in c.execute("EXPLAIN (VERBOSE, COSTS OFF) " + sql, params))
    lines = [ln for ln in plan.splitlines() if "Tantivy Query:" in ln]
    assert lines, f"no BM25 scan in the plan:\n{plan}"
    return "\n".join(lines).replace(" ", "")


def _plan_format_known(c: psycopg.Connection) -> bool:
    """This pg_search prints one heap_filter clause per filter, the format checked here."""
    tq = _tantivy_query(
        c,
        "SELECT id FROM episodes WHERE id @@@ paradedb.match('content', 'widget') "
        "AND project = 'p' AND session_id = 's'",
        None,
    )
    return tq.count(_CLAUSE) == 2


def _unfolded(sql: str) -> str:
    """The same statement with its filters as separate conjuncts: the shape before the fold
    (Postgres flattens ``AND (a AND b)`` into the conjuncts ``a`` and ``b``)."""
    assert sql.count(_FOLD) == 1 and sql.count(_FOLD_END) == 1, sql
    return sql.replace(_FOLD, " AND (").replace(_FOLD_END, ")")


def _check(db_url: str, sent: list[tuple[str, Any]], n: int) -> None:
    statements = list(sent)  # the statements run below are BM25 statements too
    assert len(statements) == n, [s for s, _ in statements]
    served = 0
    with psycopg.connect(db_url, autocommit=True) as c:
        _session(c)
        if not _plan_format_known(c):
            pytest.skip("this pg_search prints BM25 plans in a format this test does not read")
        for sql, params in statements:
            tq = _tantivy_query(c, sql, params)
            assert tq.count(_CLAUSE) == 1, (sql, tq)
            assert _MATCH_ALL not in tq, (sql, tq)
            got = [r[0] for r in c.execute(sql, params).fetchall()]
            assert got == [r[0] for r in c.execute(_unfolded(sql), params).fetchall()], sql
            served += bool(got and got != [0])  # a count statement serves one row
    assert served, "no statement served a row: the comparison above proved nothing"


@pytest.mark.parametrize("restricted", [False, True])
def test_the_requested_groups_fact_leg(corpus, sent, db_url, restricted):
    """Four filters, eight with a restricted device's provenance and label rule."""
    with psycopg.connect(db_url, autocommit=True) as c, c.cursor() as cur:
        _session(c)
        kg_pg.search_kg_postgres(
            cur,
            "widget",
            onehot(0),
            "default",
            GROUP,
            [],
            10,
            allowed_projects=["work"] if restricted else None,
        )
    _check(db_url, sent, 1)


def test_the_other_groups_fact_leg(corpus, sent, db_url):
    with psycopg.connect(db_url, autocommit=True) as c, c.cursor() as cur:
        _session(c)
        kg_pg.cross_bm25_rows(cur, "widget", "default", ["personal"], 3)
    _check(db_url, sent, 1)


@pytest.mark.parametrize(
    "kw",
    [
        {"project": "p"},
        {"project": "p", "session_id": "s"},
        {"allowed_projects": ["work"]},
        # A credential-bound restricted device: its own-surface rule is an OR.
        {"allowed_projects": ["work"], "own_surface": "dev-a"},
        {"project": "work", "allowed_projects": ["work"], "own_surface": "dev-a"},
    ],
)
def test_the_episode_leg(corpus, sent, db_url, kw):
    sink: list[str] = []
    with warn_sink(sink):  # the leg logs a failure and serves nothing: fail on it here
        rows = Recall(db_url, "")._bm25_table(
            "episodes", "widget", kw.pop("project", None), 30, "episode", **kw
        )
    assert sink == [] and rows
    _check(db_url, sent, 1)


def test_the_extraction_candidate_leg(corpus, sent, db_url, monkeypatch):
    monkeypatch.setenv("SYNAPSE_DB_URL", db_url)
    assert KGPostgresReader().find_edges_by_fulltext("widget", GROUP)
    _check(db_url, sent, 1)


@pytest.mark.parametrize("type_", ["episodes", "facts", "events"])
@pytest.mark.parametrize("project", ["p", "untagged"])
def test_the_dashboard_search(corpus, sent, db_url, type_, project):
    """Its three counts and the requested tab's hits; 'untagged' is a NULL or empty project."""
    out = dashboard_routes._search(db_url, "widget", type_, 0, 10, project, GROUP, "s")
    assert out["hits"]
    _check(db_url, sent, 4)


def test_a_filter_that_is_null_or_false_drops_the_row_as_the_plain_where_does(conn):
    """The CASE has no ELSE: a NULL or false filter yields NULL, and NULL = 1 is not true."""
    rows = conn.execute(
        "SELECT x FROM (VALUES (1, true, true), (2, false, true), (3, NULL, true), "
        "(4, true, NULL)) v(x, a, b) WHERE true" + bm25_filter(["a", "b"]) + " ORDER BY x"
    ).fetchall()
    assert rows == [(1,)]
