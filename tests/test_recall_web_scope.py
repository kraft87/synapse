"""The web bucket for a restricted caller: only the pages its own projects (and its own
credential) fetched, through the same fail-closed scoping the episode legs use.

Captured pages carry the project and surface of the turn that fetched them in
``web_artifacts.metadata`` (``ingestion.capture.store``); a page stored before that carries
neither and is served to full trust only.
"""

from __future__ import annotations

import pytest
from psycopg.types.json import Json

from ingestion.surfaces import web_scope_sql
from mcp_server.kg_pg import _vec_literal
from mcp_server.recall import Recall

pytestmark = pytest.mark.db

DIMS = 2048

#: (project, surface) stamps: two allowlisted projects' pages, a page one device uploaded
#: from a project nobody allowed, and a page from before capture stamped anything.
PAGES = {
    "alpha": {"project": "alpha", "surface_id": None},
    "beta": {"project": "beta", "surface_id": "dev-1"},
    "gamma": {"project": "gamma", "surface_id": "dev-2"},
    "legacy": {},
}


def _axis(i: int) -> list[float]:
    v = [0.0] * DIMS
    v[i] = 1.0
    return v


@pytest.fixture()
def pages(conn):
    conn.execute("TRUNCATE web_artifacts RESTART IDENTITY CASCADE")
    for i, (name, meta) in enumerate(PAGES.items()):
        aid = conn.execute(
            "INSERT INTO web_artifacts (kind, tool_name, tool_use_id, url, title, fetched_at, "
            "  metadata) VALUES ('web_scrape', 'WebFetch', %s, %s, %s, now(), %s) RETURNING id",
            (f"tu-{name}", f"https://example.org/{name}", name, Json(meta)),
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO web_chunks (web_artifact_id, idx, content, char_start, char_end, "
            "  embedding, is_embedded) VALUES (%s, 0, %s, 0, 1, %s, TRUE)",
            (aid, f"quokka habitat notes from {name}", _vec_literal(_axis(i))),
        )
    yield
    conn.execute("TRUNCATE web_artifacts RESTART IDENTITY CASCADE")


def _served(db_url: str, **scope) -> tuple[set[str], set[str]]:
    r = Recall(db_url, "")
    q = [1.0] * DIMS
    bm25 = {row["title"] for row in r._search_bm25_web("quokka habitat", 10, **scope)}
    vector = {row["title"] for row in r._search_vector_web(q, 10, **scope)}
    return bm25, vector


@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        ({}, set(PAGES)),  # full trust: every page, the unstamped one too
        ({"allowed_projects": ["alpha"]}, {"alpha"}),
        ({"allowed_projects": ["alpha", "beta"]}, {"alpha", "beta"}),
        # the caller's own credential reads back what it uploaded, whatever the project
        ({"allowed_projects": ["alpha"], "own_surface": "dev-1"}, {"alpha", "beta"}),
        ({"allowed_projects": [], "own_surface": "dev-1"}, {"beta"}),
        # an unknown surface has an empty allowlist and no own surface: nothing, the
        # unstamped page included
        ({"allowed_projects": []}, set()),
    ],
)
def test_both_web_legs_serve_a_restricted_caller_only_its_pages(pages, db_url, scope, expected):
    bm25, vector = _served(db_url, **scope)
    assert bm25 == expected
    assert vector == expected


def test_the_scope_is_the_episode_scope_on_the_pages_provenance():
    assert web_scope_sql(None, "dev-1") is None  # full trust: no predicate
    assert web_scope_sql(["a"]) == ("a.metadata->>'project' = ANY(%s)", [["a"]])
    sql, params = web_scope_sql(["a"], "dev-1") or ("", [])
    assert "a.metadata->>'surface_id' = %s" in sql and params == [["a"], "dev-1"]
