"""The web vector leg is served by web_chunks' halfvec HNSW index, not an exact scan.

The index is built on ``embedding::halfvec(N)`` (schema 014); ordering by the full-precision
``vector`` column cannot use it, so every recall read every embedded web chunk.
"""

from __future__ import annotations

import pytest

from ingestion.vector import vec_literal
from mcp_server.recall import Recall
from mcp_server.recall_web import vector_web_sql

pytestmark = pytest.mark.db

DIMS = 2048


def _axis(i: int, scale: float = 1.0) -> list[float]:
    v = [0.0] * DIMS
    v[i] = scale
    return v


@pytest.fixture()
def web(conn):
    conn.execute("TRUNCATE web_artifacts RESTART IDENTITY CASCADE")
    aid = conn.execute(
        "INSERT INTO web_artifacts (kind, tool_name, tool_use_id, url, title, fetched_at) VALUES "
        "('web_scrape', 'WebFetch', 'tu-1', 'https://example.org/a', 'A page', now()) RETURNING id"
    ).fetchone()[0]
    for idx, (vec, embedded) in enumerate([(_axis(0), True), (_axis(1), True), (_axis(0), False)]):
        conn.execute(
            "INSERT INTO web_chunks (web_artifact_id, idx, content, char_start, char_end, "
            "  embedding, is_embedded) VALUES (%s, %s, %s, 0, 1, %s, %s)",
            (aid, idx, f"chunk {idx}", vec_literal(vec) if embedded else None, embedded),
        )
    yield aid
    conn.execute("TRUNCATE web_artifacts RESTART IDENTITY CASCADE")


def test_vector_web_returns_the_nearest_embedded_chunks_first(web, db_url):
    q = [0.0] * DIMS
    q[0], q[1] = 1.0, 0.2
    rows = Recall(db_url, "")._search_vector_web(q, 5)
    assert [r["content"] for r in rows] == ["chunk 0", "chunk 1"]  # the unembedded one is out
    assert rows[0]["url"] == "https://example.org/a" and rows[0]["title"] == "A page"
    assert rows[0]["vec_distance"] < rows[1]["vec_distance"]


def test_vector_web_query_is_served_by_the_hnsw_index(conn):
    # Seqscan off is how the repo checks index ELIGIBILITY on a table too small for the
    # planner to prefer the index (see the kg_pg module docstring).
    conn.execute("SET enable_seqscan = off")
    try:
        plan = "\n".join(
            r[0]
            for r in conn.execute(
                "EXPLAIN " + vector_web_sql(DIMS),
                (vec_literal(_axis(0)), vec_literal(_axis(0)), 5),
            ).fetchall()
        )
    finally:
        conn.execute("RESET enable_seqscan")
    assert "web_chunks_embedding_hnsw" in plan
