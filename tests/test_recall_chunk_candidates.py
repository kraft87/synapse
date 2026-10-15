"""Chunks as an extra candidate source for the episode pool (RecallConfig.chunk_candidates).

A chunk is four consecutive turns of a session, embedded and keyword-indexed at ingest. With
the setting on, the top chunks by keyword and by embedding name their episodes, which join the
pool the cross-encoder reranks. A chunk is never served: only its episodes enter the pool, and
through the same project and surface predicates as the episode legs. Off (the default) the
pool is exactly the two episode legs.

DB-backed: the chunk sources are SQL (ParadeDB BM25 and a pgvector HNSW index), so they are
tested where they run.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest

import mcp_server.recall_episode_search as search_mod
from ingestion.schema_check import ColumnProbe
from ingestion.surfaces import FULL_TRUST
from mcp_server.recall import Recall
from mcp_server.recall_config import RecallConfig
from mcp_server.recall_warnings import warn_sink
from tests.helpers.kg import DIM, _axis_list


def _vec(i: int) -> str:
    return "[" + ",".join(map(str, _axis_list(i))) + "]"


class _Emb:
    """Embeds every query onto axis 0."""

    def embed(self, texts, task="query"):
        return [_axis_list(0) for _ in texts]


def _engine(db_url: str, chunk_candidates: bool = False) -> Recall:
    r = Recall(db_url, "", config=RecallConfig(chunk_candidates=chunk_candidates))
    r._ensure_embedder = lambda: _Emb()  # type: ignore[method-assign]
    return r


def _episode(conn, session: str, seq: int, content: str, project: str = "chunky") -> int:
    return conn.execute(
        "INSERT INTO episodes (session_id, sequence, content, project) "
        "VALUES (%s, %s, %s, %s) RETURNING id",
        (session, seq, content, project),
    ).fetchone()[0]


def _chunk(conn, session: str, episode_ids: list[int], content: str, axis: int, project="chunky"):
    conn.execute(
        "INSERT INTO chunks (session_id, start_sequence, end_sequence, episode_ids, content, "
        "  project, embedding, is_embedded) VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, TRUE)",
        (
            session,
            episode_ids[0],
            episode_ids[-1],
            json.dumps(episode_ids),
            content,
            project,
            _vec(axis),
        ),
    )


@pytest.fixture()
def corpus(conn, clean_tables):
    """One session of four turns whose chunk talks about a zebracorn the turns themselves
    never name, a second session whose chunk is unrelated, and a chunk in another project."""
    conn.execute("TRUNCATE chunks RESTART IDENTITY CASCADE")
    tag = uuid.uuid4().hex[:8]
    s1, s2, s3 = f"chunk-a-{tag}", f"chunk-b-{tag}", f"chunk-c-{tag}"
    a = [_episode(conn, s1, i, f"turn {i} about deployment details") for i in range(1, 5)]
    b = [_episode(conn, s2, i, f"turn {i} about something else entirely") for i in range(1, 5)]
    other = [_episode(conn, s3, i, f"turn {i} elsewhere", project="elsewhere") for i in range(1, 3)]
    _chunk(conn, s1, a, "the zebracorn migration was planned across these four turns", axis=0)
    _chunk(conn, s2, b, "grocery list and weekend plans", axis=7)
    _chunk(
        conn,
        s3,
        other,
        "another zebracorn mention, in a different project",
        axis=0,
        project="elsewhere",
    )
    yield {"a": a, "b": b, "other": other}
    conn.execute("TRUNCATE chunks RESTART IDENTITY CASCADE")


def _ids(items: list[dict[str, Any]]) -> list[str]:
    return [it["id"] for it in items]


def test_keyword_source_names_the_episodes_of_the_matching_chunk(corpus, db_url):
    r = _engine(db_url)
    found = r._search_chunks_bm25("zebracorn", None)
    # Both projects' chunks match; every episode of a matching chunk, in chunk order.
    ids = _ids(found)
    assert set(ids) == {f"e:{i}" for i in corpus["a"] + corpus["other"]}
    # A chunk's episodes come out together, in the chunk's own order.
    a_ids = [f"e:{i}" for i in corpus["a"]]
    start = ids.index(a_ids[0])
    assert ids[start : start + 4] == a_ids
    # The shape the other episode legs return, so the pool treats them alike.
    first = found[0]
    assert first["doc_type"] == "episode" and first["content"].startswith("turn ")
    assert {"project", "created_at", "retrieval_count", "session_id"} <= set(first)
    assert r._search_chunks_bm25("zebracorn", "chunky") != []
    assert set(_ids(r._search_chunks_bm25("zebracorn", "chunky"))) == {
        f"e:{i}" for i in corpus["a"]
    }


def test_vector_source_names_the_episodes_of_the_nearest_chunks(corpus, db_url):
    r = _engine(db_url)
    found = r._search_chunks_vector(_axis_list(0), "chunky")
    assert set(_ids(found)) >= {f"e:{i}" for i in corpus["a"]}
    assert not set(_ids(found)) & {f"e:{i}" for i in corpus["other"]}  # other project
    assert len(_axis_list(0)) == DIM


def test_a_restricted_caller_gets_only_the_episodes_it_could_have_searched(corpus, db_url):
    r = _engine(db_url)
    found = r._search_chunks_bm25("zebracorn", None, allowed_projects=["elsewhere"])
    assert set(_ids(found)) == {f"e:{i}" for i in corpus["other"]}
    assert r._search_chunks_bm25("zebracorn", None, allowed_projects=[]) == []
    assert r._search_chunks_vector(_axis_list(0), None, allowed_projects=["nowhere"]) == []


def test_an_unindexed_vector_source_says_so_and_serves_nothing(corpus, db_url, monkeypatch):
    monkeypatch.setattr(
        search_mod,
        "CHUNK_VECTOR_INDEX",
        ColumnProbe("SELECT 1 WHERE false", "the chunks vector index"),
    )
    r = _engine(db_url)
    notices: list[str] = []
    with warn_sink(notices):
        assert r._search_chunks_vector(_axis_list(0), None) == []
    assert len(notices) == 1 and "chunks vector index" in notices[0]


def test_chunk_episodes_join_the_pool_only_when_the_setting_is_on(corpus, db_url):
    # The turns say "deployment details"; only their chunk says "zebracorn".
    def pool(on: bool) -> list[str]:
        r = _engine(db_url, on)
        legs = r._start_episode_legs("zebracorn", "chunky")
        assert (legs.f_chunk_bm25 is not None) is on and (legs.f_chunk_vec is not None) is on
        return _ids(r._join_episode_pool(legs).items)

    expected = {f"e:{i}" for i in corpus["a"]}
    assert not expected & set(pool(False))
    assert expected <= set(pool(True))


def test_chunk_sources_stay_out_of_a_session_scoped_drill_down(corpus, db_url):
    r = _engine(db_url, True)
    legs = r._start_episode_legs("zebracorn", None, session_id="some-session")
    assert legs.f_chunk_bm25 is None and legs.f_chunk_vec is None
    r._join_episode_pool(legs)


def test_recall_with_chunk_candidates_still_serves_episodes_not_chunks(corpus, db_url):
    r = _engine(db_url, True)
    r._rerank_pool_scored = lambda q, pl: [(i, 0.9 - 0.01 * i) for i in range(len(pl))]  # type: ignore[method-assign]
    r._record_metrics = lambda m: None  # type: ignore[method-assign]
    out = r.recall_episodes("zebracorn", project="chunky", limit=5, trust=FULL_TRUST)
    served = _ids(out["episodes"])
    # Episodes the chunk legs named twice (by keyword and by embedding) lead the pool.
    assert served[:4] == [f"e:{i}" for i in corpus["a"]]
    assert all(i.startswith("e:") for i in served)
