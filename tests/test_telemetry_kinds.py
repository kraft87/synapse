"""Row-shape pins for every recall_metrics ``kind`` — the telemetry contract tests.

One test per kind ('recall', 'episodes', 'fetch', 'remember', 'board', 'timeline') drives
the REAL emitting surface with the network legs stubbed (the tests/test_recall_floor.py
harness; the tests/test_remember_notes.py server stubs) and asserts the row that lands in
recall_metrics carries the columns + served_ids envelope keys that SQL counters rely on.
A refactor that stops a surface emitting, or renames an envelope key, fails HERE — not
silently in a dashboard query.

Sync discipline: the metrics write is fire-and-forget on the engine's single-worker FIFO
executor, so a no-op barrier submitted after the call proves the row landed. Row selection
is deterministic (an earlier review caught a flake from an unordered fetchone()): each
test snapshots max(id) BEFORE driving the surface, then selects its kind above that
watermark with ORDER BY id DESC LIMIT 1 — a stale fire-and-forget write from a prior
test's engine can never win the assertion.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import psycopg
import pytest

_DB_URL = os.environ.get(
    "SYNAPSE_TEST_URL", "postgresql://synapse:synapse@127.0.0.1:5432/synapse_test"
)

# Skip the whole module if the shared Postgres test DB isn't up — these tests are DB-only.
try:
    _probe = psycopg.connect(_DB_URL, connect_timeout=2)
    _probe.close()
except Exception:  # pragma: no cover - environment dependent
    pytest.skip("no test DB reachable", allow_module_level=True)

import mcp_server.recall as recall_mod  # noqa: E402
from ingestion.db import Database  # noqa: E402
from mcp_server import server  # noqa: E402
from mcp_server.board import _OWNER, build_board, record_board_metrics  # noqa: E402
from mcp_server.recall import Recall  # noqa: E402


class _FakeEmbedder:
    def embed(self, texts, task="query"):
        return [[0.0] * 4 for _ in texts]


class _FakeTimeline:
    def recall_timeline(self, **kwargs):
        return {"items": []}


def _pool(n: int = 4) -> list[dict]:
    return [
        {
            "id": f"e:{i}",
            "content": f"pool doc {i}",
            "doc_type": "episode",
            "project": None,
            "created_at": None,
            "retrieval_count": 0,
        }
        for i in range(n)
    ]


def _wired(db_url: str, scores: list[float]) -> Recall:
    """A Recall whose search/feedback legs are stubbed (same set as test_recall_floor's
    harness) but whose telemetry writer is REAL — the row lands in the test DB."""
    r = Recall(db_url, "")
    p = _pool()
    r._ensure_embedder = lambda: _FakeEmbedder()
    r._ensure_timeline = lambda: _FakeTimeline()
    r._search_bm25_episodes = lambda q, proj, limit: list(p)
    r._search_vector_episodes = lambda emb, proj, limit: []
    r._search_vector_web = lambda emb, n: []
    r._search_kg = lambda *a, **k: ([], [])
    r._search_preferences = lambda emb, gid, limit: []
    r._fetch_history_pairs_pg = lambda gid, uuids, cap: []
    r._surface_supersessions = lambda *a, **k: []
    r._episode_supersessions = lambda *a, **k: {}
    # Compaction serves compact passages of the top reranked eps; stub returns them
    # directly (no more full-episode fallback — an empty return omits the bucket).
    r._compact_to_passages = lambda q, eps, n: [
        {"id": e["id"], "content": f"passage {e['id']}"} for e in eps[:n]
    ]
    r._increment_fact_retrieval_counts = lambda *a, **k: None
    r._increment_retrieval_counts = lambda ids: None
    r._rerank_pool_scored = lambda q, pl: [(i, scores[i]) for i in range(min(len(scores), len(pl)))]
    return r


def _watermark(conn) -> int:
    return conn.execute("SELECT coalesce(max(id), 0) FROM recall_metrics").fetchone()[0]


def _newest(conn, engine: Recall, kind: str, cols: str, after: int):
    """Barrier the engine's FIFO writer, then fetch the newest row of ``kind`` above the
    pre-call watermark. Returns None when the surface emitted nothing."""
    engine._async_executor.submit(lambda: None).result(timeout=10)
    return conn.execute(
        f"SELECT {cols} FROM recall_metrics WHERE kind = %s AND id > %s ORDER BY id DESC LIMIT 1",
        (kind, after),
    ).fetchone()


# ---------------------------------------------------------------------------
# kind='recall' — Recall.recall()
# ---------------------------------------------------------------------------


def test_recall_kind_row_shape(conn, db_url, monkeypatch):
    monkeypatch.setattr(recall_mod, "_RECALL_FLOOR", 0.58)
    q = f"telemetry shape pin recall {uuid.uuid4().hex[:8]}"
    engine = _wired(db_url, [0.91, 0.80, 0.60, 0.59])  # real scores, above the floor
    mark = _watermark(conn)
    out = engine.recall(q, source="mcp-tool")
    assert out["episodes"]

    row = _newest(
        conn,
        engine,
        "recall",
        "source, query, group_id, ms_total, rerank_top_score, emb_ok, "
        "n_episodes, chars, est_tokens, served_ids",
        mark,
    )
    assert row is not None, "recall() emitted no kind='recall' telemetry row"
    source, query, group_id, ms_total, top, emb_ok, n_eps, chars, est_tokens, served = row
    assert source == "mcp-tool" and query == q and group_id == "technical"
    assert ms_total is not None and ms_total >= 0
    assert top == pytest.approx(0.91)  # scores exist -> the raw pre-recency top is recorded
    assert emb_ok is True
    assert n_eps == len(out["episodes"]) and chars > 0 and est_tokens == chars // 4
    # served_ids envelope contract: per-bucket serve lists + the echo counter. Above the
    # floor there is no shadow marker — the key set is exactly this.
    assert set(served) == {
        "episodes",
        "facts",
        "web",
        "timeline",
        "prefs",
        "n_echo_suppressed",
        "n_bm25_lifted",
    }
    assert served["episodes"] == [it["id"] for it in out["episodes"]]
    assert served["facts"] == [] and served["timeline"] == [] and served["prefs"] == []
    assert served["web"] == []
    assert served["n_echo_suppressed"] == 0
    assert served["n_bm25_lifted"] == 0  # stub pool has no bm25_score -> fusion is a no-op


# ---------------------------------------------------------------------------
# kind='episodes' — Recall.recall_episodes() (+ the abstention-shadow keys)
# ---------------------------------------------------------------------------


def test_episodes_kind_row_shape_records_abstention_shadow(conn, db_url, monkeypatch):
    monkeypatch.setattr(recall_mod, "_RECALL_FLOOR", 0.58)
    q = f"telemetry shape pin episodes {uuid.uuid4().hex[:8]}"
    p = _pool()
    engine = _wired(db_url, [0.31, 0.22, 0.11, 0.05])  # real scores, below the floor
    engine._episode_pool = lambda q_, emb, proj: list(p)
    mark = _watermark(conn)
    out = engine.recall_episodes(q, source="mcp-tool")
    assert len(out["episodes"]) == 4  # shadow only — served in full

    row = _newest(
        conn,
        engine,
        "episodes",
        "source, query, ms_total, rerank_top_score, emb_ok, n_episodes, "
        "chars, est_tokens, served_ids",
        mark,
    )
    assert row is not None, "recall_episodes() emitted no kind='episodes' telemetry row"
    source, query, ms_total, top, emb_ok, n_eps, chars, est_tokens, served = row
    assert source == "mcp-tool" and query == q
    assert ms_total is not None and ms_total >= 0
    assert top == pytest.approx(0.31)  # the column recall() always had — now recorded here too
    assert emb_ok is True
    assert n_eps == 4 and chars > 0 and est_tokens == chars // 4
    # Below the floor the envelope gains the shadow keys alongside the serve list.
    assert set(served) == {"episodes", "n_echo_suppressed", "would_abstain", "floor"}
    assert served["episodes"] == [it["id"] for it in out["episodes"]]
    assert served["would_abstain"] is True
    assert served["floor"] == pytest.approx(0.58)


# ---------------------------------------------------------------------------
# kind='fetch' — Recall.fetch()
# ---------------------------------------------------------------------------


def test_fetch_kind_row_shape(conn, db_url):
    eid = conn.execute(
        "INSERT INTO episodes (session_id, sequence, content) VALUES (%s, 1, %s) RETURNING id",
        (f"telemetry-kinds-{uuid.uuid4().hex[:8]}", "full text of the fetched turn"),
    ).fetchone()[0]
    engine = Recall(db_url, "")  # fetch path is pure SQL — no leg stubs needed
    mark = _watermark(conn)
    out = engine.fetch([f"e:{eid}"], source="mcp-tool")
    assert [e["id"] for e in out["episodes"]] == [f"e:{eid}"]

    row = _newest(
        conn,
        engine,
        "fetch",
        "source, query, ms_total, n_episodes, chars, est_tokens, served_ids",
        mark,
    )
    assert row is not None, "fetch() emitted no kind='fetch' telemetry row"
    source, query, ms_total, n_eps, chars, est_tokens, served = row
    assert source == "mcp-tool"
    assert query == f"e:{eid}"  # comma-joined normalized parsed ids
    assert ms_total is not None and ms_total >= 0
    assert n_eps == 1 and chars > 0
    # Fetch's row stays lean: no est_tokens (the requested ids already live in `query`);
    # served_ids carries only the per-kind serve counts (test_tool_surface.py pins the
    # mixed-kind counts).
    assert est_tokens is None
    assert served == {"kinds": {"e": 1, "n": 0}}


# ---------------------------------------------------------------------------
# kind='timeline' — the recall_timeline() MCP tool
# ---------------------------------------------------------------------------


def test_timeline_kind_row_shape(conn, db_url, monkeypatch):
    engine = Recall(db_url, "")
    monkeypatch.setattr(server, "_recall_engine", engine)

    class _StubTimelineEngine:
        def recall_timeline(self, **kwargs):
            return {
                "query_shape": "time",
                "items": [{"_id": 7, "date": "2026-07-01", "fact": "shipped the widget"}],
            }

    monkeypatch.setattr(server, "_timeline_engine", _StubTimelineEngine())
    mark = _watermark(conn)
    out = server.recall_timeline(since="2026-07-01")
    assert len(out["items"]) == 1
    assert "_id" not in out["items"][0]  # internal key popped before serving

    row = _newest(conn, engine, "timeline", "source, ms_total, served_ids", mark)
    assert row is not None, "recall_timeline() emitted no kind='timeline' telemetry row"
    source, ms_total, served = row
    assert source == "mcp-tool"
    assert ms_total is not None and ms_total >= 0
    assert served == {"n_events": 1}


# ---------------------------------------------------------------------------
# kind='remember' — the remember() MCP tool
# ---------------------------------------------------------------------------


def test_remember_kind_row_shape(conn, db_url, monkeypatch):
    engine = Recall(db_url, "")
    monkeypatch.setattr(server, "DB_URL", db_url)
    monkeypatch.setattr(server, "_recall_engine", engine)
    # Keyless deps: NULL note embedding, dedup KNN skipped -> always 'created', no LLM.
    monkeypatch.setattr(server, "_notes_deps", lambda: (None, None))
    mark = _watermark(conn)
    out = asyncio.run(
        server.remember(hook="Telemetry shape pin note", body="Full body.", type="user")
    )
    assert out["status"] == "ok" and out["outcome"] == "created"

    row = _newest(conn, engine, "remember", "source, ms_total, served_ids", mark)
    assert row is not None, "remember() emitted no kind='remember' telemetry row"
    source, ms_total, served = row
    assert source == "mcp-tool"
    assert ms_total is not None and ms_total >= 0
    assert served == {"note": out["note_id"], "outcome": "created", "type": "user"}


# ---------------------------------------------------------------------------
# kind='board' — the /context serve path's writer (record_board_metrics). The
# route integration is covered in test_board.py; this pins the row SHAPE the
# only remaining serve source ('http' — the MCP board tool was removed) writes.
# ---------------------------------------------------------------------------


def test_board_kind_row_shape(conn, db_url):
    conn.execute("DELETE FROM notes")
    conn.execute("DELETE FROM timeline_events")
    db = Database(db_url)
    nid = db.insert_note(
        owner_id=_OWNER,
        group_id="technical",
        project=None,
        type="user",
        hook="Board telemetry shape pin",
        body="Body.",
        embedding=None,
        embed_model=None,
        source_ref=None,
    )
    db.close()

    engine = Recall(db_url, "")
    mark = _watermark(conn)
    board = build_board(db_url, None)
    assert board["status"] == "ok"
    record_board_metrics(engine, "http", 1.0, board)

    row = _newest(conn, engine, "board", "source, ms_total, chars, est_tokens, served_ids", mark)
    assert row is not None, "board serve emitted no kind='board' telemetry row"
    source, ms_total, chars, est_tokens, served = row
    assert source == "http"
    assert ms_total is not None and ms_total >= 0
    assert chars == len(board["text"]) and chars > 0
    assert est_tokens == chars // 4
    assert served == {"notes": [nid], "n_notes": 1, "overflow": 0}
