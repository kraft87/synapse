"""Session drill-down: fetch_session (the Read analog) + session_id scoping
(the Grep analog) + the `session` pivot key on served items.

fetch_session must serve the anchor FULL and neighbors as bounded heads (the
over-serving lesson: 7 full turns is a 30K-token dump), clamp radius/limit,
and answer an unknown session with an explicit error — never a silent empty —
so the caller knows to fall back to the on-disk transcript. The session filter
must reach both episode search legs' SQL, and an explicit session ask must
skip self-exclusion (a scoped call to your own session is intentional).

Pure-logic like test_recall_notes.py: bare Recall instances, pg stubbed on the
instance, no live DB.
"""

from __future__ import annotations

from typing import Any, ClassVar

from mcp_server.recall import Recall, _to_recall_item


def _bare() -> Recall:
    r = object.__new__(Recall)
    r._db_url = "postgresql://stub"
    return r


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _PG:
    """Dispatches on SQL substrings; records every (sql, params) call."""

    calls: ClassVar[list[tuple[str, tuple]]] = []

    def __init__(self, session_rows: list[dict], anchor_seq: int | None = None):
        self._session_rows = session_rows
        self._anchor_seq = anchor_seq

    def execute(self, sql, params=()):
        _PG.calls.append((sql, tuple(params)))
        if "count(*)" in sql:
            n = len(self._session_rows)
            first = self._session_rows[0]["created_at"] if n else None
            last = self._session_rows[-1]["created_at"] if n else None
            return _Rows([{"n": n, "first": first, "last": last, "project": "proj" if n else None}])
        if "SELECT sequence" in sql:
            if self._anchor_seq is None:
                return _Rows([])
            return _Rows([{"sequence": self._anchor_seq}])
        if "BETWEEN" in sql:
            lo, hi = params[1], params[2]
            return _Rows([r for r in self._session_rows if lo <= r["sequence"] <= hi])
        if "OFFSET" in sql:
            limit, offset = params[1], params[2]
            return _Rows(self._session_rows[offset : offset + limit])
        raise AssertionError(f"unexpected SQL: {sql}")


def _turn(eid: int, seq: int, content: str = "") -> dict:
    return {
        "id": eid,
        "sequence": seq,
        "content": content or f"turn {seq} " + "x" * 600,
        "created_at": f"2026-08-0{min(seq, 7)} 10:00:00",
        "has_h": True,
        "has_a": seq % 2 == 0,
    }


def _wire(r: Recall, pg: _PG, metrics: list) -> None:
    _PG.calls = []
    r._ensure_pg = lambda: pg  # type: ignore[method-assign]
    r._record_metrics = metrics.append  # type: ignore[method-assign]
    r._increment_retrieval_counts = lambda ids: None  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# The pivot key
# ---------------------------------------------------------------------------


def test_recall_item_carries_session():
    item = _to_recall_item(
        {
            "id": "e:7",
            "content": "c",
            "project": "p",
            "created_at": "2026-08-07 10:00",
            "session_id": "sess-1",
        }
    )
    assert item["session"] == "sess-1"


def test_recall_item_omits_session_when_absent():
    assert "session" not in _to_recall_item({"id": "e:7", "content": "c"})


# ---------------------------------------------------------------------------
# fetch_session — anchor mode
# ---------------------------------------------------------------------------


def test_anchor_full_neighbors_heads():
    rows = [_turn(100 + s, s) for s in range(1, 8)]
    r, metrics = _bare(), []
    _wire(r, _PG(rows, anchor_seq=4), metrics)

    out = r.fetch_session("sess-1", around="e:104", radius=2)
    assert out["total_turns"] == 7 and out["project"] == "proj"
    assert [t["seq"] for t in out["turns"]] == [2, 3, 4, 5, 6]
    anchor = next(t for t in out["turns"] if t["seq"] == 4)
    assert "content" in anchor and "head" not in anchor  # served whole
    for t in out["turns"]:
        if t["seq"] != 4:
            assert "content" not in t and len(t["head"]) <= 500
            assert t["full_chars"] > 500  # tells the caller what fetch() expands to
    assert metrics and metrics[0]["kind"] == "fetch_session"


def test_anchor_not_in_session_is_explicit_error():
    r, metrics = _bare(), []
    _wire(r, _PG([_turn(101, 1)], anchor_seq=None), metrics)
    out = r.fetch_session("sess-1", around="e:999")
    assert "not in session" in out["error"]


def test_unparseable_anchor_is_explicit_error():
    r, metrics = _bare(), []
    _wire(r, _PG([_turn(101, 1)]), metrics)
    assert "unparseable" in r.fetch_session("sess-1", around="banana")["error"]


# ---------------------------------------------------------------------------
# fetch_session — anchorless paging + clamps + unknown session
# ---------------------------------------------------------------------------


def test_paging_serves_heads_only():
    rows = [_turn(100 + s, s) for s in range(1, 8)]
    r, metrics = _bare(), []
    _wire(r, _PG(rows), metrics)
    out = r.fetch_session("sess-1", offset=2, limit=3)
    assert [t["seq"] for t in out["turns"]] == [3, 4, 5]
    assert all("content" not in t and "head" in t for t in out["turns"])
    assert {t["role"] for t in out["turns"]} == {"user", "mixed"}


def test_radius_and_limit_clamped():
    rows = [_turn(100 + s, s) for s in range(1, 8)]
    r, metrics = _bare(), []
    _wire(r, _PG(rows, anchor_seq=4), metrics)
    r.fetch_session("sess-1", around="e:104", radius=99)
    between = next(p for sql, p in _PG.calls if "BETWEEN" in sql)
    assert (between[1], between[2]) == (4 - 10, 4 + 10)  # radius capped at 10

    _wire(r, _PG(rows), metrics)
    r.fetch_session("sess-1", limit=999)
    paged = next(p for sql, p in _PG.calls if "OFFSET" in sql)
    assert paged[1] == 25  # page cap


def test_unknown_session_is_explicit_error_not_empty():
    r, metrics = _bare(), []
    _wire(r, _PG([]), metrics)
    out = r.fetch_session("nope")
    assert "not indexed" in out["error"] and "on disk" in out["error"]
    assert metrics[0]["served_ids"]["error"]  # the miss is telemetry-visible


# ---------------------------------------------------------------------------
# Session filter reaches both search legs; scoped calls skip self-exclusion
# ---------------------------------------------------------------------------


class _CapturePG:
    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []

    def execute(self, sql, params=()):
        self.calls.append((sql, tuple(params)))
        return _Rows([])


def test_bm25_leg_scopes_to_session():
    r = _bare()
    pg = _CapturePG()
    r._ensure_pg = lambda: pg  # type: ignore[method-assign]
    r._search_bm25_episodes("q", None, 10, session_id="sess-1")
    sql, params = pg.calls[0]
    assert "session_id = %s" in sql and "sess-1" in params


def test_vector_leg_scopes_to_session():
    r = _bare()
    pg = _CapturePG()
    r._ensure_pg = lambda: pg  # type: ignore[method-assign]
    r._search_vector_episodes([0.1, 0.2], "proj", 10, session_id="sess-1")
    sql, params = pg.calls[0]
    assert "session_id = %s" in sql and "sess-1" in params and "proj" in params


def test_session_scoped_recall_skips_self_exclusion(monkeypatch):
    """An explicit session ask must never be suppressed — even for the caller's
    own session (the whole point is reading yourself back)."""
    import mcp_server.recall as recall_mod

    r = _bare()
    seen: dict[str, Any] = {}

    monkeypatch.setattr(recall_mod, "_RECALL_SELF_EXCLUDE", True)
    r._ensure_embedder = lambda: (_ for _ in ()).throw(RuntimeError("no embedder"))  # type: ignore[method-assign]

    def _pool(q, e, p, session_id=None):
        seen["session_id"] = session_id
        return []

    def _excl(pool, sid):
        seen["excluded"] = True
        return pool

    r._episode_pool = _pool  # type: ignore[method-assign]
    r._exclude_self = _excl  # type: ignore[method-assign]
    r._select_episodes = lambda q, pool, limit: (pool, 0, 0.0)  # type: ignore[method-assign]
    r._record_metrics = lambda m: seen.setdefault("metrics", m)  # type: ignore[method-assign]
    r._increment_retrieval_counts = lambda ids: None  # type: ignore[method-assign]

    r.recall_episodes("q", self_session="me", session_id="sess-1")
    assert seen["session_id"] == "sess-1"  # filter reached the pool
    assert "excluded" not in seen  # self-exclusion skipped for scoped calls
    assert seen["metrics"]["served_ids"]["session_id"] == "sess-1"

    seen.clear()
    r.recall_episodes("q", self_session="me")
    assert "excluded" in seen  # unscoped calls still self-exclude
