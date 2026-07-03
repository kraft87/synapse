"""DB-backed tests for the preferences store + read route (schema 035).

Covers the reconciliation round-trip the gate relies on (insert / cosine KNN / reassert /
supersede) and the session-start route serialization + ordering. Mirrors the timeline
route tests. Skips cleanly when no test DB is reachable (the pure-logic decision + parser
coverage lives in test_preferences_gate.py)."""

from __future__ import annotations

import os

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

from ingestion.db import Database  # noqa: E402
from ingestion.embedding import embed_dims  # noqa: E402
from mcp_server.preferences_routes import _OWNER, _top_preferences  # noqa: E402

_DIMS = embed_dims()
GROUP = "technical"


def _vec(slot: int) -> list[float]:
    """A one-hot 2048-dim unit vector: identical slots -> cosine sim 1, distinct -> 0.
    Lets the KNN ordering be asserted deterministically without a real embedder."""
    v = [0.0] * _DIMS
    v[slot % _DIMS] = 1.0
    return v


def _wipe(conn):
    conn.execute("DELETE FROM preferences WHERE owner_id = %s", (_OWNER,))


def test_insert_and_cosine_knn(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    a = db.insert_preference(
        owner_id=_OWNER,
        group_id=GROUP,
        project="synapse",
        pref="User prefers bullet lists over tables",
        polarity="like",
        embedding=_vec(1),
        embed_model="test",
        source_ref="ep:1",
    )
    db.insert_preference(
        owner_id=_OWNER,
        group_id=GROUP,
        project="synapse",
        pref="User dislikes em-dashes",
        polarity="dislike",
        embedding=_vec(2),
        embed_model="test",
        source_ref="ep:2",
    )
    # Query nearest to slot-1: the matching pref comes back with sim ~1, ahead of the other.
    hits = db.find_live_preferences(_OWNER, GROUP, _vec(1), limit=5)
    assert hits[0]["id"] == a
    assert hits[0]["sim"] == pytest.approx(1.0, abs=1e-4)
    assert hits[1]["sim"] == pytest.approx(0.0, abs=1e-4)
    db.close()
    _wipe(conn)


def test_reassert_bumps_count_and_keeps_text(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    pid = db.insert_preference(
        owner_id=_OWNER,
        group_id=GROUP,
        project=None,
        pref="User prefers concise answers",
        polarity="like",
        embedding=_vec(3),
        embed_model="test",
        source_ref="ep:3",
    )
    db.reassert_preference(pid)
    db.reassert_preference(pid)
    row = conn.execute(
        "SELECT pref, assert_count FROM preferences WHERE id = %s", (pid,)
    ).fetchone()
    assert row[0] == "User prefers concise answers"  # older text kept
    assert row[1] == 3  # 1 (insert) + 2 reasserts
    db.close()
    _wipe(conn)


def test_supersede_retires_old_and_hides_from_live(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    old = db.insert_preference(
        owner_id=_OWNER,
        group_id=GROUP,
        project=None,
        pref="User prefers dark mode",
        polarity="like",
        embedding=_vec(4),
        embed_model="test",
        source_ref="ep:4",
    )
    new = db.insert_preference(
        owner_id=_OWNER,
        group_id=GROUP,
        project=None,
        pref="User now prefers light mode",
        polarity="dislike",
        embedding=_vec(4),
        embed_model="test",
        source_ref="ep:5",
    )
    db.supersede_preference(old, new)
    # The retired row carries the link and is gone from the live set.
    row = conn.execute(
        "SELECT t_invalid, superseded_by FROM preferences WHERE id = %s", (old,)
    ).fetchone()
    assert row[0] is not None and row[1] == new
    live_ids = {h["id"] for h in db.find_live_preferences(_OWNER, GROUP, _vec(4), limit=5)}
    assert new in live_ids and old not in live_ids
    db.close()
    _wipe(conn)


def test_top_preferences_route_shape_and_order(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    db.insert_preference(
        owner_id=_OWNER,
        group_id=GROUP,
        project=None,
        pref="User likes short commit messages",
        polarity="like",
        embedding=_vec(6),
        embed_model="test",
        source_ref="ep:6",
    )
    strong = db.insert_preference(
        owner_id=_OWNER,
        group_id="personal",  # cross-group: the block spans all groups for the owner
        project=None,
        pref="User never wants contract roles surfaced",
        polarity="rule",
        embedding=_vec(7),
        embed_model="test",
        source_ref="ep:7",
    )
    db.reassert_preference(strong)  # assert_count 2 -> ranks above weak
    retired = db.insert_preference(
        owner_id=_OWNER,
        group_id=GROUP,
        project=None,
        pref="User prefers tabs",
        polarity="like",
        embedding=_vec(8),
        embed_model="test",
        source_ref="ep:8",
    )
    db.supersede_preference(retired, strong)  # retired must not appear

    items = _top_preferences(db_url, limit=8)
    prefs = [i["pref"] for i in items]
    assert prefs == [
        "User never wants contract roles surfaced",  # assert_count 2, first
        "User likes short commit messages",
    ]
    top = items[0]
    assert set(top) == {"pref", "polarity", "assert_count", "since"}
    assert top["polarity"] == "rule" and top["assert_count"] == 2
    assert top["since"] and len(top["since"]) == 10  # YYYY-MM-DD
    db.close()
    _wipe(conn)
