"""DB-backed tests for the preferences store + read route (schema 035).

Covers the reconciliation round-trip the gate relies on (insert / cosine KNN / reassert /
supersede) and the session-start route serialization + ordering. Mirrors the timeline
route tests. Skips cleanly when no test DB is reachable (the pure-logic decision + parser
coverage lives in test_preferences_gate.py)."""

from __future__ import annotations

import math
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

import ingestion.preferences_gate as pg  # noqa: E402
from ingestion.db import Database  # noqa: E402
from ingestion.db_connection import _EMBED_DIMS, _vector_literal  # noqa: E402
from ingestion.surfaces import FULL_TRUST, UNKNOWN_SURFACE, SurfaceTrust  # noqa: E402
from mcp_server.preferences_routes import (  # noqa: E402
    _OWNER,
    BLOCK_LINES,
    _top_preferences,
    top_block,
)
from tests.helpers.embed import onehot  # noqa: E402


def _wipe(conn):
    conn.execute("DELETE FROM preferences WHERE owner_id = %s", (_OWNER,))


def test_insert_and_cosine_knn(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    a = db.insert_preference(
        owner_id=_OWNER,
        project="synapse",
        pref="User prefers bullet lists over tables",
        polarity="like",
        embedding=onehot(1),
        embed_model="test",
        source_ref="ep:1",
    )
    db.insert_preference(
        owner_id=_OWNER,
        project="synapse",
        pref="User dislikes em-dashes",
        polarity="dislike",
        embedding=onehot(2),
        embed_model="test",
        source_ref="ep:2",
    )
    # Query nearest to slot-1: the matching pref comes back with sim ~1, ahead of the other.
    hits = db.find_live_preferences(_OWNER, onehot(1), limit=5)
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
        project=None,
        pref="User prefers concise answers",
        polarity="like",
        embedding=onehot(3),
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
        project=None,
        pref="User prefers dark mode",
        polarity="like",
        embedding=onehot(4),
        embed_model="test",
        source_ref="ep:4",
    )
    new = db.insert_preference(
        owner_id=_OWNER,
        project=None,
        pref="User now prefers light mode",
        polarity="dislike",
        embedding=onehot(4),
        embed_model="test",
        source_ref="ep:5",
    )
    db.supersede_preference(old, new)
    # The retired row carries the link and is gone from the live set.
    row = conn.execute(
        "SELECT t_invalid, superseded_by FROM preferences WHERE id = %s", (old,)
    ).fetchone()
    assert row[0] is not None and row[1] == new
    live_ids = {h["id"] for h in db.find_live_preferences(_OWNER, onehot(4), limit=5)}
    assert new in live_ids and old not in live_ids
    db.close()
    _wipe(conn)


def test_top_preferences_route_shape_and_order(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    db.insert_preference(
        owner_id=_OWNER,
        project=None,
        pref="User likes short commit messages",
        polarity="like",
        embedding=onehot(6),
        embed_model="test",
        source_ref="ep:6",
    )
    strong = db.insert_preference(
        owner_id=_OWNER,
        project=None,
        pref="User never wants contract roles surfaced",
        polarity="rule",
        embedding=onehot(7),
        embed_model="test",
        source_ref="ep:7",
    )
    db.reassert_preference(strong)  # assert_count 2 -> ranks above weak
    retired = db.insert_preference(
        owner_id=_OWNER,
        project=None,
        pref="User prefers tabs",
        polarity="like",
        embedding=onehot(8),
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


def _prefer(db, slot, pref, polarity="rule", project=None):
    return db.insert_preference(
        owner_id=_OWNER,
        project=project,
        pref=pref,
        polarity=polarity,
        embedding=onehot(slot),
        embed_model="test",
        source_ref=f"ep:{slot}",
    )


def test_the_block_served_with_the_board_is_scoped_like_the_route(conn, db_url):
    """``GET /context?parts=preferences`` renders these rows on the server (top_block):
    the same rows, for the same caller, as ``/preferences/top``."""
    _wipe(conn)
    db = Database(db_url)
    _prefer(db, 10, "User wants short answers", "rule")
    _prefer(db, 11, "User likes tabs in Go", "like", "alpha")
    _prefer(db, 12, "User dislikes mocks", "dislike", "beta")
    db.close()

    header, *lines = top_block(db_url, FULL_TRUST).splitlines()
    assert header == "[Synapse preferences]"
    assert sorted(lines) == [
        "  - (dislikes) User dislikes mocks",
        "  - (likes) User likes tabs in Go",
        "  - (rule) User wants short answers",
    ]
    alpha = SurfaceTrust(trust="restricted", allowed_projects=("alpha",), known=True)
    assert top_block(db_url, alpha) == "[Synapse preferences]\n  - (likes) User likes tabs in Go"
    # Nothing to show is no block at all, header included.
    assert top_block(db_url, UNKNOWN_SURFACE) == ""
    _wipe(conn)


def test_the_block_holds_the_strongest_few(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    ids = [_prefer(db, 20 + n, f"User rule number {n}") for n in range(BLOCK_LINES + 2)]
    db.reassert_preference(ids[-1])  # asserted twice: first, though it was stored last
    db.close()
    header, *lines = top_block(db_url, FULL_TRUST).splitlines()
    assert header == "[Synapse preferences]" and len(lines) == BLOCK_LINES
    assert lines[0] == f"  - (rule) User rule number {BLOCK_LINES + 1}"
    _wipe(conn)


# ---------------------------------------------------------------------------
# The gate against the real store — the owner's live preferences are one pool for a
# fresh turn, whatever its project
# ---------------------------------------------------------------------------


def _tilted(slot, cos):
    """A unit vector at cosine ``cos`` to ``onehot(slot)``."""
    v = onehot(slot)
    v[slot] = cos
    v[slot + 1] = math.sqrt(1 - cos * cos)
    return v


class _VecEmb:
    model_name = "test"

    def __init__(self, vec):
        self._vec = vec

    def embed(self, texts, task):
        return [self._vec for _ in texts]


def _seed_pref(conn, *, slot, polarity):
    return conn.execute(
        "INSERT INTO preferences (owner_id, pref, polarity, embedding, embed_model) "
        f"VALUES (%s, 'User prefers dark mode', %s, %s::vector({_EMBED_DIMS}), 'test') "
        "RETURNING id",
        (_OWNER, polarity, _vector_literal(onehot(slot))),
    ).fetchone()[0]


def _run_gate(db, monkeypatch, vec, *, polarity, project):
    monkeypatch.setenv("SYNAPSE_PREFS_GATE", "1")
    stated = [{"pref": "User prefers light mode", "polarity": polarity}]
    monkeypatch.setattr(pg, "parse_with_retry", lambda *a, **k: stated)
    gate = pg.PreferencesGate(db=db, llm_client=object(), embedder=_VecEmb(vec))
    gate.process({"id": 1, "episode_id": 5, "content": "x" * 500, "project": project})


# The project of the new turn: none, and one the old per-group lookup sent elsewhere.
_PROJECTS = [None, "jobs"]


@pytest.mark.parametrize("project", _PROJECTS)
def test_restated_preference_reasserts_the_live_one_whatever_the_project(
    conn, db_url, monkeypatch, project
):
    _wipe(conn)
    db = Database(db_url)
    old = _seed_pref(conn, slot=30, polarity="like")
    _run_gate(db, monkeypatch, onehot(30), polarity="like", project=project)
    rows = conn.execute("SELECT id, assert_count FROM preferences WHERE t_invalid IS NULL")
    assert rows.fetchall() == [(old, 2)]
    db.close()
    _wipe(conn)


@pytest.mark.parametrize("project", _PROJECTS)
def test_reversed_preference_supersedes_the_live_one_whatever_the_project(
    conn, db_url, monkeypatch, project
):
    _wipe(conn)
    db = Database(db_url)
    old = _seed_pref(conn, slot=31, polarity="like")
    _run_gate(db, monkeypatch, _tilted(31, 0.85), polarity="dislike", project=project)
    live = conn.execute("SELECT id FROM preferences WHERE t_invalid IS NULL").fetchall()
    assert len(live) == 1 and live[0][0] != old
    retired = conn.execute("SELECT superseded_by FROM preferences WHERE id = %s", (old,))
    assert retired.fetchone()[0] == live[0][0]
    db.close()
    _wipe(conn)
