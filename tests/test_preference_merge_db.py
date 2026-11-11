"""A preference text merge and the device allowlists, against the real store.

A restricted device is served a preference only when its ``project`` is on that device's
allowlist, and the merge of two preferences rewrites the stored text while the row keeps
its own project. So a merge could carry text from a project a device may not read into a
row the device is served. The rule under test: a merge may not make text readable by a
device that could not read the project it came from. When it would, the old text stays
and the restatement is only counted.

Every test runs the real gate over the real store; the embedder puts the new preference at
cosine 0.85 to the stored one (inside the confirm band) and the confirm call is a stub
that says "duplicate" with a merged text.
"""

from __future__ import annotations

import math
import os

import psycopg
import pytest

_DB_URL = os.environ.get(
    "SYNAPSE_TEST_URL", "postgresql://synapse:synapse@127.0.0.1:5432/synapse_test"
)

try:
    _probe = psycopg.connect(_DB_URL, connect_timeout=2)
    _probe.close()
except Exception:  # pragma: no cover - environment dependent
    pytest.skip("no test DB reachable", allow_module_level=True)

import ingestion.preferences_gate as pg  # noqa: E402
from ingestion.db import Database  # noqa: E402
from mcp_server.preferences_routes import _OWNER  # noqa: E402
from tests.helpers.embed import onehot  # noqa: E402
from tests.helpers.surfaces import (  # noqa: E402
    clear_surfaces,
    register_device,
    register_restricted,
)

_STORED = "User prefers short replies"
_MERGED = "User prefers short replies, and a table when the answer is a comparison"
_SLOT = 50


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


@pytest.fixture()
def store(conn, db_url):
    """A Database on the scratch store, with no preferences and no registered surfaces."""
    conn.execute("DELETE FROM preferences WHERE owner_id = %s", (_OWNER,))
    clear_surfaces(conn)
    db = Database(db_url)
    yield db
    db.close()
    conn.execute("DELETE FROM preferences WHERE owner_id = %s", (_OWNER,))
    clear_surfaces(conn)


def _seed(db, project):
    return db.insert_preference(
        owner_id=_OWNER,
        project=project,
        pref=_STORED,
        polarity="like",
        embedding=onehot(_SLOT),
        embed_model="test",
        source_ref="ep:1",
    )


def _restate(db, monkeypatch, project):
    """One turn from ``project`` restating the stored preference, which the confirm
    judge calls a duplicate and offers a merged text for."""
    monkeypatch.setenv("SYNAPSE_PREFS_GATE", "1")
    stated = [
        {"pref": "User prefers short replies with tables for comparisons", "polarity": "like"}
    ]

    def _fake(*a, **k):
        if k.get("parser") is pg._parse_confirm:
            return {"duplicate": True, "updated_pref": _MERGED}
        return stated

    monkeypatch.setattr(pg, "parse_with_retry", _fake)
    gate = pg.PreferencesGate(db=db, llm_client=object(), embedder=_VecEmb(_tilted(_SLOT, 0.85)))
    gate.process({"id": 1, "episode_id": 5, "content": "x" * 500, "project": project})


def _state(conn, pref_id):
    pref, embedding, count = conn.execute(
        "SELECT pref, embedding::text, assert_count FROM preferences WHERE id = %s", (pref_id,)
    ).fetchone()
    return pref, embedding, count


def _live_count(conn):
    return conn.execute(
        "SELECT count(*) FROM preferences WHERE owner_id = %s AND t_invalid IS NULL", (_OWNER,)
    ).fetchone()[0]


def _assert_not_merged(conn, pref_id, before):
    after = _state(conn, pref_id)
    assert after[0] == _STORED and after[1] == before[1]  # old text and its embedding stay
    assert after[2] == before[2] + 1  # the restatement is still counted
    assert _live_count(conn) == 1  # and nothing new was inserted


def _assert_merged(conn, pref_id, before):
    after = _state(conn, pref_id)
    assert after[0] == _MERGED and after[1] != before[1]
    assert after[2] == before[2] + 1
    assert _live_count(conn) == 1


def test_text_from_outside_an_allowlist_is_not_merged_into_a_preference_inside_it(
    conn, store, monkeypatch
):
    register_restricted(conn, ["alpha"])
    pref = _seed(store, "alpha")
    before = _state(conn, pref)
    _restate(store, monkeypatch, "gamma")
    _assert_not_merged(conn, pref, before)


def test_a_turn_with_no_project_is_outside_every_allowlist(conn, store, monkeypatch):
    register_restricted(conn, ["alpha"])
    pref = _seed(store, "alpha")
    before = _state(conn, pref)
    _restate(store, monkeypatch, None)
    _assert_not_merged(conn, pref, before)


def test_text_from_inside_an_allowlist_merges_into_a_preference_outside_it(
    conn, store, monkeypatch
):
    register_restricted(conn, ["alpha"])
    pref = _seed(store, "gamma")
    before = _state(conn, pref)
    _restate(store, monkeypatch, "alpha")
    _assert_merged(conn, pref, before)


def test_two_projects_on_the_same_allowlist_merge(conn, store, monkeypatch):
    register_restricted(conn, ["alpha", "beta"])
    pref = _seed(store, "alpha")
    before = _state(conn, pref)
    _restate(store, monkeypatch, "BETA")  # compared case-insensitively
    _assert_merged(conn, pref, before)


def test_a_preference_on_no_allowlist_merges_freely(conn, store, monkeypatch):
    pref = _seed(store, "gamma")
    before = _state(conn, pref)
    _restate(store, monkeypatch, "delta")
    _assert_merged(conn, pref, before)


def test_each_device_is_checked_on_its_own_allowlist_not_on_the_union(conn, store, monkeypatch):
    """The first device reads the stored preference's project, the second reads the new
    turn's, and neither reads both. The union of the two lists holds both projects, so a
    union check would let the second project's text reach the first device."""
    register_restricted(conn, ["alpha"], "first-device")
    register_restricted(conn, ["beta"], "second-device")
    pref = _seed(store, "alpha")
    before = _state(conn, pref)
    _restate(store, monkeypatch, "beta")
    _assert_not_merged(conn, pref, before)


def test_a_device_that_is_not_approved_reads_nothing_and_blocks_nothing(conn, store, monkeypatch):
    register_device(conn, "tok-revoked", trust="restricted", projects=["alpha"], status="revoked")
    pref = _seed(store, "alpha")
    before = _state(conn, pref)
    _restate(store, monkeypatch, "gamma")
    _assert_merged(conn, pref, before)


def test_an_unreadable_allowlist_keeps_the_old_text(conn, store, monkeypatch):
    pref = _seed(store, "gamma")  # a merge that would otherwise go through
    before = _state(conn, pref)

    def _boom(self):
        raise psycopg.OperationalError("surfaces unreadable")

    monkeypatch.setattr(Database, "restricted_surface_allowlists", _boom)
    _restate(store, monkeypatch, "alpha")
    _assert_not_merged(conn, pref, before)


def test_allowlists_come_one_per_approved_restricted_device(conn, store):
    register_restricted(conn, ["alpha", "beta"], "first-device")
    register_restricted(conn, [], "second-device")
    register_device(conn, "tok-full", trust="full", projects=["gamma"])
    register_device(conn, "tok-revoked", trust="restricted", projects=["delta"], status="revoked")
    assert sorted(store.restricted_surface_allowlists()) == [[], ["alpha", "beta"]]
