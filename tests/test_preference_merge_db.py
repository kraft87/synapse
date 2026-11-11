"""Which stored preferences a new one may touch, against the real store.

A restricted device is served a preference only when its ``project`` is on that device's
allowlist. So a turn that re-asserts, merges into or retires a preference of another
project could change text, or a served count, that a device reads differently from the
turn's own project, and a device that states a stance could learn from its own preference
block whether a stored one existed. The rule under test: a new preference is matched only
against live preferences with the same readers (for every approved restricted device, both
projects on its allowlist or both off it, compared case-insensitively). A row with other
readers is never a candidate, so it is never re-asserted, merged into or retired by that
write, and the judge never sees its text; the new item is stored as its own row.

Every test runs the real gate over the real store. The embedder puts the new preference at
``onehot(_SLOT)``; a stored row sits at the cosine the test names (0.90 and above is a
re-assertion, 0.78 to 0.90 is the confirm band) and the confirm call is a stub that says
"duplicate" with a merged text and counts how often it was asked.
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
_NEW = "User prefers short replies with tables for comparisons"
_MERGED = "User prefers short replies, and a table when the answer is a comparison"
_SLOT = 50


def _at(cos, other=_SLOT + 1):
    """A unit vector at cosine ``cos`` to ``onehot(_SLOT)``; ``other`` is the axis the rest
    of its length lies on, so two rows built on different axes are not near each other."""
    v = onehot(_SLOT)
    v[_SLOT] = cos
    v[other] = math.sqrt(1 - cos * cos)
    return v


class _NewItemEmb:
    """Every text embeds to the new item's vector, so a merged text re-embeds there too."""

    model_name = "test"

    def embed(self, texts, task):
        return [onehot(_SLOT) for _ in texts]


class _Judge:
    """Stands in for both LLM calls the gate makes: the per-turn gate (one new
    preference) and the gray-band confirm (always "duplicate", with a merged text)."""

    def __init__(self, polarity="like"):
        self.polarity = polarity
        self.confirms = 0

    def __call__(self, *a, **k):
        if k.get("parser") is pg._parse_confirm:
            self.confirms += 1
            return {"duplicate": True, "updated_pref": _MERGED}
        return [{"pref": _NEW, "polarity": self.polarity}]


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


def _seed(db, project, *, cos=1.0, polarity="like", other=_SLOT + 1, pref=_STORED):
    """A stored live preference of ``project`` at cosine ``cos`` to the new item."""
    return db.insert_preference(
        owner_id=_OWNER,
        project=project,
        pref=pref,
        polarity=polarity,
        embedding=_at(cos, other),
        embed_model="test",
        source_ref="ep:1",
    )


def _turn(db, monkeypatch, project, *, polarity="like"):
    """One turn from ``project`` stating the new preference. Returns the judge stub."""
    monkeypatch.setenv("SYNAPSE_PREFS_GATE", "1")
    judge = _Judge(polarity)
    monkeypatch.setattr(pg, "parse_with_retry", judge)
    gate = pg.PreferencesGate(db=db, llm_client=object(), embedder=_NewItemEmb())
    gate.process({"id": 1, "episode_id": 5, "content": "x" * 500, "project": project})
    return judge


def _row(conn, pref_id):
    """Everything a write could change on one row."""
    return conn.execute(
        "SELECT pref, embedding::text, assert_count, last_asserted, t_invalid, superseded_by "
        "FROM preferences WHERE id = %s",
        (pref_id,),
    ).fetchone()


def _live(conn):
    return conn.execute(
        "SELECT id, project, pref FROM preferences "
        "WHERE owner_id = %s AND t_invalid IS NULL ORDER BY id",
        (_OWNER,),
    ).fetchall()


def _assert_stored_as_own_row(conn, seeded, project, polarity="like"):
    """The turn stored its own text under ``project`` as a new live row, and nothing else
    new."""
    rows = [r for r in _live(conn) if r[0] not in seeded]
    assert [(r[1], r[2]) for r in rows] == [(project, _NEW)]
    stored = conn.execute(
        "SELECT polarity, assert_count FROM preferences WHERE id = %s", (rows[0][0],)
    ).fetchone()
    assert stored == (polarity, 1)


def _assert_other_class_untouched(conn, target, before, project, judge=None, polarity="like"):
    assert _row(conn, target) == before
    _assert_stored_as_own_row(conn, [target], project, polarity)
    if judge is not None:
        assert judge.confirms == 0


# ---- another class is never re-asserted (the target is untouched, the judge unasked) -


def test_a_turn_off_the_allowlist_does_not_reassert_a_preference_on_it(conn, store, monkeypatch):
    register_restricted(conn, ["alpha"])
    target = _seed(store, "alpha")
    before = _row(conn, target)
    judge = _turn(store, monkeypatch, "gamma")
    _assert_other_class_untouched(conn, target, before, "gamma", judge)


def test_a_turn_on_the_allowlist_does_not_reassert_a_preference_off_it(conn, store, monkeypatch):
    register_restricted(conn, ["alpha"])
    target = _seed(store, "gamma")
    before = _row(conn, target)
    judge = _turn(store, monkeypatch, "alpha")
    _assert_other_class_untouched(conn, target, before, "alpha", judge)


# ---- nor retired, merged into or shown to the judge ----------------------------------

_ACROSS = [("alpha", "gamma"), ("gamma", "alpha")]


@pytest.mark.parametrize(("stored_project", "turn_project"), _ACROSS)
def test_a_polarity_flip_across_classes_retires_nothing(
    conn, store, monkeypatch, stored_project, turn_project
):
    register_restricted(conn, ["alpha"])
    target = _seed(store, stored_project, cos=0.84)
    before = _row(conn, target)
    _turn(store, monkeypatch, turn_project, polarity="dislike")
    _assert_other_class_untouched(conn, target, before, turn_project, polarity="dislike")


@pytest.mark.parametrize(("stored_project", "turn_project"), _ACROSS)
def test_the_judge_is_never_shown_a_preference_of_another_class(
    conn, store, monkeypatch, stored_project, turn_project
):
    register_restricted(conn, ["alpha"])
    target = _seed(store, stored_project, cos=0.85)
    before = _row(conn, target)
    judge = _turn(store, monkeypatch, turn_project)
    _assert_other_class_untouched(conn, target, before, turn_project, judge)


# ---- the same class behaves as before ------------------------------------------------

_SAME_CLASS = [
    pytest.param(["alpha"], "gamma", "delta", id="both-off-the-list"),
    pytest.param(["alpha", "beta"], "alpha", "beta", id="both-on-the-same-list"),
]


@pytest.mark.parametrize(("allowed", "stored_project", "turn_project"), _SAME_CLASS)
def test_same_class_reasserts(conn, store, monkeypatch, allowed, stored_project, turn_project):
    register_restricted(conn, allowed)
    target = _seed(store, stored_project, cos=0.95)
    before = _row(conn, target)
    judge = _turn(store, monkeypatch, turn_project)
    after = _row(conn, target)
    assert after[0] == _STORED and after[1] == before[1]  # the older text stays
    assert after[2] == before[2] + 1 and after[3] > before[3]
    assert judge.confirms == 0
    assert [r[0] for r in _live(conn)] == [target]


@pytest.mark.parametrize(("allowed", "stored_project", "turn_project"), _SAME_CLASS)
def test_same_class_merges(conn, store, monkeypatch, allowed, stored_project, turn_project):
    register_restricted(conn, allowed)
    target = _seed(store, stored_project, cos=0.85)
    before = _row(conn, target)
    judge = _turn(store, monkeypatch, turn_project)
    after = _row(conn, target)
    assert after[0] == _MERGED and after[1] != before[1]
    assert after[2] == before[2] + 1
    assert judge.confirms == 1
    assert [r[0] for r in _live(conn)] == [target]


@pytest.mark.parametrize(("allowed", "stored_project", "turn_project"), _SAME_CLASS)
def test_same_class_supersedes(conn, store, monkeypatch, allowed, stored_project, turn_project):
    register_restricted(conn, allowed)
    target = _seed(store, stored_project, cos=0.84)
    _turn(store, monkeypatch, turn_project, polarity="dislike")
    retired = _row(conn, target)
    assert retired[4] is not None  # t_invalid
    live = _live(conn)
    assert [(r[1], r[2]) for r in live] == [(turn_project, _NEW)]
    assert retired[5] == live[0][0]  # superseded_by points at the new row


# ---- the filter runs before the LIMIT ------------------------------------------------


def test_a_same_class_duplicate_is_found_behind_nearer_rows_of_another_class(
    conn, store, monkeypatch
):
    """Seven rows the new item's readers do not share sit nearer than the one duplicate
    they do: a filter applied to the five nearest rows would find nothing to re-assert."""
    register_restricted(conn, ["alpha"])
    nearer = [
        _seed(store, "alpha", cos=0.99 - 0.001 * i, other=_SLOT + 2 + i, pref=f"User likes #{i}")
        for i in range(7)
    ]
    duplicate = _seed(store, "gamma", cos=0.93)
    before = {i: _row(conn, i) for i in nearer}
    count = _row(conn, duplicate)[2]
    _turn(store, monkeypatch, "delta")
    assert _row(conn, duplicate)[2] == count + 1
    assert {i: _row(conn, i) for i in nearer} == before
    assert len(_live(conn)) == 8


# ---- case is folded on every side ----------------------------------------------------


@pytest.mark.parametrize("stored_project", ["workproj", "WORKPROJ", "WorkProj"])
def test_a_stored_project_is_on_the_allowlist_whatever_its_case(
    conn, store, monkeypatch, stored_project
):
    register_restricted(conn, ["WorkProj"])
    target = _seed(store, stored_project)
    before = _row(conn, target)
    _turn(store, monkeypatch, "gamma")  # off the list, so another class
    _assert_other_class_untouched(conn, target, before, "gamma")


def test_two_spellings_of_an_allowlisted_project_are_one_class(conn, store, monkeypatch):
    register_restricted(conn, ["WorkProj"])
    target = _seed(store, "workproj")
    count = _row(conn, target)[2]
    _turn(store, monkeypatch, "WORKPROJ")
    assert _row(conn, target)[2] == count + 1
    assert [r[0] for r in _live(conn)] == [target]


def test_an_allowlisted_spelling_is_another_class_than_a_project_off_the_list(
    conn, store, monkeypatch
):
    register_restricted(conn, ["WorkProj"])
    target = _seed(store, "gamma")
    before = _row(conn, target)
    _turn(store, monkeypatch, "WORKPROJ")
    _assert_other_class_untouched(conn, target, before, "WORKPROJ")


# ---- one class per approved restricted device, not per union -------------------------


@pytest.mark.parametrize(("stored_project", "turn_project"), [("alpha", "beta"), ("beta", "alpha")])
def test_projects_on_different_devices_allowlists_are_different_classes(
    conn, store, monkeypatch, stored_project, turn_project
):
    """Each project is on one device's list and off the other's. The union of the two
    lists holds both, so a union check would call them the same."""
    register_restricted(conn, ["alpha"], "first-device")
    register_restricted(conn, ["beta"], "second-device")
    target = _seed(store, stored_project)
    before = _row(conn, target)
    _turn(store, monkeypatch, turn_project)
    _assert_other_class_untouched(conn, target, before, turn_project)


def test_two_projects_on_the_same_device_only_are_one_class(conn, store, monkeypatch):
    register_restricted(conn, ["alpha", "alpha2"], "first-device")
    register_restricted(conn, ["beta"], "second-device")
    target = _seed(store, "alpha")
    count = _row(conn, target)[2]
    _turn(store, monkeypatch, "alpha2")
    assert _row(conn, target)[2] == count + 1
    assert [r[0] for r in _live(conn)] == [target]


# ---- unreadable allowlists fall back to the same project -----------------------------


@pytest.fixture()
def unreadable_allowlists(monkeypatch):
    def _boom(self):
        raise psycopg.OperationalError("surfaces unreadable")

    monkeypatch.setattr(Database, "restricted_surface_allowlists", _boom)


@pytest.mark.parametrize(
    ("stored_project", "turn_project"),
    [("Alpha", "ALPHA"), (None, None), ("", None), (None, "")],
)
def test_unreadable_allowlists_still_reassert_the_same_project(
    conn, store, monkeypatch, unreadable_allowlists, stored_project, turn_project
):
    target = _seed(store, stored_project)
    count = _row(conn, target)[2]
    _turn(store, monkeypatch, turn_project)
    assert _row(conn, target)[2] == count + 1
    assert [r[0] for r in _live(conn)] == [target]


@pytest.mark.parametrize(
    ("stored_project", "turn_project"),
    [("gamma", "delta"), ("alpha", None), (None, "alpha")],
)
def test_unreadable_allowlists_store_another_project_as_its_own_row(
    conn, store, monkeypatch, unreadable_allowlists, stored_project, turn_project
):
    """Two projects off every list are one class when the lists can be read; with the
    lists unreadable they are not, so an outage never widens what a write may touch."""
    target = _seed(store, stored_project)
    before = _row(conn, target)
    _turn(store, monkeypatch, turn_project)
    _assert_other_class_untouched(conn, target, before, turn_project)


# ---- a missing project is on no allowlist --------------------------------------------


def test_a_missing_project_on_both_sides_is_one_class(conn, store, monkeypatch):
    register_restricted(conn, ["alpha"])
    target = _seed(store, None)
    count = _row(conn, target)[2]
    _turn(store, monkeypatch, None)
    assert _row(conn, target)[2] == count + 1
    assert [r[0] for r in _live(conn)] == [target]


@pytest.mark.parametrize(("stored_project", "turn_project"), [(None, "alpha"), ("alpha", None)])
def test_a_missing_project_is_another_class_than_an_allowlisted_one(
    conn, store, monkeypatch, stored_project, turn_project
):
    register_restricted(conn, ["alpha"])
    target = _seed(store, stored_project)
    before = _row(conn, target)
    _turn(store, monkeypatch, turn_project)
    _assert_other_class_untouched(conn, target, before, turn_project)


# ---- only approved restricted devices split classes ----------------------------------


def test_a_revoked_or_full_trust_device_does_not_split_classes(conn, store, monkeypatch):
    register_device(conn, "tok-revoked", trust="restricted", projects=["alpha"], status="revoked")
    register_device(conn, "tok-full", trust="full", projects=["alpha"])
    target = _seed(store, "alpha")
    count = _row(conn, target)[2]
    _turn(store, monkeypatch, "gamma")
    assert _row(conn, target)[2] == count + 1
    assert [r[0] for r in _live(conn)] == [target]


def test_with_no_restricted_device_projects_are_one_class(conn, store, monkeypatch):
    target = _seed(store, "alpha", cos=0.85)
    before = _row(conn, target)
    judge = _turn(store, monkeypatch, "gamma")
    after = _row(conn, target)
    assert after[0] == _MERGED and after[1] != before[1]  # merged across projects, as before
    assert after[2] == before[2] + 1
    assert judge.confirms == 1
    assert [r[0] for r in _live(conn)] == [target]


def test_allowlists_come_one_per_approved_restricted_device(conn, store):
    register_restricted(conn, ["alpha", "beta"], "first-device")
    register_restricted(conn, [], "second-device")
    register_device(conn, "tok-full", trust="full", projects=["gamma"])
    register_device(conn, "tok-revoked", trust="restricted", projects=["delta"], status="revoked")
    assert sorted(store.restricted_surface_allowlists()) == [[], ["alpha", "beta"]]
