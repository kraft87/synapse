"""A note write and the two tiers (notes.audience: 'personal' / 'work-safe'), against the
real store.

The rule under test: a write never edits a live note of another tier, and a restricted
caller never sees, edits or retires a personal note. A restricted device is served only
'work-safe' notes, so a note that changes tier by being rephrased would hand its new text
to a device that was never meant to read it (or hide a note from one that was).

Every test is a real ``reconcile_note`` call: the embedder puts every text on one slot,
so a note seeded on that slot is an exact match, and the confirm call is a stub that says
what the test needs ("same", "contradicts", or an outage).
"""

from __future__ import annotations

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

import ingestion.notes as notes_mod  # noqa: E402
from ingestion.db import Database  # noqa: E402
from ingestion.notes import _OWNER, reconcile_note  # noqa: E402
from tests.helpers.embed import OneHotEmbedder, onehot  # noqa: E402
from tests.helpers.surfaces import clear_surfaces, register_restricted  # noqa: E402

_OLD_HOOK = "User likes terse status updates"
_OLD_BODY = "Keep status updates to two lines."
_NEW_HOOK = "User likes terse status updates and tracks a private hobby budget"
_NEW_BODY = "Two-line updates; the hobby budget is tracked separately."
_NEWER_HOOK = "User likes terse status updates and has moved the hobby budget to a new tool"
_NEWER_BODY = "Two-line updates; the hobby budget now lives in a new tool."
_SLOT = 40


@pytest.fixture()
def tiers(conn, db_url):
    """A Database on the scratch store, with no notes and no registered surfaces."""
    conn.execute("DELETE FROM notes")
    clear_surfaces(conn)
    db = Database(db_url)
    yield db
    db.close()
    conn.execute("DELETE FROM notes")
    clear_surfaces(conn)


@pytest.fixture()
def confirm(monkeypatch):
    """Set what the relation confirm answers; returns the list of its calls. ``during`` runs
    inside the call, which is where another writer can change a note the write has read."""
    calls: list[dict] = []

    def _answer(relation, during=None):
        calls.clear()

        def _fake(*a, **k):
            calls.append(k)
            if during is not None:
                during()
            if isinstance(relation, Exception):
                raise relation
            return relation

        monkeypatch.setattr(notes_mod, "parse_with_retry", _fake)
        return calls

    return _answer


def _seed(db, *, audience, hook=_OLD_HOOK, body=_OLD_BODY, type="user", slot=_SLOT):
    return db.insert_note(
        owner_id=_OWNER,
        project=None,
        type=type,
        hook=hook,
        body=body,
        embedding=onehot(slot),
        embed_model="test",
        source_ref=None,
        audience=audience,
    )


def _write(db, *, hook=_NEW_HOOK, body=_NEW_BODY, type="user", slot=_SLOT, **kw):
    kw.setdefault("project", None)
    return reconcile_note(
        db, OneHotEmbedder(slot), object(), hook=hook, body=body, type=type, source_ref="ep:1", **kw
    )


def _row(conn, note_id):
    return conn.execute(
        "SELECT hook, body, audience, superseded_by FROM notes WHERE id = %s", (note_id,)
    ).fetchone()


def _live(conn):
    rows = conn.execute(
        "SELECT id, audience, hook FROM notes WHERE superseded_by IS NULL ORDER BY id"
    ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def _count_containing(conn, audience, needle):
    return conn.execute(
        "SELECT count(*) FROM notes WHERE audience = %s AND (hook LIKE %s OR body LIKE %s)",
        (audience, f"%{needle}%", f"%{needle}%"),
    ).fetchone()[0]


# ---------------------------------------------------------------------------
# Full trust: a restatement never edits a note of the other tier
# ---------------------------------------------------------------------------


def test_personal_write_never_edits_a_matching_work_safe_note(conn, tiers, confirm):
    register_restricted(conn, ["alpha"])
    work = _seed(tiers, audience="work-safe")
    confirm("same")
    res = _write(tiers, project="gamma")  # not on any allowlist: derives personal
    assert res["outcome"] == "created" and res["note_id"] != work
    assert res["audience"] == "personal" and res["prev_id"] is None
    assert _row(conn, work) == (_OLD_HOOK, _OLD_BODY, "work-safe", None)
    assert _row(conn, res["note_id"]) == (_NEW_HOOK, _NEW_BODY, "personal", None)
    assert _count_containing(conn, "work-safe", "hobby budget") == 0


@pytest.mark.parametrize("outage", ["llm_error", "kill_switch"])
def test_a_confirm_that_cannot_answer_never_edits_the_other_tier(
    conn, tiers, confirm, monkeypatch, outage
):
    """An LLM failure and SYNAPSE_NOTES_CONFIRM=0 both answer "same". Across tiers that
    answer must still leave the other tier's note alone."""
    register_restricted(conn, ["alpha"])
    work = _seed(tiers, audience="work-safe")
    if outage == "llm_error":
        confirm(RuntimeError("LLM down"))
    else:
        monkeypatch.setenv("SYNAPSE_NOTES_CONFIRM", "0")
    res = _write(tiers, project="gamma")
    assert res["outcome"] == "created" and res["note_id"] != work
    assert _row(conn, work) == (_OLD_HOOK, _OLD_BODY, "work-safe", None)
    assert _row(conn, res["note_id"])[2] == "personal"
    assert _count_containing(conn, "work-safe", "hobby budget") == 0


def test_a_second_restatement_updates_its_own_tiers_twin(conn, tiers, confirm):
    """After one cross-tier restatement the owner has a twin in each tier. Both sit on the
    same slot, so only the tier filter decides which one the next restatement updates."""
    register_restricted(conn, ["alpha"])
    work = _seed(tiers, audience="work-safe")
    confirm("same")
    first = _write(tiers, project="gamma")
    second = _write(tiers, hook=_NEWER_HOOK, body=_NEWER_BODY, project="gamma")
    assert second["outcome"] == "updated" and second["note_id"] == first["note_id"]
    assert _row(conn, first["note_id"]) == (_NEWER_HOOK, _NEWER_BODY, "personal", None)
    assert _row(conn, work) == (_OLD_HOOK, _OLD_BODY, "work-safe", None)
    assert [(a, h) for _, a, h in _live(conn)] == [
        ("work-safe", _OLD_HOOK),
        ("personal", _NEWER_HOOK),
    ]


def test_personal_write_that_contradicts_a_work_safe_note_retires_it(conn, tiers, confirm):
    register_restricted(conn, ["alpha"])
    work = _seed(tiers, audience="work-safe")
    confirm("contradicts")
    res = _write(tiers, project="gamma")
    assert res["outcome"] == "superseded" and res["prev_id"] == work
    assert res["audience"] == "personal"
    assert _row(conn, work)[3] == res["note_id"]  # retired, lineage kept
    assert _row(conn, res["note_id"]) == (_NEW_HOOK, _NEW_BODY, "personal", None)
    assert [a for _, a, _ in _live(conn)] == ["personal"]


def test_work_safe_write_never_edits_a_matching_personal_note(conn, tiers, confirm):
    register_restricted(conn, ["alpha"])
    personal = _seed(tiers, audience="personal")
    confirm("same")
    res = _write(tiers, project="alpha")  # on the allowlist: derives work-safe
    assert res["outcome"] == "created" and res["note_id"] != personal
    assert res["audience"] == "work-safe"
    assert _row(conn, personal) == (_OLD_HOOK, _OLD_BODY, "personal", None)
    assert _row(conn, res["note_id"]) == (_NEW_HOOK, _NEW_BODY, "work-safe", None)
    assert _count_containing(conn, "personal", "hobby budget") == 0


@pytest.mark.parametrize(
    ("stored", "asked"), [("personal", "work-safe"), ("work-safe", "personal")]
)
def test_an_explicit_audience_still_moves_the_matching_note_of_the_other_tier(
    conn, tiers, confirm, stored, asked
):
    """Reclassification stays possible, but only as an explicit act."""
    note = _seed(tiers, audience=stored)
    confirm("same")
    res = _write(tiers, audience=asked)
    assert res == {"outcome": "updated", "note_id": note, "prev_id": None, "audience": asked}
    assert _row(conn, note) == (_NEW_HOOK, _NEW_BODY, asked, None)
    assert [n for n, _, _ in _live(conn)] == [note]


# ---------------------------------------------------------------------------
# A restricted caller works inside the work-safe tier only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("relation", ["same", "contradicts"])
def test_restricted_write_never_matches_a_personal_note(conn, tiers, confirm, relation):
    personal = _seed(tiers, audience="personal")
    calls = confirm(relation)
    res = _write(tiers, caller_restricted=True)
    assert res["outcome"] == "created" and res["note_id"] != personal
    assert res["audience"] == "work-safe" and res["prev_id"] is None
    assert calls == []  # the personal note was never a candidate, so nothing was asked
    assert _row(conn, personal) == (_OLD_HOOK, _OLD_BODY, "personal", None)
    assert _row(conn, res["note_id"]) == (_NEW_HOOK, _NEW_BODY, "work-safe", None)
    assert {a for _, a, _ in _live(conn)} == {"personal", "work-safe"}


def test_restricted_write_naming_personal_still_never_matches_a_personal_note(conn, tiers, confirm):
    personal = _seed(tiers, audience="personal")
    calls = confirm("same")
    res = _write(tiers, caller_restricted=True, audience="personal")
    assert res["outcome"] == "created" and res["note_id"] != personal
    assert calls == []
    assert _row(conn, personal) == (_OLD_HOOK, _OLD_BODY, "personal", None)
    assert [n for n, _, _ in _live(conn)] == [personal, res["note_id"]]


def test_restricted_write_restating_a_work_safe_note_updates_it(conn, tiers, confirm):
    work = _seed(tiers, audience="work-safe")
    personal = _seed(tiers, audience="personal")  # same slot: the tier filter picks
    confirm("same")
    res = _write(tiers, caller_restricted=True)
    assert res == {"outcome": "updated", "note_id": work, "prev_id": None, "audience": None}
    assert _row(conn, work) == (_NEW_HOOK, _NEW_BODY, "work-safe", None)
    assert _row(conn, personal) == (_OLD_HOOK, _OLD_BODY, "personal", None)


# ---------------------------------------------------------------------------
# The note can change between the read and the write
# ---------------------------------------------------------------------------


def _set_audience(conn, note_id, audience):
    conn.execute("UPDATE notes SET audience = %s WHERE id = %s", (audience, note_id))


def test_a_restatement_never_lands_on_a_note_that_turned_personal_meanwhile(conn, tiers, confirm):
    """A restricted write read a work-safe note; while the confirm call ran, another write
    made it personal. The restatement must not overwrite that personal text."""
    work = _seed(tiers, audience="work-safe")
    confirm("same", during=lambda: _set_audience(conn, work, "personal"))
    res = _write(tiers, caller_restricted=True)
    assert res["outcome"] == "created" and res["note_id"] != work
    assert res["audience"] == "work-safe" and res["prev_id"] is None
    assert _row(conn, work) == (_OLD_HOOK, _OLD_BODY, "personal", None)
    assert _row(conn, res["note_id"]) == (_NEW_HOOK, _NEW_BODY, "work-safe", None)
    assert _count_containing(conn, "personal", "hobby budget") == 0


def test_a_contradiction_never_retires_a_note_that_turned_personal_meanwhile(conn, tiers, confirm):
    work = _seed(tiers, audience="work-safe")
    confirm("contradicts", during=lambda: _set_audience(conn, work, "personal"))
    res = _write(tiers, caller_restricted=True)
    assert res["outcome"] == "created" and res["note_id"] != work
    assert res["audience"] == "work-safe" and res["prev_id"] is None
    assert _row(conn, work) == (_OLD_HOOK, _OLD_BODY, "personal", None)  # still live
    assert _row(conn, res["note_id"]) == (_NEW_HOOK, _NEW_BODY, "work-safe", None)
    assert [n for n, _, _ in _live(conn)] == [work, res["note_id"]]


def test_a_restatement_never_lands_on_a_note_retired_meanwhile(conn, tiers, confirm):
    old = _seed(tiers, audience="personal")
    successor = _seed(
        tiers, audience="personal", hook="Unrelated", body="Unrelated.", slot=_SLOT + 1
    )
    confirm(
        "same",
        during=lambda: conn.execute(
            "UPDATE notes SET superseded_by = %s WHERE id = %s", (successor, old)
        ),
    )
    res = _write(tiers)
    assert res["outcome"] == "created" and res["note_id"] not in (old, successor)
    assert _row(conn, old) == (_OLD_HOOK, _OLD_BODY, "personal", successor)
    assert _row(conn, res["note_id"]) == (_NEW_HOOK, _NEW_BODY, "personal", None)
