"""Tests for the notes store (schema 041) + the reconcile path (ingestion/notes.py).

Two halves, mirroring the preferences split:
  * DB-backed accessor tests (insert / cosine KNN / update / supersede / board listing /
    source_ref probe) against the shared Postgres test DB.
  * reconcile_note unit tests with the DB, embedder, and LLM all stubbed — the threshold,
    the three-way verdict and the insert-when-unsure rule, no network and no real model
    calls."""

from __future__ import annotations

from datetime import UTC, datetime

import psycopg
import pytest

import ingestion.notes as notes_mod
from ingestion import pg_pool
from ingestion.db import Database
from ingestion.db_notes import (
    NOTES_HISTORY_PROBE,
    get_notes_by_ids,
    search_live_notes,
    select_board_notes,
)
from ingestion.notes import reconcile_note
from ingestion.settings import KG_OWNER_ID
from tests.helpers.embed import GROUP, onehot

pytestmark = pytest.mark.db


def _wipe(conn):
    conn.execute("DELETE FROM notes")


def _insert(db, *, type="user", hook="User prefers X", body="Body.", slot=1, **kw):
    args = {
        "owner_id": KG_OWNER_ID,
        "group_id": GROUP,
        "project": None,
        "embedding": onehot(slot),
        "embed_model": "test",
        "source_ref": None,
    }
    args.update(kw)
    return db.insert_note(type=type, hook=hook, body=body, **args)


# ---------------------------------------------------------------------------
# DB accessors
# ---------------------------------------------------------------------------


def test_insert_and_cosine_knn(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    a = _insert(db, hook="User prefers bullet lists", slot=1, source_ref="ep:1")
    _insert(db, hook="User dislikes em-dashes", slot=2, source_ref="ep:2")
    hits = db.find_live_notes(KG_OWNER_ID, GROUP, onehot(1), limit=5)
    assert hits[0]["id"] == a
    assert hits[0]["sim"] == pytest.approx(1.0, abs=1e-3)
    assert hits[1]["sim"] == pytest.approx(0.0, abs=1e-3)
    assert set(hits[0]) == {"id", "hook", "body", "type", "project", "updated_at", "sim"}
    db.close()
    _wipe(conn)


def test_update_refreshes_hook_body_and_updated_at(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    nid = _insert(db, hook="Old hook", body="Old body.", slot=3, source_ref="ep:1")
    before = conn.execute("SELECT updated_at FROM notes WHERE id = %s", (nid,)).fetchone()[0]
    db.update_note(
        nid,
        hook="New hook",
        body="New body.",
        embedding=onehot(4),
        embed_model="test2",
        source_ref="ep:2",
    )
    row = conn.execute(
        "SELECT hook, body, embed_model, source_ref, updated_at FROM notes WHERE id = %s", (nid,)
    ).fetchone()
    assert row[0] == "New hook" and row[1] == "New body." and row[2] == "test2"
    assert row[3] == "ep:2"  # provenance follows the wording
    assert row[4] > before
    # The refreshed embedding moved the note to slot 4 in KNN space.
    hits = db.find_live_notes(KG_OWNER_ID, GROUP, onehot(4), limit=1)
    assert hits[0]["id"] == nid and hits[0]["sim"] == pytest.approx(1.0, abs=1e-3)
    db.close()
    _wipe(conn)


def test_update_keeps_the_replaced_wording_in_history(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    nid = _insert(db, hook="Old hook", body="Old body.", slot=3, source_ref="ep:1")
    db.update_note(
        nid,
        hook="Mid hook",
        body="Mid body.",
        embedding=onehot(3),
        embed_model="t",
        source_ref="ep:2",
    )
    # Restating the same text again changes nothing worth keeping.
    db.update_note(
        nid,
        hook="Mid hook",
        body="Mid body.",
        embedding=onehot(3),
        embed_model="t",
        source_ref="ep:3",
    )
    db.update_note(nid, hook="New hook", body="New body.", embedding=onehot(3), embed_model="t")
    rows = conn.execute(
        "SELECT hook, body, source_ref FROM notes_history WHERE note_id = %s ORDER BY id", (nid,)
    ).fetchall()
    assert rows == [("Old hook", "Old body.", "ep:1"), ("Mid hook", "Mid body.", "ep:3")]
    # source_ref=None keeps the stored provenance.
    assert (
        conn.execute("SELECT source_ref FROM notes WHERE id = %s", (nid,)).fetchone()[0] == "ep:3"
    )
    db.close()
    _wipe(conn)


def test_update_without_the_history_table_still_rewrites(conn, db_url):
    """Optional migration: before notes_history exists the text is rewritten in place, and
    source_ref stays on the episode that carries the wording that was just replaced."""
    _wipe(conn)
    db = Database(db_url)
    nid = _insert(db, hook="Old hook", body="Old body.", slot=3, source_ref="ep:1")
    conn.execute("ALTER TABLE notes_history RENAME TO notes_history_away")
    try:
        NOTES_HISTORY_PROBE.reset()
        db.update_note(
            nid,
            hook="New hook",
            body="New body.",
            embedding=None,
            embed_model=None,
            source_ref="ep:2",
        )
        assert conn.execute(
            "SELECT hook, source_ref FROM notes WHERE id = %s", (nid,)
        ).fetchone() == ("New hook", "ep:1")
    finally:
        conn.execute("ALTER TABLE notes_history_away RENAME TO notes_history")
        NOTES_HISTORY_PROBE.reset()
    db.close()
    _wipe(conn)


def test_supersede_retires_old_and_links_lineage(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    old = _insert(db, hook="Project uses provider A", type="project", project="proj", slot=5)
    new = _insert(db, hook="Project moved to provider B", type="project", project="proj", slot=5)
    db.supersede_note(old, new)
    row = conn.execute("SELECT superseded_by FROM notes WHERE id = %s", (old,)).fetchone()
    assert row[0] == new  # lineage FK points at the replacement
    live_ids = {h["id"] for h in db.find_live_notes(KG_OWNER_ID, GROUP, onehot(5), limit=5)}
    assert new in live_ids and old not in live_ids
    db.close()
    _wipe(conn)


def test_type_check_rejects_invalid(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert(db, type="bogus")
    db.close()
    _wipe(conn)


def test_group_and_owner_scoping(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    mine = _insert(db, hook="Mine, technical", slot=6)
    _insert(db, hook="Mine, personal", group_id="personal", slot=6)
    _insert(db, hook="Someone else's", owner_id="other", slot=6)
    hits = db.find_live_notes(KG_OWNER_ID, GROUP, onehot(6), limit=5)
    assert [h["id"] for h in hits] == [mine]
    db.close()
    _wipe(conn)


def test_null_embedding_insert_ok_and_skipped_by_knn(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    nid = _insert(db, hook="Keyless note", embedding=None, embed_model="ignored")
    row = conn.execute("SELECT embedding, embed_model FROM notes WHERE id = %s", (nid,)).fetchone()
    assert row[0] is None and row[1] is None  # embed_model nulled alongside the vector
    assert db.find_live_notes(KG_OWNER_ID, GROUP, onehot(1), limit=5) == []
    db.close()
    _wipe(conn)


def _board(db, project):
    with db._conn() as conn:
        return select_board_notes(conn, KG_OWNER_ID, project)


def test_board_notes_ordering_and_project_filter(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    ref = _insert(db, type="reference", hook="See runbook", slot=10)
    usr = _insert(db, type="user", hook="User is left-handed", slot=11)
    fb = _insert(db, type="feedback", hook="Never use tables", slot=12)
    pa = _insert(db, type="project", project="proj-a", hook="A uses provider X", slot=13)
    _insert(db, type="project", project="proj-b", hook="B uses provider Y", slot=14)
    retired = _insert(db, type="feedback", hook="Old rule", slot=15)
    db.supersede_note(retired, fb)

    rows = _board(db, "proj-a")
    # feedback -> user -> project (only proj-a's) -> reference; the retired row is gone.
    assert [r["id"] for r in rows] == [fb, usr, pa, ref]
    assert set(rows[0]) == {"id", "hook", "type", "project", "audience", "updated_at"}

    # No project scope: the global set only (project = NULL matches nothing).
    assert [r["id"] for r in _board(db, None)] == [fb, usr, ref]
    db.close()
    _wipe(conn)


def test_get_notes_by_ids(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    a = _insert(db, hook="First", body="Body one.", slot=16)
    b = _insert(db, hook="Second", body="Body two.", slot=17)
    rows = db.get_notes_by_ids([a, b, 999999999])
    assert [r["id"] for r in rows] == [a, b]  # unknown ids silently dropped
    assert rows[0]["body"] == "Body one."
    assert set(rows[0]) == {"id", "hook", "body", "type", "project", "audience", "updated_at"}
    assert db.get_notes_by_ids([]) == []
    db.close()
    _wipe(conn)


def test_recall_reads_notes_on_an_autocommit_borrow(conn, db_url):
    """search_live_notes and get_notes_by_ids take a connection, so recall runs each as one
    statement on a pooled autocommit borrow, scoped like the board."""
    _wipe(conn)
    db = Database(db_url)
    usr = _insert(db, type="user", hook="User fact", slot=18)
    pa = _insert(db, type="project", project="proj-a", hook="A fact", slot=19)
    _insert(db, type="project", project="proj-b", hook="B fact", slot=20)
    retired = _insert(db, type="user", hook="Retired fact", slot=18)
    db.supersede_note(retired, usr)
    with pg_pool.connection(db_url) as borrowed:
        hits = search_live_notes(borrowed, KG_OWNER_ID, "proj-a", onehot(19), limit=10)
        assert hits[0]["id"] == pa  # nearest first
        assert {h["id"] for h in hits} == {usr, pa}  # proj-b and the retired row are out
        assert [h["id"] for h in search_live_notes(borrowed, KG_OWNER_ID, None, onehot(19))] == [
            usr
        ]
        assert borrowed.info.transaction_status != psycopg.pq.TransactionStatus.INTRANS
        assert [r["id"] for r in get_notes_by_ids(borrowed, [pa, usr, 999999999])] == [usr, pa]
        assert get_notes_by_ids(borrowed, []) == []
    db.close()
    _wipe(conn)


# ---------------------------------------------------------------------------
# reconcile_note — DB, embedder, and LLM stubbed
# ---------------------------------------------------------------------------


class _NotesDB:
    """Records writes; returns configurable KNN candidates."""

    def __init__(self, candidates=()):
        self._candidates = list(candidates)
        self.knn_calls = 0
        self.inserted: list[dict] = []
        self.updated: list[dict] = []
        self.superseded: list[tuple[int, int]] = []
        self._next_id = 500

    def find_live_notes(self, owner_id, group_id, embedding, limit=5):
        self.knn_calls += 1
        return self._candidates

    def insert_note(self, **kw):
        self.inserted.append(kw)
        self._next_id += 1
        return self._next_id

    def update_note(
        self, note_id, *, hook, body, embedding, embed_model, source_ref=None, audience=None
    ):
        self.updated.append(
            {
                "note_id": note_id,
                "hook": hook,
                "body": body,
                "source_ref": source_ref,
                "audience": audience,
            }
        )

    def supersede_note(self, old_id, new_id):
        self.superseded.append((old_id, new_id))

    def restricted_surface_projects(self):
        """No registered restricted surface -> audience derivation falls to 'personal'.
        The precedence rules themselves are exercised in test_audience_scoping.py."""
        return []


class _StubEmb:
    model_name = "voyage-4-large"

    def embed(self, texts, task):
        return [[0.0] * 8 for _ in texts]


class _FailEmb:
    model_name = "voyage-4-large"

    def embed(self, texts, task):
        raise RuntimeError("embed backend down")


def _cand(sim: float, type: str = "user", note_id: int = 42, project: str | None = None) -> dict:
    return {
        "id": note_id,
        "hook": "User prefers dark mode",
        "body": "Existing body.",
        "type": type,
        "project": project,
        "updated_at": datetime(2026, 1, 1, tzinfo=UTC),
        "sim": sim,
    }


_DUP = {"verdict": "DUPLICATE", "keep": None}
_DISTINCT = {"verdict": "DISTINCT", "keep": None}
_NEW_SURVIVES = {"verdict": "CORRECTION", "keep": 2}
_OLD_SURVIVES = {"verdict": "CORRECTION", "keep": 1}


def _run(monkeypatch, *, candidates=(), verdict=_DUP, type="user", project=None, raise_llm=False):
    """Drive reconcile_note with everything stubbed; returns (result, db, llm_calls)."""
    llm_calls: list[dict] = []

    def _fake_parse(*a, **k):
        llm_calls.append(k)
        if raise_llm:
            raise RuntimeError("LLM down")
        return verdict

    monkeypatch.setattr(notes_mod, "_group_for", lambda project: "technical")
    monkeypatch.setattr(notes_mod, "parse_with_retry", _fake_parse)
    db = _NotesDB(candidates)
    result = reconcile_note(
        db,
        _StubEmb(),
        object(),
        hook="User prefers light mode",
        body="New body.",
        type=type,
        project=project,
        source_ref="ep:9",
    )
    return result, db, llm_calls


def test_reconcile_creates_when_no_candidates(monkeypatch):
    result, db, llm_calls = _run(monkeypatch, candidates=[])
    assert result == {
        "outcome": "created",
        "note_id": 501,
        "prev_id": None,
        "audience": "personal",
    }
    assert len(db.inserted) == 1 and db.updated == [] and db.superseded == []
    assert db.inserted[0]["source_ref"] == "ep:9"
    assert llm_calls == []  # no candidate -> no judge call


def test_reconcile_creates_below_threshold(monkeypatch):
    result, db, llm_calls = _run(monkeypatch, candidates=[_cand(0.5)])
    assert result["outcome"] == "created"
    assert len(db.inserted) == 1 and db.updated == [] and llm_calls == []


def test_reconcile_updates_on_a_duplicate_verdict(monkeypatch):
    result, db, llm_calls = _run(monkeypatch, candidates=[_cand(0.92)], verdict=_DUP)
    assert result == {"outcome": "updated", "note_id": 42, "prev_id": None, "audience": None}
    # audience=None on the UPDATE: a restatement preserves the note's stored tier.
    assert db.updated == [
        {
            "note_id": 42,
            "hook": "User prefers light mode",
            "body": "New body.",
            "source_ref": "ep:9",
            "audience": None,
        }
    ]
    assert db.inserted == [] and db.superseded == []
    assert len(llm_calls) == 1


def test_reconcile_supersedes_when_the_new_note_corrects_the_old(monkeypatch):
    result, db, llm_calls = _run(monkeypatch, candidates=[_cand(0.92)], verdict=_NEW_SURVIVES)
    assert result == {
        "outcome": "superseded",
        "note_id": 501,
        "prev_id": 42,
        "audience": "personal",
    }
    assert len(db.inserted) == 1
    assert db.superseded == [(42, 501)]  # old retired, pointed at the fresh row
    assert db.updated == [] and len(llm_calls) == 1


@pytest.mark.parametrize("verdict", [_DISTINCT, _OLD_SURVIVES], ids=["distinct", "old-survives"])
def test_reconcile_inserts_beside_a_near_note_that_is_not_a_restatement(monkeypatch, verdict):
    """Two rules on one topic are two notes. The judge saying so, or naming the OLD note as
    the corrected-by one (the user just asserted the new one), leaves both live for the
    nightly lane to settle."""
    result, db, llm_calls = _run(monkeypatch, candidates=[_cand(0.92)], verdict=verdict)
    assert result["outcome"] == "created"
    assert len(db.inserted) == 1 and db.updated == [] and db.superseded == []
    assert len(llm_calls) == 1


def test_reconcile_llm_failure_inserts(monkeypatch):
    """A failed judgement must not overwrite a note: a missed merge costs one board line."""
    result, db, llm_calls = _run(monkeypatch, candidates=[_cand(0.92)], raise_llm=True)
    assert result["outcome"] == "created"
    assert len(db.inserted) == 1 and db.updated == [] and db.superseded == []
    assert len(llm_calls) == 1


def test_reconcile_judges_the_old_note_as_note_1(monkeypatch):
    seen: list[str] = []

    def _capture(llm, *, base_prompt, **kw):
        seen.append(base_prompt)
        return _DUP

    monkeypatch.setattr(notes_mod, "_group_for", lambda project: "technical")
    monkeypatch.setattr(notes_mod, "parse_with_retry", _capture)
    reconcile_note(
        _NotesDB([_cand(0.92)]),
        _StubEmb(),
        object(),
        hook="User prefers light mode",
        body="New body.",
        type="user",
        project=None,
        source_ref=None,
    )
    old, new = (
        seen[0].index("hook: User prefers dark mode"),
        seen[0].index("hook: User prefers light"),
    )
    assert old < new  # NOTE 1 is the stored note, NOTE 2 the one being written


def test_reconcile_exact_restatement_needs_no_judge(monkeypatch):
    """A replayed spool write carries the stored text verbatim; it converges on the note
    without a model call, so an LLM outage cannot fork a duplicate."""
    cand = _cand(1.0) | {"hook": "User prefers light mode", "body": "New body."}
    result, db, llm_calls = _run(monkeypatch, candidates=[cand], raise_llm=True)
    assert result["outcome"] == "updated" and result["note_id"] == 42
    assert db.inserted == [] and llm_calls == []


def test_reconcile_uses_the_best_same_type_candidate(monkeypatch):
    """The nearest note is another type; the second is a same-type restatement."""
    nearer_other_type = _cand(0.97, type="feedback", note_id=41)
    result, _db, llm_calls = _run(
        monkeypatch, candidates=[nearer_other_type, _cand(0.88)], verdict=_DUP
    )
    assert result["outcome"] == "updated" and result["note_id"] == 42
    assert len(llm_calls) == 1


def test_reconcile_never_restates_another_projects_note(monkeypatch):
    """A project note is only ever a restatement of its own project's note."""
    other = _cand(0.95, type="project", project="alpha")
    result, db, llm_calls = _run(
        monkeypatch, candidates=[other], verdict=_DUP, type="project", project="beta"
    )
    assert result["outcome"] == "created" and llm_calls == [] and db.updated == []
    result, db, llm_calls = _run(
        monkeypatch, candidates=[other], verdict=_DUP, type="project", project="alpha"
    )
    assert result["outcome"] == "updated"


def test_reconcile_threshold(monkeypatch):
    monkeypatch.setattr(notes_mod, "_SIM_MATCH", 0.95)
    result, _db, llm_calls = _run(monkeypatch, candidates=[_cand(0.90)])
    assert result["outcome"] == "created"  # 0.90 < raised bar
    assert llm_calls == []


def test_reconcile_keyless_embedder_none(monkeypatch):
    monkeypatch.setattr(notes_mod, "_group_for", lambda project: "technical")
    db = _NotesDB([_cand(0.99)])  # would match — but keyless skips dedup entirely
    result = reconcile_note(
        db,
        None,
        object(),
        hook="Keyless note",
        body="Body.",
        type="user",
        project=None,
        source_ref=None,
    )
    assert result["outcome"] == "created"
    assert db.knn_calls == 0  # no embedding -> no KNN
    assert db.inserted[0]["embedding"] is None and db.inserted[0]["embed_model"] is None


def test_reconcile_embed_failure_degrades_to_insert(monkeypatch):
    monkeypatch.setattr(notes_mod, "_group_for", lambda project: "technical")
    db = _NotesDB([_cand(0.99)])
    result = reconcile_note(
        db,
        _FailEmb(),
        object(),
        hook="Embed-down note",
        body="Body.",
        type="user",
        project=None,
        source_ref=None,
    )
    assert result["outcome"] == "created"
    assert db.knn_calls == 0
    assert db.inserted[0]["embedding"] is None


def test_reconcile_rejects_invalid_type():
    with pytest.raises(ValueError, match="invalid note type"):
        reconcile_note(
            _NotesDB(),
            None,
            object(),
            hook="h",
            body="b",
            type="bogus",
            project=None,
            source_ref=None,
        )
