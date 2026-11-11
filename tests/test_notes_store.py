"""Tests for the notes store (schema 041) + the reconcile path (ingestion/notes.py).

Two halves, mirroring the preferences split:
  * DB-backed accessor tests (insert / cosine KNN / update / supersede / board listing /
    source_ref probe) against the shared Postgres test DB — skip cleanly when it's down.
  * reconcile_note unit tests with the DB, embedder, and LLM all stubbed — thresholds,
    kill switch, and fail-open behavior, no network and no real model calls."""

from __future__ import annotations

import os

import psycopg
import pytest

_DB_URL = os.environ.get(
    "SYNAPSE_TEST_URL", "postgresql://synapse:synapse@127.0.0.1:5432/synapse_test"
)

# Skip the whole module if the shared Postgres test DB isn't up — the accessor tests
# are DB-only and the reconcile tests ride along (this file is pinned to the db worker).
try:
    _probe = psycopg.connect(_DB_URL, connect_timeout=2)
    _probe.close()
except Exception:  # pragma: no cover - environment dependent
    pytest.skip("no test DB reachable", allow_module_level=True)

import ingestion.notes as notes_mod  # noqa: E402
from ingestion.db import Database  # noqa: E402
from ingestion.db_connection import _EMBED_DIMS, _vector_literal  # noqa: E402
from ingestion.db_notes import note_helpful_counts  # noqa: E402
from ingestion.notes import _OWNER, reconcile_note  # noqa: E402
from tests.helpers.embed import onehot  # noqa: E402


def _wipe(conn):
    conn.execute("DELETE FROM notes")


def _insert(db, *, type="user", hook="User prefers X", body="Body.", slot=1, **kw):
    args = {
        "owner_id": _OWNER,
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
    hits = db.find_live_notes(_OWNER, onehot(1), limit=5)
    assert hits[0]["id"] == a
    assert hits[0]["sim"] == pytest.approx(1.0, abs=1e-3)
    assert hits[1]["sim"] == pytest.approx(0.0, abs=1e-3)
    assert set(hits[0]) == {"id", "hook", "body", "type", "project", "sim"}
    db.close()
    _wipe(conn)


def test_update_refreshes_hook_body_and_updated_at(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    nid = _insert(db, hook="Old hook", body="Old body.", slot=3)
    before = conn.execute("SELECT updated_at FROM notes WHERE id = %s", (nid,)).fetchone()[0]
    db.update_note(nid, hook="New hook", body="New body.", embedding=onehot(4), embed_model="test2")
    row = conn.execute(
        "SELECT hook, body, embed_model, updated_at FROM notes WHERE id = %s", (nid,)
    ).fetchone()
    assert row[0] == "New hook" and row[1] == "New body." and row[2] == "test2"
    assert row[3] > before
    # The refreshed embedding moved the note to slot 4 in KNN space.
    hits = db.find_live_notes(_OWNER, onehot(4), limit=1)
    assert hits[0]["id"] == nid and hits[0]["sim"] == pytest.approx(1.0, abs=1e-3)
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
    live_ids = {h["id"] for h in db.find_live_notes(_OWNER, onehot(5), limit=5)}
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


def test_find_live_notes_scopes_by_owner_not_by_legacy_group(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    technical = _insert(db, hook="Mine, technical", slot=6)
    personal = _insert(db, hook="Mine, personal", slot=6)
    conn.execute("UPDATE notes SET group_id = 'personal' WHERE id = %s", (personal,))
    _insert(db, hook="Someone else's", owner_id="other", slot=6)
    hits = db.find_live_notes(_OWNER, onehot(6), limit=5)
    assert {h["id"] for h in hits} == {technical, personal}
    db.close()
    _wipe(conn)


def test_new_note_takes_the_column_default_group(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    nid = _insert(db, hook="Left to the default", slot=7)
    assert conn.execute("SELECT group_id FROM notes WHERE id = %s", (nid,)).fetchone()[0] == (
        "technical"
    )
    db.close()
    _wipe(conn)


def test_null_embedding_insert_ok_and_skipped_by_knn(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    nid = _insert(db, hook="Keyless note", embedding=None, embed_model="ignored")
    row = conn.execute("SELECT embedding, embed_model FROM notes WHERE id = %s", (nid,)).fetchone()
    assert row[0] is None and row[1] is None  # embed_model nulled alongside the vector
    assert db.find_live_notes(_OWNER, onehot(1), limit=5) == []
    db.close()
    _wipe(conn)


def test_list_board_notes_ordering_and_project_filter(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    ref = _insert(db, type="reference", hook="See runbook", slot=10)
    usr = _insert(db, type="user", hook="User is left-handed", slot=11)
    fb = _insert(db, type="feedback", hook="Never use tables", slot=12)
    pa = _insert(db, type="project", project="proj-a", hook="A uses provider X", slot=13)
    _insert(db, type="project", project="proj-b", hook="B uses provider Y", slot=14)
    retired = _insert(db, type="feedback", hook="Old rule", slot=15)
    db.supersede_note(retired, fb)

    rows = db.list_board_notes(_OWNER, "proj-a")
    # feedback -> user -> project (only proj-a's) -> reference; the retired row is gone.
    assert [r["id"] for r in rows] == [fb, usr, pa, ref]
    assert set(rows[0]) == {"id", "hook", "type", "project", "audience", "updated_at"}

    # No project scope: the global set only (project = NULL matches nothing).
    assert [r["id"] for r in db.list_board_notes(_OWNER, None)] == [fb, usr, ref]
    db.close()
    _wipe(conn)


def test_note_helpful_counts_counts_reports_that_rated_a_note_helpful(conn, db_url):
    """One count per report, only for n:<id> in `helpful`: e:/f: ids, the `noise` list,
    malformed entries and repeats inside one report do not count."""
    conn.execute("DELETE FROM recall_feedback")
    rows = [
        ('["n:5", "e:5", "n:7"]', "[]"),
        ('["n:5", "n:5"]', '["n:7"]'),  # repeat in one report = one; noise is not helpful
        ('["n:5x", "n:", "nn:9", "n:-3", "n:99999999999999999999", "f:abc"]', "[]"),
        ("[]", '["n:5"]'),
    ]
    for helpful, noise in rows:
        conn.execute(
            "INSERT INTO recall_feedback (query, helpful, noise) VALUES ('q', %s::jsonb, %s::jsonb)",
            (helpful, noise),
        )
    try:
        assert note_helpful_counts(conn) == {5: 2, 7: 1}
    finally:
        conn.execute("DELETE FROM recall_feedback")
    assert note_helpful_counts(conn) == {}


def test_note_helpful_counts_is_empty_without_the_feedback_table():
    """A database that predates schema 046 ranks every note equal: the board falls back to
    recency instead of failing."""

    class NoTable:
        def execute(self, *a, **kw):
            raise psycopg.errors.UndefinedTable("relation recall_feedback does not exist")

    assert note_helpful_counts(NoTable()) == {}


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


def test_find_note_by_source_ref(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    assert db.find_note_by_source_ref("seed:missing") is None
    first = _insert(db, hook="Seeded v1", source_ref="seed:42", slot=18)
    second = _insert(db, hook="Seeded v2", source_ref="seed:42", slot=19)
    row = db.find_note_by_source_ref("seed:42")
    assert row is not None and row["id"] == second  # newest wins
    assert first != second
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

    def find_live_notes(self, owner_id, embedding, limit=5):
        self.knn_calls += 1
        return self._candidates

    def insert_note(self, **kw):
        self.inserted.append(kw)
        self._next_id += 1
        return self._next_id

    def update_note(self, note_id, *, hook, body, embedding, embed_model, audience=None):
        self.updated.append({"note_id": note_id, "hook": hook, "body": body, "audience": audience})

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


def _cand(sim: float, type: str = "user", note_id: int = 42) -> dict:
    return {
        "id": note_id,
        "hook": "User prefers dark mode",
        "body": "Existing body.",
        "type": type,
        "project": None,
        "sim": sim,
    }


def _run(monkeypatch, *, candidates=(), relation="same", type="user", raise_llm=False):
    """Drive reconcile_note with everything stubbed; returns (result, db, llm_calls)."""
    llm_calls: list[dict] = []

    def _fake_parse(*a, **k):
        llm_calls.append(k)
        if raise_llm:
            raise RuntimeError("LLM down")
        return relation

    monkeypatch.setattr(notes_mod, "parse_with_retry", _fake_parse)
    db = _NotesDB(candidates)
    result = reconcile_note(
        db,
        _StubEmb(),
        object(),
        hook="User prefers light mode",
        body="New body.",
        type=type,
        project=None,
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
    assert "group_id" not in db.inserted[0]  # the column default decides
    assert llm_calls == []  # no candidate -> no confirm call


def test_reconcile_creates_below_threshold(monkeypatch):
    result, db, llm_calls = _run(monkeypatch, candidates=[_cand(0.5)])
    assert result["outcome"] == "created"
    assert len(db.inserted) == 1 and db.updated == [] and llm_calls == []


def test_reconcile_updates_on_high_sim_same_type_same(monkeypatch):
    result, db, llm_calls = _run(monkeypatch, candidates=[_cand(0.92)], relation="same")
    assert result == {"outcome": "updated", "note_id": 42, "prev_id": None, "audience": None}
    # audience=None on the UPDATE: a restatement preserves the note's stored tier.
    assert db.updated == [
        {
            "note_id": 42,
            "hook": "User prefers light mode",
            "body": "New body.",
            "audience": None,
        }
    ]
    assert db.inserted == [] and db.superseded == []
    assert len(llm_calls) == 1


def test_reconcile_supersedes_on_contradicts(monkeypatch):
    result, db, llm_calls = _run(monkeypatch, candidates=[_cand(0.92)], relation="contradicts")
    assert result == {
        "outcome": "superseded",
        "note_id": 501,
        "prev_id": 42,
        "audience": "personal",
    }
    assert len(db.inserted) == 1
    assert db.superseded == [(42, 501)]  # old retired, pointed at the fresh row
    assert db.updated == [] and len(llm_calls) == 1


def test_reconcile_different_type_high_sim_creates(monkeypatch):
    result, db, llm_calls = _run(monkeypatch, candidates=[_cand(0.95, type="project")])
    assert result["outcome"] == "created"
    assert len(db.inserted) == 1 and llm_calls == []  # type mismatch skips the confirm


def test_reconcile_kill_switch_updates_without_llm(monkeypatch):
    monkeypatch.setenv("SYNAPSE_NOTES_CONFIRM", "0")
    result, db, llm_calls = _run(monkeypatch, candidates=[_cand(0.92)], relation="contradicts")
    assert result["outcome"] == "updated"
    assert db.updated and db.inserted == [] and db.superseded == []
    assert llm_calls == []  # the LLM was never consulted


def test_reconcile_llm_failure_fails_open_to_update(monkeypatch):
    result, db, llm_calls = _run(monkeypatch, candidates=[_cand(0.92)], raise_llm=True)
    assert result["outcome"] == "updated"
    assert db.updated and db.inserted == [] and db.superseded == []
    assert len(llm_calls) == 1


def test_reconcile_threshold_env_override(monkeypatch):
    monkeypatch.setenv("SYNAPSE_NOTES_SIM_MATCH", "0.95")
    result, _db, llm_calls = _run(monkeypatch, candidates=[_cand(0.90)])
    assert result["outcome"] == "created"  # 0.90 < raised bar
    assert llm_calls == []


def test_reconcile_keyless_embedder_none(monkeypatch):
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


# ---------------------------------------------------------------------------
# reconcile_note against the real store — a live note filed under either legacy
# group is the same pool as a fresh write
# ---------------------------------------------------------------------------


class _OneHotEmb:
    """Embeds every text to one slot, so a note seeded in that slot is an exact match."""

    model_name = "test"

    def __init__(self, slot):
        self._slot = slot

    def embed(self, texts, task):
        return [onehot(self._slot) for _ in texts]


def _seed_note(conn, *, group_id, slot):
    return conn.execute(
        "INSERT INTO notes (owner_id, group_id, type, hook, body, embedding, embed_model) "
        f"VALUES (%s, %s, 'user', 'User prefers dark mode', 'Old body.', "
        f"%s::halfvec({_EMBED_DIMS}), 'test') RETURNING id",
        (_OWNER, group_id, _vector_literal(onehot(slot))),
    ).fetchone()[0]


def _write_note(db, slot, project):
    return reconcile_note(
        db,
        _OneHotEmb(slot),
        object(),
        hook="User prefers light mode",
        body="New body.",
        type="user",
        project=project,
        source_ref="ep:1",
    )


# (group the live note is filed under, project of the new write). Each pair is one the
# old per-group lookup missed: a project-less write looked only in 'technical', a write
# for a personal-leaning project only in 'personal'.
_ACROSS_GROUPS = [("personal", None), ("technical", "jobs")]


@pytest.mark.parametrize(("legacy_group", "project"), _ACROSS_GROUPS)
def test_restated_note_updates_the_live_note_in_the_other_group(
    conn, db_url, monkeypatch, legacy_group, project
):
    _wipe(conn)
    db = Database(db_url)
    old = _seed_note(conn, group_id=legacy_group, slot=20)
    monkeypatch.setattr(notes_mod, "parse_with_retry", lambda *a, **k: "same")
    result = _write_note(db, 20, project)
    assert result["outcome"] == "updated" and result["note_id"] == old
    live = conn.execute("SELECT id, hook FROM notes WHERE superseded_by IS NULL").fetchall()
    assert live == [(old, "User prefers light mode")]
    db.close()
    _wipe(conn)


@pytest.mark.parametrize(("legacy_group", "project"), _ACROSS_GROUPS)
def test_contradicting_note_supersedes_the_live_note_in_the_other_group(
    conn, db_url, monkeypatch, legacy_group, project
):
    _wipe(conn)
    db = Database(db_url)
    old = _seed_note(conn, group_id=legacy_group, slot=21)
    monkeypatch.setattr(notes_mod, "parse_with_retry", lambda *a, **k: "contradicts")
    result = _write_note(db, 21, project)
    assert result["outcome"] == "superseded" and result["prev_id"] == old
    new = result["note_id"]
    assert (
        conn.execute("SELECT superseded_by FROM notes WHERE id = %s", (old,)).fetchone()[0] == new
    )
    live = conn.execute("SELECT id FROM notes WHERE superseded_by IS NULL").fetchall()
    assert live == [(new,)]
    db.close()
    _wipe(conn)


def test_reconciled_note_gets_the_default_group_whatever_the_project(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    result = _write_note(db, 22, "jobs")
    assert result["outcome"] == "created"
    row = conn.execute("SELECT group_id FROM notes WHERE id = %s", (result["note_id"],)).fetchone()
    assert row[0] == "technical"
    db.close()
    _wipe(conn)
