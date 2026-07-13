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
from ingestion.embedding import embed_dims  # noqa: E402
from ingestion.notes import _OWNER, reconcile_note  # noqa: E402

_DIMS = embed_dims()
GROUP = "technical"


def _vec(slot: int) -> list[float]:
    """A one-hot unit vector: identical slots -> cosine sim 1, distinct -> 0.
    Lets the KNN ordering be asserted deterministically without a real embedder."""
    v = [0.0] * _DIMS
    v[slot % _DIMS] = 1.0
    return v


def _wipe(conn):
    conn.execute("DELETE FROM notes")


def _insert(db, *, type="user", hook="User prefers X", body="Body.", slot=1, **kw):
    args = {
        "owner_id": _OWNER,
        "group_id": GROUP,
        "project": None,
        "embedding": _vec(slot),
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
    hits = db.find_live_notes(_OWNER, GROUP, _vec(1), limit=5)
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
    db.update_note(nid, hook="New hook", body="New body.", embedding=_vec(4), embed_model="test2")
    row = conn.execute(
        "SELECT hook, body, embed_model, updated_at FROM notes WHERE id = %s", (nid,)
    ).fetchone()
    assert row[0] == "New hook" and row[1] == "New body." and row[2] == "test2"
    assert row[3] > before
    # The refreshed embedding moved the note to slot 4 in KNN space.
    hits = db.find_live_notes(_OWNER, GROUP, _vec(4), limit=1)
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
    live_ids = {h["id"] for h in db.find_live_notes(_OWNER, GROUP, _vec(5), limit=5)}
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
    hits = db.find_live_notes(_OWNER, GROUP, _vec(6), limit=5)
    assert [h["id"] for h in hits] == [mine]
    db.close()
    _wipe(conn)


def test_null_embedding_insert_ok_and_skipped_by_knn(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    nid = _insert(db, hook="Keyless note", embedding=None, embed_model="ignored")
    row = conn.execute("SELECT embedding, embed_model FROM notes WHERE id = %s", (nid,)).fetchone()
    assert row[0] is None and row[1] is None  # embed_model nulled alongside the vector
    assert db.find_live_notes(_OWNER, GROUP, _vec(1), limit=5) == []
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
    assert set(rows[0]) == {"id", "hook", "type", "project", "updated_at"}

    # No project scope: the global set only (project = NULL matches nothing).
    assert [r["id"] for r in db.list_board_notes(_OWNER, None)] == [fb, usr, ref]
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
    assert set(rows[0]) == {"id", "hook", "body", "type", "project", "updated_at"}
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

    def find_live_notes(self, owner_id, group_id, embedding, limit=5):
        self.knn_calls += 1
        return self._candidates

    def insert_note(self, **kw):
        self.inserted.append(kw)
        self._next_id += 1
        return self._next_id

    def update_note(self, note_id, *, hook, body, embedding, embed_model):
        self.updated.append({"note_id": note_id, "hook": hook, "body": body})

    def supersede_note(self, old_id, new_id):
        self.superseded.append((old_id, new_id))


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
        project=None,
        source_ref="ep:9",
    )
    return result, db, llm_calls


def test_reconcile_creates_when_no_candidates(monkeypatch):
    result, db, llm_calls = _run(monkeypatch, candidates=[])
    assert result == {"outcome": "created", "note_id": 501, "prev_id": None}
    assert len(db.inserted) == 1 and db.updated == [] and db.superseded == []
    assert db.inserted[0]["source_ref"] == "ep:9"
    assert llm_calls == []  # no candidate -> no confirm call


def test_reconcile_creates_below_threshold(monkeypatch):
    result, db, llm_calls = _run(monkeypatch, candidates=[_cand(0.5)])
    assert result["outcome"] == "created"
    assert len(db.inserted) == 1 and db.updated == [] and llm_calls == []


def test_reconcile_updates_on_high_sim_same_type_same(monkeypatch):
    result, db, llm_calls = _run(monkeypatch, candidates=[_cand(0.92)], relation="same")
    assert result == {"outcome": "updated", "note_id": 42, "prev_id": None}
    assert db.updated == [{"note_id": 42, "hook": "User prefers light mode", "body": "New body."}]
    assert db.inserted == [] and db.superseded == []
    assert len(llm_calls) == 1


def test_reconcile_supersedes_on_contradicts(monkeypatch):
    result, db, llm_calls = _run(monkeypatch, candidates=[_cand(0.92)], relation="contradicts")
    assert result == {"outcome": "superseded", "note_id": 501, "prev_id": 42}
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
