"""Audience scoping end to end (schema 053): who gets served what, and why.

Private mode (050) controls what gets CAPTURED. This controls what gets SERVED. The
enforcement lives on the serving routes, so these tests drive the serving paths — not
the SQL helpers — and assert on what actually comes back.

Three things are load-bearing and each gets its own section:

  * The trust lookup fails CLOSED. No surface, no row, no database — all restricted with
    an empty allowlist. There is no error path that produces 'full'.
  * Drill-down cannot outrun the overview. e:N / n:N ids are sequential integers, so
    fetch() must enforce exactly what recall() and the board do.
  * remember() classifies on write, in a fixed precedence, and a RESTATEMENT never
    silently reclassifies an existing note.

Board filtering (notes tier + digest allowlist + banner) lives in test_board.py, and
fetch_session's predicates in test_fetch_session.py — both next to the code they cover.
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

try:
    _probe = psycopg.connect(_DB_URL, connect_timeout=2)
    _probe.close()
except Exception:  # pragma: no cover - environment dependent
    pytest.skip("no test DB reachable", allow_module_level=True)

import ingestion.notes as notes_mod  # noqa: E402
from ingestion.db import Database  # noqa: E402
from ingestion.notes import _OWNER, reconcile_note  # noqa: E402
from ingestion.surfaces import (  # noqa: E402
    UNKNOWN_SURFACE,
    derive_audience,
    lookup_surface,
    restricted_project_union,
)
from mcp_server import server  # noqa: E402
from mcp_server.recall import Recall  # noqa: E402
from tests.helpers.surfaces import (  # noqa: E402
    clear_surfaces,
    register_full,
    register_restricted,
)


@pytest.fixture()
def clean(conn):
    def _wipe():
        conn.execute("TRUNCATE episodes RESTART IDENTITY CASCADE")
        conn.execute("DELETE FROM notes")
        clear_surfaces(conn)

    _wipe()
    yield conn
    _wipe()


def _note(db_url, hook, *, audience="personal", type="user", project=None):
    db = Database(db_url)
    try:
        return db.insert_note(
            owner_id=_OWNER,
            group_id="technical",
            project=project,
            type=type,
            hook=hook,
            body=f"Body of: {hook}",
            embedding=None,
            embed_model=None,
            source_ref=None,
            audience=audience,
        )
    finally:
        db.close()


def _episode(conn, project, content="a turn about the thing"):
    return conn.execute(
        "INSERT INTO episodes (session_id, sequence, project, content) "
        "VALUES (%s, 1, %s, %s) RETURNING id",
        (f"aud-{uuid.uuid4().hex[:8]}", project, content),
    ).fetchone()[0]


# ---------------------------------------------------------------------------
# lookup_surface: every failure mode lands on restricted
# ---------------------------------------------------------------------------


def test_registered_surfaces_resolve_to_their_row(clean, db_url):
    register_full(clean, "trusted-host")
    register_restricted(clean, ["alpha", "beta"], "work-host")

    full = lookup_surface(db_url, "trusted-host")
    assert full.trust == "full" and full.known and not full.restricted
    assert full.project_filter is None and full.audience_filter is None

    work = lookup_surface(db_url, "work-host")
    assert work.trust == "restricted" and work.known and work.restricted
    assert work.project_filter == ["alpha", "beta"]
    assert work.audience_filter == "work-safe"


@pytest.mark.parametrize(
    "surface_id",
    [None, "", "   ", "never-registered"],
    ids=["none", "empty", "whitespace", "unknown"],
)
def test_absent_or_unknown_surface_is_restricted_and_empty(clean, db_url, surface_id):
    st = lookup_surface(db_url, surface_id)
    assert st.restricted and not st.known
    assert st.project_filter == []  # matches nothing, NULL project included
    assert st.audience_filter == "work-safe"


def test_unreachable_database_is_restricted_not_full(clean):
    """The lookup's own infrastructure failing must not widen access. Fail-closed here
    is the difference between an empty board and a leaked one."""
    st = lookup_surface("postgresql://synapse:synapse@127.0.0.1:1/nope", "trusted-host")
    assert st == UNKNOWN_SURFACE
    assert st.restricted and not st.known


def test_missing_surfaces_table_is_restricted(clean):
    """A deployment behind schema/053 restricts everything rather than serving it."""
    st = lookup_surface(_DB_URL.replace("/synapse_test", "/postgres"), "trusted-host")
    assert st.restricted and not st.known


# ---------------------------------------------------------------------------
# recall(): episodes by allowlist, notes by tier, KG leg skipped
# ---------------------------------------------------------------------------


class _Emb:
    def embed(self, texts, task):
        return [[0.0] * 8 for _ in texts]


def _engine(db_url, monkeypatch, notes_rows=None):
    """A Recall with the embedding/rerank/KG legs stubbed, so the assertions are about
    the FILTERS and not about retrieval quality. The episode legs stay real — the
    project predicate they gain is exactly what's under test."""
    r = Recall(db_url, "")
    r._ensure_embedder = lambda: _Emb()
    r._rerank_pool_scored = lambda q, pool: [(i, 0.9) for i in range(len(pool))]
    r._compact_to_passages = lambda q, eps, n: [
        {"id": e["id"], "text": e.get("content", "")} for e in eps
    ]
    r._search_web_reranked = lambda q, emb: []
    r._fetch_superseded_pairs_pg = lambda gid, uuids, cap: []
    r._surface_supersessions = lambda *a, **k: []
    r._episode_supersessions = lambda *a, **k: {}
    r._increment_retrieval_counts = lambda ids: None
    r._increment_fact_retrieval_counts = lambda *a, **k: None
    r._record_metrics = lambda m: None
    r._search_notes = lambda q, emb, proj, audience=None: [
        {"id": f"n:{n['id']}", "hook": n["hook"], "audience": audience} for n in (notes_rows or [])
    ]
    return r


def test_restricted_recall_filters_episodes_to_the_allowlist(clean, db_url, monkeypatch):
    _episode(clean, "alpha", "alpha discussion of the widget")
    _episode(clean, "beta", "beta discussion of the widget")
    _episode(clean, None, "unlabeled discussion of the widget")
    sid = register_restricted(clean, ["alpha"])

    r = _engine(db_url, monkeypatch)
    out = r.recall("widget", surface=sid)
    served = [it["text"] for it in out.get("episodes", [])]
    assert served == ["alpha discussion of the widget"]

    # The control: the same query on a trusted host reaches all three.
    full = register_full(clean)
    out_full = _engine(db_url, monkeypatch).recall("widget", surface=full)
    assert len(out_full.get("episodes", [])) == 3


def test_recall_without_a_surface_serves_no_episodes(clean, db_url, monkeypatch):
    """Bare calls get the empty allowlist. This is intended: a serving path that can't
    identify its caller is not a trusted caller."""
    _episode(clean, "alpha", "alpha discussion of the widget")
    out = _engine(db_url, monkeypatch).recall("widget")
    assert out.get("episodes") is None


def test_restricted_recall_passes_the_work_safe_tier_to_the_notes_leg(clean, db_url, monkeypatch):
    sid = register_restricted(clean, ["alpha"])
    r = _engine(db_url, monkeypatch, notes_rows=[{"id": 1, "hook": "User prefers tabs"}])
    out = r.recall("tabs", surface=sid)
    assert out["notes"][0]["audience"] == "work-safe"

    full = register_full(clean)
    r2 = _engine(db_url, monkeypatch, notes_rows=[{"id": 1, "hook": "User prefers tabs"}])
    assert r2.recall("tabs", surface=full)["notes"][0]["audience"] is None


def test_restricted_recall_skips_the_kg_leg_entirely(clean, db_url, monkeypatch):
    """v1 posture: kg_relationships has no project column, so facts cannot be filtered
    and the only fail-closed answer is to serve none. Skipping is also cheaper than
    filtering — the leg never runs."""
    sid = register_restricted(clean, ["alpha"])
    calls: list[tuple] = []

    r = _engine(db_url, monkeypatch)
    r._search_kg = lambda *a, **k: calls.append(a) or ([{"fact": "leaked", "_uuid": "u1"}], [])
    assert r.recall("anything", surface=sid)["facts"] == []
    assert calls == [], "the KG leg must not run at all on a restricted surface"

    full = register_full(clean)
    r2 = _engine(db_url, monkeypatch)
    r2._search_kg = lambda *a, **k: ([{"fact": "served", "_uuid": "u1"}], [])
    assert [f["fact"] for f in r2.recall("anything", surface=full)["facts"]] == ["served"]


def test_recall_records_the_trust_regime_in_telemetry(clean, db_url, monkeypatch):
    """A restricted serve is narrower by design; without this the metrics read as an
    unexplained collapse in recall quality after the rollout."""
    sid = register_restricted(clean, ["alpha"])
    rows: list[dict] = []
    r = _engine(db_url, monkeypatch)
    r._record_metrics = rows.append
    r.recall("q", surface=sid)
    assert rows[0]["served_ids"]["trust"] == "restricted"


def test_restricted_recall_full_turns_filters_episodes(clean, db_url, monkeypatch):
    """The drill-down sibling enforces the same allowlist — otherwise the cheap way
    around a filtered overview is to ask for whole turns instead."""
    _episode(clean, "alpha", "alpha raw turn about the widget")
    _episode(clean, "beta", "beta raw turn about the widget")
    sid = register_restricted(clean, ["alpha"])
    out = _engine(db_url, monkeypatch).recall_episodes("widget", surface=sid)
    assert [e["content"] for e in out["episodes"]] == ["alpha raw turn about the widget"]


# ---------------------------------------------------------------------------
# fetch(): ids are guessable, so drill-down enforces the same predicates
# ---------------------------------------------------------------------------


def test_fetch_cannot_bypass_the_episode_allowlist(clean, db_url):
    allowed = _episode(clean, "alpha", "allowed turn")
    forbidden = _episode(clean, "beta", "forbidden turn")
    unlabeled = _episode(clean, None, "unlabeled turn")
    sid = register_restricted(clean, ["alpha"])

    out = Recall(db_url, "").fetch(
        [f"e:{allowed}", f"e:{forbidden}", f"e:{unlabeled}"], surface=sid
    )
    assert [e["id"] for e in out["episodes"]] == [f"e:{allowed}"]
    # Absent, not errored: a distinct "forbidden" reply would itself confirm the id.
    assert out["skipped"] == []


def test_fetch_cannot_bypass_the_note_audience(clean, db_url):
    work = _note(db_url, "User prefers tabs", audience="work-safe")
    private = _note(db_url, "User keeps a personal journal", audience="personal")
    sid = register_restricted(clean, ["alpha"])

    out = Recall(db_url, "").fetch([f"n:{work}", f"n:{private}"], surface=sid)
    assert [n["id"] for n in out["notes"]] == [f"n:{work}"]

    full = register_full(clean)
    out_full = Recall(db_url, "").fetch([f"n:{work}", f"n:{private}"], surface=full)
    assert [n["id"] for n in out_full["notes"]] == [f"n:{work}", f"n:{private}"]


def test_fetch_without_a_surface_is_restricted(clean, db_url):
    eid = _episode(clean, "alpha", "a turn")
    private = _note(db_url, "User keeps a personal journal", audience="personal")
    out = Recall(db_url, "").fetch([f"e:{eid}", f"n:{private}"])
    assert out["episodes"] == [] and out["notes"] == []


# ---------------------------------------------------------------------------
# remember(): audience precedence on write
# ---------------------------------------------------------------------------


def test_restricted_project_union_reads_only_restricted_rows(clean, db_url):
    register_restricted(clean, ["alpha", "beta"], "work-host")
    register_restricted(clean, ["beta", "gamma"], "other-work-host")
    register_full(clean, "trusted-host")  # full surfaces contribute nothing
    db = Database(db_url)
    try:
        assert restricted_project_union(db) == {"alpha", "beta", "gamma"}
    finally:
        db.close()


class _NoSurfaces:
    """A Database whose surfaces read blows up — derivation must fall to personal."""

    def restricted_surface_projects(self):
        raise RuntimeError("surfaces unavailable")


def test_derive_audience_precedence(clean, db_url):
    register_restricted(clean, ["alpha"], "work-host")
    db = Database(db_url)
    try:
        # 1. explicit wins over everything, in both directions.
        assert (
            derive_audience(db, explicit="personal", caller_restricted=True, project="alpha")
            == "personal"
        )
        assert (
            derive_audience(db, explicit="work-safe", caller_restricted=False, project=None)
            == "work-safe"
        )
        # 2. a registered restricted caller defaults work-safe regardless of project.
        assert (
            derive_audience(db, explicit=None, caller_restricted=True, project="unrelated")
            == "work-safe"
        )
        # 3. the project rule.
        assert (
            derive_audience(db, explicit=None, caller_restricted=False, project="alpha")
            == "work-safe"
        )
        # 4. the fail-closed default.
        assert (
            derive_audience(db, explicit=None, caller_restricted=False, project="beta")
            == "personal"
        )
        assert (
            derive_audience(db, explicit=None, caller_restricted=False, project=None) == "personal"
        )
    finally:
        db.close()

    # A broken union read cannot promote a note.
    assert (
        derive_audience(_NoSurfaces(), explicit=None, caller_restricted=False, project="alpha")
        == "personal"
    )


def test_derive_audience_rejects_an_invalid_explicit_value(clean, db_url):
    db = Database(db_url)
    try:
        with pytest.raises(ValueError, match="invalid audience"):
            derive_audience(db, explicit="public", caller_restricted=False, project=None)
    finally:
        db.close()


def _remember(**kw):
    return asyncio.run(server.remember(**kw))


@pytest.fixture()
def remember_env(clean, db_url, monkeypatch):
    """remember() wired at the test DB with keyless notes deps: NULL embedding, dedup
    KNN skipped, no LLM call. Every write is therefore a clean 'created'."""
    monkeypatch.setattr(server, "DB_URL", db_url)
    monkeypatch.setattr(server, "_recall_engine", Recall(db_url, ""))
    monkeypatch.setattr(server, "_notes_deps", lambda: (None, None))
    return clean


def _audience_of(conn, note_id):
    return conn.execute("SELECT audience FROM notes WHERE id = %s", (note_id,)).fetchone()[0]


def test_remember_explicit_audience_wins(remember_env):
    out = _remember(hook="User prefers tabs", body="B.", type="user", audience="work-safe")
    assert out["audience"] == "work-safe"
    assert _audience_of(remember_env, out["note_id"]) == "work-safe"


def test_remember_from_a_registered_restricted_surface_defaults_work_safe(remember_env):
    """Symmetric with what that host can READ: otherwise notes written at work vanish
    from the work board on the next session."""
    sid = register_restricted(remember_env, ["alpha"], "work-host")
    out = _remember(hook="Beta uses a queue", body="B.", project="beta", surface=sid)
    assert out["audience"] == "work-safe"


def test_remember_from_an_unknown_surface_does_not_default_work_safe(remember_env):
    """The asymmetry that matters. An unknown surface RESTRICTS reads, but it must not
    widen a write — defaulting an unrecognised hostname's notes to work-safe would turn
    the fail-closed read rule into a leak."""
    out = _remember(
        hook="User keeps a personal journal", body="B.", type="user", surface="never-registered"
    )
    assert out["audience"] == "personal"


def test_remember_derives_work_safe_from_the_project_rule(remember_env):
    register_restricted(remember_env, ["alpha"], "work-host")
    out = _remember(hook="Alpha runs on Postgres", body="B.", project="alpha")
    assert out["audience"] == "work-safe"
    other = _remember(hook="Gamma runs on SQLite", body="B.", project="gamma")
    assert other["audience"] == "personal"


def test_remember_defaults_personal(remember_env):
    out = _remember(hook="User keeps a personal journal", body="B.", type="user")
    assert out["audience"] == "personal"
    assert _audience_of(remember_env, out["note_id"]) == "personal"


def test_remember_rejects_an_invalid_audience_before_writing(remember_env):
    out = _remember(hook="H", body="B.", audience="public")
    assert out["status"] == "error" and "invalid audience" in out["detail"]
    assert remember_env.execute("SELECT count(*) FROM notes").fetchone()[0] == 0


# ---------------------------------------------------------------------------
# Restatement preserves the stored tier
# ---------------------------------------------------------------------------


class _StubDB:
    """reconcile_note's collaborators, stubbed: one high-similarity candidate so the
    restatement path runs, and no restricted surfaces so derivation says 'personal'."""

    def __init__(self, candidate_type="user"):
        self.updates: list[dict] = []
        self.inserts: list[dict] = []
        self._candidate_type = candidate_type

    def find_live_notes(self, owner_id, group_id, embedding, limit=5):
        return [
            {
                "id": 42,
                "hook": "User prefers dark mode",
                "body": "Existing.",
                "type": self._candidate_type,
                "project": None,
                "sim": 0.95,
            }
        ]

    def insert_note(self, **kw):
        self.inserts.append(kw)
        return 501

    def update_note(self, note_id, **kw):
        self.updates.append({"note_id": note_id, **kw})

    def supersede_note(self, old_id, new_id):
        pass

    def restricted_surface_projects(self):
        return []


class _Emb8:
    model_name = "test-embed"

    def embed(self, texts, task):
        return [[0.0] * 8 for _ in texts]


def test_restatement_preserves_the_stored_audience(monkeypatch):
    """Rephrasing a note is not reclassifying it. update_note gets audience=None, which
    COALESCEs to the stored value — so a work-safe note stays work-safe through an
    update that has no idea what tier it was on."""
    monkeypatch.setattr(notes_mod, "parse_with_retry", lambda *a, **k: "same")
    db = _StubDB()
    res = reconcile_note(
        db,
        _Emb8(),
        object(),
        hook="User prefers light mode",
        body="New body.",
        type="user",
        project=None,
        source_ref="ep:1",
    )
    assert res["outcome"] == "updated" and res["audience"] is None
    assert db.updates[0]["audience"] is None


def test_restatement_with_an_explicit_audience_does_reclassify(monkeypatch):
    """The one way an existing note moves tier: someone said so."""
    monkeypatch.setattr(notes_mod, "parse_with_retry", lambda *a, **k: "same")
    db = _StubDB()
    res = reconcile_note(
        db,
        _Emb8(),
        object(),
        hook="User prefers light mode",
        body="New body.",
        type="user",
        project=None,
        source_ref="ep:1",
        audience="work-safe",
    )
    assert res["audience"] == "work-safe"
    assert db.updates[0]["audience"] == "work-safe"


def test_contradiction_derives_rather_than_inheriting(monkeypatch):
    """A contradiction is a NEW assertion. Inheriting the retired note's tier would let
    a personal statement arrive work-safe purely because it reversed a work-safe one."""
    monkeypatch.setattr(notes_mod, "parse_with_retry", lambda *a, **k: "contradicts")
    db = _StubDB()
    res = reconcile_note(
        db,
        _Emb8(),
        object(),
        hook="User prefers light mode",
        body="New body.",
        type="user",
        project=None,
        source_ref="ep:1",
    )
    assert res["outcome"] == "superseded" and res["audience"] == "personal"
    assert db.inserts[0]["audience"] == "personal"
