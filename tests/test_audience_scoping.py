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
  * An OAuth/OIDC-authenticated caller has no hook to inject a surface, so the server
    derives one from its verified identity — and that derivation is itself fail-closed.

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
    monkeypatch.setattr(server, "_recall_engine", _engine(db_url, monkeypatch))
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


# ---------------------------------------------------------------------------
# OAuth/OIDC callers: the surface comes from the identity, not a param
# ---------------------------------------------------------------------------


class _OAuthToken:
    """The two fields server.py reads off a FastMCP access token: which lane, and who."""

    def __init__(self, claims: dict, client_id: str = "claude-ai-connector") -> None:
        self.client_id = client_id
        self.claims = claims


@pytest.fixture()
def as_oauth_caller(monkeypatch):
    """Put a verified OAuth/OIDC identity in the token context — the claude.ai
    connector's shape: authenticated and allowlisted, but running no PreToolUse hook,
    so it carries no `surface` param at all."""

    def _login(login: str, *, client_id: str = "claude-ai-connector") -> None:
        monkeypatch.setattr(server, "_IDENTITY_CLAIMS", ("preferred_username", "email"))
        monkeypatch.setattr(
            server,
            "get_access_token",
            lambda: _OAuthToken({"preferred_username": login}, client_id),
        )

    return _login


def test_oauth_identity_becomes_its_own_surface_id(as_oauth_caller):
    as_oauth_caller("Kyle")
    assert server._caller_surface(None) == "oauth:kyle"  # lowercased, namespaced


def test_the_derived_surface_overrides_a_client_supplied_one(as_oauth_caller):
    """A token identity the server verified outranks a string the caller typed about
    itself — otherwise an OAuth client could name a trusted host and widen its own view."""
    as_oauth_caller("kyle")
    assert server._caller_surface("test-full-surface") == "oauth:kyle"


def test_machine_token_callers_keep_their_injected_surface(monkeypatch):
    """The machine token says "a Synapse client", never which host, so the hook-injected
    param stays the only evidence available on that lane."""
    monkeypatch.setattr(
        server,
        "get_access_token",
        lambda: _OAuthToken({"preferred_username": "kyle"}, server._MACHINE_CLIENT_ID),
    )
    assert server._caller_surface("work-host") == "work-host"
    assert server._caller_surface(None) is None


def test_no_token_context_changes_nothing(monkeypatch):
    """Open dev/stdio servers and any call outside a request: no identity evidence, so
    the param stands and the pre-existing fail-closed path applies unchanged."""
    monkeypatch.setattr(server, "get_access_token", lambda: None)
    assert server._caller_surface("work-host") == "work-host"
    assert server._caller_surface(None) is None


def test_an_oauth_token_with_no_identity_claim_derives_nothing(monkeypatch):
    """Fail-closed on a malformed token: no claim means no derived id, never a guess."""
    monkeypatch.setattr(server, "_IDENTITY_CLAIMS", ("preferred_username",))
    monkeypatch.setattr(server, "get_access_token", lambda: _OAuthToken({}))
    assert server._caller_surface(None) is None


@pytest.fixture()
def oauth_serving(clean, db_url, monkeypatch):
    """The MCP tools wired at the test DB with the retrieval legs stubbed — these
    assertions drive server.recall/server.fetch, not the engine, because the derivation
    under test lives at the tool boundary."""
    monkeypatch.setattr(server, "DB_URL", db_url)
    monkeypatch.setattr(server, "_recall_engine", _engine(db_url, monkeypatch))
    return clean


def test_registered_oauth_identity_gets_full_serving(oauth_serving, as_oauth_caller):
    """The regression: before this, an owner identity on the OAuth lane sent no surface,
    resolved to UNKNOWN, and got the near-empty restricted serve."""
    _episode(oauth_serving, "alpha", "alpha discussion of the widget")
    _episode(oauth_serving, "beta", "beta discussion of the widget")
    _episode(oauth_serving, None, "unlabeled discussion of the widget")
    register_full(oauth_serving, "oauth:kyle")
    as_oauth_caller("kyle")

    assert len(server.recall("widget")["episodes"]) == 3


def test_unregistered_oauth_identity_is_still_restricted(oauth_serving, as_oauth_caller):
    """The fail-closed half: authenticating is not the same as being trusted. The
    operator registers `oauth:<login>` to grant trust — a login never creates its own row."""
    _episode(oauth_serving, "alpha", "alpha discussion of the widget")
    as_oauth_caller("mallory")
    assert server.recall("widget").get("episodes") is None


def test_oauth_identity_scopes_a_restricted_row(oauth_serving, as_oauth_caller):
    """An identity can be registered restricted, same as a host: partial trust is a row,
    not a special case."""
    _episode(oauth_serving, "alpha", "alpha discussion of the widget")
    _episode(oauth_serving, "beta", "beta discussion of the widget")
    register_restricted(oauth_serving, ["alpha"], "oauth:kyle")
    as_oauth_caller("kyle")

    served = [it["text"] for it in server.recall("widget").get("episodes", [])]
    assert served == ["alpha discussion of the widget"]


def test_oauth_drill_down_enforces_the_same_verdict(oauth_serving, db_url, as_oauth_caller):
    """fetch() resolves the identity too — otherwise the cheap way around a restricted
    overview is to guess sequential ids and fetch them."""
    eid = _episode(oauth_serving, "alpha", "a turn")
    private = _note(db_url, "User keeps a personal journal", audience="personal")
    as_oauth_caller("kyle")

    out = server.fetch([f"e:{eid}", f"n:{private}"])
    assert out["episodes"] == [] and out["notes"] == []

    register_full(oauth_serving, "oauth:kyle")
    out_full = server.fetch([f"e:{eid}", f"n:{private}"])
    assert [e["id"] for e in out_full["episodes"]] == [f"e:{eid}"]
    assert [n["id"] for n in out_full["notes"]] == [f"n:{private}"]


def test_oauth_full_turns_resolve_the_identity(oauth_serving, as_oauth_caller):
    _episode(oauth_serving, "alpha", "alpha raw turn about the widget")
    _episode(oauth_serving, "beta", "beta raw turn about the widget")
    register_restricted(oauth_serving, ["alpha"], "oauth:kyle")
    as_oauth_caller("kyle")

    out = server.recall_full_turns("widget")
    assert [e["content"] for e in out["episodes"]] == ["alpha raw turn about the widget"]


def test_machine_token_serving_is_unchanged(oauth_serving, monkeypatch):
    """The plugin's lane, with a token in context: the hook-injected surface is still
    the whole answer, and the identity claims on that token are ignored."""
    _episode(oauth_serving, "alpha", "alpha discussion of the widget")
    _episode(oauth_serving, "beta", "beta discussion of the widget")
    sid = register_restricted(oauth_serving, ["alpha"])
    monkeypatch.setattr(
        server,
        "get_access_token",
        lambda: _OAuthToken({"preferred_username": "kyle"}, server._MACHINE_CLIENT_ID),
    )

    served = [it["text"] for it in server.recall("widget", surface=sid)["episodes"]]
    assert served == ["alpha discussion of the widget"]


def test_remember_from_a_restricted_oauth_identity_defaults_work_safe(
    remember_env, as_oauth_caller
):
    """The write side derives the same id, so an identity registered restricted can read
    back what it just wrote — the same symmetry a restricted HOST gets."""
    register_restricted(remember_env, ["alpha"], "oauth:kyle")
    as_oauth_caller("kyle")
    out = _remember(hook="Beta uses a queue", body="B.", project="beta")
    assert out["audience"] == "work-safe"


def test_remember_from_an_unregistered_oauth_identity_stays_personal(remember_env, as_oauth_caller):
    """Unknown restricts reads but must never widen a write — an unregistered identity
    is exactly as unknown as an unregistered hostname."""
    as_oauth_caller("mallory")
    out = _remember(hook="User keeps a personal journal", body="B.", type="user")
    assert out["audience"] == "personal"


# ---------------------------------------------------------------------------
# Device tokens (schema 054): the credential decides, and pending decides nothing
# ---------------------------------------------------------------------------


class _DeviceToken:
    """What SynapseTokenVerifier stamps onto an approved device's request."""

    def __init__(self, surface_id: str, trust: str, projects: tuple[str, ...] = ()) -> None:
        self.client_id = server._DEVICE_CLIENT_ID
        self.claims = {
            "kind": "device",
            "surface_id": surface_id,
            "trust": trust,
            "allowed_projects": list(projects),
        }


def test_a_device_tokens_claims_decide_what_it_is_served(clean, db_url, monkeypatch):
    """The verifier already resolved the row; serving reads THAT, not a param."""
    monkeypatch.setattr(server, "DB_URL", db_url)
    monkeypatch.setattr(server, "_recall_engine", _engine(db_url, monkeypatch))
    _episode(clean, "alpha", "alpha discussion of the widget")
    _episode(clean, "beta", "beta discussion of the widget")
    monkeypatch.setattr(
        server, "get_access_token", lambda: _DeviceToken("dev-work", "restricted", ("alpha",))
    )

    served = [it["text"] for it in server.recall("widget").get("episodes", [])]
    assert served == ["alpha discussion of the widget"]


def test_a_device_token_ignores_a_surface_param_entirely(clean, db_url, monkeypatch):
    """The whole point of 054. A restricted device that names the trusted host must get
    its OWN scope, not the one it asked for."""
    monkeypatch.setattr(server, "DB_URL", db_url)
    monkeypatch.setattr(server, "_recall_engine", _engine(db_url, monkeypatch))
    _episode(clean, "alpha", "alpha discussion of the widget")
    _episode(clean, "beta", "beta discussion of the widget")
    trusted = register_full(clean, "trusted-host")
    monkeypatch.setattr(
        server, "get_access_token", lambda: _DeviceToken("dev-work", "restricted", ("alpha",))
    )

    served = [it["text"] for it in server.recall("widget", surface=trusted).get("episodes", [])]
    assert served == ["alpha discussion of the widget"]  # NOT the full corpus


def test_a_root_token_caller_still_resolves_the_legacy_surface_param(clean, db_url, monkeypatch):
    """The migration window: a 0.16.x plugin on the shared token keeps working exactly
    as it did until it updates and enrolls."""
    monkeypatch.setattr(server, "DB_URL", db_url)
    monkeypatch.setattr(server, "_recall_engine", _engine(db_url, monkeypatch))
    _episode(clean, "alpha", "alpha discussion of the widget")
    _episode(clean, "beta", "beta discussion of the widget")
    sid = register_restricted(clean, ["alpha"], "legacy-work-host")
    monkeypatch.setattr(server, "get_access_token", lambda: None)

    served = [it["text"] for it in server.recall("widget", surface=sid).get("episodes", [])]
    assert served == ["alpha discussion of the widget"]


def test_a_pending_device_is_served_nothing(clean, db_url, monkeypatch):
    """Pending holds a real token that authenticates as nothing — including through the
    trust resolution, so even a hand-built claims dict cannot rescue it."""
    from tests.helpers.surfaces import register_device

    monkeypatch.setattr(server, "DB_URL", db_url)
    monkeypatch.setattr(server, "_recall_engine", _engine(db_url, monkeypatch))
    _episode(clean, "alpha", "alpha discussion of the widget")
    register_device(clean, "pending-tok", trust="full", surface_id="dev-new", status="pending")
    monkeypatch.setattr(server, "get_access_token", lambda: None)

    assert server.recall("widget", surface="dev-new").get("episodes") is None
    assert server._caller_trust("dev-new") == UNKNOWN_SURFACE


def test_restricted_project_union_ignores_pending_devices(clean, db_url):
    """A pending device reads nothing, so letting its allowlist widen the work-safe tier
    would classify notes for an audience that does not exist — permanently, if the
    device is never approved."""
    from tests.helpers.surfaces import register_device

    register_restricted(clean, ["alpha"], "approved-work-host")
    register_device(clean, "pending-tok", trust="restricted", projects=["secret"], status="pending")
    db = Database(db_url)
    try:
        assert restricted_project_union(db) == {"alpha"}
    finally:
        db.close()
