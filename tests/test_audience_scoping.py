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
  * Only a credential picks the row. A root-token caller that names a surface (even a
    full-trust one) is served exactly what it is served when it names none: nothing.

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
    r._compact_to_passages = lambda q, eps, n, **_k: [
        {"id": e["id"], "text": e.get("content", "")} for e in eps
    ]
    r._search_web_reranked = lambda q, emb: []
    r._search_kg = lambda *a, **k: ([], [])
    r._fetch_superseded_pairs_pg = lambda gid, uuids, cap, allowed_projects=None: []
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


def test_restricted_recall_scopes_the_kg_leg_to_the_allowlist(clean, db_url, monkeypatch):
    """Schema 056: a registered restricted surface runs the KG leg WITH its allowlist
    (the provenance filter itself is covered in test_kg_source_projects.py). Full trust
    calls it exactly as before, with no scope argument at all."""
    sid = register_restricted(clean, ["alpha"])
    calls: list[dict] = []

    r = _engine(db_url, monkeypatch)
    r._search_kg = lambda *a, **k: calls.append(k) or ([{"fact": "scoped", "_uuid": "u1"}], [])
    assert [f["fact"] for f in r.recall("anything", surface=sid)["facts"]] == ["scoped"]
    assert calls[0]["allowed_projects"] == ["alpha"]

    full = register_full(clean)
    full_calls: list[dict] = []
    r2 = _engine(db_url, monkeypatch)
    r2._search_kg = lambda *a, **k: (
        full_calls.append(k) or ([{"fact": "served", "_uuid": "u1"}], [])
    )
    assert [f["fact"] for f in r2.recall("anything", surface=full)["facts"]] == ["served"]
    assert "allowed_projects" not in full_calls[0]


def test_unknown_surface_skips_every_kg_read(clean, db_url, monkeypatch):
    """An unknown caller's allowlist is empty, and no fact can be a non-empty subset of
    nothing, so the facts leg and every KG-derived extra are skipped without a query."""
    calls: list[str] = []

    def _spy(name, result):
        return lambda *a, **k: calls.append(name) or result

    r = _engine(db_url, monkeypatch)
    r._search_kg = _spy("facts", ([{"fact": "leaked", "_uuid": "u1"}], []))
    r._fetch_superseded_pairs_pg = _spy("pairs", [{"fact": "leaked", "superseded_by": "x"}])
    r._surface_supersessions = _spy("surface", [{"fact": "leaked", "_uuid": "u2"}])
    r._episode_supersessions = _spy("overlay", {})
    out = r.recall("anything", surface="never-registered")
    assert out["facts"] == []
    assert "superseded_facts" not in out
    assert calls == []


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


def test_remember_from_a_registered_restricted_surface_defaults_work_safe(
    remember_env, monkeypatch
):
    """Symmetric with what that host can READ: otherwise notes written at work vanish
    from the work board on the next session. The host is named by its device token."""
    monkeypatch.setattr(
        server, "get_access_token", lambda: _DeviceToken("dev-work", "restricted", ("alpha",))
    )
    out = _remember(hook="Beta uses a queue", body="B.", project="beta")
    assert out["audience"] == "work-safe"


def test_a_root_token_remember_cannot_claim_a_restricted_surface(remember_env, monkeypatch):
    """The write-side half of closing the root lane. Naming a registered restricted
    surface used to default the note to work-safe on that surface's behalf; the root
    token now writes as an unknown caller, so the project rule and then `personal`
    decide, exactly as if no surface had been sent."""
    sid = register_restricted(remember_env, ["alpha"], "legacy-work-host")
    monkeypatch.setattr(server, "get_access_token", lambda: _RootToken())
    out = _remember(hook="User keeps a personal journal", body="B.", type="user", surface=sid)
    assert out["audience"] == "personal"


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
    assert server._caller_surface() == "oauth:kyle"  # lowercased, namespaced


def test_machine_token_callers_resolve_to_no_surface(monkeypatch):
    """The machine token says "a Synapse client", never which host — and identity
    claims riding on it are not an identity either."""
    monkeypatch.setattr(
        server,
        "get_access_token",
        lambda: _OAuthToken({"preferred_username": "kyle"}, server._MACHINE_CLIENT_ID),
    )
    assert server._caller_surface() is None
    assert server._caller_trust() == UNKNOWN_SURFACE


def test_no_token_context_resolves_to_no_surface(monkeypatch):
    """Open dev/stdio servers and any call outside a request: no credential evidence,
    so the fail-closed verdict applies."""
    monkeypatch.setattr(server, "get_access_token", lambda: None)
    assert server._caller_surface() is None
    assert server._caller_trust() == UNKNOWN_SURFACE


def test_an_oauth_token_with_no_identity_claim_derives_nothing(monkeypatch):
    """Fail-closed on a malformed token: no claim means no derived id, never a guess."""
    monkeypatch.setattr(server, "_IDENTITY_CLAIMS", ("preferred_username",))
    monkeypatch.setattr(server, "get_access_token", lambda: _OAuthToken({}))
    assert server._caller_surface() is None


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


def test_an_oauth_caller_cannot_name_a_trusted_surface(oauth_serving, as_oauth_caller):
    """A token identity the server verified outranks a string the caller typed about
    itself — otherwise an OAuth client could name a trusted host and widen its own view."""
    _episode(oauth_serving, "alpha", "alpha discussion of the widget")
    trusted = register_full(oauth_serving, "trusted-host")
    as_oauth_caller("mallory")  # authenticated, but no row of its own
    assert server.recall("widget", surface=trusted).get("episodes") is None


def test_oauth_full_turns_resolve_the_identity(oauth_serving, as_oauth_caller):
    _episode(oauth_serving, "alpha", "alpha raw turn about the widget")
    _episode(oauth_serving, "beta", "beta raw turn about the widget")
    register_restricted(oauth_serving, ["alpha"], "oauth:kyle")
    as_oauth_caller("kyle")

    out = server.recall_full_turns("widget")
    assert [e["content"] for e in out["episodes"]] == ["alpha raw turn about the widget"]


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


class _RootToken:
    """What SynapseTokenVerifier stamps onto a root-token request: no surface at all."""

    def __init__(self) -> None:
        self.client_id = server._MACHINE_CLIENT_ID
        self.claims = {"kind": "root"}


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


@pytest.mark.parametrize("token", [_RootToken, lambda: None], ids=["root-token", "no-token"])
def test_a_root_token_caller_cannot_borrow_a_full_trust_row(clean, db_url, monkeypatch, token):
    """The lane that is now closed. Every machine that ever ran the plugin has held the
    shared root token, and naming a full-trust legacy row's id used to serve that row's
    whole corpus, personal notes included. Every MCP read path now serves a root caller
    that names a surface exactly what it serves one that names none: nothing."""
    monkeypatch.setattr(server, "DB_URL", db_url)
    monkeypatch.setattr(server, "_recall_engine", _engine(db_url, monkeypatch))
    eid = _episode(clean, "alpha", "alpha discussion of the widget")
    session = clean.execute("SELECT session_id FROM episodes WHERE id = %s", (eid,)).fetchone()[0]
    private = _note(db_url, "User keeps a personal journal", audience="personal")
    full = register_full(clean, "legacy-full-host")
    monkeypatch.setattr(server, "get_access_token", token)

    assert server._caller_trust() == UNKNOWN_SURFACE
    assert server.recall("widget", surface=full).get("episodes") is None
    assert server.recall_full_turns("widget", surface=full).get("episodes", []) == []
    fetched = server.fetch([f"e:{eid}", f"n:{private}"], surface=full)
    assert fetched["episodes"] == [] and fetched["notes"] == []
    assert "error" in server.fetch_session(session, surface=full)


def test_a_device_token_is_unaffected_by_closing_the_root_lane(clean, db_url, monkeypatch):
    """The control for the test above: a full-trust DEVICE reads the same corpus."""
    monkeypatch.setattr(server, "DB_URL", db_url)
    monkeypatch.setattr(server, "_recall_engine", _engine(db_url, monkeypatch))
    eid = _episode(clean, "alpha", "alpha discussion of the widget")
    session = clean.execute("SELECT session_id FROM episodes WHERE id = %s", (eid,)).fetchone()[0]
    private = _note(db_url, "User keeps a personal journal", audience="personal")
    monkeypatch.setattr(server, "get_access_token", lambda: _DeviceToken("dev-home", "full"))

    assert [e["text"] for e in server.recall("widget")["episodes"]] == [
        "alpha discussion of the widget"
    ]
    fetched = server.fetch([f"e:{eid}", f"n:{private}"])
    assert [n["id"] for n in fetched["notes"]] == [f"n:{private}"]
    assert "error" not in server.fetch_session(session)


def test_a_revoked_device_is_served_nothing(clean, db_url, monkeypatch):
    """Revoking clears the token hash, so the credential matches no row — and the id
    lane refuses the row too, so nothing can resurrect it by name."""
    from tests.helpers.surfaces import register_device

    monkeypatch.setattr(server, "DB_URL", db_url)
    monkeypatch.setattr(server, "_recall_engine", _engine(db_url, monkeypatch))
    _episode(clean, "alpha", "alpha discussion of the widget")
    register_device(clean, "gone-tok", trust="full", surface_id="dev-gone", status="revoked")
    monkeypatch.setattr(server, "get_access_token", lambda: None)

    assert server.recall("widget", surface="dev-gone").get("episodes") is None
    assert server._caller_trust() == UNKNOWN_SURFACE


def test_restricted_project_union_ignores_revoked_devices(clean, db_url):
    """A revoked device reads nothing, so letting its allowlist widen the work-safe tier
    would keep classifying notes for an audience that no longer exists."""
    from tests.helpers.surfaces import register_device

    register_restricted(clean, ["alpha"], "approved-work-host")
    register_device(clean, "gone-tok", trust="restricted", projects=["secret"], status="revoked")
    db = Database(db_url)
    try:
        assert restricted_project_union(db) == {"alpha"}
    finally:
        db.close()
