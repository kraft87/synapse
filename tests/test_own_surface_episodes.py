"""Own-surface episode provenance (schema 055): a restricted surface reads what it ingested.

053/054 scope a restricted surface's episode reads to its project allowlist, so a work
laptop with an empty allowlist could not recall even its own conversations. 055 stamps
every ingested turn with the CREDENTIAL-resolved surface that wrote it, and widens the
restricted predicate to ``project = ANY(allowed) OR surface_id = <caller>``.

What has to hold, and each gets a section below:

  * Only a server-established identity (device token, verified OAuth id) stamps or
    widens. A self-reported hostname never does, and neither does an unknown caller.
  * The widening is exact: another surface's rows and legacy NULL-stamped rows stay out.
  * Every restricted read path applies it the same way (recall, recall_full_turns,
    fetch, fetch_session, board banner + digest).
  * The write side stamps from the bearer and ignores anything in the request body.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import psycopg
import pytest

from ingestion.surfaces import (
    UNKNOWN_SURFACE,
    SurfaceTrust,
    episode_scope_sql,
    resolve_caller,
    token_hash,
)
from mcp_server import caller_trust as ct

# ---------------------------------------------------------------------------
# Pure verdict logic: no database needed
# ---------------------------------------------------------------------------

_DEV_A = SurfaceTrust(surface_id="dev-a", known=True, credential_bound=True)
_DEV_B = SurfaceTrust(surface_id="dev-b", known=True, credential_bound=True)
#: A restricted row reached through the self-reported id lane: known, NOT bound.
_LEGACY_A = SurfaceTrust(surface_id="dev-a", known=True)
_FULL_BOUND = SurfaceTrust(surface_id="dev-f", trust="full", known=True, credential_bound=True)


def test_own_surface_only_for_a_credential_bound_restricted_surface():
    assert _DEV_A.own_surface == "dev-a"
    assert _DEV_A.stamp_surface_id == "dev-a"
    # Self-reported id lane: neither stamps nor widens.
    assert _LEGACY_A.own_surface is None and _LEGACY_A.stamp_surface_id is None
    # Unknown: nothing at all.
    assert UNKNOWN_SURFACE.own_surface is None and UNKNOWN_SURFACE.stamp_surface_id is None
    # A bound but unknown verdict (cannot happen via resolve_caller, but must not widen).
    assert SurfaceTrust(surface_id="x", credential_bound=True).own_surface is None
    # Full trust stamps provenance but has no filter to widen.
    assert _FULL_BOUND.stamp_surface_id == "dev-f" and _FULL_BOUND.own_surface is None


def test_episode_scope_sql_shapes():
    assert episode_scope_sql(None, "dev-a") is None  # full trust: no predicate
    assert episode_scope_sql([], None) == ("project = ANY(%s)", [[]])
    assert episode_scope_sql(["w"], "dev-a") == (
        "(project = ANY(%s) OR surface_id = %s)",
        [["w"], "dev-a"],
    )


def test_device_claims_are_credential_bound():
    st = ct.trust_from_claims({"kind": "device", "surface_id": "dev-a", "trust": "restricted"})
    assert st.credential_bound and st.own_surface == "dev-a"


def test_oauth_identity_is_bound_but_the_legacy_surface_param_is_not(monkeypatch):
    def fake_resolve(db_url, token_hash_hex=None, legacy_surface_id=None):
        return SurfaceTrust(surface_id=legacy_surface_id, known=True)

    monkeypatch.setattr(ct, "resolve_caller", fake_resolve)

    class _Tok:
        def __init__(self) -> None:
            self.client_id = "github-app"
            self.claims = {"login": "someone"}

    common = dict(
        db_url="x",
        identity_claims=("login",),
        machine_client_ids={"synapse-machine"},
        claims_identity=lambda claims, keys: claims.get("login", ""),
    )
    oauth = ct.caller_trust(surface="ignored", access_token=_Tok(), **common)
    assert oauth.surface_id == "oauth:someone" and oauth.credential_bound

    legacy = ct.caller_trust(surface="dev-a", access_token=None, **common)
    assert legacy.known and not legacy.credential_bound and legacy.own_surface is None


def test_route_trust_binds_only_a_credential_bound_verdict(monkeypatch):
    monkeypatch.setattr(ct, "resolve_caller", lambda *a, **k: UNKNOWN_SURFACE)
    common = dict(
        db_url="x",
        surface=None,
        access_token=None,
        identity_claims=(),
        machine_client_ids=set(),
        claims_identity=lambda c, k: "",
    )
    with ct.route_trust(_DEV_A):
        assert ct.caller_trust(**common) is _DEV_A
    assert ct.caller_trust(**common) is UNKNOWN_SURFACE  # reset on exit
    with ct.route_trust(_LEGACY_A):  # not bound: ignored
        assert ct.caller_trust(**common) is UNKNOWN_SURFACE


# ---------------------------------------------------------------------------
# Database-backed: serving and writing
# ---------------------------------------------------------------------------

_DB_URL = os.environ.get(
    "SYNAPSE_TEST_URL", "postgresql://synapse:synapse@127.0.0.1:5432/synapse_test"
)


def _db_reachable() -> bool:
    try:
        psycopg.connect(_DB_URL, connect_timeout=2).close()
        return True
    except Exception:  # pragma: no cover - environment dependent
        return False


needs_db = pytest.mark.skipif(not _db_reachable(), reason="no test DB reachable")


@pytest.fixture()
def clean(conn):
    from tests.helpers.surfaces import clear_surfaces

    def _wipe():
        conn.execute("TRUNCATE episodes, extraction_queue RESTART IDENTITY CASCADE")
        conn.execute("DELETE FROM timeline_events")
        conn.execute("DELETE FROM notes")
        clear_surfaces(conn)

    _wipe()
    yield conn
    _wipe()


def _episode(conn, project, content, surface_id=None, session_id=None):
    return conn.execute(
        "INSERT INTO episodes (session_id, sequence, project, content, surface_id) "
        "VALUES (%s, 1, %s, %s, %s) RETURNING id",
        (session_id or f"own-{uuid.uuid4().hex[:8]}", project, content, surface_id),
    ).fetchone()[0]


class _Emb:
    def embed(self, texts, task="document"):
        return [[0.0] * 8 for _ in texts]


def _engine(db_url):
    """A Recall whose embedding/rerank/KG legs are stubbed so assertions are about the
    FILTERS. The BM25 episode leg stays real: its predicate is what is under test."""
    from mcp_server.recall import Recall

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
    r._search_notes = lambda q, emb, proj, audience=None: []
    return r


@pytest.fixture()
def corpus(clean):
    """One row per provenance class, all mentioning 'widget', none allowlisted."""
    return {
        "own": _episode(clean, "work-x", "own widget turn", "dev-a"),
        "own_noproj": _episode(clean, None, "own unlabeled widget turn", "dev-a"),
        "other": _episode(clean, "work-x", "other surface widget turn", "dev-b"),
        "legacy": _episode(clean, "work-x", "legacy widget turn", None),
        "personal": _episode(clean, "family", "personal widget turn", None),
    }


@needs_db
def test_restricted_recall_serves_own_episodes_with_an_empty_allowlist(corpus, db_url):
    out = _engine(db_url).recall("widget", trust=_DEV_A)
    served = sorted(it["text"] for it in out.get("episodes", []))
    assert served == ["own unlabeled widget turn", "own widget turn"]


@needs_db
def test_allowlist_and_own_surface_combine(corpus, clean, db_url):
    st = SurfaceTrust(
        surface_id="dev-a", allowed_projects=("family",), known=True, credential_bound=True
    )
    served = sorted(it["text"] for it in _engine(db_url).recall("widget", trust=st)["episodes"])
    assert served == ["own unlabeled widget turn", "own widget turn", "personal widget turn"]


@needs_db
@pytest.mark.parametrize("trust", [UNKNOWN_SURFACE, _LEGACY_A], ids=["unknown", "legacy-id"])
def test_unknown_or_self_reported_surface_still_serves_nothing(corpus, db_url, trust):
    assert _engine(db_url).recall("widget", trust=trust).get("episodes") is None
    r = _engine(db_url)
    assert r.fetch([f"e:{i}" for i in corpus.values()], trust=trust)["episodes"] == []


@needs_db
def test_full_trust_is_unchanged(corpus, db_url):
    out = _engine(db_url).recall("widget", trust=_FULL_BOUND)
    assert len(out["episodes"]) == len(corpus)


@needs_db
def test_recall_full_turns_serves_own_but_not_other_or_legacy(corpus, db_url):
    out = _engine(db_url).recall_episodes("widget", trust=_DEV_A)
    served = sorted(e["content"] for e in out["episodes"])
    assert served == ["own unlabeled widget turn", "own widget turn"]
    out_b = _engine(db_url).recall_episodes("widget", trust=_DEV_B)
    assert [e["content"] for e in out_b["episodes"]] == ["other surface widget turn"]


@needs_db
def test_fetch_serves_own_ids_only(corpus, db_url):
    ids = [f"e:{i}" for i in corpus.values()]
    out = _engine(db_url).fetch(ids, trust=_DEV_A)
    assert [e["id"] for e in out["episodes"]] == [f"e:{corpus['own']}", f"e:{corpus['own_noproj']}"]


@needs_db
def test_fetch_session_serves_own_turns_and_hides_others(clean, db_url):
    sid_own = f"own-{uuid.uuid4().hex[:8]}"
    sid_other = f"own-{uuid.uuid4().hex[:8]}"
    _episode(clean, None, "own session turn", "dev-a", session_id=sid_own)
    _episode(clean, None, "other session turn", "dev-b", session_id=sid_other)
    r = _engine(db_url)
    own = r.fetch_session(sid_own, trust=_DEV_A)
    assert "error" not in own and len(own["turns"]) == 1
    hidden = r.fetch_session(sid_other, trust=_DEV_A)
    assert "error" in hidden  # indistinguishable from an unknown session


@needs_db
def test_board_banner_and_digest_include_own_rows(corpus, clean, db_url):
    from ingestion.db import Database
    from mcp_server.board import _banner_stats
    from mcp_server.timeline_routes import _recent_events

    n, projects = _banner_stats(db_url, allowed_projects=[], own_surface="dev-a")
    assert n == 2 and projects == ["work-x"]
    assert _banner_stats(db_url, allowed_projects=[], own_surface=None) == (0, [])

    db = Database(db_url)
    try:
        for key, ref in (("own", "ep:own"), ("other", "ep:other"), ("legacy", "ep:legacy")):
            db.insert_timeline_event(
                t_valid="2099-01-01T00:00:00+00:00",
                fact=f"{key} event",
                source="chat",
                source_ref=ref,
                project="work-x",
                salience=2,
                embedding=None,
                embed_model=None,
                source_episode_id=corpus[key],
            )
    finally:
        db.close()
    # The gate's copy of provenance landed on the events.
    stamps = dict(clean.execute("SELECT fact, surface_id FROM timeline_events").fetchall())
    assert stamps == {"own event": "dev-a", "other event": "dev-b", "legacy event": None}

    events = _recent_events(
        db_url, days=36500, min_salience=2, limit=10, project=None,
        allowed_projects=[], own_surface="dev-a",
    )  # fmt: skip
    assert [e["fact"] for e in events] == ["own event"]
    assert _recent_events(db_url, 36500, 2, 10, None, allowed_projects=[], own_surface=None) == []


@needs_db
def test_upsert_never_claims_or_clears_provenance(clean, db_url):
    """A re-write of the same (session_id, sequence) keeps the FIRST writer's stamp: it
    can neither claim a legacy NULL row for a surface nor wipe an existing stamp."""
    from ingestion.db import Database
    from ingestion.models import Episode

    db = Database(db_url)
    try:
        legacy = db.upsert_episode(Episode(session_id="s-legacy", sequence=1, content="old"))
        db.upsert_episode(
            Episode(session_id="s-legacy", sequence=1, content="old", surface_id="dev-a")
        )
        stamped = db.upsert_episode(
            Episode(session_id="s-own", sequence=1, content="mine", surface_id="dev-a")
        )
        db.upsert_episode(Episode(session_id="s-own", sequence=1, content="mine"))
    finally:
        db.close()
    rows = dict(clean.execute("SELECT id, surface_id FROM episodes").fetchall())
    assert rows == {legacy: None, stamped: "dev-a"}


# ---------------------------------------------------------------------------
# Write side: /ingest stamps from the bearer, never from the body
# ---------------------------------------------------------------------------

_ROOT = "root-test-token"


def _ingest_client(db_url):
    from fastmcp import FastMCP
    from starlette.testclient import TestClient

    from mcp_server.http_auth import bearer
    from mcp_server.ingest_route import register

    m = FastMCP("test-ingest-provenance")
    register(
        m,
        lambda: db_url,
        lambda r: True,
        lambda r: ct.request_trust(
            db_url=db_url, machine_token=_ROOT, bearer=bearer(r), surface=None
        ),
    )
    return TestClient(m.http_app())


def _records(tag: str) -> list[dict]:
    sid = str(uuid.uuid4())
    base = {"sessionId": sid, "cwd": "/home/user/work/proj", "timestamp": "2026-09-01T12:00:00Z"}
    return [
        {**base, "type": "user", "uuid": f"u-{tag}", "message": {"role": "user", "content": f"q {tag}"}},
        {
            **base,
            "type": "assistant",
            "uuid": f"a-{tag}",
            "message": {"role": "assistant", "content": [{"type": "text", "text": f"a {tag}"}]},
        },
    ]  # fmt: skip


@needs_db
def test_ingest_stamps_the_device_surface_and_ignores_the_body(clean, db_url):
    from tests.helpers.surfaces import register_device

    tok = "device-token-" + uuid.uuid4().hex
    sid = register_device(clean, tok, trust="restricted", projects=[])
    # Sanity: the DB token lane yields a bound verdict.
    assert resolve_caller(db_url, token_hash_hex=token_hash(tok)).own_surface == sid

    body = {"records": _records("dev"), "surface_id": "dev-evil", "surface": "dev-evil"}
    r = _ingest_client(db_url).post(
        "/ingest", json=body, headers={"Authorization": f"Bearer {tok}"}
    )
    assert r.status_code == 200 and r.json()["ingested"] == 1
    stamps = [row[0] for row in clean.execute("SELECT surface_id FROM episodes").fetchall()]
    assert stamps == [sid]


@needs_db
def test_ingest_with_the_root_token_stamps_null(clean, db_url):
    body = {"records": _records("root"), "surface_id": "dev-evil", "surface": "dev-evil"}
    r = _ingest_client(db_url).post(
        "/ingest", json=body, headers={"Authorization": f"Bearer {_ROOT}"}
    )
    assert r.status_code == 200 and r.json()["ingested"] == 1
    stamps = [row[0] for row in clean.execute("SELECT surface_id FROM episodes").fetchall()]
    assert stamps == [None]


# ---------------------------------------------------------------------------
# Write side: remember() and the spool replay
# ---------------------------------------------------------------------------


@pytest.fixture()
def remember_env(clean, db_url, monkeypatch):
    from mcp_server import server

    monkeypatch.setattr(server, "DB_URL", db_url)
    monkeypatch.setattr(server, "_recall_engine", _engine(db_url))
    monkeypatch.setattr(server, "_notes_deps", lambda: (None, None))
    return clean


def _archived_stamp(conn, episode_id):
    return conn.execute("SELECT surface_id FROM episodes WHERE id = %s", (episode_id,)).fetchone()[
        0
    ]


@needs_db
def test_remember_archive_is_stamped_from_bound_trust_only(remember_env):
    from mcp_server import server
    from tests.helpers.surfaces import register_restricted

    kw = dict(hook="Work fact about the build", body="The build uses the cache.", type="project")
    with ct.route_trust(_DEV_A):
        bound = asyncio.run(server.remember(**kw))
    assert _archived_stamp(remember_env, bound["episode_id"]) == "dev-a"

    # Self-reported id lane: even a registered restricted row does not stamp.
    register_restricted(remember_env, [], "dev-a")
    legacy = asyncio.run(server.remember(surface="dev-a", **{**kw, "hook": "Another fact"}))
    assert _archived_stamp(remember_env, legacy["episode_id"]) is None


@needs_db
def test_spool_replay_stamps_the_device_from_its_bearer(remember_env, db_url):
    from fastmcp import FastMCP
    from starlette.testclient import TestClient

    from mcp_server import server
    from mcp_server.http_auth import bearer
    from mcp_server.remember_routes import register
    from tests.helpers.surfaces import register_device

    tok = "device-token-" + uuid.uuid4().hex
    sid = register_device(remember_env, tok, trust="restricted", projects=[])

    def _trust(r):
        return ct.request_trust(db_url=db_url, machine_token=_ROOT, bearer=bearer(r), surface=None)

    m = FastMCP("test-spool-provenance")
    register(
        m,
        db_url,
        lambda r: True,
        server.remember,
        caller_surface=lambda r: _trust(r).surface_id,
        caller_trust=_trust,
    )
    remember_env.execute("DELETE FROM remember_intents")
    try:
        r = TestClient(m.http_app()).post(
            "/remember/spool",
            json={
                "intent_id": f"i-{uuid.uuid4().hex[:8]}",
                "hook": "Spooled work fact",
                "body": "Written while MCP was down.",
                "surface": "dev-evil",
            },
            headers={"Authorization": f"Bearer {tok}"},
        )
        assert r.status_code == 200, r.text
        assert _archived_stamp(remember_env, r.json()["episode_id"]) == sid
    finally:
        remember_env.execute("DELETE FROM remember_intents")


# ---------------------------------------------------------------------------
# Session-start read routes: /timeline/recent and /preferences/top are scoped too
# ---------------------------------------------------------------------------


def _route_trust(db_url):
    from mcp_server.http_auth import bearer

    return lambda r: ct.request_trust(
        db_url=db_url, machine_token=_ROOT, bearer=bearer(r), surface=None
    )


@pytest.fixture()
def route_corpus(clean, db_url):
    """Three devices (own restricted, other restricted, full) and one row of each
    provenance class in both timeline_events and preferences."""
    from ingestion.db import Database
    from tests.helpers.surfaces import register_device

    toks = {k: f"{k}-token-{uuid.uuid4().hex}" for k in ("own", "other", "full")}
    sids = {
        "own": register_device(clean, toks["own"], trust="restricted", projects=["work-x"]),
        "other": register_device(clean, toks["other"], trust="restricted", projects=[]),
        "full": register_device(clean, toks["full"], trust="full"),
    }
    eps = {
        "own": _episode(clean, "scratch-own", "own turn", sids["own"]),
        "other": _episode(clean, "scratch-other", "other turn", sids["other"]),
    }
    rows = {  # key -> (project, source episode)
        "allowlisted": ("work-x", None),
        "personal": ("family", None),
        "own": ("scratch-own", eps["own"]),
        "other": ("scratch-other", eps["other"]),
    }
    db = Database(db_url)
    try:
        for key, (project, ep) in rows.items():
            db.insert_timeline_event(
                t_valid="2099-01-01T00:00:00+00:00",
                fact=f"{key} event",
                source="chat",
                source_ref=f"ep:{ep}" if ep else f"git:{key}",
                project=project,
                salience=2,
                embedding=None,
                embed_model=None,
                source_episode_id=ep,
            )
            db.insert_preference(
                owner_id="default",
                group_id="technical",
                project=project,
                pref=f"{key} pref",
                polarity="like",
                embedding=None,
                embed_model=None,
                source_ref=f"ep:{ep}" if ep else None,
            )
    finally:
        db.close()
    yield {k: {"Authorization": f"Bearer {t}"} for k, t in toks.items()}
    clean.execute("DELETE FROM preferences")


def _route_client(db_url):
    from fastmcp import FastMCP
    from starlette.testclient import TestClient

    from mcp_server.preferences_routes import register as reg_prefs
    from mcp_server.timeline_routes import register as reg_timeline

    m = FastMCP("test-route-scope")
    reg_timeline(m, db_url, lambda r: True, "", _route_trust(db_url))
    reg_prefs(m, db_url, lambda r: True, _route_trust(db_url))
    return TestClient(m.http_app())


def _served(client, headers):
    tl = client.post("/timeline/recent", json={"days": 90, "limit": 20}, headers=headers)
    pr = client.get("/preferences/top?limit=50", headers=headers)
    assert tl.status_code == 200 and pr.status_code == 200
    return (
        sorted(i["fact"].removesuffix(" event") for i in tl.json()["items"]),
        sorted(i["pref"].removesuffix(" pref") for i in pr.json()["items"]),
    )


@needs_db
def test_session_start_routes_scope_a_restricted_device(route_corpus, db_url):
    c = _route_client(db_url)
    assert _served(c, route_corpus["own"]) == (["allowlisted", "own"], ["allowlisted", "own"])
    assert _served(c, route_corpus["other"]) == (["other"], ["other"])


@needs_db
def test_session_start_routes_full_trust_unchanged(route_corpus, db_url):
    everything = ["allowlisted", "other", "own", "personal"]
    assert _served(_route_client(db_url), route_corpus["full"]) == (everything, everything)


@needs_db
def test_session_start_routes_unknown_caller_gets_nothing(route_corpus, db_url):
    c = _route_client(db_url)
    assert _served(c, {"Authorization": f"Bearer {_ROOT}"}) == ([], [])
    assert _served(c, {}) == ([], [])


@needs_db
def test_session_start_routes_fail_closed_without_a_resolver(route_corpus, db_url):
    from fastmcp import FastMCP
    from starlette.testclient import TestClient

    from mcp_server.preferences_routes import register as reg_prefs
    from mcp_server.timeline_routes import register as reg_timeline

    m = FastMCP("test-route-unwired")
    reg_timeline(m, db_url, lambda r: True, "")
    reg_prefs(m, db_url, lambda r: True)
    assert _served(TestClient(m.http_app()), route_corpus["full"]) == ([], [])
