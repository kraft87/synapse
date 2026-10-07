"""Own-surface provenance (schema 057): a restricted device reads back what it ingested.

053/054 scope a restricted surface's episode reads to its project allowlist, so a work
laptop with an empty allowlist could not recall even its own conversations. 057 stamps
every ingested turn with the CREDENTIAL-resolved surface that wrote it, and widens the
restricted predicate to ``project = ANY(allowed) OR surface_id = <caller>``.

What has to hold, and each gets a section below:

  * Only a server-established identity (device token, verified OAuth id) stamps or
    widens. A self-reported hostname never does, nor an unknown caller, nor the root token.
  * The widening is exact: another surface's rows and unstamped rows stay out.
  * Every restricted read path applies it the same way (recall, recall_full_turns,
    fetch, fetch_session, board banner + digest, /timeline/recent).
  * The write side stamps from the bearer and ignores anything in the request body.
  * Deploy order: with the columns absent (code shipped, 057 not applied yet) ingest and
    serving behave exactly as before 057, with no errors.
  * /timeline/recent and /preferences/top are scoped by the bearer, like the board.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

import psycopg
import pytest

from ingestion import schema_check
from ingestion.surfaces import (
    OWN_SURFACE_PROBE,
    UNKNOWN_SURFACE,
    SurfaceTrust,
    episode_scope_sql,
    resolve_caller,
    token_hash,
)
from mcp_server import caller_trust as ct
from tests.helpers.embed import onehot

_SCHEMA_057 = Path(__file__).resolve().parent.parent / "schema" / "057_own_surface_episodes.sql"

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


def test_oauth_identity_is_bound_but_the_root_token_is_not(monkeypatch):
    def fake_resolve(db_url, token_hash_hex=None, legacy_surface_id=None):
        return SurfaceTrust(surface_id=legacy_surface_id, known=True)

    monkeypatch.setattr(ct, "resolve_caller", fake_resolve)

    class _Tok:
        def __init__(self, client_id: str, claims: dict) -> None:
            self.client_id = client_id
            self.claims = claims

    common = dict(
        db_url="x",
        identity_claims=("login",),
        machine_client_ids={"synapse-machine"},
        claims_identity=lambda claims, keys: claims.get("login", ""),
    )
    oauth = ct.caller_trust(access_token=_Tok("github-app", {"login": "someone"}), **common)
    assert oauth.surface_id == "oauth:someone" and oauth.credential_bound

    for tok in (None, _Tok("synapse-machine", {"kind": "root"})):
        root = ct.caller_trust(access_token=tok, **common)
        assert root is UNKNOWN_SURFACE and root.stamp_surface_id is None


def test_the_in_process_id_lane_never_stamps(monkeypatch):
    """lookup_surface (engine/test callers passing a server-derived id) resolves a row by
    id, which is not a credential: it neither stamps nor widens."""
    from ingestion import surfaces

    def fake_resolve(db_url, token_hash_hex=None, legacy_surface_id=None):
        return SurfaceTrust(surface_id=legacy_surface_id, known=True)

    monkeypatch.setattr(surfaces, "resolve_caller", fake_resolve)
    st = surfaces.lookup_surface("x", "dev-a")
    assert st.known and not st.credential_bound and st.own_surface is None


def test_schema_057_is_registered_and_optional():
    script = (Path(__file__).resolve().parent.parent / "scripts" / "apply_schema.sh").read_text()
    assert "057_own_surface_episodes.sql" in script
    # The image lands before 057 is applied by hand; the boot guard must let it start.
    assert schema_check.is_optional_migration(_SCHEMA_057)


# ---------------------------------------------------------------------------
# Database-backed fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def clean(conn):
    from tests.helpers.surfaces import clear_surfaces

    def _wipe():
        conn.execute("TRUNCATE episodes, extraction_queue RESTART IDENTITY CASCADE")
        conn.execute("DELETE FROM timeline_events")
        conn.execute("DELETE FROM notes")
        conn.execute("DELETE FROM preferences")
        clear_surfaces(conn)

    _wipe()
    OWN_SURFACE_PROBE.reset()
    yield conn
    _wipe()
    OWN_SURFACE_PROBE.reset()


def _episode(conn, project, content, surface_id=None, session_id=None):
    return conn.execute(
        "INSERT INTO episodes (session_id, sequence, project, content, surface_id) "
        "VALUES (%s, 1, %s, %s, %s) RETURNING id",
        (session_id or f"own-{uuid.uuid4().hex[:8]}", project, content, surface_id),
    ).fetchone()[0]


class _Emb:
    def embed(self, texts, task="document"):
        return [onehot(0) for _ in texts]


def _engine(db_url):
    """A Recall whose embedding/rerank/KG legs are stubbed so assertions are about the
    EPISODE filters. The BM25 and vector episode legs stay real: their predicate is what
    is under test."""
    from mcp_server.recall import Recall

    r = Recall(db_url, "")
    r._ensure_embedder = lambda: _Emb()
    r._rerank_pool_scored = lambda q, pool: [(i, 0.9) for i in range(len(pool))]
    r._compact_to_passages = lambda q, eps, n: [
        {"id": e["id"], "text": e.get("content", "")} for e in eps
    ]
    r._search_web_reranked = lambda q, emb: []
    r._search_kg = lambda *a, **k: ([], [])
    r._fetch_superseded_pairs_pg = lambda *a, **k: []
    r._surface_supersessions = lambda *a, **k: []
    r._episode_supersessions = lambda *a, **k: {}
    r._increment_retrieval_counts = lambda ids: None
    r._increment_fact_retrieval_counts = lambda *a, **k: None
    r._record_metrics = lambda m: None
    r._search_notes = lambda *a, **k: []
    return r


def _served_texts(out) -> list[str]:
    assert "warnings" not in out, out.get("warnings")
    return sorted(it["text"] for it in out.get("episodes") or [])


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


# ---------------------------------------------------------------------------
# Serving: own rows outside the allowlist, and nothing else
# ---------------------------------------------------------------------------


def test_restricted_recall_serves_own_episodes_with_an_empty_allowlist(corpus, db_url):
    out = _engine(db_url).recall("widget", trust=_DEV_A)
    assert _served_texts(out) == ["own unlabeled widget turn", "own widget turn"]


def test_another_devices_rows_are_not_visible(corpus, db_url):
    out = _engine(db_url).recall("widget", trust=_DEV_B)
    assert _served_texts(out) == ["other surface widget turn"]
    ids = [f"e:{i}" for i in corpus.values()]
    assert [e["id"] for e in _engine(db_url).fetch(ids, trust=_DEV_B)["episodes"]] == [
        f"e:{corpus['other']}"
    ]


def test_allowlist_and_own_surface_combine(corpus, db_url):
    st = SurfaceTrust(
        surface_id="dev-a", allowed_projects=("family",), known=True, credential_bound=True
    )
    assert _served_texts(_engine(db_url).recall("widget", trust=st)) == [
        "own unlabeled widget turn",
        "own widget turn",
        "personal widget turn",
    ]


@pytest.mark.parametrize("trust", [UNKNOWN_SURFACE, _LEGACY_A], ids=["unknown", "legacy-id"])
def test_unknown_or_self_reported_surface_still_serves_nothing(corpus, db_url, trust):
    assert _engine(db_url).recall("widget", trust=trust).get("episodes") is None
    r = _engine(db_url)
    assert r.fetch([f"e:{i}" for i in corpus.values()], trust=trust)["episodes"] == []


def test_full_trust_is_unchanged(corpus, db_url):
    out = _engine(db_url).recall("widget", trust=_FULL_BOUND)
    assert len(out["episodes"]) == len(corpus)


def test_unstamped_rows_serve_exactly_the_allowlist(clean, db_url):
    """Columns present but nothing stamped yet (the state right after 057 is applied): a
    bound device is served exactly what the allowlist alone serves."""
    _episode(clean, "work-x", "allowlisted widget turn")
    _episode(clean, "family", "personal widget turn")
    _episode(clean, None, "unlabeled widget turn")
    bound = SurfaceTrust(
        surface_id="dev-a", allowed_projects=("work-x",), known=True, credential_bound=True
    )
    allowlist_only = SurfaceTrust(surface_id="dev-a", allowed_projects=("work-x",), known=True)
    served = _served_texts(_engine(db_url).recall("widget", trust=bound))
    assert served == _served_texts(_engine(db_url).recall("widget", trust=allowlist_only))
    assert served == ["allowlisted widget turn"]


def test_recall_full_turns_serves_own_but_not_other_or_legacy(corpus, db_url):
    out = _engine(db_url).recall_episodes("widget", trust=_DEV_A)
    served = sorted(e["content"] for e in out["episodes"])
    assert served == ["own unlabeled widget turn", "own widget turn"]
    out_b = _engine(db_url).recall_episodes("widget", trust=_DEV_B)
    assert [e["content"] for e in out_b["episodes"]] == ["other surface widget turn"]


def test_fetch_serves_own_ids_only(corpus, db_url):
    ids = [f"e:{i}" for i in corpus.values()]
    out = _engine(db_url).fetch(ids, trust=_DEV_A)
    assert [e["id"] for e in out["episodes"]] == [f"e:{corpus['own']}", f"e:{corpus['own_noproj']}"]


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


def _timeline_event(db, key: str, project: str | None, source_episode_id: int | None) -> None:
    db.insert_timeline_event(
        t_valid="2099-01-01T00:00:00+00:00",
        fact=f"{key} event",
        source="chat",
        source_ref=f"ep:{source_episode_id}" if source_episode_id else f"git:{key}",
        project=project,
        salience=2,
        embedding=None,
        embed_model=None,
        source_episode_id=source_episode_id,
    )


def test_board_banner_and_digest_include_own_rows(corpus, clean, db_url):
    from ingestion.db import Database
    from mcp_server.board import _banner_stats
    from mcp_server.timeline_routes import _recent_events

    n, projects = _banner_stats(db_url, allowed_projects=[], own_surface="dev-a")
    assert n == 2 and projects == ["work-x"]
    assert _banner_stats(db_url, allowed_projects=[], own_surface=None) == (0, [])

    db = Database(db_url)
    try:
        for key in ("own", "other", "legacy"):
            _timeline_event(db, key, "work-x", corpus[key])
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


def test_upsert_never_claims_and_a_foreign_rewrite_clears(clean, db_url):
    """A stamp means "every byte of this row came from that surface". A rewrite of the same
    (session_id, sequence) keeps it only when the SAME surface writes again; anyone else's
    rewrite clears it, and a NULL row can never be claimed."""
    from ingestion.db import Database
    from ingestion.models import Episode

    db = Database(db_url)
    try:
        legacy = db.upsert_episode(Episode(session_id="s-legacy", sequence=1, content="old"))
        db.upsert_episode(
            Episode(session_id="s-legacy", sequence=1, content="claim", surface_id="dev-a")
        )
        same = db.upsert_episode(
            Episode(session_id="s-same", sequence=1, content="mine", surface_id="dev-a")
        )
        db.upsert_episode(
            Episode(session_id="s-same", sequence=1, content="mine v2", surface_id="dev-a")
        )
        by_root = db.upsert_episode(
            Episode(session_id="s-root", sequence=1, content="mine", surface_id="dev-a")
        )
        db.upsert_episode(Episode(session_id="s-root", sequence=1, content="root rewrite"))
        by_other = db.upsert_episode(
            Episode(session_id="s-other", sequence=1, content="mine", surface_id="dev-a")
        )
        db.upsert_episode(
            Episode(session_id="s-other", sequence=1, content="theirs", surface_id="dev-b")
        )
    finally:
        db.close()
    rows = dict(clean.execute("SELECT id, surface_id FROM episodes").fetchall())
    assert rows == {legacy: None, same: "dev-a", by_root: None, by_other: None}


# ---------------------------------------------------------------------------
# Write side: /ingest stamps from the bearer, never from the body
# ---------------------------------------------------------------------------

_ROOT = "root-test-token"


def _bearer_trust(db_url):
    from mcp_server.http_auth import bearer

    return lambda r: ct.request_trust(db_url=db_url, machine_token=_ROOT, bearer=bearer(r))


def _ingest_client(db_url, trust_db_url=None):
    from fastmcp import FastMCP
    from starlette.testclient import TestClient

    from mcp_server.ingest_route import register

    m = FastMCP("test-ingest-provenance")
    register(m, lambda: db_url, lambda r: True, _bearer_trust(trust_db_url or db_url))
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

    # ...and that device reads it back with an EMPTY allowlist.
    st = resolve_caller(db_url, token_hash_hex=token_hash(tok))
    assert st.project_filter == []
    eid = clean.execute("SELECT id FROM episodes").fetchone()[0]
    assert [e["id"] for e in _engine(db_url).fetch([f"e:{eid}"], trust=st)["episodes"]] == [
        f"e:{eid}"
    ]


def test_ingest_with_the_root_token_stamps_null(clean, db_url):
    from tests.helpers.surfaces import register_restricted

    # Even when the body names a registered restricted surface.
    register_restricted(clean, [], "dev-evil")
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
    clean.execute("DELETE FROM remember_intents")
    yield clean
    clean.execute("DELETE FROM remember_intents")


def _archived_stamp(conn, episode_id):
    row = conn.execute("SELECT surface_id FROM episodes WHERE id = %s", (episode_id,)).fetchone()
    return row[0]


def test_remember_archive_is_stamped_from_bound_trust_only(remember_env):
    from mcp_server import server

    kw = dict(hook="Work fact about the build", body="The build uses the cache.", type="project")
    bound = asyncio.run(server._remember_as(trust=_DEV_A, **kw))
    assert _archived_stamp(remember_env, bound["episode_id"]) == "dev-a"

    # A verdict reached by id rather than by credential does not stamp, nor does unknown.
    for i, trust in enumerate((_LEGACY_A, UNKNOWN_SURFACE)):
        out = asyncio.run(server._remember_as(trust=trust, **{**kw, "hook": f"Other fact {i}"}))
        assert _archived_stamp(remember_env, out["episode_id"]) is None


def _spool_client(db_url):
    from fastmcp import FastMCP
    from starlette.testclient import TestClient

    from mcp_server import server
    from mcp_server.remember_routes import register

    m = FastMCP("test-spool-provenance")
    register(m, db_url, lambda r: True, server._remember_as, resolve_trust=_bearer_trust(db_url))
    return TestClient(m.http_app())


def _spool(client, headers, hook):
    r = client.post(
        "/remember/spool",
        json={
            "intent_id": f"i-{uuid.uuid4().hex[:8]}",
            "hook": hook,
            "body": "Written while MCP was down.",
            "surface": "dev-evil",
        },
        headers=headers,
    )
    assert r.status_code == 200, r.text
    return r.json()["episode_id"]


def test_spool_replay_stamps_the_device_from_its_bearer(remember_env, db_url):
    from tests.helpers.surfaces import register_device

    tok = "device-token-" + uuid.uuid4().hex
    sid = register_device(remember_env, tok, trust="restricted", projects=[])
    eid = _spool(_spool_client(db_url), {"Authorization": f"Bearer {tok}"}, "Spooled work fact")
    assert _archived_stamp(remember_env, eid) == sid


def test_spool_replay_with_the_root_token_stamps_null(remember_env, db_url):
    """The root token naming a registered restricted surface in the body must not file the
    row under that surface."""
    from tests.helpers.surfaces import register_restricted

    register_restricted(remember_env, [], "dev-evil")
    eid = _spool(_spool_client(db_url), {"Authorization": f"Bearer {_ROOT}"}, "Root spool")
    assert _archived_stamp(remember_env, eid) is None


# ---------------------------------------------------------------------------
# Deploy order: code live, schema 057 not applied yet
# ---------------------------------------------------------------------------


@pytest.fixture()
def pre057(clean, conn, db_url):
    """A database without schema 057's columns (see tests/helpers/pre057.py)."""
    from tests.helpers.pre057 import pre057_schema

    with pre057_schema(conn, db_url) as out:
        yield out


def _pre057_episode(conn, schema, project, content, session_id=None):
    return conn.execute(
        f"INSERT INTO {schema}.episodes (session_id, sequence, project, content) "
        "VALUES (%s, 1, %s, %s) RETURNING id",
        (session_id or f"pre-{uuid.uuid4().hex[:8]}", project, content),
    ).fetchone()[0]


def test_pre057_probe_reports_the_columns_missing(pre057, db_url):
    url, _schema = pre057
    with psycopg.connect(url, autocommit=True) as c:
        assert not OWN_SURFACE_PROBE.ready(c, url)
    with psycopg.connect(db_url, autocommit=True) as c:
        assert OWN_SURFACE_PROBE.ready(c, db_url)


def test_pre057_ingest_and_timeline_write_as_before(pre057, clean, caplog):
    """A device-token ingest and a gate timeline write succeed, unstamped, on a database
    without the columns. Nothing errors."""
    from ingestion.db import Database
    from tests.helpers.surfaces import register_device

    url, schema = pre057
    tok = "device-token-" + uuid.uuid4().hex
    register_device(clean, tok, trust="restricted", projects=[])
    with caplog.at_level(logging.ERROR):
        r = _ingest_client(url).post(
            "/ingest", json={"records": _records("pre")}, headers={"Authorization": f"Bearer {tok}"}
        )
    assert r.status_code == 200 and r.json()["ingested"] == 1, r.text
    eid = clean.execute(f"SELECT id FROM {schema}.episodes").fetchone()[0]

    db = Database(url)
    try:
        _timeline_event(db, "pre", "work-x", eid)
    finally:
        db.close()
    facts = clean.execute(f"SELECT fact FROM {schema}.timeline_events").fetchall()
    assert facts == [("pre event",)]
    assert not [rec for rec in caplog.records if rec.levelno >= logging.ERROR]


def test_pre057_restricted_reads_serve_the_allowlist_only(pre057, clean):
    """Every restricted read path, on a database without the columns, serves exactly the
    pre-057 allowlist answer to a bound device: no own-surface rows, no warnings."""
    from ingestion.db import Database
    from mcp_server.board import _banner_stats
    from mcp_server.timeline_routes import _recent_events

    url, schema = pre057
    sid = f"pre-{uuid.uuid4().hex[:8]}"
    allowed = _pre057_episode(clean, schema, "work-x", "allowlisted widget turn", sid)
    other = _pre057_episode(clean, schema, "family", "personal widget turn")
    db = Database(url)
    try:
        _timeline_event(db, "allowlisted", "work-x", allowed)
        _timeline_event(db, "personal", "family", other)
    finally:
        db.close()
    dev = SurfaceTrust(
        surface_id="dev-a", allowed_projects=("work-x",), known=True, credential_bound=True
    )
    r = _engine(url)
    assert _served_texts(r.recall("widget", trust=dev)) == ["allowlisted widget turn"]
    eps = r.recall_episodes("widget", trust=dev)
    assert [e["content"] for e in eps["episodes"]] == ["allowlisted widget turn"]
    assert "warnings" not in eps
    fetched = r.fetch([f"e:{allowed}", f"e:{other}"], trust=dev)["episodes"]
    assert [e["id"] for e in fetched] == [f"e:{allowed}"]
    assert len(r.fetch_session(sid, trust=dev)["turns"]) == 1
    assert _banner_stats(url, allowed_projects=["work-x"], own_surface="dev-a") == (1, ["work-x"])
    events = _recent_events(
        url, 36500, 2, 10, None, allowed_projects=["work-x"], own_surface="dev-a"
    )
    assert [e["fact"] for e in events] == ["allowlisted event"]

    # Full trust is untouched by the missing column.
    assert len(_engine(url).recall("widget", trust=_FULL_BOUND)["episodes"]) == 2


def test_pre057_a_column_lost_after_a_positive_probe_falls_back(pre057, clean):
    """Belt and braces: if the probe already cached "applied" and the column then
    disappears (057 rolled back under a running process), the read legs and the episode
    upsert fall back to the pre-057 statements instead of failing."""
    from ingestion.db import Database
    from ingestion.models import Episode

    url, schema = pre057
    _pre057_episode(clean, schema, "work-x", "allowlisted widget turn")
    OWN_SURFACE_PROBE._ok[url] = True  # as if a probe had found the column
    dev = SurfaceTrust(
        surface_id="dev-a", allowed_projects=("work-x",), known=True, credential_bound=True
    )
    assert _served_texts(_engine(url).recall("widget", trust=dev)) == ["allowlisted widget turn"]
    assert OWN_SURFACE_PROBE._ok[url] is False

    OWN_SURFACE_PROBE._ok[url] = True
    db = Database(url)
    try:
        eid = db.upsert_episode(
            Episode(session_id="s-lost", sequence=1, content="x", surface_id="dev-a")
        )
    finally:
        db.close()
    assert clean.execute(
        f"SELECT 1 FROM {schema}.episodes WHERE id = %s",
        (eid,),
    ).fetchone()


def test_full_trust_and_unknown_callers_never_probe(monkeypatch, db_url):
    """The probe runs only for a caller that could use the column, so full-trust and
    unknown callers send exactly the pre-057 statements."""
    from mcp_server.recall import Recall

    r = Recall(db_url, "")

    def boom():
        raise AssertionError("probed")

    monkeypatch.setattr(r, "_ensure_pg", boom)
    for st in (_FULL_BOUND, UNKNOWN_SURFACE, _LEGACY_A):
        assert r._own_surface(st) is None


def test_a_negative_probe_is_rechecked_after_the_ttl(clean, db_url, monkeypatch):
    """Applying 057 under a running process takes effect without a restart."""
    OWN_SURFACE_PROBE.missing(db_url)
    with psycopg.connect(db_url, autocommit=True) as c:
        assert OWN_SURFACE_PROBE.ready(c, db_url) is False  # inside the TTL: no re-probe
        monkeypatch.setattr(OWN_SURFACE_PROBE, "reprobe_s", 0.0)
        assert OWN_SURFACE_PROBE.ready(c, db_url) is True


# ---------------------------------------------------------------------------
# Session-start read routes: /timeline/recent and /preferences/top are scoped too
# ---------------------------------------------------------------------------


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
            _timeline_event(db, key, project, ep)
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
    return {k: {"Authorization": f"Bearer {t}"} for k, t in toks.items()}


def _route_client(db_url, *, wired: bool = True):
    from fastmcp import FastMCP
    from starlette.testclient import TestClient

    from mcp_server.preferences_routes import register as reg_prefs
    from mcp_server.timeline_routes import register as reg_timeline

    m = FastMCP("test-route-scope")
    if wired:
        reg_timeline(m, db_url, lambda r: True, "", _bearer_trust(db_url))
        reg_prefs(m, db_url, lambda r: True, _bearer_trust(db_url))
    else:
        reg_timeline(m, db_url, lambda r: True, "")
        reg_prefs(m, db_url, lambda r: True)
    return TestClient(m.http_app())


def _served(client, headers):
    tl = client.post(
        "/timeline/recent",
        json={"days": 36500, "limit": 20, "project": None, "surface": "ignored"},
        headers=headers,
    )
    pr = client.get("/preferences/top?limit=50&surface=ignored", headers=headers)
    assert tl.status_code == 200 and pr.status_code == 200
    return (
        sorted(i["fact"].removesuffix(" event") for i in tl.json()["items"]),
        sorted(i["pref"].removesuffix(" pref") for i in pr.json()["items"]),
    )


def test_session_start_routes_scope_a_restricted_device(route_corpus, db_url):
    """Timeline: the allowlist plus the device's own events. Preferences: the allowlist
    only (a preference's text can be merged from later turns, so its first turn's ref
    proves nothing about which device the text came from)."""
    c = _route_client(db_url)
    assert _served(c, route_corpus["own"]) == (["allowlisted", "own"], ["allowlisted"])
    assert _served(c, route_corpus["other"]) == (["other"], [])


def test_session_start_routes_full_trust_unchanged(route_corpus, db_url):
    everything = ["allowlisted", "other", "own", "personal"]
    assert _served(_route_client(db_url), route_corpus["full"]) == (everything, everything)


def test_session_start_routes_unknown_caller_gets_nothing(route_corpus, db_url):
    c = _route_client(db_url)
    assert _served(c, {"Authorization": f"Bearer {_ROOT}"}) == ([], [])
    assert _served(c, {}) == ([], [])


def test_session_start_routes_fail_closed_without_a_resolver(route_corpus, db_url):
    assert _served(_route_client(db_url, wired=False), route_corpus["full"]) == ([], [])
