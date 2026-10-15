"""One project identity on the server: allowlists, skill scopes and /ingest labels.

A project's label keeps its folder's case, and the same folder reaches the server in
several spellings (a Windows path typed in lower case, a clone made under another name,
an older plugin that sent the whole path). The server side of that has four parts, and
each gets a section below:

  * ``ingestion.surfaces.expand_allowlist``: a restricted allowlist matches every stored
    casing of its entries, so the reads keep their exact ``project = ANY(...)``
    predicate. Any failure, or a database with no index leading with ``project``,
    matches exactly.
  * ``resolve_caller`` applies it to restricted verdicts only, and the serving paths
    (fetch, recall, the board) then serve every casing and nothing else.
  * ``derive_audience`` and ``/skills/list`` compare project names case-insensitively;
    ``_evidence_scope`` folds case and counts a path-shaped label as its folder.
  * ``/ingest`` stores a label, never a path: the body's ``project`` is cleaned, and
    without a usable one the parser labels from a Windows or POSIX cwd alike.

The first section uses fake connections; the rest runs against the shared Postgres test
DB (tagged into the ``db`` xdist group via conftest). Every row is synthetic: its session,
surface or skill id starts with a per-test ``pidtest-<hex>-`` prefix and its project names
carry the same hex, and each test deletes exactly its own rows before and after.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import psycopg
import pytest
from psycopg.rows import dict_row
from psycopg.types.json import Json

from ingestion import surfaces
from ingestion.surfaces import (
    DEFAULT_AUDIENCE,
    PROJECT_INDEX_PROBE,
    RESTRICTED_AUDIENCE,
    UNKNOWN_SURFACE,
    SurfaceTrust,
    derive_audience,
    expand_allowlist,
    lookup_surface,
    resolve_caller,
    token_hash,
)
from tests.helpers.embed import onehot

_P = "pidtest-"


@pytest.fixture(autouse=True)
def _fresh_index_probe():
    """The project-index probe caches per key for the life of the process; start each test
    from "never probed" so a fake connection's answer cannot leak into a real one."""
    PROJECT_INDEX_PROBE.reset()
    yield
    PROJECT_INDEX_PROBE.reset()


# ---------------------------------------------------------------------------
# expand_allowlist against a fake connection: no database needed
# ---------------------------------------------------------------------------


class _Result:
    def __init__(self, rows) -> None:
        self._rows = list(rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


class _FakeConn:
    """Answers the project-index probe and the variant walk, recording each call."""

    def __init__(self, *, index=True, rows=(), fail=None, probe_fail=None) -> None:
        self.index = index
        self.rows = list(rows)
        self.fail = fail
        self.probe_fail = probe_fail
        self.probes = 0
        self.variant_params: list = []

    @contextmanager
    def transaction(self):
        yield

    def execute(self, sql, params=None):
        if "pg_index" in sql:
            self.probes += 1
            if self.probe_fail is not None:
                raise self.probe_fail
            return _Result([(1,)] if self.index else [])
        assert "WITH RECURSIVE" in sql, sql
        self.variant_params.append(params)
        if self.fail is not None:
            raise self.fail
        return _Result(self.rows)


class _Untouchable:
    """A connection that must not be used at all."""

    def __getattr__(self, name):
        raise AssertionError(f"connection touched: {name}")


@pytest.fixture()
def key():
    return f"fake-{uuid.uuid4().hex}"


@pytest.fixture()
def clock(monkeypatch):
    """A settable monotonic clock for ingestion.surfaces only (the probe keeps real time)."""
    now = [1000.0]
    monkeypatch.setattr(surfaces, "time", SimpleNamespace(monotonic=lambda: now[0]))
    return now


def test_empty_allowlist_is_untouched(key):
    assert expand_allowlist(_Untouchable(), key, ()) == ()
    assert surfaces._variant_cache == {}


def test_without_the_index_the_allowlist_matches_exactly(key):
    conn = _FakeConn(index=False, rows=[("alpha",), ("ALPHA",)])
    allowed = ("Alpha",)
    assert expand_allowlist(conn, key, allowed) is allowed
    assert conn.probes == 1 and conn.variant_params == []
    assert surfaces._variant_cache == {}


def test_variants_are_appended_after_the_allowlist_without_duplicates(key):
    conn = _FakeConn(rows=[("ALPHA",), ("Alpha",), ("alpha",), ("beta",)])
    out = expand_allowlist(conn, key, ("Alpha", "Beta"))
    assert out == ("Alpha", "Beta", "ALPHA", "alpha", "beta")
    # The walk is asked for the lower-cased allowlist, once, sorted.
    assert conn.variant_params == [(["alpha", "beta"],)]


def test_dict_rows_are_read_by_column_name(key):
    conn = _FakeConn(rows=[{"project": "ALPHA"}, {"project": "Alpha"}])
    assert expand_allowlist(conn, key, ("Alpha",)) == ("Alpha", "ALPHA")


@pytest.mark.parametrize(
    "conn",
    [
        _FakeConn(fail=psycopg.OperationalError("connection lost")),
        _FakeConn(fail=RuntimeError("anything at all")),
        _FakeConn(probe_fail=psycopg.OperationalError("probe failed")),
    ],
    ids=["walk-db-error", "walk-other-error", "probe-error"],
)
def test_any_failure_leaves_the_allowlist_unchanged_and_uncached(conn, key):
    allowed = ("Alpha", "Beta")
    assert expand_allowlist(conn, key, allowed) is allowed
    assert surfaces._variant_cache == {}


def test_a_failed_walk_is_retried_on_the_next_call(key):
    conn = _FakeConn(fail=RuntimeError("blip"), rows=[("ALPHA",)])
    assert expand_allowlist(conn, key, ("Alpha",)) == ("Alpha",)
    conn.fail = None
    assert expand_allowlist(conn, key, ("Alpha",)) == ("Alpha", "ALPHA")
    assert len(conn.variant_params) == 2


def test_cache_hit_within_the_ttl_does_not_requery(key, clock):
    conn = _FakeConn(rows=[("ALPHA",), ("Alpha",), ("alpha",)])
    assert expand_allowlist(conn, key, ("Alpha",)) == ("Alpha", "ALPHA", "alpha")
    clock[0] += surfaces._VARIANT_TTL_SECONDS - 1
    # Same allowlist, and one that differs only in case: both are the cached entry.
    assert expand_allowlist(conn, key, ("Alpha",)) == ("Alpha", "ALPHA", "alpha")
    assert expand_allowlist(conn, key, ("alpha",)) == ("alpha", "ALPHA", "Alpha")
    assert len(conn.variant_params) == 1

    # Another key (another database) is its own entry.
    other = _FakeConn(rows=[("Alpha",)])
    assert expand_allowlist(other, key + "-other", ("Alpha",)) == ("Alpha",)
    assert len(other.variant_params) == 1

    # Past the TTL the walk runs again and sees a casing that appeared meanwhile.
    conn.rows.append(("aLpHa",))
    clock[0] += 2
    assert expand_allowlist(conn, key, ("Alpha",)) == ("Alpha", "ALPHA", "alpha", "aLpHa")
    assert len(conn.variant_params) == 2


def test_with_case_variants_touches_restricted_allowlists_only(key):
    full = SurfaceTrust(surface_id="dev-f", trust="full", allowed_projects=("Alpha",), known=True)
    empty = SurfaceTrust(surface_id="dev-e", known=True)
    assert surfaces._with_case_variants(_Untouchable(), key, full) is full
    assert surfaces._with_case_variants(_Untouchable(), key, empty) is empty

    restricted = SurfaceTrust(
        surface_id="dev-r", allowed_projects=("Alpha",), known=True, credential_bound=True
    )
    same = surfaces._with_case_variants(_FakeConn(rows=[("Alpha",)]), key + "-1", restricted)
    assert same is restricted  # nothing new: the verdict object is kept
    wider = surfaces._with_case_variants(_FakeConn(rows=[("alpha",)]), key + "-2", restricted)
    assert wider.allowed_projects == ("Alpha", "alpha")
    assert (wider.surface_id, wider.trust, wider.known, wider.credential_bound) == (
        "dev-r",
        "restricted",
        True,
        True,
    )


class _UnionDb:
    def __init__(self, projects) -> None:
        self.projects = list(projects)

    def restricted_surface_projects(self):
        return self.projects


@pytest.mark.parametrize(
    "union,project,expected",
    [
        ({"alpha"}, "ALPHA", RESTRICTED_AUDIENCE),
        ({"ALPHA"}, "alpha", RESTRICTED_AUDIENCE),
        ({"Alpha"}, "Alpha", RESTRICTED_AUDIENCE),
        ({"alpha"}, "Beta", DEFAULT_AUDIENCE),
        ({"alpha"}, None, DEFAULT_AUDIENCE),
        (set(), "alpha", DEFAULT_AUDIENCE),
    ],
)
def test_derive_audience_compares_the_union_case_insensitively(union, project, expected):
    db = _UnionDb(union)
    got = derive_audience(db, explicit=None, caller_restricted=False, project=project)
    assert got == expected


# ---------------------------------------------------------------------------
# Database-backed fixtures
# ---------------------------------------------------------------------------


@dataclass
class _Names:
    prefix: str  # this test's rows: session, surface and skill ids all start with it
    alpha: str  # the allowlisted spelling
    lower: str
    upper: str
    beta: str  # an unrelated project

    def sid(self) -> str:
        return f"{self.prefix}{uuid.uuid4().hex[:12]}"


def _wipe(conn, prefix: str) -> None:
    """Delete this test's rows only, so tests that land on different xdist workers
    cannot delete each other's rows mid-test."""
    like = f"{prefix}%"
    conn.execute("DELETE FROM extraction_queue WHERE session_id LIKE %s", (like,))
    conn.execute("DELETE FROM episodes WHERE session_id LIKE %s", (like,))
    conn.execute("DELETE FROM surfaces WHERE surface_id LIKE %s", (like,))
    conn.execute("DELETE FROM skills_lane.skill_files WHERE skill_name LIKE %s", (like,))
    conn.execute("DELETE FROM skills_lane.skill_registry WHERE name LIKE %s", (like,))
    conn.execute("DELETE FROM skills_lane.skill_history WHERE name LIKE %s", (like,))


@pytest.fixture()
def ident(conn):
    """Unique project spellings for this test, and a clean slate around it."""
    sfx = uuid.uuid4().hex[:8]
    prefix = f"{_P}{sfx}-"
    _wipe(conn, prefix)
    yield _Names(
        prefix=prefix,
        alpha=f"Alpha-{sfx}",
        lower=f"alpha-{sfx}",
        upper=f"ALPHA-{sfx}",
        beta=f"Beta-{sfx}",
    )
    _wipe(conn, prefix)


def _episode(conn, project, content, *, session_id: str, cwd=None) -> int:
    return conn.execute(
        "INSERT INTO episodes (session_id, sequence, project, platform, content, metadata) "
        "VALUES (%s, 1, %s, 'claude_code', %s, %s) RETURNING id",
        (
            session_id,
            project,
            content,
            Json({"cwd": cwd} if cwd else {}),
        ),
    ).fetchone()[0]


def _device(conn, n: _Names, *, trust="restricted", projects=()) -> str:
    """A credential-bound device (schema 054); returns its plaintext test token."""
    from tests.helpers.surfaces import register_device

    tok = f"test-token-{uuid.uuid4().hex}"
    sid = f"{n.prefix}dev-{tok[-8:]}"
    register_device(conn, tok, trust=trust, projects=list(projects), surface_id=sid)
    return tok


def _seed_all(conn, n: _Names) -> dict[str, int]:
    return {
        p: _episode(conn, p, f"widget turn in {p}", session_id=n.sid())
        for p in (n.alpha, n.lower, n.upper, n.beta)
    }


# ---------------------------------------------------------------------------
# resolve_caller on the real database
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spelling", ["alpha", "lower", "upper"])
def test_restricted_allowlist_resolves_every_stored_casing(ident, conn, db_url, spelling):
    _seed_all(conn, ident)
    listed = getattr(ident, spelling)
    tok = _device(conn, ident, projects=[listed])

    st = resolve_caller(db_url, token_hash_hex=token_hash(tok))

    assert st.known and st.restricted and st.credential_bound
    assert st.allowed_projects[0] == listed  # the allowlist as written comes first
    assert sorted(st.allowed_projects) == sorted([ident.alpha, ident.lower, ident.upper])
    assert ident.beta not in st.allowed_projects
    assert st.project_filter == list(st.allowed_projects)


def test_an_allowlist_entry_with_no_stored_casing_is_kept_as_written(ident, conn, db_url):
    _seed_all(conn, ident)
    tok = _device(conn, ident, projects=[ident.alpha, "Gamma-unstored"])
    st = resolve_caller(db_url, token_hash_hex=token_hash(tok))
    assert st.allowed_projects[:2] == (ident.alpha, "Gamma-unstored")
    assert set(st.allowed_projects) == {ident.alpha, ident.lower, ident.upper, "Gamma-unstored"}


def test_full_trust_and_empty_allowlists_are_untouched(ident, conn, db_url):
    _seed_all(conn, ident)
    full = resolve_caller(
        db_url,
        token_hash_hex=token_hash(_device(conn, ident, trust="full", projects=[ident.alpha])),
    )
    assert full.trust == "full" and full.allowed_projects == (ident.alpha,)
    assert full.project_filter is None

    empty = resolve_caller(db_url, token_hash_hex=token_hash(_device(conn, ident, projects=[])))
    assert empty.known and empty.restricted and empty.allowed_projects == ()


def test_unknown_token_is_the_unknown_surface(ident, conn, db_url):
    _seed_all(conn, ident)
    _device(conn, ident, projects=[ident.alpha])
    st = resolve_caller(db_url, token_hash_hex=token_hash(f"never-issued-{uuid.uuid4().hex}"))
    assert st == UNKNOWN_SURFACE
    assert st.project_filter == []


def test_the_id_lane_expands_too(ident, conn, db_url):
    from tests.helpers.surfaces import register_restricted

    _seed_all(conn, ident)
    sid = register_restricted(conn, [ident.upper], f"{ident.prefix}oauth-someone")
    st = lookup_surface(db_url, sid)
    assert st.known and not st.credential_bound
    assert st.allowed_projects[0] == ident.upper
    assert set(st.allowed_projects) == {ident.alpha, ident.lower, ident.upper}


def test_without_the_project_index_the_allowlist_matches_exactly(ident, conn, db_url, monkeypatch):
    """A database with no index leading with ``project``: the skip scan would read the
    whole table per step, so the allowlist is served exactly as written (the pre-change
    behaviour)."""
    _seed_all(conn, ident)
    tok = _device(conn, ident, projects=[ident.alpha])
    monkeypatch.setattr(PROJECT_INDEX_PROBE, "ready", lambda c, key="": False)
    st = resolve_caller(db_url, token_hash_hex=token_hash(tok))
    assert st.known and st.allowed_projects == (ident.alpha,)


# ---------------------------------------------------------------------------
# End to end: what the restricted caller is actually served
# ---------------------------------------------------------------------------


class _Emb:
    def embed(self, texts, task="document"):
        return [onehot(0) for _ in texts]


def _engine(db_url):
    """A Recall whose embedding/rerank/KG legs are stubbed (as in
    test_own_surface_episodes) so assertions are about the EPISODE filters."""
    from mcp_server.recall import Recall

    r = Recall(db_url, "")
    r._ensure_embedder = lambda: _Emb()
    r._rerank_pool_scored = lambda q, pool: [(i, 0.9) for i in range(len(pool))]
    r._compact_to_passages = lambda q, eps, n, **_k: [
        {"id": e["id"], "text": e.get("content", "")} for e in eps
    ]
    r._search_web_reranked = lambda q, emb, **_k: []
    r._search_kg = lambda *a, **k: ([], [])
    r._fetch_superseded_pairs_pg = lambda *a, **k: []
    r._surface_supersessions = lambda *a, **k: []
    r._episode_supersessions = lambda *a, **k: {}
    r._increment_retrieval_counts = lambda ids: None
    r._increment_fact_retrieval_counts = lambda *a, **k: None
    r._record_metrics = lambda m: None
    r._search_notes = lambda *a, **k: []
    return r


def test_restricted_caller_is_served_every_casing_and_not_the_other_project(ident, conn, db_url):
    from mcp_server.board import _banner_stats, build_board

    ids = _seed_all(conn, ident)
    tok = _device(conn, ident, projects=[ident.lower])  # the allowlist typed in lower case
    st = resolve_caller(db_url, token_hash_hex=token_hash(tok))
    alpha_ids = {f"e:{ids[p]}" for p in (ident.alpha, ident.lower, ident.upper)}
    every_id = [f"e:{i}" for i in ids.values()]
    r = _engine(db_url)

    # Before: the verdict exactly as stored (no expansion) is served one casing only.
    exact = SurfaceTrust(
        surface_id=st.surface_id,
        allowed_projects=(ident.lower,),
        known=True,
        credential_bound=True,
    )
    assert [e["id"] for e in r.fetch(every_id, trust=exact)["episodes"]] == [
        f"e:{ids[ident.lower]}"
    ]

    # After: fetch serves all three casings and withholds the unrelated project.
    fetched = r.fetch(every_id, trust=st)["episodes"]
    assert {e["id"] for e in fetched} == alpha_ids
    assert ident.beta not in {e["project"] for e in fetched}

    # recall's episode leg applies the same predicate.
    out = r.recall("widget", trust=st)
    assert "warnings" not in out, out.get("warnings")
    served = {it["id"] for it in out.get("episodes") or []}
    assert served == alpha_ids

    # The board's banner counts and names the three casings, never the other project.
    n, n_projects, names = _banner_stats(
        db_url, allowed_projects=st.project_filter, own_surface=st.own_surface
    )
    assert (n, n_projects) == (3, 3)
    assert set(names) == {ident.alpha, ident.lower, ident.upper}
    board = build_board(db_url, None, trust=st)
    assert board["trust"] == "restricted"
    assert ident.beta not in board["text"]
    assert any(p in board["text"] for p in (ident.alpha, ident.lower, ident.upper))


def test_derive_audience_against_the_real_union(ident, conn, db_url):
    from ingestion.db import Database
    from tests.helpers.surfaces import register_restricted

    register_restricted(conn, [ident.lower], f"{ident.prefix}union")
    db = Database(db_url)
    try:
        kw = dict(explicit=None, caller_restricted=False)
        assert derive_audience(db, project=ident.upper, **kw) == RESTRICTED_AUDIENCE
        assert derive_audience(db, project=ident.alpha, **kw) == RESTRICTED_AUDIENCE
        assert derive_audience(db, project=ident.beta, **kw) == DEFAULT_AUDIENCE
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Skills: /skills/list scope matching and the evidence scope of a derived skill
# ---------------------------------------------------------------------------


def _publish(db_url, name: str, scope: str) -> None:
    from mcp_server.skill_sync_routes import _publish_skill

    body = f"---\nname: {name}\ndescription: Demo skill.\n---\n# {name}\n\n1. Do it.\n"
    out = _publish_skill(
        db_url,
        {
            "name": name,
            "scope": scope,
            "body": body,
            "description": "",
            "content_modified_at": "2026-01-01T00:00:00+00:00",
            "files": [],
        },
    )
    assert out.get("status") != "retired", out


def _mine(listing: dict, prefix: str) -> dict[str, str]:
    return {s["name"]: s["scope"] for s in listing["skills"] if s["name"].startswith(prefix)}


def test_list_skills_matches_a_project_scope_in_any_case(ident, db_url):
    from mcp_server.skill_sync_routes import _list_skills

    mine = ident.prefix
    proj, glob = f"{mine}proj-skill", f"{mine}global-skill"
    _publish(db_url, proj, f"project:{ident.alpha}")
    _publish(db_url, glob, "global")

    for spelling in (ident.alpha, ident.lower, ident.upper):
        # Served under the scope as STORED, whatever spelling asked for it.
        assert _mine(_list_skills(db_url, f"project:{spelling}"), mine) == {
            proj: f"project:{ident.alpha}"
        }
    assert _mine(_list_skills(db_url, f"project:{ident.beta}"), mine) == {}
    assert _mine(_list_skills(db_url, "global"), mine) == {glob: "global"}


def _evidence_scope_of(db_url, sessions: list[str]) -> str:
    from mcp_server.skill_sync_routes import _evidence_scope

    with psycopg.connect(db_url, autocommit=True, row_factory=dict_row) as c:
        return _evidence_scope(c, [{"session_id": s, "class": "judge"} for s in sessions])


def _sessions(conn, n: _Names, labels: list[str]) -> list[str]:
    out = []
    for label in labels:
        sid = n.sid()
        cwd = label if ("\\" in label or "/" in label) else f"/work/{label}"
        _episode(conn, label, f"evidence turn {sid}", session_id=sid, cwd=cwd)
        out.append(sid)
    return out


def test_evidence_scope_folds_case_and_names_the_most_common_spelling(ident, conn, db_url):
    a, lo = ident.alpha, ident.lower
    assert _evidence_scope_of(db_url, _sessions(conn, ident, [a, a, lo])) == f"project:{a}"
    assert _evidence_scope_of(db_url, _sessions(conn, ident, [lo, lo, a])) == f"project:{lo}"
    # A tie is settled the same way every time (sorted first).
    assert _evidence_scope_of(db_url, _sessions(conn, ident, [lo, a])) == f"project:{min(a, lo)}"
    # Genuinely different projects stay global, as before.
    assert _evidence_scope_of(db_url, _sessions(conn, ident, [a, ident.beta])) == "global"


@pytest.mark.parametrize(
    "labels",
    [
        ["D:\\src\\{a}"],
        ["D:\\src\\{a}", "d:\\SRC\\{a}"],  # two casings of one path
        ["D:\\src\\{a}", "{a}"],  # the path and the label a 0.21 client sends
        ["/srv/work/{a}"],
    ],
    ids=["windows-path", "windows-path-two-casings", "path-and-bare-name", "posix-path"],
)
def test_evidence_scope_names_the_folder_of_a_path_label(ident, conn, db_url, labels):
    """Legacy rows labelled with their whole path (the pre-fix parsers) count as the
    folder's name: never a ``project:D:\\src\\...`` scope no client can match, and never
    'global', which relabel_projects cannot move back into the project."""
    sessions = _sessions(conn, ident, [label.format(a=ident.alpha) for label in labels])
    assert _evidence_scope_of(db_url, sessions) == f"project:{ident.alpha}"


def test_evidence_scope_keeps_a_home_directory_session_global(ident, conn, db_url):
    sessions = _sessions(conn, ident, ["C:\\Users\\someone"])
    assert _evidence_scope_of(db_url, sessions) == "global"


# ---------------------------------------------------------------------------
# POST /ingest through the real route: the stored label is never a path
# ---------------------------------------------------------------------------

#: The machine token the harness authorizes with: a test constant, not a credential.
_ROOT = "root-test-token"


def _ingest_client(db_url):
    from fastmcp import FastMCP
    from starlette.testclient import TestClient

    from mcp_server import caller_trust as ct
    from mcp_server.http_auth import bearer
    from mcp_server.ingest_route import register

    m = FastMCP("test-project-identity-ingest")
    register(
        m,
        lambda: db_url,
        lambda r: True,
        lambda r: ct.request_trust(db_url=db_url, machine_token=_ROOT, bearer=bearer(r)),
    )
    return TestClient(m.http_app())


def _cc_records(n: _Names, cwd: str) -> tuple[str, list[dict]]:
    """One Claude Code turn (user + assistant) run in ``cwd``, with unique ids/content."""
    sid = n.sid()
    tag = uuid.uuid4().hex
    base = {"sessionId": sid, "cwd": cwd, "timestamp": "2026-09-01T12:00:00Z"}
    return sid, [
        {**base, "type": "user", "uuid": f"u-{tag}", "message": {"role": "user", "content": f"q {tag}"}},
        {
            **base,
            "type": "assistant",
            "uuid": f"a-{tag}",
            "message": {"role": "assistant", "content": [{"type": "text", "text": f"a {tag}"}]},
        },
    ]  # fmt: skip


def _codex_records(n: _Names, cwd: str) -> tuple[str, list[dict]]:
    """A Codex rollout tail: session_meta (with the cwd) plus one user/assistant turn."""
    sid = n.sid()
    tag = uuid.uuid4().hex

    def item(role, kind, text, n):
        return {
            "type": "response_item",
            "timestamp": "2026-09-01T12:00:00Z",
            "payload": {
                "type": "message",
                "id": f"item-{n}-{tag}",
                "role": role,
                "content": [{"type": kind, "text": text}],
            },
        }

    return sid, [
        {"type": "session_meta", "payload": {"id": sid, "cwd": cwd}},
        item("user", "input_text", f"q {tag}", 1),
        item("assistant", "output_text", f"a {tag}", 2),
    ]


def _post(db_url, body: dict):
    return _ingest_client(db_url).post(
        "/ingest", json=body, headers={"Authorization": f"Bearer {_ROOT}"}
    )


def _stored_projects(conn, sid: str) -> list[str | None]:
    rows = conn.execute(
        "SELECT DISTINCT project FROM episodes WHERE session_id = %s", (sid,)
    ).fetchall()
    return [r[0] for r in rows]


_WIN_ALPHA = "C:\\Users\\pat\\Alpha"
_WIN_OTHER = "C:\\Users\\pat\\Other"

_MISSING = object()


@pytest.mark.parametrize(
    "project,cwd,expected",
    [
        # The body's label wins over the cwd, and a path in it is cut to its last part.
        ("C:\\Users\\pat\\Alpha\\", _WIN_OTHER, "Alpha"),
        ("/home/pat/Alpha/", _WIN_OTHER, "Alpha"),
        ("Alpha", _WIN_OTHER, "Alpha"),
        ("Al\x07pha\n", _WIN_OTHER, "Alpha"),  # control characters stripped
        # No usable label in the body: the parser labels from the cwd, either separator.
        (_MISSING, _WIN_ALPHA, "Alpha"),
        (_MISSING, _WIN_ALPHA + "\\", "Alpha"),
        (_MISSING, "c:\\users\\pat\\Alpha", "Alpha"),
        (_MISSING, "/home/pat/Alpha/", "Alpha"),
        (None, _WIN_ALPHA, "Alpha"),
        (42, _WIN_ALPHA, "Alpha"),
        (["Other"], _WIN_ALPHA, "Alpha"),
        ({"name": "Other"}, _WIN_ALPHA, "Alpha"),
        ("", _WIN_ALPHA, "Alpha"),
        ("C:", _WIN_ALPHA, "Alpha"),
        ("..", _WIN_ALPHA, "Alpha"),
        ("\x00", _WIN_ALPHA, "Alpha"),
    ],
    ids=[
        "body-windows-path",
        "body-posix-path",
        "body-label",
        "body-control-chars",
        "no-project-windows-cwd",
        "no-project-windows-cwd-trailing-sep",
        "no-project-lowercase-drive",
        "no-project-posix-cwd",
        "null",
        "int",
        "list",
        "dict",
        "empty",
        "bare-drive",
        "dotdot",
        "only-control-chars",
    ],
)
def test_ingest_stores_a_label_never_a_path(ident, conn, db_url, project, cwd, expected):
    sid, records = _cc_records(ident, cwd)
    body: dict = {"records": records}
    if project is not _MISSING:
        body["project"] = project

    r = _post(db_url, body)

    assert r.status_code == 200, r.text
    assert r.json()["ingested"] == 1
    assert _stored_projects(conn, sid) == [expected]


@pytest.mark.parametrize("folder", ["axon-test", "scratch", "tmp"])
def test_a_scratch_folder_inside_a_repository_is_still_dropped(ident, conn, db_url, folder):
    """The client sends the repository's label for a session run in its scratch folder;
    the folder still marks the turn as a smoke test (ingestion.scratch_projects)."""
    sid, records = _cc_records(ident, f"/work/{ident.alpha}/{folder}")
    out = _post(db_url, {"records": records, "project": ident.alpha}).json()
    assert out["ingested"] == 0, out
    assert _stored_projects(conn, sid) == []


def test_a_replay_across_the_label_change_is_not_stored_twice(ident, conn, db_url):
    """A session stored by an older plugin under its subfolder's label, replayed under a
    new session id by a client that now sends the repo's label: the cross-session replay
    guard (schema 036) still recognises it by its folder."""
    cwd = f"/work/{ident.alpha}/tickets"
    sid, records = _cc_records(ident, cwd)
    assert _post(db_url, {"records": records}).json()["ingested"] == 1
    assert _stored_projects(conn, sid) == ["tickets"]

    replay_sid = ident.sid()
    replay = [{**r, "sessionId": replay_sid, "uuid": f"replay-{r['uuid']}"} for r in records]
    out = _post(db_url, {"records": replay, "project": ident.alpha}).json()
    assert out["ingested"] == 0, out
    assert _stored_projects(conn, replay_sid) == []


@pytest.mark.parametrize(
    "project,cwd",
    [
        (_MISSING, _WIN_ALPHA),
        (42, _WIN_ALPHA),
        # The client resolved a subfolder's session to its repo's toplevel.
        ("Alpha", _WIN_ALPHA + "\\src\\pkg"),
    ],
    ids=["no-project", "non-string-project", "client-resolved-toplevel"],
)
def test_codex_ingest_stores_a_label_never_a_path(ident, conn, db_url, project, cwd):
    sid, records = _codex_records(ident, cwd)
    body: dict = {"records": records, "format": "codex", "session_id": sid}
    if project is not _MISSING:
        body["project"] = project

    r = _post(db_url, body)

    assert r.status_code == 200, r.text
    assert r.json()["ingested"] == 1
    assert _stored_projects(conn, sid) == ["Alpha"]


def test_a_batch_spanning_two_projects_labels_each_turn_from_its_own_cwd(ident, conn, db_url):
    """A session that moved between repositories: the client sends each cwd's label
    (cwd_projects) and the route labels every turn from its own cwd, path-cleaned."""
    sid_a, recs_a = _cc_records(ident, "/work/Alpha/sub")
    _sid_b, recs_b = _cc_records(ident, "/work/Other/deep")
    for r in recs_b:
        r["sessionId"] = sid_a
        r["timestamp"] = "2026-09-01T12:05:00Z"
    body = {
        "records": recs_a + recs_b,
        "source": "hook",
        "cwd_projects": {"/work/Alpha/sub": "Alpha", "/work/Other/deep": "C:\\x\\Other", "x": 3},
    }
    assert _post(db_url, body).status_code == 200
    rows = conn.execute(
        "SELECT metadata->>'cwd', project FROM episodes WHERE session_id = %s ORDER BY sequence",
        (sid_a,),
    ).fetchall()
    assert rows == [("/work/Alpha/sub", "Alpha"), ("/work/Other/deep", "Other")]
