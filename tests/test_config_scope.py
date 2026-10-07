"""/config/* is scoped by the caller's bearer, not by a ``surface`` field in the body.

The config lane mirrors each machine's CLAUDE.md / rules files and serves dream's proposed rule
edits. Before this, any client token could name any ``surface`` in the body and read (or
overwrite) another machine's mirrored config, and every proposal mined from every conversation
went to every caller. Now:

  * full-trust device: unchanged, may name any surface;
  * restricted device: its own surface only, and only proposals whose evidence comes entirely
    from sessions it ingested (schema 057 provenance), and nothing before 057 is applied;
  * root token / no token / no resolver: nothing (empty reads, 403 on writes).
"""

from __future__ import annotations

import json
import os
import uuid

import psycopg
import pytest

from mcp_server import caller_trust as ct

_DB_URL = os.environ.get(
    "SYNAPSE_TEST_URL", "postgresql://synapse:synapse@127.0.0.1:5432/synapse_test"
)

try:
    psycopg.connect(_DB_URL, connect_timeout=2).close()
except Exception:  # pragma: no cover - environment dependent
    pytest.skip("no test DB reachable", allow_module_level=True)

_ROOT = "root-test-token"


def _client(db_url, *, wired: bool = True):
    from fastmcp import FastMCP
    from starlette.testclient import TestClient

    from mcp_server.config_sync_routes import register
    from mcp_server.http_auth import bearer

    def trust(r):
        return ct.request_trust(db_url=db_url, machine_token=_ROOT, bearer=bearer(r))

    m = FastMCP("test-config-scope")
    register(m, db_url, lambda r: True, trust if wired else None)
    return TestClient(m.http_app())


def _h(tok: str | None) -> dict:
    return {"Authorization": f"Bearer {tok}"} if tok else {}


@pytest.fixture()
def env(conn, db_url):
    """Three devices, one mirrored file each, and proposals with every provenance shape."""
    from tests.helpers.surfaces import clear_surfaces, register_device

    def _wipe():
        conn.execute("DELETE FROM config_lane.config_registry")
        conn.execute("DELETE FROM config_lane.config_proposals")
        conn.execute("TRUNCATE episodes RESTART IDENTITY CASCADE")
        clear_surfaces(conn)

    _wipe()
    toks = {k: f"{k}-token-{uuid.uuid4().hex}" for k in ("work", "home", "full")}
    sids = {
        "work": register_device(conn, toks["work"], trust="restricted", projects=[]),
        "home": register_device(conn, toks["home"], trust="restricted", projects=[]),
        "full": register_device(conn, toks["full"], trust="full"),
    }
    for k, sid in sids.items():
        conn.execute(
            "INSERT INTO config_lane.config_registry "
            "(surface_id, file_key, abs_path, content, content_hash) VALUES (%s, %s, %s, %s, %s)",
            (sid, "CLAUDE.md", f"/x/{k}/CLAUDE.md", f"{k} rules", f"h-{k}"),
        )
    # One stamped session per restricted device (schema 057 provenance), plus a session
    # that holds a work turn AND an unstamped one (a later root-token upload, say).
    for k in ("work", "home"):
        conn.execute(
            "INSERT INTO episodes (session_id, sequence, content, surface_id) VALUES (%s, 1, %s, %s)",
            (f"sess-{k}", f"{k} turn", sids[k]),
        )
    conn.execute(
        "INSERT INTO episodes (session_id, sequence, content, surface_id) VALUES "
        "('sess-shared', 1, 'work turn', %s), ('sess-shared', 2, 'unstamped turn', NULL)",
        (sids["work"],),
    )

    def proposal(summary, evidence, surface_id=None):
        return conn.execute(
            "INSERT INTO config_lane.config_proposals "
            "(kind, file_key, scope, surface_id, summary, evidence, status) "
            "VALUES ('add', 'CLAUDE.md', %s, %s, %s, %s::jsonb, 'proposed') RETURNING id",
            ("local" if surface_id else "general", surface_id, summary, json.dumps(evidence)),
        ).fetchone()[0]

    props = {
        "work_evidence": proposal("from work sessions", [{"session_id": "sess-work"}]),
        # Aimed at the work surface, but nothing proves its CONTENT came from there.
        "work_local": proposal("targets work", [], surface_id=sids["work"]),
        "shared_session": proposal("from a mixed session", [{"session_id": "sess-shared"}]),
        "unknown_session": proposal("from an unknown session", [{"session_id": "sess-gone"}]),
        "home_evidence": proposal("from home sessions", [{"session_id": "sess-home"}]),
        "mixed": proposal(
            "mixed sessions", [{"session_id": "sess-work"}, {"session_id": "sess-home"}]
        ),
        "no_provenance": proposal("no session ids", [{"why": "x"}]),
        "empty": proposal("no evidence", []),
    }
    yield {"tok": toks, "sid": sids, "prop": props}
    _wipe()


def _list(c, tok, surface=None):
    body = {"surface": surface} if surface else {}
    r = c.post("/config/list", json=body, headers=_h(tok))
    assert r.status_code == 200
    return r.json()["files"]


def _fetch(c, tok, surface=None):
    body = {"file_key": "CLAUDE.md", **({"surface": surface} if surface else {})}
    r = c.post("/config/fetch", json=body, headers=_h(tok))
    assert r.status_code == 200
    return r.json()


def _proposal_ids(c, tok):
    r = c.post("/config/proposals", json={}, headers=_h(tok))
    assert r.status_code == 200
    return sorted(p["id"] for p in r.json()["proposals"])


# --------------------------------------------------------------------------- mirror reads


def test_restricted_reads_only_its_own_mirror_and_body_surface_is_ignored(env, db_url):
    c = _client(db_url)
    work = env["tok"]["work"]
    # No surface in the body: the bearer decides.
    assert _fetch(c, work)["content"] == "work rules"
    assert [f["content_hash"] for f in _list(c, work)] == ["h-work"]
    # Naming another surface is ignored, not honoured.
    assert _fetch(c, work, surface=env["sid"]["home"])["content"] == "work rules"
    assert [f["content_hash"] for f in _list(c, work, surface=env["sid"]["full"])] == ["h-work"]


def test_full_trust_may_still_name_any_surface(env, db_url):
    c = _client(db_url)
    full = env["tok"]["full"]
    assert _fetch(c, full, surface=env["sid"]["home"])["content"] == "home rules"
    assert [f["content_hash"] for f in _list(c, full, surface=env["sid"]["work"])] == ["h-work"]


@pytest.mark.parametrize("tok", [_ROOT, None], ids=["root", "no-token"])
def test_unknown_callers_read_nothing(env, db_url, tok):
    c = _client(db_url)
    assert _list(c, tok, surface=env["sid"]["home"]) == []
    assert _fetch(c, tok, surface=env["sid"]["home"]) == {"found": False}
    assert _proposal_ids(c, tok) == []


def test_unwired_routes_fail_closed(env, db_url):
    c = _client(db_url, wired=False)
    assert _list(c, env["tok"]["full"], surface=env["sid"]["home"]) == []
    assert _proposal_ids(c, env["tok"]["full"]) == []


# --------------------------------------------------------------------------- mirror writes


def test_restricted_publish_lands_on_its_own_surface_only(env, conn, db_url):
    c = _client(db_url)
    r = c.post(
        "/config/publish",
        json={
            "surface": env["sid"]["home"],  # spoof attempt
            "file_key": "CLAUDE.md",
            "abs_path": "/x/CLAUDE.md",
            "content": "overwritten",
        },
        headers=_h(env["tok"]["work"]),
    )
    assert r.status_code == 200, r.text
    rows = dict(
        conn.execute(
            "SELECT surface_id, content FROM config_lane.config_registry WHERE file_key='CLAUDE.md'"
        ).fetchall()
    )
    assert rows[env["sid"]["home"]] == "home rules"  # untouched
    assert rows[env["sid"]["work"]] == "overwritten"


@pytest.mark.parametrize("tok", [_ROOT, None], ids=["root", "no-token"])
def test_unknown_callers_cannot_publish(env, conn, db_url, tok):
    r = _client(db_url).post(
        "/config/publish",
        json={"surface": env["sid"]["home"], "file_key": "CLAUDE.md", "content": "x"},
        headers=_h(tok),
    )
    assert r.status_code == 403
    content = conn.execute(
        "SELECT content FROM config_lane.config_registry WHERE surface_id = %s",
        (env["sid"]["home"],),
    ).fetchone()[0]
    assert content == "home rules"


def test_full_trust_publish_unchanged(env, conn, db_url):
    r = _client(db_url).post(
        "/config/publish",
        json={"surface": "some-host", "file_key": "CLAUDE.md", "content": "c"},
        headers=_h(env["tok"]["full"]),
    )
    assert r.status_code == 200
    assert conn.execute(
        "SELECT 1 FROM config_lane.config_registry WHERE surface_id = 'some-host'"
    ).fetchone()


# --------------------------------------------------------------------------- proposals


def test_restricted_sees_only_its_own_proposals(env, db_url):
    c = _client(db_url)
    p = env["prop"]
    assert _proposal_ids(c, env["tok"]["work"]) == [p["work_evidence"]]
    assert _proposal_ids(c, env["tok"]["home"]) == [p["home_evidence"]]
    assert _proposal_ids(c, env["tok"]["full"]) == sorted(p.values())


def test_restricted_detail_hides_other_proposals(env, db_url):
    c = _client(db_url)
    work = env["tok"]["work"]
    mine = c.post("/config/proposals", json={"id": env["prop"]["work_evidence"]}, headers=_h(work))
    assert mine.json()["found"] is True
    for key in ("home_evidence", "mixed", "no_provenance", "work_local", "shared_session"):
        r = c.post("/config/proposals", json={"id": env["prop"][key]}, headers=_h(work))
        assert r.json() == {"found": False}, key


def test_restricted_can_act_only_on_its_own_proposals(env, conn, db_url):
    c = _client(db_url)
    work = env["tok"]["work"]
    ok = c.post(
        "/config/proposals/act",
        json={"id": env["prop"]["work_evidence"], "action": "reject"},
        headers=_h(work),
    )
    assert ok.status_code == 200 and ok.json()["status"] == "rejected"
    for key in ("home_evidence", "mixed", "work_local"):
        r = c.post(
            "/config/proposals/act",
            json={"id": env["prop"][key], "action": "accept"},
            headers=_h(work),
        )
        assert r.status_code == 403, key
    status = conn.execute(
        "SELECT status FROM config_lane.config_proposals WHERE id = %s",
        (env["prop"]["home_evidence"],),
    ).fetchone()[0]
    assert status == "proposed"


def test_unknown_and_full_act_paths(env, db_url):
    c = _client(db_url)
    r = c.post(
        "/config/proposals/act",
        json={"id": env["prop"]["mixed"], "action": "reject"},
        headers=_h(_ROOT),
    )
    assert r.status_code == 403
    r = c.post(
        "/config/proposals/act",
        json={"id": env["prop"]["mixed"], "action": "reject"},
        headers=_h(env["tok"]["full"]),
    )
    assert r.status_code == 200 and r.json()["status"] == "rejected"


# --------------------------------------------------------------------------- deploy order


def test_before_057_a_restricted_device_owns_no_proposals(env, conn, db_url):
    """Without the provenance column nothing can be proven a device's own, so a restricted
    caller sees no proposals (and gets 403 acting on one) rather than an error. Full trust
    and the mirror lane do not depend on the column at all."""
    from tests.helpers.pre057 import pre057_schema

    with pre057_schema(conn, db_url) as (url, _schema):
        c = _client(url)
        work = env["tok"]["work"]
        assert _proposal_ids(c, work) == []
        r = c.post("/config/proposals", json={"id": env["prop"]["work_evidence"]}, headers=_h(work))
        assert r.json() == {"found": False}
        r = c.post(
            "/config/proposals/act",
            json={"id": env["prop"]["work_evidence"], "action": "reject"},
            headers=_h(work),
        )
        assert r.status_code == 403
        assert _fetch(c, work)["content"] == "work rules"
        assert _proposal_ids(c, env["tok"]["full"]) == sorted(env["prop"].values())
