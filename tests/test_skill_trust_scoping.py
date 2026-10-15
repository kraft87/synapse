"""Skills are scoped by the caller's trust, the board's rule (schema 053/054) applied to skills.

Before this, /skills/list was gated by the machine token alone and served every active
global skill to every approved device, restricted ones included: the leak schema 053 closed
for notes. Now (mcp_server/skill_sync_routes.py ``SkillAccess``):

  * full-trust device: every scope, read and write, and the admin routes (retire, rescope,
    the proposal review lane);
  * restricted device: only ``project:<p>`` skills whose project is in its allowlist
    (case-insensitively), never ``global``; may write only there; no admin route;
  * the root token, no token, or routes registered without a trust resolver: nothing.

A skill a device is no longer served (moved to another scope, or not its to read) comes
back as "withdrawn" for the names it says it holds, with the hashes of every version
Synapse delivered, and the client engine (plugin/scripts/skills_sync.py) deletes exactly
those copies through the same path as a retirement. PgSkillsProvider (skill:// MCP
resources) applies the same filter.

Everything runs through the real HTTP app with real device tokens resolved from the
surfaces table, and the end-to-end tests drive the real client engine against it. DB-backed
(tagged into the `db` xdist group via conftest). Every skill, candidate and surface this
file writes carries the ``tsc-`` prefix and is removed around each test. No real tokens.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType

import pytest
from psycopg.types.json import Json

from ingestion.surfaces import (
    FULL_TRUST,
    UNKNOWN_SURFACE,
    SurfaceTrust,
    resolve_caller,
    token_hash,
)
from mcp_server import caller_trust as ct
from mcp_server.skill_sync_routes import (
    FULL_ACCESS,
    SkillAccess,
    _publish_skill,
    safe_file_path,
    safe_skill_name,
    whole_skill_hash,
)
from mcp_server.skills_provider import PgSkillsProvider

pytestmark = pytest.mark.db

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "plugin" / "scripts"

_P = "tsc-"  # every skill / candidate name this file writes
_ROOT = "root-test-token"  # the server's shared machine token in these tests
_OLD = "2026-01-01T00:00:00+00:00"
_REF = [("ref/notes.md", b"reference notes\n")]


def _body(name: str, step: str) -> str:
    return f"---\nname: {name}\ndescription: A demo skill.\n---\n# {name}\n\n1. {step}\n"


GLOB = _body("tsc-glob", "Run the global step.")
GLOB_V2 = _body("tsc-glob", "Run the global step, then check it.")
GLOB2 = _body("tsc-glob2", "Run another global step.")
ACME = _body("tsc-acme", "Run the project step.")
OTHER = _body("tsc-other", "Run the other project's step.")
LOCAL = _body("tsc-glob", "Run it my way.")
MINE = _body("tsc-mine", "A skill only this machine has.")

#: name -> (registry scope, body, files) as the fixture seeds it. The project skill is
#: stored as "project:acme" while the restricted device's allowlist says "Acme": the
#: allowlist and the scope match case-insensitively.
_SEED = {
    "tsc-glob": ("global", GLOB, _REF),
    "tsc-glob2": ("global", GLOB2, ()),
    "tsc-acme": ("project:acme", ACME, _REF),
    "tsc-other": ("project:Other", OTHER, _REF),
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _file(path: str, content: bytes) -> dict:
    return {
        "path": path,
        "content_b64": base64.b64encode(content).decode(),
        "sha256": _sha(content),
        "size": len(content),
        "is_executable": False,
    }


def _payload(name: str, body: str, scope, when: str | None = _OLD, files=()) -> dict:
    p = {
        "name": name,
        "scope": scope,
        "body": body,
        "description": "",
        "files": [_file(rel, content) for rel, content in files],
    }
    if when is not None:
        p["content_modified_at"] = when
    return p


def _hash(body: str, files=()) -> str:
    return whole_skill_hash(body, [(rel, _sha(content)) for rel, content in files])


def _later(seconds: float = 3600) -> str:
    """A timestamp after anything this test wrote (retired_at is the DB's now())."""
    return (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat()


def _h(tok: str | None) -> dict:
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _mine(entries) -> list[dict]:
    return sorted((e for e in entries or [] if e["name"].startswith(_P)), key=lambda e: e["name"])


def _names(listing: dict, key: str = "skills") -> list[str]:
    return [e["name"] for e in _mine(listing.get(key))]


def _row(conn, name: str):
    return conn.execute(
        "SELECT scope, status, body FROM skills_lane.skill_registry WHERE name=%s", (name,)
    ).fetchone()


def _history(conn, name: str) -> list[tuple]:
    return conn.execute(
        "SELECT scope, body, op FROM skills_lane.skill_history WHERE name=%s ORDER BY id",
        (name,),
    ).fetchall()


def _app(db_url: str, *, wired: bool = True):
    """The real /skills/* routes, trust resolved from the bearer exactly as the server
    does (``request_trust``: a device token by its hash; the root token and no token are
    unknown). ``wired=False`` registers them without a resolver, the wiring slip that must
    fail closed."""
    from fastmcp import FastMCP
    from starlette.testclient import TestClient

    from mcp_server.http_auth import bearer
    from mcp_server.skill_sync_routes import register

    def trust(request):
        return ct.request_trust(db_url=db_url, machine_token=_ROOT, bearer=bearer(request))

    m = FastMCP("test-skill-trust-scoping")
    register(m, db_url, lambda request: True, trust if wired else None)
    return TestClient(m.http_app())


#: Callers that resolve to no device: the shared root token, no bearer at all, and a
#: full-trust device's token on routes registered without a resolver.
_UNKNOWN = ("root", "no-token", "unwired")


@pytest.fixture()
def env(conn, db_url):
    """Two devices (restricted to ["Acme"], full), four seeded skills, a clean slate."""
    from tests.helpers.surfaces import register_device

    def _clean():
        conn.execute("DELETE FROM skills_lane.skill_gap_candidates WHERE name LIKE 'tsc-%%'")
        conn.execute("DELETE FROM skills_lane.skill_registry WHERE name LIKE 'tsc-%%'")
        conn.execute("DELETE FROM skills_lane.skill_history WHERE name LIKE 'tsc-%%'")
        conn.execute("DELETE FROM surfaces WHERE surface_id LIKE 'dev-tsc-%%'")

    _clean()
    tag = uuid.uuid4().hex[:10]
    tok = {k: f"tsc-{k}-device-{uuid.uuid4().hex}" for k in ("restricted", "full")}
    sid = {
        "restricted": register_device(
            conn,
            tok["restricted"],
            trust="restricted",
            projects=["Acme"],
            surface_id=f"dev-tsc-r-{tag}",
        ),
        "full": register_device(conn, tok["full"], trust="full", surface_id=f"dev-tsc-f-{tag}"),
    }
    for name, (scope, body, files) in _SEED.items():
        r = _publish_skill(db_url, _payload(name, body, scope, files=files), FULL_ACCESS)
        assert r == {"status": "ok", "name": name}
    yield {"tok": tok, "sid": sid}
    _clean()


@pytest.fixture()
def http(env, db_url):
    """``http(who, path, body)`` -> the response, for who in restricted / full / root /
    no-token / unwired."""
    wired, unwired = _app(db_url), _app(db_url, wired=False)

    def call(who: str, path: str, body):
        if who == "unwired":
            return unwired.post(path, json=body, headers=_h(env["tok"]["full"]))
        tok = {
            "restricted": env["tok"]["restricted"],
            "full": env["tok"]["full"],
            "root": _ROOT,
            "no-token": None,
        }[who]
        return wired.post(path, json=body, headers=_h(tok))

    return call


def _ok(http, who: str, path: str, body) -> dict:
    r = http(who, path, body)
    assert r.status_code == 200, r.text
    return r.json()


def _list(http, who: str, scope: str, **extra) -> dict:
    return _ok(http, who, "/skills/list", {"scope": scope, **extra})


# ---------------------------------------------------------------------------
# SkillAccess
# ---------------------------------------------------------------------------


def test_full_trust_serves_every_scope_but_only_string_scopes():
    a = SkillAccess.of(FULL_TRUST)
    assert a == FULL_ACCESS and a.full and a.known
    for scope in ("global", "project:Acme", "project:anything", "project:x y"):
        assert a.serves(scope)
    for scope in (None, 7, b"global", ["global"], {"scope": "global"}):
        assert not a.serves(scope)


def test_restricted_serves_only_its_allowlisted_projects_in_any_case():
    st = SurfaceTrust(
        surface_id="dev-a",
        trust="restricted",
        allowed_projects=("Acme", "beta-app"),
        known=True,
        credential_bound=True,
    )
    a = SkillAccess.of(st)
    assert not a.full and a.known and a.projects == frozenset({"acme", "beta-app"})
    for scope in (
        "project:Acme",
        "project:acme",
        "project:ACME",
        "PROJECT:acme",
        "project:Beta-App",
    ):
        assert a.serves(scope), scope
    for scope in (
        "global",
        "GLOBAL",
        "project:Other",
        "project:",
        "project:acme/sub",
        "project:acmex",
        "project:Acme ",
        "acme",
        "Acme",
        None,
        7,
        b"project:acme",
    ):
        assert not a.serves(scope), scope


@pytest.mark.parametrize(
    ("st", "known"),
    [
        (UNKNOWN_SURFACE, False),
        (SurfaceTrust(surface_id="some-host"), False),  # a token that matched no row
        (SurfaceTrust(trust="full"), False),  # "full" that no surfaces row confirmed
        (SurfaceTrust(trust="restricted", known=True), True),  # a device, empty allowlist
    ],
    ids=["unknown", "unmatched", "unconfirmed-full", "empty-allowlist"],
)
def test_an_unknown_or_empty_verdict_serves_nothing(st, known):
    a = SkillAccess.of(st)
    assert not a.full and a.projects == frozenset()
    assert a.known is known  # only a surfaces row is answered about the names it holds
    for scope in ("global", "project:Acme", "project:", "project:anything"):
        assert not a.serves(scope)


# ---------------------------------------------------------------------------
# /skills/list
# ---------------------------------------------------------------------------

_LISTINGS = [
    ("restricted", "global", [], False),
    ("restricted", "project:Acme", ["tsc-acme"], True),
    ("restricted", "project:acme", ["tsc-acme"], True),
    ("restricted", "project:ACME", ["tsc-acme"], True),
    ("restricted", "project:Other", [], False),
    ("full", "global", ["tsc-glob", "tsc-glob2"], True),
    ("full", "project:Acme", ["tsc-acme"], True),
    ("full", "project:Other", ["tsc-other"], True),
    *(
        (who, scope, [], False)
        for who in _UNKNOWN
        for scope in ("global", "project:Acme", "project:Other")
    ),
]


@pytest.mark.parametrize(("who", "scope", "names", "publish"), _LISTINGS)
def test_list_serves_by_trust_and_says_whether_the_caller_may_publish(
    http, who, scope, names, publish
):
    listing = _list(http, who, scope)
    assert _names(listing) == names
    assert listing["publish"] is publish
    if not publish:
        assert listing["skills"] == []  # nothing at all, not just nothing of ours
    assert "withdrawn" not in listing  # only answered when the client says what it holds


def test_a_listed_project_skill_keeps_the_registry_spelling_and_carries_its_files(http):
    (entry,) = _mine(_list(http, "restricted", "project:Acme")["skills"])
    assert entry["scope"] == "project:acme"
    assert entry["body"] == ACME
    assert entry["files"] == [{"path": "ref/notes.md", "sha256": _sha(_REF[0][1])}]


# ---------------------------------------------------------------------------
# /skills/fetch
# ---------------------------------------------------------------------------

_FETCHES = {
    "restricted": {"tsc-acme"},
    "full": {"tsc-glob", "tsc-glob2", "tsc-acme", "tsc-other"},
    **{who: set() for who in _UNKNOWN},
}


@pytest.mark.parametrize("who", list(_FETCHES))
def test_fetch_finds_only_served_skills(http, who):
    for name, (scope, body, files) in _SEED.items():
        got = _ok(http, who, "/skills/fetch", {"name": name})
        if name in _FETCHES[who]:
            assert got["found"] is True and got["scope"] == scope and got["body"] == body
            assert [f["path"] for f in got["files"]] == [rel for rel, _ in files]
        else:
            # Exactly what a missing skill returns: no body, no scope, no hint.
            assert got == {"found": False}, (who, name)
    assert _ok(http, who, "/skills/fetch", {"name": "tsc-nonexistent"}) == {"found": False}


# ---------------------------------------------------------------------------
# /skills/publish
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("who", "scope"),
    [
        ("restricted", "global"),
        ("restricted", "project:Other"),
        *((who, scope) for who in _UNKNOWN for scope in ("global", "project:Acme")),
    ],
)
def test_publishing_to_an_unserved_scope_is_forbidden_and_writes_nothing(http, conn, who, scope):
    r = http(who, "/skills/publish", _payload("tsc-new", MINE, scope, when=_later()))
    # 200, not 403: an older client treats an HTTP error as the end of its whole sync.
    assert r.status_code == 200
    assert r.json() == {"status": "forbidden", "name": "tsc-new"}
    assert _row(conn, "tsc-new") is None


@pytest.mark.parametrize("who", ["restricted", "full"])
@pytest.mark.parametrize("scope", ["project:a/b", "Project:Acme", "project:", "", 5, None])
def test_an_invalid_scope_is_an_error(http, conn, who, scope):
    got = _ok(http, who, "/skills/publish", _payload("tsc-new", MINE, scope))
    assert got["status"] == "error"
    assert _row(conn, "tsc-new") is None


@pytest.mark.parametrize(
    ("who", "name", "scope", "revealed"),
    [
        # A restricted device is told the name lives elsewhere, never where.
        ("restricted", "tsc-glob", "project:Acme", None),
        ("restricted", "tsc-other", "project:Acme", None),
        # Full trust is told where it lives.
        ("full", "tsc-acme", "global", "project:acme"),
        ("full", "tsc-acme", "project:Other", "project:acme"),
        ("full", "tsc-glob", "project:Acme", "global"),
    ],
)
def test_a_push_into_another_scopes_active_skill_is_refused_and_moves_nothing(
    http, conn, who, name, scope, revealed
):
    before = _row(conn, name)
    got = _ok(http, who, "/skills/publish", _payload(name, LOCAL, scope, when=_later()))
    assert got == {"status": "scoped", "name": name, "scope": revealed}
    assert _row(conn, name) == before
    assert _history(conn, name) == []


def test_a_push_to_the_same_scope_in_another_case_keeps_the_registry_spelling(http, conn):
    edited = ACME + "2. And a new step.\n"
    got = _ok(
        http,
        "restricted",
        "/skills/publish",
        _payload("tsc-acme", edited, "project:ACME", when=_later(), files=[("x.txt", b"x")]),
    )
    assert got == {"status": "ok", "name": "tsc-acme"}
    assert _row(conn, "tsc-acme") == ("project:acme", "active", edited)
    files = conn.execute(
        "SELECT path FROM skills_lane.skill_files WHERE skill_name='tsc-acme'"
    ).fetchall()
    assert files == [("x.txt",)]


def test_a_restricted_device_publishes_a_new_skill_into_its_allowlisted_project(http, conn):
    got = _ok(http, "restricted", "/skills/publish", _payload("tsc-new", MINE, "project:Acme"))
    assert got == {"status": "ok", "name": "tsc-new"}
    assert _row(conn, "tsc-new") == ("project:Acme", "active", MINE)
    # Every spelling of the project serves it, to this device and to full trust.
    assert _names(_list(http, "restricted", "project:acme")) == ["tsc-acme", "tsc-new"]
    assert _names(_list(http, "full", "project:Acme")) == ["tsc-acme", "tsc-new"]
    assert _names(_list(http, "full", "global")) == ["tsc-glob", "tsc-glob2"]
    assert _ok(http, "restricted", "/skills/fetch", {"name": "tsc-new"})["body"] == MINE


def test_a_restricted_device_cannot_take_over_a_retired_skill_of_a_scope_it_is_not_served(
    http, conn
):
    for name in ("tsc-glob", "tsc-other", "tsc-acme"):
        assert _ok(http, "full", "/skills/retire", {"name": name})["status"] == "retired"

    # A newer edit of different content would reactivate it, but not from this device.
    for name in ("tsc-glob", "tsc-other"):
        before = _row(conn, name)
        got = _ok(
            http,
            "restricted",
            "/skills/publish",
            _payload(name, LOCAL, "project:Acme", when=_later()),
        )
        assert got == {"status": "forbidden", "name": name}
        assert _row(conn, name) == before and before[1] == "retired"

    # Its own project's retired skill: newest edit wins, as for any device.
    edited = ACME + "2. Revived.\n"
    got = _ok(
        http,
        "restricted",
        "/skills/publish",
        _payload("tsc-acme", edited, "project:Acme", _later()),
    )
    assert got == {"status": "ok", "name": "tsc-acme"}
    scope, status, body = _row(conn, "tsc-acme")
    assert (scope.lower(), status, body) == ("project:acme", "active", edited)


# ---------------------------------------------------------------------------
# /skills/overwrite
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("who", "scope", "allowed"),
    [
        ("restricted", "global", False),
        ("restricted", None, False),  # no scope reads as global
        ("restricted", "project:Other", False),
        ("restricted", "project:Acme", True),
        ("full", "global", True),
        ("full", "project:Other", True),
        *((who, "project:Acme", False) for who in _UNKNOWN),
        *((who, "global", False) for who in _UNKNOWN),
    ],
)
def test_overwrite_records_history_only_in_a_served_scope(http, conn, who, scope, allowed):
    name = "tsc-acme" if scope == "project:Acme" else "tsc-glob"
    body = {"name": name, "body": LOCAL, "content_modified_at": _OLD}
    if scope is not None:
        body["scope"] = scope
    got = _ok(http, who, "/skills/overwrite", body)
    if allowed:
        assert got == {"status": "ok"}
        assert _history(conn, name) == [(scope, LOCAL, "disk_overwrite")]
    else:
        assert got == {"status": "forbidden"}
        assert _history(conn, name) == []


def test_overwrite_checks_the_skills_stored_scope_not_the_claimed_one(http, conn):
    """A restricted device can't write history under a global skill's name by claiming
    one of its own project scopes for it."""
    body = {"name": "tsc-glob", "scope": "project:Acme", "body": LOCAL, "content_modified_at": _OLD}
    assert _ok(http, "restricted", "/skills/overwrite", body) == {"status": "forbidden"}
    assert _history(conn, "tsc-glob") == []


# ---------------------------------------------------------------------------
# admin routes: retire, rescope, proposals, proposals/act
# ---------------------------------------------------------------------------


def _candidate(conn) -> int:
    return conn.execute(
        "INSERT INTO skills_lane.skill_gap_candidates "
        "(kind, name, direction, target_skills, status, summary, evidence, proposal_body, "
        " judge_sessions, judge_weight) "
        "VALUES ('retune', 'tsc-glob', 'widen', %s, 'proposed', 'synthetic', %s, %s, 1, 1.0) "
        "RETURNING id",
        (["tsc-glob"], Json([{"session_id": "tsc-s1", "class": "judge"}]), GLOB_V2),
    ).fetchone()[0]


def _admin_calls(cid: int) -> list[tuple[str, dict]]:
    return [
        ("/skills/retire", {"name": "tsc-glob2"}),
        ("/skills/rescope", {"name": "tsc-glob", "scope": "project:Acme"}),
        ("/skills/proposals", {}),
        ("/skills/proposals", {"id": cid}),
        ("/skills/proposals/act", {"id": cid, "action": "reject", "reason": "tsc"}),
    ]


@pytest.mark.parametrize("who", ["restricted", *_UNKNOWN])
def test_admin_routes_are_forbidden_below_full_trust_and_change_nothing(http, conn, who):
    cid = _candidate(conn)
    for path, body in _admin_calls(cid):
        r = http(who, path, body)
        assert r.status_code == 403, (path, r.text)
        assert r.json() == {"status": "error", "detail": "forbidden for this surface"}
    assert _row(conn, "tsc-glob2") == ("global", "active", GLOB2)
    assert _row(conn, "tsc-glob") == ("global", "active", GLOB)
    assert _history(conn, "tsc-glob") == [] and _history(conn, "tsc-glob2") == []
    status = conn.execute(
        "SELECT status FROM skills_lane.skill_gap_candidates WHERE id=%s", (cid,)
    ).fetchone()
    assert status == ("proposed",)


def test_admin_routes_work_for_full_trust(http, conn):
    cid = _candidate(conn)
    (retire, rescope, listing, detail, act) = (
        _ok(http, "full", path, body) for path, body in _admin_calls(cid)
    )
    assert (retire["status"], retire["name"]) == ("retired", "tsc-glob2")
    assert rescope == {
        "status": "rescoped",
        "name": "tsc-glob",
        "scope": "project:Acme",
        "from": "global",
    }
    assert cid in {p["id"] for p in listing["proposals"]}
    assert detail["found"] is True and detail["id"] == cid
    assert act == {"status": "rejected", "id": cid, "reason": "tsc"}
    assert _row(conn, "tsc-glob2")[1] == "retired"
    assert _row(conn, "tsc-glob")[0] == "project:Acme"


# ---------------------------------------------------------------------------
# retired entries
# ---------------------------------------------------------------------------


@pytest.fixture()
def retired(http, db_url):
    """One retired skill per scope: global, the device's project, another project."""
    seeds = {
        "tsc-ret-glob": ("global", _body("tsc-ret-glob", "Old global."), _REF),
        "tsc-ret-acme": ("project:acme", _body("tsc-ret-acme", "Old project."), ()),
        "tsc-ret-other": ("project:Other", _body("tsc-ret-other", "Old other."), _REF),
    }
    out = {}
    for name, (scope, body, files) in seeds.items():
        assert _publish_skill(db_url, _payload(name, body, scope, files=files))["status"] == "ok"
        r = _ok(http, "full", "/skills/retire", {"name": name})
        h = _hash(body, files)
        out[name] = {"scope": scope, "content_hash": h, "retired_at": r["retired_at"]}
    return out


def _retired_view(listing: dict) -> dict:
    return {e["name"]: e for e in _mine(listing.get("retired"))}


def test_full_trust_is_told_every_retirement_with_its_scope(http, retired):
    for scope in ("global", "project:Acme", "project:Other"):
        got = _retired_view(_list(http, "full", scope))
        assert got == {
            name: {
                "name": name,
                "scope": r["scope"],
                "content_hash": r["content_hash"],
                "content_hashes": [r["content_hash"]],
                "retired_at": r["retired_at"],
            }
            for name, r in retired.items()
        }


def test_a_restricted_device_is_told_only_retirements_it_was_served_or_holds(http, retired):
    # Its own project's, with the scope shown because it is the scope asked about.
    got = _retired_view(_list(http, "restricted", "project:Acme"))
    assert list(got) == ["tsc-ret-acme"]
    assert got["tsc-ret-acme"]["scope"] == "project:acme"
    assert got["tsc-ret-acme"]["content_hash"] == retired["tsc-ret-acme"]["content_hash"]
    assert got["tsc-ret-acme"]["retired_at"] == retired["tsc-ret-acme"]["retired_at"]

    # A held name is answered too, with "" for a scope that is not the one asked about.
    got = _retired_view(
        _list(http, "restricted", "project:ACME", held=["tsc-ret-glob", "tsc-unknown"])
    )
    assert list(got) == ["tsc-ret-acme", "tsc-ret-glob"]
    assert got["tsc-ret-acme"]["scope"] == "project:acme"
    assert got["tsc-ret-glob"]["scope"] == ""
    assert got["tsc-ret-glob"]["content_hashes"] == [retired["tsc-ret-glob"]["content_hash"]]

    # Its global folder: a scope is shown only when this device is served it, so the held
    # global retirement shows "" (only Synapse's own versions vouch for a copy there) and
    # its own project's shows its scope. The other project's never appears.
    got = _retired_view(_list(http, "restricted", "global", held=["tsc-ret-glob"]))
    assert {n: e["scope"] for n, e in got.items()} == {
        "tsc-ret-acme": "project:acme",
        "tsc-ret-glob": "",
    }


@pytest.mark.parametrize("who", _UNKNOWN)
def test_an_unknown_caller_is_told_nothing_even_about_the_names_it_holds(http, retired, who):
    """Answering held names would let the shared root token probe which skills exist."""
    held = ["tsc-ret-glob", "tsc-ret-acme", "tsc-ret-other", "tsc-glob", "tsc-acme"]
    for scope in ("global", "project:Acme"):
        for extra in ({}, {"held": held}):
            listing = _list(http, who, scope, **extra)
            assert listing["skills"] == [] and listing["publish"] is False
            assert _retired_view(listing) == {}
            assert "withdrawn" not in listing


# ---------------------------------------------------------------------------
# withdrawn entries
# ---------------------------------------------------------------------------


def _withdrawn(listing: dict) -> dict:
    return {e["name"]: e for e in _mine(listing.get("withdrawn"))}


def test_a_rescoped_skill_held_in_the_folder_it_left_is_withdrawn(http):
    assert (
        _ok(http, "full", "/skills/rescope", {"name": "tsc-glob", "scope": "project:Acme"})[
            "status"
        ]
        == "rescoped"
    )
    h = _hash(GLOB, _REF)

    # Full trust holding it in its global folder: told where it went.
    listing = _list(http, "full", "global", held=["tsc-glob", "tsc-glob2", "tsc-unknown"])
    assert _names(listing) == ["tsc-glob2"]
    assert listing["withdrawn"] == [
        {"name": "tsc-glob", "scope": "project:Acme", "content_hash": h, "content_hashes": [h]}
    ]
    # A restricted device holding it in its global folder is told where it went, because
    # the new scope is one it reads (a scope it may not read would show as "").
    got = _withdrawn(_list(http, "restricted", "global", held=["tsc-glob"]))
    assert got == {
        "tsc-glob": {
            "name": "tsc-glob",
            "scope": "project:Acme",
            "content_hash": h,
            "content_hashes": [h],
        }
    }
    # ...and the new scope's folder lists it, so nothing is withdrawn there.
    listing = _list(http, "restricted", "project:Acme", held=["tsc-glob", "tsc-acme"])
    assert _names(listing) == ["tsc-acme", "tsc-glob"] and _withdrawn(listing) == {}


def test_a_skill_moved_out_of_a_restricted_devices_project_is_withdrawn_without_its_scope(http):
    _ok(http, "full", "/skills/rescope", {"name": "tsc-acme", "scope": "project:Other"})
    h = _hash(ACME, _REF)

    got = _withdrawn(_list(http, "restricted", "project:Acme", held=["tsc-acme"]))
    assert got["tsc-acme"]["scope"] == ""  # never names a project it may not read
    assert got["tsc-acme"]["content_hash"] == h and "retired_at" not in got["tsc-acme"]
    got = _withdrawn(_list(http, "full", "project:acme", held=["tsc-acme"]))
    assert got["tsc-acme"]["scope"] == "project:Other"


def test_a_global_skill_held_by_a_restricted_device_is_withdrawn_by_trust(http):
    listing = _list(
        http, "restricted", "global", held=["tsc-glob", "tsc-glob2", "tsc-mine", "tsc-glob"]
    )
    assert listing["skills"] == [] and listing["publish"] is False
    got = _withdrawn(listing)
    # Held and active: answered. tsc-mine (never a Synapse skill) and tsc-acme / tsc-other
    # (not held) are not.
    assert list(got) == ["tsc-glob", "tsc-glob2"]
    assert got["tsc-glob"] == {
        "name": "tsc-glob",
        "scope": "",  # not a scope this device is served: only Synapse's versions vouch
        "content_hash": _hash(GLOB, _REF),
        "content_hashes": [_hash(GLOB, _REF)],
    }
    assert got["tsc-glob2"]["content_hash"] == _hash(GLOB2)


def test_withdrawn_answers_only_a_list_of_held_names(http):
    # Served and listed: not withdrawn.
    assert _withdrawn(_list(http, "full", "global", held=["tsc-glob"])) == {}
    assert _withdrawn(_list(http, "restricted", "project:Acme", held=["tsc-acme"])) == {}
    # Not a list: no "withdrawn" key at all, as from an older client.
    for held in ("tsc-glob", {"tsc-glob": 1}, 7, None):
        assert "withdrawn" not in _list(http, "restricted", "global", held=held)
    # Non-string entries are ignored; the string ones still answer.
    got = _withdrawn(_list(http, "restricted", "global", held=[7, None, "", ["x"], "tsc-glob"]))
    assert list(got) == ["tsc-glob"]
    # A retired held name is reported as retired, never as withdrawn.
    _ok(http, "full", "/skills/retire", {"name": "tsc-glob2"})
    listing = _list(http, "restricted", "global", held=["tsc-glob2"])
    assert _withdrawn(listing) == {} and _names(listing, "retired") == ["tsc-glob2"]


def test_withdrawn_hashes_include_a_superseded_version_but_never_a_disk_overwrite(http, conn):
    """A copy pulled before the latest edit is still one Synapse delivered; a local edit a
    pull clobbered (recorded by /skills/overwrite) never was."""
    got = _ok(
        http, "full", "/skills/publish", _payload("tsc-glob", GLOB_V2, "global", _later(), _REF)
    )
    assert got["status"] == "ok"
    _ok(
        http,
        "full",
        "/skills/overwrite",
        {"name": "tsc-glob", "scope": "global", "body": LOCAL, "content_modified_at": _OLD},
    )
    assert [op for _s, _b, op in _history(conn, "tsc-glob")] == ["superseded", "disk_overwrite"]

    entry = _withdrawn(_list(http, "restricted", "global", held=["tsc-glob"]))["tsc-glob"]
    assert entry["content_hash"] == _hash(GLOB_V2, _REF)
    assert entry["content_hashes"] == [_hash(GLOB_V2, _REF), _hash(GLOB, _REF)]
    assert _hash(LOCAL, _REF) not in entry["content_hashes"]


# ---------------------------------------------------------------------------
# /skills/rescope
# ---------------------------------------------------------------------------


def test_rescope_moves_the_skill_and_records_the_move(http, conn):
    got = _ok(http, "full", "/skills/rescope", {"name": "tsc-glob", "scope": "project:Acme"})
    assert got == {
        "status": "rescoped",
        "name": "tsc-glob",
        "scope": "project:Acme",
        "from": "global",
    }
    assert _row(conn, "tsc-glob") == ("project:Acme", "active", GLOB)
    # One row: the scope it left and the body it carried. A scope change is not a body
    # change, so the supersede trigger adds nothing.
    assert _history(conn, "tsc-glob") == [("global", GLOB, "rescoped")]
    assert _names(_list(http, "full", "global")) == ["tsc-glob2"]
    assert _names(_list(http, "restricted", "project:acme")) == ["tsc-acme", "tsc-glob"]

    again = _ok(http, "full", "/skills/rescope", {"name": "tsc-glob", "scope": "project:Acme"})
    assert again == {"status": "unchanged", "name": "tsc-glob", "scope": "project:Acme"}
    assert len(_history(conn, "tsc-glob")) == 1


def test_rescope_refuses_what_it_cannot_move(http, conn):
    got = _ok(http, "full", "/skills/rescope", {"name": "tsc-none", "scope": "global"})
    assert got == {"status": "not_found", "name": "tsc-none"}

    _ok(http, "full", "/skills/retire", {"name": "tsc-glob2"})
    got = _ok(http, "full", "/skills/rescope", {"name": "tsc-glob2", "scope": "project:Acme"})
    assert got["status"] == "error" and "retired" in got["detail"]
    assert _row(conn, "tsc-glob2")[:2] == ("global", "retired")

    for scope in ("project:a/b", "Global", "", "project:", None, 5):
        got = _ok(http, "full", "/skills/rescope", {"name": "tsc-glob", "scope": scope})
        assert got["status"] == "error", scope
    for name in ("", "   ", None, 5):
        got = _ok(http, "full", "/skills/rescope", {"name": name, "scope": "global"})
        assert got["status"] == "error", name
    assert _ok(http, "full", "/skills/rescope", ["tsc-glob"])["status"] == "error"
    assert _row(conn, "tsc-glob") == ("global", "active", GLOB)
    assert _history(conn, "tsc-glob") == []


# ---------------------------------------------------------------------------
# PgSkillsProvider (skill:// MCP resources)
# ---------------------------------------------------------------------------


def _device_trust(db_url: str, tok: str):
    """The verdict the server's ``_caller_trust`` resolves for this device's bearer."""
    return lambda: resolve_caller(db_url, token_hash_hex=token_hash(tok))


async def _uris(p: PgSkillsProvider) -> set[str]:
    return {str(r.uri) for r in await p._list_resources() if str(r.uri).startswith("skill://tsc-")}


async def _templates(p: PgSkillsProvider) -> set[str]:
    return {t.name for t in await p._list_resource_templates() if t.name.startswith(_P)}


async def test_provider_serves_a_restricted_device_only_its_projects_skills(env, db_url):
    p = PgSkillsProvider(db_url, _device_trust(db_url, env["tok"]["restricted"]))
    assert await _uris(p) == {"skill://tsc-acme/SKILL.md", "skill://tsc-acme/_manifest"}
    assert await _templates(p) == {"tsc-acme_files"}

    for name in ("tsc-glob", "tsc-glob2", "tsc-other"):
        assert await p._get_resource(f"skill://{name}/SKILL.md") is None
        assert await p._get_resource(f"skill://{name}/_manifest") is None
        assert await p._get_resource_template(f"skill://{name}/ref/notes.md") is None

    res = await p._get_resource("skill://tsc-acme/SKILL.md")
    assert res is not None and await res.read() == ACME
    man = await p._get_resource("skill://tsc-acme/_manifest")
    assert man is not None
    assert {f["path"] for f in json.loads(await man.read())["files"]} == {
        "SKILL.md",
        "ref/notes.md",
    }
    tmpl = await p._get_resource_template("skill://tsc-acme/ref/notes.md")
    assert tmpl is not None and await tmpl.read({"path": "ref/notes.md"}) == "reference notes\n"


async def test_provider_serves_full_trust_every_skill(env, db_url):
    p = PgSkillsProvider(db_url, _device_trust(db_url, env["tok"]["full"]))
    assert await _uris(p) == {
        f"skill://{name}/{leaf}" for name in _SEED for leaf in ("SKILL.md", "_manifest")
    }
    assert await _templates(p) == {f"{name}_files" for name in _SEED}
    res = await p._get_resource("skill://tsc-glob/SKILL.md")
    assert res is not None and await res.read() == GLOB
    assert await p._get_resource_template("skill://tsc-other/ref/notes.md") is not None


@pytest.mark.parametrize("trust", [None, lambda: UNKNOWN_SURFACE], ids=["unwired", "unknown"])
async def test_provider_serves_no_one_else_anything(env, db_url, trust):
    p = PgSkillsProvider(db_url, trust)
    assert await _uris(p) == set() and await _templates(p) == set()
    for name in _SEED:
        assert await p._get_resource(f"skill://{name}/SKILL.md") is None
        assert await p._get_resource(f"skill://{name}/_manifest") is None
        assert await p._get_resource_template(f"skill://{name}/ref/notes.md") is None


# ---------------------------------------------------------------------------
# end to end: the real client engine against the real routes
# ---------------------------------------------------------------------------

_CLIENT_MODULES = (
    "config",
    "synapse_home",
    "synapse_project",
    "enroll",
    "synapse_login",
    "harness",
    "synapse_cli",
    "common",
)
_CLIENT_ENV = (
    "SYNAPSE_URL",
    "SYNAPSE_INGEST_URL",
    "SYNAPSE_INGEST_TOKEN",
    "SYNAPSE_MCP_URL",
    "SYNAPSE_SKILLS_SYNC",
    "SYNAPSE_SURFACE",
    "CLAUDE_PROJECT_DIR",
)


@pytest.fixture()
def engine(monkeypatch, tmp_path) -> ModuleType:
    """plugin/scripts/skills_sync.py loaded by path, with a scratch HOME / Synapse home
    (its delivered ledger lives there), no Synapse env, and no cached client modules."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("SYNAPSE_HOME", str(tmp_path / "synapse-home"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("SYNAPSE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CLAUDE_SKILLS_DIR", str(tmp_path / "skills"))
    for var in _CLIENT_ENV:
        monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv(f"CLAUDE_PLUGIN_OPTION_{var}", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", [*sys.path])  # the engine prepends its own directory
    for name in _CLIENT_MODULES:
        sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(
        "tsc_skills_sync_engine", _SCRIPTS / "skills_sync.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    yield mod
    for name in _CLIENT_MODULES:
        sys.modules.pop(name, None)


class _Device:
    """The engine's ``client``: config.post_json's contract (raise on an HTTP error, else
    the parsed reply) over the real routes, with this device's bearer."""

    def __init__(self, app, token: str):
        self.app, self.token = app, token
        self.calls: list[tuple[str, dict]] = []

    def post_json(self, path, payload, timeout=30.0):
        self.calls.append((path, payload))
        r = self.app.post(path, json=payload, headers=_h(self.token))
        r.raise_for_status()
        return r.json()

    def paths(self) -> list[str]:
        return [p for p, _ in self.calls]


def _write_skill(root: Path, name: str, body: str, files=(), *, mtime: float | None = None):
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(body, encoding="utf-8", newline="\n")
    for rel, content in files:
        (folder / rel).parent.mkdir(parents=True, exist_ok=True)
        (folder / rel).write_bytes(content)
    if mtime is not None:
        for fp in folder.rglob("*"):
            if fp.is_file():
                os.utime(fp, (mtime, mtime))
    return folder


def _git(*args: str, cwd: Path) -> str:
    r = subprocess.run(
        [
            "git",
            "-c",
            "user.email=t@example.com",
            "-c",
            "user.name=t",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return r.stdout


def test_e2e_a_restricted_device_drops_delivered_global_copies_and_keeps_its_own(
    env, db_url, conn, engine, tmp_path
):
    """The leak, closed on a machine that already has it: a restricted device whose
    global folder holds the global skills it was served before."""
    dev = _Device(_app(db_url), env["tok"]["restricted"])
    folder = tmp_path / "claude-skills"
    # Exact delivered copies, with a fresh mtime (a copy renews an mtime, not the content).
    now = time.time()
    _write_skill(folder, "tsc-glob", GLOB, _REF, mtime=now)
    _write_skill(folder, "tsc-glob2", GLOB2, mtime=now)
    _write_skill(folder, "tsc-mine", MINE, mtime=now)

    assert engine._sync("global", folder, client=dev) == (0, 0, 2)

    assert sorted(p.name for p in folder.iterdir()) == ["tsc-mine"]
    assert dev.calls == [
        ("/skills/list", {"scope": "global", "held": ["tsc-glob", "tsc-glob2", "tsc-mine"]})
    ]  # nothing published: not its own skill (publish=false), not the withdrawn copies
    assert _row(conn, "tsc-mine") is None
    assert _row(conn, "tsc-glob") == ("global", "active", GLOB)
    assert _row(conn, "tsc-glob2") == ("global", "active", GLOB2)
    assert engine._sync("global", folder, client=dev) == (0, 0, 0)
    assert (folder / "tsc-mine" / "SKILL.md").read_text(encoding="utf-8") == MINE

    # Its own project's skills still reach it, in any spelling of the label.
    proj = tmp_path / "acme" / ".claude" / "skills"
    assert engine._sync("project:Acme", proj, client=dev) == (1, 0, 0)
    assert (proj / "tsc-acme" / "SKILL.md").read_text(encoding="utf-8") == ACME
    assert (proj / "tsc-acme" / "ref" / "notes.md").read_bytes() == _REF[0][1]
    # Another project's folder: nothing served, nothing pulled, no empty folder made.
    other = tmp_path / "other" / ".claude" / "skills"
    assert engine._sync("project:Other", other, client=dev) == (0, 0, 0)
    assert not other.exists()
    assert "/skills/publish" not in dev.paths()


def test_e2e_a_full_device_moves_a_rescoped_skill_from_global_into_its_project(
    env, db_url, conn, engine, tmp_path
):
    dev = _Device(_app(db_url), env["tok"]["full"])
    glob_dir = tmp_path / "claude-skills"
    engine._sync("global", glob_dir, client=dev)
    assert (glob_dir / "tsc-glob" / "SKILL.md").read_text(encoding="utf-8") == GLOB
    assert (glob_dir / "tsc-glob2" / "SKILL.md").read_text(encoding="utf-8") == GLOB2

    for name in ("tsc-glob", "tsc-glob2"):
        r = dev.post_json("/skills/rescope", {"name": name, "scope": "project:Acme"})
        assert r["status"] == "rescoped"
    # A local edit of one of them: not a copy Synapse delivered, so it is left alone.
    edited = GLOB2 + "2. A local step.\n"
    (glob_dir / "tsc-glob2" / "SKILL.md").write_text(edited, encoding="utf-8", newline="\n")
    dev.calls.clear()

    assert engine._sync("global", glob_dir, client=dev) == (0, 0, 1)

    assert not (glob_dir / "tsc-glob").exists()
    assert (glob_dir / "tsc-glob2" / "SKILL.md").read_text(encoding="utf-8") == edited
    assert "/skills/publish" not in dev.paths()  # never pushed back into global
    assert _row(conn, "tsc-glob") == ("project:Acme", "active", GLOB)
    assert _row(conn, "tsc-glob2") == ("project:Acme", "active", GLOB2)

    # The project pass, in a real repository: pulls it (and the project's other skills),
    # and keeps the pulled folders out of the repository through info/exclude.
    repo = tmp_path / "acme-repo"
    repo.mkdir()
    _git("init", "-q", cwd=repo)
    (repo / "README.md").write_text("demo\n", encoding="utf-8")
    _git("add", "README.md", cwd=repo)
    _git("commit", "-q", "-m", "init", cwd=repo)
    proj = repo / ".claude" / "skills"

    assert engine._sync("project:Acme", proj, client=dev, git_exclude=True) == (3, 0, 0)

    assert (proj / "tsc-glob" / "SKILL.md").read_text(encoding="utf-8") == GLOB
    assert (proj / "tsc-glob" / "ref" / "notes.md").read_bytes() == _REF[0][1]
    assert sorted(p.name for p in proj.iterdir()) == ["tsc-acme", "tsc-glob", "tsc-glob2"]
    exclude = (repo / ".git" / "info" / "exclude").read_text(encoding="utf-8").splitlines()
    for name in ("tsc-acme", "tsc-glob", "tsc-glob2"):
        assert f"/.claude/skills/{name}/" in exclude
    assert _git("status", "--porcelain", "--untracked-files=all", cwd=repo) == ""
    assert _row(conn, "tsc-glob") == ("project:Acme", "active", GLOB)

    # Settled: the next session start moves nothing either way.
    assert engine._sync("global", glob_dir, client=dev) == (0, 0, 0)
    assert engine._sync("project:Acme", proj, client=dev, git_exclude=True) == (0, 0, 0)
    assert not (glob_dir / "tsc-glob").exists()
    assert "/skills/publish" not in dev.paths()


# ---------------------------------------------------------------------------
# Names and file paths stay inside a skill's folder; one publish per name at a time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["..", ".", "tsc/../x", "tsc\\x", "C:", "tsc:x", " tsc", "tsc\x00", ""]
)
def test_a_publish_with_a_name_that_is_not_one_folder_is_refused(http, conn, name):
    """A skill name is a folder on every machine the skill reaches: '..' would make a
    pull write (and mirror-delete) outside the skills folder."""
    got = _ok(http, "full", "/skills/publish", _payload(name, GLOB, "global"))
    assert got["status"] == "error"
    if "\x00" not in name:  # Postgres text can't hold NUL: it never got near the table
        n = conn.execute("SELECT count(*) FROM skills_lane.skill_registry WHERE name=%s", (name,))
        assert n.fetchone()[0] == 0


@pytest.mark.parametrize(
    "path", ["../escape.md", "a/../../b", "/abs.md", "a\\b.md", "C:/x.md", "a//b", "./a", "a/ b"]
)
def test_a_publish_with_a_file_path_outside_the_folder_is_refused(http, conn, path):
    got = _ok(
        http,
        "restricted",
        "/skills/publish",
        _payload("tsc-paths", GLOB, "project:Acme", files=[(path, b"x")]),
    )
    assert got["status"] == "error"
    n = conn.execute("SELECT count(*) FROM skills_lane.skill_registry WHERE name='tsc-paths'")
    assert n.fetchone()[0] == 0


def test_safe_names_and_paths_accept_ordinary_skills():
    for name in ("react-best-practices", "the-humanizer", "docx", "My Skill", "a.b_c"):
        assert safe_skill_name(name), name
    for path in ("scripts/run.py", "ref/notes.md", "deep/er/x.txt", "x"):
        assert safe_file_path(path), path
