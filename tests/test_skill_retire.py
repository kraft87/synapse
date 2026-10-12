"""Retiring a skill — DB-backed tests (migration 060 + skill_sync_routes).

A retirement is an edit with a time: /skills/retire stamps retired_at, /skills/list reports
it beside the active skills, and a publish whose edit is not newer is refused, so a machine
that still holds the skill can't bring it back at its next session start. A newer edit
does bring it back (newest edit wins). Runs against the shared Postgres test DB (tagged
into the `db` xdist group via conftest). Every skill is synthetic (ret- prefix) and cleaned
up around each test.
"""

from __future__ import annotations

import base64
import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from psycopg.types.json import Json
from starlette.testclient import TestClient

import mcp_server.skill_sync_routes as R
from dream.skills.skill_doc import body_hash
from ingestion import schema_check
from mcp_server.skill_sync_routes import (
    _list_skills,
    _proposal_act,
    _proposal_detail,
    _publish_skill,
    _retire_skill,
    whole_skill_hash,
)

_ROOT = Path(__file__).resolve().parent.parent
_OLD = "2026-01-01T00:00:00+00:00"
_TOKEN = "test-machine-token"
_H = {"Authorization": f"Bearer {_TOKEN}"}


def _skill(name: str, step: str) -> str:
    return f"---\nname: {name}\ndescription: Back up the demo db.\n---\n# {name}\n\n1. {step}\n"


V1 = _skill("ret-backup", "Dump the database.")
V2 = _skill("ret-backup", "Dump the database, then verify it.")
SCRIPT = b"#!/bin/sh\necho dump\n"


@pytest.fixture()
def ret(conn):
    def _clean():
        conn.execute("DELETE FROM skills_lane.skill_gap_candidates WHERE name LIKE 'ret-%%'")
        conn.execute("DELETE FROM skills_lane.skill_registry WHERE name LIKE 'ret-%%'")
        conn.execute("DELETE FROM skills_lane.skill_history WHERE name LIKE 'ret-%%'")

    _clean()
    yield conn
    _clean()


def _file(path: str, content: bytes) -> dict:
    return {
        "path": path,
        "content_b64": base64.b64encode(content).decode(),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _payload(name="ret-backup", body=V1, scope="global", when=_OLD, files=()) -> dict:
    p = {"name": name, "scope": scope, "body": body, "description": "", "files": list(files)}
    if when is not None:
        p["content_modified_at"] = when
    return p


def _publish(db_url, **kw) -> dict:
    return _publish_skill(db_url, _payload(**kw))


def _row(conn, name="ret-backup"):
    return conn.execute(
        "SELECT status, retired_at, body FROM skills_lane.skill_registry WHERE name=%s", (name,)
    ).fetchone()


def _retired(listing: dict) -> list[dict]:
    return [r for r in listing["retired"] if r["name"].startswith("ret-")]


def _active_names(listing: dict) -> set[str]:
    return {s["name"] for s in listing["skills"] if s["name"].startswith("ret-")}


# ------------------------------------------------------------------ retire


def test_retire_stamps_the_row_and_records_history(ret, db_url):
    _publish(db_url)

    r = _retire_skill(db_url, "ret-backup")

    assert (r["status"], r["name"], r["scope"]) == ("retired", "ret-backup", "global")
    status, retired_at, body = _row(ret)
    assert status == "retired" and body == V1
    assert r["retired_at"] == retired_at.isoformat()
    hist = ret.execute(
        "SELECT body, content_modified_at, op FROM skills_lane.skill_history "
        "WHERE name='ret-backup'"
    ).fetchall()
    assert hist == [(V1, datetime.fromisoformat(_OLD), "retired")]


def test_retire_again_unknown_and_invalid_names(ret, db_url):
    _publish(db_url)
    first = _retire_skill(db_url, "ret-backup")

    again = _retire_skill(db_url, "ret-backup")
    assert again == {
        "status": "already_retired",
        "name": "ret-backup",
        "retired_at": first["retired_at"],
    }
    n = ret.execute("SELECT count(*) FROM skills_lane.skill_history WHERE name='ret-backup'")
    assert n.fetchone()[0] == 1  # no second history row
    assert _retire_skill(db_url, "ret-nope") == {"status": "not_found", "name": "ret-nope"}
    for bad in (None, "", "   ", 7, ["ret-backup"]):
        assert _retire_skill(db_url, bad)["status"] == "error"


def test_a_hand_retired_row_is_not_enforced_until_retire_stamps_it(ret, db_url):
    _publish(db_url)
    ret.execute("UPDATE skills_lane.skill_registry SET status='retired' WHERE name='ret-backup'")
    assert _retired(_list_skills(db_url, "global")) == []  # no time to compare against

    r = _retire_skill(db_url, "ret-backup")

    assert r["status"] == "retired" and _row(ret)[1] is not None
    assert [e["name"] for e in _retired(_list_skills(db_url, "global"))] == ["ret-backup"]


# ------------------------------------------------------------------ list


def test_list_reports_retired_skills_apart_from_the_active_ones(ret, db_url):
    _publish(db_url, files=[_file("scripts/dump.sh", SCRIPT), _file("a.txt", b"a")])
    _publish(db_url, name="ret-live", body=_skill("ret-live", "Run."), files=[_file("x", b"x")])
    _publish(db_url, name="ret-proj", body=_skill("ret-proj", "Run."), scope="project:ret-p")
    r = _retire_skill(db_url, "ret-backup")
    _retire_skill(db_url, "ret-proj")

    listing = _list_skills(db_url, "global")

    assert _active_names(listing) == {"ret-live"}  # an older client never sees a retirement
    live = next(s for s in listing["skills"] if s["name"] == "ret-live")
    assert live["files"] == [{"path": "x", "sha256": hashlib.sha256(b"x").hexdigest()}]
    shas = [  # the hash sorts by path itself, like the client's
        ("scripts/dump.sh", hashlib.sha256(SCRIPT).hexdigest()),
        ("a.txt", hashlib.sha256(b"a").hexdigest()),
    ]
    retired = _retired(listing)
    assert retired[0] == {
        "name": "ret-backup",
        "scope": "global",
        "content_hash": whole_skill_hash(V1, shas),
        "retired_at": r["retired_at"],
    }
    # A name is one registry row, its scope whichever folder published it last: a copy in
    # another scope's folder is just as retired, so every listing carries every retirement.
    assert [e["name"] for e in retired] == ["ret-backup", "ret-proj"]
    assert retired[1]["scope"] == "project:ret-p"
    assert _retired(_list_skills(db_url, "project:ret-p")) == retired


# ------------------------------------------------------------------ publish guard


def _shifted(iso: str, seconds: float, *, naive: bool = False) -> str:
    dt = datetime.fromisoformat(iso).astimezone(UTC) + timedelta(seconds=seconds)
    return (dt.replace(tzinfo=None) if naive else dt).isoformat()


@pytest.mark.parametrize(
    "when",
    [
        pytest.param(lambda at: _OLD, id="older"),
        pytest.param(lambda at: at, id="equal"),
        pytest.param(lambda at: None, id="missing"),
        pytest.param(lambda at: "not a timestamp", id="unparseable"),
        pytest.param(lambda at: _shifted(at, -60, naive=True), id="naive-older"),
    ],
)
def test_a_publish_not_newer_than_the_retirement_is_refused(ret, db_url, when):
    _publish(db_url, files=[_file("a.txt", b"a")])
    at = _retire_skill(db_url, "ret-backup")["retired_at"]

    r = _publish(db_url, body=V2, when=when(at), files=[])

    assert r == {"status": "retired", "name": "ret-backup", "retired_at": at}
    status, retired_at, body = _row(ret)
    assert (status, retired_at.isoformat(), body) == ("retired", at, V1)  # nothing written
    files = ret.execute(
        "SELECT path FROM skills_lane.skill_files WHERE skill_name='ret-backup'"
    ).fetchall()
    assert files == [("a.txt",)]


def test_the_retired_content_itself_never_reactivates_whatever_its_time(ret, db_url):
    """A clone, a restore, a copy or a fast clock renews an mtime, not the skill; and an
    older client pushes any copy it holds."""
    files = [_file("a.txt", b"a")]
    _publish(db_url, files=files)
    at = _retire_skill(db_url, "ret-backup")["retired_at"]

    r = _publish(db_url, when=_shifted(at, 3600), files=files)

    assert r == {"status": "retired", "name": "ret-backup", "retired_at": at}
    assert _row(ret)[0] == "retired"
    # Any change at all is an edit: newest edit wins.
    r = _publish(db_url, when=_shifted(at, 3600), files=[_file("a.txt", b"b")])
    assert r == {"status": "ok", "name": "ret-backup"}


@pytest.mark.parametrize("naive", [False, True], ids=["aware", "naive"])
def test_a_newer_edit_reactivates_the_skill(ret, db_url, naive):
    _publish(db_url)
    at = _retire_skill(db_url, "ret-backup")["retired_at"]

    r = _publish(db_url, body=V2, when=_shifted(at, 3600, naive=naive))

    assert r == {"status": "ok", "name": "ret-backup"}
    assert _row(ret) == ("active", None, V2)
    listing = _list_skills(db_url, "global")
    assert _active_names(listing) == {"ret-backup"} and _retired(listing) == []


# ------------------------------------------------------------------ review accept


def _candidate(conn, body: str, base: str) -> int:
    return conn.execute(
        "INSERT INTO skills_lane.skill_gap_candidates "
        "(kind, name, direction, target_skills, status, summary, evidence, proposal_body, "
        " base_body_hash, judge_sessions, judge_weight) "
        "VALUES ('retune','ret-backup','fix',%s,'proposed','synthetic',%s,%s,%s,1,1.0) "
        "RETURNING id",
        (["ret-backup"], Json([{"session_id": "ret-s1", "class": "judge"}]), body, base),
    ).fetchone()[0]


def _accept(db_url, cid: int, **kw) -> dict:
    return _proposal_act(db_url, cid, "accept", None, lambda *_: "routing-eval: stub", **kw)


def test_accepting_onto_a_retired_skill_needs_force(ret, db_url):
    _publish(db_url)
    _retire_skill(db_url, "ret-backup")
    cid = _candidate(ret, V2, body_hash(V1))

    issues = _proposal_detail(db_url, cid)["apply_issues"]
    assert [(i["code"], i["force"]) for i in issues] == [("retired", True)]

    r = _accept(db_url, cid)
    assert r["status"] == "refused" and r["reason"] == "retired"
    assert _row(ret)[0] == "retired"

    r = _accept(db_url, cid, force=True)
    assert r["status"] == "promoted" and r["forced"] == ["retired"]
    assert _row(ret) == ("active", None, V2)  # a forced accept is the newest edit


# ------------------------------------------------------------------ HTTP


def _client(db_url) -> TestClient:
    from fastmcp import FastMCP

    def authorized(request):
        return request.headers.get("authorization", "") == f"Bearer {_TOKEN}"

    test_mcp = FastMCP("test-skill-retire")
    R.register(test_mcp, db_url, authorized)
    return TestClient(test_mcp.http_app())


def test_routes_require_the_machine_token_and_retire_over_http(ret, db_url):
    _publish(db_url)
    with _client(db_url) as client:
        r = client.post("/skills/retire", json={"name": "ret-backup"})
        assert r.status_code == 401
        assert r.json() == {"status": "error", "detail": "unauthorized"}
        assert _row(ret)[0] == "active"

        r = client.post("/skills/retire", json={"name": "ret-backup"}, headers=_H)
        assert r.status_code == 200 and r.json()["status"] == "retired"
        at = r.json()["retired_at"]
        again = client.post("/skills/retire", json={"name": "ret-backup"}, headers=_H)
        assert again.json()["status"] == "already_retired"
        for bad in ({}, {"name": ""}, ["ret-backup"]):
            r = client.post("/skills/retire", json=bad, headers=_H)
            assert r.status_code == 200 and r.json()["status"] == "error"

        listing = client.post("/skills/list", json={"scope": "global"}, headers=_H).json()
        assert [e["retired_at"] for e in _retired(listing)] == [at]
        r = client.post("/skills/publish", json=_payload(body=V2), headers=_H)
        assert r.json() == {"status": "retired", "name": "ret-backup", "retired_at": at}


# ------------------------------------------------------------------ schema 060


def test_schema_060_is_registered_and_optional():
    script = (_ROOT / "scripts" / "apply_schema.sh").read_text()
    assert "060_skill_retired_at.sql" in script
    # The image lands before 060 is applied by hand; the boot guard must let it start.
    assert schema_check.is_optional_migration(_ROOT / "schema" / "060_skill_retired_at.sql")


def test_without_schema_060_everything_behaves_as_before(ret, db_url, monkeypatch):
    _publish(db_url)
    _retire_skill(db_url, "ret-backup")
    monkeypatch.setattr(R.RETIRED_AT_PROBE, "ready", lambda *a, **k: False)

    assert _retire_skill(db_url, "ret-backup") == {
        "status": "error",
        "detail": "schema 060 (skill_registry.retired_at) not applied",
    }
    assert "retired" not in _list_skills(db_url, "global")
    assert _publish(db_url, body=V2) == {"status": "ok", "name": "ret-backup"}
    assert _row(ret)[0] == "active"  # pre-060: any publish reactivates
