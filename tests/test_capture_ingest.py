"""Web capture through the episode writer and POST /ingest, against the scratch DB.

The writer captures the research of the turns it stored (and of turns it already had),
after their commit, and a capture that fails costs the capture only.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from ingestion import capture
from ingestion.db import Database
from ingestion.db_episodes import DELETED_TURNS_PROBE
from ingestion.episode_writer import store_episodes
from ingestion.jsonl_client import JSONLParser

PAGE = "# A Real Article\n\n" + "The measured latency was low and the design held up well. " * 12


@pytest.fixture(scope="module")
def db(db_url):
    d = Database(db_url)
    yield d
    d.close()


@pytest.fixture(autouse=True)
def _clean(clean_tables, conn):
    conn.execute(
        "TRUNCATE web_artifacts, research_archive, private_sessions RESTART IDENTITY CASCADE"
    )
    DELETED_TURNS_PROBE.reset()
    yield
    conn.execute(
        "TRUNCATE web_artifacts, research_archive, private_sessions RESTART IDENTITY CASCADE"
    )
    DELETED_TURNS_PROBE.reset()


@pytest.fixture()
def capture_on(monkeypatch):
    monkeypatch.setenv("SYNAPSE_WEB_CAPTURE", "on")


def records(
    tag: str,
    session: str,
    *,
    url: str = "https://example.com/post",
    text: str = PAGE,
    name: str = "mcp__acme__read",
    tool_input: dict | None = None,
) -> list[dict]:
    base = {"sessionId": session, "cwd": "/work/app", "timestamp": "2026-09-01T12:00:00Z"}
    use = {
        "type": "tool_use",
        "id": f"toolu_{tag}",
        "name": name,
        "input": tool_input or {"url": url},
    }
    res = {"type": "tool_result", "tool_use_id": f"toolu_{tag}", "content": text}

    def rec(kind: str, uid: str, content: Any) -> dict:
        return {**base, "type": kind, "uuid": uid, "message": {"role": kind, "content": content}}

    return [
        rec("user", f"u-{tag}", f"question {tag}"),
        rec("assistant", f"a-{tag}", [use]),
        rec("user", f"r-{tag}", [res]),
        rec("assistant", f"z-{tag}", [{"type": "text", "text": f"answer {tag}"}]),
    ]  # fmt: skip


def parse(recs: list[dict], project: str | None = "app"):
    return JSONLParser().parse_records(recs, "hook", project, with_records=True)


def artifacts(conn) -> list[tuple]:
    return conn.execute(
        "SELECT tool_use_id, session_id, parent_episode_id, metadata FROM web_artifacts ORDER BY id"
    ).fetchall()


def test_capture_is_off_unless_switched_on(db, conn, monkeypatch):
    monkeypatch.delenv("SYNAPSE_WEB_CAPTURE", raising=False)
    stats = store_episodes(db, parse(records("a", "s-off")))
    assert (stats.written, stats.captured) == (1, 0)
    assert artifacts(conn) == []


def test_a_new_turns_research_is_captured_and_linked_to_its_episode(db, conn, capture_on):
    eps = parse(records("a", "s-1"))
    eps[0].surface_id = "surf-1"
    stats = store_episodes(db, eps)
    assert (stats.written, stats.captured, stats.capture_errors) == (1, 1, 0)
    ((tid, session, parent, meta),) = artifacts(conn)
    episode_id = conn.execute("SELECT id FROM episodes WHERE session_id = 's-1'").fetchone()[0]
    assert (tid, session, parent) == ("toolu_a", "s-1", episode_id)
    assert meta == {
        "capture": "generic",
        "project": "app",
        "surface_id": "surf-1",
        "kg_extract": False,
    }


def test_a_reshipped_turn_captures_what_an_earlier_run_could_not(db, conn, monkeypatch):
    recs = records("a", "s-2")
    monkeypatch.delenv("SYNAPSE_WEB_CAPTURE", raising=False)
    assert store_episodes(db, parse(recs)).written == 1  # stored while capture was off
    assert artifacts(conn) == []

    monkeypatch.setenv("SYNAPSE_WEB_CAPTURE", "on")
    stats = store_episodes(db, parse(recs))  # the same records again: the turn is stored already
    assert (stats.written, stats.captured) == (0, 1)
    episode_id = conn.execute("SELECT id FROM episodes WHERE session_id = 's-2'").fetchone()[0]
    assert [(a[0], a[2]) for a in artifacts(conn)] == [("toolu_a", episode_id)]

    again = store_episodes(db, parse(recs))  # and once more: nothing new
    assert (again.written, again.captured) == (0, 0)
    assert len(artifacts(conn)) == 1


def test_the_guards_that_refuse_a_turn_refuse_its_research(db, conn, capture_on):
    private = parse(records("p", "s-private"))
    conn.execute("INSERT INTO private_sessions (session_id) VALUES ('s-private')")
    scratch = parse(records("s", "s-scratch"), project="scratch")
    stats = store_episodes(db, private + scratch)
    assert (stats.written, stats.private, stats.dropped, stats.captured) == (0, 1, 1, 0)
    assert artifacts(conn) == []


def test_research_of_a_stored_turn_in_a_session_made_private_since_is_not_captured(
    db, conn, monkeypatch
):
    recs = records("a", "s-3")
    monkeypatch.delenv("SYNAPSE_WEB_CAPTURE", raising=False)
    store_episodes(db, parse(recs))
    conn.execute("INSERT INTO private_sessions (session_id) VALUES ('s-3')")
    monkeypatch.setenv("SYNAPSE_WEB_CAPTURE", "on")
    assert store_episodes(db, parse(recs)).captured == 0
    assert artifacts(conn) == []


def test_a_deleted_turn_is_not_captured_when_it_is_shipped_again(db, db_url, conn, capture_on):
    from mcp_server.dashboard_routes import _delete_episode

    if not conn.execute("SELECT to_regclass('deleted_turns')").fetchone()[0]:
        pytest.skip("schema 123 (deleted_turns) is not applied")
    recs = records("a", "s-4")
    store_episodes(db, parse(recs))
    conn.execute("TRUNCATE web_artifacts CASCADE")
    eid = conn.execute("SELECT id FROM episodes WHERE session_id = 's-4'").fetchone()[0]
    assert _delete_episode(db_url, eid) is not None
    stats = store_episodes(db, parse(recs))
    assert (stats.written, stats.tombstoned, stats.captured) == (0, 1, 0)
    assert artifacts(conn) == []


def test_a_capture_failure_does_not_touch_the_episode_write(db, conn, capture_on, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("capture is down")

    monkeypatch.setattr(capture, "capture_calls", boom)
    stats = store_episodes(db, parse(records("a", "s-5")))
    assert (stats.written, stats.captured, stats.capture_errors) == (1, 0, 1)
    assert conn.execute("SELECT count(*) FROM episodes").fetchone()[0] == 1


def test_a_misspelled_switch_loses_the_capture_not_the_turn(db, conn, monkeypatch):
    monkeypatch.setenv("SYNAPSE_WEB_CAPTURE", "yes")
    stats = store_episodes(db, parse(records("a", "s-6")))
    assert (stats.written, stats.captured, stats.capture_errors) == (1, 0, 1)


def test_calls_that_belong_to_no_turn_are_captured_too(db, conn, capture_on):
    side = [
        {**r, "isSidechain": True} for r in records("side", "s-7")[1:3]
    ]  # a sub-agent: no stored turn
    calls = capture.sidechain_tool_calls(side, lambda cwd: "app", None)
    stats = store_episodes(db, [], extra_calls=calls)
    assert (stats.written, stats.captured) == (0, 1)
    assert [(a[0], a[2]) for a in artifacts(conn)] == [("toolu_side", None)]


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------


def _client(db_url):
    from fastmcp import FastMCP
    from starlette.testclient import TestClient

    from mcp_server.ingest_route import register

    m = FastMCP("test-capture-ingest")
    register(m, lambda: db_url, lambda r: True)
    return TestClient(m.http_app())


def test_ingest_reports_what_it_captured(db_url, conn, capture_on):
    sid = f"r-{uuid.uuid4().hex}"
    r = _client(db_url).post("/ingest", json={"records": records("a", sid), "project": "app"})
    assert r.status_code == 200 and r.json() == {
        "status": "ok",
        "ingested": 1,
        "captured": 1,
        "capture": True,
    }
    assert [a[:2] for a in artifacts(conn)] == [("toolu_a", sid)]


def test_ingest_of_records_that_make_no_turn_is_a_clean_no_op(db_url, conn, capture_on):
    r = _client(db_url).post("/ingest", json={"records": []})
    nothing = {"status": "ok", "ingested": 0, "captured": 0, "capture": True}
    assert r.status_code == 200 and r.json() == nothing
    r = _client(db_url).post("/ingest", json={"records": [{"type": "summary", "summary": "x"}]})
    assert r.status_code == 200 and r.json() == nothing


def test_ingest_captures_a_post_of_nothing_but_sub_agent_records(db_url, conn, capture_on):
    sid = f"r-{uuid.uuid4().hex}"
    side = [{**r, "isSidechain": True} for r in records("side", sid)[1:3]]
    r = _client(db_url).post("/ingest", json={"records": side, "project": "app"})
    assert r.status_code == 200 and r.json() == {
        "status": "ok",
        "ingested": 0,
        "captured": 1,
        "capture": True,
    }
    assert conn.execute("SELECT count(*) FROM episodes").fetchone()[0] == 0
    assert [(a[0], a[3]["project"]) for a in artifacts(conn)] == [("toolu_side", "app")]


def test_ingest_with_capture_off_stores_the_turn_and_no_research(db_url, conn, monkeypatch):
    monkeypatch.delenv("SYNAPSE_WEB_CAPTURE", raising=False)
    sid = f"r-{uuid.uuid4().hex}"
    body = {
        "records": records("a", sid) + [{**r, "isSidechain": True} for r in records("s", sid)[1:3]]
    }
    r = _client(db_url).post("/ingest", json=body)
    # `capture` tells the hooks whether shipping sub-agent files and saved outputs is of use
    assert r.json() == {"status": "ok", "ingested": 1, "captured": 0, "capture": False}
    assert artifacts(conn) == []


def test_ingest_of_a_misspelled_switch_says_capture_is_off(db_url, conn, monkeypatch):
    monkeypatch.setenv("SYNAPSE_WEB_CAPTURE", "yes")
    r = _client(db_url).post("/ingest", json={"records": records("a", f"r-{uuid.uuid4().hex}")})
    assert r.status_code == 200 and r.json()["capture"] is False


def test_ingest_stamps_the_research_with_the_callers_surface(db_url, conn, capture_on):
    from fastmcp import FastMCP
    from starlette.testclient import TestClient

    from ingestion.surfaces import SurfaceTrust
    from mcp_server.ingest_route import register

    trust = SurfaceTrust(surface_id="dev-1", known=True, credential_bound=True)
    m = FastMCP("test-capture-surface")
    register(m, lambda: db_url, lambda r: True, lambda r: trust)
    body = {"records": records("a", f"r-{uuid.uuid4().hex}"), "surface_id": "dev-evil"}
    assert TestClient(m.http_app()).post("/ingest", json=body).json()["captured"] == 1
    assert artifacts(conn)[0][3]["surface_id"] == "dev-1"


def test_a_shell_command_is_captured_only_by_its_rule_into_the_archive(db, conn, capture_on):
    thread = (
        '[{"author": {"username": "steipete"}, "text": "' + "A point worth keeping. " * 20 + '"}]'
    )
    recs = records(
        "a", "s-3", name="Bash", tool_input={"command": "bird thread 1790000000000000001 --json"},
        text=thread,
    )  # fmt: skip
    recs += records("b", "s-3", name="Bash", tool_input={"command": "curl https://example.com"})
    stats = store_episodes(db, parse(recs))
    assert stats.captured == 1 and artifacts(conn) == []
    ((kind, source_id, session, parent),) = conn.execute(
        "SELECT kind, source_id, session_id, parent_episode_id FROM research_archive"
    ).fetchall()
    episode_id = conn.execute("SELECT id FROM episodes WHERE session_id = 's-3'").fetchone()[0]
    assert (kind, source_id, session) == ("x_thread", "thread:x:1790000000000000001", "s-3")
    assert parent == episode_id
