"""POST /capture/dry-run: what capture would store from posted records, and nothing stored.

The route runs the live path (parse, detect, rules, private sessions, duplicates, caps) with
``dry_run=True``; its reply is ``ingestion.capture.report``'s account, grouped by tool.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from ingestion.capture.ruleset import load_rules
from ingestion.db import Database
from ingestion.episode_writer import store_episodes
from tests.helpers.db import research_archive_exists, truncate_research_archive
from tests.test_capture_ingest import PAGE, parse, records

pytestmark = pytest.mark.db


@pytest.fixture(autouse=True)
def _clean(clean_tables, conn, monkeypatch):
    monkeypatch.delenv("SYNAPSE_WEB_CAPTURE", raising=False)
    monkeypatch.delenv("SYNAPSE_CAPTURE_DIR", raising=False)
    load_rules.cache_clear()
    tables = "web_artifacts, private_sessions"
    conn.execute(f"TRUNCATE {tables} RESTART IDENTITY CASCADE")  # nosec B608 - fixed names
    truncate_research_archive(conn)
    yield
    conn.execute(f"TRUNCATE {tables} RESTART IDENTITY CASCADE")  # nosec B608 - fixed names
    truncate_research_archive(conn)
    load_rules.cache_clear()


def _client(db_url, allowed: bool = True) -> TestClient:
    from fastmcp import FastMCP

    from mcp_server.ingest_route import register

    m = FastMCP("test-capture-dry-run")
    register(m, lambda: db_url, lambda r: allowed)
    return TestClient(m.http_app())


def _dry_run(db_url, recs, **body):
    return _client(db_url).post("/capture/dry-run", json={"records": recs, **body})


def _stored(conn) -> tuple[int, int, int]:
    """Rows in episodes, web_artifacts and the cold archive (0 where that is not applied)."""

    def count(table: str) -> int:
        return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]  # nosec B608 - fixed

    archived = count("research_archive") if research_archive_exists(conn) else 0
    return (count("episodes"), count("web_artifacts"), archived)


def test_it_reports_by_tool_what_a_live_capture_would_store_and_stores_nothing(db_url, conn):
    recs = records("a", "s-1", url="https://example.com/a")
    recs += records("b", "s-1", name="mcp__acme__read", tool_input={"url": "http://localhost/x"})
    recs += records("c", "s-1", name="Bash", tool_input={"command": "curl https://example.com"})
    recs += records("d", "s-1", name="Read", tool_input={"file_path": "/etc/hosts"})
    r = _dry_run(db_url, recs, project="app")

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok" and body["enabled"] is False
    assert (body["calls"], body["ignored"], body["errors"]) == (4, 1, 0)
    assert body["tools"]["mcp__acme__read"]["captured"] == 1
    assert body["tools"]["mcp__acme__read"]["skipped"] == {"non_public_url": 1}
    assert body["tools"]["mcp__acme__read"]["samples"] == [
        {
            "url": "https://example.com/a",
            "title": "A Real Article",
            "kind": "web_scrape",
            "destination": "searchable",
            "source": "generic",
        }
    ]
    assert body["tools"]["Bash"] == {"captured": 0, "skipped": {"shell": 1}, "samples": []}
    assert _stored(conn) == (0, 0, 0)  # no turn, no artifact


def test_it_answers_with_the_switch_on_and_says_so(db_url, monkeypatch):
    monkeypatch.setenv("SYNAPSE_WEB_CAPTURE", "on")
    assert _dry_run(db_url, []).json() == {
        "status": "ok",
        "enabled": True,
        "calls": 0,
        "ignored": 0,
        "errors": 0,
        "tools": {},
    }


def test_it_sees_the_same_duplicates_and_private_sessions_a_live_capture_would(
    db_url, conn, monkeypatch
):
    monkeypatch.setenv("SYNAPSE_WEB_CAPTURE", "on")
    db = Database(db_url)
    try:
        store_episodes(db, parse(records("a", "s-2")))
    finally:
        db.close()
    assert conn.execute("SELECT count(*) FROM web_artifacts").fetchone()[0] == 1
    conn.execute("INSERT INTO private_sessions (session_id) VALUES ('s-private')")

    recs = records("a", "s-2") + records("p", "s-private", url="https://example.com/p")
    body = _dry_run(db_url, recs).json()

    assert body["tools"]["mcp__acme__read"]["captured"] == 0
    assert body["tools"]["mcp__acme__read"]["skipped"] == {"duplicate": 1, "private": 1}
    assert conn.execute("SELECT count(*) FROM web_artifacts").fetchone()[0] == 1


def test_it_reads_sub_agent_records_and_codex_rollouts(db_url):
    side = [{**r, "isSidechain": True} for r in records("side", "s-3")[1:3]]
    claude = _dry_run(db_url, side).json()
    assert claude["tools"]["mcp__acme__read"]["captured"] == 1

    rollout = [
        {
            "timestamp": "2026-09-01T12:00:00Z",
            "type": "response_item",
            "payload": payload,
        }
        for payload in (
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "q"}]},
            {
                "type": "function_call",
                "name": "mcp__acme__read",
                "call_id": "c1",
                "arguments": '{"url": "https://example.com/codex"}',
            },
            {"type": "function_call_output", "call_id": "c1", "output": PAGE},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "a"}]},
        )
    ]  # fmt: skip
    codex = _dry_run(db_url, rollout, format="codex", session_id="codex-1").json()
    assert codex["tools"]["mcp__acme__read"]["captured"] == 1


@pytest.mark.usefixtures("research_archive")
def test_an_archive_rule_is_reported_with_its_destination(db_url):
    thread = '[{"author": {"username": "u"}, "text": "' + "A point worth keeping. " * 20 + '"}]'
    recs = records(
        "a", "s-4", name="Bash", tool_input={"command": "bird thread 1790000000000000001 --json"},
        text=thread,
    )  # fmt: skip
    sample = _dry_run(db_url, recs).json()["tools"]["Bash"]["samples"][0]
    assert (sample["destination"], sample["kind"], sample["source"]) == (
        "archive",
        "x_thread",
        "x_thread",
    )


def test_it_has_the_ingest_gate_and_the_ingest_errors(db_url):
    assert _client(db_url, allowed=False).post("/capture/dry-run", json={}).status_code == 401
    client = _client(db_url)
    assert client.post("/capture/dry-run", content=b"not json").json()["detail"] == (
        "invalid JSON body"
    )
    r = client.post("/capture/dry-run", json={"nope": 1})
    assert (r.status_code, r.json()["detail"]) == (400, "body must contain a 'records' list")
    r = client.post("/capture/dry-run", json={"records": [], "format": "cursor"})
    assert (r.status_code, r.json()["detail"]) == (400, "unknown format 'cursor'")


def test_a_bad_rule_file_is_a_named_server_error_not_a_bare_500(db_url, tmp_path, monkeypatch):
    (tmp_path / "broken.toml").write_text("[match\n")
    monkeypatch.setenv("SYNAPSE_CAPTURE_DIR", str(tmp_path))
    r = _dry_run(db_url, [])
    assert r.status_code == 500
    assert "web capture is misconfigured" in r.json()["detail"]
    assert "broken.toml" in r.json()["detail"]
