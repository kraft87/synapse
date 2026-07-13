"""The MCP tool-surface contract: names, ORDER, hidden tools, negative triggers,
and the unified fetch (e:/n:) behavior.

Registration order is asserted exactly — tool-list position biases which tool a model
reaches for, so a reorder is a behavior change and must fail here, deliberately.
issue_machine_token must stay callable via tools/call (the `synapse login` CLI invokes
it by name over raw JSON-RPC) while never appearing in tools/list — the middleware
half of that contract is exercised against the real FastMCP app via the in-process
client, which drives the same request pipeline (middleware included) as HTTP.
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

# Skip the whole module if the shared Postgres test DB isn't up — the fetch tests are
# DB-only and the surface pins are cheap enough to re-run wherever the DB lives.
try:
    _probe = psycopg.connect(_DB_URL, connect_timeout=2)
    _probe.close()
except Exception:  # pragma: no cover - environment dependent
    pytest.skip("no test DB reachable", allow_module_level=True)

from fastmcp import Client  # noqa: E402

from ingestion.db import Database  # noqa: E402
from mcp_server import server  # noqa: E402
from mcp_server.board import _OWNER  # noqa: E402
from mcp_server.recall import Recall  # noqa: E402

# ---------------------------------------------------------------------------
# The listed surface: names, order, absences, descriptions
# ---------------------------------------------------------------------------

_EXPECTED_ORDER = [
    "recall",
    "get_context",
    "fetch",
    "remember",
    "recall_timeline",
    "recall_episodes",
]


def _listed():
    return asyncio.run(server.mcp.list_tools())


def test_tool_list_is_exactly_the_expected_names_in_order():
    """The whole listed surface, pinned in registration order (positional bias is
    deliberate — see the Tools section comment in server.py)."""
    assert [t.name for t in _listed()] == _EXPECTED_ORDER


def test_removed_tools_are_gone():
    names = {t.name for t in _listed()}
    assert "query_graph" not in names
    assert "list_projects" not in names
    assert "fetch_episode" not in names  # replaced by fetch (e:/n:)


def test_issue_machine_token_hidden_from_list_but_callable():
    """Integration-style, against the real app: tools/list omits the hidden tool,
    tools/call still executes it (the `synapse login` compatibility contract)."""

    async def _run():
        async with Client(server.mcp) as c:
            listed = [t.name for t in await c.list_tools()]
            res = await c.call_tool("issue_machine_token", {})
            return listed, res

    listed, res = asyncio.run(_run())
    assert listed == _EXPECTED_ORDER
    assert "issue_machine_token" not in listed
    assert res.data == {"token": server.MACHINE_TOKEN}


def test_every_listed_description_carries_a_negative_trigger():
    """Trigger-inventory description style: every model-facing tool must tell the
    model when NOT to call it, not just when to call it."""
    for t in _listed():
        assert "Do NOT" in (t.description or ""), f"{t.name} has no negative trigger"


# ---------------------------------------------------------------------------
# Unified fetch: e:/n: ids, back-compat, skip reporting, the cross-kind cap
# ---------------------------------------------------------------------------


def _episode(conn, content: str) -> int:
    return conn.execute(
        "INSERT INTO episodes (session_id, sequence, content) VALUES (%s, 1, %s) RETURNING id",
        (f"tool-surface-{uuid.uuid4().hex[:8]}", content),
    ).fetchone()[0]


def _note(db_url: str, hook: str, body: str) -> int:
    db = Database(db_url)
    try:
        return db.insert_note(
            owner_id=_OWNER,
            group_id="technical",
            project="tool-surface",
            type="project",
            hook=hook,
            body=body,
            embedding=None,
            embed_model=None,
            source_ref=None,
        )
    finally:
        db.close()


def test_fetch_episode_ids_only(conn, db_url):
    e1 = _episode(conn, "turn one full text")
    e2 = _episode(conn, "turn two full text")
    out = Recall(db_url, "").fetch([f"e:{e1}", f"e:{e2}"])
    assert [e["id"] for e in out["episodes"]] == [f"e:{e1}", f"e:{e2}"]
    assert out["episodes"][0]["content"] == "turn one full text"
    assert out["notes"] == [] and out["skipped"] == []


def test_fetch_note_ids_only(conn, db_url):
    n1 = _note(db_url, "Hook one", "Body one.")
    out = Recall(db_url, "").fetch([f"n:{n1}"])
    assert out["episodes"] == []
    (note,) = out["notes"]
    # The served note shape, pinned: {id, hook, body, type, project, updated}.
    assert set(note) == {"id", "hook", "body", "type", "project", "updated"}
    assert note["id"] == f"n:{n1}" and note["hook"] == "Hook one" and note["body"] == "Body one."
    assert note["type"] == "project" and note["project"] == "tool-surface"
    assert len(note["updated"]) == 10  # ISO date, not a full timestamp


def test_fetch_mixed_ids(conn, db_url):
    e1 = _episode(conn, "mixed-fetch turn")
    n1 = _note(db_url, "Mixed hook", "Mixed body.")
    out = Recall(db_url, "").fetch([f"n:{n1}", f"e:{e1}"])
    assert [e["id"] for e in out["episodes"]] == [f"e:{e1}"]
    assert [n["id"] for n in out["notes"]] == [f"n:{n1}"]
    assert out["skipped"] == []


def test_fetch_bare_ids_are_episodes_backcompat(conn, db_url):
    """Bare "N" strings and bare ints stay episode ids (the old fetch_episode inputs),
    and dedupe against their prefixed form."""
    e1 = _episode(conn, "bare-id turn")
    out = Recall(db_url, "").fetch([str(e1), e1, f"e:{e1}"])
    assert [e["id"] for e in out["episodes"]] == [f"e:{e1}"]
    assert out["skipped"] == []


def test_fetch_unknown_ids_reported_as_skipped(conn, db_url):
    e1 = _episode(conn, "the one good id")
    out = Recall(db_url, "").fetch(["x:5", "e:abc", "wat", f"e:{e1}"])
    assert out["skipped"] == ["x:5", "e:abc", "wat"]
    assert [e["id"] for e in out["episodes"]] == [f"e:{e1}"]


def test_fetch_cap_applies_across_kinds(conn, db_url):
    """_FETCH_MAX bounds the TOTAL expanded across kinds, first-come: 18 episodes +
    4 notes requested -> 18 episodes + 2 notes served. Over-cap ids drop silently
    (matching the old episode-only path), never into `skipped`."""
    eps = [_episode(conn, f"cap turn {i}") for i in range(18)]
    notes = [_note(db_url, f"Cap hook {i}", "Body.") for i in range(4)]
    ids = [f"e:{e}" for e in eps] + [f"n:{n}" for n in notes]
    out = Recall(db_url, "").fetch(ids)
    assert len(out["episodes"]) == 18
    assert [n["id"] for n in out["notes"]] == [f"n:{notes[0]}", f"n:{notes[1]}"]
    assert out["skipped"] == []


def test_fetch_kinds_telemetry_counts(conn, db_url):
    """The kind='fetch' telemetry row carries per-kind serve counts in served_ids
    (the single-kind row shape is pinned in test_telemetry_kinds.py)."""
    e1 = _episode(conn, "telemetry mixed turn")
    n1 = _note(db_url, "Telemetry hook A", "Body.")
    n2 = _note(db_url, "Telemetry hook B", "Body.")
    engine = Recall(db_url, "")
    mark = conn.execute("SELECT coalesce(max(id), 0) FROM recall_metrics").fetchone()[0]
    engine.fetch([f"e:{e1}", f"n:{n1}", f"n:{n2}", "x:1"], source="mcp-tool")
    engine._async_executor.submit(lambda: None).result(timeout=10)  # barrier the writer
    row = conn.execute(
        "SELECT query, served_ids FROM recall_metrics "
        "WHERE kind = 'fetch' AND id > %s ORDER BY id DESC LIMIT 1",
        (mark,),
    ).fetchone()
    assert row is not None, "fetch() emitted no kind='fetch' telemetry row"
    query, served = row
    assert query == f"e:{e1},n:{n1},n:{n2}"  # normalized accepted ids; skipped ids absent
    assert served == {"kinds": {"e": 1, "n": 2}}
