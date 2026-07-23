"""recall_feedback — offline labeled retrieval-quality capture (schema 046).

Pure-unit: the served-id validator and the tool's validation error dicts (no DB
touched — validation rejects before any connection opens). DB-backed (skipped
when the test DB is unreachable): the insert round-trip through the shared
_file_recall_feedback worker, for both the MCP tool and the /feedback HTTP seam.
All data is synthetic — no real queries or session ids.
"""

from __future__ import annotations

import os

import psycopg
import pytest

from mcp_server import server
from mcp_server.server import _feedback_ids_error

_DB_URL = os.environ.get(
    "SYNAPSE_TEST_URL", "postgresql://synapse:synapse@127.0.0.1:5432/synapse_test"
)

try:
    _probe = psycopg.connect(_DB_URL, connect_timeout=2)
    _probe.close()
    _DB_OK = True
except Exception:  # pragma: no cover - environment dependent
    _DB_OK = False

_needs_db = pytest.mark.skipif(not _DB_OK, reason="no test DB reachable")

# ---------------------------------------------------------------------------
# Validator — pure unit
# ---------------------------------------------------------------------------


def test_validator_accepts_none_empty_and_served_forms():
    assert _feedback_ids_error("helpful", None) is None
    assert _feedback_ids_error("helpful", []) is None
    assert _feedback_ids_error("helpful", ["e:1", "n:45", "e:227168"]) is None


def test_validator_accepts_all_bucket_id_forms():
    # Every recall bucket now carries an id; all forms are ratable.
    assert (
        _feedback_ids_error(
            "helpful",
            [
                "e:1",  # episode
                "n:45",  # note
                "t:9",  # timeline
                "w:7",  # web
                "p:3",  # preference
                "f:0b3c1d2e-4f56-7890-abcd-ef0123456789",  # fact (KG edge uuid)
            ],
        )
        is None
    )


@pytest.mark.parametrize(
    "bad",
    [
        ["x:5"],  # unknown kind
        ["e:abc"],  # non-numeric id
        ["e:1", "wat"],  # one good id does not excuse a bad one
        ["E:1"],  # kind is lowercase by contract
        ["e:1 "],  # no trailing whitespace
        ["e:-2"],  # no signs
        ["n:"],  # empty numeric part
        ["e:1\n"],  # fullmatch, not match — trailing newline rejected
        [123],  # non-string entries rejected, not coerced
        ["f:1"],  # fact id is a uuid, not an int
        ["f:short"],  # non-hex / too-short uuid rejected
        ["t:abc"],  # timeline id must be numeric
        ["g:1"],  # unknown numeric kind
    ],
)
def test_validator_rejects_malformed_ids(bad):
    err = _feedback_ids_error("noise", bad)
    assert err is not None and "noise" in err


def test_validator_rejects_non_list():
    err = _feedback_ids_error("helpful", "e:1")  # a bare string, not a list
    assert err is not None and "helpful" in err


# ---------------------------------------------------------------------------
# Tool validation — error dicts, no DB touched
# ---------------------------------------------------------------------------


def test_tool_rejects_bad_helpful_ids_before_any_db_work():
    out = server.recall_feedback(query="synthetic query", helpful=["x:9"])
    assert out["status"] == "error" and "helpful" in out["detail"]


def test_tool_rejects_bad_noise_ids():
    out = server.recall_feedback(query="synthetic query", noise=["e:abc"])
    assert out["status"] == "error" and "noise" in out["detail"]


def test_tool_rejects_blank_query():
    out = server.recall_feedback(query="   ")
    assert out["status"] == "error" and "query" in out["detail"]


# ---------------------------------------------------------------------------
# Insert round-trip — real Postgres
# ---------------------------------------------------------------------------


@pytest.fixture()
def _test_db(monkeypatch, conn):
    monkeypatch.setattr(server, "DB_URL", _DB_URL)
    conn.execute("DELETE FROM recall_feedback")
    yield
    conn.execute("DELETE FROM recall_feedback")


@_needs_db
def test_tool_round_trip_inserts_labeled_row(conn, _test_db):
    out = server.recall_feedback(
        query="synthetic: how does the widget frobnicate",
        helpful=["e:1", "n:2"],
        noise=["e:3"],
        missing="the frobnication config file path",
        note="serve config-file chunks for widget queries",
        session_id="synthetic-session-0001",
        project="demo",
    )
    assert out["status"] == "ok" and isinstance(out["feedback_id"], int)

    row = conn.execute(
        "SELECT query, helpful, noise, missing, note, session_id, project, created_at "
        "FROM recall_feedback WHERE id = %s",
        (out["feedback_id"],),
    ).fetchone()
    assert row is not None
    query, helpful, noise, missing, note, session_id, project, created_at = row
    assert query == "synthetic: how does the widget frobnicate"
    assert helpful == ["e:1", "n:2"]  # jsonb round-trips as a real list
    assert noise == ["e:3"]
    assert missing == "the frobnication config file path"
    assert note == "serve config-file chunks for widget queries"
    assert session_id == "synthetic-session-0001" and project == "demo"
    assert created_at is not None


@_needs_db
def test_tool_defaults_store_empty_lists_and_nulls(conn, _test_db):
    out = server.recall_feedback(query="synthetic: bare minimum report")
    assert out["status"] == "ok"
    row = conn.execute(
        "SELECT helpful, noise, missing, note, session_id, project "
        "FROM recall_feedback WHERE id = %s",
        (out["feedback_id"],),
    ).fetchone()
    assert row == ([], [], None, None, None, None)


@_needs_db
def test_http_worker_shares_tool_validation_and_insert(conn, _test_db):
    """The /feedback route delegates to _file_recall_feedback — same rejection,
    same insert, so the seams cannot drift."""
    from mcp_server.server import _file_recall_feedback

    bad = _file_recall_feedback("synthetic q", ["nope"], None, None, None, None, None)
    assert bad["status"] == "error"
    assert conn.execute("SELECT count(*) FROM recall_feedback").fetchone()[0] == 0

    ok = _file_recall_feedback("synthetic q", ["e:7"], None, "nothing", None, None, None)
    assert ok["status"] == "ok"
    assert conn.execute(
        "SELECT helpful FROM recall_feedback WHERE id = %s", (ok["feedback_id"],)
    ).fetchone()[0] == ["e:7"]


@_needs_db
def test_tool_round_trip_via_mcp_client(conn, _test_db):
    """Through the real FastMCP pipeline (middleware included), like the
    tool-surface tests drive it."""
    import asyncio

    from fastmcp import Client

    async def _run():
        async with Client(server.mcp) as c:
            return await c.call_tool(
                "recall_feedback",
                {"query": "synthetic: pipeline round-trip", "helpful": ["e:11"]},
            )

    res = asyncio.run(_run())
    assert res.data["status"] == "ok"
    row = conn.execute(
        "SELECT query, helpful FROM recall_feedback WHERE id = %s",
        (res.data["feedback_id"],),
    ).fetchone()
    assert row == ("synthetic: pipeline round-trip", ["e:11"])
