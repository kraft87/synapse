"""DB-backed tests for the session-start serve record (schema 071, mcp_server/board_serves.py).

One ``board_serves`` row per ``GET /context`` that built a part: the note, event and
preference ids in the order the reply shows them, a hash of each block, and who it went
to. The record is written after the reply, never changes it, records nothing for a
private session, and records nothing (without an error) where 071 is not applied.
Skips cleanly when no test DB is reachable (mirrors test_board.py)."""

from __future__ import annotations

import hashlib
import os
import re

import psycopg
import pytest

_DB_URL = os.environ.get(
    "SYNAPSE_TEST_URL", "postgresql://synapse:synapse@127.0.0.1:5432/synapse_test"
)

try:
    _probe = psycopg.connect(_DB_URL, connect_timeout=2)
    _probe.close()
except Exception:  # pragma: no cover - environment dependent
    pytest.skip("no test DB reachable", allow_module_level=True)

from fastmcp import FastMCP  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from ingestion.db import Database  # noqa: E402
from ingestion.schema_check import ColumnProbe  # noqa: E402
from ingestion.surfaces import FULL_TRUST, UNKNOWN_SURFACE, SurfaceTrust  # noqa: E402
from mcp_server import board_serves  # noqa: E402
from mcp_server.board import _OWNER, build_board, register  # noqa: E402
from mcp_server.preferences_routes import top_block  # noqa: E402

_TOKEN = "test-board-serves-token"
_AUTH = {"Authorization": f"Bearer {_TOKEN}"}
_SID = "0b6c2f1e-4d3a-4c55-9a7e-2f1d8b9c0a11"


def _wipe(conn):
    conn.execute("TRUNCATE episodes RESTART IDENTITY CASCADE")
    conn.execute("DELETE FROM notes")
    conn.execute("DELETE FROM timeline_events")
    conn.execute("DELETE FROM preferences WHERE owner_id = %s", (_OWNER,))
    conn.execute("DELETE FROM private_sessions")
    conn.execute("DELETE FROM board_serves")


@pytest.fixture(autouse=True)
def _clean(conn):
    _wipe(conn)
    yield
    _wipe(conn)


def _rows(conn) -> list[dict]:
    cur = conn.cursor(row_factory=psycopg.rows.dict_row)
    return cur.execute("SELECT * FROM board_serves ORDER BY id").fetchall()


def _fixture(db_url, conn) -> None:
    """Notes of every type, events on three days and three preferences: enough that
    the board's section order differs from the notes' list order."""
    db = Database(db_url)
    for kind, hook in [
        ("reference", "Reference doc lives at docs/x.md"),
        ("user", "User lives in a cold place"),
        ("feedback", "User prefers the proper fix"),
        ("user", "User owns two cats"),
        ("feedback", "User wants short answers"),
    ]:
        db.insert_note(
            owner_id=_OWNER,
            project=None,
            type=kind,
            hook=hook,
            body=f"Body of: {hook}",
            embedding=None,
            embed_model=None,
            source_ref=None,
        )
    for i, pref in enumerate(["User likes tabs", "User dislikes emoji", "User wants tests"]):
        db.insert_preference(
            owner_id=_OWNER,
            project=None,
            pref=pref,
            polarity=("like", "dislike", "rule")[i],
            embedding=None,
            embed_model=None,
            source_ref=f"ep:{i}",
        )
    db.close()
    for days in (1, 2, 3):
        conn.execute(
            "INSERT INTO timeline_events (t_valid, fact, source, source_ref, project, salience) "
            "VALUES (now() - make_interval(days => %s), %s, 'test', %s, 'alpha', 2)",
            (days, f"Event {days} days ago", f"tb:{days}"),
        )


def _client(db_url, resolve_trust=None) -> TestClient:
    def authorized(request):
        return request.headers.get("authorization", "") == f"Bearer {_TOKEN}"

    app = FastMCP("test-board-serves")
    register(app, db_url, authorized, resolve_trust=resolve_trust)
    return TestClient(app.http_app())


def _ids(text: str, prefix: str) -> list[int]:
    return [int(m) for m in re.findall(rf"\({prefix}:(\d+)", text)]


def test_a_session_start_records_what_it_showed_in_order(conn, db_url):
    _fixture(db_url, conn)
    url = f"/context?parts=board,preferences&session={_SID}"
    headers = {**_AUTH, "User-Agent": "synapse-client/0.24.21 hermes"}
    with _client(db_url, resolve_trust=lambda _r: FULL_TRUST) as client:
        body = client.get(url, headers=headers).json()
    text, prefs = body["text"], body["preferences"]
    # The reply is byte for byte what the builders render: recording changes nothing.
    assert text == build_board(db_url, None, trust=FULL_TRUST)["text"]
    assert prefs == top_block(db_url, FULL_TRUST)
    assert set(body) == {"status", "trust", "text", "n_notes", "overflow", "preferences"}

    (row,) = _rows(conn)
    assert row["session_id"] == _SID and row["project"] is None
    assert row["trust"] == "full" and row["client"] == "synapse-client/0.24.21 hermes"
    assert row["parts"] == ["board", "preferences"]
    # In the order the board shows them, which is section order, not list order.
    assert row["note_ids"] == _ids(text, "n") and len(row["note_ids"]) == 5
    assert row["event_ids"] == _ids(text, "t") and len(row["event_ids"]) == 3
    by_pref = dict(
        conn.execute("SELECT pref, id FROM preferences WHERE owner_id = %s", (_OWNER,)).fetchall()
    )
    shown = [line.split(") ", 1)[1] for line in prefs.splitlines()[1:]]
    assert row["pref_ids"] == [by_pref[p] for p in shown]
    assert row["board_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert row["prefs_sha256"] == hashlib.sha256(prefs.encode()).hexdigest()
    assert row["prefs_text"] == prefs
    assert row["board_lines"] == text.count("\n") + 1
    assert row["board_tokens"] == len(text) // 4
    assert row["notes_dropped"] == 0 and row["ms_build"] is not None


def test_the_analysis_questions_are_one_query_each(conn, db_url):
    _fixture(db_url, conn)
    with _client(db_url, resolve_trust=lambda _r: FULL_TRUST) as client:
        for sid in ("s-1", "s-2", "s-2"):  # s-2 is served twice (a resume)
            client.get(f"/context?parts=board,preferences&session={sid}", headers=_AUTH)
    note = _rows(conn)[0]["note_ids"][0]
    on_board = conn.execute(
        "SELECT 1 FROM board_serves WHERE session_id = %s AND note_ids @> ARRAY[%s]::bigint[]",
        ("s-1", note),
    ).fetchone()
    assert on_board is not None
    sessions = conn.execute(
        "SELECT count(DISTINCT session_id) FROM board_serves WHERE note_ids @> ARRAY[%s]::bigint[]",
        (note,),
    ).fetchone()[0]
    assert sessions == 2


def test_a_private_session_is_not_recorded(conn, db_url):
    _fixture(db_url, conn)
    conn.execute("INSERT INTO private_sessions (session_id) VALUES (%s)", (_SID,))
    with _client(db_url, resolve_trust=lambda _r: FULL_TRUST) as client:
        body = client.get(f"/context?session={_SID}", headers=_AUTH).json()
        client.get("/context?session=on-the-record", headers=_AUTH)
    assert "User prefers the proper fix" in body["text"]  # still served its board
    assert [r["session_id"] for r in _rows(conn)] == ["on-the-record"]


def test_a_restricted_serve_records_only_what_it_was_shown(conn, db_url):
    """The ids are the reply's own, so a restricted device's row names nothing it was
    not served; the surface comes from the credential."""
    _fixture(db_url, conn)
    work = SurfaceTrust(
        surface_id="dev-work",
        trust="restricted",
        allowed_projects=("alpha",),
        known=True,
        credential_bound=True,
    )
    with _client(db_url, resolve_trust=lambda _r: work) as client:
        body = client.get("/context?parts=board,preferences&session=w", headers=_AUTH).json()
    (row,) = _rows(conn)
    assert row["trust"] == "restricted" and row["surface_id"] == "dev-work"
    assert row["note_ids"] == _ids(body["text"], "n") == []  # every note is personal
    assert row["event_ids"] == _ids(body["text"], "t")
    assert row["pref_ids"] == [] and row["prefs_text"] == body["preferences"] == ""


def test_an_unknown_caller_is_recorded_without_a_surface(conn, db_url):
    with _client(db_url) as client:
        client.get("/context", headers=_AUTH)
    (row,) = _rows(conn)
    assert row["trust"] == UNKNOWN_SURFACE.trust and row["surface_id"] is None


def test_without_the_table_nothing_is_recorded_and_nothing_fails(conn, db_url, monkeypatch):
    """A database without schema 071: the probe says absent, so no INSERT is tried."""
    monkeypatch.setattr(
        board_serves,
        "TABLE_READY",
        ColumnProbe("SELECT 1 WHERE to_regclass('board_serves_absent') IS NOT NULL", "absent"),
    )
    with _client(db_url, resolve_trust=lambda _r: FULL_TRUST) as client:
        reply = client.get("/context", headers=_AUTH)
    assert reply.status_code == 200 and _rows(conn) == []


def test_record_never_raises():
    serve = board_serves.Serve(trust=FULL_TRUST, parts=["board"], board_text="x")
    board_serves.record("postgresql://nobody@127.0.0.1:1/nothing_test", serve)


def test_old_rows_are_pruned(conn, db_url, monkeypatch):
    conn.execute(
        "INSERT INTO board_serves (served_at, trust, parts) VALUES "
        "(now() - make_interval(days => %s), 'full', '{board}'), "
        "(now() - make_interval(days => 1), 'full', '{board}')",
        (board_serves.RETENTION_DAYS + 1,),
    )
    monkeypatch.setattr(board_serves, "_pruned_at", {})
    serve = board_serves.Serve(trust=FULL_TRUST, parts=["board"], board_text="x")
    board_serves.record(db_url, serve)
    ages = conn.execute(
        "SELECT served_at > now() - interval '2 days' FROM board_serves ORDER BY id"
    ).fetchall()
    assert ages == [(True,), (True,)]  # the 181-day-old row is gone


def test_note_order_follows_the_board_sections():
    notes = [
        {"id": 1, "type": "reference"},
        {"id": 2, "type": "user"},
        {"id": 3, "type": "feedback"},
        {"id": 4, "type": "user"},
        {"id": 5, "type": "unknown"},
    ]
    order = ("feedback", "user", "project", "reference")
    assert board_serves.note_ids_in_board_order(notes, order) == [3, 2, 4, 1]
