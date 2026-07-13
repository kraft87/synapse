"""DB-backed tests for the board (mcp_server/board.py) + the /context route.

Covers the render contract (banner, grouping/order, line format, project scoping,
superseded exclusion), the hard cap (line overflow drop order + token cap), the
timeline section, the machine-token route gate, and kind='board' telemetry. Skips
cleanly when no test DB is reachable (mirrors test_notes_store.py)."""

from __future__ import annotations

import os
import re

import psycopg
import pytest

_DB_URL = os.environ.get(
    "SYNAPSE_TEST_URL", "postgresql://synapse:synapse@127.0.0.1:5432/synapse_test"
)

# Skip the whole module if the shared Postgres test DB isn't up — these tests are DB-only.
try:
    _probe = psycopg.connect(_DB_URL, connect_timeout=2)
    _probe.close()
except Exception:  # pragma: no cover - environment dependent
    pytest.skip("no test DB reachable", allow_module_level=True)

from starlette.testclient import TestClient  # noqa: E402

from ingestion.db import Database  # noqa: E402
from mcp_server.board import _OWNER, build_board, record_board_metrics, register  # noqa: E402

GROUP = "technical"


def _wipe(conn):
    conn.execute("TRUNCATE episodes RESTART IDENTITY CASCADE")
    conn.execute("DELETE FROM notes")
    conn.execute("DELETE FROM timeline_events")
    conn.execute("DELETE FROM recall_metrics")


def _note(db, *, type="user", hook="User prefers X", project=None):
    """Board tests never exercise the KNN, so notes land with NULL embedding."""
    return db.insert_note(
        owner_id=_OWNER,
        group_id=GROUP,
        project=project,
        type=type,
        hook=hook,
        body=f"Body of: {hook}",
        embedding=None,
        embed_model=None,
        source_ref=None,
    )


def _episode(conn, session, seq, project, days_ago=0):
    conn.execute(
        "INSERT INTO episodes (session_id, sequence, project, content, created_at) "
        "VALUES (%s, %s, %s, %s, now() - make_interval(days => %s))",
        (session, seq, project, f"turn {seq}", days_ago),
    )


def _event(conn, fact, *, salience=2, days_ago=1, project="alpha", ref="tb:0"):
    conn.execute(
        "INSERT INTO timeline_events (t_valid, fact, source, source_ref, project, salience) "
        "VALUES (now() - make_interval(days => %s), %s, 'test', %s, %s, %s)",
        (days_ago, fact, ref, project, salience),
    )


# ---------------------------------------------------------------------------
# Render contract
# ---------------------------------------------------------------------------


def test_empty_db_renders_banner_only(conn, db_url):
    _wipe(conn)
    out = build_board(db_url, None)
    assert out["status"] == "ok"
    assert out["n_notes"] == 0 and out["overflow"] == 0 and out["note_ids"] == []
    text = out["text"]
    assert text.startswith("[Synapse board — project: all]")
    assert "0 episodes across 0 projects." in text
    assert "Absence from this board means SEARCH (recall)" in text
    assert "##" not in text  # empty sections omitted


def test_banner_counts_and_recent_projects(conn, db_url):
    _wipe(conn)
    _episode(conn, "s1", 1, "alpha", days_ago=5)
    _episode(conn, "s1", 2, "alpha", days_ago=5)
    _episode(conn, "s2", 1, "beta", days_ago=1)
    text = build_board(db_url, "alpha")["text"]
    assert text.startswith("[Synapse board — project: alpha]")
    # beta has the most recent activity, so it leads the most-recent list.
    assert "3 episodes across 2 projects (most recent: beta, alpha)." in text


def test_banner_null_project_episodes_do_not_consume_slots(conn, db_url):
    """A NULL-project episode group must not eat one of the 12 most-recent slots or
    skew the project count: 12 real projects + newer project-less episodes -> all 12
    listed, P counts 12."""
    _wipe(conn)
    for i in range(12):
        _episode(conn, f"s{i}", 1, f"proj{i:02d}", days_ago=i + 1)
    # Most recent activity of all — would win a LIMIT slot if not excluded in SQL.
    _episode(conn, "s-null", 1, None, days_ago=0)
    text = build_board(db_url, None)["text"]
    assert "13 episodes across 12 projects" in text
    for i in range(12):
        assert f"proj{i:02d}" in text


def test_grouping_order_and_line_format(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    nid = _note(db, type="feedback", hook="Never use tables in answers")
    _note(db, type="user", hook="User works in a homelab")
    _note(db, type="project", hook="Board PR in flight", project="alpha")
    _note(db, type="reference", hook="See the notes design doc")
    db.close()
    text = build_board(db_url, "alpha")["text"]
    order = [
        text.index("## Rules & feedback"),
        text.index("## User"),
        text.index("## Project: alpha"),
        text.index("## References"),
    ]
    assert order == sorted(order)  # feedback -> user -> project -> reference
    line = next(ln for ln in text.splitlines() if "Never use tables" in ln)
    assert re.fullmatch(rf"- Never use tables in answers \(n:{nid}, upd \d{{2}}-\d{{2}}\)", line)


def test_project_scoping(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    _note(db, type="project", hook="Alpha project state", project="alpha")
    _note(db, type="project", hook="Beta project state", project="beta")
    _note(db, type="user", hook="Global user fact")
    db.close()
    text = build_board(db_url, "alpha")["text"]
    assert "Alpha project state" in text
    assert "Beta project state" not in text  # other projects' notes excluded
    assert "Global user fact" in text  # globals always present
    # No project scope -> global set only, no project section at all.
    text_all = build_board(db_url, None)["text"]
    assert "Alpha project state" not in text_all and "## Project" not in text_all


def test_superseded_notes_excluded(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    old = _note(db, type="user", hook="User prefers dark mode")
    new = _note(db, type="user", hook="User now prefers light mode")
    db.supersede_note(old, new)
    db.close()
    out = build_board(db_url, None)
    assert "User now prefers light mode" in out["text"]
    assert "User prefers dark mode" not in out["text"]
    assert out["note_ids"] == [new]


# ---------------------------------------------------------------------------
# Hard cap / overflow
# ---------------------------------------------------------------------------


def test_line_overflow_drops_oldest_project_notes_first(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    _note(db, type="feedback", hook="Feedback rule one")
    _note(db, type="feedback", hook="Feedback rule two")
    ids = [
        _note(db, type="project", hook=f"Project note {i:03d}", project="alpha") for i in range(90)
    ]
    db.close()
    # Stagger updated_at: note 000 oldest ... 089 newest.
    for age, nid in enumerate(reversed(ids)):
        conn.execute(
            "UPDATE notes SET updated_at = now() - make_interval(days => %s) WHERE id = %s",
            (age, nid),
        )
    out = build_board(db_url, "alpha")
    text = out["text"]
    assert out["overflow"] > 0
    assert text.count("\n") + 1 <= 80
    assert f"(+ {out['overflow']} older notes not shown)" in text
    # Oldest-updated project notes dropped first; newest survive; feedback survives.
    assert "Project note 000" not in text
    assert "Project note 089" in text
    assert "Feedback rule one" in text and "Feedback rule two" in text
    assert out["n_notes"] == 92 - out["overflow"]


def test_token_cap_on_long_hooks(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    for i in range(5):
        _note(db, type="user", hook=f"Long note {i} " + "x" * 2000)
    db.close()
    out = build_board(db_url, None)
    assert out["overflow"] >= 1  # far under 80 lines — the token cap did this
    assert len(out["text"]) // 4 <= 2000


def test_long_event_facts_clamped_and_feedback_survives(conn, db_url):
    """Verbose timeline facts (POST /timeline/events accepts unbounded text) must not
    bust the hard cap or evict curated notes: event lines are clamped at render, so
    the fixed section stays bounded and the cap loop never drains feedback."""
    _wipe(conn)
    for i in range(5):
        _event(conn, f"Event {i} " + "y" * 3000, salience=2, days_ago=1, ref=f"tb:long{i}")
    db = Database(db_url)
    _note(db, type="feedback", hook="Feedback must survive event floods")
    db.close()
    out = build_board(db_url, None)
    text = out["text"]
    assert out["overflow"] == 0
    assert "Feedback must survive event floods" in text
    assert text.count("\n") + 1 <= 80 and len(text) // 4 <= 2000  # both caps hold
    assert "y" * 250 not in text  # each event fact clamped, not served whole
    assert all(len(ln) <= 250 for ln in text.splitlines())


def test_cap_loop_keeps_notes_when_floor_over_cap(conn, db_url):
    """If the fixed portion alone busts the token cap (pathological project names in
    the banner), dropping notes cannot reach the cap — the loop must keep them all
    instead of draining the board for zero benefit."""
    _wipe(conn)
    for i in range(12):
        _episode(conn, f"s{i}", 1, f"project-{i}-" + "z" * 800, days_ago=i)
    db = Database(db_url)
    _note(db, type="feedback", hook="Keep me")
    db.close()
    out = build_board(db_url, None)
    assert out["overflow"] == 0 and out["n_notes"] == 1
    assert "Keep me" in out["text"]


def test_overflow_line_separated_when_all_notes_dropped(conn, db_url):
    """A board whose every note was evicted still renders the overflow line with a
    blank line after the banner, not jammed directly under it."""
    _wipe(conn)
    db = Database(db_url)
    _note(db, type="user", hook="Giant " + "x" * 9000)  # one note, over the cap alone
    db.close()
    out = build_board(db_url, None)
    assert out["n_notes"] == 0 and out["overflow"] == 1
    assert out["text"].endswith("\n\n(+ 1 older notes not shown)")


# ---------------------------------------------------------------------------
# Timeline section
# ---------------------------------------------------------------------------


def test_timeline_section_present_and_absent(conn, db_url):
    _wipe(conn)
    _event(conn, "Shipped the board", salience=2, days_ago=1, ref="tb:1")
    text = build_board(db_url, None)["text"]
    assert "## Last 7 days" in text
    assert re.search(r"- \d{2}-\d{2} \(alpha\): Shipped the board", text)

    _wipe(conn)
    _event(conn, "Routine tweak", salience=1, days_ago=1, ref="tb:2")  # below min_salience
    _event(conn, "Old milestone", salience=2, days_ago=30, ref="tb:3")  # outside window
    assert "## Last 7 days" not in build_board(db_url, None)["text"]


# ---------------------------------------------------------------------------
# /context route + telemetry
# ---------------------------------------------------------------------------

_TOKEN = "test-board-token"


def _client(db_url, get_recall=None):
    from fastmcp import FastMCP

    def authorized(request):
        return request.headers.get("authorization", "") == f"Bearer {_TOKEN}"

    test_mcp = FastMCP("test-board")
    register(test_mcp, db_url, authorized, get_recall=get_recall)
    return TestClient(test_mcp.http_app())


def test_route_auth_and_project_param(conn, db_url):
    _wipe(conn)
    db = Database(db_url)
    _note(db, type="project", hook="Alpha project state", project="alpha")
    db.close()
    with _client(db_url) as client:
        assert client.get("/context").status_code == 401  # no token
        r = client.get("/context", headers={"Authorization": f"Bearer {_TOKEN}"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok" and "note_ids" not in body
        assert "Alpha project state" not in body["text"]  # unscoped -> global set only
        r2 = client.get("/context?project=alpha", headers={"Authorization": f"Bearer {_TOKEN}"})
        assert "Alpha project state" in r2.json()["text"]


def test_route_records_board_telemetry(conn, db_url):
    from mcp_server.recall import Recall

    _wipe(conn)
    db = Database(db_url)
    kept = _note(db, type="user", hook="Telemetry fixture note")
    db.close()
    engine = Recall(db_url=db_url, voyage_api_key="")
    with _client(db_url, get_recall=lambda: engine) as client:
        r = client.get("/context", headers={"Authorization": f"Bearer {_TOKEN}"})
        assert r.status_code == 200
    # The metrics write is fire-and-forget on a single-worker FIFO pool: a barrier
    # task completing proves the row insert before ours has finished.
    engine._async_executor.submit(lambda: None).result(timeout=10)
    row = conn.execute(
        "SELECT source, ms_total, chars, est_tokens, served_ids FROM recall_metrics "
        "WHERE kind = 'board'"
    ).fetchone()
    assert row is not None
    source, ms_total, chars, est_tokens, served_ids = row
    assert source == "http"
    assert ms_total is not None and chars > 0 and est_tokens == chars // 4
    assert served_ids["notes"] == [kept]
    assert served_ids["n_notes"] == 1 and served_ids["overflow"] == 0


def test_record_board_metrics_is_fail_soft():
    class Boom:
        def record_event(self, m):
            raise RuntimeError("nope")

    # Must swallow, never raise back into the serve path.
    record_board_metrics(Boom(), "http", 1.0, {"text": "t", "note_ids": [1]})
