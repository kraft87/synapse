"""DB-backed tests for the timeline event-ingest seam (mcp_server/timeline_routes).

Covers what the plugin's git feeder relies on: keyless insert (NULL embedding), full-batch
idempotency on re-push, partial-batch dedup (only genuinely-new events count as inserted),
and per-field storage (project / salience / t_valid land as sent).
"""

from __future__ import annotations

from mcp_server.timeline_routes import _ingest_events

SRC = "test-evt"


def _ev(ref: str, fact: str = "committed to demo: a change", sal: int = 1, et: str | None = None):
    return {
        "t_valid": "2026-07-01T12:00:00+00:00",
        "fact": fact,
        "source": SRC,
        "source_ref": ref,
        "project": "demo",
        "salience": sal,
        "event_type": et,
    }


def _wipe(conn):
    conn.execute("DELETE FROM timeline_events WHERE source = %s", (SRC,))


def test_insert_and_repush_idempotent(conn, db_url):
    _wipe(conn)
    batch = [_ev("sha1"), _ev("sha2", "decided demo events are naked", 2)]
    assert _ingest_events(db_url, "", batch) == (2, 0)
    # full re-push: everything skips, nothing double-inserts
    assert _ingest_events(db_url, "", batch) == (0, 2)
    n = conn.execute("SELECT count(*) FROM timeline_events WHERE source = %s", (SRC,)).fetchone()[0]
    assert n == 2
    _wipe(conn)


def test_partial_batch_dedup(conn, db_url):
    _wipe(conn)
    assert _ingest_events(db_url, "", [_ev("sha1")]) == (1, 0)
    # feeder re-pushes history plus one new commit — only the new one inserts
    assert _ingest_events(db_url, "", [_ev("sha1"), _ev("sha3")]) == (1, 1)
    _wipe(conn)


def test_fields_stored(conn, db_url):
    _wipe(conn)
    _ingest_events(db_url, "", [_ev("sha9", "shipped the demo milestone", 2, "milestone")])
    row = conn.execute(
        "SELECT fact, project, salience, embedding, embed_model, event_type, "
        "       left(t_valid::text, 10) AS d "
        "FROM timeline_events WHERE source = %s AND source_ref = 'sha9'",
        (SRC,),
    ).fetchone()
    fact, project, salience, embedding, embed_model, event_type, d = row
    assert fact == "shipped the demo milestone"
    assert project == "demo" and salience == 2 and d == "2026-07-01"
    assert event_type == "milestone"
    # keyless path: no embedding, no model recorded (a re-embed pass fills later)
    assert embedding is None and embed_model is None
    _wipe(conn)


def test_recent_events_window_and_salience(conn, db_url):
    from mcp_server.timeline_routes import _recent_events

    _wipe(conn)
    _ingest_events(
        db_url,
        "",
        [
            _ev("m1", "shipped the timeline to prod", 2, "milestone"),
            _ev("a1", "ran a routine benchmark", 1, "action"),
        ],
    )
    items = _recent_events(db_url, days=36500, min_salience=2, limit=5, project="demo")
    assert [i["fact"] for i in items] == ["shipped the timeline to prod"]
    assert items[0]["event_type"] == "milestone" and items[0]["date"] == "2026-07-01"
    # salience floor 0 -> both come back, newest-first cap respected
    items = _recent_events(db_url, days=36500, min_salience=0, limit=5, project="demo")
    assert len(items) == 2
    _wipe(conn)


def test_timeline_ident_exists_window(conn, db_url):
    from ingestion.db import Database

    _wipe(conn)
    _ingest_events(db_url, "", [_ev("shaX", "committed to demo: fix the gate (#321)")])
    db = Database(db_url)
    hit = db.timeline_ident_exists(["#321"], "demo", "2026-07-01T18:00:00+00:00", 72)
    miss_ident = db.timeline_ident_exists(["#999"], "demo", "2026-07-01T18:00:00+00:00", 72)
    miss_window = db.timeline_ident_exists(["#321"], "demo", "2026-07-20T18:00:00+00:00", 72)
    assert hit and not miss_ident and not miss_window
    db.close()
    _wipe(conn)
