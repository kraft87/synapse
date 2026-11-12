"""POST /ingest: a turn whose span_id is already stored under ANOTHER session is skipped.

``episodes_span_id_idx`` is global (UNIQUE on span_id), but the route's own dedup only
loads the incoming session's span index. A transcript that replays a turn of another
session (a resumed or forked session, its records keeping their uuids) therefore reached
the INSERT, raised UniqueViolation and 500'd the whole POST. The hook never advances its
cursor on a failure, so every later turn of that transcript was lost on every retry: in
prod, 15 POSTs from 2026-10-30 to 11-05 all failed on the one span
``jsonl:6033958e-...`` (its spelling then), stored once as session 24d5dada's turn 12.

Two layers, one test each: the route probes the span globally before writing, and an
append that still hits the span index (another writer committed the span between the
probe and the INSERT) skips that one turn instead of failing the batch.

Runs against the shared Postgres test DB (tagged into the ``db`` xdist group via
conftest). Every session id starts with a per-test ``spantest-<hex>-`` prefix and each
test deletes exactly its own rows before and after.
"""

from __future__ import annotations

import logging
import uuid

import pytest


@pytest.fixture()
def tag(conn):
    hx = uuid.uuid4().hex[:12]

    def _wipe():  # extraction_queue rows go with their episodes (ON DELETE CASCADE)
        conn.execute("DELETE FROM episodes WHERE session_id LIKE %s", (f"spantest-{hx}-%",))

    _wipe()
    yield hx
    _wipe()


def _client(db_url):
    from fastmcp import FastMCP
    from starlette.testclient import TestClient

    from mcp_server.ingest_route import register

    m = FastMCP("test-ingest-span-collision")
    register(m, lambda: db_url, lambda r: True)
    return TestClient(m.http_app())


def _post(db_url, records: list[dict]):
    return _client(db_url).post("/ingest", json={"records": records})


def _turn(sid: str, cwd: str, text: str, last_uuid: str) -> list[dict]:
    """One Claude Code turn; its span_id is ``claude_code:<last_uuid>``."""
    base = {"sessionId": sid, "cwd": cwd, "timestamp": "2026-10-30T13:41:00Z"}
    return [
        {**base, "type": "user", "uuid": f"u-{last_uuid}", "message": {"role": "user", "content": f"q {text}"}},
        {
            **base,
            "type": "assistant",
            "uuid": last_uuid,
            "message": {"role": "assistant", "content": [{"type": "text", "text": f"a {text}"}]},
        },
    ]  # fmt: skip


def _rows(conn, sid: str) -> list[tuple[int, str]]:
    return [
        (r[0], r[1])
        for r in conn.execute(
            "SELECT sequence, span_id FROM episodes WHERE session_id = %s ORDER BY sequence",
            (sid,),
        ).fetchall()
    ]


def _seed_and_replay(conn, db_url, hx: str) -> tuple[str, str, str, list[dict]]:
    """Store one turn under session A, then build a batch for session B whose middle turn
    carries A's span_id. Its content and folder differ from A's, so the cross-session
    content guard does not absorb it: the span index is the only thing that matches."""
    sid_a, sid_b = f"spantest-{hx}-a", f"spantest-{hx}-b"
    shared = f"shared-{hx}"
    seeded = _post(db_url, _turn(sid_a, f"/work/{hx}/alpha", f"original {hx}", shared))
    assert seeded.status_code == 200 and seeded.json()["ingested"] == 1, seeded.text

    batch = (
        _turn(sid_b, f"/work/{hx}/beta", f"first {hx}", f"b1-{hx}")
        + _turn(sid_b, f"/work/{hx}/beta", f"replayed {hx}", shared)
        + _turn(sid_b, f"/work/{hx}/beta", f"third {hx}", f"b3-{hx}")
    )
    return sid_a, sid_b, f"claude_code:{shared}", batch


def _assert_skipped(conn, r, hx: str, sid_a: str, sid_b: str, span: str) -> None:
    assert r.status_code == 200, r.text
    assert r.json()["ingested"] == 2, r.json()
    # The third count is what tells this skip apart from the content guard's.
    assert r.json()["span_collisions"] == 1, r.json()
    # The turns either side of the collision landed, numbered with no gap.
    assert _rows(conn, sid_b) == [(1, f"claude_code:b1-{hx}"), (2, f"claude_code:b3-{hx}")]
    # The stored turn is untouched and still the only row holding that span.
    holders = conn.execute(
        "SELECT session_id, sequence FROM episodes WHERE span_id = %s", (span,)
    ).fetchall()
    assert [tuple(h) for h in holders] == [(sid_a, 1)]


def test_a_span_stored_under_another_session_is_skipped_not_a_500(
    tag, conn, db_url, caplog, monkeypatch
):
    from ingestion.db import Database

    sid_a, sid_b, span, batch = _seed_and_replay(conn, db_url, tag)
    # Every span the route tries to INSERT. The except leg alone would also return 200,
    # so this is what holds the probe to its job: the stored span never reaches the INSERT.
    attempted: list[str] = []
    real_append = Database.append_episode

    def _spy(self, ep):
        attempted.append(ep.span_id)
        return real_append(self, ep)

    monkeypatch.setattr(Database, "append_episode", _spy)

    with caplog.at_level(logging.WARNING, logger="mcp_server.ingest_route"):
        r = _post(db_url, batch)

    _assert_skipped(conn, r, tag, sid_a, sid_b, span)
    assert attempted == [f"claude_code:b1-{tag}", f"claude_code:b3-{tag}"]
    warnings = [m for m in caplog.records if m.levelno == logging.WARNING]
    assert len(warnings) == 1, [m.getMessage() for m in warnings]
    assert sid_b in warnings[0].getMessage() and span in warnings[0].getMessage()

    # The client re-ships the same tail until a POST succeeds; a re-ship is a no-op.
    again = _post(db_url, batch)
    assert again.status_code == 200 and again.json()["ingested"] == 0, again.text
    assert len(_rows(conn, sid_b)) == 2


def test_a_span_written_between_the_probe_and_the_insert_skips_one_turn(
    tag, conn, db_url, monkeypatch
):
    """The race: another writer commits the span after the probe has answered "not
    stored". The INSERT then hits the span index; that turn alone is skipped."""
    from ingestion.db import Database

    sid_a, sid_b, span, batch = _seed_and_replay(conn, db_url, tag)
    monkeypatch.setattr(Database, "span_id_exists", lambda self, span_id: False)

    r = _post(db_url, batch)

    _assert_skipped(conn, r, tag, sid_a, sid_b, span)


def test_a_row_another_writer_adds_mid_batch_is_never_overwritten(tag, conn, db_url, monkeypatch):
    """Another writer adds a row to the session after the route has read its span index.
    The batch's turns land after that row; the row keeps its own text, project and stamp."""
    from ingestion.db import Database
    from ingestion.models import Episode

    sid = f"spantest-{tag}-a"
    batch = _turn(sid, f"/work/{tag}/alpha", f"first {tag}", f"m1-{tag}") + _turn(
        sid, f"/work/{tag}/alpha", f"second {tag}", f"m2-{tag}"
    )
    real_index = Database.get_session_span_index

    def _index_then_another_writer(self, session_id):
        out = real_index(self, session_id)
        other = Database(db_url)
        try:
            other.append_episode(
                Episode(
                    session_id=session_id,
                    sequence=1,
                    project="other-project",
                    content="the other writer's text",
                    surface_id="dev-other",
                )
            )
        finally:
            other.close()
        return out

    monkeypatch.setattr(Database, "get_session_span_index", _index_then_another_writer)

    r = _post(db_url, batch)

    assert r.status_code == 200 and r.json()["ingested"] == 2, r.text
    assert conn.execute(
        "SELECT sequence, span_id, content, project, surface_id FROM episodes "
        "WHERE session_id = %s ORDER BY sequence",
        (sid,),
    ).fetchall()[0] == (1, None, "the other writer's text", "other-project", "dev-other")
    assert _rows(conn, sid) == [
        (1, None),
        (2, f"claude_code:m1-{tag}"),
        (3, f"claude_code:m2-{tag}"),
    ]
