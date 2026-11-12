"""Appending a turn to a session never rewrites another writer's row.

Every stored turn is a row keyed by ``(session_id, sequence)``, and a row carries a
project and a provenance stamp. The live writers used to work the next number out from
an earlier read and then upsert, so two writers into one session could pick the same
number and the second one replaced the first one's text (keeping its project when it
passed none). ``append_episode`` takes the number inside the INSERT, makes appenders to one session
take turns on an advisory lock, and never rewrites: a row's text, project and stamp
always belong to one write.

Runs against the shared Postgres test DB (tagged into the ``db`` xdist group via
conftest). Every session id starts with a per-test ``append-<hex>-`` prefix and each
test deletes exactly its own rows before and after.
"""

from __future__ import annotations

import threading
import time
import uuid

import psycopg
import pytest

from ingestion import db_episodes
from ingestion.db import Database
from ingestion.models import Episode
from ingestion.surfaces import OWN_SURFACE_PROBE
from tests.helpers.pre057 import pre057_schema

#: A hung thread fails the test instead of hanging the suite.
_JOIN_S = 60.0


@pytest.fixture()
def tag(conn):
    hx = uuid.uuid4().hex[:12]

    def _wipe():  # extraction_queue rows go with their episodes (ON DELETE CASCADE)
        conn.execute("DELETE FROM episodes WHERE session_id LIKE %s", (f"append-{hx}-%",))

    _wipe()
    OWN_SURFACE_PROBE.reset()
    yield hx
    _wipe()
    OWN_SURFACE_PROBE.reset()


@pytest.fixture()
def db(db_url):
    d = Database(db_url)
    yield d
    d.close()


def _row(conn, episode_id: int):
    return conn.execute("SELECT * FROM episodes WHERE id = %s", (episode_id,)).fetchone()


def _rows(conn, sid: str) -> list[tuple]:
    return conn.execute(
        "SELECT sequence, content, project, surface_id FROM episodes "
        "WHERE session_id = %s ORDER BY sequence",
        (sid,),
    ).fetchall()


def _session_lock_is_free(conn, sid: str) -> bool:
    """Whether a third connection can take the session's append lock right now."""
    return conn.execute(
        "SELECT pg_try_advisory_xact_lock(hashtext(%s))", (db_episodes._SESSION_LOCK_KEY + sid,)
    ).fetchone()[0]


def _wait_until_blocked_on_a_row(conn, thread: threading.Thread, outcome: dict) -> None:
    """Block until the thread's append is waiting on a row another transaction holds open."""
    deadline = time.monotonic() + 15
    waiting = 0
    while thread.is_alive() and time.monotonic() < deadline:
        waiting = conn.execute(
            "SELECT count(*) FROM pg_stat_activity WHERE wait_event_type = 'Lock' "
            "AND datname = current_database() AND query LIKE 'INSERT INTO episodes%'"
        ).fetchone()[0]
        if waiting:
            break
        time.sleep(0.05)
    assert thread.is_alive(), f"the append did not wait for the open row: {outcome}"
    assert waiting, "the append never showed up waiting on the open row"


def _no_upsert(monkeypatch) -> None:
    """The writers that used to number their own rows read the highest sequence and upserted
    at the next one, which lands in an empty slot and so looks like an append until another
    writer is there. Appending is the only way a live writer may write, so an upsert here
    is that old path."""

    def _boom(self, ep):
        raise AssertionError("a live writer upserted a row instead of appending it")

    monkeypatch.setattr(Database, "upsert_episode", _boom)


def test_appends_number_from_one_and_leave_earlier_rows_alone(tag, conn, db):
    sid = f"append-{tag}-a"
    # The sequence the caller passes is not used.
    first = Episode(
        session_id=sid,
        sequence=40,
        project="proj-a",
        content="first turn",
        surface_id="dev-a",
        span_id=f"claude_code:{tag}-1",
        metadata={"cwd": "/work/a"},
    )
    id1, seq1 = db.append_episode(first)
    before = _row(conn, id1)

    second = Episode(
        session_id=sid, sequence=1, project="proj-b", content="second turn", surface_id="dev-b"
    )
    id2, seq2 = db.append_episode(second)

    assert (seq1, seq2) == (1, 2)
    assert id1 != id2
    assert _row(conn, id1) == before
    assert _rows(conn, sid) == [
        (1, "first turn", "proj-a", "dev-a"),
        (2, "second turn", "proj-b", "dev-b"),
    ]


def test_an_append_into_another_session_starts_at_one(tag, db):
    ids = [
        db.append_episode(Episode(session_id=f"append-{tag}-{s}", sequence=7, content=s))
        for s in ("x", "y", "x")
    ]
    assert [seq for _, seq in ids] == [1, 1, 2]


def test_append_strips_nul_and_keeps_event_time(tag, conn, db):
    from datetime import UTC, datetime

    when = datetime(2025, 3, 14, 15, 9, 26, tzinfo=UTC)
    eid, _ = db.append_episode(
        Episode(
            session_id=f"append-{tag}-n",
            sequence=1,
            content="clean\x00text",
            human_turn="a\x00b",
            metadata={"k": "v\x00w"},
            created_at=when,
        )
    )
    row = db.get_episode(eid)
    assert row["content"] == "cleantext" and row["human_turn"] == "ab"
    assert row["metadata"] == {"k": "vw"}
    assert row["created_at"] == when


def test_the_span_index_violation_reaches_the_caller(tag, db):
    span = f"claude_code:{tag}-shared"
    db.append_episode(Episode(session_id=f"append-{tag}-a", sequence=1, content="x", span_id=span))
    with pytest.raises(psycopg.errors.UniqueViolation) as exc:
        db.append_episode(
            Episode(session_id=f"append-{tag}-b", sequence=1, content="y", span_id=span)
        )
    assert exc.value.diag.constraint_name == "episodes_span_id_idx"
    # The failed attempt left nothing behind and the connection is usable.
    _, seq = db.append_episode(Episode(session_id=f"append-{tag}-b", sequence=1, content="z"))
    assert seq == 1


def test_a_writer_that_loses_the_number_appends_after_the_winner(tag, conn, db_url):
    """The backstop, made deterministic: a writer outside the session lock (a raw INSERT)
    holds sequence 1 in an open transaction, so the append's INSERT waits on the index,
    still holding the session lock it took first. When that writer commits the INSERT
    does nothing, is run again, and the append lands at 2 with its own text, project and
    stamp; the winner's row stays."""
    sid = f"append-{tag}-c"
    outcome: dict = {}

    def _append():
        d = Database(db_url)
        try:
            outcome["ret"] = d.append_episode(
                Episode(
                    session_id=sid,
                    sequence=1,
                    project="proj-late",
                    content="late writer text",
                    surface_id="dev-late",
                )
            )
        except BaseException as exc:  # reported by the assertions below
            outcome["err"] = exc
        finally:
            d.close()

    winner = psycopg.connect(db_url, autocommit=False)
    thread = threading.Thread(target=_append, daemon=True)
    try:
        winner.execute(
            "INSERT INTO episodes (session_id, sequence, project, content, surface_id) "
            "VALUES (%s, 1, 'proj-first', 'first writer text', 'dev-first')",
            (sid,),
        )
        thread.start()
        _wait_until_blocked_on_a_row(conn, thread, outcome)
        # The lock was taken in the transaction the INSERT is waiting in. Taken in a block
        # of its own it would already be released, and nothing else here would notice.
        assert not _session_lock_is_free(conn, sid), "the session lock was not held at the INSERT"
        winner.commit()
    finally:
        winner.rollback()
        winner.close()
    thread.join(timeout=15)
    assert not thread.is_alive(), "the append hung after the other writer committed"

    assert "err" not in outcome, outcome.get("err")
    assert outcome["ret"][1] == 2
    assert _rows(conn, sid) == [
        (1, "first writer text", "proj-first", "dev-first"),
        (2, "late writer text", "proj-late", "dev-late"),
    ]


def test_the_pre057_fallback_takes_the_session_lock_again(tag, conn, db_url):
    """The probe said the provenance column exists and it is gone (schema 057 rolled back
    under a running process): the stamped INSERT fails, its transaction and the lock with
    it, and the pre-057 INSERT runs in a new one. That one must hold the lock too, so the
    open row it then waits on is seen with the lock still taken."""
    sid = f"append-{tag}-f"
    outcome: dict = {}

    with pre057_schema(conn, db_url) as (url, _schema):

        def _append():
            d = Database(url)
            try:
                outcome["ret"] = d.append_episode(Episode(session_id=sid, sequence=1, content="x"))
            except BaseException as exc:  # reported by the assertions below
                outcome["err"] = exc
            finally:
                d.close()

        OWN_SURFACE_PROBE._ok[url] = True
        winner = psycopg.connect(url, autocommit=False)
        thread = threading.Thread(target=_append, daemon=True)
        try:
            winner.execute(
                "INSERT INTO episodes (session_id, sequence, content) VALUES (%s, 1, 'held')",
                (sid,),
            )
            thread.start()
            _wait_until_blocked_on_a_row(conn, thread, outcome)
            assert not _session_lock_is_free(conn, sid), "the fallback ran without the lock"
            winner.commit()
        finally:
            winner.rollback()
            winner.close()
        thread.join(timeout=15)
        assert not thread.is_alive(), "the append hung after the other writer committed"
        assert "err" not in outcome, outcome.get("err")
        assert outcome["ret"][1] == 2


def test_many_appends_at_once_get_distinct_numbers_and_keep_their_own_rows(tag, conn, db_url):
    """Six writers append flat out into one session, ten each, after one barrier. They take
    turns on the session lock, so none loses a number and none runs out of attempts."""
    sid = f"append-{tag}-m"
    writers, per_writer = 6, 10
    start = threading.Barrier(writers)
    results: list[list[tuple[int, int]]] = [[] for _ in range(writers)]
    errors: list[BaseException] = []

    def _writer(w: int) -> None:
        d = Database(db_url)
        try:
            start.wait(timeout=30)
            for n in range(per_writer):
                results[w].append(
                    d.append_episode(
                        Episode(
                            session_id=sid,
                            sequence=1,
                            project=f"proj-{w}",
                            content=f"writer {w} turn {n}",
                            surface_id=f"dev-{w}",
                        )
                    )
                )
        except BaseException as exc:
            errors.append(exc)
        finally:
            d.close()

    threads = [threading.Thread(target=_writer, args=(w,), daemon=True) for w in range(writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=_JOIN_S)
    assert not any(t.is_alive() for t in threads), "an append hung"
    assert not errors, errors

    total = writers * per_writer
    returned = sorted(seq for per in results for _, seq in per)
    assert returned == list(range(1, total + 1))
    stored = {r[0]: r[1:] for r in _rows(conn, sid)}
    assert sorted(stored) == returned
    for w, per in enumerate(results):
        for n, (_, seq) in enumerate(per):
            assert stored[seq] == (f"writer {w} turn {n}", f"proj-{w}", f"dev-{w}")


def test_an_append_waits_on_the_session_lock_and_lands_after_the_holder(
    tag, conn, db_url, monkeypatch
):
    """Another appender holds the session's advisory lock in an open transaction and has
    written sequence 1 under it. The append waits on the lock (not on the index), and once
    the holder commits it lands at 2 on its first try: its INSERT is taken after the lock
    is granted, so it already sees the holder's row."""
    sid = f"append-{tag}-l"
    tries: list[int] = []
    real = Database._run_episode_sql

    def _counting(self, *args, **kwargs):
        tries.append(1)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Database, "_run_episode_sql", _counting)
    outcome: dict = {}

    def _append():
        d = Database(db_url)
        try:
            outcome["ret"] = d.append_episode(
                Episode(session_id=sid, sequence=1, project="proj-late", content="late text")
            )
        except BaseException as exc:  # reported by the assertions below
            outcome["err"] = exc
        finally:
            d.close()

    holder = psycopg.connect(db_url, autocommit=False)
    thread = threading.Thread(target=_append, daemon=True)
    try:
        holder.execute(
            "SELECT pg_advisory_xact_lock(hashtext(%s))", (db_episodes._SESSION_LOCK_KEY + sid,)
        )
        holder.execute(
            "INSERT INTO episodes (session_id, sequence, project, content) "
            "VALUES (%s, 1, 'proj-holder', 'holder text')",
            (sid,),
        )
        thread.start()
        deadline = time.monotonic() + 15
        waiting = 0
        while thread.is_alive() and time.monotonic() < deadline:
            waiting = conn.execute(
                "SELECT count(*) FROM pg_stat_activity WHERE wait_event_type = 'Lock' "
                "AND wait_event = 'advisory' AND datname = current_database()"
            ).fetchone()[0]
            if waiting:
                break
            time.sleep(0.05)
        assert thread.is_alive(), f"the append did not wait for the lock: {outcome}"
        assert waiting, "the append never showed up waiting on the advisory lock"
        holder.commit()
    finally:
        holder.rollback()
        holder.close()
    thread.join(timeout=15)
    assert not thread.is_alive(), "the append hung after the holder committed"

    assert "err" not in outcome, outcome.get("err")
    assert outcome["ret"][1] == 2
    assert len(tries) == 1
    assert [r[:3] for r in _rows(conn, sid)] == [
        (1, "holder text", "proj-holder"),
        (2, "late text", "proj-late"),
    ]


def test_the_append_gives_up_after_its_bound_and_names_the_session(tag, db, monkeypatch):
    calls: list[dict] = []

    def _never_a_row(self, pre057, stamped, params, lock_session=False):
        calls.append(params)
        return None

    monkeypatch.setattr(Database, "_run_episode_sql", _never_a_row)
    sid = f"append-{tag}-g"
    with pytest.raises(RuntimeError) as exc:
        db.append_episode(Episode(session_id=sid, sequence=1, content="x"))
    assert len(calls) == db_episodes._APPEND_ATTEMPTS == 8
    assert sid in str(exc.value) and "colliding" in str(exc.value)


# ---------------------------------------------------------------------------
# Backfill: rows are written in parsed order, after what the session already holds
# ---------------------------------------------------------------------------


def _parsed(sid: str, seq: int, span: str | None, content: str, project: str | None = None):
    return Episode(session_id=sid, sequence=seq, span_id=span, content=content, project=project)


def _queued(conn, sid: str) -> int:
    return conn.execute(
        "SELECT count(*) FROM extraction_queue WHERE session_id = %s", (sid,)
    ).fetchone()[0]


def test_write_backfill_session_skips_known_spans_and_appends_in_order(tag, conn, db, monkeypatch):
    from ingestion.backfill import write_backfill_session

    sid = f"append-{tag}-b1"
    held, _ = db.append_episode(_parsed(sid, 1, f"imp:{tag}:held", "held turn", "proj-held"))
    before = _row(conn, held)
    _no_upsert(monkeypatch)

    parsed = [
        _parsed(sid, 3, f"imp:{tag}:c", "turn c"),
        _parsed(sid, 1, f"imp:{tag}:held", "held turn again"),
        _parsed(sid, 2, f"imp:{tag}:b", "turn b"),
    ]
    assert write_backfill_session(db, sid, parsed) == 2

    assert [r[:3] for r in _rows(conn, sid)] == [
        (1, "held turn", "proj-held"),
        (2, "turn b", None),
        (3, "turn c", None),
    ]
    assert _row(conn, held) == before
    assert _queued(conn, sid) == 2


def test_write_session_episodes_skips_known_spans_and_appends_in_order(tag, conn, db, monkeypatch):
    from ingestion.backfill import _write_session_episodes

    sid = f"append-{tag}-b2"
    held, _ = db.append_episode(
        _parsed(sid, 1, f"claude_code:{tag}-held", "held turn", "proj-held")
    )
    before = _row(conn, held)
    _no_upsert(monkeypatch)

    parsed = [
        _parsed(sid, 3, f"claude_code:{tag}-c", "turn c", "proj-c"),
        _parsed(sid, 1, f"claude_code:{tag}-held", "held turn again"),
        _parsed(sid, 2, f"claude_code:{tag}-b", "turn b"),
    ]
    assert _write_session_episodes(db, sid, parsed, "proj-default") == 2

    assert [r[:3] for r in _rows(conn, sid)] == [
        (1, "held turn", "proj-held"),
        (2, "turn b", "proj-default"),
        (3, "turn c", "proj-c"),
    ]
    assert _row(conn, held) == before
    assert _queued(conn, sid) == 2


def test_write_session_episodes_copes_with_a_row_that_has_no_span_id(tag, conn, db):
    """A remember row has no span id. The sweep of a session that holds one must read past
    it, then append after it."""
    from ingestion.backfill import _write_session_episodes

    sid = f"append-{tag}-b3"
    db.append_episode(_parsed(sid, 1, None, "a manual note"))
    db.append_episode(_parsed(sid, 2, f"claude_code:{tag}-held", "held turn"))

    parsed = [
        _parsed(sid, 1, f"claude_code:{tag}-held", "held turn again"),
        _parsed(sid, 2, f"claude_code:{tag}-new", "new turn"),
    ]
    assert _write_session_episodes(db, sid, parsed, "proj-default") == 1

    assert [r[:2] for r in _rows(conn, sid)] == [
        (1, "a manual note"),
        (2, "held turn"),
        (3, "new turn"),
    ]
