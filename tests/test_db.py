"""Integration tests for the DB layer (runs against synapse_test)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ingestion.db import Database
from ingestion.db_queue import CHUNK_ID_PROBE
from ingestion.models import Episode, ExtractionItem


@pytest.fixture(scope="module")
def db(db_url):
    d = Database(db_url)
    yield d
    d.close()


@pytest.fixture(autouse=True)
def _clean(clean_tables):
    """Auto-clean tables before every test in this module."""


class TestEpisodeWriter:
    def test_insert_episode(self, db):
        ep = Episode(
            session_id="db-test-1",
            sequence=1,
            content="Testing the DB layer",
            platform="claude_code",
            model="claude-sonnet-4-6",
            human_turn="write some tests",
            assistant_turn="here are the tests",
        )
        ep_id = db.upsert_episode(ep)
        assert isinstance(ep_id, int)
        assert ep_id > 0

    def test_nul_bytes_stripped_before_insert(self, db):
        # TEXT columns reject NUL bytes and jsonb rejects the u0000 escape —
        # a stray NUL in a transcript must not fail the INSERT.
        ep = Episode(
            session_id="db-test-nul",
            sequence=1,
            content="clean\x00text",
            human_turn="a\x00b",
            metadata={"k": "v\x00w", "nested": ["x\x00y"]},
        )
        ep_id = db.upsert_episode(ep)
        row = db.get_episode(ep_id)
        assert row["content"] == "cleantext"
        assert row["human_turn"] == "ab"
        assert row["metadata"] == {"k": "vw", "nested": ["xy"]}

    def test_upsert_is_idempotent(self, db):
        ep = Episode(session_id="db-test-2", sequence=1, content="idempotent test")
        id1 = db.upsert_episode(ep)
        id2 = db.upsert_episode(ep)  # same session_id + sequence
        assert id1 == id2

    def test_upsert_updates_content(self, db):
        ep = Episode(session_id="db-test-3", sequence=1, content="original content")
        db.upsert_episode(ep)
        ep2 = Episode(session_id="db-test-3", sequence=1, content="updated content")
        ep_id = db.upsert_episode(ep2)
        row = db.get_episode(ep_id)
        assert row["content"] == "updated content"

    def test_created_at_is_event_time_not_ingest_time(self, db):
        """Issue #43: the transcript's own timestamp must survive to the row.
        Imported/backfilled episodes carry when the conversation HAPPENED;
        dropping it re-dates history to import day and poisons served dates,
        recency ranking, and fact t_valid."""
        when = datetime(2025, 3, 14, 15, 9, 26, tzinfo=UTC)
        ep = Episode(session_id="db-test-ts", sequence=1, content="old turn", created_at=when)
        row = db.get_episode(db.upsert_episode(ep))
        assert row["created_at"] == when
        # re-upsert with a corrected ts wins; without one the stored ts survives
        row = db.get_episode(
            db.upsert_episode(Episode(session_id="db-test-ts", sequence=1, content="old turn v2"))
        )
        assert row["created_at"] == when

    def test_created_at_defaults_to_now_for_live_ingest(self, db):
        row = db.get_episode(
            db.upsert_episode(Episode(session_id="db-test-now", sequence=1, content="live"))
        )
        assert abs((datetime.now(UTC) - row["created_at"]).total_seconds()) < 60

    def test_insert_multiple_sequences(self, db):
        for seq in range(1, 6):
            db.upsert_episode(
                Episode(
                    session_id="db-test-multi",
                    sequence=seq,
                    content=f"turn {seq}",
                )
            )
        episodes = db.get_session_episodes("db-test-multi")
        assert len(episodes) == 5
        assert [e["sequence"] for e in episodes] == [1, 2, 3, 4, 5]

    def test_metadata_roundtrip(self, db):
        ep = Episode(
            session_id="db-test-meta",
            sequence=1,
            content="meta test",
            metadata={"cwd": "/home/user/services/synapse", "exit_code": 0},
        )
        ep_id = db.upsert_episode(ep)
        row = db.get_episode(ep_id)
        assert row["metadata"]["cwd"] == "/home/user/services/synapse"

    def test_span_index_returns_span_ids_and_max_seq(self, db):
        """/ingest dedups by span_id and appends at max(sequence)+1 — the span
        index must report exactly the stored span_ids and the high-water sequence."""
        for seq in (1, 2, 3):
            db.upsert_episode(
                Episode(
                    session_id="db-test-span",
                    sequence=seq,
                    content=f"turn {seq}",
                    span_id=f"jsonl:u{seq}",
                )
            )
        span_ids, max_seq = db.get_session_span_index("db-test-span")
        assert span_ids == {"jsonl:u1", "jsonl:u2", "jsonl:u3"}
        assert max_seq == 3

    def test_span_index_empty_session(self, db):
        """An unseen session reports no span_ids and a 0 high-water mark, so the
        first new turn appends at sequence 1."""
        span_ids, max_seq = db.get_session_span_index("db-test-span-empty")
        assert span_ids == set()
        assert max_seq == 0

    def test_content_dup_exists_cross_session_same_project(self, db):
        """A retried session re-ships identical content under a new session id and
        new span id — the content probe must flag it so /ingest can skip it."""
        db.upsert_episode(
            Episode(
                session_id="db-test-dup-a",
                sequence=1,
                project="proj-x",
                content="identical replayed turn",
                span_id="jsonl:dup-a1",
            )
        )
        assert db.content_dup_exists("proj-x", "identical replayed turn") is True
        assert db.content_dup_exists("proj-x", "a different turn") is False

    def test_content_dup_scoped_to_project(self, db):
        """Identical content in ANOTHER project is not a replay — the probe is
        project-scoped (NULL project matches only NULL)."""
        db.upsert_episode(
            Episode(
                session_id="db-test-dup-b",
                sequence=1,
                project="proj-x",
                content="same words, other silo",
                span_id="jsonl:dup-b1",
            )
        )
        assert db.content_dup_exists("proj-y", "same words, other silo") is False
        assert db.content_dup_exists(None, "same words, other silo") is False
        db.upsert_episode(
            Episode(
                session_id="db-test-dup-c",
                sequence=1,
                project=None,
                content="null-project turn",
                span_id="jsonl:dup-c1",
            )
        )
        assert db.content_dup_exists(None, "null-project turn") is True

    def test_content_dup_matches_the_same_folder_under_an_older_label(self, db):
        """A turn first stored under the label an older plugin gave its folder (a
        subfolder name, a whole Windows path) and replayed with today's label is
        still a replay: same content, same working directory."""
        cwd = "C:\\Users\\someone\\src\\Acme\\tickets"
        db.upsert_episode(
            Episode(
                session_id="db-test-dup-d",
                sequence=1,
                project="tickets",
                content="replayed across the label change",
                span_id="jsonl:dup-d1",
                metadata={"cwd": cwd},
            )
        )
        text = "replayed across the label change"
        assert db.content_dup_exists("Acme", text, cwd) is True
        # another folder, or no folder at all, is still project-scoped
        assert db.content_dup_exists("Acme", text, "C:\\Users\\someone\\src\\Other") is False
        assert db.content_dup_exists("Acme", text) is False
        assert db.content_dup_exists("Acme", text, "") is False


class TestTimelineDedup:
    """timeline_near_candidates + bump_timeline_reported (schema 037)."""

    PROJ = "db-test-tl-dedup"

    @pytest.fixture(autouse=True)
    def _clean_timeline(self, conn):
        conn.execute("DELETE FROM timeline_events WHERE project = %s", (self.PROJ,))
        yield
        conn.execute("DELETE FROM timeline_events WHERE project = %s", (self.PROJ,))

    def _vec(self, hot: int):
        from ingestion.db import _EMBED_DIMS

        v = [0.0] * _EMBED_DIMS
        v[hot] = 1.0
        return v

    def _insert(self, db, ref: str, vec, t_valid="2026-07-01T10:00:00+00:00"):
        db.insert_timeline_event(
            t_valid=t_valid,
            fact=f"event {ref}",
            source="chat",
            source_ref=ref,
            project=self.PROJ,
            salience=1,
            embedding=vec,
            embed_model="test",
        )

    def test_near_candidates_excludes_siblings_and_far(self, db):
        near = self._vec(0)
        self._insert(db, "ep:900", near)  # near, different turn -> candidate
        self._insert(db, "ep:901", near)  # the new event's own turn -> excluded
        self._insert(db, "ep:901#2", near)  # sibling of the new turn -> excluded
        self._insert(db, "ep:902", self._vec(1))  # orthogonal (dist 1.0) -> over max_dist

        cands = db.timeline_near_candidates(
            near, self.PROJ, "2026-07-02T10:00:00+00:00", exclude_episode_ref="ep:901"
        )
        assert [c["source_ref"] for c in cands] == ["ep:900"]
        assert cands[0]["dist"] < 0.01

    def test_near_candidates_respects_time_window(self, db):
        near = self._vec(0)
        self._insert(db, "ep:910", near, t_valid="2026-05-01T10:00:00+00:00")  # 2 months back
        cands = db.timeline_near_candidates(
            near, self.PROJ, "2026-07-02T10:00:00+00:00", exclude_episode_ref="ep:999"
        )
        assert cands == []

    def test_bump_keeps_earliest_t_valid(self, db, conn):
        self._insert(db, "ep:920", self._vec(0), t_valid="2026-07-01T10:00:00+00:00")
        row = conn.execute("SELECT id FROM timeline_events WHERE source_ref = 'ep:920'").fetchone()
        eid = row[0] if not isinstance(row, dict) else row["id"]

        # re-tell resolves an EARLIER date -> t_valid corrects backward
        db.bump_timeline_reported(eid, "2026-06-28T12:00:00+00:00")
        # re-tell with a LATER date -> earliest wins, count still increments
        db.bump_timeline_reported(eid, "2026-07-03T12:00:00+00:00")

        r = conn.execute(
            "SELECT reported_count, t_valid FROM timeline_events WHERE id = %s", (eid,)
        ).fetchone()
        cnt, tv = (r[0], r[1]) if not isinstance(r, dict) else (r["reported_count"], r["t_valid"])
        assert cnt == 3
        assert str(tv).startswith("2026-06-28")


def _pending(db, limit: int) -> list[dict]:
    """The queue's pending rows, oldest first."""
    with db._conn() as conn:
        return conn.execute(
            "SELECT * FROM extraction_queue WHERE status = 'pending' "
            "ORDER BY priority, enqueued_at LIMIT %s",
            (limit,),
        ).fetchall()


class TestExtractionQueue:
    def test_enqueue_episode(self, db):
        ep_id = db.upsert_episode(Episode(session_id="eq-test-1", sequence=1, content="extract me"))
        item = ExtractionItem(
            episode_id=ep_id,
            content="extract me",
            content_type="episode",
            project="synapse",
        )
        db.enqueue_extraction(item)
        pending = _pending(db, 10)
        assert any(p["episode_id"] == ep_id for p in pending)

    def test_an_episode_item_stores_no_text(self, db):
        ep_id = db.upsert_episode(Episode(session_id="eq-notext", sequence=1, content="the turn"))
        db.enqueue_extraction(ExtractionItem(episode_id=ep_id, content_type="episode"))
        (row,) = [p for p in _pending(db, 100) if p["episode_id"] == ep_id]
        assert row["content"] == ""

    def test_an_item_that_names_no_source_is_refused(self, db):
        with pytest.raises(ValueError, match="needs an episode_id"):
            db.enqueue_extraction(
                ExtractionItem(session_id="eq-sum-1", content="text", content_type="summary")
            )
        with pytest.raises(ValueError, match="chunk_id or its content"):
            db.enqueue_extraction(ExtractionItem(session_id="eq-c", content_type="chunk"))

    def test_no_duplicate_enqueue(self, db):
        ep_id = db.upsert_episode(Episode(session_id="eq-dedup", sequence=1, content="no dups"))
        item = ExtractionItem(episode_id=ep_id, content="no dups", content_type="episode")
        db.enqueue_extraction(item)
        db.enqueue_extraction(item)  # second call should be a no-op
        pending = _pending(db, 100)
        matches = [p for p in pending if p["episode_id"] == ep_id]
        assert len(matches) == 1

    def test_mark_done(self, db):
        ep_id = db.upsert_episode(Episode(session_id="eq-done", sequence=1, content="mark me done"))
        item = ExtractionItem(episode_id=ep_id, content="mark me done", content_type="episode")
        db.enqueue_extraction(item)
        pending = _pending(db, 10)
        queue_id = next(p["id"] for p in pending if p["episode_id"] == ep_id)
        db.mark_extraction_done(queue_id)
        still_pending = _pending(db, 100)
        assert not any(p["id"] == queue_id for p in still_pending)

    # --- failed items are retried, not buried ---------------------------------
    #
    # `failed` used to be terminal: claim_pending_extractions only ever looked at
    # `pending`, so an item that failed during an outage (or against a wrong model id)
    # stayed graph-less forever no matter what was fixed afterwards.

    def _queued(self, db, tag: str) -> int:
        ep_id = db.upsert_episode(Episode(session_id=tag, sequence=1, content=f"content {tag}"))
        db.enqueue_extraction(
            ExtractionItem(episode_id=ep_id, content=f"content {tag}", content_type="episode")
        )
        return next(p["id"] for p in _pending(db, 100) if p["episode_id"] == ep_id)

    def _age(self, db, queue_id: int, *, minutes: int, attempts: int = 1) -> None:
        """Backdate a failed row so the retry window is expressible without sleeping."""
        with db._conn() as conn:
            conn.execute(
                "UPDATE extraction_queue SET processed_at = now() - make_interval(mins => %s), "
                "attempts = %s WHERE id = %s",
                (minutes, attempts, queue_id),
            )

    def test_a_cooled_off_failure_is_reclaimed(self, db):
        qid = self._queued(db, "eq-retry")
        db.claim_pending_extractions(limit=10)
        db.mark_extraction_failed(qid, error="model returned no output on any attempt")
        assert qid not in [r["id"] for r in db.claim_pending_extractions(limit=10)]  # too fresh
        self._age(db, qid, minutes=Database.FAILED_RETRY_MINUTES + 1)
        assert qid in [r["id"] for r in db.claim_pending_extractions(limit=10)]

    def test_an_exhausted_failure_is_left_alone(self, db):
        """A genuinely poisonous item must stop costing LLM calls."""
        qid = self._queued(db, "eq-exhausted")
        db.claim_pending_extractions(limit=10)
        db.mark_extraction_failed(qid, error="nope")
        self._age(db, qid, minutes=600, attempts=Database.MAX_EXTRACTION_ATTEMPTS)
        assert db.claim_pending_extractions(limit=10) == []

    def test_pending_is_selected_before_retries(self, db):
        """A retry backlog must never starve new ingest: with room for one item, the
        pending row wins even though the failed one is older."""
        old_fail = self._queued(db, "eq-order-failed")
        db.claim_pending_extractions(limit=10)
        db.mark_extraction_failed(old_fail, error="nope")
        self._age(db, old_fail, minutes=600)
        fresh = self._queued(db, "eq-order-pending")
        assert [r["id"] for r in db.claim_pending_extractions(limit=1)] == [fresh]
        assert [r["id"] for r in db.claim_pending_extractions(limit=1)] == [old_fail]

    def test_mark_failed_with_error(self, db):
        ep_id = db.upsert_episode(Episode(session_id="eq-fail", sequence=1, content="fail me"))
        item = ExtractionItem(episode_id=ep_id, content="fail me", content_type="episode")
        db.enqueue_extraction(item)
        pending = _pending(db, 10)
        queue_id = next(p["id"] for p in pending if p["episode_id"] == ep_id)
        db.mark_extraction_failed(queue_id, error="Connection refused")
        with db._conn() as conn:
            row = conn.execute(
                "SELECT status, error FROM extraction_queue WHERE id=%s", (queue_id,)
            ).fetchone()
        assert row["status"] == "failed"
        assert "Connection refused" in row["error"]


class TestPendingChunks:
    """Which sessions the poller re-reads to chunk: a new complete window, or a quiet tail."""

    @pytest.fixture(autouse=True)
    def _no_chunks(self, conn):
        conn.execute("DELETE FROM chunks WHERE session_id LIKE 'pc-%'")
        yield
        conn.execute("DELETE FROM chunks WHERE session_id LIKE 'pc-%'")

    def _session(self, db, sid: str, n: int, *, idle: bool, manual_last: bool = False) -> None:
        last = datetime.now(UTC) - (timedelta(hours=3) if idle else timedelta(minutes=1))
        for seq in range(1, n + 1):
            db.upsert_episode(
                Episode(
                    session_id=sid,
                    sequence=seq,
                    content=f"turn {seq}",
                    source="manual" if manual_last and seq == n else "hook",
                    created_at=last - timedelta(minutes=n - seq),
                )
            )

    def _chunk(self, db, sid: str, start: int, end: int) -> int | None:
        return db.upsert_chunk(sid, start, end, [start, end], f"{start}-{end}", None)

    def test_upsert_chunk_returns_the_id_and_none_for_an_existing_window(self, db):
        first = self._chunk(db, "pc-id", 1, 4)
        assert isinstance(first, int)
        assert self._chunk(db, "pc-id", 1, 4) is None

    def test_a_live_session_waits_for_a_full_window(self, db):
        self._session(db, "pc-live-3", 3, idle=False)
        self._session(db, "pc-live-4", 4, idle=False)
        self._session(db, "pc-live-tail", 5, idle=False)
        self._chunk(db, "pc-live-tail", 1, 4)
        pending = db.sessions_with_pending_chunks()
        assert "pc-live-4" in pending
        assert "pc-live-3" not in pending and "pc-live-tail" not in pending

    def test_an_idle_session_is_pending_until_its_tail_is_chunked(self, db):
        self._session(db, "pc-idle-short", 2, idle=True)
        self._session(db, "pc-idle-tail", 5, idle=True)
        self._chunk(db, "pc-idle-tail", 1, 4)
        assert {"pc-idle-short", "pc-idle-tail"} <= set(db.sessions_with_pending_chunks())
        self._chunk(db, "pc-idle-short", 1, 2)
        self._chunk(db, "pc-idle-tail", 4, 5)
        pending = db.sessions_with_pending_chunks()
        assert "pc-idle-short" not in pending and "pc-idle-tail" not in pending

    def test_a_remember_archive_does_not_keep_an_idle_session_pending(self, db):
        self._session(db, "pc-idle-manual", 5, idle=True, manual_last=True)
        self._chunk(db, "pc-idle-manual", 1, 4)
        assert "pc-idle-manual" not in db.sessions_with_pending_chunks()

    def test_the_session_read_leaves_out_the_vector_and_the_duplicated_text(self, db):
        self._session(db, "pc-cols", 1, idle=False)
        (row,) = db.get_session_episodes("pc-cols")
        assert set(row) == {"id", "sequence", "content", "project", "source", "created_at"}


class TestChunkQueueItems:
    """A chunk's queue row names the chunk (schema: extraction_queue.chunk_id) and stores
    no text; before that migration it keeps the text and is found by it."""

    @pytest.fixture(autouse=True)
    def _chunks(self, conn):
        conn.execute("DELETE FROM chunks WHERE session_id LIKE 'cq-%'")
        CHUNK_ID_PROBE.reset()
        yield
        conn.execute("DELETE FROM chunks WHERE session_id LIKE 'cq-%'")
        CHUNK_ID_PROBE.reset()

    def _chunk(self, db, sid: str = "cq-1", text: str = "one\n\n---\n\ntwo") -> int:
        chunk_id = db.upsert_chunk(sid, 1, 4, [1, 2], text, "synapse")
        assert chunk_id is not None
        return chunk_id

    def _rows(self, conn, sid: str):
        return conn.execute(
            "SELECT chunk_id, content, status FROM extraction_queue WHERE session_id = %s", (sid,)
        ).fetchall()

    def test_the_row_names_the_chunk_and_stores_no_text(self, db, conn):
        chunk_id = self._chunk(db)
        item = ExtractionItem(
            session_id="cq-1", chunk_id=chunk_id, content="one", content_type="chunk"
        )
        db.enqueue_extraction(item)
        db.enqueue_extraction(item)  # a double run is a no-op
        assert self._rows(conn, "cq-1") == [(chunk_id, "", "pending")]

    def test_a_finished_chunk_is_not_queued_again(self, db, conn):
        chunk_id = self._chunk(db)
        item = ExtractionItem(session_id="cq-1", chunk_id=chunk_id, content_type="chunk")
        db.enqueue_extraction(item)
        (queue_id,) = [r["id"] for r in db.claim_pending_extractions(limit=10)]
        db.mark_extraction_done(queue_id)
        db.enqueue_extraction(item)
        assert [r[2] for r in self._rows(conn, "cq-1")] == ["done"]

    def test_deleting_the_chunk_deletes_its_queue_row(self, db, conn):
        chunk_id = self._chunk(db)
        db.enqueue_extraction(
            ExtractionItem(session_id="cq-1", chunk_id=chunk_id, content_type="chunk")
        )
        conn.execute("DELETE FROM chunks WHERE id = %s", (chunk_id,))
        assert self._rows(conn, "cq-1") == []

    def test_a_claimed_row_carries_its_chunk_id(self, db):
        chunk_id = self._chunk(db)
        db.enqueue_extraction(
            ExtractionItem(session_id="cq-1", chunk_id=chunk_id, content_type="chunk")
        )
        (row,) = db.claim_pending_extractions(limit=10)
        assert row["chunk_id"] == chunk_id and row["content"] == ""
        assert db.get_chunk(chunk_id) == {"content": "one\n\n---\n\ntwo", "episode_ids": [1, 2]}

    def test_without_the_column_the_row_keeps_the_text_and_dedups_on_it(self, db, conn):
        CHUNK_ID_PROBE.missing(db._url)  # the migration is not applied here yet
        item = ExtractionItem(
            session_id="cq-2", chunk_id=7, content="the window", content_type="chunk"
        )
        db.enqueue_extraction(item)
        db.enqueue_extraction(item)
        assert self._rows(conn, "cq-2") == [(None, "the window", "pending")]

    def test_a_chunk_gone_reads_as_none(self, db):
        assert db.get_chunk(2_000_000_000) is None


class TestBlankFinishedQueueText:
    def _row(self, conn, status: str, age_days: int, content: str = "some turn text") -> int:
        (qid,) = conn.execute(
            "INSERT INTO extraction_queue (session_id, content, content_type, status, processed_at) "
            "VALUES ('bq', %s, 'chunk', %s, now() - make_interval(days => %s)) RETURNING id",
            (content, status, age_days),
        ).fetchone()
        return qid

    def test_only_old_finished_rows_lose_their_text(self, db, conn, clean_tables):
        old_done = self._row(conn, "done", 10)
        fresh_done = self._row(conn, "done", 1)
        old_failed = self._row(conn, "failed", 10)
        pending = self._row(conn, "pending", 10)
        assert db.blank_finished_queue_text(older_than_days=7, batch=2) == 1
        text = dict(conn.execute("SELECT id, content FROM extraction_queue").fetchall())
        assert text[old_done] == ""
        assert text[fresh_done] == text[old_failed] == text[pending] == "some turn text"
        # the row itself stays: the dashboard's series and web_enqueue read it
        assert conn.execute("SELECT count(*) FROM extraction_queue").fetchone()[0] == 4

    def test_it_works_through_more_rows_than_one_batch(self, db, conn, clean_tables):
        for _ in range(5):
            self._row(conn, "done", 30)
        assert db.blank_finished_queue_text(older_than_days=7, batch=2) == 5
        assert db.blank_finished_queue_text(older_than_days=7, batch=2) == 0
