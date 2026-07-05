"""Integration tests for the DB layer (runs against synapse_test)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ingestion.db import Database
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


class TestIngestionState:
    def test_get_watermark_returns_none_initially(self, db):
        wm = db.get_watermark("test-source-new")
        assert wm is None

    def test_set_and_get_watermark(self, db):
        ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        db.set_watermark("test-source", ts)
        result = db.get_watermark("test-source")
        assert result is not None
        assert result.replace(microsecond=0) == ts.replace(microsecond=0)

    def test_update_watermark(self, db):
        t1 = datetime(2026, 1, 1, tzinfo=UTC)
        t2 = datetime(2026, 1, 2, tzinfo=UTC)
        db.set_watermark("wm-update", t1)
        db.set_watermark("wm-update", t2)
        assert db.get_watermark("wm-update").date() == t2.date()


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
        pending = db.get_pending_extractions(limit=10)
        assert any(p["episode_id"] == ep_id for p in pending)

    def test_enqueue_summary(self, db):
        item = ExtractionItem(
            session_id="eq-sum-1",
            content="Summary text for extraction",
            content_type="summary",
            project="synapse",
        )
        db.enqueue_extraction(item)
        pending = db.get_pending_extractions(limit=10)
        assert any(p["session_id"] == "eq-sum-1" for p in pending)

    def test_no_duplicate_enqueue(self, db):
        ep_id = db.upsert_episode(Episode(session_id="eq-dedup", sequence=1, content="no dups"))
        item = ExtractionItem(episode_id=ep_id, content="no dups", content_type="episode")
        db.enqueue_extraction(item)
        db.enqueue_extraction(item)  # second call should be a no-op
        pending = db.get_pending_extractions(limit=100)
        matches = [p for p in pending if p["episode_id"] == ep_id]
        assert len(matches) == 1

    def test_mark_done(self, db):
        ep_id = db.upsert_episode(Episode(session_id="eq-done", sequence=1, content="mark me done"))
        item = ExtractionItem(episode_id=ep_id, content="mark me done", content_type="episode")
        db.enqueue_extraction(item)
        pending = db.get_pending_extractions(limit=10)
        queue_id = next(p["id"] for p in pending if p["episode_id"] == ep_id)
        db.mark_extraction_done(queue_id)
        still_pending = db.get_pending_extractions(limit=100)
        assert not any(p["id"] == queue_id for p in still_pending)

    def test_mark_failed_with_error(self, db):
        ep_id = db.upsert_episode(Episode(session_id="eq-fail", sequence=1, content="fail me"))
        item = ExtractionItem(episode_id=ep_id, content="fail me", content_type="episode")
        db.enqueue_extraction(item)
        pending = db.get_pending_extractions(limit=10)
        queue_id = next(p["id"] for p in pending if p["episode_id"] == ep_id)
        db.mark_extraction_failed(queue_id, error="Connection refused")
        with db._conn() as conn:
            row = conn.execute(
                "SELECT status, error FROM extraction_queue WHERE id=%s", (queue_id,)
            ).fetchone()
        assert row["status"] == "failed"
        assert "Connection refused" in row["error"]
