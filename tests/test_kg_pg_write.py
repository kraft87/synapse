"""DB-backed tests for ingestion/kg_pg_write.py (#67 PR 3 primary writer).

Exercises every mutation against the real kg_entities / kg_relationships
schema (017) on the shared test DB, asserting the contracts the pipeline
depends on: upsert updates only the rename-able fields (created_at /
entity_type are insert-only, embedding survives a no-embedding update),
edge re-creation is a retry no-op, invalidation sets both t_invalid and the
legacy invalid_at, and failures RAISE (Postgres is the source of truth now —
no best-effort swallowing).
"""

from __future__ import annotations

import pytest

from ingestion.kg_pg_write import KGPostgresWriter, _ts, _vec

GROUP = "technical"
DIM = 2048


def _axis_list(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i] = 1.0
    return v


@pytest.fixture()
def kg_writer(conn, monkeypatch, db_url):
    """Clean KG tables and point the writer at the test DB."""
    monkeypatch.setenv("SYNAPSE_DB_URL", db_url)
    conn.execute("TRUNCATE kg_entities, kg_relationships RESTART IDENTITY CASCADE")
    yield KGPostgresWriter()


class TestHelpers:
    def test_ts_nulls_garbage_timestamps(self):
        # The edge-date extractor occasionally emits out-of-range years that
        # TIMESTAMPTZ rejects — those become NULL, not a failed write.
        assert _ts("-4599999974-05-27T00:00:00Z") is None
        assert _ts("2026-06-01T00:00:00+00:00") == "2026-06-01T00:00:00+00:00"
        assert _ts("2026-06-01T00:00:00Z") == "2026-06-01T00:00:00Z"
        assert _ts(None) is None
        assert _ts("") is None

    def test_vec_literal(self):
        assert _vec([1.0, 2.5]) == "[1.0,2.5]"
        assert _vec(None) is None
        assert _vec([]) is None

    def test_raises_when_db_url_unset(self, monkeypatch):
        monkeypatch.delenv("SYNAPSE_DB_URL", raising=False)
        with pytest.raises(RuntimeError, match="SYNAPSE_DB_URL"):
            KGPostgresWriter().upsert_node(
                uuid="u",
                name="n",
                normalized_name="n",
                entity_type="",
                summary="",
                group_id=GROUP,
                project=None,
                created_at=None,
                valid_at=None,
                embedding=None,
            )


class TestUpsertNode:
    def test_nul_bytes_stripped_from_text_fields(self, kg_writer, conn):
        # Postgres TEXT rejects NUL bytes — an LLM response carrying one must
        # not fail the write (and the queue-item retry loop behind it).
        kg_writer.upsert_node(
            uuid="e-nul",
            name="Name\x00X",
            normalized_name="name\x00x",
            entity_type="Tool",
            summary="sum\x00mary",
            group_id=GROUP,
            project=None,
            created_at=None,
            valid_at=None,
            embedding=None,
        )
        row = conn.execute(
            "SELECT name, normalized_name, summary FROM kg_entities WHERE uuid = 'e-nul'"
        ).fetchone()
        assert row == ("NameX", "namex", "summary")

    def test_insert_then_update_semantics(self, kg_writer, conn):
        kg_writer.upsert_node(
            uuid="e-1",
            name="Synapse",
            normalized_name="synapse",
            entity_type="Tool",
            summary="memory layer",
            group_id=GROUP,
            project="synapse",
            created_at="2026-06-01T00:00:00+00:00",
            valid_at="2026-06-01T00:00:00+00:00",
            embedding=_axis_list(0),
        )
        row = conn.execute(
            "SELECT owner_id, group_id, name, normalized_name, entity_type, summary, "
            "created_at, embedding IS NOT NULL FROM kg_entities WHERE uuid = 'e-1'"
        ).fetchone()
        assert row[:6] == ("default", GROUP, "Synapse", "synapse", "Tool", "memory layer")
        created_at_first, has_emb = row[6], row[7]
        assert has_emb

        # Update: rename + longer summary, NO embedding this time. The
        # existing embedding and the insert-only fields must survive.
        kg_writer.upsert_node(
            uuid="e-1",
            name="Synapse v2",
            normalized_name="synapse v2",
            entity_type="SHOULD-NOT-CHANGE",
            summary="memory layer, renamed",
            group_id=GROUP,
            project="synapse",
            created_at="2026-06-09T00:00:00+00:00",
            valid_at="2026-06-09T00:00:00+00:00",
            embedding=None,
        )
        row = conn.execute(
            "SELECT name, normalized_name, entity_type, summary, created_at, "
            "embedding IS NOT NULL FROM kg_entities WHERE uuid = 'e-1'"
        ).fetchone()
        assert row[0] == "Synapse v2"
        assert row[1] == "synapse v2"
        assert row[2] == "Tool"  # insert-only
        assert row[3] == "memory layer, renamed"
        assert row[4] == created_at_first  # insert-only
        assert row[5]  # embedding preserved through a no-embedding update


class TestCreateEdges:
    def _row(self, uuid: str, **over):
        base = {
            "src": "e-s",
            "tgt": "e-t",
            "edge_uuid": uuid,
            "name": "USES",
            "fact": f"fact for {uuid}",
            "episodes": [1, 2],
            "created_at": "2026-06-01T00:00:00+00:00",
            "t_created": "2026-06-01T00:00:00+00:00",
            "valid_at": "2026-06-01T00:00:00+00:00",
            "t_valid": "2026-06-01T00:00:00+00:00",
            "emb": _axis_list(0),
        }
        base.update(over)
        return base

    def test_nul_bytes_stripped_from_fact(self, kg_writer, conn):
        kg_writer.create_edges([self._row("r-nul", fact="a\x00fact")], GROUP)
        row = conn.execute("SELECT fact FROM kg_relationships WHERE uuid = 'r-nul'").fetchone()
        assert row == ("afact",)

    def test_insert_round_trip(self, kg_writer, conn):
        kg_writer.create_edges(
            [
                self._row("r-1"),
                self._row("r-2", emb=None, web_artifact_id=42),
            ],
            GROUP,
        )
        rows = conn.execute(
            "SELECT uuid, owner_id, group_id, src_uuid, tgt_uuid, fact, episodes, "
            "retrieval_count, fact_embedding IS NOT NULL, web_artifact_id, t_invalid "
            "FROM kg_relationships ORDER BY uuid"
        ).fetchall()
        assert len(rows) == 2
        r1, r2 = rows
        assert r1[:5] == ("r-1", "default", GROUP, "e-s", "e-t")
        assert r1[6] == [1, 2]  # episodes jsonb round-trip
        assert r1[7] == 0 and r1[8] and r1[9] is None and r1[10] is None
        assert not r2[8]  # no embedding
        assert r2[9] == 42  # web-lane provenance

    def test_recreate_is_retry_noop(self, kg_writer, conn):
        kg_writer.create_edges([self._row("r-1")], GROUP)
        kg_writer.create_edges([self._row("r-1", fact="DIFFERENT fact")], GROUP)
        rows = conn.execute("SELECT fact FROM kg_relationships WHERE uuid = 'r-1'").fetchall()
        assert rows == [("fact for r-1",)]  # original wins; conflict is a no-op

    def test_garbage_valid_at_falls_back_to_created(self, kg_writer, conn):
        kg_writer.create_edges(
            [self._row("r-1", valid_at="-4599999974-05-27T00:00:00Z", t_valid=None)],
            GROUP,
        )
        row = conn.execute(
            "SELECT valid_at, created_at FROM kg_relationships WHERE uuid = 'r-1'"
        ).fetchone()
        assert row[0] == row[1]  # nulled garbage falls back to created_at

    def test_empty_rows_is_noop(self, kg_writer):
        kg_writer.create_edges([], GROUP)  # must not raise or connect


class TestInvalidateEdges:
    def test_sets_both_lifecycle_fields(self, kg_writer, conn):
        kg_writer.create_edges(
            [
                {
                    "src": "e-s",
                    "tgt": "e-t",
                    "edge_uuid": "r-1",
                    "name": "USES",
                    "fact": "f",
                    "episodes": None,
                    "created_at": "2026-06-01T00:00:00+00:00",
                    "valid_at": "2026-06-01T00:00:00+00:00",
                },
                {
                    "src": "e-s",
                    "tgt": "e-t",
                    "edge_uuid": "r-2",
                    "name": "USES",
                    "fact": "g",
                    "episodes": None,
                    "created_at": "2026-06-01T00:00:00+00:00",
                    "valid_at": "2026-06-01T00:00:00+00:00",
                },
            ],
            GROUP,
        )
        kg_writer.invalidate_edges([("r-1", "2026-06-05T00:00:00+00:00"), ("r-2", None)], GROUP)
        rows = conn.execute(
            "SELECT uuid, t_invalid, invalid_at FROM kg_relationships ORDER BY uuid"
        ).fetchall()
        for _uuid, t_inv, inv in rows:
            assert t_inv is not None
            assert t_inv == inv  # legacy field mirrors the canonical one
        assert str(rows[0][1]).startswith("2026-06-05")  # explicit ts honored

    def test_records_superseder_when_provided(self, kg_writer, conn):
        kg_writer.create_edges(
            [
                {
                    "src": "e-s",
                    "tgt": "e-t",
                    "edge_uuid": "old-1",
                    "name": "USES",
                    "fact": "f",
                    "episodes": None,
                    "created_at": "2026-06-01T00:00:00+00:00",
                    "valid_at": "2026-06-01T00:00:00+00:00",
                },
            ],
            GROUP,
        )
        kg_writer.invalidate_edges([("old-1", None)], GROUP, invalidated_by="new-1")
        (by,) = conn.execute(
            "SELECT invalidated_by FROM kg_relationships WHERE uuid = 'old-1'"
        ).fetchone()
        assert by == "new-1"
        # COALESCE preserves it when a later invalidate passes None (idempotent re-invalidation).
        kg_writer.invalidate_edges([("old-1", None)], GROUP)
        (by2,) = conn.execute(
            "SELECT invalidated_by FROM kg_relationships WHERE uuid = 'old-1'"
        ).fetchone()
        assert by2 == "new-1"
        # r-2 used the now() fallback — just has to be set (asserted above)

    def test_missing_edge_is_zero_row_update(self, kg_writer):
        # An edge that was never created yields a 0-row UPDATE, not an error
        # (idempotent retries can re-invalidate after a partial failure).
        kg_writer.invalidate_edges([("never-existed", None)], GROUP)


class TestReinforceEdges:
    def _row(self, uuid: str, **over):
        base = {
            "src": "e-s",
            "tgt": "e-t",
            "edge_uuid": uuid,
            "name": "USES",
            "fact": f"fact for {uuid}",
            "episodes": [1, 2],
            "created_at": "2026-06-01T00:00:00+00:00",
            "valid_at": "2026-06-01T00:00:00+00:00",
        }
        base.update(over)
        return base

    def test_bumps_count_and_unions_episodes(self, kg_writer, conn):
        kg_writer.create_edges([self._row("r-1")], GROUP)  # episodes [1,2], mention_count default 1
        kg_writer.reinforce_edges([("r-1", [2, 3])], GROUP)
        cnt, eps = conn.execute(
            "SELECT mention_count, episodes FROM kg_relationships WHERE uuid = 'r-1'"
        ).fetchone()
        assert cnt == 2  # one re-assertion captured
        assert set(eps) == {1, 2, 3}  # new episode unioned, deduped

    def test_idempotent_when_episodes_already_present(self, kg_writer, conn):
        kg_writer.create_edges([self._row("r-1")], GROUP)  # episodes [1,2]
        kg_writer.reinforce_edges([("r-1", [1, 2])], GROUP)  # nothing new
        cnt, eps = conn.execute(
            "SELECT mention_count, episodes FROM kg_relationships WHERE uuid = 'r-1'"
        ).fetchone()
        assert cnt == 1  # no double-count on re-processing the same chunk
        assert set(eps) == {1, 2}

    def test_empty_episode_list_is_per_item_noop(self, kg_writer, conn):
        kg_writer.create_edges([self._row("r-1")], GROUP)
        kg_writer.reinforce_edges([("r-1", [])], GROUP)
        cnt = conn.execute(
            "SELECT mention_count FROM kg_relationships WHERE uuid = 'r-1'"
        ).fetchone()[0]
        assert cnt == 1

    def test_empty_items_is_noop(self, kg_writer):
        kg_writer.reinforce_edges([], GROUP)  # must not raise or connect
