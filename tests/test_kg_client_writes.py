"""DB-backed tests for the KGClient writes (#67 PR 3 primary writer).

Exercises every mutation against the real kg_entities / kg_relationships
schema (017) on the shared test DB, asserting the contracts the pipeline
depends on: a matched entity's update carries only a longer summary and a known
supertype (never its name, embedding or subtype), edge re-creation is a retry no-op,
only the t_* time columns are written (never the FalkorDB-era mirrors), and failures
RAISE (Postgres is the source of truth now — no best-effort swallowing).
"""

from __future__ import annotations

import pytest

from ingestion.kg_client import KGClient, _ts, _vec
from tests.helpers.embed import GROUP
from tests.helpers.kg import _axis_list, make_edge_row


@pytest.fixture()
def kg_writer(conn, db_url):
    """Clean KG tables and yield a client on the test DB."""
    conn.execute("TRUNCATE kg_entities, kg_relationships RESTART IDENTITY CASCADE")
    client = KGClient(db_url)
    yield client
    client._reset()


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


class TestInsertNode:
    def test_nul_bytes_stripped_from_text_fields(self, kg_writer, conn):
        # Postgres TEXT rejects NUL bytes — an LLM response carrying one must
        # not fail the write (and the queue-item retry loop behind it).
        kg_writer.insert_node(
            node_uuid="e-nul",
            name="Name\x00X",
            entity_type="Tool",
            summary="sum\x00mary",
            group_id=GROUP,
        )
        row = conn.execute(
            "SELECT name, normalized_name, summary FROM kg_entities WHERE uuid = 'e-nul'"
        ).fetchone()
        assert row == ("NameX", "namex", "summary")

    def test_insert_round_trip(self, kg_writer, conn):
        kg_writer.insert_node(
            node_uuid="e-1",
            name="Synapse",
            entity_type="Tool",
            summary="memory layer",
            group_id=GROUP,
            project="synapse",
            embedding=_axis_list(0),
            supertype="Service",
        )
        row = conn.execute(
            "SELECT owner_id, group_id, project, name, normalized_name, summary, "
            "entity_subtype, entity_supertype, embedding IS NOT NULL "
            "FROM kg_entities WHERE uuid = 'e-1'"
        ).fetchone()
        assert row == (
            "default",
            GROUP,
            "synapse",
            "Synapse",
            "synapse",
            "memory layer",
            "Tool",
            "Service",
            True,
        )

    def test_only_the_t_columns_and_the_subtype_are_written(self, kg_writer, conn):
        kg_writer.insert_node(
            node_uuid="e-1", name="Synapse", entity_type="Tool", summary="", group_id=GROUP
        )
        assert conn.execute(
            "SELECT entity_type, created_at, valid_at, t_created IS NOT NULL, t_valid IS NOT NULL "
            "FROM kg_entities WHERE uuid = 'e-1'"
        ).fetchone() == (None, None, None, True, True)

    def test_reinserting_a_uuid_is_a_retry_no_op(self, kg_writer, conn):
        kg_writer.insert_node(
            node_uuid="e-1", name="Synapse", entity_type="Tool", summary="a", group_id=GROUP
        )
        kg_writer.insert_node(
            node_uuid="e-1", name="Renamed", entity_type="Other", summary="b", group_id=GROUP
        )
        assert conn.execute(
            "SELECT name, summary FROM kg_entities WHERE uuid = 'e-1'"
        ).fetchone() == (
            "Synapse",
            "a",
        )


class TestUpdateNode:
    """A mention that matched an existing entity may only carry a longer summary and a
    newly known supertype onto it. Its identity (name, normalized_name, embedding,
    subtype) is whatever the entity was first written with."""

    @pytest.fixture()
    def entity(self, kg_writer, conn):
        kg_writer.insert_node(
            node_uuid="e-pg",
            name="PostgreSQL",
            entity_type="Database",
            summary="relational database",
            group_id=GROUP,
            embedding=_axis_list(0),
            supertype="Database",
        )
        return kg_writer

    def _row(self, conn):
        return conn.execute(
            "SELECT name, normalized_name, entity_subtype, entity_supertype, summary, "
            "embedding IS NOT NULL, xmin::text FROM kg_entities WHERE uuid = 'e-pg'"
        ).fetchone()

    def test_longer_summary_wins_and_identity_is_untouched(self, entity, conn):
        entity.update_node("e-pg", summary="relational database used for the memory layer")
        name, norm, subtype, supertype, summary, has_emb, _ = self._row(conn)
        assert (name, norm, subtype, supertype) == (
            "PostgreSQL",
            "postgresql",
            "Database",
            "Database",
        )
        assert summary == "relational database used for the memory layer"
        assert has_emb

    def test_shorter_or_empty_summary_keeps_the_stored_one(self, entity, conn):
        entity.update_node("e-pg", summary="database")
        entity.update_node("e-pg", summary="")
        assert self._row(conn)[4] == "relational database"

    def test_supertype_is_set_when_known_and_kept_when_not(self, entity, conn):
        entity.update_node("e-pg", summary="", supertype="Tool")
        assert self._row(conn)[3] == "Tool"
        entity.update_node("e-pg", summary="")  # taxonomy map not loaded: nothing to say
        assert self._row(conn)[3] == "Tool"

    def test_a_mention_that_adds_nothing_writes_nothing(self, entity, conn):
        before = self._row(conn)[6]
        entity.update_node("e-pg", summary="database", supertype="Database")
        assert self._row(conn)[6] == before  # same row version: no write, no new HNSW tuple

    def test_unknown_uuid_is_a_zero_row_update(self, entity):
        entity.update_node("never-existed", summary="x")


class TestEntitySummaries:
    def test_summaries_of_the_asked_uuids_in_the_group(self, kg_writer, conn):
        for uuid, name, summary, group in [
            ("e-1", "A", "first", GROUP),
            ("e-2", "B", "second", GROUP),
            ("e-3", "C", "other group", "personal"),
        ]:
            kg_writer.insert_node(
                node_uuid=uuid, name=name, entity_type="Tool", summary=summary, group_id=group
            )
        assert kg_writer.entity_summaries(["e-1", "e-3", "e-9"], GROUP) == {"e-1": "first"}
        assert kg_writer.entity_summaries([], GROUP) == {}


class TestCreateEdges:
    def _row(self, uuid: str, **over):
        # Create path uses the full row contract verbatim (incl. emb).
        return make_edge_row(uuid, **over)

    def test_nul_bytes_stripped_from_fact(self, kg_writer, conn):
        kg_writer.create_edges_batch([self._row("r-nul", fact="a\x00fact")], GROUP)
        row = conn.execute("SELECT fact FROM kg_relationships WHERE uuid = 'r-nul'").fetchone()
        assert row == ("afact",)

    def test_insert_round_trip(self, kg_writer, conn):
        kg_writer.create_edges_batch(
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
        kg_writer.create_edges_batch([self._row("r-1")], GROUP)
        kg_writer.create_edges_batch([self._row("r-1", fact="DIFFERENT fact")], GROUP)
        rows = conn.execute("SELECT fact FROM kg_relationships WHERE uuid = 'r-1'").fetchall()
        assert rows == [("fact for r-1",)]  # original wins; conflict is a no-op

    def test_garbage_t_valid_falls_back_to_t_created(self, kg_writer, conn):
        kg_writer.create_edges_batch(
            [self._row("r-1", t_valid="-4599999974-05-27T00:00:00Z")],
            GROUP,
        )
        row = conn.execute(
            "SELECT t_valid, t_created FROM kg_relationships WHERE uuid = 'r-1'"
        ).fetchone()
        assert row[0] == row[1]  # nulled garbage falls back to t_created

    def test_legacy_time_columns_are_not_written(self, kg_writer, conn):
        kg_writer.create_edges_batch([self._row("r-1")], GROUP)
        legacy = conn.execute(
            "SELECT created_at, valid_at, invalid_at, t_created IS NOT NULL, t_valid IS NOT NULL "
            "FROM kg_relationships WHERE uuid = 'r-1'"
        ).fetchone()
        assert legacy == (None, None, None, True, True)

    def test_empty_rows_is_noop(self, kg_writer):
        kg_writer.create_edges_batch([], GROUP)  # must not raise or connect


class TestInvalidateEdges:
    def test_sets_t_invalid_and_never_the_legacy_mirror(self, kg_writer, conn):
        kg_writer.create_edges_batch(
            [
                {
                    "src": "e-s",
                    "tgt": "e-t",
                    "edge_uuid": "r-1",
                    "name": "USES",
                    "fact": "f",
                    "episodes": None,
                    "t_created": "2026-06-01T00:00:00+00:00",
                    "t_valid": "2026-06-01T00:00:00+00:00",
                },
                {
                    "src": "e-s",
                    "tgt": "e-t",
                    "edge_uuid": "r-2",
                    "name": "USES",
                    "fact": "g",
                    "episodes": None,
                    "t_created": "2026-06-01T00:00:00+00:00",
                    "t_valid": "2026-06-01T00:00:00+00:00",
                },
            ],
            GROUP,
        )
        kg_writer.invalidate_edges_batch(
            [("r-1", "2026-06-05T00:00:00+00:00"), ("r-2", None)], GROUP
        )
        rows = conn.execute(
            "SELECT uuid, t_invalid, invalid_at FROM kg_relationships ORDER BY uuid"
        ).fetchall()
        for _uuid, t_inv, inv in rows:
            assert t_inv is not None
            assert inv is None  # the FalkorDB-era mirror is no longer written
        assert str(rows[0][1]).startswith("2026-06-05")  # explicit ts honored

    def test_records_superseder_when_provided(self, kg_writer, conn):
        kg_writer.create_edges_batch(
            [
                {
                    "src": "e-s",
                    "tgt": "e-t",
                    "edge_uuid": "old-1",
                    "name": "USES",
                    "fact": "f",
                    "episodes": None,
                    "t_created": "2026-06-01T00:00:00+00:00",
                    "t_valid": "2026-06-01T00:00:00+00:00",
                },
            ],
            GROUP,
        )
        kg_writer.invalidate_edges_batch([("old-1", None)], GROUP, invalidated_by="new-1")
        (by,) = conn.execute(
            "SELECT invalidated_by FROM kg_relationships WHERE uuid = 'old-1'"
        ).fetchone()
        assert by == "new-1"
        # COALESCE preserves it when a later invalidate passes None (idempotent re-invalidation).
        kg_writer.invalidate_edges_batch([("old-1", None)], GROUP)
        (by2,) = conn.execute(
            "SELECT invalidated_by FROM kg_relationships WHERE uuid = 'old-1'"
        ).fetchone()
        assert by2 == "new-1"
        # r-2 used the now() fallback — just has to be set (asserted above)

    def test_missing_edge_is_zero_row_update(self, kg_writer):
        # An edge that was never created yields a 0-row UPDATE, not an error
        # (idempotent retries can re-invalidate after a partial failure).
        kg_writer.invalidate_edges_batch([("never-existed", None)], GROUP)


class TestReinforceEdges:
    def _row(self, uuid: str, **over):
        # Reinforce path drops the create-time embedding.
        return make_edge_row(uuid, drop=("emb",), **over)

    def test_bumps_count_and_unions_episodes(self, kg_writer, conn):
        kg_writer.create_edges_batch(
            [self._row("r-1")], GROUP
        )  # episodes [1,2], mention_count default 1
        kg_writer.reinforce_edges([("r-1", [2, 3])], GROUP)
        cnt, eps = conn.execute(
            "SELECT mention_count, episodes FROM kg_relationships WHERE uuid = 'r-1'"
        ).fetchone()
        assert cnt == 2  # one re-assertion captured
        assert set(eps) == {1, 2, 3}  # new episode unioned, deduped

    def test_idempotent_when_episodes_already_present(self, kg_writer, conn):
        kg_writer.create_edges_batch([self._row("r-1")], GROUP)  # episodes [1,2]
        kg_writer.reinforce_edges([("r-1", [1, 2])], GROUP)  # nothing new
        cnt, eps = conn.execute(
            "SELECT mention_count, episodes FROM kg_relationships WHERE uuid = 'r-1'"
        ).fetchone()
        assert cnt == 1  # no double-count on re-processing the same chunk
        assert set(eps) == {1, 2}

    def test_empty_episode_list_is_per_item_noop(self, kg_writer, conn):
        kg_writer.create_edges_batch([self._row("r-1")], GROUP)
        kg_writer.reinforce_edges([("r-1", [])], GROUP)
        cnt = conn.execute(
            "SELECT mention_count FROM kg_relationships WHERE uuid = 'r-1'"
        ).fetchone()[0]
        assert cnt == 1

    def test_empty_items_is_noop(self, kg_writer):
        kg_writer.reinforce_edges([], GROUP)  # must not raise or connect


class TestEvidenceCurrency:
    """Schema 058: ongoing / last_supported_at / last_supported_by, and the clamped
    invalidation time."""

    def _row(self, uuid: str, **over):
        return make_edge_row(uuid, drop=("emb",), **over)

    def test_create_writes_evidence_fields(self, kg_writer, conn):
        kg_writer.create_edges_batch(
            [
                self._row(
                    "r-1",
                    ongoing=True,
                    last_supported_at="2026-06-01T00:00:00+00:00",
                    last_supported_by=2,
                ),
                self._row("r-2"),  # legacy caller: nothing supplied -> NULLs (unknown)
            ],
            GROUP,
        )
        rows = conn.execute(
            "SELECT uuid, ongoing, last_supported_at, last_supported_by "
            "FROM kg_relationships ORDER BY uuid"
        ).fetchall()
        assert rows[0][1] is True
        assert str(rows[0][2]).startswith("2026-06-01")
        assert rows[0][3] == 2
        assert rows[1][1] is None and rows[1][2] is None and rows[1][3] is None

    def test_reinforce_refreshes_support_forward_only_on_new_provenance(self, kg_writer, conn):
        kg_writer.create_edges_batch(
            [self._row("r-1", last_supported_at="2026-06-01T00:00:00+00:00", last_supported_by=2)],
            GROUP,
        )  # episodes [1,2]
        # (a) assistant restatement (no support) with new provenance: count bumps, support stays
        kg_writer.reinforce_edges([("r-1", [3], None, None)], GROUP)
        cnt, sup, by = conn.execute(
            "SELECT mention_count, last_supported_at, last_supported_by FROM kg_relationships "
            "WHERE uuid = 'r-1'"
        ).fetchone()
        assert cnt == 2 and str(sup).startswith("2026-06-01") and by == 2
        # (b) user restatement, newer, new provenance: support moves forward
        kg_writer.reinforce_edges([("r-1", [4], "2026-08-01T00:00:00+00:00", 4)], GROUP)
        cnt, sup, by = conn.execute(
            "SELECT mention_count, last_supported_at, last_supported_by FROM kg_relationships "
            "WHERE uuid = 'r-1'"
        ).fetchone()
        assert cnt == 3 and str(sup).startswith("2026-08-01") and by == 4
        # (c) a replayed chunk (provenance already present) refreshes nothing, even as user
        kg_writer.reinforce_edges([("r-1", [4], "2026-09-01T00:00:00+00:00", 4)], GROUP)
        # (d) an OLDER user statement (backfilled transcript) never moves support backward
        kg_writer.reinforce_edges([("r-1", [5], "2026-01-01T00:00:00+00:00", 5)], GROUP)
        cnt, sup, by = conn.execute(
            "SELECT mention_count, last_supported_at, last_supported_by FROM kg_relationships "
            "WHERE uuid = 'r-1'"
        ).fetchone()
        assert cnt == 4 and str(sup).startswith("2026-08-01") and by == 4

    def test_reinforce_sets_support_when_never_confirmed(self, kg_writer, conn):
        kg_writer.create_edges_batch([self._row("r-1")], GROUP)  # last_supported_at NULL
        kg_writer.reinforce_edges([("r-1", [9], "2026-07-01T00:00:00+00:00", 9)], GROUP)
        sup, by = conn.execute(
            "SELECT last_supported_at, last_supported_by FROM kg_relationships WHERE uuid = 'r-1'"
        ).fetchone()
        assert str(sup).startswith("2026-07-01") and by == 9

    def test_invalidate_clamps_to_the_edge_own_valid_from(self, kg_writer, conn):
        # Stale edge became live 2026-06-01; the correction is dated 2026-03-01 (the user said
        # "stopped in March") -> retired as of 2026-06-01, never a negative lifetime.
        kg_writer.create_edges_batch([self._row("r-1")], GROUP)  # t_valid 2026-06-01
        kg_writer.invalidate_edges_batch(
            [("r-1", "2026-03-01T00:00:00+00:00")], GROUP, invalidated_by="n"
        )
        t_inv, by = conn.execute(
            "SELECT t_invalid, invalidated_by FROM kg_relationships WHERE uuid = 'r-1'"
        ).fetchone()
        assert str(t_inv).startswith("2026-06-01") and by == "n"

    def test_invalidate_keeps_the_earlier_retirement(self, kg_writer, conn):
        kg_writer.create_edges_batch([self._row("r-1")], GROUP)
        kg_writer.invalidate_edges_batch([("r-1", "2026-07-01T00:00:00+00:00")], GROUP)
        kg_writer.invalidate_edges_batch(
            [("r-1", "2026-09-01T00:00:00+00:00")], GROUP, invalidated_by="n"
        )
        t_inv, by = conn.execute(
            "SELECT t_invalid, invalidated_by FROM kg_relationships WHERE uuid = 'r-1'"
        ).fetchone()
        assert str(t_inv).startswith("2026-07-01")  # first retirement stands
        assert by == "n"  # but the later call may still add the missing link


class TestDeleteOrphanEntities:
    def _entity(self, conn, uuid: str, age_hours: int, group: str = GROUP) -> None:
        conn.execute(
            "INSERT INTO kg_entities (uuid, group_id, name, t_created) "
            "VALUES (%s, %s, %s, now() - make_interval(hours => %s))",
            (uuid, group, uuid, age_hours),
        )

    def _edge(self, conn, uuid: str, src: str, tgt: str, *, retired: bool = False, group=GROUP):
        conn.execute(
            "INSERT INTO kg_relationships (uuid, group_id, src_uuid, tgt_uuid, fact, t_invalid) "
            "VALUES (%s, %s, %s, %s, 'f', CASE WHEN %s THEN now() END)",
            (uuid, group, src, tgt, retired),
        )

    def test_only_old_entities_without_any_edge_go(self, kg_writer, conn):
        for name in ("src", "tgt", "retired-only", "cross-group", "orphan-old"):
            self._entity(conn, name, age_hours=48)
        self._entity(conn, "orphan-new", age_hours=1)
        self._entity(conn, "other", age_hours=48, group="personal")
        self._edge(conn, "r1", "src", "other", group="personal")
        self._edge(conn, "r2", "other", "tgt", group="personal")
        self._edge(conn, "r3", "retired-only", "other", retired=True, group="personal")
        # an edge in another group still counts (a few edges cross groups on prod)
        self._edge(conn, "r4", "other", "cross-group", group="personal")
        assert kg_writer.delete_orphan_entities(24) == 1
        left = {r[0] for r in conn.execute("SELECT uuid FROM kg_entities").fetchall()}
        assert left == {"src", "tgt", "retired-only", "cross-group", "orphan-new", "other"}
        assert kg_writer.delete_orphan_entities(24) == 0  # idempotent
