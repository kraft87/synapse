from __future__ import annotations

from typing import Any, cast

from ingestion.db_connection import _EMBED_DIMS, DatabaseConnection, _vector_literal


class TimelineStore(DatabaseConnection):
    def insert_timeline_event(
        self,
        *,
        t_valid: str,
        fact: str,
        source: str,
        source_ref: str,
        project: str | None,
        salience: int,
        embedding: list[float] | None,
        embed_model: str | None,
        event_type: str | None = None,
        domain: str | None = None,
    ) -> int:
        """Append one event to the episodic timeline (schema 033). Idempotent on
        UNIQUE(source, source_ref) — re-processing a turn never duplicates. Returns
        rows inserted (0 = already present)."""
        vlit = _vector_literal(embedding)
        with self._conn() as conn:
            return conn.execute(
                "INSERT INTO timeline_events "  # nosec B608 — _EMBED_DIMS is a validated int, not user input
                "(t_valid, fact, source, source_ref, project, salience, embedding, embed_model, "
                " event_type, domain) "
                f"VALUES (%s,%s,%s,%s,%s,%s,%s::vector({_EMBED_DIMS}),%s,%s,%s) "
                "ON CONFLICT (source, source_ref) DO NOTHING",
                (
                    t_valid,
                    fact,
                    source,
                    source_ref,
                    project,
                    salience,
                    vlit,
                    embed_model if embedding is not None else None,
                    event_type,
                    domain,
                ),
            ).rowcount

    def timeline_near_candidates(
        self,
        embedding: list[float],
        project: str | None,
        t_valid: str,
        exclude_episode_ref: str,
        window_days: int = 14,
        max_dist: float = 0.20,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Nearest same-project chat events within ±window_days of ``t_valid`` and
        under ``max_dist`` cosine distance — the dedup confirm call's candidate pool.
        Excludes the new event's own turn (the ``ep:<id>`` base ref and its ``#k``
        siblings): a turn's multiple events are intentionally distinct, never dups."""
        vlit = _vector_literal(embedding)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT id, fact, t_valid, source_ref, "  # nosec B608 — _EMBED_DIMS is a validated int, not user input
                f"       (embedding <=> %s::vector({_EMBED_DIMS})) AS dist "
                "FROM timeline_events "
                "WHERE source = 'chat' AND project IS NOT DISTINCT FROM %s "
                "AND embedding IS NOT NULL "
                "AND source_ref != %s AND source_ref NOT LIKE %s "
                "AND t_valid BETWEEN %s::timestamptz - make_interval(days => %s) "
                "                AND %s::timestamptz + make_interval(days => %s) "
                f"AND (embedding <=> %s::vector({_EMBED_DIMS})) < %s "
                "ORDER BY dist LIMIT %s",
                (
                    vlit,
                    project,
                    exclude_episode_ref,
                    exclude_episode_ref + "#%",
                    t_valid,
                    window_days,
                    t_valid,
                    window_days,
                    vlit,
                    max_dist,
                    limit,
                ),
            ).fetchall()
        return cast(list[dict[str, Any]], rows)

    def bump_timeline_reported(self, event_id: int, t_valid: str) -> None:
        """Record a re-assertion of an existing timeline event (dedup merge outcome):
        increment reported_count and keep the EARLIEST t_valid — the canonical date of
        a date-split re-telling is the first-resolved one, and a re-tell that resolves
        an earlier true date corrects the row."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE timeline_events SET reported_count = reported_count + 1, "
                "t_valid = LEAST(t_valid, %s::timestamptz) WHERE id = %s",
                (t_valid, event_id),
            )

    def timeline_ident_exists(
        self, idents: list[str], project: str | None, t_valid: str, window_hours: int
    ) -> bool:
        """True if any timeline event in the project/time window already carries one of
        these identifiers (PR ref / SHA) in its fact text. The write-time cross-source
        dedup key — exact identifier match, deliberately NOT embedding similarity."""
        if not idents:
            return False
        pats = ["%" + i + "%" for i in idents]
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM timeline_events "
                "WHERE (project = %s OR %s::text IS NULL) "
                "AND t_valid BETWEEN %s::timestamptz - make_interval(hours => %s) "
                "                AND %s::timestamptz + make_interval(hours => %s) "
                "AND lower(fact) LIKE ANY(%s) LIMIT 1",
                (project, project, t_valid, window_hours, t_valid, window_hours, pats),
            ).fetchone()
        return row is not None
