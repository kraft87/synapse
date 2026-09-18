from __future__ import annotations

from typing import Any, cast

from ingestion.db_connection import _EMBED_DIMS, DatabaseConnection, _vector_literal


class PreferenceStore(DatabaseConnection):
    def find_live_preferences(
        self, owner_id: str, group_id: str, embedding: list[float], limit: int = 5
    ) -> list[dict[str, Any]]:
        """Live preferences for one owner/group, nearest to ``embedding`` by cosine.
        Returns ``[{id, pref, polarity, sim}]`` in DESCENDING similarity (sim = 1 -
        cosine_distance in [0, 2]). The gate's dedup/supersession decision reads this."""
        vlit = _vector_literal(embedding)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT id, pref, polarity, 1 - (embedding <=> %s::vector({_EMBED_DIMS})) AS sim "  # nosec B608 — _EMBED_DIMS is a validated int, not user input
                "FROM preferences "
                "WHERE owner_id = %s AND group_id = %s AND t_invalid IS NULL "
                "AND embedding IS NOT NULL "
                f"ORDER BY embedding <=> %s::vector({_EMBED_DIMS}) ASC LIMIT %s",
                (vlit, owner_id, group_id, vlit, limit),
            ).fetchall()
        return [
            {"id": r["id"], "pref": r["pref"], "polarity": r["polarity"], "sim": float(r["sim"])}
            for r in rows
        ]

    def insert_preference(
        self,
        *,
        owner_id: str,
        group_id: str,
        project: str | None,
        pref: str,
        polarity: str,
        embedding: list[float] | None,
        embed_model: str | None,
        source_ref: str | None,
    ) -> int:
        """Append one live preference (assert_count=1). Returns the new row id."""
        vlit = _vector_literal(embedding)
        with self._conn() as conn:
            row = conn.execute(
                "INSERT INTO preferences "  # nosec B608 — _EMBED_DIMS is a validated int, not user input
                "(owner_id, group_id, project, pref, polarity, embedding, embed_model, source_ref) "
                f"VALUES (%s,%s,%s,%s,%s,%s::vector({_EMBED_DIMS}),%s,%s) RETURNING id",
                (
                    owner_id,
                    group_id,
                    project,
                    pref,
                    polarity,
                    vlit,
                    embed_model if embedding is not None else None,
                    source_ref,
                ),
            ).fetchone()
        assert row is not None, "INSERT RETURNING id returned nothing"
        return cast(int, row["id"])

    def merge_preference_text(
        self, pref_id: int, pref: str, embedding: list[float], embed_model: str
    ) -> None:
        """The gray-band judge folded NEW information into an existing preference:
        replace the anchor text and its embedding (the pair must never diverge) and
        count the re-assertion in the same write. Deliberately departs from
        ``reassert_preference``'s keep-the-older-text rule: the merged text IS the
        older anchor plus the newly stated detail."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE preferences "  # nosec B608 — _EMBED_DIMS is a validated int, not user input
                f"SET pref = %s, embedding = %s::vector({_EMBED_DIMS}), embed_model = %s, "
                "assert_count = assert_count + 1, last_asserted = now() WHERE id = %s",
                (pref, _vector_literal(embedding), embed_model, pref_id),
            )

    def reassert_preference(self, pref_id: int) -> None:
        """A restated preference: bump assert_count + last_asserted, keep the older text
        (the first phrasing is the anchor; recurrence is the strength signal)."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE preferences "
                "SET assert_count = assert_count + 1, last_asserted = now() WHERE id = %s",
                (pref_id,),
            )

    def supersede_preference(self, old_id: int, new_id: int) -> None:
        """A contradicting preference won: retire the old row (t_invalid=now, superseded_by
        = the new row) so the live set carries only the current stance, auditably linked."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE preferences SET t_invalid = now(), superseded_by = %s WHERE id = %s",
                (new_id, old_id),
            )

    def top_preferences(self, owner_id: str, limit: int = 8) -> list[dict[str, Any]]:
        """Live preferences for the session-start block: strongest first — most-reasserted,
        then most-recent. Across ALL groups (a standing preference shapes every session)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT pref, polarity, assert_count, left(first_seen::text, 10) AS since "
                "FROM preferences WHERE owner_id = %s AND t_invalid IS NULL "
                "ORDER BY assert_count DESC, last_asserted DESC LIMIT %s",
                (owner_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]
