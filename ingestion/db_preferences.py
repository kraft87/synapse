from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from ingestion.db_connection import _EMBED_DIMS, DatabaseConnection, _vector_literal


@dataclass(frozen=True)
class SameReaders:
    """Which stored preferences have the same readers as a new item, for
    :meth:`PreferenceStore.find_live_preferences`.

    A restricted device is served a preference only when its project is on that device's
    allowlist, so two projects have the same readers when, for every approved restricted
    surface, both are on its allowlist or both are off it. ``surfaces`` holds one entry
    per such surface: its allowlist, lower-cased, and whether the new item's project is on
    it. Case is folded because the read side matches allowlists case-insensitively, and a
    missing project is on no allowlist.

    ``match_project`` is the stricter fallback for when the allowlists could not be read:
    only rows of ``project`` qualify (``None`` matches rows that have none)."""

    surfaces: tuple[tuple[tuple[str, ...], bool], ...] = ()
    match_project: bool = False
    project: str | None = None


class PreferenceStore(DatabaseConnection):
    def find_live_preferences(
        self,
        owner_id: str,
        embedding: list[float],
        limit: int = 5,
        same_readers: SameReaders | None = None,
    ) -> list[dict[str, Any]]:
        """Live preferences for one owner, nearest to ``embedding`` by cosine.
        Returns ``[{id, pref, polarity, sim}]`` in DESCENDING similarity (sim
        = 1 - cosine_distance in [0, 2]).

        ``same_readers`` keeps only rows whose readers match the new item's. It is applied
        in the query, before the LIMIT, so a duplicate with the same readers is found even
        when ``limit`` nearer rows have other readers. The gate's dedup/supersession
        decision reads this and may re-assert, merge into or retire what it returns, so the
        gate always passes it. Passing nothing returns every live row: only a caller that
        merely looks at the store (a read-only listing, a test) may do that."""
        vlit = _vector_literal(embedding)
        where = ["owner_id = %s", "t_invalid IS NULL", "embedding IS NOT NULL"]
        params: list[Any] = [vlit, owner_id]
        if same_readers is not None:
            for names, on_list in same_readers.surfaces:
                if on_list:
                    where.append("lower(project) = ANY(%s::text[])")
                else:
                    where.append("(project IS NULL OR NOT (lower(project) = ANY(%s::text[])))")
                params.append(list(names))
            if same_readers.match_project:
                where.append(
                    "NULLIF(lower(project), '') IS NOT DISTINCT FROM NULLIF(lower(%s::text), '')"
                )
                params.append(same_readers.project)
        params += [vlit, limit]
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT id, pref, polarity, "  # nosec B608 — _EMBED_DIMS is a validated int and the predicates are fixed strings; every value is a bound parameter
                f"1 - (embedding <=> %s::vector({_EMBED_DIMS})) AS sim "
                f"FROM preferences WHERE {' AND '.join(where)} "
                f"ORDER BY embedding <=> %s::vector({_EMBED_DIMS}) ASC LIMIT %s",
                params,
            ).fetchall()
        return [
            {
                "id": r["id"],
                "pref": r["pref"],
                "polarity": r["polarity"],
                "sim": float(r["sim"]),
            }
            for r in rows
        ]

    def insert_preference(
        self,
        *,
        owner_id: str,
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
                "(owner_id, project, pref, polarity, embedding, embed_model, source_ref) "
                f"VALUES (%s,%s,%s,%s,%s::vector({_EMBED_DIMS}),%s,%s) RETURNING id",
                (
                    owner_id,
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
        then most-recent. Across every project (a standing preference shapes every session)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT pref, polarity, assert_count, left(first_seen::text, 10) AS since "
                "FROM preferences WHERE owner_id = %s AND t_invalid IS NULL "
                "ORDER BY assert_count DESC, last_asserted DESC LIMIT %s",
                (owner_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]
