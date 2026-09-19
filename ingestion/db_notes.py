from __future__ import annotations

from typing import Any, cast

import orjson

from ingestion.db_connection import _EMBED_DIMS, DatabaseConnection, _vector_literal


class NoteStore(DatabaseConnection):
    def find_live_notes(
        self, owner_id: str, group_id: str, embedding: list[float], limit: int = 5
    ) -> list[dict[str, Any]]:
        """Live notes for one owner/group, nearest to ``embedding`` by cosine (over the
        HOOK — the embed target). Returns ``[{id, hook, body, type, project, sim}]`` in
        DESCENDING similarity. The reconcile path's dedup/supersession decision reads this."""
        vlit = _vector_literal(embedding)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT id, hook, body, type, project, 1 - (embedding <=> %s::halfvec({_EMBED_DIMS})) AS sim "  # nosec B608 — _EMBED_DIMS is a validated int, not user input
                "FROM notes "
                "WHERE owner_id = %s AND group_id = %s AND superseded_by IS NULL "
                "AND embedding IS NOT NULL "
                f"ORDER BY embedding <=> %s::halfvec({_EMBED_DIMS}) ASC LIMIT %s",
                (vlit, owner_id, group_id, vlit, limit),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "hook": r["hook"],
                "body": r["body"],
                "type": r["type"],
                "project": r["project"],
                "sim": float(r["sim"]),
            }
            for r in rows
        ]

    def search_live_notes(
        self,
        owner_id: str,
        project: str | None,
        embedding: list[float],
        limit: int = 24,
        audience: str | None = None,
    ) -> list[dict[str, Any]]:
        """Live notes nearest to ``embedding`` (hook KNN), scoped like the board: global
        types (user/feedback/reference) always in, project notes only for the caller's
        project (``project = NULL`` matches global types only). No group filter — notes
        span groups and the board doesn't filter either. recall()'s notes leg reads this;
        ``find_live_notes`` (above) stays the reconcile path's owner/group-exact probe.

        ``audience`` (schema 053) narrows to one tier — recall passes 'work-safe' for a
        restricted surface, None for a full-trust one. Filtering here rather than after
        the KNN keeps a restricted caller's serve at full width instead of spending its
        _NOTES_FETCH budget on rows it can never see."""
        vlit = _vector_literal(embedding)
        clauses = [
            "owner_id = %s",
            "superseded_by IS NULL",
            "embedding IS NOT NULL",
            "(type IN ('user','feedback','reference') OR project = %s)",
        ]
        params: list[Any] = [vlit, owner_id, project]
        if audience is not None:
            clauses.append("audience = %s")
            params.append(audience)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT id, hook, body, type, project, audience, updated_at, "  # nosec B608 — _EMBED_DIMS is a validated int; the WHERE clauses are literals with bound params
                f"1 - (embedding <=> %s::halfvec({_EMBED_DIMS})) AS sim "
                "FROM notes "
                f"WHERE {' AND '.join(clauses)} "
                f"ORDER BY embedding <=> %s::halfvec({_EMBED_DIMS}) ASC LIMIT %s",
                (*params, vlit, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def insert_note(
        self,
        *,
        owner_id: str,
        group_id: str,
        project: str | None,
        type: str,
        hook: str,
        body: str,
        embedding: list[float] | None,
        embed_model: str | None,
        source_ref: str | None,
        audience: str | None = None,
    ) -> int:
        """Append one live note. Returns the new row id. NULL embedding is allowed
        (keyless dev/test; dedup KNN simply skips such rows).

        ``audience=None`` leaves the column at its schema default ('personal') — the
        fail-closed tier. Callers that know better (remember(), the backfill apply)
        pass it explicitly; see ingestion/surfaces.derive_audience for the rule."""
        vlit = _vector_literal(embedding)
        with self._conn() as conn:
            row = conn.execute(
                "INSERT INTO notes "  # nosec B608 — _EMBED_DIMS is a validated int, not user input
                "(owner_id, group_id, project, type, hook, body, embedding, embed_model, "
                " source_ref, audience) "
                f"VALUES (%s,%s,%s,%s,%s,%s,%s::halfvec({_EMBED_DIMS}),%s,%s,"
                "COALESCE(%s,'personal')) RETURNING id",
                (
                    owner_id,
                    group_id,
                    project,
                    type,
                    hook,
                    body,
                    vlit,
                    embed_model if embedding is not None else None,
                    source_ref,
                    audience,
                ),
            ).fetchone()
        assert row is not None, "INSERT RETURNING id returned nothing"
        return cast(int, row["id"])

    def update_note(
        self,
        note_id: int,
        *,
        hook: str,
        body: str,
        embedding: list[float] | None,
        embed_model: str | None,
        audience: str | None = None,
    ) -> None:
        """A restated note: refresh hook/body/embedding in place and bump updated_at
        (the note keeps its id — the board line just gets the newer phrasing).

        ``audience=None`` PRESERVES the stored tier. A restatement must never silently
        re-classify a note: the only thing that moves an existing note between audiences
        is an explicit override (COALESCE keeps the current value otherwise)."""
        vlit = _vector_literal(embedding)
        with self._conn() as conn:
            conn.execute(
                "UPDATE notes "  # nosec B608 — _EMBED_DIMS is a validated int, not user input
                f"SET hook = %s, body = %s, embedding = %s::halfvec({_EMBED_DIMS}), "
                "embed_model = %s, audience = COALESCE(%s, audience), updated_at = now() "
                "WHERE id = %s",
                (
                    hook,
                    body,
                    vlit,
                    embed_model if embedding is not None else None,
                    audience,
                    note_id,
                ),
            )

    def supersede_note(self, old_id: int, new_id: int) -> None:
        """A contradicting note won: retire the old row (superseded_by = the new row) so
        the live set carries only the current statement, auditably linked."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE notes SET superseded_by = %s WHERE id = %s",
                (new_id, old_id),
            )

    def list_board_notes(
        self, owner_id: str, project: str | None, audience: str | None = None
    ) -> list[dict[str, Any]]:
        """Live notes for the board: every global-scope type (user/feedback/reference)
        plus the current project's project-notes. Ordered feedback -> user -> project ->
        reference, newest-updated first within each type. ``project=None`` serves the
        global set only (``project = NULL`` matches nothing).

        ``audience`` (schema 053) narrows to one tier — the board passes 'work-safe' for
        a restricted surface, None for a full-trust one."""
        clauses = [
            "owner_id = %s",
            "superseded_by IS NULL",
            "(type IN ('user','feedback','reference') OR project = %s)",
        ]
        params: list[Any] = [owner_id, project]
        if audience is not None:
            clauses.append("audience = %s")
            params.append(audience)
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, hook, type, project, audience, updated_at FROM notes "  # nosec B608 — clause list is literal, values are bound
                f"WHERE {' AND '.join(clauses)} "
                "ORDER BY CASE type WHEN 'feedback' THEN 0 WHEN 'user' THEN 1 "
                "WHEN 'project' THEN 2 ELSE 3 END, updated_at DESC",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def get_notes_by_ids(self, ids: list[int], audience: str | None = None) -> list[dict[str, Any]]:
        """Fetch note bodies by id — the on-demand half of the board (hook on the board,
        body behind the id). Silently drops unknown ids.

        ``audience`` applies the same tier filter the overview paths use. Ids are
        guessable, so drill-down MUST NOT be a way around the board's filter: a
        restricted caller asking for an id it was never served simply gets nothing."""
        if not ids:
            return []
        clauses = ["id = ANY(%s)"]
        params: list[Any] = [ids]
        if audience is not None:
            clauses.append("audience = %s")
            params.append(audience)
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, hook, body, type, project, audience, updated_at FROM notes "  # nosec B608 — clause list is literal, values are bound
                f"WHERE {' AND '.join(clauses)} ORDER BY id",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def restricted_surface_projects(self) -> list[str]:
        """Every project any REGISTERED restricted surface may read (schema 053).

        The provenance half of audience derivation: a note filed under a project some
        work host already reads episodes from is work-safe by construction.

        APPROVED rows only (schema 054). A pending enrollment reads nothing, so letting
        its allowlist widen the work-safe tier would classify notes for an audience that
        does not exist yet — and a device that never gets approved would leave a
        permanent, invisible widening behind it."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT unnest(allowed_projects) AS project FROM surfaces "
                "WHERE trust = 'restricted' AND status = 'approved'"
            ).fetchall()
        return [cast(str, r["project"]) for r in rows if r["project"]]

    def find_note_by_source_ref(self, source_ref: str) -> dict[str, Any] | None:
        """Newest note carrying this provenance ref, live or retired — the seed
        importer's idempotency probe (re-imports must not duplicate)."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, hook, body, type, project, superseded_by, updated_at "
                "FROM notes WHERE source_ref = %s ORDER BY id DESC LIMIT 1",
                (source_ref,),
            ).fetchone()
        return dict(row) if row is not None else None

    # ------------------------------------------------------------------
    # Notes curation (schema 051) — the read/write surface for the nightly
    # dream→notes lane (dream/notes/). The lane owns the policy (what counts as
    # a duplicate, which note wins, the caps); these accessors own the SQL only.
    # notes_curation doubles as the memo (a judged pair is never re-sent to the
    # LLM) and the audit log, so every candidate query LEFT JOINs it and treats
    # `judged_at < updated_at` as "the note changed, judge it again".
    # ------------------------------------------------------------------

    def find_note_pair_candidates(
        self, owner_id: str, *, sim_floor: float, limit: int
    ) -> list[dict[str, Any]]:
        """Unjudged (or stale-judged) live note pairs whose HOOK embeddings are within
        ``sim_floor`` cosine, most-similar first.

        Pure SQL over the vectors already stored by the write path — the lane adds no
        embedding calls. The ``b.id > a.id`` join condition yields each unordered pair
        exactly once AND normalizes it to (least, greatest), which is the shape the
        memo index keys on. The floor is deliberately permissive (corrections often
        share little hook wording); the LLM judge downstream is the precision layer.

        Returns ``[{a: {...}, b: {...}, sim: float}]`` where each side carries
        ``id, hook, body, type, project, updated_at``."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT a.id AS a_id, a.hook AS a_hook, a.body AS a_body, a.type AS a_type, "
                "       a.project AS a_project, a.updated_at AS a_updated_at, "
                "       b.id AS b_id, b.hook AS b_hook, b.body AS b_body, b.type AS b_type, "
                "       b.project AS b_project, b.updated_at AS b_updated_at, "
                "       1 - (a.embedding <=> b.embedding) AS sim "
                "FROM notes a "
                "JOIN notes b ON b.owner_id = a.owner_id AND b.id > a.id "
                "            AND b.superseded_by IS NULL AND b.embedding IS NOT NULL "
                "LEFT JOIN notes_curation c ON c.op = 'pair' "
                "            AND least(c.note_a, c.note_b) = a.id "
                "            AND greatest(c.note_a, c.note_b) = b.id "
                "WHERE a.owner_id = %s AND a.superseded_by IS NULL AND a.embedding IS NOT NULL "
                "  AND 1 - (a.embedding <=> b.embedding) >= %s "
                "  AND (c.id IS NULL OR c.judged_at < greatest(a.updated_at, b.updated_at)) "
                "ORDER BY sim DESC, a.id, b.id LIMIT %s",
                (owner_id, sim_floor, limit),
            ).fetchall()
        return [
            {
                "a": {k: r[f"a_{k}"] for k in ("id", "hook", "body", "type", "project")}
                | {"updated_at": r["a_updated_at"]},
                "b": {k: r[f"b_{k}"] for k in ("id", "hook", "body", "type", "project")}
                | {"updated_at": r["b_updated_at"]},
                "sim": float(r["sim"]),
            }
            for r in rows
        ]

    def find_retype_candidates(self, owner_id: str, *, limit: int) -> list[dict[str, Any]]:
        """Live GLOBAL notes (type user/feedback) not yet scope-judged, oldest id first.

        The memo doubles as the cursor: once a note is judged it drops out of this
        query, so successive runs walk forward through the global set without any
        separate watermark row. An edit to the note (updated_at past judged_at) puts
        it back in the queue."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT n.id, n.hook, n.body, n.type, n.project, n.updated_at FROM notes n "
                "LEFT JOIN notes_curation c ON c.op = 'retype' AND c.note_a = n.id "
                "WHERE n.owner_id = %s AND n.superseded_by IS NULL "
                "  AND n.type IN ('user','feedback') "
                "  AND (c.id IS NULL OR c.judged_at < n.updated_at) "
                "ORDER BY n.id LIMIT %s",
                (owner_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def known_project_slugs(self, limit: int = 40) -> list[str]:
        """The project ids that actually exist in this store — notes first (a project
        already carrying curated memory is the likeliest target), then episode projects
        by volume. Handed to the retype judge as a closed vocabulary, which is what
        keeps it from inventing a plausible-looking slug nothing is filed under."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT project FROM ("
                "  SELECT project, 0 AS tier, count(*) AS n FROM notes "
                "    WHERE project IS NOT NULL AND superseded_by IS NULL GROUP BY project "
                "  UNION ALL "
                "  SELECT project, 1 AS tier, count(*) AS n FROM episodes "
                "    WHERE project IS NOT NULL GROUP BY project"
                ") s GROUP BY project ORDER BY min(tier), max(n) DESC, project LIMIT %s",
                (limit,),
            ).fetchall()
        return [cast(str, r["project"]) for r in rows]

    def project_slug_exists(self, slug: str) -> bool:
        """True if ``slug`` is already a known project id — either on a note or on an
        episode. The retype apply is gated on this so a hallucinated slug can only ever
        be recorded, never written into the store as a new (unreachable) scope."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT EXISTS (SELECT 1 FROM notes WHERE project = %s) "
                "    OR EXISTS (SELECT 1 FROM episodes WHERE project = %s) AS found",
                (slug, slug),
            ).fetchone()
        return bool(row is not None and row["found"])

    def retype_note(
        self, note_id: int, *, type: str, project: str | None, audience: str | None = None
    ) -> None:
        """Re-scope one note (global -> project). Reversible and content-preserving:
        hook/body/embedding are untouched, and ``updated_at`` is deliberately NOT
        bumped — the note did not change, only its shelf, and a bump would invalidate
        the memo row the lane is about to write.

        ``audience`` re-derives the tier when the note's PROJECT moves (spec §1: the
        project rule is what decides an unclassified note's audience, so changing the
        project changes the answer). ``None`` preserves the stored tier."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE notes SET type = %s, project = %s, "
                "audience = COALESCE(%s, audience) WHERE id = %s",
                (type, project, audience, note_id),
            )

    def record_curation(
        self,
        *,
        op: str,
        note_a: int,
        note_b: int | None,
        verdict: str,
        applied: bool,
        detail: dict[str, Any] | None = None,
    ) -> int:
        """Upsert one audit/memo row. Re-judging an already-judged pair or note
        overwrites the verdict in place (and refreshes judged_at) — the row is the
        current answer, not an append-only history. Returns the row id."""
        payload = orjson.dumps(detail).decode() if detail is not None else None
        target = (
            "(least(note_a, note_b), greatest(note_a, note_b)) WHERE op = 'pair'"
            if op == "pair"
            else "(note_a) WHERE op = 'retype'"
        )
        with self._conn() as conn:
            row = conn.execute(
                "INSERT INTO notes_curation (op, note_a, note_b, verdict, applied, detail) "  # nosec B608 — `target` is a literal chosen by `op`, never interpolated user input
                "VALUES (%s, %s, %s, %s, %s, %s::jsonb) "
                f"ON CONFLICT {target} DO UPDATE SET "
                "  verdict = EXCLUDED.verdict, applied = EXCLUDED.applied, "
                "  detail = EXCLUDED.detail, judged_at = now() "
                "RETURNING id",
                (op, note_a, note_b, verdict, applied, payload),
            ).fetchone()
        assert row is not None, "INSERT ... RETURNING id returned nothing"
        return cast(int, row["id"])
