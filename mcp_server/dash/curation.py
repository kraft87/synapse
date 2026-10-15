"""Curation: the soft flags and the two DESTRUCTIVE routes (phase 7) that hard-delete an
episode (with its content cascade) or a single KG edge.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json
from starlette.requests import Request

from ingestion import pg_pool
from ingestion.db_episodes import DELETED_TURNS_PROBE
from ingestion.db_timeline import REPORTED_REFS_PROBE
from ingestion.surfaces import OWN_SURFACE_PROBE
from ingestion.web_artifacts import ARCHIVE_PROBE
from mcp_server.dash.common import _episode_id, _episode_id_from_ref, _gist, _provenance
from mcp_server.http_helpers import RouteError, _iso, route

_FLAG_KINDS = ("episode", "fact", "timeline_event", "preference", "note")

# Per-kind resolver for the best-effort gist, source episode and (notes only) full body
# shown on the flags list. The flags list is operator-sized (a handful of rows), so a small
# lookup per row is cheap.
# kind -> (table, text column, id column, column naming the row's source episode, body column)
_FLAG_SOURCES = {
    "episode": ("episodes", "content", "id", None, None),
    "fact": ("kg_relationships", "fact", "uuid", "episodes", None),
    "timeline_event": ("timeline_events", "fact", "id", "source_ref", None),
    "preference": ("preferences", "pref", "id", "source_ref", None),
    "note": ("notes", "hook", "id", "source_ref", "body"),
}


def _resolve_flag_item(
    conn: psycopg.Connection[dict[str, Any]], kind: str, item_id: str
) -> tuple[str, int | None, str | None]:
    """A flagged row's gist, the episode the Flags tab can open for it, and its body.

    The episode is the episode itself, a fact's first provenance episode, or the
    ``ep:<id>`` turn a timeline event, preference or note was born from; None when the
    row or its episode is gone, or the ref names something that isn't a turn (a git SHA).
    The body is a note's full text (the Flags tab shows it inline beside the hook), None
    for every other kind."""
    spec = _FLAG_SOURCES.get(kind)
    if not spec:
        return "", None, None
    table, text_col, id_col, ref_col, body_col = spec
    # id_col is a fixed identifier from the trusted map above, never user input; item_id
    # is bound as a parameter. Numeric-keyed tables get a digit guard so a bad id can't
    # raise a cast error inside the read.
    if id_col == "id" and not str(item_id).isdigit():
        return "", None, None
    ref_sql = f", {ref_col} AS ref" if ref_col else ""
    body_sql = f", {body_col} AS body" if body_col else ""
    try:
        row = conn.execute(
            f"SELECT {text_col} AS t{ref_sql}{body_sql} FROM {table} WHERE {id_col} = %s LIMIT 1",  # nosec B608
            (item_id,),
        ).fetchone()
    except Exception:  # pragma: no cover - defensive, e.g. a missing table
        return "", None, None
    if row is None:
        return "", None, None
    if kind == "episode":
        return _gist(row["t"]), int(item_id), None
    ref = row["ref"]
    episode_id = _provenance(ref) if isinstance(ref, list) else _episode_id_from_ref(ref)
    if (
        episode_id is not None
        and conn.execute("SELECT 1 FROM episodes WHERE id = %s", (episode_id,)).fetchone() is None
    ):
        episode_id = None
    return _gist(row["t"]), episode_id, row.get("body")


def _flags_list(db_url: str) -> dict[str, Any]:
    with pg_pool.connection(db_url, row_factory=dict_row) as conn:
        rows = conn.execute(
            "SELECT id, kind, item_id, note, created_at FROM dashboard_flags "
            "WHERE removed_at IS NULL ORDER BY created_at DESC, id DESC"
        ).fetchall()
        flags = []
        for r in rows:
            gist, episode_id, body = _resolve_flag_item(conn, r["kind"], r["item_id"])
            flags.append(
                {
                    "id": r["id"],
                    "kind": r["kind"],
                    "item_id": r["item_id"],
                    "note": r["note"],
                    "created_at": _iso(r["created_at"]),
                    "gist": gist,
                    "episode_id": episode_id,
                    "body": body,
                }
            )
        return {"flags": flags}


def _set_flag(db_url: str, kind: str, item_id: str, note: str | None, flagged: bool | None) -> bool:
    """Put a flag in the state the operator asked for and return the state it ends in.

    ``flagged`` True / False is idempotent (a repeat changes nothing and appends no audit
    row), so a stale button or list can't invert the intent; None toggles, for older
    clients. Each branch is one statement against the partial-unique index, so concurrent
    requests settle instead of colliding on it. A change appends a dashboard_audit row."""
    detail = Json({"note": note}) if note else None
    with pg_pool.connection(db_url, autocommit=False, row_factory=dict_row) as conn:
        with conn.transaction():
            if flagged is not True:
                retired = conn.execute(
                    "UPDATE dashboard_flags SET removed_at = now() "
                    "WHERE kind = %s AND item_id = %s AND removed_at IS NULL RETURNING id",
                    (kind, item_id),
                ).fetchone()
                if retired is not None:
                    conn.execute(
                        "INSERT INTO dashboard_audit (action, kind, item_id, detail) "
                        "VALUES ('unflag', %s, %s, %s)",
                        (kind, item_id, detail),
                    )
                if retired is not None or flagged is False:
                    return False
            created = conn.execute(
                "INSERT INTO dashboard_flags (kind, item_id, note) VALUES (%s, %s, %s) "
                "ON CONFLICT (kind, item_id) WHERE removed_at IS NULL DO NOTHING RETURNING id",
                (kind, item_id, note),
            ).fetchone()
            if created is not None:
                conn.execute(
                    "INSERT INTO dashboard_audit (action, kind, item_id, detail) "
                    "VALUES ('flag', %s, %s, %s)",
                    (kind, item_id, detail),
                )
            return True


# Real deletes, not tombstones: the operator's intent is "gone from memory", so a
# soft-delete column that recall() would have to learn to skip is the wrong shape.
# Each route runs in ONE transaction (autocommit is OFF here, unlike every read above)
# so a partial cascade is never committed.
#
# THE rule, from a 2026-06 KG cleanup that nearly went wrong: provenance naming ONLY the
# deleted episode is deleted; provenance SHARED with surviving episodes is UNLINKED. A
# blanket "delete every fact whose provenance mentions these episodes" would have taken
# 1467 legitimate facts out with the 330 bad ones. See docs/dashboard-contract.md §Phase 7
# for the full cascade table, including what is deliberately left alone (preferences,
# notes, recall_metrics/recall_feedback, kg_entities) and why. The web research captured
# from a deleted turn is not left alone: it is that turn's content.
#
# Note the NOTIFY trigger (schema 043) is AFTER INSERT only, so a delete does not push
# anything down the SSE stream; the deleting client drops the row locally and other
# clients see it gone on their next full fetch.


def _is_episode_ref(elem: Any, episode_ids: set[int]) -> bool:
    """True when a provenance-array element names one of ``episode_ids``.

    Matches the JSON number and (defensively) the numeric-string form — the extractor's
    array element type isn't guaranteed, which is why every read path here checks both.
    bool is an int subclass, so it's excluded explicitly."""
    if isinstance(elem, bool):
        return False
    if isinstance(elem, int):
        return elem in episode_ids
    if isinstance(elem, str):
        return elem.strip().isdigit() and int(elem) in episode_ids
    return False


def _other_episode_refs(raw: Any, episode_ids: set[int]) -> list[Any]:
    """The provenance elements that are NOT in ``episode_ids`` (order preserved).

    Empty ⇒ SOLE provenance (delete the row). Non-empty ⇒ SHARED (unlink and keep). A
    malformed / non-array value counts as sole: it names nothing that would survive."""
    if not isinstance(raw, list):
        return []
    return [e for e in raw if not _is_episode_ref(e, episode_ids)]


def _clear_invalidated_by(conn: psycopg.Connection[dict[str, Any]], uuids: list[str]) -> int:
    """NULL the schema-028 supersession pointers aimed at edges we're deleting, so no
    surviving edge cites a uuid that no longer exists. Returns rows touched."""
    if not uuids:
        return 0
    cur = conn.execute(
        "UPDATE kg_relationships SET invalidated_by = NULL WHERE invalidated_by = ANY(%s)",
        (uuids,),
    )
    return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0


def _repoint_event(
    conn: psycopg.Connection[dict[str, Any]], event_id: int, episode_id: int
) -> bool:
    """Make the surviving turn ``episode_id`` the owner of a timeline event: its
    source_ref becomes the first free ``ep:<id>`` / ``ep:<id>#<k>`` (the turn's own events
    may hold some), and the event takes that turn's provenance stamp (schema 057), since it
    is now served on that turn's account. False when the turn no longer exists."""
    stamp = ", surface_id = e.surface_id" if OWN_SURFACE_PROBE.ready(conn) else ""
    return bool(
        conn.execute(
            "UPDATE timeline_events t SET source_ref = free.ref"  # nosec B608 — a fixed fragment
            + stamp
            + " FROM episodes e, LATERAL (SELECT 'ep:' || e.id || CASE WHEN n = 1 THEN '' "
            "ELSE '#' || n END AS ref FROM generate_series(1, 50) n WHERE NOT EXISTS "
            "(SELECT 1 FROM timeline_events o WHERE o.source = 'chat' AND "
            "o.source_ref = 'ep:' || e.id || CASE WHEN n = 1 THEN '' ELSE '#' || n END) "
            "ORDER BY n LIMIT 1) free WHERE t.id = %s AND e.id = %s",
            (event_id, episode_id),
        ).rowcount
    )


def delete_episode_content(
    conn: psycopg.Connection[dict[str, Any]],
    episode_ids: list[int],
    session_id: str | None = None,
) -> dict[str, Any]:
    """The episode-delete cascade: everything carrying the content of ``episode_ids``.

    Runs on ``conn`` inside the CALLER's transaction, with the episode rows already locked:
    chunks citing any of them and those chunks' extraction_queue rows (by chunk_id, else by
    session and text, so the 001 FK misses them), their sole-provenance facts (shared ones are
    unlinked), their chat timeline events, a tombstone per turn (``deleted_turns``, so a
    re-ship, catch-up sweep or backfill of the transcript on disk cannot bring the turn
    back; ``ingestion.episode_writer`` checks it), the rows themselves (their own
    extraction_queue rows follow via the 001 FK), and any active flag on the facts, events
    and episodes that went. The web research captured from these turns (``web_artifacts``
    and the cold ``research_archive``, by ``parent_episode_id``) goes with them, and so do
    the facts the web lane read out of those pages. ``session_id`` (a session purge) also
    takes every capture of that session, which includes a sub-agent's: those belong to no
    turn, so a single-turn delete leaves them. Returns the counts plus the uuids of the
    deleted facts and the ids of the deleted events, so a client can drop them from its
    lists. Shared by the dashboard's single-turn delete and private mode's session purge
    (``mcp_server/private_session_routes``).

    Set-based on purpose: the purge passes every turn of a session, and the provenance
    arrays have no usable index (029's GIN is partial), so each lookup is one pass over
    the table. "Sole" and "shared" are judged against the whole set, so a fact citing two
    deleted turns and one surviving turn is unlinked once, not once per turn."""
    ids = set(episode_ids)
    id_texts = [str(i) for i in episode_ids]
    # --- chunks -------------------------------------------------------------
    # A chunk's `content` is the verbatim concatenation of its 3-5 episode window
    # (schema 008), so there is no way to excise one turn from it: every chunk
    # citing one of these episodes goes. Chunks whose window ALSO named surviving
    # episodes are counted separately — that's real collateral the UI has to admit to.
    # jsonb_array_elements_text renders the number 12 and the string "12" alike, which is
    # both forms _is_episode_ref accepts; a non-array value is read as a one-element array.
    chunk_rows = conn.execute(
        "SELECT id, episode_ids FROM chunks c WHERE EXISTS ("
        "SELECT 1 FROM jsonb_array_elements_text(CASE WHEN jsonb_typeof(c.episode_ids) = 'array' "
        "THEN c.episode_ids ELSE jsonb_build_array(c.episode_ids) END) AS v "
        "WHERE v = ANY(%s::text[])) FOR UPDATE",
        (id_texts,),
    ).fetchall()
    chunk_ids = [r["id"] for r in chunk_rows]
    chunks_mixed = sum(1 for r in chunk_rows if _other_episode_refs(r["episode_ids"], ids))
    if chunk_ids:
        # A chunk's queue row follows it by chunk_id (cascade) once that migration is
        # applied. Rows without one, from before it or from a database without it, are
        # keyed by (session_id, content) with no episode_id, so no FK reaches them: a
        # pending one would re-extract the deleted text into facts with no provenance, and
        # a done one still stores it verbatim. Every status goes, before the chunk text is
        # gone.
        conn.execute(
            "DELETE FROM extraction_queue q USING chunks c "
            "WHERE c.id = ANY(%s) AND q.content_type = 'chunk' "
            "AND q.session_id = c.session_id AND q.content = c.content",
            (chunk_ids,),
        )
        conn.execute("DELETE FROM chunks WHERE id = ANY(%s)", (chunk_ids,))

    # --- captured web research -------------------------------------------------------
    # A page's chunks (and their queue rows) follow it by FK. The facts the web lane read out
    # of it name the page as their only source: they go, or are unlinked when a surviving
    # episode backs them too, joining the sole/shared judgement below.
    ids_arg = (episode_ids, session_id)
    web_ids = [
        r["id"]
        for r in conn.execute(
            "DELETE FROM web_artifacts WHERE parent_episode_id = ANY(%s::bigint[]) "
            "OR session_id = %s RETURNING id",
            ids_arg,
        ).fetchall()
    ]
    archive_deleted = 0
    if ARCHIVE_PROBE.ready(conn):  # the cold archive (optional migration)
        archive_deleted = conn.execute(
            "DELETE FROM research_archive WHERE parent_episode_id = ANY(%s::bigint[]) "
            "OR session_id = %s",
            ids_arg,
        ).rowcount

    # --- KG facts (sole vs shared provenance) --------------------------------
    fact_rows = conn.execute(
        "SELECT uuid, episodes FROM kg_relationships r WHERE EXISTS ("
        "SELECT 1 FROM jsonb_array_elements_text(CASE WHEN jsonb_typeof(r.episodes) = 'array' "
        "THEN r.episodes ELSE jsonb_build_array(r.episodes) END) AS v "
        "WHERE v = ANY(%s::text[])) OR r.web_artifact_id = ANY(%s::bigint[]) FOR UPDATE",
        (id_texts, web_ids),
    ).fetchall()
    sole: list[str] = []
    shared: list[tuple[str, list[Any]]] = []
    for r in fact_rows:
        others = _other_episode_refs(r["episodes"], ids)
        if others:
            shared.append((r["uuid"], others))
        else:
            sole.append(r["uuid"])
    for uuid, others in shared:
        conn.execute(
            "UPDATE kg_relationships SET episodes = %s WHERE uuid = %s",
            (Json(others), uuid),
        )
    if sole:
        _clear_invalidated_by(conn, sole)
        conn.execute("DELETE FROM kg_relationships WHERE uuid = ANY(%s)", (sole,))
        retire_flags(conn, "fact", sole)
    if web_ids:  # the facts that stay are backed by a surviving turn: no page to cite
        conn.execute(
            "UPDATE kg_relationships SET web_artifact_id = NULL WHERE web_artifact_id = ANY(%s)",
            (web_ids,),
        )

    # --- timeline events -----------------------------------------------------
    # Chat events carry source_ref 'ep:<id>' (or 'ep:<id>#<k>' when one turn
    # yielded several — ingestion/timeline_gate.py). UNIQUE (source, source_ref)
    # makes the first telling's turn the row's owner. An event no other stored turn
    # backs goes with it: keeping it would go on serving the deleted turn's text. One a
    # later turn re-told (the reported_refs column) moves to the first such turn that
    # survives. Rows merged before reported_refs existed list no turns, so they are deleted.
    has_refs = REPORTED_REFS_PROBE.ready(conn)
    owned = conn.execute(
        "SELECT id, " + ("reported_refs" if has_refs else "'{}'::text[] AS reported_refs") + " "  # nosec B608 — a fixed fragment
        "FROM timeline_events WHERE source_ref = ANY(%s) OR source_ref LIKE ANY(%s) FOR UPDATE",
        ([f"ep:{i}" for i in episode_ids], [f"ep:{i}#%" for i in episode_ids]),
    ).fetchall()
    deleted_events = []
    for r in owned:
        survivors = [
            ep
            for ref in r["reported_refs"]
            if (ep := _episode_id_from_ref(ref)) is not None and ep not in ids
        ]
        if not any(_repoint_event(conn, r["id"], ep) for ep in survivors):
            deleted_events.append(r["id"])
    if deleted_events:
        conn.execute("DELETE FROM timeline_events WHERE id = ANY(%s)", (deleted_events,))
        retire_flags(conn, "timeline_event", [str(i) for i in deleted_events])
    if has_refs:
        # A turn that is gone has re-told nothing: a later cascade must not move an
        # event onto it.
        conn.execute(
            "UPDATE timeline_events SET reported_refs = ARRAY(SELECT x FROM "
            "unnest(reported_refs) x WHERE x <> ALL(%s)) WHERE reported_refs && %s",
            ([f"ep:{i}" for i in episode_ids],) * 2,
        )

    # --- tombstones, while the rows still say what the turns were ------------------------
    # Without the table (its migration is optional) there is nothing to record and the
    # turn can return; the probe logs that once.
    tombstoned = 0
    if DELETED_TURNS_PROBE.ready(conn):
        tombstoned = conn.execute(
            "INSERT INTO deleted_turns (span_id, session_id, content_md5, project, cwd) "
            "SELECT span_id, session_id, md5(content), project, metadata->>'cwd' "
            "FROM episodes WHERE id = ANY(%s)",
            (episode_ids,),
        ).rowcount

    # --- the episodes themselves (their extraction_queue rows follow via the 001 FK) ----
    conn.execute("DELETE FROM episodes WHERE id = ANY(%s)", (episode_ids,))

    retire_flags(conn, "episode", id_texts)
    return {
        "chunks_deleted": len(chunk_ids),
        "chunks_mixed_deleted": chunks_mixed,
        "facts_deleted": len(sole),
        "facts_unlinked": len(shared),
        "events_deleted": len(deleted_events),
        "web_deleted": len(web_ids) + archive_deleted,
        "tombstoned": tombstoned,
        "deleted_fact_ids": sole,
        "deleted_event_ids": deleted_events,
    }


def _delete_episode(db_url: str, episode_id: int) -> dict[str, Any] | None:
    """Hard-delete an episode and everything carrying ITS content, in one transaction.

    Returns the cascade summary the contract pins, or None when the episode doesn't
    exist (the route turns that into the 404)."""
    with pg_pool.connection(db_url, autocommit=False, row_factory=dict_row) as conn:
        with conn.transaction():
            row = conn.execute(
                "SELECT id FROM episodes WHERE id = %s FOR UPDATE", (episode_id,)
            ).fetchone()
            if row is None:
                return None
            summary = {
                "episode_id": episode_id,
                **delete_episode_content(conn, [episode_id]),
            }
            conn.execute(
                "INSERT INTO dashboard_audit (action, kind, item_id, detail) "
                "VALUES ('delete_episode', 'episode', %s, %s)",
                (str(episode_id), Json(summary)),
            )
            return summary


def _delete_fact(db_url: str, uuid: str) -> dict[str, Any] | None:
    """Hard-delete ONE KG edge by uuid. No cascade — a fact is a leaf; the only cleanup is
    the dangling supersession pointer. None when the uuid doesn't exist (→ 404)."""
    with pg_pool.connection(db_url, autocommit=False, row_factory=dict_row) as conn:
        with conn.transaction():
            row = conn.execute(
                "SELECT uuid FROM kg_relationships WHERE uuid = %s FOR UPDATE", (uuid,)
            ).fetchone()
            if row is None:
                return None
            links_cleared = _clear_invalidated_by(conn, [uuid])
            conn.execute("DELETE FROM kg_relationships WHERE uuid = %s", (uuid,))
            result = {
                "fact_id": uuid,
                "deleted": True,
                "supersession_links_cleared": links_cleared,
            }
            retire_flags(conn, "fact", [uuid])
            conn.execute(
                "INSERT INTO dashboard_audit (action, kind, item_id, detail) "
                "VALUES ('delete_fact', 'fact', %s, %s)",
                (uuid, Json(result)),
            )
            return result


def retire_flags(conn: psycopg.Connection[dict[str, Any]], kind: str, item_ids: list[str]) -> None:
    """Retire the active flag (if any) on items we just deleted, so the Flags list can't
    point at a row that no longer exists. The dashboard_audit trail is NOT touched — it is
    append-only and deliberately outlives the content."""
    conn.execute(
        "UPDATE dashboard_flags SET removed_at = now() "
        "WHERE kind = %s AND item_id = ANY(%s) AND removed_at IS NULL",
        (kind, item_ids),
    )


def register(mcp: Any, db_url: str, authorized: Callable[[Request], bool]) -> None:
    """Mount the flag routes and the two delete routes (admin gated)."""

    @mcp.custom_route("/dash/api/flags", methods=["GET"])  # type: ignore[misc]
    @route(authorized)
    def dash_flags(request: Request) -> dict[str, Any]:
        return _flags_list(db_url)

    @mcp.custom_route("/dash/api/flag", methods=["POST"])  # type: ignore[misc]
    @route(authorized, body=True)
    def dash_flag(request: Request, body: dict[str, Any]) -> dict[str, Any]:
        kind = body.get("kind")
        item_id = body.get("id")
        note = body.get("note")
        want = body.get("flagged")
        if kind not in _FLAG_KINDS:
            raise RouteError(f"invalid kind {kind!r}")
        if not item_id or not isinstance(item_id, str):
            raise RouteError("missing 'id' (item_id string)")
        if note is not None and not isinstance(note, str):
            raise RouteError("'note' must be a string")
        if want is not None and not isinstance(want, bool):
            raise RouteError("'flagged' must be a boolean")
        return {"status": "ok", "flagged": _set_flag(db_url, kind, item_id, note, want)}

    # Plural collection paths, unlike the singular phase-1 read routes — the DELETE verb
    # acts on the collection member. Contract §Phase 7 has the cascade table.

    @mcp.custom_route("/dash/api/episodes/{id}", methods=["DELETE"])  # type: ignore[misc]
    @route(authorized, not_found="episode not found")
    def dash_delete_episode(request: Request) -> dict[str, Any] | None:
        return _delete_episode(db_url, _episode_id(request))

    @mcp.custom_route("/dash/api/facts/{id}", methods=["DELETE"])  # type: ignore[misc]
    @route(authorized, not_found="fact not found")
    def dash_delete_fact(request: Request) -> dict[str, Any] | None:
        uuid = request.path_params["id"]
        if not uuid:
            raise RouteError("missing fact id")
        return _delete_fact(db_url, uuid)
