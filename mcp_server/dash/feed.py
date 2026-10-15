"""Catalog, the paged feed, episode detail (with the facts and events derived from it) and
the session view: the dashboard's main reading surface.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import psycopg
from psycopg.rows import dict_row
from starlette.requests import Request

from ingestion import pg_pool
from mcp_server.dash.common import (
    _GIST_SOURCE_MAX,
    _SAL_MAP,
    _decode_cursor,
    _domain_filter,
    _encode_cursor,
    _episode_id,
    _episode_id_from_ref,
    _flag_set,
    _gist,
    _label_filter,
    _limit,
    _provenance,
    _TtlCache,
)
from mcp_server.http_helpers import _iso, route

# Server-side caps (contract: "All list endpoints enforce server-side limit caps").
_FEED_LIMIT_DEFAULT = 30
_FEED_LIMIT_MAX = 100
_SESSION_CAP = 500
_DERIVED_CAP = 200
_CATALOG_TTL_S = 300  # ~5 min in-process cache


def _catalog(db_url: str) -> dict[str, Any]:
    with pg_pool.connection(db_url, row_factory=dict_row) as conn:
        # NULL / '' project collapses to the single bucket name "untagged" (contract);
        # sources get the same defensive treatment so a NULL never lands as a null name.
        projects = conn.execute(
            "SELECT COALESCE(NULLIF(project, ''), 'untagged') AS name, count(*) AS count "
            "FROM episodes GROUP BY 1 ORDER BY count DESC, name"
        ).fetchall()
        sources = conn.execute(
            "SELECT COALESCE(NULLIF(source, ''), 'untagged') AS name, count(*) AS count "
            "FROM episodes GROUP BY 1 ORDER BY count DESC, name"
        ).fetchall()
        # group_ids: the distinct group scopes across the group-filterable surfaces
        # (facts.group_id, entities.group_id, timeline.domain), as a plain string array.
        gids = conn.execute(
            "SELECT DISTINCT g FROM ("
            "  SELECT group_id AS g FROM kg_relationships "
            "  UNION SELECT group_id FROM kg_entities "
            "  UNION SELECT domain FROM timeline_events"
            ") s WHERE g IS NOT NULL AND g <> '' ORDER BY g"
        ).fetchall()
        return {
            "projects": [{"name": r["name"], "count": r["count"]} for r in projects],
            "sources": [{"name": r["name"], "count": r["count"]} for r in sources],
            "group_ids": [r["g"] for r in gids],
        }


# The per-type SELECT column lists and row→FeedItem mappers are shared between the
# paged /dash/api/feed endpoint (below) and the SSE stream's single-row hydration
# (Phase 3, further down) so BOTH produce byte-identical wire shapes from ONE code path.
# The endpoint appends a keyset WHERE + ORDER BY + LIMIT to the SELECT; the stream appends
# a "WHERE <pk> = %s". The mappers emit the final item minus `flagged` (added after the
# per-request flag set is loaded) and minus the internal `_ts`/`_idstr` merge-sort keys.

_EP_SELECT = (
    "SELECT id, created_at, project, source, session_id, sequence, "
    f"left(content, {_GIST_SOURCE_MAX}) AS content FROM episodes"
)
# A fact's feed time is t_created, its write time (schema 017); the FalkorDB-era
# created_at mirror is not read.
_FACT_SELECT = (
    "SELECT r.uuid, r.t_created, r.group_id, r.fact, r.t_valid, r.t_invalid, r.episodes, "
    "       se.name AS src_name, te.name AS tgt_name "
    "FROM kg_relationships r "
    "LEFT JOIN kg_entities se ON se.uuid = r.src_uuid "
    "LEFT JOIN kg_entities te ON te.uuid = r.tgt_uuid"
)
_TL_SELECT = (
    "SELECT id, ingested_at, project, fact, t_valid, source, source_ref, salience, domain "
    "FROM timeline_events"
)


def _episode_item(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "episode",
        "id": str(r["id"]),
        "ts": _iso(r["created_at"]),
        "project": r["project"],
        "source": r["source"],
        "gist": _gist(r["content"]),
        "data": {"session_id": r["session_id"], "sequence": r["sequence"]},
    }


def _fact_item(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "fact",
        "id": r["uuid"],
        "ts": _iso(r["t_created"]),
        "group_id": r["group_id"],
        "gist": _gist(r["fact"]),
        "data": {
            "fact": r["fact"],
            "src_name": r["src_name"],
            "tgt_name": r["tgt_name"],
            "t_valid": _iso(r["t_valid"]),
            "t_invalid": _iso(r["t_invalid"]),
            "provenance_episode_id": _provenance(r["episodes"]),
        },
    }


def _timeline_item(r: dict[str, Any]) -> dict[str, Any]:
    sal = _SAL_MAP.get(int(r["salience"]) if r["salience"] is not None else 1, 0.6)
    return {
        "type": "timeline_event",
        "id": str(r["id"]),
        "ts": _iso(r["ingested_at"]),
        "project": r["project"],
        # domain is the timeline's group scope (schema 038); exposed as group_id so the live
        # SSE client can apply the group filter uniformly across all three feed types.
        "group_id": r["domain"],
        "sal": sal,
        "gist": _gist(r["fact"]),
        "data": {
            "fact": r["fact"],
            "t_valid": _iso(r["t_valid"]),
            "source": r["source"],
            "episode_id": _episode_id_from_ref(r["source_ref"]),
        },
    }


def _feed(
    db_url: str,
    cursor: str | None,
    limit: int,
    project: str | None,
    group_id: str | None,
    source: str | None,
) -> dict[str, Any]:
    """Reverse-chron keyset merge of episodes + KG facts + timeline events.

    Three bounded per-type SELECTs (each ORDER BY ts DESC, id DESC LIMIT `limit`, with a
    keyset predicate when a cursor is present), merged in Python and cut to `limit`. The
    merge key and the SQL predicate are the SAME total order — (ts DESC, id-as-text DESC)
    — so pages don't drop or duplicate rows. Filters apply only where a type has the
    column (contract): episodes have project+source but no group_id — the group filter
    must NOT exclude them; facts have group_id but no project/source; timeline has
    project and (schema 038) a domain column that stands in for group_id.
    """
    cur = _decode_cursor(cursor) if cursor else None
    cts, cid = cur if cur else (None, None)

    with pg_pool.connection(db_url, row_factory=dict_row) as conn:
        candidates: list[dict[str, Any]] = []

        # --- episodes (ts = created_at). Filters: project, source. No group_id column. ---
        ep_where = ["1=1"]
        ep_params: list[Any] = []
        for col, value in (("project", project), ("source", source)):
            if value is not None:
                clause, args = _label_filter(col, value)
                ep_where.append(clause)
                ep_params += args
        if cts is not None:
            ep_where.append(
                "(created_at < %s::timestamptz OR (created_at = %s::timestamptz AND id::text < %s))"
            )
            ep_params += [cts, cts, cid]
        ep_rows = conn.execute(
            f"{_EP_SELECT} WHERE {' AND '.join(ep_where)} "
            "ORDER BY created_at DESC, id::text DESC LIMIT %s",
            (*ep_params, limit),
        ).fetchall()
        for r in ep_rows:
            item = _episode_item(r)
            item["_ts"] = r["created_at"]
            item["_idstr"] = str(r["id"])
            candidates.append(item)

        # --- facts (ts = t_created). Filter: group_id. No project/source column. ---
        # Columns are r.-qualified: the entity joins below also expose t_created/group_id.
        # The row-wise keyset and ORDER BY match kg_rel_t_created_idx (a later optional
        # migration); without that index the same query sorts.
        f_where = ["r.t_created IS NOT NULL"]
        f_params: list[Any] = []
        if group_id is not None:
            f_where.append("r.group_id = %s")
            f_params.append(group_id)
        if cts is not None:
            f_where.append("(r.t_created, r.uuid) < (%s::timestamptz, %s)")
            f_params += [cts, cid]
        fact_rows = conn.execute(
            f"{_FACT_SELECT} WHERE {' AND '.join(f_where)} "
            "ORDER BY r.t_created DESC, r.uuid DESC LIMIT %s",
            (*f_params, limit),
        ).fetchall()
        for r in fact_rows:
            item = _fact_item(r)
            item["_ts"] = r["t_created"]
            item["_idstr"] = str(r["uuid"])
            candidates.append(item)

        # --- timeline (ts = ingested_at). Filters: project, and domain for group_id. ---
        t_where = ["1=1"]
        t_params: list[Any] = []
        if project is not None:
            clause, args = _label_filter("project", project)
            t_where.append(clause)
            t_params += args
        if group_id is not None:
            # 038 added `domain` (technical/personal) as the timeline's group scope; the
            # contract's feed-filter line names only project for timeline, but the domain
            # column IS a group_id-equivalent, so the group filter applies here too.
            clause, args = _domain_filter(group_id)
            t_where.append(clause)
            t_params += args
        if cts is not None:
            t_where.append(
                "(ingested_at < %s::timestamptz OR (ingested_at = %s::timestamptz AND id::text < %s))"
            )
            t_params += [cts, cts, cid]
        tl_rows = conn.execute(
            f"{_TL_SELECT} WHERE {' AND '.join(t_where)} "
            "ORDER BY ingested_at DESC, id::text DESC LIMIT %s",
            (*t_params, limit),
        ).fetchall()
        for r in tl_rows:
            item = _timeline_item(r)
            item["_ts"] = r["ingested_at"]
            item["_idstr"] = str(r["id"])
            candidates.append(item)

        # Merge on the same total order the per-type keyset used, cut to limit.
        candidates.sort(key=lambda c: (c["_ts"], c["_idstr"]), reverse=True)
        emitted = candidates[:limit]

        flags = _flag_set(conn)
        for item in emitted:
            item["flagged"] = (item["type"], item["id"]) in flags

        next_cursor = None
        # A full page means there may be more; a short page means every per-type stream
        # is drained (union < limit), so we stop.
        if len(emitted) == limit and emitted:
            last = emitted[-1]
            next_cursor = _encode_cursor(_iso(last["_ts"]) or "", last["type"], last["_idstr"])

        for item in emitted:
            item.pop("_ts", None)
            item.pop("_idstr", None)
        return {"items": emitted, "next_cursor": next_cursor}


def _episode(db_url: str, episode_id: int) -> dict[str, Any] | None:
    with pg_pool.connection(db_url, row_factory=dict_row) as conn:
        row = conn.execute(
            "SELECT id, session_id, sequence, project, source, platform, model, created_at, "
            "       human_turn, assistant_turn, content "
            "FROM episodes WHERE id = %s",
            (episode_id,),
        ).fetchone()
        if row is None:
            return None
        flagged = conn.execute(
            "SELECT 1 FROM dashboard_flags "
            "WHERE kind = 'episode' AND item_id = %s AND removed_at IS NULL",
            (str(episode_id),),
        ).fetchone()
        return {
            "id": row["id"],
            "session_id": row["session_id"],
            "sequence": row["sequence"],
            "project": row["project"],
            "source": row["source"],
            "platform": row["platform"],
            "model": row["model"],
            "created_at": _iso(row["created_at"]),
            "flagged": flagged is not None,
            "human_turn": row["human_turn"],
            "assistant_turn": row["assistant_turn"],
            "content": row["content"],
        }


def _episode_derived(db_url: str, episode_id: int) -> dict[str, Any]:
    with pg_pool.connection(db_url, row_factory=dict_row) as conn:
        # Facts whose `episodes` jsonb array contains this id — as a number OR (defensively)
        # a numeric string, since the extractor's array element type isn't guaranteed.
        facts = conn.execute(
            "SELECT uuid, fact, group_id, t_valid, t_invalid FROM kg_relationships "
            "WHERE episodes @> to_jsonb(%s::bigint) OR episodes @> to_jsonb(%s::text) "
            "ORDER BY t_valid DESC NULLS LAST, uuid LIMIT %s",
            (episode_id, str(episode_id), _DERIVED_CAP),
        ).fetchall()
        # A turn's events carry 'ep:<id>', then 'ep:<id>#<k>' (ingestion/timeline_gate.py).
        events = conn.execute(
            "SELECT id, fact, t_valid, salience FROM timeline_events "
            "WHERE source_ref = %s OR source_ref LIKE %s ORDER BY t_valid DESC LIMIT %s",
            (f"ep:{episode_id}", f"ep:{episode_id}#%", _DERIVED_CAP),
        ).fetchall()
        return {
            "facts": [
                {
                    "uuid": r["uuid"],
                    "fact": r["fact"],
                    "group_id": r["group_id"],
                    "t_valid": _iso(r["t_valid"]),
                    "t_invalid": _iso(r["t_invalid"]),
                }
                for r in facts
            ],
            "timeline_events": [
                {
                    "id": r["id"],
                    "fact": r["fact"],
                    "t_valid": _iso(r["t_valid"]),
                    "salience": r["salience"],
                }
                for r in events
            ],
        }


_SESSION_COLS = "id, sequence, created_at, project, source, human_turn, assistant_turn"


def _session_rows(
    conn: psycopg.Connection[dict[str, Any]], session_id: str, highlight: Any
) -> list[dict[str, Any]]:
    """Up to _SESSION_CAP turns of the session, in sequence order: the first ones, or,
    when ``highlight`` is a turn of this session, a window around it (about half before,
    sliding toward whichever side has the turns) so the drawer can show the turn it was
    opened for however long the session is."""
    cap = _SESSION_CAP
    hl = None
    if isinstance(highlight, int):
        hl = conn.execute(
            "SELECT sequence FROM episodes WHERE id = %s AND session_id = %s",
            (highlight, session_id),
        ).fetchone()
    if hl is None:
        return conn.execute(
            f"SELECT {_SESSION_COLS} FROM episodes "  # nosec B608 - fixed column list
            "WHERE session_id = %s ORDER BY sequence LIMIT %s",
            (session_id, cap),
        ).fetchall()
    after = conn.execute(
        f"SELECT {_SESSION_COLS} FROM episodes "  # nosec B608 - fixed column list
        "WHERE session_id = %s AND sequence >= %s ORDER BY sequence LIMIT %s",
        (session_id, hl["sequence"], cap),
    ).fetchall()
    before = conn.execute(
        f"SELECT {_SESSION_COLS} FROM episodes "  # nosec B608 - fixed column list
        "WHERE session_id = %s AND sequence < %s ORDER BY sequence DESC LIMIT %s",
        (session_id, hl["sequence"], cap - min(len(after), cap - cap // 2)),
    ).fetchall()
    return before[::-1] + after[: cap - len(before)]


def _session(db_url: str, session_id: str, highlight: Any) -> dict[str, Any] | None:
    """The session drawer's payload, or None when no turn is stored under the id (the route
    turns that into the 404). ``total`` counts the whole session: ``episodes`` is only the
    window of it that _SESSION_CAP allows."""
    with pg_pool.connection(db_url, row_factory=dict_row) as conn:
        rows = _session_rows(conn, session_id, highlight)
        if not rows:
            return None
        first = rows[0]
        total = conn.execute(
            "SELECT count(*) AS n FROM episodes WHERE session_id = %s", (session_id,)
        ).fetchone()
        return {
            "session_id": session_id,
            "project": first["project"],
            "source": first["source"],
            "total": total["n"] if total else len(rows),
            "highlight": highlight,
            "episodes": [
                {
                    "id": r["id"],
                    "sequence": r["sequence"],
                    "created_at": _iso(r["created_at"]),
                    "human_turn": r["human_turn"],
                    "assistant_turn": r["assistant_turn"],
                }
                for r in rows
            ],
        }


def register(mcp: Any, db_url: str, authorized: Callable[[Request], bool]) -> None:
    """Mount catalog, feed, episode detail and session (admin gated)."""
    # Contract: the catalog is cached ~5 min in-process.
    catalog_cache = _TtlCache(_CATALOG_TTL_S)

    @mcp.custom_route("/dash/api/catalog", methods=["GET"])  # type: ignore[misc]
    @route(authorized)
    def dash_catalog(request: Request) -> dict[str, Any]:
        return catalog_cache.get(lambda: _catalog(db_url))

    @mcp.custom_route("/dash/api/feed", methods=["GET"])  # type: ignore[misc]
    @route(authorized)
    def dash_feed(request: Request) -> dict[str, Any]:
        qp = request.query_params
        limit = _limit(qp.get("limit"), _FEED_LIMIT_DEFAULT, _FEED_LIMIT_MAX)
        cursor = qp.get("cursor") or None
        project = qp.get("project") or None
        group_id = qp.get("group_id") or None
        source = qp.get("source") or None
        return _feed(db_url, cursor, limit, project, group_id, source)

    @mcp.custom_route("/dash/api/episode/{id}", methods=["GET"])  # type: ignore[misc]
    @route(authorized, not_found="episode not found")
    def dash_episode(request: Request) -> dict[str, Any] | None:
        return _episode(db_url, _episode_id(request))

    @mcp.custom_route("/dash/api/episode/{id}/derived", methods=["GET"])  # type: ignore[misc]
    @route(authorized)
    def dash_episode_derived(request: Request) -> dict[str, Any]:
        return _episode_derived(db_url, _episode_id(request))

    @mcp.custom_route("/dash/api/session/{id}", methods=["GET"])  # type: ignore[misc]
    @route(authorized, not_found="session not found")
    def dash_session(request: Request) -> dict[str, Any] | None:
        raw_hl = request.query_params.get("highlight")
        highlight: Any = None
        if raw_hl is not None:
            highlight = int(raw_hl) if raw_hl.isdigit() else raw_hl
        return _session(db_url, request.path_params["id"], highlight)
