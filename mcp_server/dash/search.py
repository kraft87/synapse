"""Search across episodes, facts, entities and events."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from psycopg.rows import dict_row
from starlette.requests import Request

from ingestion import pg_pool
from ingestion.db_connection import bm25_query_text
from mcp_server.dash.common import (
    _SNIPPET_MAX,
    _count,
    _domain_filter,
    _episode_id_from_ref,
    _label_filter,
    _like_contains,
    _limit,
    _offset,
    _provenance,
)
from mcp_server.dash.kg import _LIVE_DEGREE_SQL
from mcp_server.http_helpers import _iso, route

_SEARCH_LIMIT_DEFAULT = 20
_SEARCH_LIMIT_MAX = 50
_SEARCH_TYPES = ("episodes", "facts", "entities", "events")


def _search(
    db_url: str,
    q: str,
    type_: str,
    offset: int,
    limit: int,
    project: str | None,
    group_id: str | None,
    source: str | None,
) -> dict[str, Any]:
    """BM25 (ParadeDB @@@) for episodes/facts/events, ILIKE-on-name for entities.

    total_by_type is computed for all four tabs on a query's first page (offset 0) and left
    out of later pages, which the client pages with the totals it already holds; `hits`
    only for the requested type. Snippet is the first 240 chars of the matched text — paradedb.snippet() with
    real term highlighting is a later upgrade.
    """
    empty = {
        "hits": [],
        "total_by_type": dict.fromkeys(_SEARCH_TYPES, 0),
        "offset": offset,
        "limit": limit,
    }
    if not q:
        return empty
    safe = bm25_query_text(q)
    like = _like_contains(q)

    with pg_pool.connection(db_url, row_factory=dict_row) as conn:
        # --- per-type filter fragments (applied to both count and hit queries) ---
        ep_filt: list[str] = []
        ep_p: list[Any] = []
        for col, value in (("project", project), ("source", source)):
            if value is not None:
                clause, args = _label_filter(col, value)
                ep_filt.append(clause)
                ep_p += args
        ep_and = "".join(f" AND {c}" for c in ep_filt)

        f_and, f_p = "", []
        if group_id is not None:
            f_and, f_p = " AND group_id = %s", [group_id]

        ev_filt: list[str] = []
        ev_p: list[Any] = []
        if project is not None:
            clause, args = _label_filter("project", project)
            ev_filt.append(clause)
            ev_p += args
        if group_id is not None:
            clause, args = _domain_filter(group_id)
            ev_filt.append(clause)
            ev_p += args
        ev_and = "".join(f" AND {c}" for c in ev_filt)

        en_and, en_p = "", []
        if group_id is not None:
            en_and, en_p = " AND group_id = %s", [group_id]

        # --- total_by_type: all four counts, first page only ---
        total: dict[str, int] | None = None
        if offset == 0:
            total = dict.fromkeys(_SEARCH_TYPES, 0)
            if safe:
                total["episodes"] = _count(
                    conn,
                    f"SELECT count(*) AS c FROM episodes WHERE id @@@ paradedb.match('content', %s){ep_and}",
                    (safe, *ep_p),
                )
                total["facts"] = _count(
                    conn,
                    f"SELECT count(*) AS c FROM kg_relationships WHERE id @@@ paradedb.match('fact', %s){f_and}",
                    (safe, *f_p),
                )
                total["events"] = _count(
                    conn,
                    f"SELECT count(*) AS c FROM timeline_events WHERE id @@@ paradedb.match('fact', %s){ev_and}",
                    (safe, *ev_p),
                )
            total["entities"] = _count(
                conn,
                f"SELECT count(*) AS c FROM kg_entities WHERE name ILIKE %s{en_and}",
                (like, *en_p),
            )

        # --- hits for the requested type only ---
        hits: list[dict[str, Any]] = []
        if type_ == "episodes" and safe:
            for r in conn.execute(
                f"SELECT id, left(content, {_SNIPPET_MAX}) AS snippet, project, source, "  # nosec B608 — ep_and joins fixed column predicates and the snippet width is a constant; the values are bound
                "       created_at, session_id, paradedb.score(id) AS sc FROM episodes "
                f"WHERE id @@@ paradedb.match('content', %s){ep_and} "
                "ORDER BY sc DESC, id DESC OFFSET %s LIMIT %s",
                (safe, *ep_p, offset, limit),
            ).fetchall():
                hits.append(
                    {
                        "type": "episodes",
                        "id": str(r["id"]),
                        "snippet": r["snippet"] or "",
                        "meta": {
                            "project": r["project"],
                            "source": r["source"],
                            "ts": _iso(r["created_at"]),
                            "session_id": r["session_id"],
                        },
                    }
                )
        elif type_ == "facts" and safe:
            for r in conn.execute(
                "SELECT uuid, fact, group_id, t_valid, t_invalid, t_created, episodes, "  # nosec B608 — f_and is a fixed predicate; the values are bound
                "       paradedb.score(id) AS sc FROM kg_relationships "
                f"WHERE id @@@ paradedb.match('fact', %s){f_and} "
                "ORDER BY sc DESC, uuid DESC OFFSET %s LIMIT %s",
                (safe, *f_p, offset, limit),
            ).fetchall():
                hits.append(
                    {
                        "type": "facts",
                        "id": r["uuid"],
                        "snippet": (r["fact"] or "")[:_SNIPPET_MAX],
                        # episode_id (fact provenance) lets the client deep-link a hit to
                        # the episode overlay; contract pins only episodes/entity hit meta,
                        # so this extra field is additive.
                        "meta": {
                            "group_id": r["group_id"],
                            "t_valid": _iso(r["t_valid"]),
                            "t_invalid": _iso(r["t_invalid"]),
                            "ts": _iso(r["t_created"]),
                            "episode_id": _provenance(r["episodes"]),
                        },
                    }
                )
        elif type_ == "events" and safe:
            for r in conn.execute(
                "SELECT id, fact, project, t_valid, source, source_ref, ingested_at, "  # nosec B608 — ev_and joins fixed column predicates; the values are bound
                "       paradedb.score(id) AS sc FROM timeline_events "
                f"WHERE id @@@ paradedb.match('fact', %s){ev_and} "
                "ORDER BY sc DESC, id DESC OFFSET %s LIMIT %s",
                (safe, *ev_p, offset, limit),
            ).fetchall():
                hits.append(
                    {
                        "type": "events",
                        "id": str(r["id"]),
                        "snippet": (r["fact"] or "")[:_SNIPPET_MAX],
                        "meta": {
                            "project": r["project"],
                            "t_valid": _iso(r["t_valid"]),
                            "source": r["source"],
                            "ts": _iso(r["ingested_at"]),
                            "episode_id": _episode_id_from_ref(r["source_ref"]),
                            # the only provenance a git-sourced event has (a SHA)
                            "source_ref": r["source_ref"],
                        },
                    }
                )
        elif type_ == "entities":
            for r in conn.execute(
                "SELECT e.uuid, e.name, e.entity_subtype, e.summary, "  # nosec B608
                f"       {_LIVE_DEGREE_SQL} AS degree FROM kg_entities e "
                f"WHERE e.name ILIKE %s{en_and} "
                "ORDER BY degree DESC, e.name OFFSET %s LIMIT %s",
                (like, *en_p, offset, limit),
            ).fetchall():
                hits.append(
                    {
                        "type": "entities",
                        "id": r["uuid"],
                        "snippet": ((r["summary"] or r["name"]) or "")[:_SNIPPET_MAX],
                        "meta": {
                            "name": r["name"],
                            "entity_type": r["entity_subtype"],
                            "degree": r["degree"],
                        },
                    }
                )

        page = {"hits": hits, "offset": offset, "limit": limit}
        return page if total is None else {**page, "total_by_type": total}


def register(mcp: Any, db_url: str, authorized: Callable[[Request], bool]) -> None:
    """Mount search (admin gated)."""

    @mcp.custom_route("/dash/api/search", methods=["GET"])  # type: ignore[untyped-decorator]
    @route(authorized)
    def dash_search(request: Request) -> dict[str, Any]:
        qp = request.query_params
        q = (qp.get("q") or "").strip()
        type_ = qp.get("type") or "episodes"
        if type_ not in _SEARCH_TYPES:
            type_ = "episodes"
        offset = _offset(qp.get("offset"))
        limit = _limit(qp.get("limit"), _SEARCH_LIMIT_DEFAULT, _SEARCH_LIMIT_MAX)
        project = qp.get("project") or None
        group_id = qp.get("group_id") or None
        source = qp.get("source") or None
        return _search(db_url, q, type_, offset, limit, project, group_id, source)
