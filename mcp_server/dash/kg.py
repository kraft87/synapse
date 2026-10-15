"""Knowledge-graph views: the entity dossier and the graph explorer (phase 6)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import psycopg
from psycopg.rows import dict_row
from starlette.requests import Request

from ingestion import pg_pool
from ingestion.dedup import _normalize_name
from mcp_server.dash.common import (
    _GIST_SOURCE_MAX,
    _flag_set,
    _gist,
    _like_contains,
    _limit,
    _offset,
    _provenance,
)
from mcp_server.http_helpers import RouteError, _iso, route

_MENTIONS_PAGE = 20
_ENTITY_FACTS_PAGE = 200  # dossier facts per page; the example has ~15


# Graph explorer (phase 6). Typeahead + BFS neighborhood over kg_relationships. The 150
# hard cap mirrors the client's render budget (spec §4); when a neighborhood overflows we
# keep the highest-degree nodes so the important structure survives truncation. Every scan
# is bounded (LIMIT + a statement_timeout) — no unbounded walks on the ~75K-edge graph.
_GRAPH_TYPEAHEAD_DEFAULT = 10
_GRAPH_TYPEAHEAD_MAX = 25
_GRAPH_NODE_CAP = 150  # hard cap on rendered nodes (contract + spec §4)
# Edges need their own cap: the node cap alone let a dense hub return 150 nodes with
# ~2K interconnecting edges (the live "Synapse" entity, degree 707), which hung the
# browser tab in cose-bilkent layout. Rank seed-adjacent first, then most-retrieved,
# then newest; nodes orphaned by the cut are pruned (the seed always stays).
_GRAPH_EDGE_CAP = 500
_GRAPH_EDGE_SCAN_CAP = 6000  # per-level BFS + final-edge scan safety valve
_GRAPH_STMT_TIMEOUT_MS = 5000  # belt-and-suspenders bound on any single graph query


# kg_entities.degree is a one-off backfill (schema 017) that nothing maintains: stale for
# older entities, 0 for every one extraction wrote since. Reads count live edges instead.
# One scalar subquery per side keeps each count on kg_rel_pair / kg_rel_tgt (an src-OR-tgt
# predicate can use neither). `e` is the kg_entities alias of the surrounding query.
_LIVE_DEGREE_SQL = (
    "((SELECT count(*) FROM kg_relationships r WHERE r.owner_id = e.owner_id "
    "AND r.group_id = e.group_id AND r.src_uuid = e.uuid AND r.t_invalid IS NULL) "
    "+ (SELECT count(*) FROM kg_relationships r WHERE r.owner_id = e.owner_id "
    "AND r.group_id = e.group_id AND r.tgt_uuid = e.uuid AND r.t_invalid IS NULL))"
)


def _entity(
    db_url: str, uuid: str, mentions_offset: int, facts_offset: int = 0
) -> dict[str, Any] | None:
    with pg_pool.connection(db_url, row_factory=dict_row) as conn:
        ent = conn.execute(
            "SELECT uuid, owner_id, group_id, name, entity_subtype, summary, t_created "
            "FROM kg_entities WHERE uuid = %s",
            (uuid,),
        ).fetchone()
        if ent is None:
            return None

        flags = _flag_set(conn)
        # An entity's edges live in its own (owner_id, group_id): extraction and dedup run
        # per group. Naming both lets kg_rel_pair / kg_rel_tgt serve the src-OR-tgt reads.
        scope = {"u": uuid, "o": ent["owner_id"], "g": ent["group_id"]}
        touching = "r.owner_id = %(o)s AND r.group_id = %(g)s AND (r.src_uuid = %(u)s OR r.tgt_uuid = %(u)s)"

        # Facts = live + superseded edges touching the uuid, live first so a page cut never
        # hides a current fact behind superseded ones; the "other" endpoint's name comes from
        # a join on whichever side isn't this entity. Paged: stats.facts is the total.
        fact_rows = conn.execute(
            "SELECT r.uuid, r.name, r.fact, r.t_valid, r.t_invalid, r.episodes, "  # nosec B608 — {touching} is a fixed predicate; the ids are bound
            "       CASE WHEN r.src_uuid = %(u)s THEN r.tgt_uuid ELSE r.src_uuid END AS other_uuid, "
            "       oe.name AS other_name "
            "FROM kg_relationships r "
            "LEFT JOIN kg_entities oe "
            "  ON oe.uuid = CASE WHEN r.src_uuid = %(u)s THEN r.tgt_uuid ELSE r.src_uuid END "
            f"WHERE {touching} "
            "ORDER BY (r.t_invalid IS NOT NULL), r.t_valid DESC NULLS LAST, r.uuid "
            "OFFSET %(off)s LIMIT %(lim)s",
            {**scope, "off": facts_offset, "lim": _ENTITY_FACTS_PAGE},
        ).fetchall()

        stats = (
            conn.execute(
                "SELECT count(*) FILTER (WHERE t_invalid IS NULL) AS edges, "  # nosec B608 — {touching} is a fixed predicate; the ids are bound
                "       count(*) AS facts, "
                "       COALESCE(sum(retrieval_count), 0) AS served "
                f"FROM kg_relationships r WHERE {touching}",
                scope,
            ).fetchone()
            or {}
        )

        # Mentions = distinct episode ids across the touching edges' `episodes` arrays,
        # resolved to real episodes, newest first, paged. Non-numeric elements are skipped.
        # The CASE guard coerces NULL / non-array episodes to '[]' so the lateral never
        # errors on a malformed row (it would raise at execution, before any WHERE). The
        # ids are deduplicated before the episodes join, so no episode text is sorted.
        mentioned = (
            "SELECT elem.v::bigint FROM kg_relationships r "
            "CROSS JOIN LATERAL jsonb_array_elements_text("
            "  CASE WHEN jsonb_typeof(r.episodes) = 'array' THEN r.episodes ELSE '[]'::jsonb END"
            ") AS elem(v) "
            f"WHERE {touching} AND elem.v ~ '^[0-9]+$'"
        )
        total_row = (
            conn.execute(
                f"SELECT count(*) AS total FROM episodes e WHERE e.id IN ({mentioned})",  # nosec B608 — {mentioned} is a fixed subquery; the ids are bound
                scope,
            ).fetchone()
            or {}
        )
        mention_rows = conn.execute(
            f"SELECT e.id, e.created_at, left(e.content, {_GIST_SOURCE_MAX}) AS content "  # nosec B608 — {mentioned} is a fixed subquery, the width a constant; the ids are bound
            f"FROM episodes e WHERE e.id IN ({mentioned}) "
            "ORDER BY e.created_at DESC, e.id DESC OFFSET %(off)s LIMIT %(lim)s",
            {**scope, "off": mentions_offset, "lim": _MENTIONS_PAGE},
        ).fetchall()

        return {
            "entity": {
                "uuid": ent["uuid"],
                "name": ent["name"],
                # The wire keeps the name entity_type; the value is the taxonomy subtype
                # (schema 020), not the FalkorDB-era raw entity_type column.
                "entity_type": ent["entity_subtype"],
                "summary": ent["summary"],
                "degree": int(stats.get("edges") or 0),
                "created_at": _iso(ent["t_created"]),
            },
            "stats": {
                "edges": int(stats.get("edges") or 0),
                "served": int(stats.get("served") or 0),
                "facts": int(stats.get("facts") or 0),
            },
            "facts": [
                {
                    "uuid": r["uuid"],
                    "fact": r["fact"],
                    "name": r["name"],
                    "t_valid": _iso(r["t_valid"]),
                    "t_invalid": _iso(r["t_invalid"]),
                    "other": {"uuid": r["other_uuid"], "name": r["other_name"]},
                    "provenance_episode_id": _provenance(r["episodes"]),
                    "flagged": ("fact", r["uuid"]) in flags,
                }
                for r in fact_rows
            ],
            "mentions": {
                "items": [
                    {
                        "episode_id": r["id"],
                        "created_at": _iso(r["created_at"]),
                        "gist": _gist(r["content"]),
                    }
                    for r in mention_rows
                ],
                "offset": mentions_offset,
                "limit": _MENTIONS_PAGE,
                "total": int(total_row.get("total") or 0),
            },
        }


def _parse_as_of(raw: str | None) -> datetime | None:
    """Parse the as-of ISO timestamp, or None (unset / unparseable → no time filter)."""
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _graph_entities(db_url: str, q: str, limit: int) -> list[dict[str, Any]]:
    """Typeahead over entity names — shares the search endpoint's entity leg SQL
    (name ILIKE, live degree DESC). Global (no group scope): the graph is seeded explicitly."""
    if not q:
        return []
    with pg_pool.connection(db_url, row_factory=dict_row) as conn:
        rows = conn.execute(
            "SELECT e.uuid, e.name, e.entity_subtype, e.entity_supertype, "  # nosec B608
            f"       {_LIVE_DEGREE_SQL} AS degree FROM kg_entities e "
            "WHERE e.name ILIKE %s ORDER BY degree DESC, e.name LIMIT %s",
            (_like_contains(q), limit),
        ).fetchall()
        return [
            {
                "uuid": r["uuid"],
                "name": r["name"],
                "entity_type": r["entity_subtype"],
                "supertype": r["entity_supertype"],
                "degree": r["degree"],
            }
            for r in rows
        ]


def _resolve_seed(conn: psycopg.Connection[dict[str, Any]], entity: str) -> dict[str, Any] | None:
    """Resolve a seed given a uuid OR a name: exact uuid → exact normalized_name →
    best ILIKE match (highest live degree). None when nothing matches."""
    cols = "e.uuid, e.owner_id, e.group_id"
    row = conn.execute(f"SELECT {cols} FROM kg_entities e WHERE e.uuid = %s", (entity,)).fetchone()  # nosec B608
    if row:
        return row
    norm = _normalize_name(entity)
    if norm:
        row = conn.execute(
            f"SELECT {cols} FROM kg_entities e WHERE e.normalized_name = %s "  # nosec B608
            f"ORDER BY {_LIVE_DEGREE_SQL} DESC LIMIT 1",
            (norm,),
        ).fetchone()
        if row:
            return row
    return conn.execute(
        f"SELECT {cols} FROM kg_entities e WHERE e.name ILIKE %s "  # nosec B608
        f"ORDER BY {_LIVE_DEGREE_SQL} DESC, e.name LIMIT 1",
        (_like_contains(entity),),
    ).fetchone()


def _graph_neighborhood(
    db_url: str,
    entity: str,
    depth: int,
    as_of: datetime | None,
    limit: int,
    edge_cap: int = _GRAPH_EDGE_CAP,
) -> dict[str, Any] | None:
    """BFS from the resolved seed over kg_relationships (both directions), depth ≤ 2,
    hard-capped at `limit` (≤150) nodes AND `edge_cap` edges. Truncation keeps the
    highest-degree nodes / best-ranked edges (the seed is always kept) and sets
    truncated=True; nodes orphaned by the edge cut are pruned.

    as_of visibility (traversal AND returned edges use the SAME predicate): when set,
    exclude edges not-yet-valid (t_valid > as_of) but KEEP superseded edges (the client
    dashes them); when unset, return live + superseded and let the client scrub. Every
    scan is LIMIT-bounded and a statement_timeout guards against a pathological hub.
    """
    node_budget = max(1, min(limit, _GRAPH_NODE_CAP))
    depth = 2 if depth >= 2 else 1
    with pg_pool.connection(db_url, autocommit=False, row_factory=dict_row) as conn:
        # Bound every query of this read (SET can't bind a param). One transaction, with a
        # transaction-local timeout: a session-level one would outlive the borrow.
        conn.execute(
            "SELECT set_config('statement_timeout', %s, true)", (str(_GRAPH_STMT_TIMEOUT_MS),)
        )

        seed = _resolve_seed(conn, entity)
        if seed is None:
            return None
        seed_uuid = seed["uuid"]

        # Edges stay within the seed's (owner_id, group_id) — extraction and dedup run per
        # group — and naming both lets kg_rel_pair / kg_rel_tgt serve every
        # scan below. Plus the shared as-of edge-visibility fragment (empty when unset).
        edge_scope = "owner_id = %s AND group_id = %s"
        edge_params: list[Any] = [seed["owner_id"], seed["group_id"]]
        as_of_sql, as_of_params = "", []
        if as_of is not None:
            as_of_sql = " AND (t_valid IS NULL OR t_valid <= %s)"
            as_of_params = [as_of]

        # BFS both directions; one bounded query per level.
        visited: set[str] = {seed_uuid}
        frontier: list[str] = [seed_uuid]
        for _ in range(depth):
            if not frontier:
                break
            rows = conn.execute(
                "SELECT src_uuid, tgt_uuid FROM kg_relationships "  # nosec B608 — edge_scope and as_of_sql are fixed fragments; the values are bound
                f"WHERE {edge_scope} AND (src_uuid = ANY(%s) OR tgt_uuid = ANY(%s)){as_of_sql} "
                "LIMIT %s",
                (*edge_params, frontier, frontier, *as_of_params, _GRAPH_EDGE_SCAN_CAP),
            ).fetchall()
            nxt: list[str] = []
            for r in rows:
                for other in (r["src_uuid"], r["tgt_uuid"]):
                    if other not in visited:
                        visited.add(other)
                        nxt.append(other)
            frontier = nxt

        # Materialize real entity rows for the reached uuids (edge endpoints without an
        # entity row are dropped — edges aren't FKs, so a dangling endpoint has no node).
        ent_rows = conn.execute(
            "SELECT e.uuid, e.name, e.entity_subtype, e.entity_supertype, e.summary, "  # nosec B608
            f"       {_LIVE_DEGREE_SQL} AS degree FROM kg_entities e WHERE e.uuid = ANY(%s)",
            (list(visited),),
        ).fetchall()

        # Rank seed-first, then highest degree; truncate to the budget.
        ent_rows.sort(key=lambda r: (r["uuid"] != seed_uuid, -(r["degree"] or 0), r["name"] or ""))
        truncated = len(ent_rows) > node_budget
        if truncated:
            ent_rows = ent_rows[:node_budget]
        kept = [r["uuid"] for r in ent_rows]

        # Edges with BOTH endpoints kept, under the same as-of visibility.
        edge_rows = conn.execute(
            "SELECT uuid, src_uuid, tgt_uuid, name, fact, t_valid, t_invalid, episodes, "  # nosec B608 — edge_scope and as_of_sql are fixed fragments; the values are bound
            "       retrieval_count FROM kg_relationships "
            f"WHERE {edge_scope} AND src_uuid = ANY(%s) AND tgt_uuid = ANY(%s){as_of_sql} "
            "LIMIT %s",
            (*edge_params, kept, kept, *as_of_params, _GRAPH_EDGE_SCAN_CAP),
        ).fetchall()

        # Edge cap (see _GRAPH_EDGE_CAP): keep seed-adjacent edges first, then the
        # most-retrieved, then the newest; prune nodes the cut left edgeless.
        if len(edge_rows) > edge_cap:
            _epoch = datetime(1970, 1, 1, tzinfo=UTC)
            edge_rows.sort(
                key=lambda r: (
                    seed_uuid not in (r["src_uuid"], r["tgt_uuid"]),
                    -(r["retrieval_count"] or 0),
                    -(r["t_valid"] or _epoch).timestamp(),
                )
            )
            edge_rows = edge_rows[:edge_cap]
            connected = {seed_uuid}
            for r in edge_rows:
                connected.add(r["src_uuid"])
                connected.add(r["tgt_uuid"])
            ent_rows = [r for r in ent_rows if r["uuid"] in connected]
            truncated = True

        return {
            "nodes": [
                {
                    "uuid": r["uuid"],
                    "name": r["name"],
                    "entity_type": r["entity_subtype"],
                    # canonical coarse layer (020) — the client colors by this
                    "supertype": r["entity_supertype"],
                    "degree": r["degree"],
                    "summary": r["summary"],
                }
                for r in ent_rows
            ],
            "edges": [
                {
                    "uuid": r["uuid"],
                    "src": r["src_uuid"],
                    "tgt": r["tgt_uuid"],
                    "name": r["name"],
                    "fact": r["fact"],
                    "t_valid": _iso(r["t_valid"]),
                    "t_invalid": _iso(r["t_invalid"]),
                    "provenance_episode_id": _provenance(r["episodes"]),
                    "retrieval_count": r["retrieval_count"],
                }
                for r in edge_rows
            ],
            "truncated": truncated,
            "seed": seed_uuid,
        }


def register(mcp: Any, db_url: str, authorized: Callable[[Request], bool]) -> None:
    """Mount the entity dossier and the graph explorer (admin gated)."""

    @mcp.custom_route("/dash/api/entity/{uuid}", methods=["GET"])  # type: ignore[misc]
    @route(authorized, not_found="entity not found")
    def dash_entity(request: Request) -> dict[str, Any] | None:
        m_off = _offset(request.query_params.get("mentions_offset"))
        f_off = _offset(request.query_params.get("facts_offset"))
        return _entity(db_url, request.path_params["uuid"], m_off, f_off)

    @mcp.custom_route("/dash/api/graph/entities", methods=["GET"])  # type: ignore[misc]
    @route(authorized)
    def dash_graph_entities(request: Request) -> list[dict[str, Any]]:
        q = (request.query_params.get("q") or "").strip()
        limit = _limit(
            request.query_params.get("limit"), _GRAPH_TYPEAHEAD_DEFAULT, _GRAPH_TYPEAHEAD_MAX
        )
        return _graph_entities(db_url, q, limit)

    @mcp.custom_route("/dash/api/graph/neighborhood", methods=["GET"])  # type: ignore[misc]
    @route(authorized, not_found="entity not found")
    def dash_graph_neighborhood(request: Request) -> dict[str, Any] | None:
        qp = request.query_params
        entity = (qp.get("entity") or "").strip()
        if not entity:
            raise RouteError("missing 'entity'")
        depth = 2 if qp.get("depth") == "2" else 1
        as_of = _parse_as_of(qp.get("as_of"))
        limit = _limit(qp.get("limit"), _GRAPH_NODE_CAP, _GRAPH_NODE_CAP)
        return _graph_neighborhood(db_url, entity, depth, as_of, limit)
