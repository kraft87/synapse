"""Operator dashboard routes (issue #12) — the server half of the /dash surface.

Serves ONE React/esbuild single-page bundle (web/dist, built by the other agent's
web/ tree) plus a small read/flag API over the existing memory tables. The wire
contract is pinned in docs/dashboard-contract.md; this module builds exactly to it.

Two auth postures, by design (mirrors the contract's Namespace & auth section):
  * The static routes (/dash, /dash/app.js, /dash/assets/*) are UNAUTHENTICATED — the
    bundle is public code carrying no memory, so gating it buys nothing and would break
    the paste-once login screen (which needs to load before it has a token).
  * Every /dash/api/* route is machine-token gated through the same ``authorized`` seam
    the other route modules use (custom routes bypass FastMCP's auth middleware by
    design — issue #3704 — so each handler checks explicitly). 401 body is the contract's
    {"status":"error","detail":"unauthorized"}.

Shape rules shared by every api route: PG work runs in a threadpool on ONE short-lived
psycopg connection per request, and the boundary is fail-soft — an exception becomes a
500 JSON and is logged, never raised past the JSONResponse. Reads are owner-agnostic
(an operator sees the whole store); the only group_id/project/source scoping is the
per-endpoint filtering the contract spells out.

This is the ONLY writable surface the dashboard adds: dashboard_flags + dashboard_audit
(schema 042). Everything else is read-only over episodes / the KG / timeline /
preferences / notes.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse

# Phase 2b: the two proposal lanes own their state transitions + side effects; the
# dashboard REUSES their _proposal_act / _proposal_detail rather than reimplementing
# them (see docs/dashboard-contract.md §"Phase 2b").
from mcp_server.config_sync_routes import _proposal_act as _config_proposal_act
from mcp_server.config_sync_routes import _proposal_detail as _config_proposal_detail
from mcp_server.skill_sync_routes import _proposal_act as _skill_proposal_act
from mcp_server.skill_sync_routes import _proposal_detail as _skill_proposal_detail

logger = logging.getLogger(__name__)

# The built bundle lives at <repo>/web/dist. Module-level so tests can monkeypatch it
# to a throwaway dir; the handlers read it as a global at request time (not captured
# into register()'s closure) so the patch takes effect.
_DIST_DIR = Path(__file__).resolve().parent.parent / "web" / "dist"

# Server-side caps (contract: "All list endpoints enforce server-side limit caps").
_FEED_LIMIT_DEFAULT = 30
_FEED_LIMIT_MAX = 100
_SEARCH_LIMIT_DEFAULT = 20
_SEARCH_LIMIT_MAX = 50
_MENTIONS_PAGE = 20
_SESSION_CAP = 500
_RECALL_HISTORY_DEFAULT = 50
_RECALL_HISTORY_MAX = 200
_ENTITY_FACTS_CAP = 200  # bound the dossier payload; the example has ~15
_DERIVED_CAP = 200
_GIST_MAX = 200
_SNIPPET_MAX = 240
_CATALOG_TTL_S = 300  # ~5 min in-process cache

_FLAG_KINDS = ("episode", "fact", "timeline_event", "preference", "note")
_SEARCH_TYPES = ("episodes", "facts", "entities", "events")

# Phase 3 (SSE live stream). The LISTEN worker hydrates each NOTIFY into a full FeedItem and
# appends it to a ring buffer; the /dash/api/stream route replays from it on resume and streams
# live. Feed-item event names by type; every other tunable for the stream loop.
_STREAM_EVENT_NAMES = {
    "episode": "new_episode",
    "fact": "new_fact",
    "timeline_event": "new_timeline_event",
}
_STREAM_CHANNEL = "dash_feed"  # matches schema/043_dash_notify.sql
_STREAM_BUFFER_CAP = 512  # ring-buffer slots (contract: 512-event replay window)
_STREAM_POLL_S = 0.25  # per-connection buffer poll cadence (new-event latency ceiling)
_STREAM_STATUS_S = 1.0  # processing_status emit cadence (contract: ~1/s)
_STREAM_HEARTBEAT_S = 15.0  # comment heartbeat so idle proxies don't kill the stream

# Proposals (phase 2b): per-lane row cap for the unified list, and the review-relevant
# statuses. 'observe' (pre-graduation) and skills' 'retired' (decayed) are NOT proposals
# an operator reviews, so the unified list excludes them (the nav badge still only counts
# 'proposed'). skills terminal = 'promoted' (fs mv), config terminal = 'applied' (disk write) —
# both reached OUTSIDE the dashboard; the dashboard only moves proposed→accepted / →rejected.
_PROPOSALS_LANE_CAP = 200
_SKILL_REVIEW_STATUSES = ("proposed", "accepted", "promoted", "rejected")
_CONFIG_REVIEW_STATUSES = ("proposed", "accepted", "applied", "rejected")
_PROVENANCE_CAP = 40  # bound the best-effort session→episode resolution

# Long-cache immutable fonts (hashed filenames from the build); no-cache the rest.
_FONT_EXTS = {".woff2", ".woff", ".ttf", ".otf"}
_CONTENT_TYPES = {
    ".js": "application/javascript",
    ".css": "text/css",
    ".woff2": "font/woff2",
    ".woff": "font/woff",
    ".ttf": "font/ttf",
    ".otf": "font/otf",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".ico": "image/x-icon",
    ".json": "application/json",
    ".map": "application/json",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _iso(dt: Any) -> str | None:
    return dt.isoformat() if dt is not None else None


def _gist(text: str | None) -> str:
    """First non-empty line, internal whitespace collapsed, hard-capped at 200 chars."""
    if not text:
        return ""
    for line in text.splitlines():
        collapsed = " ".join(line.split())
        if collapsed:
            return collapsed[:_GIST_MAX]
    return ""


def _provenance(episodes: Any) -> int | None:
    """First element of a fact's ``episodes`` jsonb array as an int, else None.

    psycopg parses jsonb to a Python list; the array holds source episode ids as
    numbers or (defensively) numeric strings."""
    if isinstance(episodes, list) and episodes:
        first = episodes[0]
        if isinstance(first, bool):  # bool is an int subclass — exclude it
            return None
        if isinstance(first, int):
            return first
        if isinstance(first, str) and first.isdigit():
            return int(first)
    return None


def _episode_id_from_ref(source_ref: Any) -> int | None:
    """Resolve a timeline event's 'ep:<id>' source_ref to an episode id, else None
    (git-sourced refs are SHAs, which resolve to null)."""
    if isinstance(source_ref, str) and source_ref.startswith("ep:"):
        rest = source_ref[3:].strip()
        if rest.isdigit():
            return int(rest)
    return None


def _bm25_sanitize(q: str) -> str:
    """Alnum/space-only, mirroring recall/kg_pg: keeps identifiers/error strings while
    stripping the ParadeDB query-language metacharacters that would raise a parse error."""
    return "".join(c if (c.isalnum() or c.isspace()) else " " for c in q).strip()


def _encode_cursor(ts_iso: str, type_: str, id_str: str) -> str:
    return base64.urlsafe_b64encode(f"{ts_iso}|{type_}|{id_str}".encode()).decode()


def _decode_cursor(cursor: str) -> tuple[str, str] | None:
    """Return (ts_iso, id_str) from an opaque cursor, or None if unparseable.

    The keyset predicate is (ts, id) per the contract; the type segment rides in the
    cursor string for debuggability but is not used in the SQL comparison."""
    try:
        raw = base64.urlsafe_b64decode(cursor.encode()).decode()
        ts_iso, _type, id_str = raw.split("|", 2)
        return ts_iso, id_str
    except Exception:
        return None


def _flag_set(conn: psycopg.Connection[dict[str, Any]]) -> set[tuple[str, str]]:
    """The active (kind, item_id) flag pairs — loaded once per request to mark items."""
    rows = conn.execute(
        "SELECT kind, item_id FROM dashboard_flags WHERE removed_at IS NULL"
    ).fetchall()
    return {(r["kind"], r["item_id"]) for r in rows}


def _count(conn: psycopg.Connection[dict[str, Any]], sql: str, params: Any) -> int:
    """Scalar COUNT for a query aliased AS c (0 if — impossibly — no row comes back)."""
    row = conn.execute(sql, params).fetchone()
    return int(row["c"]) if row else 0


def _limit(raw: str | None, default: int, cap: int) -> int:
    try:
        return max(1, min(int(raw), cap)) if raw is not None else default
    except (TypeError, ValueError):
        return default


def _offset(raw: str | None) -> int:
    try:
        return max(0, int(raw)) if raw is not None else 0
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def _catalog(db_url: str) -> dict[str, Any]:
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    try:
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
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Feed
# ---------------------------------------------------------------------------
#
# The per-type SELECT column lists and row→FeedItem mappers are shared between the
# paged /dash/api/feed endpoint (below) and the SSE stream's single-row hydration
# (Phase 3, further down) so BOTH produce byte-identical wire shapes from ONE code path.
# The endpoint appends a keyset WHERE + ORDER BY + LIMIT to the SELECT; the stream appends
# a "WHERE <pk> = %s". The mappers emit the final item minus `flagged` (added after the
# per-request flag set is loaded) and minus the internal `_ts`/`_idstr` merge-sort keys.

_EP_SELECT = "SELECT id, created_at, project, source, session_id, sequence, content FROM episodes"
_FACT_SELECT = (
    "SELECT r.uuid, r.created_at, r.group_id, r.fact, r.t_valid, r.t_invalid, r.episodes, "
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
        "ts": _iso(r["created_at"]),
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
    sal = {0: 0.3, 1: 0.6, 2: 0.9}.get(int(r["salience"]) if r["salience"] is not None else 1, 0.6)
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

    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    try:
        candidates: list[dict[str, Any]] = []

        # --- episodes (ts = created_at). Filters: project, source. No group_id column. ---
        ep_where = ["1=1"]
        ep_params: list[Any] = []
        if project is not None:
            ep_where.append("COALESCE(NULLIF(project, ''), 'untagged') = %s")
            ep_params.append(project)
        if source is not None:
            ep_where.append("COALESCE(NULLIF(source, ''), 'untagged') = %s")
            ep_params.append(source)
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

        # --- facts (ts = created_at). Filter: group_id. No project/source column. ---
        # Columns are r.-qualified: the entity joins below also expose created_at/group_id.
        f_where = ["r.created_at IS NOT NULL"]
        f_params: list[Any] = []
        if group_id is not None:
            f_where.append("r.group_id = %s")
            f_params.append(group_id)
        if cts is not None:
            f_where.append(
                "(r.created_at < %s::timestamptz OR (r.created_at = %s::timestamptz AND r.uuid < %s))"
            )
            f_params += [cts, cts, cid]
        fact_rows = conn.execute(
            f"{_FACT_SELECT} WHERE {' AND '.join(f_where)} "
            "ORDER BY r.created_at DESC, r.uuid DESC LIMIT %s",
            (*f_params, limit),
        ).fetchall()
        for r in fact_rows:
            item = _fact_item(r)
            item["_ts"] = r["created_at"]
            item["_idstr"] = str(r["uuid"])
            candidates.append(item)

        # --- timeline (ts = ingested_at). Filters: project, and domain for group_id. ---
        t_where = ["1=1"]
        t_params: list[Any] = []
        if project is not None:
            t_where.append("COALESCE(NULLIF(project, ''), 'untagged') = %s")
            t_params.append(project)
        if group_id is not None:
            # 038 added `domain` (technical/personal) as the timeline's group scope; the
            # contract's feed-filter line names only project for timeline, but the domain
            # column IS a group_id-equivalent, so the group filter applies here too.
            t_where.append("domain = %s")
            t_params.append(group_id)
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
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Episode detail + derived
# ---------------------------------------------------------------------------


def _episode(db_url: str, episode_id: int) -> dict[str, Any] | None:
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    try:
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
    finally:
        conn.close()


def _episode_derived(db_url: str, episode_id: int) -> dict[str, Any]:
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    try:
        # Facts whose `episodes` jsonb array contains this id — as a number OR (defensively)
        # a numeric string, since the extractor's array element type isn't guaranteed.
        facts = conn.execute(
            "SELECT uuid, fact, group_id, t_valid, t_invalid FROM kg_relationships "
            "WHERE episodes @> to_jsonb(%s::bigint) OR episodes @> to_jsonb(%s::text) "
            "ORDER BY t_valid DESC NULLS LAST, uuid LIMIT %s",
            (episode_id, str(episode_id), _DERIVED_CAP),
        ).fetchall()
        events = conn.execute(
            "SELECT id, fact, t_valid, salience FROM timeline_events "
            "WHERE source_ref = %s ORDER BY t_valid DESC LIMIT %s",
            (f"ep:{episode_id}", _DERIVED_CAP),
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
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


def _session(db_url: str, session_id: str, highlight: Any) -> dict[str, Any]:
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    try:
        rows = conn.execute(
            "SELECT id, sequence, created_at, project, source, human_turn, assistant_turn, content "
            "FROM episodes WHERE session_id = %s ORDER BY sequence LIMIT %s",
            (session_id, _SESSION_CAP),
        ).fetchall()
        first = rows[0] if rows else None
        return {
            "session_id": session_id,
            "project": first["project"] if first else None,
            "source": first["source"] if first else None,
            "highlight": highlight,
            "episodes": [
                {
                    "id": r["id"],
                    "sequence": r["sequence"],
                    "created_at": _iso(r["created_at"]),
                    "human_turn": r["human_turn"],
                    "assistant_turn": r["assistant_turn"],
                    "content": r["content"],
                }
                for r in rows
            ],
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Entity dossier
# ---------------------------------------------------------------------------


def _entity(db_url: str, uuid: str, mentions_offset: int) -> dict[str, Any] | None:
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    try:
        ent = conn.execute(
            "SELECT uuid, name, entity_type, summary, degree, created_at "
            "FROM kg_entities WHERE uuid = %s",
            (uuid,),
        ).fetchone()
        if ent is None:
            return None

        flags = _flag_set(conn)

        # Facts = live + superseded edges touching the uuid; the "other" endpoint's name
        # comes from a join on whichever side isn't this entity.
        fact_rows = conn.execute(
            "SELECT r.uuid, r.name, r.fact, r.t_valid, r.t_invalid, r.episodes, "
            "       CASE WHEN r.src_uuid = %(u)s THEN r.tgt_uuid ELSE r.src_uuid END AS other_uuid, "
            "       oe.name AS other_name "
            "FROM kg_relationships r "
            "LEFT JOIN kg_entities oe "
            "  ON oe.uuid = CASE WHEN r.src_uuid = %(u)s THEN r.tgt_uuid ELSE r.src_uuid END "
            "WHERE r.src_uuid = %(u)s OR r.tgt_uuid = %(u)s "
            "ORDER BY r.t_valid DESC NULLS LAST, r.uuid LIMIT %(lim)s",
            {"u": uuid, "lim": _ENTITY_FACTS_CAP},
        ).fetchall()

        stats = (
            conn.execute(
                "SELECT count(*) FILTER (WHERE t_invalid IS NULL) AS edges, "
                "       count(*) AS facts, "
                "       COALESCE(sum(retrieval_count), 0) AS served "
                "FROM kg_relationships WHERE src_uuid = %(u)s OR tgt_uuid = %(u)s",
                {"u": uuid},
            ).fetchone()
            or {}
        )

        # Mentions = distinct episode ids across the touching edges' `episodes` arrays,
        # resolved to real episodes, newest first, paged. Non-numeric elements are skipped.
        # The CASE guard coerces NULL / non-array episodes to '[]' so the lateral never
        # errors on a malformed row (it would raise at execution, before any WHERE).
        _mentions_from = (
            "kg_relationships r "
            "CROSS JOIN LATERAL jsonb_array_elements_text("
            "  CASE WHEN jsonb_typeof(r.episodes) = 'array' THEN r.episodes ELSE '[]'::jsonb END"
            ") AS elem(v) "
            "JOIN episodes e ON e.id = elem.v::bigint "
            "WHERE (r.src_uuid = %(u)s OR r.tgt_uuid = %(u)s) AND elem.v ~ '^[0-9]+$'"
        )
        total_row = (
            conn.execute(
                f"SELECT count(*) AS total FROM (SELECT DISTINCT e.id FROM {_mentions_from}) m",
                {"u": uuid},
            ).fetchone()
            or {}
        )
        mention_rows = conn.execute(
            "SELECT e.id, e.created_at, e.content FROM ("
            f"  SELECT DISTINCT e.id, e.created_at, e.content FROM {_mentions_from}"
            ") e ORDER BY e.created_at DESC, e.id DESC OFFSET %(off)s LIMIT %(lim)s",
            {"u": uuid, "off": mentions_offset, "lim": _MENTIONS_PAGE},
        ).fetchall()

        return {
            "entity": {
                "uuid": ent["uuid"],
                "name": ent["name"],
                "entity_type": ent["entity_type"],
                "summary": ent["summary"],
                "degree": ent["degree"],
                "created_at": _iso(ent["created_at"]),
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
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


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

    total_by_type is always computed for all four tabs; `hits` only for the requested
    type. Snippet is the first 240 chars of the matched text — paradedb.snippet() with
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
    safe = _bm25_sanitize(q)
    like = f"%{q}%"

    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    try:
        # --- per-type filter fragments (applied to both count and hit queries) ---
        ep_filt, ep_p = [], []
        if project is not None:
            ep_filt.append("COALESCE(NULLIF(project, ''), 'untagged') = %s")
            ep_p.append(project)
        if source is not None:
            ep_filt.append("COALESCE(NULLIF(source, ''), 'untagged') = %s")
            ep_p.append(source)
        ep_and = "".join(f" AND {c}" for c in ep_filt)

        f_and, f_p = "", []
        if group_id is not None:
            f_and, f_p = " AND group_id = %s", [group_id]

        ev_filt, ev_p = [], []
        if project is not None:
            ev_filt.append("COALESCE(NULLIF(project, ''), 'untagged') = %s")
            ev_p.append(project)
        if group_id is not None:
            ev_filt.append("domain = %s")
            ev_p.append(group_id)
        ev_and = "".join(f" AND {c}" for c in ev_filt)

        en_and, en_p = "", []
        if group_id is not None:
            en_and, en_p = " AND group_id = %s", [group_id]

        # --- total_by_type: all four counts, every call ---
        total: dict[str, int] = dict.fromkeys(_SEARCH_TYPES, 0)
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
                "SELECT id, content, project, source, created_at, session_id, "
                "       paradedb.score(id) AS sc FROM episodes "
                f"WHERE id @@@ paradedb.match('content', %s){ep_and} "
                "ORDER BY sc DESC, id DESC OFFSET %s LIMIT %s",
                (safe, *ep_p, offset, limit),
            ).fetchall():
                hits.append(
                    {
                        "type": "episodes",
                        "id": str(r["id"]),
                        "snippet": (r["content"] or "")[:_SNIPPET_MAX],
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
                "SELECT uuid, fact, group_id, t_valid, t_invalid, created_at, episodes, "
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
                            "ts": _iso(r["created_at"]),
                            "episode_id": _provenance(r["episodes"]),
                        },
                    }
                )
        elif type_ == "events" and safe:
            for r in conn.execute(
                "SELECT id, fact, project, t_valid, source, source_ref, ingested_at, "
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
                        },
                    }
                )
        elif type_ == "entities":
            for r in conn.execute(
                "SELECT uuid, name, entity_type, summary, degree FROM kg_entities "
                f"WHERE name ILIKE %s{en_and} "
                "ORDER BY degree DESC, name OFFSET %s LIMIT %s",
                (like, *en_p, offset, limit),
            ).fetchall():
                hits.append(
                    {
                        "type": "entities",
                        "id": r["uuid"],
                        "snippet": ((r["summary"] or r["name"]) or "")[:_SNIPPET_MAX],
                        "meta": {
                            "name": r["name"],
                            "entity_type": r["entity_type"],
                            "degree": r["degree"],
                        },
                    }
                )

        return {"hits": hits, "total_by_type": total, "offset": offset, "limit": limit}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Recall history (phase 2)
# ---------------------------------------------------------------------------


def _recall_history(db_url: str, limit: int) -> dict[str, Any]:
    """Recent recall() calls from the recall_metrics telemetry log (kind='recall').

    A dedicated slim endpoint for the Recall console's History tab — newest first,
    just the columns the table renders. Deviates from spec §8 (which routed history
    through the phase-4 /metrics/recall aggregate) by shipping now; the aggregate can
    still supersede it later. recall_episodes / fetch / remember rows are excluded by
    the kind filter so the console shows only true recall() calls.
    """
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    try:
        rows = conn.execute(
            "SELECT id, created_at, query, source, ms_total, est_tokens, rerank_top_score "
            "FROM recall_metrics WHERE kind = 'recall' "
            "ORDER BY created_at DESC, id DESC LIMIT %s",
            (limit,),
        ).fetchall()
        return {
            "items": [
                {
                    "id": r["id"],
                    "created_at": _iso(r["created_at"]),
                    "query": r["query"],
                    "source": r["source"],
                    "ms_total": r["ms_total"],
                    "est_tokens": r["est_tokens"],
                    "rerank_top_score": r["rerank_top_score"],
                }
                for r in rows
            ]
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

# Per-kind resolver for the best-effort gist shown on the flags list. The flags list is
# operator-sized (a handful of rows), so a small lookup per row is cheap.
_GIST_SOURCES = {
    "episode": ("episodes", "content", "id"),
    "fact": ("kg_relationships", "fact", "uuid"),
    "timeline_event": ("timeline_events", "fact", "id"),
    "preference": ("preferences", "pref", "id"),
    "note": ("notes", "hook", "id"),
}


def _resolve_flag_gist(conn: psycopg.Connection[dict[str, Any]], kind: str, item_id: str) -> str:
    spec = _GIST_SOURCES.get(kind)
    if not spec:
        return ""
    table, text_col, id_col = spec
    # id_col is a fixed identifier from the trusted map above, never user input; item_id
    # is bound as a parameter. Numeric-keyed tables get a digit guard so a bad id can't
    # raise a cast error inside the read.
    if id_col == "id" and not str(item_id).isdigit():
        return ""
    try:
        row = conn.execute(
            f"SELECT {text_col} AS t FROM {table} WHERE {id_col} = %s LIMIT 1",  # nosec B608
            (item_id,),
        ).fetchone()
    except Exception:  # pragma: no cover - defensive, e.g. a missing table
        return ""
    return _gist(row["t"]) if row else ""


def _flags_list(db_url: str) -> dict[str, Any]:
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    try:
        rows = conn.execute(
            "SELECT id, kind, item_id, note, created_at FROM dashboard_flags "
            "WHERE removed_at IS NULL ORDER BY created_at DESC, id DESC"
        ).fetchall()
        return {
            "flags": [
                {
                    "id": r["id"],
                    "kind": r["kind"],
                    "item_id": r["item_id"],
                    "note": r["note"],
                    "created_at": _iso(r["created_at"]),
                    "gist": _resolve_flag_gist(conn, r["kind"], r["item_id"]),
                }
                for r in rows
            ]
        }
    finally:
        conn.close()


def _flag_toggle(db_url: str, kind: str, item_id: str, note: str | None) -> bool:
    """Insert a flag if none is active, else retire the active one. Returns the new
    flagged state and appends a dashboard_audit row either way."""
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    try:
        active = conn.execute(
            "SELECT id FROM dashboard_flags "
            "WHERE kind = %s AND item_id = %s AND removed_at IS NULL",
            (kind, item_id),
        ).fetchone()
        if active is not None:
            conn.execute(
                "UPDATE dashboard_flags SET removed_at = now() WHERE id = %s", (active["id"],)
            )
            conn.execute(
                "INSERT INTO dashboard_audit (action, kind, item_id, detail) "
                "VALUES ('unflag', %s, %s, %s)",
                (kind, item_id, Json({"note": note}) if note else None),
            )
            return False
        conn.execute(
            "INSERT INTO dashboard_flags (kind, item_id, note) VALUES (%s, %s, %s)",
            (kind, item_id, note),
        )
        conn.execute(
            "INSERT INTO dashboard_audit (action, kind, item_id, detail) "
            "VALUES ('flag', %s, %s, %s)",
            (kind, item_id, Json({"note": note}) if note else None),
        )
        return True
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Proposals (phase 2b) — unified review over the skills + config lanes
# ---------------------------------------------------------------------------
#
# The dashboard is a THIN review console over two independent lanes. It reuses each
# lane's own _proposal_act (the state transition + side effects) and _proposal_detail
# (the row read), and adds three things on top: a lane-merged list, a normalized detail
# envelope, and a dashboard_audit trail keyed by the namespaced id ("skill:<n>" /
# "config:<n>"). It never materializes an accepted change — promote (skills) and apply
# (config) stay with the lanes. See docs/dashboard-contract.md §"Phase 2b".


def _skill_routing_eval_stub(_conn: Any, _cid: int, _name: str) -> str:
    """The skills lane's accept path takes an advisory ``llm`` callable that runs a
    routing-eval on RETUNE candidates — an Anthropic API call that only annotates the
    accept result. The dashboard decision path is contracted to write ONLY state + notes,
    so it passes THIS stub in place of the real ``_routing_eval``: accept still flips
    status→accepted and records the grounded 'accept' signal, but no LLM call is made from
    the request path. Running the eval stays with the skills review CLI."""
    return "routing-eval: skipped (dashboard decision path — advisory eval runs in the skills CLI)"


def _parse_proposal_id(pid: str) -> tuple[str, int] | None:
    """'skill:<n>' | 'config:<n>' -> ('skill'|'config', n); None if malformed."""
    lane, _, rest = pid.partition(":")
    if lane not in ("skill", "config") or not rest.isdigit():
        return None
    return lane, int(rest)


def _norm_kind(lane: str) -> str:
    return "skill" if lane == "skill" else "config-edit"


def _age_days(created: Any) -> int:
    if not isinstance(created, datetime):
        return 0
    now = datetime.now(created.tzinfo) if created.tzinfo else datetime.now(UTC).replace(tzinfo=None)
    return max(0, int((now - created).total_seconds() // 86400))


def _provenance_episodes(conn: psycopg.Connection[dict[str, Any]], evidence: Any) -> list[int]:
    """Best-effort episode ids behind a proposal, from its evidence entries. Two sources:
    an explicit episode ref on an entry (``episode_id`` / an ``ep:<id>`` ``source_ref``),
    and — since lane evidence is keyed by session, not episode — the episodes of each
    distinct evidence ``session_id`` (bounded). Empty list when nothing resolves."""
    if not isinstance(evidence, list):
        return []
    explicit: set[int] = set()
    sessions: set[str] = set()
    for e in evidence:
        if not isinstance(e, dict):
            continue
        v = e.get("episode_id")
        if isinstance(v, bool):
            v = None
        if isinstance(v, int):
            explicit.add(v)
        elif isinstance(v, str) and v.isdigit():
            explicit.add(int(v))
        rid = _episode_id_from_ref(e.get("source_ref") or e.get("ref"))
        if rid is not None:
            explicit.add(rid)
        sid = e.get("session_id")
        if isinstance(sid, str) and sid:
            sessions.add(sid)
    ids = set(explicit)
    if sessions:
        rows = conn.execute(
            "SELECT id FROM episodes WHERE session_id = ANY(%s) ORDER BY id LIMIT %s",
            (list(sessions), _PROVENANCE_CAP),
        ).fetchall()
        ids.update(int(r["id"]) for r in rows)
    return sorted(ids)


def _proposals_unified(db_url: str, status: str | None, kind: str | None) -> dict[str, Any]:
    """Lane-merged proposal list + the nav-badge pending_count. Bounded per lane (newest
    first), then merged and re-sorted by created_at. ``status``/``kind`` narrow the view;
    ``pending_count`` is always both lanes' 'proposed' rows, independent of the view."""
    want_skill = kind in (None, "skill")
    want_config = kind in (None, "config-edit")

    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    try:
        out: list[dict[str, Any]] = []
        if want_skill:
            for r in conn.execute(
                "SELECT id, name, summary, status, created_at "
                "FROM skills_lane.skill_gap_candidates WHERE status = ANY(%s) "
                "ORDER BY created_at DESC, id DESC LIMIT %s",
                (list(_SKILL_REVIEW_STATUSES), _PROPOSALS_LANE_CAP),
            ).fetchall():
                if status and r["status"] != status:
                    continue
                out.append(
                    {
                        "id": f"skill:{r['id']}",
                        "kind": "skill",
                        "name": r["name"],
                        "gist": _gist(r["summary"]),
                        "status": r["status"],
                        "age_days": _age_days(r["created_at"]),
                        "created_at": _iso(r["created_at"]),
                    }
                )
        if want_config:
            for r in conn.execute(
                "SELECT id, file_key, summary, status, created_at "
                "FROM config_lane.config_proposals WHERE status = ANY(%s) "
                "ORDER BY created_at DESC, id DESC LIMIT %s",
                (list(_CONFIG_REVIEW_STATUSES), _PROPOSALS_LANE_CAP),
            ).fetchall():
                if status and r["status"] != status:
                    continue
                out.append(
                    {
                        "id": f"config:{r['id']}",
                        "kind": "config-edit",
                        "name": r["file_key"],
                        "gist": _gist(r["summary"]),
                        "status": r["status"],
                        "age_days": _age_days(r["created_at"]),
                        "created_at": _iso(r["created_at"]),
                    }
                )
        out.sort(key=lambda p: p["created_at"] or "", reverse=True)
        pending = _count(
            conn,
            "SELECT (SELECT count(*) FROM skills_lane.skill_gap_candidates WHERE status='proposed') "
            "+ (SELECT count(*) FROM config_lane.config_proposals WHERE status='proposed') AS c",
            None,
        )
        return {"proposals": out, "pending_count": pending}
    finally:
        conn.close()


def _proposal_detail(db_url: str, lane: str, n: int) -> dict[str, Any] | None:
    """Normalized detail envelope over a lane's own _proposal_detail. payload is markdown
    (skills SKILL.md draft) or diff (config unified diff); audit_log is the dashboard_audit
    trail for this id, plus the lane's own reject_reason when a decision left no audit row."""
    if lane == "skill":
        d = _skill_proposal_detail(db_url, n)
        if not d.get("found"):
            return None
        kind, name, status = "skill", d.get("name") or "", d.get("status") or ""
        evidence = d.get("evidence") or []
        payload = {"type": "markdown", "content": d.get("proposal_body") or d.get("summary") or ""}
        rr_sql = "SELECT reject_reason FROM skills_lane.skill_gap_candidates WHERE id=%s"
    else:
        d = _config_proposal_detail(db_url, n)
        if not d.get("found"):
            return None
        kind, name, status = "config-edit", d.get("file_key") or "", d.get("status") or ""
        evidence = d.get("evidence") or []
        payload = {"type": "diff", "content": d.get("diff") or ""}
        rr_sql = "SELECT reject_reason FROM config_lane.config_proposals WHERE id=%s"

    full_id = f"{lane}:{n}"
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    try:
        provenance = _provenance_episodes(conn, evidence)
        audit_rows = conn.execute(
            "SELECT ts, action, detail FROM dashboard_audit WHERE item_id = %s ORDER BY ts, id",
            (full_id,),
        ).fetchall()
        audit_log = [
            {
                "ts": _iso(r["ts"]),
                "action": r["action"],
                "note": r["detail"].get("note") if isinstance(r["detail"], dict) else None,
            }
            for r in audit_rows
        ]
        rr = conn.execute(rr_sql, (n,)).fetchone()
        reject_reason = rr["reject_reason"] if rr else None
        # Surface a lane-side reject reason (e.g. a CLI reject) only if no dashboard reject
        # row already carries it, so a dashboard reject isn't shown twice.
        if reject_reason and not any(a["action"] == "proposal_reject" for a in audit_log):
            audit_log.append({"ts": None, "action": "reject_reason", "note": reject_reason})
    finally:
        conn.close()

    return {
        "id": full_id,
        "kind": kind,
        "name": name,
        "status": status,
        "evidence": evidence,
        "provenance_episodes": provenance,
        "payload": payload,
        "audit_log": audit_log,
    }


def _proposal_decision(
    db_url: str, lane: str, n: int, action: str, note: str | None
) -> dict[str, Any] | None:
    """Delegate to the lane's _proposal_act (approve→accept, reject→reject-with-reason),
    then append a dashboard_audit row. Returns the lane result, or None if the row is gone."""
    lane_action = "accept" if action == "approve" else "reject"
    if lane == "skill":
        result = _skill_proposal_act(db_url, n, lane_action, note, _skill_routing_eval_stub)
    else:
        # config accept takes an optional scope override; the dashboard never re-scopes,
        # so pass None (the lane keeps the proposal's stored blast radius).
        result = _config_proposal_act(db_url, n, lane_action, note, None)
    if isinstance(result, dict) and result.get("found") is False:
        return None

    audit_action = "proposal_approve" if action == "approve" else "proposal_reject"
    conn = psycopg.connect(db_url, autocommit=True)
    try:
        conn.execute(
            "INSERT INTO dashboard_audit (action, kind, item_id, detail) VALUES (%s,%s,%s,%s)",
            (
                audit_action,
                _norm_kind(lane),
                f"{lane}:{n}",
                Json({"note": note, "lane_result": result}),
            ),
        )
    finally:
        conn.close()
    return result


# ---------------------------------------------------------------------------
# Phase 3 — SSE live stream (LISTEN/NOTIFY + ring buffer)
# ---------------------------------------------------------------------------
#
# schema/043 arms AFTER INSERT triggers on episodes / kg_relationships / timeline_events
# that pg_notify('dash_feed', {type,id}). A single lazily-started LISTEN worker (one
# dedicated async psycopg connection, started on the first SSE subscriber and stopped when
# the last disconnects) hydrates each notification into a full FeedItem — via the SAME
# per-type SELECT + mapper the /feed endpoint uses — and appends it to a 512-slot ring
# buffer keyed by a monotonically increasing integer event id. The /dash/api/stream route
# replays from the buffer on resume (Last-Event-ID) and streams live thereafter.


def _hydrate_feed_item(db_url: str, type_: Any, id_: Any) -> dict[str, Any] | None:
    """Re-SELECT one feed row by id and map it to the SAME FeedItem shape /feed emits
    (including `flagged`). Returns None for an unknown type or a row that vanished between
    the NOTIFY and this read. Reuses _EP_SELECT/_FACT_SELECT/_TL_SELECT + the mappers."""
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    try:
        if type_ == "episode":
            if not str(id_).isdigit():
                return None
            row = conn.execute(f"{_EP_SELECT} WHERE id = %s", (int(id_),)).fetchone()
            item = _episode_item(row) if row else None
        elif type_ == "fact":
            row = conn.execute(f"{_FACT_SELECT} WHERE r.uuid = %s", (str(id_),)).fetchone()
            item = _fact_item(row) if row else None
        elif type_ == "timeline_event":
            if not str(id_).isdigit():
                return None
            row = conn.execute(f"{_TL_SELECT} WHERE id = %s", (int(id_),)).fetchone()
            item = _timeline_item(row) if row else None
        else:
            return None
        if item is None:
            return None
        item["flagged"] = (item["type"], item["id"]) in _flag_set(conn)
        return item
    finally:
        conn.close()


def _processing_status(db_url: str) -> dict[str, Any]:
    """Extraction-queue health for the header live badge: {queue_depth, active}. ONE cheap
    indexed count (extraction_queue_status_idx) — the status refresher runs it once/second,
    shared across all connected SSE clients, never per-connection."""
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    try:
        row = (
            conn.execute(
                "SELECT count(*) FILTER (WHERE status = 'pending') AS queue_depth, "
                "       count(*) FILTER (WHERE status = 'processing') AS processing "
                "FROM extraction_queue WHERE status IN ('pending', 'processing')"
            ).fetchone()
            or {}
        )
        return {
            "queue_depth": int(row.get("queue_depth") or 0),
            "active": int(row.get("processing") or 0) > 0,
        }
    finally:
        conn.close()


class _FeedEventBuffer:
    """A fixed-capacity ring of hydrated feed events with monotonically increasing integer
    ids. Standalone (no I/O) so the replay/reset logic is unit-testable without a DB or HTTP.

    Resume semantics for a client's Last-Event-ID L:
      * L == head            -> replay []      (already current; stream live from here)
      * L  > head            -> reset          (client is ahead of us; our buffer was reset,
                                                e.g. a server restart — client refetches page 1)
      * L  < oldest - 1      -> reset          (a gap: events L+1..oldest-1 were evicted)
      * else                 -> replay since L (contiguous; hand back what was missed)
    """

    def __init__(self, capacity: int = _STREAM_BUFFER_CAP) -> None:
        self._events: deque[tuple[int, dict[str, Any]]] = deque(maxlen=capacity)
        self._next_id = 1

    def append(self, item: dict[str, Any]) -> int:
        eid = self._next_id
        self._next_id += 1
        self._events.append((eid, item))
        return eid

    @property
    def head(self) -> int:
        """The last assigned event id (0 when nothing has ever been appended)."""
        return self._next_id - 1

    def oldest(self) -> int | None:
        return self._events[0][0] if self._events else None

    def since(self, after_id: int) -> list[tuple[int, dict[str, Any]]]:
        return [(eid, it) for eid, it in self._events if eid > after_id]

    def resume(self, last_event_id: int) -> tuple[str, list[tuple[int, dict[str, Any]]]]:
        head = self.head
        if last_event_id >= head:
            # Current, or ahead of us (buffer was reset under the client). Ahead -> resync.
            return ("reset", []) if last_event_id > head else ("replay", [])
        oldest = self.oldest()
        if oldest is not None and last_event_id < oldest - 1:
            return ("reset", [])  # evicted gap between what the client has and what we kept
        return ("replay", self.since(last_event_id))


class _StreamManager:
    """Owns the lazily-started LISTEN worker + status refresher and the shared ring buffer.
    Zero background cost with no clients: both tasks start on the first subscriber and are
    cancelled when the last one disconnects. Everything runs in the server's single event
    loop, so subscribe/unsubscribe are plain synchronous ref-count flips (no lock needed —
    asyncio has no preemption between awaits)."""

    def __init__(self, db_url: str) -> None:
        self._db_url = db_url
        self.buffer = _FeedEventBuffer()
        self.status: dict[str, Any] = {"queue_depth": 0, "active": False}
        # Set once the LISTEN is actually issued — lets a caller (and the tests) know a
        # NOTIFY inserted from now on will be delivered, not raced against worker startup.
        self.listening = asyncio.Event()
        self._subscribers = 0
        self._tasks: list[asyncio.Task[Any]] = []

    def subscribe(self) -> None:
        self._subscribers += 1
        if self._subscribers == 1:
            loop = asyncio.get_running_loop()
            self._tasks = [
                loop.create_task(self._listen_loop()),
                loop.create_task(self._status_loop()),
            ]

    def unsubscribe(self) -> None:
        self._subscribers -= 1
        if self._subscribers <= 0:
            self._subscribers = 0
            for t in self._tasks:
                t.cancel()
            self._tasks = []
            self.listening.clear()

    async def _listen_loop(self) -> None:
        """One dedicated async connection LISTENing on the dash_feed channel; hydrates each
        notification off the event loop (a thread executor) and appends it to the ring
        buffer. run_in_executor (not starlette's run_in_threadpool) keeps this background
        task off anyio's request-scoped context."""
        loop = asyncio.get_running_loop()
        aconn = None
        try:
            aconn = await psycopg.AsyncConnection.connect(self._db_url, autocommit=True)
            await aconn.execute(f"LISTEN {_STREAM_CHANNEL}")
            self.listening.set()
            async for notify in aconn.notifies():
                try:
                    payload = json.loads(notify.payload)
                    item = await loop.run_in_executor(
                        None,
                        _hydrate_feed_item,
                        self._db_url,
                        payload.get("type"),
                        payload.get("id"),
                    )
                    if item is not None:
                        self.buffer.append(item)
                except Exception as e:  # one bad notification must not kill the stream
                    logger.warning("dash stream: hydrate failed: %s", e)
        except asyncio.CancelledError:
            pass  # intended shutdown when the last subscriber leaves
        except Exception as e:  # pragma: no cover - defensive (connect/LISTEN failure)
            logger.warning("dash stream: listen loop error: %s", e)
        finally:
            self.listening.clear()
            if aconn is not None:
                try:
                    await aconn.close()
                except Exception:  # pragma: no cover - best-effort close during cancel
                    pass

    async def _status_loop(self) -> None:
        """Refresh the shared processing_status snapshot once/second (one query total, not
        per-connection). Connections read self.status and emit it on their own cadence."""
        loop = asyncio.get_running_loop()
        try:
            while True:
                try:
                    self.status = await loop.run_in_executor(None, _processing_status, self._db_url)
                except Exception as e:  # pragma: no cover - defensive
                    logger.warning("dash stream: status refresh failed: %s", e)
                await asyncio.sleep(_STREAM_STATUS_S)
        except asyncio.CancelledError:
            pass


def _sse_frame(event: str, data: str, event_id: int | None = None) -> str:
    """One SSE frame: an `event:` line, an optional `id:` line (feed events only — so the
    client's Last-Event-ID tracks the resumable feed stream, not status/heartbeat), a
    `data:` line, terminated by a blank line."""
    lines = [f"event: {event}"]
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"data: {data}")
    return "\n".join(lines) + "\n\n"


# ---------------------------------------------------------------------------
# Static bundle serving
# ---------------------------------------------------------------------------

_BUNDLE_MISSING = ("bundle not built",)  # sentinel detail text


def _no_cache(resp: Response) -> Response:
    resp.headers["Cache-Control"] = "no-cache"
    return resp


def _serve_static_file(name: str) -> Response:
    """Serve a top-level bundle file (index.html / app.js). 503 when the bundle is
    absent — the deployment shipped without a built web/dist (contract)."""
    path = _DIST_DIR / name
    if not path.is_file():
        return JSONResponse({"status": "error", "detail": "bundle not built"}, status_code=503)
    media = _CONTENT_TYPES.get(path.suffix, "application/octet-stream")
    if name == "index.html":
        media = "text/html; charset=utf-8"
    return _no_cache(FileResponse(path, media_type=media))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(mcp: Any, db_url: str, authorized: Callable[[Request], bool]) -> None:
    """Mount /dash (static bundle) + /dash/api/* (machine-token gated). No-op w/o DB_URL,
    matching the sibling route modules."""
    if not db_url:
        logger.info("dashboard routes disabled (no DB_URL)")
        return

    # Per-register catalog cache (contract: cached ~5 min in-process). Kept in this
    # closure, not a module global, so a fresh register() in tests starts empty — one
    # test's inserts never serve stale counts to another's.
    catalog_cache: dict[str, Any] = {"ts": 0.0, "data": None}

    # Per-register SSE manager (LISTEN worker + ring buffer). One per server; a fresh
    # register() in tests gets an isolated manager (own buffer/worker), same as the cache.
    stream_manager = _StreamManager(db_url)

    async def _api(request: Request, work: Callable[[], Any]) -> JSONResponse:
        """Shared api boundary: auth gate + threadpool + fail-soft 500."""
        if not authorized(request):
            return JSONResponse({"status": "error", "detail": "unauthorized"}, status_code=401)
        try:
            result = await run_in_threadpool(work)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("dashboard api failed: %s", e)
            return JSONResponse({"status": "error", "detail": str(e)[:200]}, status_code=500)
        return JSONResponse(result)

    # ---- static (UNAUTHENTICATED) ----

    @mcp.custom_route("/dash", methods=["GET"])  # type: ignore[misc]
    async def dash_index(request: Request) -> Response:
        return _serve_static_file("index.html")

    @mcp.custom_route("/dash/app.js", methods=["GET"])  # type: ignore[misc]
    async def dash_appjs(request: Request) -> Response:
        return _serve_static_file("app.js")

    @mcp.custom_route("/dash/assets/{name}", methods=["GET"])  # type: ignore[misc]
    async def dash_assets(request: Request) -> Response:
        name = request.path_params["name"]
        assets_dir = _DIST_DIR / "assets"
        if not assets_dir.is_dir():
            return JSONResponse({"status": "error", "detail": "not found"}, status_code=404)
        # Whitelist by exact basename against the directory listing — never join a raw
        # user path. A traversal attempt ("../x", an absolute/encoded path) simply won't
        # be a member of the listed set, so it 404s.
        allowed = {p.name for p in assets_dir.iterdir() if p.is_file()}
        if name not in allowed:
            return JSONResponse({"status": "error", "detail": "not found"}, status_code=404)
        path = assets_dir / name
        media = _CONTENT_TYPES.get(path.suffix, "application/octet-stream")
        headers = (
            {"Cache-Control": "public, max-age=31536000, immutable"}
            if path.suffix in _FONT_EXTS
            else {"Cache-Control": "no-cache"}
        )
        return FileResponse(path, media_type=media, headers=headers)

    # ---- api (machine-token gated) ----

    @mcp.custom_route("/dash/api/catalog", methods=["GET"])  # type: ignore[misc]
    async def dash_catalog(request: Request) -> JSONResponse:
        if not authorized(request):
            return JSONResponse({"status": "error", "detail": "unauthorized"}, status_code=401)
        now = time.monotonic()
        if catalog_cache["data"] is not None and now - catalog_cache["ts"] < _CATALOG_TTL_S:
            return JSONResponse(catalog_cache["data"])
        try:
            data = await run_in_threadpool(_catalog, db_url)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("dashboard catalog failed: %s", e)
            return JSONResponse({"status": "error", "detail": str(e)[:200]}, status_code=500)
        catalog_cache["ts"] = now
        catalog_cache["data"] = data
        return JSONResponse(data)

    @mcp.custom_route("/dash/api/feed", methods=["GET"])  # type: ignore[misc]
    async def dash_feed(request: Request) -> JSONResponse:
        qp = request.query_params
        limit = _limit(qp.get("limit"), _FEED_LIMIT_DEFAULT, _FEED_LIMIT_MAX)
        cursor = qp.get("cursor") or None
        project = qp.get("project") or None
        group_id = qp.get("group_id") or None
        source = qp.get("source") or None
        return await _api(request, lambda: _feed(db_url, cursor, limit, project, group_id, source))

    @mcp.custom_route("/dash/api/stream", methods=["GET"])  # type: ignore[misc]
    async def dash_stream(request: Request) -> Response:
        """SSE live feed (Phase 3). Machine-token gated like every /dash/api/* route — the
        client uses fetch + a ReadableStream parser (not EventSource) so it CAN send the
        Authorization header. On connect: if Last-Event-ID (header or ?last_event_id=) is
        still inside the ring buffer, replay from it; if it aged out or is ahead of us, emit
        a `reset` event (client refetches page 1). Then stream live feed events
        (new_episode|new_fact|new_timeline_event, data = the FeedItem JSON) plus a
        processing_status snapshot ~1/s and a heartbeat comment every 15s."""
        if not authorized(request):
            return JSONResponse({"status": "error", "detail": "unauthorized"}, status_code=401)
        raw = request.headers.get("last-event-id") or request.query_params.get("last_event_id")
        last_id = int(raw) if (raw and raw.lstrip("-").isdigit()) else None

        async def gen() -> Any:
            stream_manager.subscribe()
            try:
                buf = stream_manager.buffer
                if last_id is not None:
                    mode, events = buf.resume(last_id)
                    if mode == "reset":
                        yield _sse_frame("reset", "{}")
                        sent = buf.head
                    else:
                        sent = last_id
                        for eid, item in events:
                            yield _sse_frame(
                                _STREAM_EVENT_NAMES[item["type"]], json.dumps(item), eid
                            )
                            sent = eid
                else:
                    # Fresh connect: the client already has page 1 from /feed, so start live
                    # from the current head — only genuinely new events stream.
                    sent = buf.head
                # Push status immediately so the header live badge is correct on connect.
                yield _sse_frame("processing_status", json.dumps(stream_manager.status))
                last_status = last_hb = time.monotonic()
                while True:
                    if await request.is_disconnected():
                        break
                    for eid, item in buf.since(sent):
                        yield _sse_frame(_STREAM_EVENT_NAMES[item["type"]], json.dumps(item), eid)
                        sent = eid
                    now = time.monotonic()
                    if now - last_status >= _STREAM_STATUS_S:
                        yield _sse_frame("processing_status", json.dumps(stream_manager.status))
                        last_status = now
                    if now - last_hb >= _STREAM_HEARTBEAT_S:
                        yield ": heartbeat\n\n"
                        last_hb = now
                    await asyncio.sleep(_STREAM_POLL_S)
            finally:
                stream_manager.unsubscribe()

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # tell nginx/proxies not to buffer the stream
            },
        )

    @mcp.custom_route("/dash/api/episode/{id}", methods=["GET"])  # type: ignore[misc]
    async def dash_episode(request: Request) -> JSONResponse:
        if not authorized(request):
            return JSONResponse({"status": "error", "detail": "unauthorized"}, status_code=401)
        raw = request.path_params["id"]
        if not str(raw).isdigit():
            return JSONResponse({"status": "error", "detail": "bad episode id"}, status_code=400)
        try:
            result = await run_in_threadpool(_episode, db_url, int(raw))
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("dashboard episode failed: %s", e)
            return JSONResponse({"status": "error", "detail": str(e)[:200]}, status_code=500)
        if result is None:
            return JSONResponse({"status": "error", "detail": "episode not found"}, status_code=404)
        return JSONResponse(result)

    @mcp.custom_route("/dash/api/episode/{id}/derived", methods=["GET"])  # type: ignore[misc]
    async def dash_episode_derived(request: Request) -> JSONResponse:
        if not authorized(request):
            return JSONResponse({"status": "error", "detail": "unauthorized"}, status_code=401)
        raw = request.path_params["id"]
        if not str(raw).isdigit():
            return JSONResponse({"status": "error", "detail": "bad episode id"}, status_code=400)
        return await _api(request, lambda: _episode_derived(db_url, int(raw)))

    @mcp.custom_route("/dash/api/session/{id}", methods=["GET"])  # type: ignore[misc]
    async def dash_session(request: Request) -> JSONResponse:
        raw_hl = request.query_params.get("highlight")
        highlight: Any = None
        if raw_hl is not None:
            highlight = int(raw_hl) if raw_hl.isdigit() else raw_hl
        session_id = request.path_params["id"]
        return await _api(request, lambda: _session(db_url, session_id, highlight))

    @mcp.custom_route("/dash/api/entity/{uuid}", methods=["GET"])  # type: ignore[misc]
    async def dash_entity(request: Request) -> JSONResponse:
        if not authorized(request):
            return JSONResponse({"status": "error", "detail": "unauthorized"}, status_code=401)
        uuid = request.path_params["uuid"]
        m_off = _offset(request.query_params.get("mentions_offset"))
        try:
            result = await run_in_threadpool(_entity, db_url, uuid, m_off)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("dashboard entity failed: %s", e)
            return JSONResponse({"status": "error", "detail": str(e)[:200]}, status_code=500)
        if result is None:
            return JSONResponse({"status": "error", "detail": "entity not found"}, status_code=404)
        return JSONResponse(result)

    @mcp.custom_route("/dash/api/search", methods=["GET"])  # type: ignore[misc]
    async def dash_search(request: Request) -> JSONResponse:
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
        return await _api(
            request,
            lambda: _search(db_url, q, type_, offset, limit, project, group_id, source),
        )

    @mcp.custom_route("/dash/api/recall/history", methods=["GET"])  # type: ignore[misc]
    async def dash_recall_history(request: Request) -> JSONResponse:
        limit = _limit(
            request.query_params.get("limit"), _RECALL_HISTORY_DEFAULT, _RECALL_HISTORY_MAX
        )
        return await _api(request, lambda: _recall_history(db_url, limit))

    @mcp.custom_route("/dash/api/flags", methods=["GET"])  # type: ignore[misc]
    async def dash_flags(request: Request) -> JSONResponse:
        return await _api(request, lambda: _flags_list(db_url))

    @mcp.custom_route("/dash/api/flag", methods=["POST"])  # type: ignore[misc]
    async def dash_flag(request: Request) -> JSONResponse:
        if not authorized(request):
            return JSONResponse({"status": "error", "detail": "unauthorized"}, status_code=401)
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"status": "error", "detail": "invalid JSON body"}, status_code=400)
        kind = body.get("kind")
        item_id = body.get("id")
        note = body.get("note")
        if kind not in _FLAG_KINDS:
            return JSONResponse(
                {"status": "error", "detail": f"invalid kind {kind!r}"}, status_code=400
            )
        if not item_id or not isinstance(item_id, str):
            return JSONResponse(
                {"status": "error", "detail": "missing 'id' (item_id string)"}, status_code=400
            )
        if note is not None and not isinstance(note, str):
            return JSONResponse(
                {"status": "error", "detail": "'note' must be a string"}, status_code=400
            )
        try:
            flagged = await run_in_threadpool(_flag_toggle, db_url, kind, item_id, note)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("dashboard flag failed: %s", e)
            return JSONResponse({"status": "error", "detail": str(e)[:200]}, status_code=500)
        return JSONResponse({"status": "ok", "flagged": flagged})

    # ---- proposals (phase 2b) ----

    @mcp.custom_route("/dash/api/proposals", methods=["GET"])  # type: ignore[misc]
    async def dash_proposals(request: Request) -> JSONResponse:
        qp = request.query_params
        status = qp.get("status") or None
        if status == "all":
            status = None
        kind = qp.get("kind") or None
        if kind not in (None, "skill", "config-edit"):
            kind = None
        return await _api(request, lambda: _proposals_unified(db_url, status, kind))

    @mcp.custom_route("/dash/api/proposals/{id}", methods=["GET"])  # type: ignore[misc]
    async def dash_proposal_detail(request: Request) -> JSONResponse:
        if not authorized(request):
            return JSONResponse({"status": "error", "detail": "unauthorized"}, status_code=401)
        parsed = _parse_proposal_id(request.path_params["id"])
        if parsed is None:
            return JSONResponse({"status": "error", "detail": "bad proposal id"}, status_code=400)
        lane, n = parsed
        try:
            result = await run_in_threadpool(_proposal_detail, db_url, lane, n)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("dashboard proposal detail failed: %s", e)
            return JSONResponse({"status": "error", "detail": str(e)[:200]}, status_code=500)
        if result is None:
            return JSONResponse(
                {"status": "error", "detail": "proposal not found"}, status_code=404
            )
        return JSONResponse(result)

    @mcp.custom_route("/dash/api/proposals/{id}/decision", methods=["POST"])  # type: ignore[misc]
    async def dash_proposal_decision(request: Request) -> JSONResponse:
        if not authorized(request):
            return JSONResponse({"status": "error", "detail": "unauthorized"}, status_code=401)
        parsed = _parse_proposal_id(request.path_params["id"])
        if parsed is None:
            return JSONResponse({"status": "error", "detail": "bad proposal id"}, status_code=400)
        lane, n = parsed
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"status": "error", "detail": "invalid JSON body"}, status_code=400)
        action = body.get("action")
        note = body.get("note")
        if action not in ("approve", "reject"):
            return JSONResponse(
                {"status": "error", "detail": "action must be 'approve' or 'reject'"},
                status_code=400,
            )
        if note is not None and not isinstance(note, str):
            return JSONResponse(
                {"status": "error", "detail": "'note' must be a string"}, status_code=400
            )
        # A reject must carry a reason — it's the lane's reject_reason and the audit note.
        if action == "reject" and not (note and note.strip()):
            return JSONResponse(
                {"status": "error", "detail": "reject requires a non-empty note"}, status_code=400
            )
        try:
            result = await run_in_threadpool(_proposal_decision, db_url, lane, n, action, note)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("dashboard proposal decision failed: %s", e)
            return JSONResponse({"status": "error", "detail": str(e)[:200]}, status_code=500)
        if result is None:
            return JSONResponse(
                {"status": "error", "detail": "proposal not found"}, status_code=404
            )
        return JSONResponse(result)
