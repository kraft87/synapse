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
import re
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)

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

# Phase 5 (Timeline / Preferences / Dream report / Behavior files).
_TIMELINE_LIMIT_DEFAULT = 50
_TIMELINE_LIMIT_MAX = 200
_PREFERENCES_CAP = 500  # a single owner accrues dozens; this only bounds a runaway read
_DREAM_REPORT_DEFAULT = 20
_DREAM_REPORT_MAX = 100
# Coarse timeline salience (0/1/2) → the 0.3/0.6/0.9 readout the feed + type ramp use.
_SAL_MAP = {0: 0.3, 1: 0.6, 2: 0.9}
# Behavior-file grouping by config_registry file_key path shape, in display order.
_BEHAVIOR_GROUP_ORDER = ("CLAUDE.md", "rules", "memory notes", "other")
# Obsidian-style [[wikilink]] target (capture up to a '|' alias or the closing ']]').
_WIKILINK_RE = re.compile(r"\[\[([^\[\]|]+)")
# Preference sort → ORDER BY fragment (trusted map, never user text — safe to interpolate).
_PREF_SORTS = {
    "recency": "p.last_asserted DESC",
    "assert_count": "p.assert_count DESC, p.last_asserted DESC",
}
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
# Graph explorer (phase 6)
# ---------------------------------------------------------------------------


def _normalize_seed_name(name: str) -> str:
    """Mirror ingestion.dedup._normalize_name for the exact-name seed lookup:
    lowercase, collapse internal whitespace, strip bookend whitespace/punctuation."""
    collapsed = " ".join(name.lower().split())
    return collapsed.strip(" \t\n\r\f\v.,;:!?\"'()[]{}<>")


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
    (name ILIKE, degree DESC). Global (no group scope): the graph is seeded explicitly."""
    if not q:
        return []
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    try:
        rows = conn.execute(
            "SELECT uuid, name, entity_type, entity_supertype, degree FROM kg_entities "
            "WHERE name ILIKE %s ORDER BY degree DESC, name LIMIT %s",
            (f"%{q}%", limit),
        ).fetchall()
        return [
            {
                "uuid": r["uuid"],
                "name": r["name"],
                "entity_type": r["entity_type"],
                "supertype": r["entity_supertype"],
                "degree": r["degree"],
            }
            for r in rows
        ]
    finally:
        conn.close()


def _resolve_seed(conn: psycopg.Connection[dict[str, Any]], entity: str) -> dict[str, Any] | None:
    """Resolve a seed given a uuid OR a name: exact uuid → exact normalized_name →
    best ILIKE match (highest degree). None when nothing matches."""
    cols = "uuid, name, entity_type, degree, summary"
    row = conn.execute(f"SELECT {cols} FROM kg_entities WHERE uuid = %s", (entity,)).fetchone()
    if row:
        return row
    norm = _normalize_seed_name(entity)
    if norm:
        row = conn.execute(
            f"SELECT {cols} FROM kg_entities WHERE normalized_name = %s "
            "ORDER BY degree DESC LIMIT 1",
            (norm,),
        ).fetchone()
        if row:
            return row
    return conn.execute(
        f"SELECT {cols} FROM kg_entities WHERE name ILIKE %s ORDER BY degree DESC, name LIMIT 1",
        (f"%{entity}%",),
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
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    try:
        # Bound every query on this short-lived connection (SET can't bind a param).
        conn.execute(
            "SELECT set_config('statement_timeout', %s, false)", (str(_GRAPH_STMT_TIMEOUT_MS),)
        )

        seed = _resolve_seed(conn, entity)
        if seed is None:
            return None
        seed_uuid = seed["uuid"]

        # Shared as-of edge-visibility fragment (empty when unset).
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
                "SELECT src_uuid, tgt_uuid FROM kg_relationships "
                f"WHERE (src_uuid = ANY(%s) OR tgt_uuid = ANY(%s)){as_of_sql} LIMIT %s",
                (frontier, frontier, *as_of_params, _GRAPH_EDGE_SCAN_CAP),
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
            "SELECT uuid, name, entity_type, entity_supertype, degree, summary "
            "FROM kg_entities WHERE uuid = ANY(%s)",
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
            "SELECT uuid, src_uuid, tgt_uuid, name, fact, t_valid, t_invalid, episodes, "
            "       retrieval_count FROM kg_relationships "
            f"WHERE src_uuid = ANY(%s) AND tgt_uuid = ANY(%s){as_of_sql} LIMIT %s",
            (kept, kept, *as_of_params, _GRAPH_EDGE_SCAN_CAP),
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
                    "entity_type": r["entity_type"],
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
# Metrics (phase 4) — Recall / Ingestion / Corpus ops aggregates
# ---------------------------------------------------------------------------
#
# Three read-only aggregate endpoints over the telemetry + queue + corpus tables. The
# honesty rules (docs/dashboard-contract.md §"Phase 4"):
#   * recall percentiles come straight from recall_metrics (kind='recall') — the SAME
#     numbers the engine already logs; no re-instrumentation.
#   * ingestion series carry ONLY what extraction_queue's real columns support. Historical
#     queue DEPTH is NOT reconstructable (a row's status is overwritten in place), so we
#     never fabricate a depth history — we series enqueue/hour (enqueued_at) and
#     completions/hour (processed_at, the only completion timestamp) and report depth as a
#     LIVE snapshot only.
#   * corpus headline counts use pg_class.reltuples (a fast estimate) for large tables and
#     an exact count(*) for small ones; the estimated flag is surfaced per-table.

_METRICS_WINDOW_MAX_S = 30 * 86400  # window is capped at 30d (contract)
_RECALL_WINDOW_DEFAULT_S = 7 * 86400  # /metrics/recall default 7d
_INGEST_WINDOW_DEFAULT_S = 48 * 3600  # /metrics/ingestion default 48h
# Recall legs timed by the engine (recall_metrics ms_* columns). timeline/prefs are only
# populated when those legs are enabled — a bucket with none stays absent for that leg.
_RECALL_LEGS = ("embed", "bm25", "vector", "kg", "web", "rerank", "timeline", "prefs")
_SLOWEST_CAP = 10
_SCORE_BINS = 10
_INGEST_FAILURES_CAP = 20
_CORPUS_TTL_S = 3600  # 1h in-process cache (row counts over 40K+ rows are the cost)
# Below this reltuples estimate, count exactly; at/above it, trust the estimate (contract).
_CORPUS_ESTIMATE_FLOOR = 50000
# (table, time_column|None) — the corpus tables to report + which column drives spark_30d.
_CORPUS_TABLES: tuple[tuple[str, str | None], ...] = (
    ("episodes", "created_at"),
    ("kg_entities", "created_at"),
    ("kg_relationships", "created_at"),
    ("timeline_events", "ingested_at"),
    ("preferences", "ingested_at"),
    ("notes", "created_at"),
    ("chunks", "created_at"),
)


def _parse_window(raw: str | None, default_s: int) -> int:
    """A window string ('7d' / '48h' / '90m' / a bare seconds int) → seconds, floored at 1h
    and capped at 30d. Anything unparseable falls back to the endpoint default."""
    if not raw:
        return default_s
    raw = raw.strip().lower()
    try:
        if raw.endswith("d"):
            secs = int(float(raw[:-1]) * 86400)
        elif raw.endswith("h"):
            secs = int(float(raw[:-1]) * 3600)
        elif raw.endswith("m"):
            secs = int(float(raw[:-1]) * 60)
        else:
            secs = int(float(raw))
    except (TypeError, ValueError):
        return default_s
    return max(3600, min(secs, _METRICS_WINDOW_MAX_S))


def _r1(v: Any) -> float | None:
    """Round a numeric to 1 decimal, passing None through."""
    return round(float(v), 1) if v is not None else None


def _metrics_recall(db_url: str, window_s: int) -> dict[str, Any]:
    """Hourly recall latency percentiles + per-leg p50 + slowest queries + rerank-score
    histogram, over recall_metrics (kind='recall') within the window. percentile_cont is a
    true continuous percentile; NULL leg timings are ignored by the aggregate (a disabled
    leg simply doesn't contribute)."""
    leg_select = ", ".join(
        f"percentile_cont(0.5) WITHIN GROUP (ORDER BY ms_{leg}::double precision) AS leg_{leg}"
        for leg in _RECALL_LEGS
    )
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    try:
        series_rows = conn.execute(
            "SELECT date_trunc('hour', created_at) AS t, count(*) AS calls, "
            "  percentile_cont(0.5) WITHIN GROUP (ORDER BY ms_total::double precision) AS p50, "
            "  percentile_cont(0.95) WITHIN GROUP (ORDER BY ms_total::double precision) AS p95, "
            "  percentile_cont(0.5) WITHIN GROUP (ORDER BY est_tokens::double precision) AS tok, "
            f"  {leg_select} "
            "FROM recall_metrics WHERE kind = 'recall' AND created_at >= now() - %s::interval "
            "GROUP BY 1 ORDER BY 1",
            (f"{window_s} seconds",),
        ).fetchall()
        series = []
        for r in series_rows:
            legs = {
                leg: _r1(r[f"leg_{leg}"]) for leg in _RECALL_LEGS if r[f"leg_{leg}"] is not None
            }
            series.append(
                {
                    "t": _iso(r["t"]),
                    "p50": _r1(r["p50"]),
                    "p95": _r1(r["p95"]),
                    "calls": int(r["calls"]),
                    "tokens_p50": int(r["tok"]) if r["tok"] is not None else None,
                    "legs_p50": legs,
                }
            )

        slow_rows = conn.execute(
            "SELECT query, ms_total, created_at FROM recall_metrics "
            "WHERE kind = 'recall' AND ms_total IS NOT NULL AND created_at >= now() - %s::interval "
            "ORDER BY ms_total DESC, id DESC LIMIT %s",
            (f"{window_s} seconds", _SLOWEST_CAP),
        ).fetchall()
        slowest = [
            {
                "query": r["query"],
                "ms_total": _r1(r["ms_total"]),
                "created_at": _iso(r["created_at"]),
            }
            for r in slow_rows
        ]

        # width_bucket(x, 0, 1, 10): 1..10 for x in [0,1); 0 for x<0; 11 for x>=1. Fold the
        # out-of-range edges into the terminal bins so a 1.0 score lands in the top bin.
        hist_rows = conn.execute(
            "SELECT width_bucket(rerank_top_score, 0, 1, %s) AS b, count(*) AS n "
            "FROM recall_metrics WHERE kind = 'recall' AND rerank_top_score IS NOT NULL "
            "AND created_at >= now() - %s::interval GROUP BY 1",
            (_SCORE_BINS, f"{window_s} seconds"),
        ).fetchall()
        counts_by_bin = [0] * _SCORE_BINS
        for r in hist_rows:
            b = int(r["b"])
            idx = min(max(b - 1, 0), _SCORE_BINS - 1)  # clamp 0→bin0 and 11→bin9
            counts_by_bin[idx] += int(r["n"])
        score_hist = [
            {"lo": round(i / _SCORE_BINS, 2), "hi": round((i + 1) / _SCORE_BINS, 2), "n": n}
            for i, n in enumerate(counts_by_bin)
        ]
        return {"series": series, "slowest": slowest, "score_hist": score_hist}
    finally:
        conn.close()


def _metrics_ingestion(db_url: str, window_s: int) -> dict[str, Any]:
    """Extraction-queue health. queue_depth is a LIVE snapshot (pending/processing/failed
    right now); throughput series are enqueue/hour (enqueued_at) + completions/hour
    (processed_at) — the ONLY honest series the columns support. There is no historical
    depth series because a row's status is overwritten in place (contract §Phase 4)."""
    interval = f"{window_s} seconds"
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    try:
        depth = conn.execute(
            "SELECT count(*) FILTER (WHERE status = 'pending') AS pending, "
            "  count(*) FILTER (WHERE status = 'processing') AS processing, "
            "  count(*) FILTER (WHERE status = 'failed') AS failed "
            "FROM extraction_queue"
        ).fetchone() or {"pending": 0, "processing": 0, "failed": 0}

        enq = conn.execute(
            "SELECT date_trunc('hour', enqueued_at) AS t, count(*) AS n FROM extraction_queue "
            "WHERE enqueued_at >= now() - %s::interval GROUP BY 1 ORDER BY 1",
            (interval,),
        ).fetchall()
        comp = conn.execute(
            "SELECT date_trunc('hour', processed_at) AS t, count(*) AS n FROM extraction_queue "
            "WHERE processed_at IS NOT NULL AND processed_at >= now() - %s::interval "
            "GROUP BY 1 ORDER BY 1",
            (interval,),
        ).fetchall()
        fails = conn.execute(
            "SELECT id, episode_id, error, enqueued_at, processed_at, attempts "
            "FROM extraction_queue WHERE status = 'failed' "
            "ORDER BY COALESCE(processed_at, enqueued_at) DESC, id DESC LIMIT %s",
            (_INGEST_FAILURES_CAP,),
        ).fetchall()

        last = conn.execute(
            "SELECT id, started_at, finished_at, stages, counts, samples, errors, ok, "
            "  EXTRACT(EPOCH FROM (finished_at - started_at)) AS duration_s "
            "FROM dream_runs ORDER BY started_at DESC, id DESC LIMIT 1"
        ).fetchone()

        return {
            "queue_depth": int(depth["pending"] or 0),
            "queue": {
                "pending": int(depth["pending"] or 0),
                "processing": int(depth["processing"] or 0),
                "failed": int(depth["failed"] or 0),
            },
            "throughput": {
                "enqueued_per_hour": [{"t": _iso(r["t"]), "n": int(r["n"])} for r in enq],
                "completed_per_hour": [{"t": _iso(r["t"]), "n": int(r["n"])} for r in comp],
            },
            "failures": [
                {
                    "id": r["id"],
                    "episode_id": r["episode_id"],
                    "error": (r["error"] or "")[:_SNIPPET_MAX],
                    "enqueued_at": _iso(r["enqueued_at"]),
                    "processed_at": _iso(r["processed_at"]),
                    "attempts": r["attempts"],
                }
                for r in fails
            ],
            "last_dream": _dream_run_json(last) if last else None,
        }
    finally:
        conn.close()


def _dream_run_json(r: dict[str, Any]) -> dict[str, Any]:
    """Shape a dream_runs row for the wire (used by the ingestion endpoint's last_dream)."""
    return {
        "id": r["id"],
        "started_at": _iso(r["started_at"]),
        "finished_at": _iso(r["finished_at"]),
        "duration_s": _r1(r["duration_s"]) if r.get("duration_s") is not None else None,
        "ok": r["ok"],
        "stages": r["stages"] or {},
        "counts": r["counts"] or {},
        "samples": r["samples"] or {},
        "errors": r["errors"] or [],
    }


def _spark_30d(conn: psycopg.Connection[dict[str, Any]], table: str, tcol: str) -> list[int]:
    """30 daily row counts (oldest→newest) off a table's own time column. table/tcol are from
    the trusted _CORPUS_TABLES constant, never user input (safe to interpolate)."""
    rows = conn.execute(
        f"SELECT (date_trunc('day', {tcol}))::date AS d, count(*) AS n FROM {table} "  # nosec B608
        f"WHERE {tcol} >= (now() - interval '30 days') GROUP BY 1",
        (),
    ).fetchall()
    by_day = {r["d"]: int(r["n"]) for r in rows}
    today = datetime.now(UTC).date()
    return [by_day.get(today - timedelta(days=29 - i), 0) for i in range(30)]


def _metrics_corpus(db_url: str) -> dict[str, Any]:
    """Per-table row counts (+ 30-day sparkline + 30d delta) and the episodes-by-project /
    by-source proportions. Whole response is cached 1h in the route (row counts are the cost).
    Headline counts are pg_class estimates for big tables, exact for small ones."""
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    try:
        tables = []
        for name, tcol in _CORPUS_TABLES:
            present = conn.execute("SELECT to_regclass(%s) AS r", (f"public.{name}",)).fetchone()
            if not present or present["r"] is None:
                continue
            est_row = conn.execute(
                "SELECT reltuples::bigint AS n FROM pg_class WHERE relname = %s AND relkind = 'r'",
                (name,),
            ).fetchone()
            est = est_row["n"] if est_row else None
            if est is None or est < _CORPUS_ESTIMATE_FLOOR:
                exact = conn.execute(f"SELECT count(*) AS c FROM {name}").fetchone()  # nosec B608
                rows_n = int(exact["c"]) if exact else 0
                estimated = False
            else:
                rows_n = int(est)
                estimated = True
            spark = _spark_30d(conn, name, tcol) if tcol else []
            tables.append(
                {
                    "name": name,
                    "rows": rows_n,
                    "rows_estimated": estimated,
                    "spark_30d": spark,
                    "delta_30d": sum(spark),
                }
            )

        by_project = [
            {"name": r["name"], "n": int(r["n"])}
            for r in conn.execute(
                "SELECT COALESCE(NULLIF(project, ''), 'untagged') AS name, count(*) AS n "
                "FROM episodes GROUP BY 1 ORDER BY n DESC, name LIMIT 12"
            ).fetchall()
        ]
        by_source = [
            {"name": r["name"], "n": int(r["n"])}
            for r in conn.execute(
                "SELECT COALESCE(NULLIF(source, ''), 'untagged') AS name, count(*) AS n "
                "FROM episodes GROUP BY 1 ORDER BY n DESC, name LIMIT 12"
            ).fetchall()
        ]
        return {"tables": tables, "by_project": by_project, "by_source": by_source}
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
# Phase 5 — Timeline (Events + Preferences), Dream report, Behavior files
# ---------------------------------------------------------------------------
#
# Read-only surfaces over the episodic timeline (033), the preference log (035), the
# dream-run bookkeeping (044), and the config-lane file mirror (030/031). Same posture as
# the rest of /dash/api: one short-lived connection per call, bounded, fail-soft. Honesty
# notes (mirrored into docs/dashboard-contract.md §"Phase 5"):
#   * timeline_events.event_type is sparsely populated (schema 033 deferred it) — a `type`
#     chip filter narrows to rows with that exact event_type; UNTYPED events show only when
#     no chip is active (the default view is the full stream).
#   * config_lane.config_registry stores the CURRENT file content only (content + hash +
#     updated_at) — there is NO version history table, so /behavior/file returns no history
#     and the client's change-history timeline stays hidden. Not fabricated this phase.


def _parse_before(before: str | None) -> tuple[str, int | None] | None:
    """Timeline keyset cursor. A bare ISO timestamp (the jump-to-date case) → (ts, None),
    read as a strict ``t_valid < ts``. A compound ``ts|id`` (next_before from a prior page)
    → (ts, id), read as the full (t_valid, id) keyset so ties on t_valid never drop or
    duplicate a row across a page boundary."""
    if not before:
        return None
    ts, sep, rest = before.partition("|")
    if sep and rest.isdigit():
        return ts, int(rest)
    return before, None


def _timeline(
    db_url: str, before: str | None, limit: int, type_: str | None, group_id: str | None
) -> dict[str, Any]:
    """t_valid-DESC keyset page of timeline events. `type` filters event_type (sparse —
    see honesty note); `group_id` maps to the `domain` column (schema 038). Each event
    carries the coarse `salience` plus the 0.3/0.6/0.9 `sal` readout the type ramp reads,
    the resolved `ep:N` episode id, and its flag state."""
    cur = _parse_before(before)
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    try:
        where = ["1=1"]
        params: list[Any] = []
        if type_ is not None:
            where.append("event_type = %s")
            params.append(type_)
        if group_id is not None:
            where.append("domain = %s")
            params.append(group_id)
        if cur is not None:
            ts, cid = cur
            if cid is not None:
                where.append(
                    "(t_valid < %s::timestamptz OR (t_valid = %s::timestamptz AND id < %s))"
                )
                params += [ts, ts, cid]
            else:
                where.append("t_valid < %s::timestamptz")
                params.append(ts)
        rows = conn.execute(
            "SELECT id, t_valid, fact, source, project, salience, event_type, source_ref, domain "
            f"FROM timeline_events WHERE {' AND '.join(where)} "
            "ORDER BY t_valid DESC, id DESC LIMIT %s",
            (*params, limit),
        ).fetchall()
        flags = _flag_set(conn)
        events = [
            {
                "id": r["id"],
                "t_valid": _iso(r["t_valid"]),
                "fact": r["fact"],
                "source": r["source"],
                "project": r["project"],
                "salience": r["salience"],
                "sal": _SAL_MAP.get(int(r["salience"]) if r["salience"] is not None else 1, 0.6),
                "event_type": r["event_type"],
                "episode_id": _episode_id_from_ref(r["source_ref"]),
                "flagged": ("timeline_event", str(r["id"])) in flags,
            }
            for r in rows
        ]
        next_before = None
        if len(rows) == limit and rows:
            last = rows[-1]
            next_before = f"{_iso(last['t_valid'])}|{last['id']}"
        return {"events": events, "next_before": next_before}
    finally:
        conn.close()


def _preferences(db_url: str, sort: str) -> dict[str, Any]:
    """The full preference log — LIVE rows first, then superseded (struck in the UI), each
    ordered by the chosen sort. `superseded_by_text` is the replacing row's pref, joined in
    so the UI can name what won without a second fetch."""
    order = _PREF_SORTS.get(sort, _PREF_SORTS["recency"])
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    try:
        rows = conn.execute(
            "SELECT p.id, p.pref, p.polarity, p.first_seen, p.last_asserted, p.assert_count, "
            "  p.superseded_by, p.t_invalid, sp.pref AS superseded_by_text "
            "FROM preferences p LEFT JOIN preferences sp ON sp.id = p.superseded_by "
            f"ORDER BY (p.t_invalid IS NOT NULL), {order}, p.id DESC LIMIT %s",  # nosec B608
            (_PREFERENCES_CAP,),
        ).fetchall()
        flags = _flag_set(conn)
        return {
            "preferences": [
                {
                    "id": r["id"],
                    "pref": r["pref"],
                    "polarity": r["polarity"],
                    "first_seen": _iso(r["first_seen"]),
                    "last_asserted": _iso(r["last_asserted"]),
                    "assert_count": r["assert_count"],
                    "superseded_by": r["superseded_by"],
                    "superseded_by_text": r["superseded_by_text"],
                    "t_invalid": _iso(r["t_invalid"]),
                    "flagged": ("preference", str(r["id"])) in flags,
                }
                for r in rows
            ]
        }
    finally:
        conn.close()


def _dream_report(db_url: str, limit: int) -> dict[str, Any]:
    """Recent dream_runs (044), newest first — the full per-run drill-in the Dream-report
    page renders. Reuses the same row shaper the Metrics ingestion endpoint's last_dream
    uses. Empty table → {"runs": []}."""
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    try:
        rows = conn.execute(
            "SELECT id, started_at, finished_at, stages, counts, samples, errors, ok, "
            "  EXTRACT(EPOCH FROM (finished_at - started_at)) AS duration_s "
            "FROM dream_runs ORDER BY started_at DESC, id DESC LIMIT %s",
            (limit,),
        ).fetchall()
        return {"runs": [_dream_run_json(r) for r in rows]}
    finally:
        conn.close()


def _wikilinks(content: str | None) -> list[str]:
    """Unique [[wikilink]] targets in first-seen order (alias suffix after '|' dropped)."""
    if not content:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for m in _WIKILINK_RE.finditer(content):
        target = m.group(1).strip()
        if target and target not in seen:
            seen.add(target)
            out.append(target)
    return out


def _behavior_group(file_key: str) -> str:
    """Bucket a config_registry file_key by path shape (the left-list grouping, README §6)."""
    base = file_key.rsplit("/", 1)[-1]
    if base == "CLAUDE.md":
        return "CLAUDE.md"
    if file_key.startswith("rules/") or "/rules/" in file_key:
        return "rules"
    if (
        file_key.startswith(("memory/", "notes/"))
        or "/memory/" in file_key
        or "/notes/" in file_key
    ):
        return "memory notes"
    return "other"


def _behavior_files(db_url: str) -> dict[str, Any]:
    """The config-lane file mirror, grouped for the left list. One entry per registry row
    (surface_id + scope + file_key is the PK), so the same file_key on two surfaces lists
    twice — the client disambiguates the detail fetch with scope + surface."""
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    try:
        rows = conn.execute(
            "SELECT surface_id, scope, file_key, updated_at, octet_length(content) AS size "
            "FROM config_lane.config_registry ORDER BY file_key, scope, surface_id"
        ).fetchall()
    finally:
        conn.close()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault(_behavior_group(r["file_key"]), []).append(
            {
                "file_key": r["file_key"],
                "surface_id": r["surface_id"],
                "scope": r["scope"],
                "updated_at": _iso(r["updated_at"]),
                "size": int(r["size"] or 0),
            }
        )
    groups = [{"name": g, "files": grouped[g]} for g in _BEHAVIOR_GROUP_ORDER if g in grouped]
    return {"groups": groups}


def _behavior_file(db_url: str, key: str, scope: str, surface: str | None) -> dict[str, Any] | None:
    """One mirrored file's content + meta + parsed [[wikilinks]]. Mirrors config_sync_routes'
    _fetch_config; when `surface` is omitted, the most-recently-updated surface for that
    (scope, file_key) is served. No history — the registry keeps only the current version
    (contract §Phase 5)."""
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    try:
        cols = (
            "surface_id, scope, file_key, abs_path, content, content_hash, modified_at, updated_at"
        )
        if surface:
            row = conn.execute(
                f"SELECT {cols} FROM config_lane.config_registry "
                "WHERE file_key = %s AND scope = %s AND surface_id = %s",
                (key, scope, surface),
            ).fetchone()
        else:
            row = conn.execute(
                f"SELECT {cols} FROM config_lane.config_registry "
                "WHERE file_key = %s AND scope = %s ORDER BY updated_at DESC LIMIT 1",
                (key, scope),
            ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    content = row["content"]
    return {
        "file_key": row["file_key"],
        "content": content,
        "meta": {
            "surface_id": row["surface_id"],
            "scope": row["scope"],
            "abs_path": row["abs_path"],
            "content_hash": row["content_hash"],
            "modified_at": _iso(row["modified_at"]),
            "updated_at": _iso(row["updated_at"]),
            "size": len(content or ""),
        },
        "links": _wikilinks(content),
    }


def _behavior_linkgraph(db_url: str) -> dict[str, Any]:
    """Adjacency over every registry file's [[wikilinks]]. Nodes are keyed by logical
    file_key (deduped across surfaces); edges are file_key → wikilink target. A target need
    not be a node (many point at memory notes, which are not mirrored files) — the client
    resolves what it can and renders the rest as inert leaves."""
    conn = psycopg.connect(db_url, autocommit=True, row_factory=dict_row)
    try:
        rows = conn.execute(
            "SELECT scope, file_key, content FROM config_lane.config_registry"
        ).fetchall()
    finally:
        conn.close()
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []
    for r in rows:
        fk = r["file_key"]
        nodes.setdefault(fk, {"file_key": fk, "scope": r["scope"], "group": _behavior_group(fk)})
        for target in _wikilinks(r["content"]):
            edges.append({"source": fk, "target": target})
    return {"nodes": list(nodes.values()), "edges": edges}


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
    # Corpus metrics cache (contract: 1h). Same closure-scoping rationale as the catalog.
    corpus_cache: dict[str, Any] = {"ts": 0.0, "data": None}

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

    # The server has no other browser-facing root — a bare vhost hit (Caddy proxies
    # the whole port) 404'd. Send humans to the dashboard; machine paths unaffected.
    @mcp.custom_route("/", methods=["GET"])  # type: ignore[misc]
    async def root_redirect(request: Request) -> Response:
        return RedirectResponse("/dash", status_code=302)

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

    # ---- graph explorer (phase 6) ----

    @mcp.custom_route("/dash/api/graph/entities", methods=["GET"])  # type: ignore[misc]
    async def dash_graph_entities(request: Request) -> JSONResponse:
        q = (request.query_params.get("q") or "").strip()
        limit = _limit(
            request.query_params.get("limit"), _GRAPH_TYPEAHEAD_DEFAULT, _GRAPH_TYPEAHEAD_MAX
        )
        return await _api(request, lambda: _graph_entities(db_url, q, limit))

    @mcp.custom_route("/dash/api/graph/neighborhood", methods=["GET"])  # type: ignore[misc]
    async def dash_graph_neighborhood(request: Request) -> JSONResponse:
        if not authorized(request):
            return JSONResponse({"status": "error", "detail": "unauthorized"}, status_code=401)
        qp = request.query_params
        entity = (qp.get("entity") or "").strip()
        if not entity:
            return JSONResponse({"status": "error", "detail": "missing 'entity'"}, status_code=400)
        depth = 2 if qp.get("depth") == "2" else 1
        as_of = _parse_as_of(qp.get("as_of"))
        limit = _limit(qp.get("limit"), _GRAPH_NODE_CAP, _GRAPH_NODE_CAP)
        try:
            result = await run_in_threadpool(
                _graph_neighborhood, db_url, entity, depth, as_of, limit
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("dashboard graph neighborhood failed: %s", e)
            return JSONResponse({"status": "error", "detail": str(e)[:200]}, status_code=500)
        if result is None:
            return JSONResponse({"status": "error", "detail": "entity not found"}, status_code=404)
        return JSONResponse(result)

    @mcp.custom_route("/dash/api/recall/history", methods=["GET"])  # type: ignore[misc]
    async def dash_recall_history(request: Request) -> JSONResponse:
        limit = _limit(
            request.query_params.get("limit"), _RECALL_HISTORY_DEFAULT, _RECALL_HISTORY_MAX
        )
        return await _api(request, lambda: _recall_history(db_url, limit))

    # ---- metrics (phase 4) ----

    @mcp.custom_route("/dash/api/metrics/recall", methods=["GET"])  # type: ignore[misc]
    async def dash_metrics_recall(request: Request) -> JSONResponse:
        window_s = _parse_window(request.query_params.get("window"), _RECALL_WINDOW_DEFAULT_S)
        return await _api(request, lambda: _metrics_recall(db_url, window_s))

    @mcp.custom_route("/dash/api/metrics/ingestion", methods=["GET"])  # type: ignore[misc]
    async def dash_metrics_ingestion(request: Request) -> JSONResponse:
        window_s = _parse_window(request.query_params.get("window"), _INGEST_WINDOW_DEFAULT_S)
        return await _api(request, lambda: _metrics_ingestion(db_url, window_s))

    @mcp.custom_route("/dash/api/metrics/corpus", methods=["GET"])  # type: ignore[misc]
    async def dash_metrics_corpus(request: Request) -> JSONResponse:
        if not authorized(request):
            return JSONResponse({"status": "error", "detail": "unauthorized"}, status_code=401)
        now = time.monotonic()
        if corpus_cache["data"] is not None and now - corpus_cache["ts"] < _CORPUS_TTL_S:
            return JSONResponse(corpus_cache["data"])
        try:
            data = await run_in_threadpool(_metrics_corpus, db_url)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("dashboard corpus metrics failed: %s", e)
            return JSONResponse({"status": "error", "detail": str(e)[:200]}, status_code=500)
        corpus_cache["ts"] = now
        corpus_cache["data"] = data
        return JSONResponse(data)

    # ---- timeline + preferences (phase 5) ----

    @mcp.custom_route("/dash/api/timeline", methods=["GET"])  # type: ignore[misc]
    async def dash_timeline(request: Request) -> JSONResponse:
        qp = request.query_params
        limit = _limit(qp.get("limit"), _TIMELINE_LIMIT_DEFAULT, _TIMELINE_LIMIT_MAX)
        before = qp.get("before") or None
        type_ = qp.get("type") or None
        group_id = qp.get("group_id") or None
        return await _api(request, lambda: _timeline(db_url, before, limit, type_, group_id))

    @mcp.custom_route("/dash/api/preferences", methods=["GET"])  # type: ignore[misc]
    async def dash_preferences(request: Request) -> JSONResponse:
        sort = request.query_params.get("sort") or "recency"
        return await _api(request, lambda: _preferences(db_url, sort))

    # ---- dream report + behavior files (phase 5) ----

    @mcp.custom_route("/dash/api/dream/report", methods=["GET"])  # type: ignore[misc]
    async def dash_dream_report(request: Request) -> JSONResponse:
        limit = _limit(request.query_params.get("limit"), _DREAM_REPORT_DEFAULT, _DREAM_REPORT_MAX)
        return await _api(request, lambda: _dream_report(db_url, limit))

    @mcp.custom_route("/dash/api/behavior/files", methods=["GET"])  # type: ignore[misc]
    async def dash_behavior_files(request: Request) -> JSONResponse:
        return await _api(request, lambda: _behavior_files(db_url))

    @mcp.custom_route("/dash/api/behavior/file", methods=["GET"])  # type: ignore[misc]
    async def dash_behavior_file(request: Request) -> JSONResponse:
        if not authorized(request):
            return JSONResponse({"status": "error", "detail": "unauthorized"}, status_code=401)
        qp = request.query_params
        key = qp.get("key")
        if not key:
            return JSONResponse({"status": "error", "detail": "missing 'key'"}, status_code=400)
        scope = qp.get("scope") or "global"
        surface = qp.get("surface") or None
        try:
            result = await run_in_threadpool(_behavior_file, db_url, key, scope, surface)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("dashboard behavior file failed: %s", e)
            return JSONResponse({"status": "error", "detail": str(e)[:200]}, status_code=500)
        if result is None:
            return JSONResponse({"status": "error", "detail": "file not found"}, status_code=404)
        return JSONResponse(result)

    @mcp.custom_route("/dash/api/behavior/linkgraph", methods=["GET"])  # type: ignore[misc]
    async def dash_behavior_linkgraph(request: Request) -> JSONResponse:
        return await _api(request, lambda: _behavior_linkgraph(db_url))

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
