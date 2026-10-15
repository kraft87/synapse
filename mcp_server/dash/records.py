"""Phase 5: Timeline (events), Preferences, the Dream report and Behavior files."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from psycopg.rows import dict_row
from starlette.requests import Request

from ingestion import pg_pool
from ingestion.llm_schemas import EVENT_TYPES
from mcp_server.dash.common import (
    _SAL_MAP,
    _domain_filter,
    _episode_id_from_ref,
    _flag_set,
    _limit,
)
from mcp_server.dash.metrics import _dream_run_json
from mcp_server.http_helpers import RouteError, _iso, route

# Phase 5 (Timeline / Preferences / Dream report / Behavior files).
_TIMELINE_LIMIT_DEFAULT = 50
_TIMELINE_LIMIT_MAX = 200
_PREFERENCES_CAP = 500  # a single owner accrues dozens; this only bounds a runaway read
_DREAM_REPORT_DEFAULT = 20
_DREAM_REPORT_MAX = 100
# Behavior-file grouping by config_registry file_key path shape, in display order.
_BEHAVIOR_GROUP_ORDER = ("CLAUDE.md", "rules", "memory notes", "other")
# Obsidian-style [[wikilink]] target (capture up to a '|' alias or the closing ']]').
_WIKILINK_RE = re.compile(r"\[\[([^\[\]|]+)")
# Preference sort → ORDER BY fragment (trusted map, never user text — safe to interpolate).
_PREF_SORTS = {
    "recency": "p.last_asserted DESC",
    "assert_count": "p.assert_count DESC, p.last_asserted DESC",
}


# Read-only surfaces over the episodic timeline (033), the preference log (035), the
# dream-run bookkeeping (044), and the config-lane file mirror (030/031). Same posture as
# the rest of /dash/api: one pooled connection per call, bounded, fail-soft. Honesty
# notes (mirrored into docs/dashboard-contract.md §"Phase 5"):
#   * timeline_events.event_type is decision|action|finding|milestone or NULL (untyped, the
#     sparse legacy majority) — a `type` chip filter narrows to rows with that exact
#     event_type and any other value is a 400; UNTYPED events show only when no chip is
#     active (the default view is the full stream).
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
    see honesty note); `group_id` maps to the `domain` column (schema 038), an unlabelled
    event passing either group. Each event carries the coarse `salience` plus the
    0.3/0.6/0.9 `sal` readout the type ramp reads, the resolved `ep:N` episode id, and its
    flag state."""
    cur = _parse_before(before)
    with pg_pool.connection(db_url, row_factory=dict_row) as conn:
        where = ["1=1"]
        params: list[Any] = []
        if type_ is not None:
            where.append("event_type = %s")
            params.append(type_)
        if group_id is not None:
            clause, args = _domain_filter(group_id)
            where.append(clause)
            params += args
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
            "SELECT id, t_valid, fact, source, project, salience, event_type, source_ref, domain "  # nosec B608 — where holds fixed predicates; the values are bound
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


def _timeline_months(
    db_url: str, type_: str | None, group_id: str | None, tz: str
) -> dict[str, Any]:
    """Every month that has events under the Events tab's filters, newest first, bucketed
    in the viewer's zone `tz` (an event said at 20:30 on the 30th belongs to that month,
    not to the next one its UTC stamp falls in). `before` is the first instant after the
    month, the cursor that makes /timeline start at the month's last event."""
    where = ["1=1"]
    params: list[Any] = [tz]
    if type_ is not None:
        where.append("event_type = %s")
        params.append(type_)
    if group_id is not None:
        clause, args = _domain_filter(group_id)
        where.append(clause)
        params += args
    with pg_pool.connection(db_url, row_factory=dict_row) as conn:
        rows = conn.execute(
            "SELECT to_char(m, 'YYYY-MM') AS month, n, "
            "       (m + interval '1 month') AT TIME ZONE %s AS before "
            "FROM (SELECT date_trunc('month', t_valid AT TIME ZONE %s) AS m, count(*) AS n "
            f"      FROM timeline_events WHERE {' AND '.join(where)} GROUP BY 1) AS s "  # nosec B608
            "ORDER BY m DESC",
            (tz, *params),
        ).fetchall()
        return {
            "months": [
                {"month": r["month"], "n": r["n"], "before": _iso(r["before"])} for r in rows
            ]
        }


def _preferences(db_url: str, sort: str) -> dict[str, Any]:
    """The full preference log — LIVE rows first, then superseded (struck in the UI), each
    ordered by the chosen sort. `superseded_by_text` is the replacing row's pref, joined in
    so the UI can name what won without a second fetch."""
    order = _PREF_SORTS.get(sort, _PREF_SORTS["recency"])
    with pg_pool.connection(db_url, row_factory=dict_row) as conn:
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


def _dream_report(db_url: str, limit: int) -> dict[str, Any]:
    """Recent dream_runs (044), newest first — the full per-run drill-in the Dream-report
    page renders. Reuses the same row shaper the Metrics ingestion endpoint's last_dream
    uses. Empty table → {"runs": []}."""
    with pg_pool.connection(db_url, row_factory=dict_row) as conn:
        rows = conn.execute(
            "SELECT id, started_at, finished_at, stages, counts, samples, errors, ok, "
            "  EXTRACT(EPOCH FROM (finished_at - started_at)) AS duration_s "
            "FROM dream_runs ORDER BY started_at DESC, id DESC LIMIT %s",
            (limit,),
        ).fetchall()
        return {"runs": [_dream_run_json(r) for r in rows]}


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
    with pg_pool.connection(db_url, row_factory=dict_row) as conn:
        rows = conn.execute(
            "SELECT surface_id, scope, file_key, updated_at, octet_length(content) AS size "
            "FROM config_lane.config_registry ORDER BY file_key, scope, surface_id"
        ).fetchall()
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
    """One mirrored file's content + meta + parsed [[wikilinks]]. When `surface` is omitted,
    the most-recently-updated surface for that (scope, file_key) is served. No history — the
    registry keeps only the current version (contract §Phase 5)."""
    with pg_pool.connection(db_url, row_factory=dict_row) as conn:
        cols = (
            "surface_id, scope, file_key, abs_path, content, content_hash, modified_at, updated_at"
        )
        if surface:
            row = conn.execute(
                f"SELECT {cols} FROM config_lane.config_registry "  # nosec B608 — cols is a literal column list
                "WHERE file_key = %s AND scope = %s AND surface_id = %s",
                (key, scope, surface),
            ).fetchone()
        else:
            row = conn.execute(
                f"SELECT {cols} FROM config_lane.config_registry "  # nosec B608 — cols is a literal column list
                "WHERE file_key = %s AND scope = %s ORDER BY updated_at DESC LIMIT 1",
                (key, scope),
            ).fetchone()
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


def register(mcp: Any, db_url: str, authorized: Callable[[Request], bool]) -> None:
    """Mount timeline, preferences, the dream report and behavior files (admin gated)."""

    @mcp.custom_route("/dash/api/timeline", methods=["GET"])  # type: ignore[misc]
    @route(authorized)
    def dash_timeline(request: Request) -> dict[str, Any]:
        qp = request.query_params
        limit = _limit(qp.get("limit"), _TIMELINE_LIMIT_DEFAULT, _TIMELINE_LIMIT_MAX)
        before = qp.get("before") or None
        type_ = qp.get("type") or None
        group_id = qp.get("group_id") or None
        if type_ is not None and type_ not in EVENT_TYPES:
            raise RouteError(f"unknown type {type_!r}")
        return _timeline(db_url, before, limit, type_, group_id)

    @mcp.custom_route("/dash/api/timeline/months", methods=["GET"])  # type: ignore[misc]
    @route(authorized)
    def dash_timeline_months(request: Request) -> dict[str, Any]:
        qp = request.query_params
        type_ = qp.get("type") or None
        group_id = qp.get("group_id") or None
        tz = qp.get("tz") or ""
        if type_ is not None and type_ not in EVENT_TYPES:
            raise RouteError(f"unknown type {type_!r}")
        try:
            ZoneInfo(tz)
        except (ZoneInfoNotFoundError, ValueError) as e:
            raise RouteError("missing or unknown 'tz' (an IANA zone name)") from e
        return _timeline_months(db_url, type_, group_id, tz)

    @mcp.custom_route("/dash/api/preferences", methods=["GET"])  # type: ignore[misc]
    @route(authorized)
    def dash_preferences(request: Request) -> dict[str, Any]:
        return _preferences(db_url, request.query_params.get("sort") or "recency")

    @mcp.custom_route("/dash/api/dream/report", methods=["GET"])  # type: ignore[misc]
    @route(authorized)
    def dash_dream_report(request: Request) -> dict[str, Any]:
        limit = _limit(request.query_params.get("limit"), _DREAM_REPORT_DEFAULT, _DREAM_REPORT_MAX)
        return _dream_report(db_url, limit)

    @mcp.custom_route("/dash/api/behavior/files", methods=["GET"])  # type: ignore[misc]
    @route(authorized)
    def dash_behavior_files(request: Request) -> dict[str, Any]:
        return _behavior_files(db_url)

    @mcp.custom_route("/dash/api/behavior/file", methods=["GET"])  # type: ignore[misc]
    @route(authorized, not_found="file not found")
    def dash_behavior_file(request: Request) -> dict[str, Any] | None:
        qp = request.query_params
        key = qp.get("key")
        if not key:
            raise RouteError("missing 'key'")
        return _behavior_file(db_url, key, qp.get("scope") or "global", qp.get("surface") or None)
