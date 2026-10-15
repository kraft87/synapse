"""Proposals (phase 2b): the unified review console over the skills and config lanes."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Json
from starlette.requests import Request
from starlette.responses import JSONResponse

from ingestion import pg_pool
from mcp_server.config_sync_routes import _proposal_act as _config_proposal_act
from mcp_server.config_sync_routes import _proposal_detail as _config_proposal_detail
from mcp_server.dash.common import _count, _episode_id_from_ref, _gist
from mcp_server.http_helpers import RouteError, _iso, route
from mcp_server.skill_sync_routes import _proposal_act as _skill_proposal_act
from mcp_server.skill_sync_routes import _proposal_detail as _skill_proposal_detail

# Proposals (phase 2b): per-lane row cap for the unified list, and the review-relevant
# statuses. 'observe' (pre-graduation) and skills' 'retired' (decayed) are NOT proposals
# an operator reviews, so the unified list excludes them (the nav badge still only counts
# 'proposed'). skills terminal = 'promoted' (reached by approve: the lane's accept applies
# the proposal to the skill registry), config terminal = 'applied' (disk write, reached outside
# the dashboard).
_PROPOSALS_LANE_CAP = 200
_SKILL_REVIEW_STATUSES = ("proposed", "accepted", "promoted", "rejected")
_CONFIG_REVIEW_STATUSES = ("proposed", "accepted", "applied", "rejected")


# The dashboard is a THIN review console over two independent lanes. It reuses each
# lane's own _proposal_act (the state transition + side effects) and _proposal_detail
# (the row read), and adds three things on top: a lane-merged list, a normalized detail
# envelope, and a dashboard_audit trail keyed by the namespaced id ("skill:<n>" /
# "config:<n>"). Approve does whatever the lane's accept does: for skills that APPLIES the
# proposal (registry write, status 'promoted'); config apply stays with the config CLI.
# See docs/dashboard-contract.md §"Phase 2b".


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


def _proposal_provenance(evidence: Any) -> tuple[list[int], list[str]]:
    """What a proposal's evidence points at: the episodes it names outright (an
    ``episode_id`` or an ``ep:<id>`` ``source_ref`` on an entry), and the distinct
    ``session_id``s in first-seen order. Lane evidence is keyed by session, not turn, so
    a session is as precise as it gets; guessing turns from it would show arbitrary
    ones, and the starting turns of the earliest sessions at that."""
    if not isinstance(evidence, list):
        return [], []
    explicit: set[int] = set()
    sessions: dict[str, None] = {}
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
            sessions[sid] = None
    return sorted(explicit), list(sessions)


def _proposals_unified(db_url: str, status: str | None, kind: str | None) -> dict[str, Any]:
    """Lane-merged proposal list + the nav-badge pending_count. Each lane is bounded
    (still-'proposed' rows first, then newest first, so a backlog of decided rows can't
    push an unreviewed candidate out of the window), then merged and re-sorted by
    created_at. ``status``/``kind`` narrow the view; ``pending_count`` is always both
    lanes' 'proposed' rows, independent of the view."""
    want_skill = kind in (None, "skill")
    want_config = kind in (None, "config-edit")
    skill_statuses = [s for s in _SKILL_REVIEW_STATUSES if status in (None, s)]
    config_statuses = [s for s in _CONFIG_REVIEW_STATUSES if status in (None, s)]

    with pg_pool.connection(db_url, row_factory=dict_row) as conn:
        out: list[dict[str, Any]] = []
        if want_skill:
            for r in conn.execute(
                "SELECT id, name, summary, status, created_at "
                "FROM skills_lane.skill_gap_candidates WHERE status = ANY(%s) "
                "ORDER BY (status = 'proposed') DESC, created_at DESC, id DESC LIMIT %s",
                (skill_statuses, _PROPOSALS_LANE_CAP),
            ).fetchall():
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
                "ORDER BY (status = 'proposed') DESC, created_at DESC, id DESC LIMIT %s",
                (config_statuses, _PROPOSALS_LANE_CAP),
            ).fetchall():
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
    provenance_episodes, provenance_sessions = _proposal_provenance(evidence)
    with pg_pool.connection(db_url, row_factory=dict_row) as conn:
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

    return {
        "id": full_id,
        "kind": kind,
        "name": name,
        "status": status,
        "evidence": evidence,
        "provenance_episodes": provenance_episodes,
        "provenance_sessions": provenance_sessions,
        "payload": payload,
        "audit_log": audit_log,
    }


def _proposal_decision(
    db_url: str, lane: str, n: int, action: str, note: str | None
) -> dict[str, Any] | None:
    """Delegate to the lane's _proposal_act (approve→accept, reject→reject-with-reason),
    then append a dashboard_audit row. Returns the lane result, or None if the row is gone.
    Only a 'proposed' row takes a decision: the lane checks that status in the same statement
    that changes it, so a stale pane, a double click or a CLI decision that landed first gets a
    'conflict' result (the route answers 409) instead of overwriting an applied, promoted or
    already-decided row. A refusal the lane makes (skills accept: no draft yet, stale draft,
    name taken) changed nothing too, so both are returned as-is and leave no audit row."""
    lane_action = "accept" if action == "approve" else "reject"
    if lane == "skill":
        result = _skill_proposal_act(db_url, n, lane_action, note, expected=("proposed",))
    else:
        # config accept takes an optional scope override; the dashboard never re-scopes,
        # so pass None (the lane keeps the proposal's stored blast radius).
        result = _config_proposal_act(db_url, n, lane_action, note, None, expected=("proposed",))
    if isinstance(result, dict) and result.get("found") is False:
        return None
    if isinstance(result, dict) and result.get("status") in ("refused", "conflict"):
        return result

    audit_action = "proposal_approve" if action == "approve" else "proposal_reject"
    with pg_pool.connection(db_url) as conn:
        conn.execute(
            "INSERT INTO dashboard_audit (action, kind, item_id, detail) VALUES (%s,%s,%s,%s)",
            (
                audit_action,
                _norm_kind(lane),
                f"{lane}:{n}",
                Json({"note": note, "lane_result": result}),
            ),
        )
    return result


def _proposal_ref(request: Request) -> tuple[str, int]:
    """The ``{id}`` path parameter as ``(lane, n)`` (the 400 when it is not a proposal id)."""
    parsed = _parse_proposal_id(request.path_params["id"])
    if parsed is None:
        raise RouteError("bad proposal id")
    return parsed


def register(mcp: Any, db_url: str, authorized: Callable[[Request], bool]) -> None:
    """Mount the unified proposal list, detail and decision (admin gated)."""

    @mcp.custom_route("/dash/api/proposals", methods=["GET"])  # type: ignore[misc]
    @route(authorized)
    def dash_proposals(request: Request) -> dict[str, Any]:
        qp = request.query_params
        status = qp.get("status") or None
        if status == "all":
            status = None
        kind = qp.get("kind") or None
        if kind not in (None, "skill", "config-edit"):
            kind = None
        return _proposals_unified(db_url, status, kind)

    @mcp.custom_route("/dash/api/proposals/{id}", methods=["GET"])  # type: ignore[misc]
    @route(authorized, not_found="proposal not found")
    def dash_proposal_detail(request: Request) -> dict[str, Any] | None:
        lane, n = _proposal_ref(request)
        return _proposal_detail(db_url, lane, n)

    @mcp.custom_route("/dash/api/proposals/{id}/decision", methods=["POST"])  # type: ignore[misc]
    @route(authorized, body=True, not_found="proposal not found")
    def dash_proposal_decision(
        request: Request, body: dict[str, Any]
    ) -> dict[str, Any] | JSONResponse | None:
        lane, n = _proposal_ref(request)
        action = body.get("action")
        note = body.get("note")
        if action not in ("approve", "reject"):
            raise RouteError("action must be 'approve' or 'reject'")
        if note is not None and not isinstance(note, str):
            raise RouteError("'note' must be a string")
        # A reject must carry a reason — it's the lane's reject_reason and the audit note.
        if action == "reject" and not (note and note.strip()):
            raise RouteError("reject requires a non-empty note")
        result = _proposal_decision(db_url, lane, n, action, note)
        # The row moved under the pane (a double click, a CLI decision that landed first):
        # nothing changed, and the pane re-reads on the 409.
        if result is not None and result.get("status") == "conflict":
            return JSONResponse(result, status_code=409)
        return result
