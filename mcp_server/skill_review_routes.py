# mypy: ignore-errors
# Deliberately untyped route-handler module (see skill_sync_routes.py); the pragma keeps it
# clean when a typed module (mcp_server/dash/review.py) imports its _proposal_* helpers via
# follow-imports.
"""Plain-HTTP skill proposal review routes — the dream→skills review CLI's DB seam.

Machine-token-gated like the sync routes (skill_sync_routes.py), and for full-trust callers
only: accepting a proposal changes what every machine is served.

Routes (all POST):
  /skills/proposals   {id?}                          -> list proposed candidates, or one's detail
                                                        (detail: + where accept would apply it,
                                                        + why its last draft failed)
  /skills/proposals/act {id, action, reason?, body?, scope?, force?}
                                                     -> accept (applies) | reject

Accept APPLIES a proposal: the drafted (or reviewer-supplied) SKILL.md is written to the
registry through _upsert_registry — the same write /skills/publish does — and the candidate
goes straight to 'promoted'. Delivery to the machine that owns the skill is the existing
two-way sync's job: the registry now carries a different whole-skill hash with a newer
content_modified_at, so that machine pulls it at its next SessionStart.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from psycopg.rows import dict_row
from psycopg.types.json import Json
from starlette.requests import Request
from starlette.responses import JSONResponse

from dream.skills.skill_doc import (
    body_hash,
    frontmatter_name,
    skill_description,
    valid_skill_name,
)
from dream.skills.skill_ledger import DRAFT_STATE, rollup
from ingestion import pg_pool
from ingestion.project_label import label_from_path
from ingestion.surfaces import SurfaceTrust
from mcp_server.http_helpers import status_conflict
from mcp_server.skill_sync_routes import _SCOPE_RE, _lock_name, _upsert_registry, full_only_guard

# A session started in a home directory carries that directory's basename as its project,
# but the client sync never syncs project:<home> (it would sync the global folder twice), so
# a skill scoped there would reach no machine. Evidence from a home-dir cwd scopes global.
_HOME_DIR_RE = re.compile(
    r"^(?:/home/[^/]+|/Users/[^/]+|/root|[A-Za-z]:[\\/]+Users[\\/]+[^\\/]+)[\\/]*$"
)
_CANDIDATE_COLS = (
    "id, kind, name, direction, target_skills, status, score, evidence, "
    "proposal_body, base_body_hash, summary, salience, source_detector, proposed_patch"
)
_NO_DRAFT = "no draft yet; re-drafts on the next nightly run, or pass --body-file"


def _proposals_list(db_url: str) -> dict:
    with pg_pool.connection(db_url, autocommit=False, row_factory=dict_row) as c:
        rows = c.execute(
            """SELECT id, kind, name, direction, target_skills, status, score,
                      grounded_sessions, judge_sessions, summary, trigger_phrasings,
                      salience, source_detector, (proposal_body IS NOT NULL) AS has_draft
                 FROM skills_lane.skill_gap_candidates
                WHERE status='proposed' AND (rejected_until IS NULL OR rejected_until < now())
                ORDER BY score DESC""",
        ).fetchall()
    return {"proposals": [dict(r) for r in rows]}


def _evidence_scope(c, evidence) -> str:
    """'project:<name>' when every evidence session ran in that one project, else 'global'.
    Unprovable (no sessions, a session with no episodes, an untagged or home-dir session)
    stays global. A label stored as a whole path (the parsers before the backslash fix)
    counts as its folder's name, the label a client sends today, so relabel_projects can
    still move the scope; labels that differ only in case are one project (the most
    common spelling names it); a label that still could not be a scope stays global."""
    sids = sorted(
        {e["session_id"] for e in evidence or [] if isinstance(e, dict) and e.get("session_id")}
    )
    if not sids:
        return "global"
    rows = c.execute(
        "SELECT DISTINCT session_id, project, metadata->>'cwd' AS cwd FROM episodes "
        "WHERE session_id = ANY(%s)",
        (sids,),
    ).fetchall()
    if {r["session_id"] for r in rows} != set(sids):
        return "global"
    projects = [label_from_path(r["project"]) for r in rows]
    if None in projects or len({p.lower() for p in projects}) != 1:
        return "global"
    if any(r["cwd"] and _HOME_DIR_RE.match(r["cwd"]) for r in rows):
        return "global"
    name = max(sorted(set(projects)), key=projects.count)
    scope = f"project:{name}"
    return scope if _SCOPE_RE.match(scope) else "global"


def _plan_apply(c, row, body: str, scope_override: str | None, *, lock: bool = False) -> dict:
    """Where accepting `row` with `body` lands in the registry, and what stands in the way.

    issues: [{code, detail, force}] — force=True issues yield to --force, the others never do.
    Scope: a retune keeps the existing skill's scope; a new skill (derive) is global unless
    all its evidence is from one project; an explicit scope_override wins either way."""
    kind = row["kind"]
    if kind == "retune":
        skill = (row["target_skills"] or [row["name"]])[0]
    else:
        fm = frontmatter_name(body)
        skill = fm if valid_skill_name(fm) else re.sub(r"[^a-z0-9-]", "", row["name"].lower())
    reg = c.execute(
        "SELECT scope, status, COALESCE(body, '') AS body FROM skills_lane.skill_registry "  # nosec B608 — a fixed FOR UPDATE suffix; the name is bound
        "WHERE name=%s" + (" FOR UPDATE" if lock else ""),
        (skill,),
    ).fetchone()
    exists = reg is not None
    registry_body = reg["body"] if exists else None
    current_hash = body_hash(registry_body)

    issues: list[dict] = []

    def issue(code: str, detail: str, force: bool = False) -> None:
        issues.append({"code": code, "detail": detail, "force": force})

    if not body.strip():
        issue("no_draft", _NO_DRAFT)
    if scope_override is not None and not _SCOPE_RE.match(scope_override):
        issue("bad_scope", f"scope must be 'global' or 'project:<name>', got {scope_override!r}")
    if kind not in ("derive", "retune"):  # a leftover 'consolidate' row: nothing applies it
        issue("bad_kind", f"{kind} candidates can't be applied; reject it")
    elif kind == "derive":
        if not valid_skill_name(skill):
            issue(
                "bad_name", f"no usable skill name (frontmatter name or candidate {row['name']!r})"
            )
        elif exists:
            issue(
                "exists",
                f"a skill named '{skill}' already exists in the registry ({reg['scope']}); "
                "accepting would overwrite it — rename it in --body-file, or pass --force",
                force=True,
            )
    elif kind == "retune":
        if not exists:
            issue(
                "no_target",
                f"retune target '{skill}' is not in the skill registry; pass --force to create it",
                force=True,
            )
        else:
            # The name may not change. A registry body with no frontmatter name (a legacy
            # row) only requires the new name, if any, to match the registry key.
            was, now_ = frontmatter_name(registry_body), frontmatter_name(body)
            if body.strip() and (now_ != was if was else now_ not in ("", skill)):
                issue(
                    "renamed",
                    f"the proposed body renames the skill ({was or skill!r} -> {now_!r}); "
                    "a retune must keep its name",
                )
            base = row.get("base_body_hash")
            if base and base != current_hash:
                issue(
                    "stale",
                    f"the registry body of '{skill}' changed since this proposal was drafted; "
                    "it re-drafts against the new body on the next nightly run, or pass --force "
                    "to apply this draft anyway",
                    force=True,
                )
    if exists and reg["status"] == "retired":
        issue(
            "retired",
            f"'{skill}' is retired; accepting would reactivate it — pass --force to do that",
            force=True,
        )

    if scope_override is not None:
        scope = scope_override
    elif exists:
        scope = reg["scope"]
    elif kind == "derive":
        scope = _evidence_scope(c, row["evidence"])
    else:
        scope = "global"
    return {
        "skill": skill,
        "scope": scope,
        "exists": exists,
        "registry_body": registry_body,
        "current_hash": current_hash,
        "issues": issues,
    }


def _proposal_detail(db_url: str, cid: int) -> dict:
    with pg_pool.connection(db_url, autocommit=False, row_factory=dict_row) as c:
        r = c.execute(
            f"SELECT {_CANDIDATE_COLS} FROM skills_lane.skill_gap_candidates WHERE id=%s",  # nosec B608 — _CANDIDATE_COLS is a literal column list
            (cid,),
        ).fetchone()
        if not r:
            return {"found": False}
        d = dict(r)
        if DRAFT_STATE.ready(c, c.info.dsn):  # why the nightly draft failed, if it did
            d.update(
                c.execute(
                    "SELECT draft_attempts, last_draft_error "
                    "FROM skills_lane.skill_gap_candidates WHERE id=%s",
                    (cid,),
                ).fetchone()
            )
        plan = _plan_apply(c, r, r["proposal_body"] or "", None)
        d["apply_to"] = {"skill": plan["skill"], "scope": plan["scope"], "exists": plan["exists"]}
        d["registry_body"] = plan["registry_body"]
        d["apply_issues"] = plan["issues"]
    d["found"] = True
    return d


def _apply_accept(
    c, cid: int, body: str | None, scope: str | None, force: bool, allowed: tuple[str, ...]
) -> dict:
    """One transaction: lock the candidate, re-check its status and the plan, write the registry
    the way a publish does, and mark the candidate promoted with the grounded accept recorded."""
    row = c.execute(
        f"SELECT {_CANDIDATE_COLS} FROM skills_lane.skill_gap_candidates WHERE id=%s FOR UPDATE",  # nosec B608 — _CANDIDATE_COLS is a literal column list
        (cid,),
    ).fetchone()
    if not row:
        c.rollback()
        return {"found": False}
    if row["status"] not in allowed:
        c.rollback()
        return status_conflict(cid, row["status"], allowed)
    for target in row["target_skills"] or [row["name"]]:
        if isinstance(target, str):
            _lock_name(c, target)  # the same lock a publish takes
    text = body if body is not None else (row["proposal_body"] or "")
    plan = _plan_apply(c, row, text, scope, lock=True)
    blocking = [i for i in plan["issues"] if not (force and i["force"])]
    if blocking:
        c.rollback()
        return {
            "status": "refused",
            "id": cid,
            "name": row["name"],
            "reason": blocking[0]["code"],
            "detail": "; ".join(i["detail"] for i in blocking),
            "issues": blocking,
        }

    now = c.execute("SELECT now() AS now").fetchone()["now"]
    _upsert_registry(
        c.cursor(), plan["skill"], plan["scope"], text, skill_description(text) or None, now
    )
    ev = row["evidence"] or []
    ev.append({"session_id": None, "class": "grounded", "signal": "accept"})
    _js, gs, _jw, gw = rollup(ev)
    c.execute(
        "UPDATE skills_lane.skill_gap_candidates SET status='promoted', proposal_body=%s, "
        "evidence=%s::jsonb, grounded_sessions=%s, grounded_weight=%s, updated_at=now() "
        "WHERE id=%s",
        (text, Json(ev), gs, gw, cid),
    )
    c.commit()
    forced = [i["code"] for i in plan["issues"] if i["force"]]
    return {
        "status": "promoted",
        "applied": True,
        "id": cid,
        "name": row["name"],
        "skill": plan["skill"],
        "scope": plan["scope"],
        "created": not plan["exists"],
        "forced": forced,
        "previous_body_hash": plan["current_hash"],
        "body_hash": body_hash(text),
        "detail": f"applied: '{plan['skill']}' ({plan['scope']}) written to the skill registry",
    }


# The statuses accept and reject act on: a decision is final, so a promoted, rejected or
# retired candidate refuses both (a late click must not restart a rejection's cooling window).
_DECIDABLE = ("observe", "proposed", "accepted")


def _proposal_act(
    db_url: str,
    cid: int,
    action: str,
    reason: str | None,
    *,
    body: str | None = None,
    scope: str | None = None,
    force: bool = False,
    expected: tuple[str, ...] | None = None,
) -> dict:
    """accept applies the proposal (see _apply_accept); reject drops it for 30 days. ``expected``
    narrows the statuses the action acts on (the dashboard decides only from 'proposed'); the
    status is checked under the candidate's row lock, so two decisions landing together cannot
    both win: the loser gets a 'conflict' result and the row is untouched."""
    allowed = expected or _DECIDABLE
    with pg_pool.connection(db_url, autocommit=False, row_factory=dict_row) as c:
        if action == "accept":
            return _apply_accept(c, cid, body, scope, force, allowed)

        if action == "reject":
            row = c.execute(
                "SELECT status, evidence FROM skills_lane.skill_gap_candidates "
                "WHERE id=%s FOR UPDATE",
                (cid,),
            ).fetchone()
            if not row:
                return {"found": False}
            if row["status"] not in allowed:
                c.rollback()
                return status_conflict(cid, row["status"], allowed)
            ev = row["evidence"] or []
            ev.append({"session_id": None, "class": "grounded", "signal": "reject"})
            _js, gs, _jw, gw = rollup(ev)
            c.execute(
                "UPDATE skills_lane.skill_gap_candidates SET status='rejected', reject_reason=%s, "
                "rejected_until=now() + interval '30 days', evidence=%s::jsonb, "
                "grounded_sessions=%s, grounded_weight=%s, updated_at=now() WHERE id=%s",
                (reason or "user_rejected", Json(ev), gs, gw, cid),
            )
            c.commit()
            return {"status": "rejected", "id": cid, "reason": reason or "user_rejected"}

    return {"status": "error", "detail": f"unknown action {action!r}"}


def register(
    mcp,
    db_url: str,
    machine_authorized: Callable[[Request], bool],
    resolve_trust: Callable[[Request], SurfaceTrust] | None = None,
) -> None:
    """Wire the /skills/proposals routes onto the FastMCP app. No-op when db_url is empty
    (dev/stdio). Without ``resolve_trust`` every caller is unknown and refused, so a wiring
    slip fails closed."""
    if not db_url:
        return

    full_only = full_only_guard(machine_authorized, resolve_trust)

    @mcp.custom_route("/skills/proposals", methods=["POST"])
    async def _proposals(request: Request) -> JSONResponse:
        def work(b):
            cid = b.get("id")
            return _proposal_detail(db_url, int(cid)) if cid else _proposals_list(db_url)

        return await full_only(request, work)

    @mcp.custom_route("/skills/proposals/act", methods=["POST"])
    async def _act(request: Request) -> JSONResponse:
        def work(b):
            body, scope = b.get("body"), b.get("scope")
            if body is not None and not isinstance(body, str):
                return {"status": "error", "detail": "'body' must be a string"}
            if scope is not None and not isinstance(scope, str):
                return {"status": "error", "detail": "'scope' must be a string"}
            return _proposal_act(
                db_url,
                int(b["id"]),
                b["action"],
                b.get("reason"),
                body=body,
                scope=scope,
                force=bool(b.get("force")),
            )

        return await full_only(request, work)
