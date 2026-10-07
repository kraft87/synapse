# mypy: ignore-errors
# Deliberately untyped route-handler module (FastMCP @custom_route decorators are inherently
# untyped) — already in the mypy pre-commit exclude; the pragma keeps it clean when a typed
# module (mcp_server/dashboard_routes.py) imports its _proposal_* helpers via follow-imports.
"""Plain-HTTP skill sync + review routes — the DB seam that lets the plugin stay DSN-free.

The Claude Code plugin used to reach Postgres directly (SYNAPSE_DB_URL) for two things:
two-way skill sync (the SessionStart hook) and the dream→skills proposal review CLI. Both
now go over these machine-token-gated HTTP routes instead, so a client needs ONE base URL +
an optional bearer — never database access. Mirrors the /ingest + /recall custom routes:
auth via the shared machine token (custom routes bypass FastMCP's auth middleware by design),
PG work in a threadpool, fail-soft JSON.

Routes (all POST):
  /skills/list        {scope}                       -> active skills (name, desc, mtime, file shas)
  /skills/fetch       {name}                         -> body + file contents (base64) for a pull
  /skills/publish     {name, scope, body, ...files}  -> upsert registry + replace files
  /skills/overwrite   {name, scope, body, mtime}     -> record a clobbered local edit in history
  /skills/proposals   {id?}                          -> list proposed candidates, or one's detail
                                                        (detail: + where accept would apply it)
  /skills/proposals/act {id, action, reason?, body?, scope?, force?}
                                                     -> accept (applies) | reject | promote (no-op)
"""

from __future__ import annotations

import base64
import logging
import re
from collections.abc import Callable
from datetime import datetime

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json
from starlette.requests import Request
from starlette.responses import JSONResponse

from dream.skills.skill_doc import body_hash, frontmatter_name, valid_skill_name
from dream.skills.skill_measure import skill_description
from mcp_server.http_helpers import _iso, guarded_json

logger = logging.getLogger(__name__)


def _dt(iso: str | None) -> datetime | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- sync


def _list_skills(db_url: str, scope: str) -> dict:
    with psycopg.connect(db_url, row_factory=dict_row) as c:
        skills = c.execute(
            "SELECT name, scope, COALESCE(body,'') AS body, COALESCE(description,'') AS description, "
            "content_modified_at FROM skills_lane.skill_registry "
            "WHERE status='active' AND scope=%s ORDER BY name",
            (scope,),
        ).fetchall()
        out = []
        for s in skills:
            files = c.execute(
                "SELECT path, sha256 FROM skills_lane.skill_files WHERE skill_name=%s ORDER BY path",
                (s["name"],),
            ).fetchall()
            out.append(
                {
                    "name": s["name"],
                    "scope": s["scope"],
                    "body": s["body"],
                    "description": s["description"],
                    "content_modified_at": _iso(s["content_modified_at"]),
                    "files": [{"path": f["path"], "sha256": f["sha256"]} for f in files],
                }
            )
    return {"skills": out}


def _fetch_skill(db_url: str, name: str) -> dict:
    with psycopg.connect(db_url, row_factory=dict_row) as c:
        row = c.execute(
            "SELECT name, scope, COALESCE(body,'') AS body, COALESCE(description,'') AS description, "
            "content_modified_at FROM skills_lane.skill_registry WHERE name=%s AND status='active'",
            (name,),
        ).fetchone()
        if not row:
            return {"found": False}
        files = c.execute(
            "SELECT path, content, is_executable FROM skills_lane.skill_files "
            "WHERE skill_name=%s ORDER BY path",
            (name,),
        ).fetchall()
    return {
        "found": True,
        "name": row["name"],
        "scope": row["scope"],
        "body": row["body"],
        "description": row["description"],
        "content_modified_at": _iso(row["content_modified_at"]),
        "files": [
            {
                "path": f["path"],
                "content_b64": base64.b64encode(bytes(f["content"])).decode(),
                "is_executable": f["is_executable"],
            }
            for f in files
        ],
    }


def _upsert_registry(
    cur, name: str, scope: str, body: str, description: str | None, content_modified_at
) -> None:
    """THE registry write for a skill's SKILL.md — /skills/publish and the review accept both
    go through it, so an accepted proposal lands exactly like a client publish.

    Nulls the description embedding when the description changed so the server lane
    re-embeds it (overlap detection). The skill_history trigger fires automatically on a body
    change, so a superseded body stays recoverable."""
    cur.execute(
        """INSERT INTO skills_lane.skill_registry
               (name, scope, body, description, status, content_modified_at)
             VALUES (%s,%s,%s,%s,'active',%s)
           ON CONFLICT (name) DO UPDATE SET
             scope=EXCLUDED.scope, body=EXCLUDED.body, description=EXCLUDED.description,
             status='active', content_modified_at=EXCLUDED.content_modified_at,
             description_embedding = CASE
               WHEN skills_lane.skill_registry.description IS DISTINCT FROM EXCLUDED.description
               THEN NULL ELSE skills_lane.skill_registry.description_embedding END,
             updated_at=now()""",
        (name, scope, body, description, content_modified_at),
    )


def _publish_skill(db_url: str, p: dict) -> dict:
    name = p["name"]
    with psycopg.connect(db_url) as c:
        cur = c.cursor()
        _upsert_registry(
            cur,
            name,
            p.get("scope", "global"),
            p.get("body", ""),
            p.get("description") or None,
            _dt(p.get("content_modified_at")),
        )
        cur.execute("DELETE FROM skills_lane.skill_files WHERE skill_name=%s", (name,))
        for f in p.get("files", []):
            content = base64.b64decode(f["content_b64"])
            cur.execute(
                """INSERT INTO skills_lane.skill_files
                       (skill_name, path, content, sha256, size, is_executable)
                     VALUES (%s,%s,%s,%s,%s,%s)""",
                (
                    name,
                    f["path"],
                    content,
                    f["sha256"],
                    f.get("size", len(content)),
                    bool(f.get("is_executable", False)),
                ),
            )
        c.commit()
    return {"status": "ok", "name": name}


def _record_overwrite(db_url: str, p: dict) -> dict:
    with psycopg.connect(db_url) as c:
        c.execute(
            "INSERT INTO skills_lane.skill_history (name, scope, body, content_modified_at, op) "
            "VALUES (%s,%s,%s,%s,'disk_overwrite')",
            (p["name"], p.get("scope"), p.get("body", ""), _dt(p.get("content_modified_at"))),
        )
        c.commit()
    return {"status": "ok"}


# ----------------------------------------------------------------------- review
#
# Accept APPLIES a proposal: the drafted (or reviewer-supplied) SKILL.md is written to the
# registry through _upsert_registry — the same write /skills/publish does — and the candidate
# goes straight to 'promoted'. Delivery to the machine that owns the skill is the existing
# two-way sync's job: the registry now carries a different whole-skill hash with a newer
# content_modified_at, so that machine pulls it at its next SessionStart.

# --scope values: 'global' or 'project:<dir basename>' (what the client sync keys on).
_SCOPE_RE = re.compile(r"^(global|project:[^/\\\r\n]+)$")
# A session started in a home directory carries that directory's basename as its project,
# but the client sync never syncs project:<home> (it would sync the global folder twice), so
# a skill scoped there would reach no machine. Evidence from a home-dir cwd scopes global.
_HOME_DIR_RE = re.compile(
    r"^(?:/home/[^/]+|/Users/[^/]+|/root|[A-Za-z]:[\\/]+Users[\\/]+[^\\/]+)[\\/]*$"
)
_CANDIDATE_COLS = (
    "id, kind, name, direction, target_skills, status, score, evidence, proposal_path, "
    "proposal_body, base_body_hash, summary, salience, source_detector, proposed_patch"
)
_NO_DRAFT = "no draft yet; re-drafts on the next nightly run, or pass --body-file"


def _proposals_list(db_url: str) -> dict:
    with psycopg.connect(db_url, row_factory=dict_row) as c:
        rows = c.execute(
            """SELECT id, kind, name, direction, target_skills, status, score,
                      grounded_sessions, judge_sessions, summary, proposal_path, trigger_phrasings,
                      salience, source_detector, (proposal_body IS NOT NULL) AS has_draft
                 FROM skills_lane.skill_gap_candidates
                WHERE status='proposed' AND (rejected_until IS NULL OR rejected_until < now())
                ORDER BY score DESC""",
        ).fetchall()
    return {"proposals": [dict(r) for r in rows]}


def _evidence_scope(c, evidence) -> str:
    """'project:<name>' when every evidence session ran in that one project, else 'global'.
    Unprovable (no sessions, a session with no episodes, an untagged or home-dir session)
    stays global."""
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
    projects = {r["project"] for r in rows}
    if len(projects) != 1 or None in projects:
        return "global"
    if any(r["cwd"] and _HOME_DIR_RE.match(r["cwd"]) for r in rows):
        return "global"
    return f"project:{projects.pop()}"


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
        "SELECT scope, COALESCE(body, '') AS body FROM skills_lane.skill_registry WHERE name=%s"
        + (" FOR UPDATE" if lock else ""),
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
    if kind == "derive":
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
    with psycopg.connect(db_url, row_factory=dict_row) as c:
        r = c.execute(
            f"SELECT {_CANDIDATE_COLS} FROM skills_lane.skill_gap_candidates WHERE id=%s",
            (cid,),
        ).fetchone()
        if not r:
            return {"found": False}
        d = dict(r)
        if r["kind"] != "consolidate":  # consolidate is never applied (merge by hand)
            plan = _plan_apply(c, r, r["proposal_body"] or "", None)
            d["apply_to"] = {
                "skill": plan["skill"],
                "scope": plan["scope"],
                "exists": plan["exists"],
            }
            d["registry_body"] = plan["registry_body"]
            d["apply_issues"] = plan["issues"]
    d["found"] = True
    return d


def _apply_accept(c, cid: int, body: str | None, scope: str | None, force: bool) -> dict:
    """One transaction: lock the candidate, re-check the plan, write the registry the way a
    publish does, and mark the candidate promoted with the grounded accept recorded."""
    from dream.skills.skill_ledger import _rollup

    row = c.execute(
        f"SELECT {_CANDIDATE_COLS} FROM skills_lane.skill_gap_candidates WHERE id=%s FOR UPDATE",
        (cid,),
    ).fetchone()
    if not row:
        c.rollback()
        return {"found": False}
    if row["status"] == "promoted":
        c.rollback()
        return {
            "status": "refused",
            "id": cid,
            "reason": "already_applied",
            "detail": f"[{cid}] {row['name']} was already applied",
        }
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
    _js, gs, _jw, gw = _rollup(ev)
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


def _proposal_act(
    db_url: str,
    cid: int,
    action: str,
    reason: str | None,
    llm: Callable,
    *,
    body: str | None = None,
    scope: str | None = None,
    force: bool = False,
) -> dict:
    from dream.skills.skill_ledger import _rollup

    with psycopg.connect(db_url, row_factory=dict_row) as c:
        cur = c.cursor()
        row = c.execute(
            "SELECT kind, status, name, proposal_path, proposal_body, evidence "
            "FROM skills_lane.skill_gap_candidates WHERE id=%s",
            (cid,),
        ).fetchone()
        if not row:
            return {"found": False}

        if action == "accept" and row["kind"] != "consolidate":
            note = ""
            drafted = (body if body is not None else row["proposal_body"] or "").strip()
            if row["kind"] == "retune" and row["status"] != "promoted" and drafted:
                note = llm(c, cid, row["name"])  # advisory routing-eval, never blocks
            c.commit()  # close the read; the apply runs in its own locked transaction
            res = _apply_accept(c, cid, body, scope, force)
            if note and res.get("status") == "promoted":
                res["routing_eval"] = note
            return res

        if action == "accept":  # consolidate: a merge is done by hand, accept only records it
            ev = row["evidence"] or []
            ev.append({"session_id": None, "class": "grounded", "signal": "accept"})
            _js, gs, _jw, gw = _rollup(ev)
            cur.execute(
                "UPDATE skills_lane.skill_gap_candidates SET status='accepted', evidence=%s::jsonb, "
                "grounded_sessions=%s, grounded_weight=%s, updated_at=now() WHERE id=%s",
                (Json(ev), gs, gw, cid),
            )
            c.commit()
            return {
                "status": "accepted",
                "id": cid,
                "name": row["name"],
                "proposal_path": row["proposal_path"],
                "proposal_body": row["proposal_body"],
                "routing_eval": "",
            }

        if action == "reject":
            ev = row["evidence"] or []
            ev.append({"session_id": None, "class": "grounded", "signal": "reject"})
            _js, gs, _jw, gw = _rollup(ev)
            cur.execute(
                "UPDATE skills_lane.skill_gap_candidates SET status='rejected', reject_reason=%s, "
                "rejected_until=now() + interval '30 days', evidence=%s::jsonb, "
                "grounded_sessions=%s, grounded_weight=%s, updated_at=now() WHERE id=%s",
                (reason or "user_rejected", Json(ev), gs, gw, cid),
            )
            c.commit()
            return {"status": "rejected", "id": cid, "reason": reason or "user_rejected"}

        if action == "promote":
            # Kept for older clients: accept now applies, so promote has nothing left to do
            # except confirm a hand-merged consolidate (the one kind accept doesn't apply).
            if row["status"] == "promoted":
                return {
                    "status": "promoted",
                    "noop": True,
                    "id": cid,
                    "name": row["name"],
                    "detail": f"[{cid}] {row['name']} was applied at accept; nothing to promote",
                }
            if row["status"] == "accepted" and row["kind"] == "consolidate":
                cur.execute(
                    "UPDATE skills_lane.skill_gap_candidates SET status='promoted', "
                    "updated_at=now() WHERE id=%s",
                    (cid,),
                )
                c.commit()
                return {"status": "promoted", "id": cid, "name": row["name"]}
            if row["status"] == "accepted":
                detail = (
                    f"[{cid}] was accepted before accept applied changes; "
                    f"run `accept {cid}` to apply it now"
                )
            else:
                detail = (
                    f"candidate is '{row['status']}'; `accept {cid}` applies the change, "
                    "there is no separate promote step"
                )
            return {"status": "refused", "id": cid, "detail": detail}

    return {"status": "error", "detail": f"unknown action {action!r}"}


def _routing_eval(c, cid: int, name: str) -> str:
    """Advisory: would the under-trigger phrasings route to `name` under current descriptions?
    Best-effort single LLM pass; any failure degrades to a skip note (never blocks accept)."""
    try:
        import json as _json

        from ingestion.llm_client import create_llm_client

        cur = c.cursor()
        ev = cur.execute(
            "SELECT evidence FROM skills_lane.skill_gap_candidates WHERE id=%s", (cid,)
        ).fetchone()[0]
        phrasings = [
            e.get("phrasing", "")
            for e in ev
            if e.get("signal") == "under_trigger" and e.get("phrasing")
        ]
        if not phrasings:
            return "routing-eval: no phrasings to eval"
        catalog = "\n".join(
            f"- {n}: {d}"
            for n, d in cur.execute(
                "SELECT name, description FROM skills_lane.skill_registry ORDER BY name"
            ).fetchall()
        )
        prompt = (
            f"Skill catalog:\n{catalog}\n\nFor each user phrasing, name the ONE skill whose "
            f"description best matches, or 'none'. We expect '{name}' to be the match.\n"
            + "\n".join(f"- {p}" for p in phrasings[:8])
            + '\n\nOutput JSON only: {"matches":["skill-or-none", ...]} in phrasing order.'
        )

        def _gen() -> str:
            from ingestion.llm_client import stage_model

            resp = create_llm_client().messages.create(
                model=stage_model("SKILLS", "claude-opus-4-8"),
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            return str(resp.content[0].text)

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1) as pool:
            raw = pool.submit(_gen).result(timeout=120)
        s, e = raw.find("{"), raw.rfind("}")
        matches = _json.loads(raw[s : e + 1]).get("matches", []) if s >= 0 else []
        hit = sum(1 for m in matches if m == name)
        return f"routing-eval: {hit}/{len(matches)} missed phrasings already map to '{name}'"
    except Exception as ex:  # pragma: no cover - advisory only
        return f"routing-eval skipped: {ex}"


# -------------------------------------------------------------------- register


def register(mcp, db_url: str, machine_authorized: Callable[[Request], bool]) -> None:
    """Wire the /skills/* routes onto the FastMCP app. No-op when db_url is empty (dev/stdio)."""
    if not db_url:
        return

    async def _guarded(request: Request, work):
        return await guarded_json(request, machine_authorized, work, label="skill route")

    @mcp.custom_route("/skills/list", methods=["POST"])
    async def _list(request: Request) -> JSONResponse:
        return await _guarded(request, lambda b: _list_skills(db_url, b.get("scope", "global")))

    @mcp.custom_route("/skills/fetch", methods=["POST"])
    async def _fetch(request: Request) -> JSONResponse:
        return await _guarded(request, lambda b: _fetch_skill(db_url, b["name"]))

    @mcp.custom_route("/skills/publish", methods=["POST"])
    async def _publish(request: Request) -> JSONResponse:
        return await _guarded(request, lambda b: _publish_skill(db_url, b))

    @mcp.custom_route("/skills/overwrite", methods=["POST"])
    async def _overwrite(request: Request) -> JSONResponse:
        return await _guarded(request, lambda b: _record_overwrite(db_url, b))

    @mcp.custom_route("/skills/proposals", methods=["POST"])
    async def _proposals(request: Request) -> JSONResponse:
        def work(b):
            cid = b.get("id")
            return _proposal_detail(db_url, int(cid)) if cid else _proposals_list(db_url)

        return await _guarded(request, work)

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
                _routing_eval,
                body=body,
                scope=scope,
                force=bool(b.get("force")),
            )

        return await _guarded(request, work)
