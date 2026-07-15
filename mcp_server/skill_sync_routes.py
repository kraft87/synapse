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
  /skills/proposals/act {id, action, reason?}        -> accept | reject | promote
"""

from __future__ import annotations

import base64
import logging
from collections.abc import Callable
from datetime import datetime

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


def _dt(iso: str | None) -> datetime | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return None


def _iso(dt) -> str | None:
    return dt.isoformat() if dt else None


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


def _publish_skill(db_url: str, p: dict) -> dict:
    name = p["name"]
    with psycopg.connect(db_url) as c:
        cur = c.cursor()
        # Upsert the registry row. Null the description embedding when the description changed
        # so the server lane re-embeds it (overlap detection). The skill_history trigger fires
        # automatically on a body change, so a superseded body stays recoverable.
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
            (
                name,
                p.get("scope", "global"),
                p.get("body", ""),
                p.get("description") or None,
                _dt(p.get("content_modified_at")),
            ),
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


def _proposals_list(db_url: str) -> dict:
    with psycopg.connect(db_url, row_factory=dict_row) as c:
        rows = c.execute(
            """SELECT id, kind, name, direction, target_skills, status, score,
                      grounded_sessions, judge_sessions, summary, proposal_path, trigger_phrasings
                 FROM skills_lane.skill_gap_candidates
                WHERE status='proposed' AND (rejected_until IS NULL OR rejected_until < now())
                ORDER BY score DESC""",
        ).fetchall()
    return {"proposals": [dict(r) for r in rows]}


def _proposal_detail(db_url: str, cid: int) -> dict:
    with psycopg.connect(db_url, row_factory=dict_row) as c:
        r = c.execute(
            "SELECT id, kind, name, direction, target_skills, status, score, evidence, "
            "proposal_path, proposal_body, summary FROM skills_lane.skill_gap_candidates WHERE id=%s",
            (cid,),
        ).fetchone()
    if not r:
        return {"found": False}
    d = dict(r)
    d["found"] = True
    return d


def _proposal_act(db_url: str, cid: int, action: str, reason: str | None, llm: Callable) -> dict:
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

        if action == "accept":
            note = ""
            if row["kind"] == "retune":
                note = llm(c, cid, row["name"])  # advisory routing-eval
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
                "routing_eval": note,
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
            if row["status"] != "accepted":
                return {
                    "status": "refused",
                    "detail": f"candidate is '{row['status']}', not 'accepted'",
                }
            cur.execute(
                "UPDATE skills_lane.skill_gap_candidates SET status='promoted', updated_at=now() WHERE id=%s",
                (cid,),
            )
            c.commit()
            return {"status": "promoted", "id": cid, "name": row["name"]}

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
        if not machine_authorized(request):
            return JSONResponse({"status": "error", "detail": "unauthorized"}, status_code=401)
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"status": "error", "detail": "invalid JSON"}, status_code=400)
        try:
            out = await run_in_threadpool(work, body)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("skill route failed")
            return JSONResponse({"status": "error", "detail": str(exc)[:200]}, status_code=500)
        return JSONResponse(out)

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
        return await _guarded(
            request,
            lambda b: _proposal_act(
                db_url, int(b["id"]), b["action"], b.get("reason"), _routing_eval
            ),
        )
