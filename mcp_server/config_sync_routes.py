# mypy: ignore-errors
# Deliberately untyped route-handler module (FastMCP @custom_route decorators are inherently
# untyped) — already in the mypy pre-commit exclude; the pragma keeps it clean when a typed
# module (mcp_server/dashboard_routes.py) imports its _proposal_* helpers via follow-imports.
"""Plain-HTTP config-lane routes — the DB seam that mirrors a machine's config into Postgres.

Same shape as skill_sync_routes: machine-token-gated custom routes (they bypass FastMCP's auth
middleware by design), PG work in a threadpool, fail-soft JSON. The plugin's SessionStart hook
PUSHES each surface's opted-in config files here so the dream pipeline (server-side) can read them
and propose edits. The plugin needs only a base URL + bearer — never database access.

A "surface" is a machine/install. file_key is the file's path relative to the config root, stable
across machines.

Which surface a call acts on comes from the caller's BEARER (schema 054), never from the body:

  * full-trust device   -> unchanged: may name any ``surface`` in the body (the operator's view).
  * restricted device   -> its OWN surface only; a body ``surface`` is ignored. Proposals are
                           limited to ones whose provenance is that surface (see _OWN_PROPOSAL).
  * unknown / root / no resolver -> nothing: empty reads, 403 on writes (fail closed).

A restricted host mirrors config from employer hardware and must not read (or overwrite) the
personal machines' CLAUDE.md, nor review rules mined from their conversations.

Routes (all POST):
  /config/publish       {surface, scope?, file_key, abs_path, content, modified_at?} -> upsert mirror row
  /config/list          {surface}                                  -> [{scope, file_key, hash, mtime}]
  /config/fetch         {surface, file_key, scope?}                -> {found, content, abs_path}
  /config/proposals     {id?}                                      -> proposed rule edits, or one's detail
  /config/proposals/act {id, action, reason?, scope?}              -> accept | reject | apply
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from datetime import datetime

import psycopg
from psycopg.rows import dict_row
from starlette.requests import Request
from starlette.responses import JSONResponse

from ingestion.surfaces import UNKNOWN_SURFACE, SurfaceTrust
from mcp_server.http_helpers import _iso, err, guarded_json

logger = logging.getLogger(__name__)


def _dt(iso: str | None) -> datetime | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return None


def _publish_config(db_url: str, p: dict) -> dict:
    """Upsert one mirror row for (surface, scope, file_key). Idempotent: a re-push of unchanged
    content is a no-op (the hash matches), so a SessionStart that changed nothing writes nothing."""
    surface = p["surface"]
    scope = p.get("scope", "global")
    file_key = p["file_key"]
    content = p.get("content", "") or ""
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        row = conn.execute(
            "INSERT INTO config_lane.config_registry "
            "  (surface_id, scope, file_key, abs_path, content, content_hash, modified_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, now()) "
            "ON CONFLICT (surface_id, scope, file_key) DO UPDATE SET "
            "  abs_path = EXCLUDED.abs_path, content = EXCLUDED.content, "
            "  content_hash = EXCLUDED.content_hash, modified_at = EXCLUDED.modified_at, "
            "  updated_at = now() "
            "WHERE config_lane.config_registry.content_hash <> EXCLUDED.content_hash "
            "RETURNING content_hash",
            (
                surface,
                scope,
                file_key,
                p.get("abs_path", ""),
                content,
                digest,
                _dt(p.get("modified_at")),
            ),
        ).fetchone()
    # RETURNING is NULL when the WHERE (hash changed) filtered the UPDATE out — i.e. unchanged.
    return {"status": "ok", "changed": row is not None, "content_hash": digest}


def _list_configs(db_url: str, surface: str) -> dict:
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        rows = conn.execute(
            "SELECT scope, file_key, content_hash, modified_at FROM config_lane.config_registry "
            "WHERE surface_id = %s ORDER BY scope, file_key",
            (surface,),
        ).fetchall()
    return {
        "files": [
            {
                "scope": r["scope"],
                "file_key": r["file_key"],
                "content_hash": r["content_hash"],
                "modified_at": _iso(r["modified_at"]),
            }
            for r in rows
        ]
    }


def _fetch_config(db_url: str, surface: str, file_key: str, scope: str = "global") -> dict:
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        row = conn.execute(
            "SELECT content, abs_path, content_hash, modified_at FROM config_lane.config_registry "
            "WHERE surface_id = %s AND scope = %s AND file_key = %s",
            (surface, scope, file_key),
        ).fetchone()
    if not row:
        return {"found": False}
    return {
        "found": True,
        "content": row["content"],
        "abs_path": row["abs_path"],
        "content_hash": row["content_hash"],
        "modified_at": _iso(row["modified_at"]),
    }


# ----------------------------------------------------------------------- review


#: A proposal "belongs" to a surface when it targets that surface (scope='local') or when ALL of
#: its evidence comes from sessions that surface ingested (episodes.surface_id, schema 055). An
#: evidence item with no session id, or a session with no stamped turn, proves nothing, so a
#: proposal mined from mixed or personal sessions never reaches a restricted host. Literal SQL;
#: only %(own)s is bound.
_OWN_PROPOSAL = (
    " AND (surface_id = %(own)s OR (jsonb_array_length(evidence) > 0 AND NOT EXISTS ("
    "SELECT 1 FROM jsonb_array_elements(evidence) e WHERE NOT EXISTS ("
    "SELECT 1 FROM episodes ep WHERE ep.session_id = e->>'session_id' "
    "AND ep.surface_id = %(own)s))))"
)


def _proposals_list(db_url: str, own_surface: str | None = None) -> dict:
    """The rule edits awaiting review — only 'proposed' (the dream gate graduated them past observe).

    ``own_surface`` limits the list to that surface's proposals (_OWN_PROPOSAL), the restricted
    caller's view. None is the unfiltered (full-trust / dashboard) view."""
    sql = (
        "SELECT id, kind, file_key, scope, summary, evidence, updated_at "
        "FROM config_lane.config_proposals WHERE status='proposed'"
    )
    if own_surface:
        sql += _OWN_PROPOSAL
    sql += " ORDER BY updated_at DESC"
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        rows = conn.execute(sql, {"own": own_surface}).fetchall()
    out = []
    for r in rows:
        ev = r["evidence"] or []
        out.append(
            {
                "id": r["id"],
                "kind": r["kind"],
                "file_key": r["file_key"],
                "scope": r["scope"],
                "summary": r["summary"],
                "sessions": len({e.get("session_id") for e in ev if e.get("session_id")}),
                "evidence_count": len(ev),
            }
        )
    return {"proposals": out}


def _proposal_owned(db_url: str, cid: int, own_surface: str) -> bool:
    """True when proposal ``cid`` belongs to ``own_surface`` (see _OWN_PROPOSAL)."""
    with psycopg.connect(db_url) as conn:
        row = conn.execute(
            "SELECT 1 FROM config_lane.config_proposals WHERE id = %(id)s" + _OWN_PROPOSAL,
            {"id": cid, "own": own_surface},
        ).fetchone()
    return row is not None


def _proposal_detail(db_url: str, cid: int, own_surface: str | None = None) -> dict:
    """One proposal. ``own_surface`` (restricted caller) makes anything outside that surface's
    proposals read as not found, indistinguishable from a missing id, so ids can't be probed."""
    sql = (
        "SELECT id, kind, file_key, scope, diff, summary, evidence, status "
        "FROM config_lane.config_proposals WHERE id=%(id)s"
    )
    if own_surface:
        sql += _OWN_PROPOSAL
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        r = conn.execute(sql, {"id": cid, "own": own_surface}).fetchone()
    if not r:
        return {"found": False}
    d = dict(r)
    d["found"] = True
    return d


def _proposal_act(
    db_url: str, cid: int, action: str, reason: str | None, scope: str | None
) -> dict:
    """Walk a proposal through the gate. accept records the human yes (and can override the blast
    radius via scope); apply marks it written-to-disk (the client did the actual write); reject drops
    it. The two-step accept->apply mirrors skills' accept->promote: 'accepted' = approved, 'applied' =
    on disk, so a failed disk write leaves it recoverable at 'accepted'."""
    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        cur = conn.cursor()
        row = conn.execute(
            "SELECT status, kind, file_key, scope, diff, summary "
            "FROM config_lane.config_proposals WHERE id=%s",
            (cid,),
        ).fetchone()
        if not row:
            return {"found": False}

        if action == "accept":
            new_scope = scope if scope in ("local", "general") else row["scope"]
            cur.execute(
                "UPDATE config_lane.config_proposals SET status='accepted', scope=%s, updated_at=now() "
                "WHERE id=%s",
                (new_scope, cid),
            )
            conn.commit()
            return {
                "status": "accepted",
                "id": cid,
                "kind": row["kind"],
                "file_key": row["file_key"],
                "scope": new_scope,
                "rule": row["diff"] or row["summary"],
                "summary": row["summary"],
            }

        if action == "reject":
            cur.execute(
                "UPDATE config_lane.config_proposals SET status='rejected', reject_reason=%s, "
                "updated_at=now() WHERE id=%s",
                (reason or "user_rejected", cid),
            )
            conn.commit()
            return {"status": "rejected", "id": cid, "reason": reason or "user_rejected"}

        if action == "apply":
            if row["status"] != "accepted":
                return {
                    "status": "refused",
                    "detail": f"proposal is '{row['status']}', not 'accepted'",
                }
            cur.execute(
                "UPDATE config_lane.config_proposals SET status='applied', updated_at=now() WHERE id=%s",
                (cid,),
            )
            conn.commit()
            return {"status": "applied", "id": cid}

    return {"status": "error", "detail": f"unknown action {action!r}"}


def _target_surface(st: SurfaceTrust, body_surface: str | None) -> str | None:
    """The surface a mirror read/write acts on, or None for "nothing".

    Full trust keeps the pre-055 behaviour (the body names the surface). A restricted caller is
    pinned to its own credential-bound id and the body is ignored. Anyone else gets None."""
    if st.known and not st.restricted:
        return body_surface or st.surface_id
    return st.stamp_surface_id if st.restricted else None


def _proposal_scope(st: SurfaceTrust) -> tuple[bool, str | None]:
    """(allowed, own_surface) for the review routes: full trust -> (True, None) = unfiltered;
    a credential-bound restricted caller -> (True, its id); anyone else -> (False, None)."""
    if st.known and not st.restricted:
        return True, None
    own = st.stamp_surface_id if st.restricted else None
    return (own is not None), own


def register(
    mcp,
    db_url: str,
    machine_authorized: Callable[[Request], bool],
    resolve_trust: Callable[[Request], SurfaceTrust] | None = None,
) -> None:
    """Wire the /config/* routes onto the FastMCP app. No-op when db_url is empty (dev/stdio).

    ``resolve_trust`` maps a request to its bearer-resolved verdict. Without it every caller is
    treated as unknown and served nothing, so a wiring slip fails closed."""
    if not db_url:
        return

    def _trust(request: Request) -> SurfaceTrust:
        # Called inside the threadpool work: resolution may read Postgres.
        return resolve_trust(request) if resolve_trust else UNKNOWN_SURFACE

    def _forbidden():
        return err("forbidden for this surface", 403)

    async def _guarded(request: Request, work):
        return await guarded_json(request, machine_authorized, work, label="config route")

    @mcp.custom_route("/config/publish", methods=["POST"])
    async def _publish(request: Request) -> JSONResponse:
        def work(b):
            target = _target_surface(_trust(request), b.get("surface"))
            if not target:
                return _forbidden()
            return _publish_config(db_url, {**b, "surface": target})

        return await _guarded(request, work)

    @mcp.custom_route("/config/list", methods=["POST"])
    async def _list(request: Request) -> JSONResponse:
        def work(b):
            target = _target_surface(_trust(request), b.get("surface"))
            return _list_configs(db_url, target) if target else {"files": []}

        return await _guarded(request, work)

    @mcp.custom_route("/config/fetch", methods=["POST"])
    async def _fetch(request: Request) -> JSONResponse:
        def work(b):
            target = _target_surface(_trust(request), b.get("surface"))
            if not target:
                return {"found": False}
            return _fetch_config(db_url, target, b["file_key"], b.get("scope", "global"))

        return await _guarded(request, work)

    @mcp.custom_route("/config/proposals", methods=["POST"])
    async def _proposals(request: Request) -> JSONResponse:
        def work(b):
            allowed, own = _proposal_scope(_trust(request))
            cid = b.get("id")
            if not allowed:
                return {"found": False} if cid else {"proposals": []}
            return _proposal_detail(db_url, int(cid), own) if cid else _proposals_list(db_url, own)

        return await _guarded(request, work)

    @mcp.custom_route("/config/proposals/act", methods=["POST"])
    async def _act(request: Request) -> JSONResponse:
        def work(b):
            allowed, own = _proposal_scope(_trust(request))
            cid = int(b["id"])
            if not allowed or (own and not _proposal_owned(db_url, cid, own)):
                return _forbidden()
            return _proposal_act(db_url, cid, b["action"], b.get("reason"), b.get("scope"))

        return await _guarded(request, work)
