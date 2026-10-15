# mypy: ignore-errors
# Deliberately untyped route-handler module (FastMCP @custom_route decorators are inherently
# untyped) — already in the mypy pre-commit exclude; the pragma keeps it clean when a typed
# module (mcp_server/dashboard_routes.py) imports its _proposal_* helpers via follow-imports.
"""Plain-HTTP config-lane routes — the DB seam that mirrors a machine's config into Postgres.

Same shape as skill_sync_routes: machine-token-gated custom routes (they bypass FastMCP's auth
middleware by design), PG work in a threadpool, fail-soft JSON. The plugin's SessionStart hook
PUSHES each surface's opted-in config files here; the server keeps them as the mirror the dashboard
lists, and the nightly config lane skips corrections that restate a line of it. The plugin needs
only a base URL + bearer — never database access.

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
  /config/publish       {surface, files: [{scope?, file_key, abs_path, content, modified_at?}]}
                        -> upsert one mirror row per file. A body that is itself one file (the
                        form clients sent before the batch) is still accepted.
  /config/proposals     {id?}                                      -> rule edits awaiting a decision
                        ('proposed') or a disk write ('accepted'), or one's detail
  /config/proposals/act {id, action, reason?, scope?}              -> accept | reject | apply
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from datetime import datetime

from psycopg.rows import dict_row
from starlette.requests import Request
from starlette.responses import JSONResponse

from ingestion import pg_pool
from ingestion.surfaces import UNKNOWN_SURFACE, SurfaceTrust, own_surface_columns_ready
from mcp_server.http_helpers import err, guarded_json, status_conflict

logger = logging.getLogger(__name__)


def _dt(iso: str | None) -> datetime | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return None


def _publish_config(db_url: str, surface: str, files: list[dict]) -> dict:
    """Upsert one mirror row per file for `surface` (keyed by scope and file_key), in one
    transaction. Idempotent: a re-push of unchanged content is a no-op (the hash matches), so a
    SessionStart that changed nothing writes nothing. `changed` counts the rows that did."""
    changed = 0
    with pg_pool.connection(db_url, autocommit=False) as conn:
        for p in files:
            content = p.get("content", "") or ""
            row = conn.execute(
                "INSERT INTO config_lane.config_registry "
                "  (surface_id, scope, file_key, abs_path, content, content_hash, modified_at, "
                "   updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, now()) "
                "ON CONFLICT (surface_id, scope, file_key) DO UPDATE SET "
                "  abs_path = EXCLUDED.abs_path, content = EXCLUDED.content, "
                "  content_hash = EXCLUDED.content_hash, modified_at = EXCLUDED.modified_at, "
                "  updated_at = now() "
                "WHERE config_lane.config_registry.content_hash <> EXCLUDED.content_hash "
                "RETURNING content_hash",
                (
                    surface,
                    p.get("scope", "global"),
                    p["file_key"],
                    p.get("abs_path", ""),
                    content,
                    hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    _dt(p.get("modified_at")),
                ),
            ).fetchone()
            # RETURNING is NULL when the WHERE (hash changed) filtered the UPDATE out.
            changed += row is not None
    return {"status": "ok", "changed": changed}


# ----------------------------------------------------------------------- review


#: A restricted caller sees a proposal only when it is provably that caller's own: the evidence is
#: a non-empty array, and EVERY evidence item names a session whose turns were ALL ingested by
#: that surface (episodes.surface_id, schema 057). An item with no session id, a session with no
#: stored turn, or a session holding even one turn from anywhere else (another device, the root
#: token, a pre-057 upload) proves nothing, so a rule mined from personal or mixed sessions never
#: reaches a restricted host. The nested CASE keeps a malformed (non-array) evidence value from
#: erroring the whole statement. Literal SQL; only %(own)s is bound.
_OWN_PROPOSAL = (
    " AND CASE WHEN jsonb_typeof(evidence) <> 'array' THEN false"
    " WHEN jsonb_array_length(evidence) = 0 THEN false"
    " ELSE NOT EXISTS ("
    "SELECT 1 FROM jsonb_array_elements(evidence) e"
    " WHERE NOT EXISTS (SELECT 1 FROM episodes ep WHERE ep.session_id = e->>'session_id')"
    " OR EXISTS (SELECT 1 FROM episodes ep WHERE ep.session_id = e->>'session_id'"
    " AND ep.surface_id IS DISTINCT FROM %(own)s)) END"
)


def _own_ready(conn, db_url: str, own_surface: str | None) -> bool:
    """False when a restricted (``own_surface``) read must serve nothing because schema 057's
    provenance column is not there yet: without it no proposal can be proven anyone's own."""
    return own_surface is None or own_surface_columns_ready(conn, db_url)


def _proposals_list(db_url: str, own_surface: str | None = None) -> dict:
    """The rule edits awaiting action: 'proposed' (the dream gate graduated them past observe)
    awaiting a decision, and 'accepted' (approved, on the dashboard or by an accept whose disk
    write failed) awaiting the write that marks them applied.

    ``own_surface`` limits the list to that surface's own proposals (_OWN_PROPOSAL), the
    restricted caller's view. None is the unfiltered (full-trust / dashboard) view."""
    sql = (
        "SELECT id, kind, file_key, scope, diff, summary, evidence, status "
        "FROM config_lane.config_proposals WHERE status IN ('proposed', 'accepted')"
    )
    if own_surface:
        sql += _OWN_PROPOSAL
    sql += " ORDER BY updated_at DESC"
    with pg_pool.connection(db_url, autocommit=False, row_factory=dict_row) as conn:
        if not _own_ready(conn, db_url, own_surface):
            return {"proposals": []}
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
                "status": r["status"],
                "rule": r["diff"] or r["summary"],
                "summary": r["summary"],
                "sessions": len({e.get("session_id") for e in ev if e.get("session_id")}),
                "evidence_count": len(ev),
            }
        )
    return {"proposals": out}


def _proposal_owned(db_url: str, cid: int, own_surface: str) -> bool:
    """True when proposal ``cid`` is ``own_surface``'s own (see _OWN_PROPOSAL)."""
    with pg_pool.connection(db_url, autocommit=False) as conn:
        if not _own_ready(conn, db_url, own_surface):
            return False
        row = conn.execute(
            "SELECT 1 FROM config_lane.config_proposals WHERE id = %(id)s" + _OWN_PROPOSAL,  # nosec B608 — _OWN_PROPOSAL is a literal predicate; the values are bound
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
    with pg_pool.connection(db_url, autocommit=False, row_factory=dict_row) as conn:
        if not _own_ready(conn, db_url, own_surface):
            return {"found": False}
        r = conn.execute(sql, {"id": cid, "own": own_surface}).fetchone()
    if not r:
        return {"found": False}
    d = dict(r)
    d["found"] = True
    return d


# The statuses accept and reject act on. 'accepted' stays in: accepting it again is how the CLI
# finishes a dashboard approval.
_DECIDABLE = ("observe", "proposed", "accepted")


def _proposal_act(
    db_url: str,
    cid: int,
    action: str,
    reason: str | None,
    scope: str | None,
    expected: tuple[str, ...] | None = None,
) -> dict:
    """Walk a proposal through the gate. accept records the human yes (and can override the blast
    radius via scope); apply marks it written-to-disk (the client did the actual write); reject drops
    it. The two-step accept->apply mirrors skills' accept->promote: 'accepted' = approved, 'applied' =
    on disk, so a failed disk write leaves it recoverable at 'accepted'.

    A decision is final: an applied or rejected proposal refuses accept and reject (a late click
    must not un-reject a rule or restart its cooling window). ``expected`` narrows the statuses the
    action acts on (the dashboard decides only from 'proposed'). The status check is part of the
    UPDATE, so two decisions landing together cannot both win: the loser gets a 'conflict' result
    and the row is untouched."""
    if action not in ("accept", "reject", "apply"):
        return {"status": "error", "detail": f"unknown action {action!r}"}
    allowed = expected or (("accepted",) if action == "apply" else _DECIDABLE)
    with pg_pool.connection(db_url, autocommit=False, row_factory=dict_row) as conn:
        if action == "accept":
            new_scope = (
                scope if scope in ("local", "general") else None
            )  # None keeps the stored one
            row = conn.execute(
                "UPDATE config_lane.config_proposals SET status='accepted', "
                "scope=COALESCE(%s, scope), updated_at=now() WHERE id=%s AND status = ANY(%s) "
                "RETURNING kind, file_key, scope, diff, summary",
                (new_scope, cid, list(allowed)),
            ).fetchone()
            result = row and {
                "status": "accepted",
                "id": cid,
                "kind": row["kind"],
                "file_key": row["file_key"],
                "scope": row["scope"],
                "rule": row["diff"] or row["summary"],
                "summary": row["summary"],
            }
        elif action == "reject":
            row = conn.execute(
                "UPDATE config_lane.config_proposals SET status='rejected', reject_reason=%s, "
                "updated_at=now() WHERE id=%s AND status = ANY(%s) RETURNING id",
                (reason or "user_rejected", cid, list(allowed)),
            ).fetchone()
            result = row and {"status": "rejected", "id": cid, "reason": reason or "user_rejected"}
        else:
            row = conn.execute(
                "UPDATE config_lane.config_proposals SET status='applied', updated_at=now() "
                "WHERE id=%s AND status = ANY(%s) RETURNING id",
                (cid, list(allowed)),
            ).fetchone()
            result = row and {"status": "applied", "id": cid}
        if result is None:  # gone, or in a status this action does not act on
            current = conn.execute(
                "SELECT status FROM config_lane.config_proposals WHERE id=%s", (cid,)
            ).fetchone()
            if current is None:
                return {"found": False}
            return status_conflict(cid, current["status"], allowed)
    return result


def _target_surface(st: SurfaceTrust, body_surface: str | None) -> str | None:
    """The surface a mirror read/write acts on, or None for "nothing".

    Full trust keeps its earlier behaviour (the body names the surface). A restricted caller is
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
            return _publish_config(db_url, target, b.get("files") or [b])

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
