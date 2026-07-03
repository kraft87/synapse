"""Timeline event ingest route — the DB seam that keeps timeline feeders DSN-free.

Feeders run where the source lives (the plugin's git feeder reads local checkouts on
each surface) and can't hold a database DSN or a Voyage key, so they POST plain event
rows here; the SERVER embeds and upserts. Mirrors /ingest + the skill/config sync
routes: machine-token auth (custom routes bypass FastMCP's auth middleware by design),
PG work in a threadpool, fail-soft JSON.

Route (POST):
  /timeline/events  {"events": [{t_valid, fact, source, source_ref, project?, salience?}]}
                    -> {"status": "ok", "inserted": N, "skipped": M}

Idempotent: UNIQUE(source, source_ref) + ON CONFLICT DO NOTHING, and already-present
events are filtered BEFORE embedding, so a feeder can re-push its full history for free
(the git feeder's model: one event per SHA, re-runs skip).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import psycopg
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse

from ingestion.embedding import create_embedder, embed_dims, embed_provider

logger = logging.getLogger(__name__)

_MAX_BATCH = 1000  # bounds per-call embed cost


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def _embed_text(project: str | None, fact: str) -> str:
    # Metadata-enriched embed string — short naked facts under-fill a 2048-dim space,
    # so give the model the project for grip (2026-07-01 design review).
    return f"Project: {project or '-'} | {fact}"


def _ingest_events(
    db_url: str, voyage_api_key: str, cleaned: list[dict[str, Any]]
) -> tuple[int, int]:
    """Embed + upsert validated events. Returns (inserted, skipped).

    Already-present (source, source_ref) pairs are filtered BEFORE embedding so
    re-pushes cost nothing. Without a Voyage key, rows land with NULL embedding
    (keyless dev/test; a later re-embed pass can fill them) rather than failing.
    """
    conn = psycopg.connect(db_url, autocommit=True)
    try:
        # Composite-tuple ANY isn't supported by psycopg; two plain ANYs over-fetch
        # (cross-product) but the result is only a membership set, so that's harmless.
        rows = conn.execute(
            "SELECT source, source_ref FROM timeline_events "
            "WHERE source = ANY(%s) AND source_ref = ANY(%s)",
            ([e["source"] for e in cleaned], [e["source_ref"] for e in cleaned]),
        ).fetchall()
        existing = {(r[0], r[1]) for r in rows}
        fresh = [e for e in cleaned if (e["source"], e["source_ref"]) not in existing]
        if not fresh:
            return 0, len(cleaned)

        vecs: list[list[float] | None]
        embed_model: str | None = None
        if embed_provider() == "voyage" and not voyage_api_key:
            # Keyless dev/test on the default (Voyage) backend: rows land with
            # NULL embedding; a later re-embed pass can fill them.
            vecs = [None] * len(fresh)
        else:
            emb = create_embedder(voyage_api_key=voyage_api_key, db_url=db_url)
            embed_model = emb.model_name
            vecs = list(
                emb.embed([_embed_text(e["project"], e["fact"]) for e in fresh], task="document")
            )

        inserted = 0
        for e, vec in zip(fresh, vecs, strict=True):
            inserted += conn.execute(
                "INSERT INTO timeline_events "
                "(t_valid, fact, source, source_ref, project, salience, embedding, embed_model, "
                " event_type) "
                f"VALUES (%s,%s,%s,%s,%s,%s,%s::vector({embed_dims()}),%s,%s) "
                "ON CONFLICT (source, source_ref) DO NOTHING",
                (
                    e["t_valid"],
                    e["fact"],
                    e["source"],
                    e["source_ref"],
                    e["project"],
                    e["salience"],
                    _vec_literal(vec) if vec is not None else None,
                    embed_model if vec is not None else None,
                    e.get("event_type"),
                ),
            ).rowcount
        return inserted, len(cleaned) - inserted
    finally:
        conn.close()


def _recent_events(
    db_url: str, days: int, min_salience: int, limit: int, project: str | None
) -> list[dict[str, Any]]:
    """Pure time-window read (no embeddings): the session-start milestones feed."""
    conn = psycopg.connect(db_url, autocommit=True)
    try:
        q = (
            "SELECT left(t_valid::text, 10) AS date, fact, project, salience, event_type "
            "FROM timeline_events "
            "WHERE t_valid > now() - make_interval(days => %s) AND salience >= %s "
        )
        params: list[Any] = [days, min_salience]
        if project:
            q += "AND project = %s "
            params.append(project)
        q += "ORDER BY t_valid DESC LIMIT %s"
        params.append(limit)
        rows = conn.execute(q, params).fetchall()
        return [
            {"date": r[0], "fact": r[1], "project": r[2], "salience": r[3], "event_type": r[4]}
            for r in rows
        ]
    finally:
        conn.close()


def register(
    mcp: Any, db_url: str, authorized: Callable[[Request], bool], voyage_api_key: str
) -> None:
    if not db_url:
        logger.info("timeline routes disabled (no DB_URL)")
        return

    @mcp.custom_route("/timeline/recent", methods=["POST"])  # type: ignore[misc]
    async def timeline_recent(request: Request) -> JSONResponse:
        """Recent high-salience events for the plugin's session-start milestones block.
        Body: {days?=7, min_salience?=2, limit?=5, project?}. Time-scoped and tiny by
        design — this is a bounded factual block, not query-blind recall injection."""
        if not authorized(request):
            return JSONResponse({"status": "error", "detail": "unauthorized"}, status_code=401)
        try:
            body = await request.json()
        except Exception:
            body = {}
        days = min(int(body.get("days") or 7), 90)
        min_sal = int(body.get("min_salience") or 2)
        limit = min(int(body.get("limit") or 5), 20)
        try:
            items = await run_in_threadpool(
                _recent_events, db_url, days, min_sal, limit, body.get("project")
            )
        except Exception as e:
            logger.warning("timeline recent failed: %s", e)
            return JSONResponse({"status": "error", "detail": str(e)[:200]}, status_code=500)
        return JSONResponse({"status": "ok", "items": items})

    @mcp.custom_route("/timeline/events", methods=["POST"])  # type: ignore[misc]
    async def timeline_events(request: Request) -> JSONResponse:
        if not authorized(request):
            return JSONResponse({"status": "error", "detail": "unauthorized"}, status_code=401)
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"status": "error", "detail": "invalid JSON body"}, status_code=400)
        events = body.get("events")
        if not isinstance(events, list) or not events:
            return JSONResponse(
                {"status": "error", "detail": "body must contain a non-empty 'events' list"},
                status_code=400,
            )
        if len(events) > _MAX_BATCH:
            return JSONResponse(
                {"status": "error", "detail": f"max {_MAX_BATCH} events per call"},
                status_code=400,
            )
        cleaned: list[dict[str, Any]] = []
        for i, e in enumerate(events):
            if not isinstance(e, dict):
                return JSONResponse(
                    {"status": "error", "detail": f"events[{i}] not an object"}, status_code=400
                )
            missing = [k for k in ("t_valid", "fact", "source", "source_ref") if not e.get(k)]
            if missing:
                return JSONResponse(
                    {"status": "error", "detail": f"events[{i}] missing {missing}"},
                    status_code=400,
                )
            sal = e.get("salience", 1)
            et = e.get("event_type")
            cleaned.append(
                {
                    "t_valid": str(e["t_valid"]),
                    "fact": str(e["fact"]).strip(),
                    "source": str(e["source"]),
                    "source_ref": str(e["source_ref"]),
                    "project": e.get("project"),
                    "salience": sal if isinstance(sal, int) and 0 <= sal <= 2 else 1,
                    "event_type": et
                    if et in ("decision", "action", "finding", "milestone")
                    else None,
                }
            )

        try:
            inserted, skipped = await run_in_threadpool(
                _ingest_events, db_url, voyage_api_key, cleaned
            )
        except psycopg.errors.UndefinedTable:
            return JSONResponse(
                {"status": "error", "detail": "timeline_events missing (apply schema/033)"},
                status_code=503,
            )
        except Exception as e:
            logger.warning("timeline ingest failed: %s", e)
            return JSONResponse({"status": "error", "detail": str(e)[:200]}, status_code=500)
        return JSONResponse({"status": "ok", "inserted": inserted, "skipped": skipped})
