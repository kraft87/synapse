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

from ingestion import pg_pool
from ingestion.embedding import create_embedder, embed_dims, embed_provider
from ingestion.scope import coerce_group
from ingestion.surfaces import (
    UNKNOWN_SURFACE,
    SurfaceTrust,
    episode_scope_sql,
    own_surface_if_ready,
)
from mcp_server.http_helpers import err, unauthorized

logger = logging.getLogger(__name__)

_MAX_BATCH = 1000  # bounds per-call embed cost


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def _embed_text(project: str | None, fact: str) -> str:
    # Metadata-enriched embed string — short naked facts under-fill a 2048-dim space,
    # so give the model the project for grip (2026-07-01 design review).
    return f"Project: {project or '-'} | {fact}"


def _event_domain(e: dict[str, Any]) -> str | None:
    """The stored ``domain`` for one pushed event (schema 038).

    Git-sourced events are technical by construction; other pushers may label
    explicitly; anything else stays NULL, which fails open at read. With the
    personal scope off (SYNAPSE_PERSONAL_SCOPE=0) only one domain exists, so a
    pushed "personal" label is stored as technical.
    """
    domain = e.get("domain")
    if domain not in ("personal", "technical"):
        domain = "technical" if str(e["source"]).startswith("git:") else None
    return coerce_group(domain)


def _ingest_events(
    db_url: str, voyage_api_key: str, cleaned: list[dict[str, Any]], stamp: str | None = None
) -> tuple[int, int]:
    """Embed + upsert validated events. Returns (inserted, skipped).

    Already-present (source, source_ref) pairs are filtered BEFORE embedding so
    re-pushes cost nothing. Without a Voyage key, rows land with NULL embedding
    (keyless dev/test; a later re-embed pass can fill them) rather than failing.

    ``stamp`` is the pushing device's provenance (schema 057, ``stamp_surface_id``), so
    a restricted device's own feeder events are served back to it like its chat-derived
    ones. None (the root token, an unknown caller) and a database without 057 write the
    pre-057 row. A re-push of an existing (source, source_ref) keeps the first stamp.
    """
    with pg_pool.connection(db_url) as conn:
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

    # Embedding is a network call and may itself read the database (the embedder's
    # meta check), so no connection is held across it.
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
    with pg_pool.connection(db_url) as conn:
        own = own_surface_if_ready(conn, db_url, stamp)  # None before schema 057
        cols = "event_type, domain, surface_id" if own else "event_type, domain"
        marks = "%s,%s,%s" if own else "%s,%s"
        for e, vec in zip(fresh, vecs, strict=True):
            params: list[Any] = [
                e["t_valid"],
                e["fact"],
                e["source"],
                e["source_ref"],
                e["project"],
                e["salience"],
                _vec_literal(vec) if vec is not None else None,
                embed_model if vec is not None else None,
                e.get("event_type"),
                _event_domain(e),
            ]
            inserted += conn.execute(
                "INSERT INTO timeline_events "  # nosec B608 — fixed column lists and a validated int
                "(t_valid, fact, source, source_ref, project, salience, embedding, embed_model, "
                f" {cols}) "
                f"VALUES (%s,%s,%s,%s,%s,%s,%s::vector({embed_dims()}),%s,{marks}) "
                "ON CONFLICT (source, source_ref) DO NOTHING",
                [*params, own] if own else params,
            ).rowcount
    return inserted, len(cleaned) - inserted


# Diversity cap on the digest: max N events per (project, day). Recency-only
# selection lets one intense work session monopolize every slot — observed
# 2026-08-15, when a single day's PR arc filled 7 of the board's 10 lines and
# pushed the rest of the week out entirely. Within a day, higher salience wins
# a slot before recency. The cap trades arc completeness for week coverage.
_PER_PROJECT_DAY = 3


def _recent_events(
    db_url: str,
    days: int,
    min_salience: int,
    limit: int,
    project: str | None,
    allowed_projects: list[str] | None = None,
    own_surface: str | None = None,
) -> list[dict[str, Any]]:
    """Pure time-window read (no embeddings): the session-start milestones feed.

    ``allowed_projects`` (schema 053) is the board's restricted-surface filter. There is
    deliberately no ``audience`` column on timeline_events — it already carries `domain`
    (038), which fails OPEN, and a second overlapping label with the opposite fail
    semantics would drift. The project allowlist is the filter instead, and
    ``project = ANY(...)`` excludes NULL-project events by construction: an unlabeled
    event has no provenance to clear it, so it stays off restricted boards.

    ``own_surface`` (schema 057) also admits events whose source turn the calling
    surface ingested itself; the gate copies that stamp onto the event at write time.
    """
    with pg_pool.connection(db_url) as conn:
        q = (
            "SELECT id, date, fact, project, salience, event_type FROM ("
            "  SELECT id, left(t_valid::text, 10) AS date, fact, project, salience, "
            "         event_type, t_valid, row_number() OVER ("
            "           PARTITION BY project, t_valid::date "
            "           ORDER BY salience DESC, t_valid DESC) AS day_rank "
            "  FROM timeline_events "
            "  WHERE t_valid > now() - make_interval(days => %s) AND salience >= %s "
        )
        params: list[Any] = [days, min_salience]
        if project:
            q += "AND project = %s "
            params.append(project)
        own = own_surface_if_ready(conn, db_url, own_surface)  # None before schema 057
        scope = episode_scope_sql(allowed_projects, own)
        if scope is not None:
            q += f"AND {scope[0]} "
            params.extend(scope[1])
        q += ") ranked WHERE day_rank <= %s ORDER BY t_valid DESC LIMIT %s"
        params.extend([_PER_PROJECT_DAY, limit])
        rows = conn.execute(q, params).fetchall()
        return [
            {
                "id": r[0],
                "date": r[1],
                "fact": r[2],
                "project": r[3],
                "salience": r[4],
                "event_type": r[5],
            }
            for r in rows
        ]


def register(
    mcp: Any,
    db_url: str,
    authorized: Callable[[Request], bool],
    voyage_api_key: str,
    resolve_trust: Callable[[Request], SurfaceTrust] | None = None,
) -> None:
    """``resolve_trust`` maps a request to its bearer-resolved verdict (schema 054). Its
    provenance stamp goes on every event the caller pushes; when it is not supplied
    nothing is stamped, which serves the events by project alone, as before 057."""
    if not db_url:
        logger.info("timeline routes disabled (no DB_URL)")
        return

    @mcp.custom_route("/timeline/events", methods=["POST"])  # type: ignore[misc]
    async def timeline_events(request: Request) -> JSONResponse:
        if not authorized(request):
            return unauthorized()
        try:
            body = await request.json()
        except Exception:
            return err("invalid JSON body", 400)
        events = body.get("events")
        if not isinstance(events, list) or not events:
            return err("body must contain a non-empty 'events' list", 400)
        if len(events) > _MAX_BATCH:
            return err(f"max {_MAX_BATCH} events per call", 400)
        cleaned: list[dict[str, Any]] = []
        for i, e in enumerate(events):
            if not isinstance(e, dict):
                return err(f"events[{i}] not an object", 400)
            missing = [k for k in ("t_valid", "fact", "source", "source_ref") if not e.get(k)]
            if missing:
                return err(f"events[{i}] missing {missing}", 400)
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
            st = (
                await run_in_threadpool(resolve_trust, request)
                if resolve_trust
                else UNKNOWN_SURFACE
            )
            inserted, skipped = await run_in_threadpool(
                _ingest_events, db_url, voyage_api_key, cleaned, st.stamp_surface_id
            )
        except psycopg.errors.UndefinedTable:
            return err("timeline_events missing (apply schema/033)", 503)
        except Exception as e:
            logger.warning("timeline ingest failed: %s", e)
            return err(str(e)[:200], 500)
        return JSONResponse({"status": "ok", "inserted": inserted, "skipped": skipped})
