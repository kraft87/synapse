"""recall_timeline() — serving for the episodic timeline store (timeline_events).

The timeline is the EPISODIC complement to the semantic KG: an append-only log of dated
point-events, never deduplicated (schema 033). Serving follows the 2026-07-01 design
review (opus-Oracle + Gemini):

  - QUERY-SHAPE BRANCH. A topical query ("the login work") fuses BM25 + vector via
    rank-based RRF and then RE-SORTS the survivors by t_valid — the timeline's value is
    ORDER, so results are never presented by relevance score. A pure-time query
    ("what happened last week") is a range scan, salience-ranked, family-collapsed.
  - NO time-decay in the fusion: since/until give temporal control; a decay scorer
    would bury old-but-relevant events on topical queries.
  - READ-TIME FAMILY-COLLAPSE. Per-turn/per-commit granularity fragments dense windows;
    same-project low-salience runs collapse to ONE line carrying count + first/last
    anchors, so any computed duration stays auditable (never a bare "47 days").
    Milestones (salience=2) always survive individually.
  - BM25 out-weighs the vector leg naturally here: episodic facts are identifier-dense
    (PR #193, a SHA, "N=6") where short-text embeddings are weakest.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import psycopg
from psycopg.rows import dict_row

from ingestion.embedding import create_embedder, embed_dims
from ingestion.timeline_gate import extract_idents

logger = logging.getLogger(__name__)

_RRF_K = 60  # same rank-based RRF constant as recall()
_LEG_POOL = 40  # candidates per leg before fusion
_TOPICAL_N = 12  # events returned for a topical query
_FAMILY_MIN = 2  # a same-project salience<2 run of >= this many collapses
_TIME_LIMIT = 20  # default cap for pure-time queries (post-collapse)

_COLS = "id, t_valid, fact, source, source_ref, project, salience, event_type"


def _rrf(*lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank-based Reciprocal Rank Fusion. No recency/feedback multipliers — episodic
    order is carried by t_valid at presentation time, not by the scorer."""
    scores: dict[Any, float] = {}
    items: dict[Any, dict[str, Any]] = {}
    for lst in lists:
        for rank, it in enumerate(lst):
            i = it["id"]
            scores[i] = scores.get(i, 0.0) + 1.0 / (_RRF_K + rank + 1)
            items.setdefault(i, it)
    return sorted(items.values(), key=lambda x: scores[x["id"]], reverse=True)


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def _filters(
    since: str | None,
    until: str | None,
    project: str | None,
    min_salience: int,
    group_id: str | None = None,
) -> tuple[list[str], list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if since:
        clauses.append("t_valid >= %s")
        params.append(since)
    if until:
        clauses.append("t_valid < %s")
        params.append(until)
    if project:
        clauses.append("project = %s")
        params.append(project)
    if min_salience:
        clauses.append("salience >= %s")
        params.append(min_salience)
    # Domain scoping (schema 038, issue #17): an explicit personal scope drops
    # technical events — measured 2026-07-05 that cross-domain junk can sit CLOSER
    # in embedding space than the true personal events (0.704 vs 0.748), so no
    # relevance floor can do this job; only the domain label can. Unlabeled (NULL)
    # rows FAIL OPEN. The default/technical scope stays unfiltered: most callers
    # never set group_id, and hiding personal events from them would be a
    # regression, not hygiene. Kill switch SYNAPSE_TIMELINE_GROUP_SCOPE=0.
    if group_id == "personal" and os.environ.get("SYNAPSE_TIMELINE_GROUP_SCOPE", "1") != "0":
        clauses.append("(domain = 'personal' OR domain IS NULL)")
    return clauses, params


def _event(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "_id": r["id"],  # internal — recall()'s served_ids telemetry; stripped at the MCP boundary
        "kind": "event",
        "t_valid": str(r["t_valid"]),
        "project": r["project"],
        "salience": r["salience"],
        "fact": r["fact"],
        "source": r["source"],
        "source_ref": r["source_ref"],
        "event_type": r.get("event_type"),
    }


def _collapse(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Read-time family-collapse for dense windows. Milestones (salience 2) survive
    individually; same-project salience<2 runs become one family line with count +
    first/last anchor events (auditable — never a bare number). Chronological output."""
    out: list[dict[str, Any]] = []
    for r in rows:
        if r["salience"] == 2:
            e = _event(r)
            e["_sort"] = r["t_valid"]
            out.append(e)

    families: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if r["salience"] < 2:
            families.setdefault(r["project"] or "-", []).append(r)
    for proj, grp in families.items():
        grp.sort(key=lambda r: r["t_valid"])
        if len(grp) >= _FAMILY_MIN:
            out.append(
                {
                    "kind": "family",
                    "project": proj,
                    "count": len(grp),
                    "t_start": str(grp[0]["t_valid"]),
                    "t_end": str(grp[-1]["t_valid"]),
                    "first": grp[0]["fact"],
                    "last": grp[-1]["fact"],
                    "_sort": grp[0]["t_valid"],
                }
            )
        else:
            for r in grp:
                e = _event(r)
                e["_sort"] = r["t_valid"]
                out.append(e)
    out.sort(key=lambda x: x.pop("_sort"))
    return out


def _collapse_idents(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Read-time identifier-collapse for the topical view: events sharing a PR-ref/SHA
    are stages of ONE happening (merge -> ship -> deploy of the same PR); keep the
    richest (longest fact) and note the fold. Identifier match, not embedding sim —
    measured sims can't separate restatement (0.63) from related-but-distinct (0.83)."""
    by_ident: dict[str, int] = {}
    out: list[dict[str, Any]] = []
    for it in items:
        idents = extract_idents(it.get("fact") or "")
        hit = next((i for i in idents if i in by_ident), None)
        if hit is None:
            for i in idents:
                by_ident[i] = len(out)
            out.append(it)
            continue
        keep = out[by_ident[hit]]
        richer = it if len(it.get("fact") or "") > len(keep.get("fact") or "") else keep
        richer["folded"] = int(keep.get("folded") or 1) + 1
        out[by_ident[hit]] = richer
        for i in extract_idents(richer.get("fact") or ""):
            by_ident[i] = by_ident[hit]
    return out


class TimelineRecall:
    """Stateful timeline retrieval engine. One instance per MCP server process."""

    def __init__(self, db_url: str, voyage_api_key: str) -> None:
        self._db_url = db_url
        self._voyage_key = voyage_api_key
        self._embedder: Any = None
        self._pg_local = threading.local()

    def _ensure_pg(self) -> Any:
        conn = getattr(self._pg_local, "conn", None)
        if conn is not None and not conn.closed:
            try:
                conn.execute("SELECT 1")
                return conn
            except Exception:
                conn = None
        conn = psycopg.connect(self._db_url, row_factory=dict_row, autocommit=True)
        self._pg_local.conn = conn
        return conn

    def _ensure_embedder(self) -> Any:
        if self._embedder is None:
            self._embedder = create_embedder(voyage_api_key=self._voyage_key, db_url=self._db_url)
        return self._embedder

    # -- legs -------------------------------------------------------------
    def _search_vector(
        self,
        conn: Any,
        query: str,
        filt: list[str],
        params: list[Any],
        query_emb: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        qvec = (
            query_emb
            if query_emb is not None
            else (self._ensure_embedder().embed([query], task="query")[0])
        )
        where = " AND ".join(["embedding IS NOT NULL", *filt])
        rows: list[dict[str, Any]] = conn.execute(
            f"SELECT {_COLS}, embedding <=> %s::vector({embed_dims()}) AS dist "
            f"FROM timeline_events WHERE {where} ORDER BY dist ASC LIMIT %s",
            [_vec_literal(qvec), *params, _LEG_POOL],
        ).fetchall()
        return rows

    def _search_bm25(
        self, conn: Any, query: str, filt: list[str], params: list[Any]
    ) -> list[dict[str, Any]]:
        where = " AND ".join(["id @@@ paradedb.match('fact', %s)", *filt])
        rows: list[dict[str, Any]] = conn.execute(
            f"SELECT {_COLS}, paradedb.score(id) AS bm25 "
            f"FROM timeline_events WHERE {where} ORDER BY bm25 DESC LIMIT %s",
            [query, *params, _LEG_POOL],
        ).fetchall()
        return rows

    # -- public -----------------------------------------------------------
    def recall_timeline(
        self,
        query: str | None = None,
        since: str | None = None,
        until: str | None = None,
        project: str | None = None,
        min_salience: int = 0,
        limit: int = _TIME_LIMIT,
        query_emb: list[float] | None = None,
        group_id: str | None = None,
    ) -> dict[str, Any]:
        try:
            conn = self._ensure_pg()
            filt, params = _filters(since, until, project, min_salience, group_id)
            where = " AND ".join(filt) or "TRUE"
            total = conn.execute(
                f"SELECT count(*) AS n FROM timeline_events WHERE {where}", params
            ).fetchone()["n"]

            if query and query.strip():
                # TOPICAL: hybrid fuse, then chronological presentation.
                bm = self._search_bm25(conn, query, filt, params)
                try:
                    vec = self._search_vector(conn, query, filt, params, query_emb=query_emb)
                except Exception as e:  # embedder down -> BM25-only, don't fail the call
                    logger.warning("timeline vector leg failed: %s", e)
                    vec = []
                fused = _rrf(bm, vec)[:_TOPICAL_N]
                fused.sort(key=lambda r: r["t_valid"])
                items = _collapse_idents([_event(r) for r in fused])
                shape = "topical"
            else:
                # PURE-TIME: range scan, salience-rank via family-collapse.
                rows = conn.execute(
                    f"SELECT {_COLS} FROM timeline_events WHERE {where} ORDER BY t_valid",
                    params,
                ).fetchall()
                items = _collapse(rows)[:limit]
                shape = "time"

            return {
                "query_shape": shape,
                "window": {
                    "since": since,
                    "until": until,
                    "project": project,
                    "min_salience": min_salience,
                },
                "total_in_window": total,
                "returned": len(items),
                "items": items,
            }
        except psycopg.errors.UndefinedTable:
            # Migration 033 not applied on this deployment yet — degrade, don't crash.
            return {
                "error": "timeline_events not present (apply schema/033_timeline.sql)",
                "items": [],
            }
        except Exception as e:
            logger.warning("recall_timeline failed: %s", e)
            return {"error": str(e)[:200], "items": []}
