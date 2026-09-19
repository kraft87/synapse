"""RecallSourcesMixin operations."""

from __future__ import annotations

import json
import logging
from typing import Any

from psycopg.rows import tuple_row

from mcp_server.kg_pg import _vec_literal, search_kg_postgres
from mcp_server.recall_ranking import merge_rrf as _merge_rrf
from mcp_server.recall_settings import _SUP_LIMIT
from mcp_server.recall_warnings import error_brief as _err_brief
from mcp_server.recall_warnings import warn as _warn

logger = logging.getLogger(__name__)


class RecallSourcesMixin:
    def _search_bm25_web(self, query: str, limit: int) -> list[dict[str, Any]]:
        # Same no-alphanumeric-tokens short-circuit as _bm25_table.
        if not any(c.isalnum() for c in query):
            return []
        pg = self._ensure_pg()
        try:
            rows = pg.execute(
                """
                SELECT c.id, c.content, c.context_prefix, c.web_artifact_id,
                       c.content_ts AS created_at,
                       a.url, a.title, a.tool_name,
                       paradedb.score(c.id) AS bm25_score
                FROM web_chunks c
                JOIN web_artifacts a ON a.id = c.web_artifact_id
                WHERE c.id @@@ paradedb.match('content', %s)
                ORDER BY bm25_score DESC LIMIT %s
                """,
                (query, limit),
            ).fetchall()
            return [{**dict(r), "doc_type": "web", "id": f"w:{r['id']}"} for r in rows]
        except Exception as e:
            logger.warning("BM25 web search failed: %s", e)
            _warn(f"BM25 web search failed ({_err_brief(e)}): web bucket is degraded.")
            return []

    def _search_vector_web(self, query_emb: list[float], limit: int) -> list[dict[str, Any]]:
        pg = self._ensure_pg()
        emb_literal = _vec_literal(query_emb)
        try:
            rows = pg.execute(
                """
                SELECT c.id, c.content, c.context_prefix, c.web_artifact_id,
                       c.content_ts AS created_at,
                       a.url, a.title, a.tool_name,
                       (c.embedding <=> %s::vector) AS vec_distance
                FROM web_chunks c
                JOIN web_artifacts a ON a.id = c.web_artifact_id
                WHERE c.is_embedded = TRUE
                ORDER BY vec_distance ASC LIMIT %s
                """,
                (emb_literal, limit),
            ).fetchall()
            return [{**dict(r), "doc_type": "web", "id": f"w:{r['id']}"} for r in rows]
        except Exception as e:
            logger.warning("Vector web search failed: %s", e)
            _warn(f"vector web search failed ({_err_brief(e)}): web bucket is degraded.")
            return []

    @staticmethod
    def _dedupe_by_artifact(chunks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        """Keep only the highest-ranked chunk per parent web_artifact_id.

        Input is assumed already RRF-ordered. First occurrence of an
        artifact_id wins; subsequent chunks from the same page drop.
        """
        seen: set[int] = set()
        out: list[dict[str, Any]] = []
        for c in chunks:
            aid = c.get("web_artifact_id")
            if aid in seen:
                continue
            if aid is not None:
                seen.add(aid)
            out.append(c)
            if len(out) >= limit:
                break
        return out

    def _search_web_reranked(self, query: str, query_emb: list[float]) -> list[dict[str, Any]]:
        """Web bucket (2026-07-24): fuse BM25 + vector via RRF, cross-encoder rerank the pool,
        DROP chunks below _WEB_FLOOR (off-topic collisions), dedupe by parent page, cap to
        _WEB_LIMIT. May return [] when nothing clears the floor — that self-gates web on intent.

        Replaces the old vector-only leg: BM25 was written (_search_bm25_web) but never wired,
        and the raw bi-encoder ranking served ~2.6% precision (feedback). The rerank is the active
        ingredient (it reads content topicality, not surface tokens); fusion widens the candidate
        pool the reranker sees. Degrades safe: on reranker outage/disable (top score 0.0) it serves
        the fused RRF order deduped rather than blanking the bucket or hard-failing."""
        settings = self._settings()
        bm25 = self._search_bm25_web(query, settings._WEB_FETCH)
        vec = self._search_vector_web(query_emb, settings._WEB_FETCH)
        fused = _merge_rrf(bm25, vec, id_key="id")[: settings._WEB_RERANK_POOL]
        if not fused:
            return []
        scored = self._rerank_pool_scored(query, fused)
        if settings._WEB_FLOOR > 0 and scored and scored[0][1] > 0.0:
            ranked = [fused[i] for i, s in scored if s >= settings._WEB_FLOOR]
        else:  # floor disabled, or reranker degraded to RRF order (no score signal)
            ranked = [fused[i] for i, _ in scored] if scored else fused
        return self._dedupe_by_artifact(ranked, settings._WEB_LIMIT)

    def _search_notes(
        self,
        query: str,
        query_emb: list[float],
        project: str | None,
        audience: str | None = None,
    ) -> list[dict[str, Any]]:
        """Notes bucket: hook-KNN top-_NOTES_FETCH live notes (global types + this
        project's), floored by the cross-encoder on "hook — body" text, top
        _NOTES_LIMIT served. May return [] when nothing clears the floor — notes
        self-gate on relevance like web. Fail-soft: any error serves no bucket.
        Uses a short-lived Database like the other notes-store paths.

        ``audience`` is the restricted surface's tier filter (schema 053) — 'work-safe'
        for a restricted caller, None for full trust. Fail-soft here means an EMPTY
        bucket, which is also the fail-closed answer; there is no path where the filter
        is dropped and the leg still serves."""
        settings = self._settings()
        from ingestion.db import Database

        try:
            db = Database(self._db_url)
            try:
                rows = db.search_live_notes(
                    settings._KG_OWNER,
                    project,
                    query_emb,
                    limit=settings._NOTES_FETCH,
                    audience=audience,
                )
            finally:
                db.close()
        except Exception as e:
            logger.warning("notes leg fetch failed: %s", e)
            _warn(f"notes leg failed ({_err_brief(e)}): no notes served.")
            return []
        if not rows:
            return []
        if settings._NOTES_FLOOR > 0:
            docs = [
                {"fact": f"{r['hook']} — {r['body']}"[: settings._RERANK_DOC_CAP], "_r": r}
                for r in rows
            ]
            rows = [d["_r"] for d in self._floor_by_rerank(query, docs, settings._NOTES_FLOOR)]
        served = []
        for r in rows[: settings._NOTES_LIMIT]:
            body = r["body"]
            if len(body) > settings._NOTES_BODY_CAP:
                body = body[: settings._NOTES_BODY_CAP] + "…"
            item: dict[str, Any] = {"id": f"n:{r['id']}", "hook": r["hook"], "note": body}
            if r.get("project"):
                item["project"] = r["project"]
            served.append(item)
        return served

    def _search_kg(
        self,
        query: str,
        query_emb: list[float],
        group_id: str,
        session_focus: list[str],
        fact_limit: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """KG facts leg over kg_entities / kg_relationships (task #67).

        Runs inside one transaction so the planner GUCs are SET LOCAL — scoped
        to this query, not the thread-local connection that other legs may
        reuse later (enable_seqscan=off session-wide would be a foot-gun for
        any future unindexed query). hnsw.ef_search is already set
        session-wide by _ensure_pg. A failure degrades to an empty facts
        bucket rather than failing the whole recall.
        """
        settings = self._settings()
        try:
            conn = self._ensure_pg()
            with conn.transaction():
                # kg_pg indexes rows positionally; recall's thread-local conns
                # default to dict_row, so override at the cursor.
                cur = conn.cursor(row_factory=tuple_row)
                cur.execute("SET LOCAL enable_seqscan = off")
                cur.execute("SET LOCAL max_parallel_workers_per_gather = 0")
                return search_kg_postgres(
                    cur, query, query_emb, settings._KG_OWNER, group_id, session_focus, fact_limit
                )
        except Exception as e:
            logger.warning("KG search failed: %s", e)
            _warn(f"KG facts leg failed ({_err_brief(e)}): no facts served.")
            return [], []

    def _fetch_superseded_pairs_pg(
        self,
        group_id: str,
        active_edge_uuids: list[str],
        cap: int,
    ) -> list[dict[str, Any]]:
        """Superseded-fact pairs for served edges (Postgres port of the old
        FalkorDB _fetch_history_pairs).

        DISTINCT ON picks the most recently invalidated predecessor per active
        edge in SQL (the FalkorDB path does this dedup in Python).
        """
        settings = self._settings()
        if not active_edge_uuids or cap <= 0:
            return []
        try:
            conn = self._ensure_pg()
            rows = conn.execute(
                """
                SELECT DISTINCT ON (a.uuid) a.uuid AS uid, a.fact AS now_fact,
                       o.uuid AS old_uid, o.fact AS old_fact
                FROM kg_relationships a
                JOIN kg_relationships o
                  ON o.src_uuid = a.src_uuid AND o.tgt_uuid = a.tgt_uuid
                 AND o.owner_id = a.owner_id AND o.group_id = a.group_id
                WHERE a.owner_id = %s AND a.group_id = %s
                  AND a.uuid = ANY(%s) AND a.t_invalid IS NULL
                  AND o.t_invalid IS NOT NULL AND o.uuid <> a.uuid
                  AND o.fact IS NOT NULL AND a.fact IS NOT NULL
                ORDER BY a.uuid, o.t_invalid DESC
                """,
                (settings._KG_OWNER, group_id, active_edge_uuids),
            ).fetchall()
        except Exception as e:
            logger.debug("PG superseded-pairs query failed: %s", e)
            return []
        by_uid = {r["uid"]: r for r in rows}
        out: list[dict[str, Any]] = []
        for uid in active_edge_uuids:
            r = by_uid.get(uid)
            if r is None:
                continue
            # id is the invalidated (old) edge's uuid, "f:<uuid>" like a regular fact.
            # Collision-free: the facts bucket serves the CURRENT edge (a.uuid), never
            # the superseded predecessor (o.uuid, enforced <> a.uuid above). Lets the
            # stale pair be cited in recall_feedback.
            out.append(
                {
                    "id": f"f:{r['old_uid']}",
                    "fact": r["old_fact"],
                    "superseded_by": r["now_fact"],
                }
            )
            if len(out) >= cap:
                break
        return out

    def _episode_supersessions(
        self, episode_ids: list[int], group_id: str, cap: int = 6
    ) -> dict[int, list[str]]:
        """Map served episode ids -> the CURRENT facts that superseded a claim each made.

        A retired edge P citing the episode (episodes @> [id]) links to its superseding live edge N
        via P.invalidated_by (schema 028 + backfill); N.fact is the "now" value. Hits the partial GIN
        (schema 029, WHERE invalidated_by IS NOT NULL) so the per-recall lookup is cheap. Fail-open —
        a lookup error just yields no annotations, never breaks recall."""
        settings = self._settings()
        if not episode_ids:
            return {}
        ors = " OR ".join(["p.episodes @> %s::jsonb"] * len(episode_ids))
        params: list[Any] = [json.dumps([i]) for i in episode_ids] + [
            settings._KG_OWNER,
            group_id,
            cap,
        ]
        try:
            conn = self._ensure_pg()
            rows = conn.execute(
                "SELECT p.episodes, n.fact FROM kg_relationships p "
                "JOIN kg_relationships n ON n.uuid = p.invalidated_by "
                f"WHERE p.invalidated_by IS NOT NULL AND ({ors}) "
                "  AND p.owner_id = %s AND p.group_id = %s LIMIT %s",
                params,
            ).fetchall()
        except Exception as e:
            logger.warning("episode supersession lookup failed: %s", e)
            return {}
        idset = set(episode_ids)
        out: dict[int, list[str]] = {}
        for r in rows:
            fact = r.get("fact")
            if not fact:
                continue
            for eid in r.get("episodes") or []:
                if eid in idset:
                    out.setdefault(int(eid), []).append(fact)
        return out

    def _surface_supersessions(
        self,
        query_emb: list[float] | None,
        group_id: str,
        served_uuids: set[str],
        cap: int = _SUP_LIMIT,
    ) -> list[dict[str, Any]]:
        """A query that matches a now-INVALID fact should still return the CURRENT answer.

        Finds superseded edges near the query that carry a precise successor link (invalidated_by,
        schema 028), resolves the live successor, and returns those not already served — deduped by
        uuid and distance-gated (_SUP_MAX_DIST) so only on-topic superseded facts pull their
        correction in. Go-forward coverage only (no link => skipped); never returns the stale fact
        itself. Shape matches the KG fact leg ({fact, _uuid, _date}) so it flows through fact serving.
        Fail-open — a lookup error just yields no extras."""
        settings = self._settings()
        if query_emb is None:
            return []
        vec = _vec_literal(query_emb)
        try:
            conn = self._ensure_pg()
            rows = conn.execute(
                "SELECT n.uuid, n.fact, n.t_valid, "
                f"  (p.fact_embedding::halfvec({settings._EMBED_DIMS}) <=> %s::halfvec({settings._EMBED_DIMS})) AS d "
                "FROM kg_relationships p "
                "JOIN kg_relationships n ON n.uuid = p.invalidated_by AND n.t_invalid IS NULL "
                "WHERE p.t_invalid IS NOT NULL AND p.invalidated_by IS NOT NULL "
                "  AND p.fact_embedding IS NOT NULL AND p.owner_id = %s AND p.group_id = %s "
                f"ORDER BY p.fact_embedding::halfvec({settings._EMBED_DIMS}) <=> %s::halfvec({settings._EMBED_DIMS}) LIMIT %s",
                (vec, settings._KG_OWNER, group_id, vec, settings._SUP_CANDIDATES),
            ).fetchall()
        except Exception as e:
            logger.warning("supersession surface failed: %s", e)
            return []
        out: list[dict[str, Any]] = []
        for r in rows:
            d = r.get("d")
            if d is None or d > settings._SUP_MAX_DIST:
                continue
            u, fact = r.get("uuid"), r.get("fact")
            if u and fact and u not in served_uuids:
                out.append({"fact": fact, "_uuid": u, "_date": r.get("t_valid")})
                served_uuids.add(u)
                if len(out) >= cap:
                    break
        return out
