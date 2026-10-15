"""RecallSourcesMixin operations."""

from __future__ import annotations

import json
import logging
from typing import Any

from psycopg.rows import tuple_row

from ingestion.db_notes import search_live_notes
from ingestion.schema_check import ColumnProbe
from mcp_server.kg_pg import _vec_literal, scope_predicate, search_kg_postgres
from mcp_server.recall_engine import RecallEngine
from mcp_server.recall_presentation import OVERLAY_MIN_SUPPORT, claim_support
from mcp_server.recall_warnings import error_brief as _err_brief
from mcp_server.recall_warnings import warn as _warn

logger = logging.getLogger(__name__)

# Restricted KG serving needs kg_relationships.source_projects (schema 056), and the code
# ships before that migration is applied by hand. Until it exists every restricted KG path
# serves nothing, which is exactly the pre-056 posture.
KG_SCOPE_PROBE = ColumnProbe(
    "SELECT 1 FROM pg_attribute "
    "WHERE attrelid = to_regclass('kg_relationships') "
    "  AND attname = 'source_projects' AND NOT attisdropped",
    "kg_relationships.source_projects (schema 056, restricted KG serving)",
)


def _self_only_sql(alias: str) -> str:
    """SQL predicate: every source episode of fact ``alias`` belongs to the session bound at %s.

    False for a fact with no recorded source (web artifacts, a malformed ``episodes`` value),
    so self-exclusion never drops what it cannot attribute."""
    return (
        f"(jsonb_typeof({alias}.episodes) = 'array' AND jsonb_array_length({alias}.episodes) > 0 "
        f"AND NOT EXISTS (SELECT 1 FROM jsonb_array_elements_text({alias}.episodes) AS src(eid) "
        "WHERE NOT EXISTS (SELECT 1 FROM episodes e "
        "WHERE e.id = CASE WHEN src.eid ~ '^[0-9]+$' THEN src.eid::bigint END "
        "AND e.session_id = %s)))"
    )


def _link_close_sql(old: str, new: str, dims: int) -> str:
    """SQL predicate: successor ``new`` is about the same thing as the retired ``old``.

    Distance bound at %s (sup_link_max_dist). A missing embedding trusts the link."""
    return (
        f"COALESCE({old}.fact_embedding::halfvec({dims}) <=> "
        f"{new}.fact_embedding::halfvec({dims}), 0) < %s"
    )


class RecallSourcesMixin(RecallEngine):
    def _kg_scope_ready(self) -> bool:
        """True when restricted KG serving can run, i.e. schema 056 is applied.

        Fail-closed: a probe error or a missing column answers False, and the restricted
        caller is served no facts. Full-trust paths never call this. They do not read
        the column, so its absence cannot break them. Applying 056 under a running server
        takes effect without a restart (see ``ColumnProbe``)."""
        cached = KG_SCOPE_PROBE.cached(self._db_url)
        if cached is not None:
            return cached
        try:
            with self._pg() as pg:
                return KG_SCOPE_PROBE.ready(pg, self._db_url)
        except Exception as e:
            logger.warning("KG provenance probe failed (%s): restricted KG serving stays off", e)
            return False

    def _kg_scope_blocked(self, allowed_projects: list[str] | None) -> bool:
        """True when a restricted KG read must return nothing without querying.

        ``None`` is full trust (never blocked). An empty allowlist cannot contain a
        non-empty provenance set, and a database without schema 056 cannot prove one."""
        if allowed_projects is None:
            return False
        return not allowed_projects or not self._kg_scope_ready()

    def _kg_scope_error(self, allowed_projects: list[str] | None, e: Exception) -> bool:
        """True when ``e`` is the restricted predicate hitting a missing column (056 rolled
        back, or dropped after the probe cached it). The caller then returns empty."""
        return allowed_projects is not None and KG_SCOPE_PROBE.note_error(e, self._db_url)

    def _search_notes(
        self,
        query: str,
        query_emb: list[float],
        project: str | None,
        audience: str | None = None,
    ) -> list[dict[str, Any]]:
        """Notes bucket: hook-KNN top-``notes_fetch`` live notes (global types + this
        project's), floored by the cross-encoder on "hook — body" text, top
        ``notes_limit`` served. May return [] when nothing clears the floor — notes
        self-gate on relevance like web. Fail-soft: any error serves no bucket.

        ``audience`` is the restricted surface's tier filter (schema 053) — 'work-safe'
        for a restricted caller, None for full trust. Fail-soft here means an EMPTY
        bucket, which is also the fail-closed answer; there is no path where the filter
        is dropped and the leg serves."""
        cfg = self._config
        try:
            with self._pg() as conn:
                rows = search_live_notes(
                    conn,
                    cfg.kg_owner,
                    project,
                    query_emb,
                    limit=cfg.notes_fetch,
                    audience=audience,
                )
        except Exception as e:
            logger.warning("notes leg fetch failed: %s", e)
            _warn(f"notes leg failed ({_err_brief(e)}): no notes served.")
            return []
        if not rows:
            return []
        if cfg.notes_floor > 0:
            docs = [
                {"fact": f"{r['hook']} — {r['body']}"[: cfg.rerank_doc_cap], "_r": r} for r in rows
            ]
            rows = [d["_r"] for d in self._floor_by_rerank(query, docs, cfg.notes_floor)]
        served = []
        for r in rows[: cfg.notes_limit]:
            body = r["body"]
            if len(body) > cfg.notes_body_cap:
                body = body[: cfg.notes_body_cap] + "…"
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
        allowed_projects: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """KG facts leg over kg_entities / kg_relationships (task #67).

        ``allowed_projects`` is a restricted surface's allowlist: only facts whose whole
        provenance lies inside it are served (schema 056, ``kg_pg.scope_predicate``).
        Without schema 056 the restricted leg returns nothing and logs one warning. It
        never adds a response warning, matching the pre-056 skip. ``None`` = full trust.

        Runs inside one transaction so the planner GUCs are SET LOCAL — scoped
        to this query, not the thread-local connection that other legs may
        reuse later (enable_seqscan=off session-wide would be a foot-gun for
        any future unindexed query). hnsw.ef_search is already set
        on every connection by ingestion/pg_pool. A failure degrades to an empty facts
        bucket rather than failing the whole recall.
        """
        cfg = self._config
        if self._kg_scope_blocked(allowed_projects):
            return [], []
        try:
            with self._pg() as conn:
                with conn.transaction():
                    # kg_pg indexes rows positionally; recall borrows dict-row
                    # connections, so override at the cursor.
                    cur = conn.cursor(row_factory=tuple_row)
                    # Both planner GUCs in one round trip; is_local = true is SET LOCAL.
                    cur.execute(
                        "SELECT set_config('enable_seqscan', 'off', true), "
                        "set_config('max_parallel_workers_per_gather', '0', true)"
                    )
                    return search_kg_postgres(
                        cur,
                        query,
                        query_emb,
                        cfg.kg_owner,
                        group_id,
                        session_focus,
                        fact_limit,
                        allowed_projects=allowed_projects,
                    )
        except Exception as e:
            if self._kg_scope_error(allowed_projects, e):
                return [], []
            logger.warning("KG search failed: %s", e)
            _warn(f"KG facts leg failed ({_err_brief(e)}): no facts served.")
            return [], []

    def _self_session_facts(self, uuids: list[str], self_session: str) -> set[str]:
        """The facts among ``uuids`` whose every source episode belongs to ``self_session``.

        The KG half of ``_exclude_self``: the drain extracts facts while a session is
        still running, so the session's own claims came back to it as memory (2.6% of
        served facts, in 10% of hook-injected recalls, over the 14 days to 2026-10-09).
        A fact the session only reinforced keeps an older source and still serves, and
        a fact with no recorded source (web artifacts, a malformed ``episodes`` value)
        is never excluded. Fails open: a lookup error serves the facts as before.
        """
        if not uuids:
            return set()
        try:
            with self._pg() as conn:
                rows = conn.execute(
                    "SELECT r.uuid FROM kg_relationships r "
                    f"WHERE r.uuid = ANY(%s) AND {_self_only_sql('r')}",
                    (uuids, self_session),
                ).fetchall()
        except Exception as e:
            logger.warning("self-session fact lookup failed: %s", e)
            return set()
        return {r["uuid"] for r in rows}

    def _fetch_superseded_pairs_pg(
        self,
        group_id: str,
        active_edge_uuids: list[str],
        cap: int,
        allowed_projects: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Superseded-fact pairs for served edges (Postgres port of the old
        FalkorDB _fetch_history_pairs).

        DISTINCT ON picks the most recently invalidated predecessor per active
        edge in SQL (the FalkorDB path does this dedup in Python).

        A predecessor must actually be one: either it carries the schema-028
        ``invalidated_by`` link to the served edge, or it has no link at all and
        shares the served edge's predicate ``name``. Endpoints alone are not enough
        (2026-10-09): every retired edge between the same two entities was being
        paired with whatever live edge between them got served, so "User told the
        assistant on 08-29 to stop X" showed up as superseded by an unrelated
        correction weeks later.

        ``allowed_projects`` (restricted surface) requires BOTH sides of a pair, the
        served edge and the displaced one, to pass the schema-056 provenance rule, so a
        pair is dropped when either side is out of scope. The filter runs before
        DISTINCT ON: when an edge's newest predecessor is out of scope, an older
        in-scope one can still be shown.
        """
        cfg = self._config
        if not active_edge_uuids or cap <= 0:
            return []
        if self._kg_scope_blocked(allowed_projects):
            return []
        scope_sql = ""
        scope_args: tuple[Any, ...] = ()
        if allowed_projects is not None:
            scope_sql = f" AND {scope_predicate('a')} AND {scope_predicate('o')}"
            scope_args = (allowed_projects, allowed_projects)
        try:
            with self._pg() as conn:
                rows = conn.execute(
                    f"""
                    SELECT DISTINCT ON (a.uuid) a.uuid AS uid, a.fact AS now_fact,
                           o.id AS old_id, o.fact AS old_fact
                    FROM kg_relationships a
                    JOIN kg_relationships o
                      ON o.src_uuid = a.src_uuid AND o.tgt_uuid = a.tgt_uuid
                     AND o.owner_id = a.owner_id AND o.group_id = a.group_id
                    WHERE a.owner_id = %s AND a.group_id = %s
                      AND a.uuid = ANY(%s) AND a.t_invalid IS NULL
                      AND o.t_invalid IS NOT NULL AND o.uuid <> a.uuid
                      AND ((o.invalidated_by = a.uuid
                            AND {_link_close_sql("o", "a", cfg.embed_dims)})
                           OR (o.invalidated_by IS NULL AND o.name = a.name))
                      AND o.fact IS NOT NULL AND a.fact IS NOT NULL{scope_sql}
                    ORDER BY a.uuid, o.t_invalid DESC
                    """,
                    (
                        cfg.kg_owner,
                        group_id,
                        active_edge_uuids,
                        cfg.sup_link_max_dist,
                        *scope_args,
                    ),
                ).fetchall()
        except Exception as e:
            if self._kg_scope_error(allowed_projects, e):
                return []
            logger.debug("PG superseded-pairs query failed: %s", e)
            return []
        by_uid = {r["uid"]: r for r in rows}
        out: list[dict[str, Any]] = []
        for uid in active_edge_uuids:
            r = by_uid.get(uid)
            if r is None:
                continue
            # id is the invalidated (old) edge's id, "f:<id>" like a regular fact.
            # Collision-free: the facts bucket serves the CURRENT edge (a.uuid), never
            # the superseded predecessor (o.uuid, enforced <> a.uuid above). Lets the
            # stale pair be cited in recall_feedback.
            out.append(
                {
                    "id": f"f:{r['old_id']}",
                    "fact": r["old_fact"],
                    "superseded_by": r["now_fact"],
                }
            )
            if len(out) >= cap:
                break
        return out

    def _episode_supersessions(
        self,
        episode_ids: list[int],
        group_id: str,
        cap: int = 6,
        allowed_projects: list[str] | None = None,
        self_session: str | None = None,
        served_text: dict[int, str] | None = None,
    ) -> dict[int, list[str]]:
        """Map served episode ids -> the CURRENT facts that superseded a claim each made.

        A retired edge P citing the episode (episodes @> [id]) links to its superseding live edge N
        via P.invalidated_by (schema 028 + backfill); N.fact is the "now" value. Hits the partial GIN
        (schema 029, WHERE invalidated_by IS NOT NULL) so the per-recall lookup is cheap. Fail-open —
        a lookup error just yields no annotations, never breaks recall.

        ``allowed_projects`` (restricted surface) requires both the retired edge P and the
        superseding edge N to pass the schema-056 provenance rule: N's text is what gets
        served, and P is what ties it to the episode.

        ``self_session`` drops a successor N sourced only from the calling session, the
        overlay half of self-exclusion (see ``_self_session_facts``).

        ``served_text`` (episode id -> the text actually served for it) keeps a successor
        only where that text states P's claim (``OVERLAY_MIN_SUPPORT``)."""
        cfg = self._config
        if not episode_ids:
            return {}
        if self._kg_scope_blocked(allowed_projects):
            return {}
        scope_sql = ""
        scope_args: list[Any] = []
        if allowed_projects is not None:
            scope_sql = f"AND {scope_predicate('p')} AND {scope_predicate('n')} "
            scope_args = [allowed_projects, allowed_projects]
        if self_session:
            scope_sql += f"AND NOT {_self_only_sql('n')} "
            scope_args.append(self_session)
        ors = " OR ".join(["p.episodes @> %s::jsonb"] * len(episode_ids))
        params: list[Any] = [
            *(json.dumps([i]) for i in episode_ids),
            cfg.kg_owner,
            group_id,
            cfg.sup_link_max_dist,
            *scope_args,
            cap if served_text is None else cap * 4,  # headroom for the support filter
        ]
        try:
            with self._pg() as conn:
                rows = conn.execute(
                    "SELECT p.episodes, p.fact AS old_fact, n.fact FROM kg_relationships p "
                    "JOIN kg_relationships n ON n.uuid = p.invalidated_by "
                    f"WHERE p.invalidated_by IS NOT NULL AND ({ors}) "
                    "  AND p.owner_id = %s AND p.group_id = %s AND n.t_invalid IS NULL "
                    f"  AND {_link_close_sql('p', 'n', cfg.embed_dims)} "
                    f"{scope_sql}"
                    "LIMIT %s",
                    params,
                ).fetchall()
        except Exception as e:
            if self._kg_scope_error(allowed_projects, e):
                return {}
            logger.warning("episode supersession lookup failed: %s", e)
            return {}
        idset = set(episode_ids)
        out: dict[int, list[str]] = {}
        n_links = 0
        for r in rows:
            fact = r.get("fact")
            if not fact:
                continue
            attached = False
            for eid in r.get("episodes") or []:
                if eid not in idset:
                    continue
                if (
                    served_text is not None
                    and claim_support(r.get("old_fact") or "", served_text.get(int(eid), ""))
                    < OVERLAY_MIN_SUPPORT
                ):
                    continue
                out.setdefault(int(eid), []).append(fact)
                attached = True
            n_links += attached
            if n_links >= cap:
                break
        return out

    def _surface_supersessions(
        self,
        query_emb: list[float] | None,
        group_id: str,
        allowed_projects: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """A query that matches a now-INVALID fact should still return the CURRENT answer.

        Finds superseded edges near the query that carry a precise successor link (invalidated_by,
        schema 028), resolves the live successor, and returns the distinct successors, nearest
        first, distance-gated (_SUP_MAX_DIST) so only on-topic superseded facts pull their
        correction in. The caller drops those it already serves and caps the rest. Go-forward
        coverage only (no link => skipped); never returns the stale fact itself. Shape matches
        the KG fact leg ({fact, _uuid, _id, _date, _ongoing, _supported}) so it flows through fact
        serving. Fail-open — a lookup error just yields no extras.

        ``allowed_projects`` (restricted surface) requires both the matched superseded edge
        and its served successor to pass the schema-056 provenance rule."""
        cfg = self._config
        if query_emb is None:
            return []
        if self._kg_scope_blocked(allowed_projects):
            return []
        scope_sql = ""
        scope_args: tuple[Any, ...] = ()
        if allowed_projects is not None:
            scope_sql = f"  AND {scope_predicate('p')} AND {scope_predicate('n')} "
            scope_args = (allowed_projects, allowed_projects)
        vec = _vec_literal(query_emb)
        try:
            with self._pg() as conn:
                rows = conn.execute(
                    "SELECT n.id, n.uuid, n.fact, n.t_valid, n.ongoing, n.last_supported_at, "
                    f"  (p.fact_embedding::halfvec({cfg.embed_dims}) <=> %s::halfvec({cfg.embed_dims})) AS d "
                    "FROM kg_relationships p "
                    "JOIN kg_relationships n ON n.uuid = p.invalidated_by AND n.t_invalid IS NULL "
                    "WHERE p.t_invalid IS NOT NULL AND p.invalidated_by IS NOT NULL "
                    "  AND p.fact_embedding IS NOT NULL AND p.owner_id = %s AND p.group_id = %s "
                    f"  AND {_link_close_sql('p', 'n', cfg.embed_dims)} "
                    f"{scope_sql}"
                    f"ORDER BY p.fact_embedding::halfvec({cfg.embed_dims}) <=> %s::halfvec({cfg.embed_dims}) LIMIT %s",
                    (
                        vec,
                        cfg.kg_owner,
                        group_id,
                        cfg.sup_link_max_dist,
                        *scope_args,
                        vec,
                        cfg.sup_candidates,
                    ),
                ).fetchall()
        except Exception as e:
            if self._kg_scope_error(allowed_projects, e):
                return []
            logger.warning("supersession surface failed: %s", e)
            return []
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for r in rows:
            d = r.get("d")
            if d is None or d > cfg.sup_max_dist:
                continue
            u, fact = r.get("uuid"), r.get("fact")
            if u and fact and u not in seen:
                out.append(
                    {
                        "fact": fact,
                        "_uuid": u,
                        "_id": r["id"],
                        "_date": r.get("t_valid"),
                        "_ongoing": r.get("ongoing"),
                        "_supported": r.get("last_supported_at"),
                    }
                )
                seen.add(u)
        return out
