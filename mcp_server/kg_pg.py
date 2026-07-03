"""Postgres read-port of the KG retrieval leg (mcp_server/recall.py::_search_kg).

Queries the ``kg_entities`` / ``kg_relationships`` tables (schema/017) that the
extraction pipeline writes (the canonical KG store since #67 PR 3), scoped by ``owner_id`` + ``group_id``.
Returns the same ``(facts, seed_entities)`` shape ``recall._search_kg`` returns, so
it can be wired into recall behind a shadow-read flag and, eventually, become the
source of truth.

Retrieval mirrors ``_search_kg`` exactly:
  1. fact-embedding vector KNN over live edges (partial HNSW: t_invalid IS NULL)
  2. fact BM25 over edge text (ParadeDB)
  3. entity-seed vector KNN + live-degree gate + session-focus bonus -> top 8 seeds
  4. 1-hop traversal from each seed (per-seed LIMIT 8)
  5. RRF fuse [vec, bm25, hop], take top ``limit``

The caller's cursor MUST have these session GUCs set (see parity_kg_read.py):
  - ``hnsw.ef_search = 200``               (matches FalkorDB's efRuntime)
  - ``enable_seqscan = off``               } pgvector won't use the HNSW index when
  - ``max_parallel_workers_per_gather = 0``} an equality filter is also present --
the planner picks a bitmap/parallel-seq-scan on the owner_id btree and sorts 47K
rows (~1.6s) instead. The fact-vector leg sidesteps this by over-fetching the
GLOBAL live-fact HNSW on the BARE partial-index predicate (no owner/group -> stays
on HNSW), then filtering the tenant scope on that small candidate set. The two
GUCs force the planner onto the index. (The long-term multi-tenant-at-scale answer
is LIST partitioning by owner_id so partition pruning removes the filter entirely;
not needed while there is one real owner + throwaway DBs for isolated runs.)
"""

from __future__ import annotations

from typing import Any

from ingestion.embedding import embed_dims

# Embedding width for the halfvec casts below — must match the provisioned schema
# (and its HNSW index expressions) verbatim. Default 2048 (Voyage prod, unchanged).
_EMBED_DIMS = embed_dims()

# RRF constant matches recall._rrf_fuse / bench_pg_kg (k=60, 1-indexed rank).
_RRF_K = 60
# Fact-vector over-fetch pool: how many global-nearest live facts to pull before
# applying the owner/group filter. Headroom for multi-tenant filtering; for a single
# owner every candidate matches and the outer LIMIT (limit*3) is what bites.
_OVERFETCH = 200


def _rrf_fuse(lists: list[list[str]], k: int = _RRF_K) -> dict[str, float]:
    scores: dict[str, float] = {}
    for lst in lists:
        for rank, uuid_ in enumerate(lst):
            scores[uuid_] = scores.get(uuid_, 0.0) + 1.0 / (k + rank + 1)
    return scores


def _vec_literal(emb: list[float]) -> str:
    return "[" + ",".join(map(str, emb)) + "]"


def search_kg_postgres(
    cur: Any,
    query: str,
    query_emb: list[float],
    owner_id: str,
    group_id: str,
    session_focus: list[str],
    limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Mirror of recall._search_kg over the Postgres KG mirror.

    Returns ``(facts, seed_entities)``: facts carry internal ``_uuid`` (for
    retrieval-count bumps + history lookup), seed_entities is the ranked list of
    connected seeds used by the entity bucket downstream.
    """
    emb_s = _vec_literal(query_emb)
    # uuid -> (fact text, t_valid). t_valid = when the fact became true (bitemporal valid-from),
    # surfaced as the fact's "as-of" date so the reader can weight currency. 100% populated on
    # live edges. Served facts are already live (t_invalid IS NULL filtered everywhere below).
    fact_by_uuid: dict[str, tuple[str, Any]] = {}

    # 1 — fact-embedding vector KNN. Over-fetch the GLOBAL live-fact HNSW on the bare
    # partial-index predicate (no owner/group -> the planner keeps the kg_rel_hnsw
    # index instead of falling back to a 47K-row bitmap+sort), then filter the tenant
    # scope on the small candidate set. See module docstring for the required GUCs.
    cur.execute(
        "SELECT uuid, fact, t_valid FROM ("
        "  SELECT uuid, fact, t_valid, owner_id, group_id FROM kg_relationships "
        "  WHERE t_invalid IS NULL AND fact_embedding IS NOT NULL "
        f"  ORDER BY fact_embedding::halfvec({_EMBED_DIMS}) <=> %s::halfvec({_EMBED_DIMS}) LIMIT %s"
        ") sub WHERE owner_id = %s AND group_id = %s LIMIT %s",
        (emb_s, _OVERFETCH, owner_id, group_id, limit * 3),
    )
    vec_uuids: list[str] = []
    for u, f, tv in cur.fetchall():
        if not f:
            continue
        fact_by_uuid.setdefault(u, (f, tv))
        vec_uuids.append(u)

    # 2 — BM25 full-text over fact text (ParadeDB). Same alnum/space sanitize as
    # recall._search_kg so identifiers/error strings survive. t_invalid filtered
    # in the WHERE (before LIMIT), matching the FalkorDB fulltext leg.
    safe = "".join(c if (c.isalnum() or c.isspace()) else " " for c in query).strip()
    bm25_uuids: list[str] = []
    if safe:
        cur.execute(
            "SELECT uuid, fact, t_valid, paradedb.score(id) AS sc FROM kg_relationships "
            "WHERE id @@@ paradedb.match('fact', %s) "
            "  AND owner_id = %s AND group_id = %s AND t_invalid IS NULL "
            "ORDER BY sc DESC LIMIT %s",
            (safe, owner_id, group_id, limit * 3),
        )
        for u, f, tv, _sc in cur.fetchall():
            if not f:
                continue
            fact_by_uuid.setdefault(u, (f, tv))
            bm25_uuids.append(u)

    # 3 — entity seed vector KNN (top 25) + live-degree gate + focus bonus -> top 8.
    # The seed KNN uses the SAME bare-predicate over-fetch trick as the fact leg
    # (step 1): an owner/group equality filter directly alongside the vector ORDER BY
    # makes the planner pick a Bitmap Heap Scan on kg_entities_owner and sort EVERY
    # in-scope entity (~1.25s once the graph passed ~36K entities — measured
    # 2026-06-20, the entity HNSW was never used). enable_seqscan=off does NOT prevent
    # a bitmap scan. Fix: ORDER the GLOBAL HNSW on the bare partial-index predicate
    # (embedding IS NOT NULL -> stays on kg_entities_hnsw), over-fetch _OVERFETCH, then
    # filter the tenant scope on that small candidate set (single owner: ~97% survive,
    # 25 found trivially). 1248ms -> 26ms verified via EXPLAIN ANALYZE.
    # Degree is computed LIVE over live edges (matches FalkorDB), batched into ONE query
    # for all 25 seeds: a per-seed correlated count with an (src OR tgt) predicate can't
    # use the btree indexes and seq-scans the whole edge table per seed. Splitting the OR
    # into two index-driven `<col> IN (seeds)` semijoins keeps it on kg_rel_src/kg_rel_tgt.
    cur.execute(
        "WITH seeds AS ("
        "  SELECT uuid, name, summary, dist FROM ("
        "    SELECT uuid, name, summary, owner_id, group_id, "
        f"           embedding::halfvec({_EMBED_DIMS}) <=> %s::halfvec({_EMBED_DIMS}) AS dist "
        "    FROM kg_entities WHERE embedding IS NOT NULL "
        f"    ORDER BY embedding::halfvec({_EMBED_DIMS}) <=> %s::halfvec({_EMBED_DIMS}) LIMIT %s"
        "  ) e WHERE owner_id = %s AND group_id = %s "
        "  ORDER BY dist LIMIT 25"
        "), deg AS ("
        "  SELECT u, count(*) AS d FROM ("
        "    SELECT src_uuid AS u FROM kg_relationships "
        "      WHERE owner_id = %s AND group_id = %s AND t_invalid IS NULL "
        "        AND src_uuid IN (SELECT uuid FROM seeds) "
        "    UNION ALL "
        "    SELECT tgt_uuid AS u FROM kg_relationships "
        "      WHERE owner_id = %s AND group_id = %s AND t_invalid IS NULL "
        "        AND tgt_uuid IN (SELECT uuid FROM seeds) "
        "  ) z GROUP BY u"
        ") "
        "SELECT s.uuid, s.name, s.summary, s.dist, COALESCE(d.d, 0) AS deg "
        "FROM seeds s LEFT JOIN deg d ON d.u = s.uuid ORDER BY s.dist",
        (emb_s, emb_s, _OVERFETCH, owner_id, group_id, owner_id, group_id, owner_id, group_id),
    )
    focus_set = set(session_focus)
    connected: list[tuple[float, str, str | None, str | None]] = []
    for uuid_, name, summary, dist, deg in cur.fetchall():
        if not deg:
            continue
        bonus = 0.3 if (name in focus_set or uuid_ in focus_set) else 0.0
        connected.append((float(dist) - bonus, uuid_, name, summary))
        if len(connected) >= 8:
            break
    connected.sort(key=lambda x: x[0])
    seed_uuids = [c[1] for c in connected]
    seed_entities = [{"uuid": u, "name": n, "summary": s} for _, u, n, s in connected]

    # 4 — 1-hop traversal facts from the seeds (per-seed LIMIT 8, undirected).
    hop_uuids: list[str] = []
    for sd in seed_uuids:
        cur.execute(
            "SELECT uuid, fact, t_valid FROM kg_relationships "
            "WHERE owner_id = %s AND group_id = %s AND t_invalid IS NULL "
            "  AND (src_uuid = %s OR tgt_uuid = %s) LIMIT 8",
            (owner_id, group_id, sd, sd),
        )
        for u, f, tv in cur.fetchall():
            if u and f:
                fact_by_uuid.setdefault(u, (f, tv))
                hop_uuids.append(u)

    # 5 — RRF fuse the three ranked lists; take the top `limit`.
    fused = _rrf_fuse([vec_uuids, bm25_uuids, hop_uuids])
    top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    results = [
        {"fact": fact_by_uuid[u][0], "_uuid": u, "_date": fact_by_uuid[u][1]}
        for u, _ in top
        if u in fact_by_uuid
    ][:limit]
    return results, seed_entities
