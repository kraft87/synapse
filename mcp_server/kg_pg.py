"""Postgres KG retrieval leg (recall's ``_search_kg``).

Queries the ``kg_entities`` / ``kg_relationships`` tables (schema/017) that the
extraction pipeline writes, scoped by ``owner_id`` + ``group_id``. Returns
``(facts, seed_entities)``.

Retrieval:
  1. fact-embedding vector KNN over live edges (partial HNSW: t_invalid IS NULL)
  2. fact BM25 over edge text (ParadeDB)
  3. entity-seed vector KNN + live-edge gate + session-focus bonus -> top 8 seeds
  4. 1-hop traversal from the seeds (per-seed LIMIT 8, one statement for all of them)
  5. RRF fuse [vec, bm25, hop], take top ``limit``

The caller's cursor MUST have these session GUCs set (recall_sources._search_kg does):
  - ``hnsw.ef_search = 200``               (set on every pooled connection)
  - ``enable_seqscan = off``               } pgvector won't use the HNSW index when
  - ``max_parallel_workers_per_gather = 0``} an equality filter is also present --
the planner picks a bitmap/parallel-seq-scan on the owner_id btree and sorts 47K
rows (~1.6s) instead. The fact-vector leg sidesteps this by over-fetching the
GLOBAL live-fact HNSW on the BARE partial-index predicate (no owner/group -> stays
on HNSW), then filtering the tenant scope on that small candidate set. The two
GUCs force the planner onto the index. (The long-term multi-tenant-at-scale answer
is LIST partitioning by owner_id so partition pruning removes the filter entirely;
not needed while there is one real owner + throwaway DBs for isolated runs.)

Restricted surfaces (schema 053/054) pass ``allowed_projects``; every leg then serves
only facts whose cached provenance set (schema 056, ``source_projects``) is known,
non-empty and inside the allowlist — see :func:`scope_predicate`. ``None`` is full trust
and runs the original SQL unchanged.
"""

from __future__ import annotations

from typing import Any

from ingestion.db_connection import bm25_query_text
from ingestion.embedding import embed_dims

# Embedding width for the halfvec casts below — must match the provisioned schema
# (and its HNSW index expressions) verbatim. Default 2048 (Voyage prod, unchanged).
_EMBED_DIMS = embed_dims()

# RRF constant (k=60, 1-indexed rank), the same as recall_ranking.merge_rrf.
_RRF_K = 60
# Fact-vector over-fetch pool: how many global-nearest live facts to pull before
# applying the owner/group filter. Headroom for multi-tenant filtering; for a single
# owner every candidate matches and the outer LIMIT (limit*3) is what bites.
_OVERFETCH = 200


def scope_predicate(alias: str = "") -> str:
    """The restricted-surface serving rule over ``kg_relationships`` (schema 056).

    A fact is servable to a restricted caller iff its source-project set is KNOWN
    (``source_projects`` is NULL whenever any source episode is missing, has no project,
    or the fact has no episode provenance at all), NON-EMPTY, and a SUBSET of the
    caller's allowlist — so a fact with even one source outside the allowlist (mixed
    provenance) is never served. One ``%s`` placeholder: the allowlist as a list.
    """
    col = f"{alias}.source_projects" if alias else "source_projects"
    return f"{col} IS NOT NULL AND cardinality({col}) > 0 AND {col} <@ %s::text[]"


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
    allowed_projects: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The facts leg: fact KNN + fact BM25 + one-hop traversal from entity seeds, RRF-fused.

    Returns ``(facts, seed_entities)``: facts carry internal ``_uuid`` (for
    retrieval-count bumps + history lookup) and ``_id`` (``kg_relationships.id``, the
    number a fact is served and rated under); seed_entities is the ranked list of
    connected seeds the hops started from, which recall does not serve (the entity
    display bucket is retired).

    ``allowed_projects`` (a restricted surface's allowlist) applies
    :func:`scope_predicate` to every fact-returning leg and to the seed degree gate,
    and returns no seed entities: an entity's name and summary are distilled from ALL
    its facts, so nothing proves them confined to the allowlist. ``None`` (full trust)
    executes exactly the statements it always did. Requires schema 056; the caller owns
    the column-missing fallback (recall_sources._search_kg).
    """
    restricted = allowed_projects is not None
    # Spliced into the full-trust statements as "" so their text is unchanged; a
    # restricted call appends the predicate and its one bound allowlist parameter.
    scope_sql = f"  AND {scope_predicate()} " if restricted else ""
    scope_args: tuple[Any, ...] = (allowed_projects,) if restricted else ()
    emb_s = _vec_literal(query_emb)
    # uuid -> (id, fact text, t_valid, ongoing, last_supported_at). t_valid = when the fact became
    # true (bitemporal valid-from), surfaced as the fact's "as-of" date; ongoing +
    # last_supported_at (schema 058) say whether the claim can lapse silently and when the
    # user last restated it, so the reader can tell "valid-from 2025, never confirmed since"
    # from "restated last week". Served facts are already live (t_invalid IS NULL filtered
    # everywhere below).
    fact_by_uuid: dict[str, tuple[int, str, Any, Any, Any]] = {}

    # 1 — fact-embedding vector KNN. Over-fetch the GLOBAL live-fact HNSW on the bare
    # partial-index predicate (no owner/group -> the planner keeps the kg_rel_hnsw
    # index instead of falling back to a 47K-row bitmap+sort), then filter the tenant
    # scope on the small candidate set. See module docstring for the required GUCs.
    if not restricted:
        cur.execute(
            "SELECT id, uuid, fact, t_valid, ongoing, last_supported_at FROM ("  # nosec B608 — _EMBED_DIMS is a validated int; the values are bound
            "  SELECT id, uuid, fact, t_valid, ongoing, last_supported_at, owner_id, group_id "
            "  FROM kg_relationships "
            "  WHERE t_invalid IS NULL AND fact_embedding IS NOT NULL "
            f"  ORDER BY fact_embedding::halfvec({_EMBED_DIMS}) <=> %s::halfvec({_EMBED_DIMS}) LIMIT %s"
            ") sub WHERE owner_id = %s AND group_id = %s LIMIT %s",
            (emb_s, _OVERFETCH, owner_id, group_id, limit * 3),
        )
    else:
        # Restricted: EXACT KNN over the allowlisted set. The global-HNSW over-fetch
        # above would under-return here — a selective filter applied to the 200 nearest
        # facts corpus-wide can leave few or none. The MATERIALIZED fence keeps the
        # planner off the HNSW ordering; the provenance filter rides the partial GIN
        # (kg_rel_source_projects_gin). Cost scales with the allowlisted fact count.
        cur.execute(
            "WITH cand AS MATERIALIZED ("  # nosec B608 — _EMBED_DIMS is a validated int and scope_predicate() a fixed predicate; the values are bound
            "  SELECT id, uuid, fact, t_valid, ongoing, last_supported_at, "
            f"         fact_embedding::halfvec({_EMBED_DIMS}) <=> %s::halfvec({_EMBED_DIMS}) AS dist "
            "  FROM kg_relationships "
            "  WHERE t_invalid IS NULL AND fact_embedding IS NOT NULL "
            "    AND owner_id = %s AND group_id = %s "
            f"   AND {scope_predicate()}"
            ") SELECT id, uuid, fact, t_valid, ongoing, last_supported_at FROM cand "
            "ORDER BY dist LIMIT %s",
            (emb_s, owner_id, group_id, allowed_projects, limit * 3),
        )
    vec_uuids: list[str] = []
    for i, u, f, tv, og, sup in cur.fetchall():
        if not f:
            continue
        fact_by_uuid.setdefault(u, (i, f, tv, og, sup))
        vec_uuids.append(u)

    # 2 — BM25 full-text over fact text (ParadeDB). Sanitized by bm25_query_text so
    # identifiers/error strings survive. t_invalid filtered in the WHERE (before LIMIT).
    safe = bm25_query_text(query)
    bm25_uuids: list[str] = []
    if safe:
        cur.execute(
            "SELECT id, uuid, fact, t_valid, ongoing, last_supported_at, paradedb.score(id) AS sc "  # nosec B608 — scope_sql is a fixed predicate; the values are bound
            "FROM kg_relationships "
            "WHERE id @@@ paradedb.match('fact', %s) "
            "  AND owner_id = %s AND group_id = %s AND t_invalid IS NULL "
            f"{scope_sql}"
            "ORDER BY sc DESC LIMIT %s",
            (safe, owner_id, group_id, *scope_args, limit * 3),
        )
        for i, u, f, tv, og, sup, _sc in cur.fetchall():
            if not f:
                continue
            fact_by_uuid.setdefault(u, (i, f, tv, og, sup))
            bm25_uuids.append(u)

    # 3 — entity seed vector KNN (top 25) + live-edge gate + focus bonus -> top 8.
    # The seed KNN uses the SAME bare-predicate over-fetch trick as the fact leg
    # (step 1): an owner/group equality filter directly alongside the vector ORDER BY
    # makes the planner pick a Bitmap Heap Scan on kg_entities_norm and sort EVERY
    # in-scope entity (~1.25s once the graph passed ~36K entities — measured
    # 2026-06-20, the entity HNSW was never used). enable_seqscan=off does NOT prevent
    # a bitmap scan. Fix: ORDER the GLOBAL HNSW on the bare predicate
    # (embedding IS NOT NULL -> stays on kg_entities_hnsw), over-fetch _OVERFETCH, then
    # filter the tenant scope on that small candidate set (single owner: ~97% survive,
    # 25 found trivially). 1248ms -> 26ms verified via EXPLAIN ANALYZE.
    # A seed must have at least one live edge, asked as two EXISTS
    # (src side, tgt side) over all 25 seeds in this ONE query: each probe stops at the
    # first live edge, where a live-degree COUNT read every edge of every seed (the User
    # supernode has thousands), and a correlated (src OR tgt) predicate could not use the
    # btree indexes (kg_rel_pair / kg_rel_tgt) and seq-scanned the whole edge table per seed.
    cur.execute(
        "WITH seeds AS ("  # nosec B608 — _EMBED_DIMS is a validated int and scope_sql a fixed predicate; the values are bound
        "  SELECT uuid, name, summary, dist FROM ("
        "    SELECT uuid, name, summary, owner_id, group_id, "
        f"           embedding::halfvec({_EMBED_DIMS}) <=> %s::halfvec({_EMBED_DIMS}) AS dist "
        "    FROM kg_entities WHERE embedding IS NOT NULL "
        f"    ORDER BY embedding::halfvec({_EMBED_DIMS}) <=> %s::halfvec({_EMBED_DIMS}) LIMIT %s"
        "  ) e WHERE owner_id = %s AND group_id = %s "
        "  ORDER BY dist LIMIT 25"
        ") "
        "SELECT s.uuid, s.name, s.summary, s.dist FROM seeds s "
        "WHERE EXISTS (SELECT 1 FROM kg_relationships "
        "      WHERE owner_id = %s AND group_id = %s AND t_invalid IS NULL "
        f"{scope_sql}"
        "        AND src_uuid = s.uuid) "
        "   OR EXISTS (SELECT 1 FROM kg_relationships "
        "      WHERE owner_id = %s AND group_id = %s AND t_invalid IS NULL "
        f"{scope_sql}"
        "        AND tgt_uuid = s.uuid) "
        "ORDER BY s.dist",
        (
            emb_s,
            emb_s,
            _OVERFETCH,
            owner_id,
            group_id,
            owner_id,
            group_id,
            *scope_args,
            owner_id,
            group_id,
            *scope_args,
        ),
    )
    focus_set = set(session_focus)
    connected: list[tuple[float, str, str | None, str | None]] = []
    for uuid_, name, summary, dist in cur.fetchall():
        bonus = 0.3 if (name in focus_set or uuid_ in focus_set) else 0.0
        connected.append((float(dist) - bonus, uuid_, name, summary))
        if len(connected) >= 8:
            break
    connected.sort(key=lambda x: x[0])
    seed_uuids = [c[1] for c in connected]
    seed_entities = (
        [] if restricted else [{"uuid": u, "name": n, "summary": s} for _, u, n, s in connected]
    )

    # 4 — 1-hop traversal facts from the seeds (per-seed LIMIT 8, undirected), ONE statement:
    # a LATERAL per seed, in seed order. ORDER BY t_valid DESC (2026-08-07): an unordered
    # LIMIT 8 was fine when every entity had a handful of edges, but a supernode seed (the User
    # node, live degree ~3.4k and growing with event facts) turned it into 8 arbitrary edges.
    # Recency is the least-wrong single ordering for a "what about X" hop sample. Each side
    # (src, tgt) takes its own newest 8 and the union its newest 8: the same 8 as sorting
    # (src OR tgt) whole, but the optional kg_rel_src_live / kg_rel_tgt_live indexes (owner,
    # group, endpoint, t_valid DESC) hand each side its 8 without reading the rest. A
    # self-loop is one hop.
    hop_uuids: list[str] = []
    if seed_uuids:
        hop_side = (
            "SELECT id, uuid, fact, t_valid, ongoing, last_supported_at FROM kg_relationships "
            "WHERE owner_id = %s AND group_id = %s AND t_invalid IS NULL "
            f"{scope_sql}"
        )
        cur.execute(
            "SELECT h.id, h.uuid, h.fact, h.t_valid, h.ongoing, h.last_supported_at "  # nosec B608 — hop_side is a fixed statement with scope_sql, a fixed predicate; the values are bound
            "FROM unnest(%s::text[]) WITH ORDINALITY AS s(sd, ord) "
            "CROSS JOIN LATERAL ("
            f"  ({hop_side}AND src_uuid = s.sd ORDER BY t_valid DESC LIMIT 8) "
            "  UNION ALL "
            f"  ({hop_side}AND tgt_uuid = s.sd AND src_uuid <> s.sd ORDER BY t_valid DESC LIMIT 8) "
            "  ORDER BY t_valid DESC LIMIT 8"
            ") h ORDER BY s.ord, h.t_valid DESC",
            (seed_uuids, owner_id, group_id, *scope_args, owner_id, group_id, *scope_args),
        )
        for i, u, f, tv, og, sup in cur.fetchall():
            if u and f:
                fact_by_uuid.setdefault(u, (i, f, tv, og, sup))
                hop_uuids.append(u)

    # 5 — RRF fuse the three ranked lists; take the top `limit`. The same fact text stored
    # as separate edges (same-batch twins: ~200 such groups were live on 2026-10-09) serves
    # once, and the next fact takes the freed slot.
    fused = _rrf_fuse([vec_uuids, bm25_uuids, hop_uuids])
    top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    results: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    for u, _ in top:
        if u not in fact_by_uuid:
            continue
        fact_id, fact, t_valid, ongoing, supported = fact_by_uuid[u]
        text_key = " ".join(fact.lower().split())
        if text_key in seen_text:
            continue
        seen_text.add(text_key)
        results.append(
            {
                "fact": fact,
                "_uuid": u,
                "_id": fact_id,
                "_date": t_valid,
                "_ongoing": ongoing,
                "_supported": supported,
            }
        )
        if len(results) >= limit:
            break
    return results, seed_entities
