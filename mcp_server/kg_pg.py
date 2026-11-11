"""Postgres read-port of the KG retrieval leg (mcp_server/recall.py::_search_kg).

Queries the ``kg_entities`` / ``kg_relationships`` tables (schema/017) that the
extraction pipeline writes (the canonical KG store since #67 PR 3), scoped by ``owner_id``.
Returns the same ``(facts, seed_entities)`` shape ``recall._search_kg`` returns, so
it can be wired into recall behind a shadow-read flag and, eventually, become the
source of truth.

Retrieval mirrors ``_search_kg`` exactly:
  1. fact-embedding vector KNN over live edges (partial HNSW: t_invalid IS NULL)
  2. fact BM25 over edge text (ParadeDB)
  3. entity-seed vector KNN + live-degree gate + session-focus bonus -> top 8 seeds
  4. 1-hop traversal from each seed (per-seed LIMIT 8, the 8 edges nearest the query)
  5. RRF fuse [vec, bm25, hop], take top ``limit``

Every leg, and the seed degree gate, skips facts extracted from web pages
(``web_artifact_id`` set, schema 018). Web content is served by recall's own web bucket
under its own floor; as facts it rated 1 helpful / 47 noise over the 60 days to
2026-11-06, the worst slice of the fact bucket. The rows stay in place.

The caller's cursor MUST have these session GUCs set (see parity_kg_read.py):
  - ``hnsw.ef_search = 200``               (matches FalkorDB's efRuntime)
  - ``enable_seqscan = off``               } pgvector won't use the HNSW index when
  - ``max_parallel_workers_per_gather = 0``} an equality filter is also present --
the planner picks a bitmap/parallel-seq-scan on the owner_id btree and sorts 47K
rows (~1.6s) instead. The fact-vector leg sidesteps this by over-fetching the
GLOBAL live-fact HNSW on the BARE partial-index predicate (no owner -> stays
on HNSW), then filtering the tenant scope on that small candidate set. The two
GUCs force the planner onto the index. (The long-term multi-tenant-at-scale answer
is LIST partitioning by owner_id so partition pruning removes the filter entirely;
not needed while there is one real owner + throwaway DBs for isolated runs.)

Restricted surfaces (schema 053/054) pass ``allowed_projects``; every leg then serves
only facts whose cached provenance set (schema 056, ``source_projects``) is known,
non-empty and inside the allowlist and, with schema 066, only facts labelled work-safe or
unlabelled and older than 066 — see :func:`scope_predicate`. ``None`` is full trust and
adds no predicate to any statement. The install owner's flagged entity (schema 066) is never
a hop seed, for any caller.

The graph is one memory per owner. Rows written before schema 067 carry a legacy
``group_id`` (``technical`` or ``personal``); no statement here reads it, so a recall ranks
one list over every row the owner has.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ingestion.bm25_sql import bm25_filter
from ingestion.embedding import embed_dims

# Embedding width for the halfvec casts below — must match the provisioned schema
# (and its HNSW index expressions) verbatim. Default 2048 (Voyage prod, unchanged).
_EMBED_DIMS = embed_dims()

# RRF constant matches recall._rrf_fuse / bench_pg_kg (k=60, 1-indexed rank).
_RRF_K = 60
# Fact-vector over-fetch pool: how many global-nearest live facts to pull before
# applying the owner/web filter. The outer LIMIT (limit*3) is what bites: across 20 real
# queries (2026-11-06) the fewest in-scope, non-web survivors of the 200 was 46.
_OVERFETCH = 200

# One fact-leg row: (uuid, fact, t_valid, ongoing, last_supported_at), the columns every
# fact-returning statement here selects, in that order.
FactRow = tuple[Any, ...]


#: synapse_meta key schema 066 stamps once: the highest kg_relationships.id when 066 ran.
FACT_AUDIENCE_CUTOVER_KEY = "fact_audience_legacy_max_id"


def scope_predicate(alias: str = "", *, fact_audience: bool = True) -> str:
    """The restricted-surface serving rule over ``kg_relationships`` (schema 056, 066).

    Provenance (056): a fact is servable to a restricted caller iff its source-project
    set is KNOWN (``source_projects`` is NULL whenever any source episode is missing, has
    no project, or the fact has no episode provenance at all), NON-EMPTY, and a SUBSET of
    the caller's allowlist, so a fact with even one source outside the allowlist (mixed
    provenance) is never served.

    Sensitivity (066, ``fact_audience``): AND the fact is labelled ``work-safe``, or is
    unlabelled and existed before 066 ran (``id`` at or below the cutover 066 stamped in
    ``synapse_meta``). A ``personal`` fact is never servable; neither is an unlabelled
    one written since, nor any unlabelled one when the stamp is missing (the comparison
    is NULL, so the row is filtered). The cutover is read by an uncorrelated subquery (an
    InitPlan, run once per copy of the rule in a statement). ``fact_audience=False`` is the
    provenance-only rule (056) as a statement shape. A restricted read never gets it:
    ``recall_sources`` raises ``FactAudienceUnknown`` when the probe is not True, and that read
    is served nothing. Only full trust passes it (on a database without 066), and full trust
    builds no predicate, so there the flag changes nothing.

    The rule is ONE expression, not ``audience = 'work-safe' OR (audience IS NULL AND id
    <= cutover)``: pg_search turns a WHERE-level OR into a Tantivy boolean of "match
    everything" sub-queries, one per clause of the BM25 scan. On prod pg_search 0.23.2
    (2026-11-09) the OR read 38x the pages (441ms against 32ms) and added score to the
    unlabelled branch, so legacy facts outranked work-safe ones and the top-K moved. As
    one expression it is a single filter over the match's own hits, like the provenance
    rule. Shapes pg_search 0.23.2 rejects with "Unsupported query shape":
    ``coalesce(<boolean OR>, false)`` and ``CASE ... THEN <boolean>``.

    One ``%s`` placeholder either way: the allowlist as a list.
    """
    p = f"{alias}." if alias else ""
    col = f"{p}source_projects"
    rule = f"{col} IS NOT NULL AND cardinality({col}) > 0 AND {col} <@ %s::text[]"
    if not fact_audience:
        return rule
    return (
        f"{rule} AND coalesce({p}audience, CASE WHEN {p}id <= "
        f"(SELECT value::bigint FROM synapse_meta WHERE key = '{FACT_AUDIENCE_CUTOVER_KEY}') "
        "THEN 'work-safe' END) = 'work-safe'"
    )


def _rrf_fuse(lists: list[list[str]], k: int = _RRF_K) -> dict[str, float]:
    """RRF over ranked uuid lists: each list credits a uuid ONCE, at its first (best) rank.

    A list can name a uuid twice: the 1-hop leg returns a fact once per seed it touches,
    so an edge between two seeds sat at (say) hop ranks 4 and 6 and was credited for both.
    Later repeats are skipped in place, so every other uuid keeps the rank it had.
    """
    scores: dict[str, float] = {}
    for lst in lists:
        seen: set[str] = set()
        for rank, uuid_ in enumerate(lst):
            if uuid_ in seen:
                continue
            seen.add(uuid_)
            scores[uuid_] = scores.get(uuid_, 0.0) + 1.0 / (k + rank + 1)
    return scores


def _vec_literal(emb: list[float]) -> str:
    return "[" + ",".join(map(str, emb)) + "]"


def search_kg_postgres(
    cur: Any,
    query: str,
    query_emb: list[float],
    owner_id: str,
    session_focus: list[str],
    limit: int,
    allowed_projects: list[str] | None = None,
    fact_audience: bool = True,
    exclude_seeds: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Mirror of recall._search_kg over the Postgres KG mirror.

    Returns ``(facts, seed_entities)``: facts carry internal ``_uuid`` (for
    retrieval-count bumps + history lookup), seed_entities is the ranked list of
    connected seeds used by the entity bucket downstream.

    ``allowed_projects`` (a restricted surface's allowlist) applies
    :func:`scope_predicate` to every fact-returning leg and to the seed degree gate,
    and returns no seed entities: an entity's name and summary are distilled from ALL
    its facts, so nothing proves them confined to the allowlist. ``None`` (full trust)
    adds nothing to any statement. Requires schema 056; the caller owns
    the column-missing fallback (recall_sources._search_kg).

    ``fact_audience`` (restricted only) adds schema 066's label rule to the predicate
    (:func:`scope_predicate`). It defaults on, so a caller that forgets it fails closed:
    on a database without 066 the statement raises UndefinedColumn and the restricted
    handler serves nothing. A restricted read whose 066 probe is not True never reaches this
    function (``FactAudienceUnknown``, served nothing); full trust on a database without 066
    passes False, which changes nothing because full trust builds no predicate.

    ``exclude_seeds`` are the install owner's own entity uuids (schema 066, ``is_self``):
    never a hop seed, for any caller. Facts about the owner are still served by the vector
    and BM25 legs; the hop never starts from the owner node, whose thousands of edges
    would spend the hop leg on one seed. Empty (no flagged owner, or no schema 066) sends
    the seed statement exactly as before.
    """
    restricted = allowed_projects is not None
    # Spliced into the full-trust statements as "" so their text is unchanged; a
    # restricted call appends the predicate and its one bound allowlist parameter.
    scope_sql = f"  AND {scope_predicate(fact_audience=fact_audience)} " if restricted else ""
    scope_args: tuple[Any, ...] = (allowed_projects,) if restricted else ()
    # Likewise "" with no flagged owner: the seed statement then reads as it always did.
    owner_sql = "  AND uuid <> ALL(%s::text[]) " if exclude_seeds else ""
    owner_args: tuple[Any, ...] = (list(exclude_seeds),) if exclude_seeds else ()
    emb_s = _vec_literal(query_emb)
    # uuid -> (fact text, t_valid, ongoing, last_supported_at). t_valid = when the fact became
    # true (bitemporal valid-from), surfaced as the fact's "as-of" date; ongoing +
    # last_supported_at (schema 058) say whether the claim can lapse silently and when the
    # user last restated it, so the reader can tell "valid-from 2025, never confirmed since"
    # from "restated last week". Served facts are already live (t_invalid IS NULL filtered
    # everywhere below).
    fact_by_uuid: dict[str, tuple[str, Any, Any, Any]] = {}

    # 1 — fact-embedding vector KNN. Over-fetch the GLOBAL live-fact HNSW on the bare
    # partial-index predicate (no owner -> the planner keeps the kg_rel_hnsw
    # index instead of falling back to a 47K-row bitmap+sort), then filter the tenant
    # scope on the small candidate set. See module docstring for the required GUCs.
    vec_fetched: list[FactRow]
    if not restricted:
        cur.execute(
            "SELECT uuid, fact, t_valid, ongoing, last_supported_at FROM ("
            "  SELECT uuid, fact, t_valid, ongoing, last_supported_at, owner_id, "
            "         web_artifact_id "
            "  FROM kg_relationships "
            "  WHERE t_invalid IS NULL AND fact_embedding IS NOT NULL "
            f"  ORDER BY fact_embedding::halfvec({_EMBED_DIMS}) <=> %s::halfvec({_EMBED_DIMS}) LIMIT %s"
            ") sub WHERE owner_id = %s AND web_artifact_id IS NULL LIMIT %s",
            (emb_s, _OVERFETCH, owner_id, limit * 3),
        )
        vec_fetched = cur.fetchall()
    else:
        # Restricted: EXACT KNN over the allowlisted set. The global-HNSW over-fetch
        # above would under-return here — a selective filter applied to the 200 nearest
        # facts corpus-wide can leave few or none. The MATERIALIZED fence keeps the
        # planner off the HNSW ordering; the provenance filter rides the partial GIN
        # (kg_rel_source_projects_gin). Cost scales with the allowlisted fact count.
        cur.execute(
            "WITH cand AS MATERIALIZED ("
            "  SELECT uuid, fact, t_valid, ongoing, last_supported_at, "
            f"         fact_embedding::halfvec({_EMBED_DIMS}) <=> %s::halfvec({_EMBED_DIMS}) AS dist "
            "  FROM kg_relationships "
            "  WHERE t_invalid IS NULL AND fact_embedding IS NOT NULL "
            "    AND owner_id = %s AND web_artifact_id IS NULL "
            f"   AND {scope_predicate(fact_audience=fact_audience)}"
            ") SELECT uuid, fact, t_valid, ongoing, last_supported_at FROM cand "
            "ORDER BY dist LIMIT %s",
            (emb_s, owner_id, allowed_projects, limit * 3),
        )
        vec_fetched = cur.fetchall()
    vec_uuids: list[str] = []
    for u, f, tv, og, sup in vec_fetched:
        if not f:
            continue
        fact_by_uuid.setdefault(u, (f, tv, og, sup))
        vec_uuids.append(u)

    # 2 — BM25 full-text over fact text (ParadeDB). Same alnum/space sanitize as
    # recall._search_kg so identifiers/error strings survive. t_invalid filtered
    # in the WHERE (before LIMIT), matching the FalkorDB fulltext leg. The filter is one
    # expression (bm25_filter): pg_search checks each WHERE conjunct as its own pass over
    # the hits, so the restricted predicate rides inside it rather than as scope_sql.
    safe = "".join(c if (c.isalnum() or c.isspace()) else " " for c in query).strip()
    bm25_uuids: list[str] = []
    if safe:
        bm25_where = ["owner_id = %s", "t_invalid IS NULL", "web_artifact_id IS NULL"]
        if restricted:
            bm25_where.append(scope_predicate(fact_audience=fact_audience))
        cur.execute(
            "SELECT uuid, fact, t_valid, ongoing, last_supported_at, paradedb.score(id) AS sc "
            "FROM kg_relationships "
            "WHERE id @@@ paradedb.match('fact', %s)"
            f"{bm25_filter(bm25_where)} "
            "ORDER BY sc DESC LIMIT %s",
            (safe, owner_id, *scope_args, limit * 3),
        )
        for u, f, tv, og, sup, _sc in cur.fetchall():
            if not f:
                continue
            fact_by_uuid.setdefault(u, (f, tv, og, sup))
            bm25_uuids.append(u)

    # 3 — entity seed vector KNN (top 25) + live-degree gate + focus bonus -> top 8.
    # The seed KNN uses the SAME bare-predicate over-fetch trick as the fact leg
    # (step 1): an owner equality filter directly alongside the vector ORDER BY
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
        "    SELECT uuid, name, summary, owner_id, "
        f"           embedding::halfvec({_EMBED_DIMS}) <=> %s::halfvec({_EMBED_DIMS}) AS dist "
        "    FROM kg_entities WHERE embedding IS NOT NULL "
        f"    ORDER BY embedding::halfvec({_EMBED_DIMS}) <=> %s::halfvec({_EMBED_DIMS}) LIMIT %s"
        "  ) e WHERE owner_id = %s "
        f"{owner_sql}"
        "  ORDER BY dist LIMIT 25"
        "), deg AS ("
        "  SELECT u, count(*) AS d FROM ("
        "    SELECT src_uuid AS u FROM kg_relationships "
        "      WHERE owner_id = %s AND t_invalid IS NULL "
        "        AND web_artifact_id IS NULL "
        f"{scope_sql}"
        "        AND src_uuid IN (SELECT uuid FROM seeds) "
        "    UNION ALL "
        "    SELECT tgt_uuid AS u FROM kg_relationships "
        "      WHERE owner_id = %s AND t_invalid IS NULL "
        "        AND web_artifact_id IS NULL "
        f"{scope_sql}"
        "        AND tgt_uuid IN (SELECT uuid FROM seeds) "
        "  ) z GROUP BY u"
        ") "
        "SELECT s.uuid, s.name, s.summary, s.dist, COALESCE(d.d, 0) AS deg "
        "FROM seeds s LEFT JOIN deg d ON d.u = s.uuid ORDER BY s.dist",
        (
            emb_s,
            emb_s,
            _OVERFETCH,
            owner_id,
            *owner_args,
            owner_id,
            *scope_args,
            owner_id,
            *scope_args,
        ),
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
    seed_entities = (
        [] if restricted else [{"uuid": u, "name": n, "summary": s} for _, u, n, s in connected]
    )

    # 4 — 1-hop traversal facts from the seeds (per-seed LIMIT 8, undirected, the 8 edges
    # nearest the query).
    hop_uuids: list[str] = []
    for sd in seed_uuids:
        cur.execute(
            # The 8 are the seed's edges whose fact embedding is nearest the query (exact cosine).
            # They were the 8 newest (2026-08-07), and arbitrary before that, so a supernode
            # seed (the User node, live degree in the thousands) offered the same 8 edges
            # whatever was asked, and hop-only facts rarely cleared the fact floor. The
            # MATERIALIZED fence is defensive, not required: the only HNSW index (kg_rel_hnsw)
            # is on fact_embedding::halfvec and partial on live, embedded edges, so no index
            # can serve this ORDER BY today; the fence keeps a future one from cutting the
            # filtered set short. An edge with no embedding stays eligible and ranks after
            # every embedded one. Equal distances and those edges fall to t_valid DESC, which
            # puts a NULL valid-from first, as the old query did.
            # Cost is one distance per live edge of the seed, on the stored full-precision
            # vector (the halfvec cast the HNSW legs need is about 3x slower here, because it
            # converts every row). Every candidate's 2048-d vector is an out-of-line TOAST
            # read the HNSW legs never warm, so a cold seed pays for it: a read-only EXPLAIN
            # ANALYZE on prod put the highest-degree seed (2,606 live edges) at 25 ms warm,
            # against 4 ms warm / 13 ms cold for the old recency statement, and a 1,625-edge
            # seed at 383 ms cold, then 37 ms. Only 25 of 45.8k entities have more than 128
            # live edges (median 2).
            "WITH cand AS MATERIALIZED ("
            "  SELECT uuid, fact, t_valid, ongoing, last_supported_at, "
            "         fact_embedding <=> %s::vector AS dist "
            "  FROM kg_relationships "
            "  WHERE owner_id = %s AND t_invalid IS NULL "
            "    AND web_artifact_id IS NULL AND (src_uuid = %s OR tgt_uuid = %s) "
            f"{scope_sql}"
            ") SELECT uuid, fact, t_valid, ongoing, last_supported_at FROM cand "
            "ORDER BY dist ASC NULLS LAST, t_valid DESC LIMIT 8",
            (emb_s, owner_id, sd, sd, *scope_args),
        )
        for u, f, tv, og, sup in cur.fetchall():
            if u and f:
                fact_by_uuid.setdefault(u, (f, tv, og, sup))
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
        fact, t_valid, ongoing, supported = fact_by_uuid[u]
        text_key = " ".join(fact.lower().split())
        if text_key in seen_text:
            continue
        seen_text.add(text_key)
        results.append(
            {
                "fact": fact,
                "_uuid": u,
                "_date": t_valid,
                "_ongoing": ongoing,
                "_supported": supported,
            }
        )
        if len(results) >= limit:
            break
    return results, seed_entities
