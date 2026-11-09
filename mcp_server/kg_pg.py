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
GLOBAL live-fact HNSW on the BARE partial-index predicate (no owner/group -> stays
on HNSW), then filtering the tenant scope on that small candidate set. The two
GUCs force the planner onto the index. (The long-term multi-tenant-at-scale answer
is LIST partitioning by owner_id so partition pruning removes the filter entirely;
not needed while there is one real owner + throwaway DBs for isolated runs.)

Restricted surfaces (schema 053/054) pass ``allowed_projects``; every leg then serves
only facts whose cached provenance set (schema 056, ``source_projects``) is known,
non-empty and inside the allowlist and, with schema 066, only facts labelled work-safe or
unlabelled and older than 066 — see :func:`scope_predicate`. ``None`` is full trust and
runs the original SQL unchanged. The install owner's flagged entity (schema 066) is never
a hop seed, for any caller.

Full trust also reads the KG group it did not ask for (:func:`search_kg`), as a
superset: the requested group's facts exactly as a one-group read serves them, then a
few vector/BM25 matches from the other group.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ingestion.embedding import embed_dims
from ingestion.scope import active_groups

# Embedding width for the halfvec casts below — must match the provisioned schema
# (and its HNSW index expressions) verbatim. Default 2048 (Voyage prod, unchanged).
_EMBED_DIMS = embed_dims()

# RRF constant matches recall._rrf_fuse / bench_pg_kg (k=60, 1-indexed rank).
_RRF_K = 60
# Fact-vector over-fetch pool: how many global-nearest live facts to pull before
# applying the owner/group/web filter. For the technical group the outer LIMIT
# (limit*3) is what bites: across 20 real queries (2026-11-06) the fewest in-group,
# non-web survivors of the 200 was 46. The small personal group can under-fill with or
# without the web filter (3 of 20 personal queries kept fewer than 36).
_OVERFETCH = 200
# Facts a full-trust recall may add from the group it did not ask for, after the
# requested group's own (see search_kg). recall_overview serves _FACT_LIMIT + this.
CROSS_GROUP_EXTRAS = 4

# One fact-leg row: (uuid, fact, t_valid, ongoing, last_supported_at), the columns every
# fact-returning statement here selects, in that order.
FactRow = tuple[Any, ...]


def read_groups(group_id: str, restricted: bool) -> str | list[str]:
    """The KG group(s) a recall reads: the requested group alone (a plain string) for a
    restricted caller; for full trust a list, the requested group first.

    The extractor routes each fact into ``technical`` or ``personal`` by a name regex, so
    the owner's facts on one topic sit in both graphs, and reading only the requested one
    missed the other half: 48% of personal-group recalls over the 60 days to 2026-11-06
    served exactly one fact, against 6% of technical ones. The group is not a trust
    boundary (a restricted surface is scoped by fact provenance, schema 056), so full
    trust reads every group the deployment keeps (:func:`ingestion.scope.active_groups`:
    with the personal scope off that is one, and the plain string comes back as before).
    """
    if restricted:
        return group_id
    groups = [group_id, *(g for g in active_groups() if g != group_id)]
    return groups if len(groups) > 1 else group_id


def group_sql(group_id: str | list[str], col: str = "group_id") -> str:
    """``col`` filtered to ``group_id``, one ``%s`` bound to it as given: equality for one
    group (a string: the statement every single-group read always sent), ``= ANY`` for a
    list (:func:`read_groups`)."""
    return f"{col} = %s" if isinstance(group_id, str) else f"{col} = ANY(%s)"


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
    one written since, nor any unlabelled one when the stamp is missing. The cutover is
    read by an uncorrelated subquery (one InitPlan per statement), which keeps the
    ParadeDB BM25 TopN pushdown. ``fact_audience=False`` is the pre-066 rule, for a
    database without the columns (``recall_sources`` decides, by probe).

    One ``%s`` placeholder either way: the allowlist as a list.
    """
    p = f"{alias}." if alias else ""
    col = f"{p}source_projects"
    rule = f"{col} IS NOT NULL AND cardinality({col}) > 0 AND {col} <@ %s::text[]"
    if not fact_audience:
        return rule
    return (
        f"{rule} AND ({p}audience = 'work-safe' OR ({p}audience IS NULL AND {p}id <= "
        f"(SELECT value::bigint FROM synapse_meta WHERE key = '{FACT_AUDIENCE_CUTOVER_KEY}')))"
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
    group_id: str,
    session_focus: list[str],
    limit: int,
    allowed_projects: list[str] | None = None,
    vec_rows: list[FactRow] | None = None,
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
    executes exactly the statements it always did. Requires schema 056; the caller owns
    the column-missing fallback (recall_sources._search_kg).

    ``fact_audience`` (restricted only) adds schema 066's label rule to the predicate
    (:func:`scope_predicate`). It defaults on, so a caller that forgets it fails closed:
    on a database without 066 the statement raises UndefinedColumn and the restricted
    handler serves nothing. ``recall_sources`` passes False for such a database, by probe.

    ``exclude_seeds`` are the install owner's own entity uuids (schema 066, ``is_self``):
    never a hop seed, for any caller. Facts about the owner are still served by the vector
    and BM25 legs; the hop never starts from the owner node, whose thousands of edges
    would spend the hop leg on one seed. Empty (no flagged owner, or no schema 066) sends
    the seed statement exactly as before.

    ``vec_rows`` (full trust only) is step 1's result already in hand: this group's first
    ``limit * 3`` rows of a shared fact-vector walk (:func:`fact_vector_walk`,
    :func:`split_walk`), so the statement is not sent a second time for the other group.
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
    # partial-index predicate (no owner/group -> the planner keeps the kg_rel_hnsw
    # index instead of falling back to a 47K-row bitmap+sort), then filter the tenant
    # scope on the small candidate set. See module docstring for the required GUCs.
    vec_fetched: list[FactRow]
    if vec_rows is not None and not restricted:
        vec_fetched = vec_rows
    elif not restricted:
        cur.execute(
            "SELECT uuid, fact, t_valid, ongoing, last_supported_at FROM ("
            "  SELECT uuid, fact, t_valid, ongoing, last_supported_at, owner_id, group_id, "
            "         web_artifact_id "
            "  FROM kg_relationships "
            "  WHERE t_invalid IS NULL AND fact_embedding IS NOT NULL "
            f"  ORDER BY fact_embedding::halfvec({_EMBED_DIMS}) <=> %s::halfvec({_EMBED_DIMS}) LIMIT %s"
            ") sub WHERE owner_id = %s AND group_id = %s AND web_artifact_id IS NULL LIMIT %s",
            (emb_s, _OVERFETCH, owner_id, group_id, limit * 3),
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
            "    AND owner_id = %s AND group_id = %s AND web_artifact_id IS NULL "
            f"   AND {scope_predicate(fact_audience=fact_audience)}"
            ") SELECT uuid, fact, t_valid, ongoing, last_supported_at FROM cand "
            "ORDER BY dist LIMIT %s",
            (emb_s, owner_id, group_id, allowed_projects, limit * 3),
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
    # in the WHERE (before LIMIT), matching the FalkorDB fulltext leg.
    safe = "".join(c if (c.isalnum() or c.isspace()) else " " for c in query).strip()
    bm25_uuids: list[str] = []
    if safe:
        cur.execute(
            "SELECT uuid, fact, t_valid, ongoing, last_supported_at, paradedb.score(id) AS sc "
            "FROM kg_relationships "
            "WHERE id @@@ paradedb.match('fact', %s) "
            "  AND owner_id = %s AND group_id = %s AND t_invalid IS NULL "
            "  AND web_artifact_id IS NULL "
            f"{scope_sql}"
            "ORDER BY sc DESC LIMIT %s",
            (safe, owner_id, group_id, *scope_args, limit * 3),
        )
        for u, f, tv, og, sup, _sc in cur.fetchall():
            if not f:
                continue
            fact_by_uuid.setdefault(u, (f, tv, og, sup))
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
        f"{owner_sql}"
        "  ORDER BY dist LIMIT 25"
        "), deg AS ("
        "  SELECT u, count(*) AS d FROM ("
        "    SELECT src_uuid AS u FROM kg_relationships "
        "      WHERE owner_id = %s AND group_id = %s AND t_invalid IS NULL "
        "        AND web_artifact_id IS NULL "
        f"{scope_sql}"
        "        AND src_uuid IN (SELECT uuid FROM seeds) "
        "    UNION ALL "
        "    SELECT tgt_uuid AS u FROM kg_relationships "
        "      WHERE owner_id = %s AND group_id = %s AND t_invalid IS NULL "
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
            group_id,
            *owner_args,
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
            "  WHERE owner_id = %s AND group_id = %s AND t_invalid IS NULL "
            "    AND web_artifact_id IS NULL AND (src_uuid = %s OR tgt_uuid = %s) "
            f"{scope_sql}"
            ") SELECT uuid, fact, t_valid, ongoing, last_supported_at FROM cand "
            "ORDER BY dist ASC NULLS LAST, t_valid DESC LIMIT 8",
            (emb_s, owner_id, group_id, sd, sd, *scope_args),
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


def _safe_query(query: str) -> str:
    """``query`` with every non-alphanumeric, non-space character blanked: the BM25 leg's
    text, so identifiers and error strings survive (same rule as ``recall._search_kg``)."""
    return "".join(c if (c.isalnum() or c.isspace()) else " " for c in query).strip()


def fact_vector_walk(
    cur: Any, query_emb: list[float], owner_id: str, groups: list[str]
) -> list[FactRow]:
    """ONE walk of the global live-fact HNSW for every group in ``groups`` (step 1 of
    :func:`search_kg_postgres`, un-limited): the :data:`_OVERFETCH` nearest live facts,
    kept when they are this owner's, in one of ``groups`` and not from a web page, in
    distance order. Rows are :data:`FactRow` plus a trailing ``group_id``; hand them to
    :func:`split_walk`.

    The statement is the one each group's own fact-vector leg sent (same inner subquery,
    same filter-only outer, no ORDER BY outside it) with the group test widened to the
    list and the group selected. Two legs used to walk the same index for the same
    vector, once per group, back to back (~0.2s each on prod).
    """
    cur.execute(
        "SELECT uuid, fact, t_valid, ongoing, last_supported_at, group_id FROM ("
        "  SELECT uuid, fact, t_valid, ongoing, last_supported_at, owner_id, group_id, "
        "         web_artifact_id "
        "  FROM kg_relationships "
        "  WHERE t_invalid IS NULL AND fact_embedding IS NOT NULL "
        f"  ORDER BY fact_embedding::halfvec({_EMBED_DIMS}) <=> %s::halfvec({_EMBED_DIMS}) LIMIT %s"
        ") sub WHERE owner_id = %s AND group_id = ANY(%s) AND web_artifact_id IS NULL",
        (_vec_literal(query_emb), _OVERFETCH, owner_id, groups),
    )
    return cur.fetchall()


def split_walk(
    walk: list[FactRow], own_group: str, own_n: int, cross_groups: list[str], cross_n: int
) -> tuple[list[FactRow], list[FactRow]]:
    """The two fact-vector legs' rows out of one :func:`fact_vector_walk`: the first
    ``own_n`` rows of ``own_group`` and the first ``cross_n`` rows of ``cross_groups``, each
    in walk order, as :data:`FactRow`. This is what each leg's own ``LIMIT`` returned. The
    cut is on the raw rows, before any empty-fact skip, because the SQL ``LIMIT`` counted
    those rows too."""
    own: list[FactRow] = []
    cross: list[FactRow] = []
    for row in walk:
        group = row[5]
        if group == own_group:
            if len(own) < own_n:
                own.append(row[:5])
        elif group in cross_groups and len(cross) < cross_n:
            cross.append(row[:5])
        if len(own) >= own_n and len(cross) >= cross_n:
            break
    return own, cross


def cross_vec_rows(
    cur: Any, query_emb: list[float], owner_id: str, groups: list[str], k: int
) -> list[FactRow]:
    """The other groups' fact-vector leg on its own: the first ``k * 3`` survivors of one
    walk. :func:`search_kg` takes these from the shared walk instead."""
    walk = fact_vector_walk(cur, query_emb, owner_id, groups)
    return split_walk(walk, "", 0, groups, k * 3)[1]


def cross_bm25_rows(
    cur: Any, query: str, owner_id: str, groups: list[str], k: int
) -> list[FactRow]:
    """The other groups' BM25 leg: up to ``k * 3`` live non-web facts of ``groups`` matching
    ``query``, best score first. Needs nothing but the query text, so a caller can run it
    on its own connection while the requested group is searched (recall_sources).

    A caller that does so must give the connection the planner GUCs :func:`search_kg`
    documents; on a tie in score the ParadeDB scan's row order follows the plan.
    """
    safe = _safe_query(query)
    if k <= 0 or not groups or not safe:
        return []
    cur.execute(
        "SELECT uuid, fact, t_valid, ongoing, last_supported_at, paradedb.score(id) AS sc "
        "FROM kg_relationships "
        "WHERE id @@@ paradedb.match('fact', %s) "
        "  AND owner_id = %s AND group_id = ANY(%s) AND t_invalid IS NULL "
        "  AND web_artifact_id IS NULL "
        "ORDER BY sc DESC LIMIT %s",
        (safe, owner_id, groups, k * 3),
    )
    return [row[:5] for row in cur.fetchall()]


def cross_group_facts(
    cur: Any,
    query: str,
    query_emb: list[float],
    owner_id: str,
    groups: list[str],
    taken: list[dict[str, Any]],
    k: int,
    vec_rows: list[FactRow] | None = None,
    bm25_rows: list[FactRow] | None = None,
) -> list[dict[str, Any]]:
    """Up to ``k`` live facts from ``groups``, the groups a full-trust recall did not ask
    for: the fact-vector over-fetch and BM25 legs, RRF-fused, skipping any text already in
    ``taken``. Same row shape as :func:`search_kg_postgres`, each marked ``_cross``.

    No seed/hop leg: the other group's copy of the owner identity node (thousands of
    live edges) would spend the extras on one seed's hop sample. Full trust only: no
    provenance predicate.

    ``vec_rows`` / ``bm25_rows`` are a leg's rows already fetched (:func:`split_walk`, the
    shared walk; :func:`cross_bm25_rows`, run concurrently); a leg not given is run on
    ``cur``, so a call without them is the same two statements as ever.
    """
    if k <= 0 or not groups:
        return []
    if vec_rows is None:
        vec_rows = cross_vec_rows(cur, query_emb, owner_id, groups, k)
    if bm25_rows is None:
        bm25_rows = cross_bm25_rows(cur, query, owner_id, groups, k)
    fact_by_uuid: dict[str, tuple[str, Any, Any, Any]] = {}
    vec_uuids: list[str] = []
    for u, f, tv, og, sup in vec_rows:
        if f:
            fact_by_uuid.setdefault(u, (f, tv, og, sup))
            vec_uuids.append(u)
    bm25_uuids: list[str] = []
    for u, f, tv, og, sup in bm25_rows:
        if f:
            fact_by_uuid.setdefault(u, (f, tv, og, sup))
            bm25_uuids.append(u)
    fused = _rrf_fuse([vec_uuids, bm25_uuids])
    seen_text = {" ".join(t["fact"].lower().split()) for t in taken}
    out: list[dict[str, Any]] = []
    for u, _ in sorted(fused.items(), key=lambda kv: kv[1], reverse=True):
        fact, t_valid, ongoing, supported = fact_by_uuid[u]
        text_key = " ".join(fact.lower().split())
        if text_key in seen_text:
            continue
        seen_text.add(text_key)
        out.append(
            {
                "fact": fact,
                "_uuid": u,
                "_date": t_valid,
                "_ongoing": ongoing,
                "_supported": supported,
                "_cross": True,
            }
        )
        if len(out) >= k:
            break
    return out


def search_kg(
    cur: Any,
    query: str,
    query_emb: list[float],
    owner_id: str,
    group_id: str | list[str],
    session_focus: list[str],
    limit: int,
    allowed_projects: list[str] | None = None,
    cross_bm25: Callable[[], list[FactRow]] | None = None,
    fact_audience: bool = True,
    exclude_seeds: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """:func:`search_kg_postgres` over the group spec :func:`read_groups` returns.

    A string is one group, read exactly as it always was. A list (full trust) reads its
    first, requested, group exactly as a one-group read would, with every leg, then
    appends up to :data:`CROSS_GROUP_EXTRAS` facts from the rest (:func:`cross_group_facts`).
    The extras never compete for the requested group's ``limit`` slots: letting them
    compete (one fused top-12 over both groups) served 14 fewer of the 352 rated-helpful
    items on the 200 most recent rated recalls, a regression the bench rejects.

    The requested group's fact-vector leg and the extras' are one walk of the HNSW
    (:func:`fact_vector_walk`), split by group, and the extras' BM25 leg is a call away:
    ``cross_bm25`` returns its rows (:func:`cross_bm25_rows`, which the caller may be
    running on another connection already) and is called once, after the requested group
    is read; without it the leg runs on ``cur``. Nothing served changes: each leg returns
    the rows its own statement did.

    ``fact_audience`` and ``exclude_seeds`` go to every :func:`search_kg_postgres` call. The
    cross-group functions take neither: they serve full trust only (a restricted caller
    returns above, and :func:`read_groups` gives it one group) and have no hop leg.
    """
    if isinstance(group_id, str):
        return search_kg_postgres(
            cur,
            query,
            query_emb,
            owner_id,
            group_id,
            session_focus,
            limit,
            allowed_projects,
            fact_audience=fact_audience,
            exclude_seeds=exclude_seeds,
        )
    own_group, cross_groups = group_id[0], group_id[1:]
    if allowed_projects is not None:
        return search_kg_postgres(
            cur,
            query,
            query_emb,
            owner_id,
            own_group,
            session_focus,
            limit,
            allowed_projects,
            fact_audience=fact_audience,
            exclude_seeds=exclude_seeds,
        )
    walk = fact_vector_walk(cur, query_emb, owner_id, group_id)
    own_vec, cross_vec = split_walk(
        walk, own_group, limit * 3, cross_groups, CROSS_GROUP_EXTRAS * 3
    )
    facts, seeds = search_kg_postgres(
        cur,
        query,
        query_emb,
        owner_id,
        own_group,
        session_focus,
        limit,
        vec_rows=own_vec,
        fact_audience=fact_audience,
        exclude_seeds=exclude_seeds,
    )
    facts += cross_group_facts(
        cur,
        query,
        query_emb,
        owner_id,
        cross_groups,
        facts,
        CROSS_GROUP_EXTRAS,
        vec_rows=cross_vec,
        bm25_rows=cross_bm25() if cross_bm25 is not None else None,
    )
    return facts, seeds
