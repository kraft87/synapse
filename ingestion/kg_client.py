"""Postgres-backed knowledge-graph client for the extraction pipeline (#67 PR 3).

The pipeline reads and writes the graph through this one class: ``kg_entities`` /
``kg_relationships`` (schema/017), scoped by ``owner_id`` + ``group_id``. An instance owns
ONE connection, opened on first use with the session options the pool's connections get
(``hnsw.ef_search``); the poller builds one per drain thread, since a connection is not
shared between threads. Every method RAISES on failure: Postgres is the source of truth,
so a failed read or write fails the extraction item and the queue retries it. A
connection error is retried once on a fresh connection (the server restarted under an
idle worker). That is safe because every statement is a read or idempotent (``ON
CONFLICT`` upsert/no-op, ``UPDATE`` by uuid).

Return shapes are contracts the callers depend on (they feed prompts and ``rrf_merge``):
``valid_at`` comes back as an ISO string and the vector ``score`` as cosine distance
(0 = identical) -- the scale the stage-6a ``distance_threshold=0.20`` gate expects.

The ParadeDB BM25 leg intentionally has NO score floor: the pool is capped by the caller
(``_SEMANTIC_POOL_LIMIT``), rank-only RRF (k=1) does the weighting, and stage 6b's LLM
confirm gates every candidate.

Vector KNN uses the same planner discipline as ``mcp_server/kg_pg.py``: over-fetch the
GLOBAL HNSW on the bare partial-index predicate (owner/group equality filters push the
planner off the index onto a bitmap+sort), then filter tenant scope on the small
candidate set. The planner also costs a seq scan as cheap (it can't see the per-row
detoast + halfvec-cast + distance work) and skips the HNSW index: ~2.5s/lookup at 44K
entities vs ~10ms indexed. The ANN statements run in a transaction that forces the index
with ``SET LOCAL enable_seqscan = off``.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import psycopg

from ingestion import pg_pool
from ingestion.db_connection import bm25_query_text
from ingestion.embedding import embed_dims
from ingestion.textsafe import strip_nul

logger = logging.getLogger(__name__)

# Embedding width for the halfvec/vector casts below -- must match the provisioned schema
# (and its HNSW index expressions) verbatim. Default 2048 (Voyage prod, unchanged).
_EMBED_DIMS = embed_dims()

# Single-owner constant today; #49 owner axis threads a real owner through env.
OWNER = os.environ.get("SYNAPSE_KG_OWNER_ID", "default")

# Over-fetch pool for global-HNSW-then-filter (see module docstring / kg_pg.py).
_OVERFETCH = 200
# The same over-fetch for the Stage 6c restatement lookup, one HNSW scan per fact. A
# restatement sits within cosine distance 0.10, among the very nearest live edges of any
# scope, so a short pool is enough and keeps the per-fact heap fetches cheap.
_RESTATED_OVERFETCH = 32


def _vec(v: Any) -> str | None:
    """Format an embedding list for pgvector text input ('[a,b,c]'), or None."""
    if not v:
        return None
    return "[" + ",".join(map(str, v)) + "]"


def _iso(v: Any) -> str | None:
    """TIMESTAMPTZ -> ISO string, mirroring FalkorDB's string-property reads."""
    return v.isoformat() if v is not None else None


def _ts(v: Any) -> str | None:
    """ISO-string timestamp, or None for absent/garbage values.

    The edge-date extractor occasionally emits out-of-range years
    (e.g. ``'-4599999974-05-27T00:00:00Z'``) that Postgres TIMESTAMPTZ rejects --
    null those rather than fail the whole write.
    """
    if not v:
        return None
    try:
        datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        return str(v)
    except (ValueError, TypeError):
        return None


def rrf_merge(
    vector_hits: list[dict[str, Any]],
    fulltext_hits: list[dict[str, Any]],
    limit: int = 20,
    k: int = 1,
) -> list[dict[str, Any]]:
    """Rank-based reciprocal rank fusion of two candidate lists.

    Score per uuid = sum over lists of 1/(k + rank), rank 0-indexed.
    Default k=1 matches Graphiti's `rrf()` — steeper than the Cormack k=60,
    appropriate when top-1 dominance matters (contradiction detection).

    Truncation happens AFTER the merge — the long tail is what RRF exists
    to rescue, so do not pre-truncate the input lists. Pass the FULL pool
    from each retrieval modality and let RRF promote items that scored
    moderately well in both.

    Dedupe by uuid; preserve the payload from the first list that yielded it.
    """
    scores: dict[str, float] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for hits in (vector_hits, fulltext_hits):
        for rank, item in enumerate(hits):
            uid = item.get("uuid")
            if not uid:
                continue
            scores[uid] = scores.get(uid, 0.0) + 1.0 / (k + rank)
            payloads.setdefault(uid, item)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [payloads[uid] for uid, _ in ordered[:limit]]


class KGClient:
    def __init__(self, db_url: str) -> None:
        self._db_url = db_url
        self._conn: psycopg.Connection[Any] | None = None

    # -- connection management ------------------------------------------------
    def _reset(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            pass
        self._conn = None

    def _run[T](self, fn: Callable[[psycopg.Connection[Any]], T]) -> T:
        """Run ``fn(connection)`` on this client's connection, opened on first use.

        Any error drops the connection so the next call reconnects instead of reusing one
        a failed statement or a restarted server left unusable. A connection error gets
        one retry on a fresh connection, so a Postgres restart under an idle worker
        costs the call a reconnect, not the item a failed attempt."""
        for attempt in (1, 2):
            if self._conn is None:
                self._conn = pg_pool.connect(self._db_url)
            try:
                return fn(self._conn)
            except psycopg.OperationalError:
                self._reset()
                if attempt == 2:
                    raise
            except Exception:
                self._reset()
                raise
        raise AssertionError("unreachable")  # pragma: no cover

    # ------------------------------------------------------------------
    # Entity reads
    # ------------------------------------------------------------------

    def find_similar_nodes(
        self, query_embedding: list[float], group_id: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Top-k entities by embedding distance. Score is cosine distance (0=identical)."""
        emb_s = _vec(query_embedding)

        def _q(conn: psycopg.Connection[Any]) -> list[dict[str, Any]]:
            with conn.transaction():
                cur = conn.cursor()
                cur.execute("SET LOCAL enable_seqscan = off")
                cur.execute(
                    "SELECT uuid, name, dist FROM ("  # nosec B608 — _EMBED_DIMS is a validated int, not user input
                    "  SELECT uuid, name, owner_id, group_id, "
                    f"         embedding::halfvec({_EMBED_DIMS}) <=> %s::halfvec({_EMBED_DIMS}) AS dist "
                    "  FROM kg_entities WHERE embedding IS NOT NULL "
                    "  ORDER BY dist LIMIT %s"
                    ") sub WHERE owner_id = %s AND group_id = %s LIMIT %s",
                    (emb_s, _OVERFETCH, OWNER, group_id, limit),
                )
                return [{"uuid": u, "name": n, "score": float(d)} for u, n, d in cur.fetchall()]

        return self._run(_q)

    def entity_uuid_by_normalized_name(self, normalized: str, group_id: str) -> str | None:
        """Exact normalized-name lookup (NodeDeduper short-circuit).

        Falls back to a lower(trim(name)) equality check for pre-migration
        rows that lack the indexed property.
        """

        def _q(conn: psycopg.Connection[Any]) -> str | None:
            row = conn.execute(
                "SELECT uuid FROM kg_entities "
                "WHERE owner_id = %s AND group_id = %s AND normalized_name = %s LIMIT 1",
                (OWNER, group_id, normalized),
            ).fetchone()
            if row:
                return str(row[0])
            # Pre-migration nodes without the property: compute on the fly (rare;
            # mirrors the FalkorDB fallback).
            row = conn.execute(
                "SELECT uuid FROM kg_entities "
                "WHERE owner_id = %s AND group_id = %s AND lower(trim(name)) = %s LIMIT 1",
                (OWNER, group_id, normalized),
            ).fetchone()
            return str(row[0]) if row else None

        return self._run(_q)

    def load_type_map(self) -> dict[str, str]:
        """subtype -> supertype from kg_entity_types (the editable taxonomy map).

        Used by entity dedup to map a new entity's extracted type to its
        supertype for the type-compatibility candidate gate. Empty dict if the
        table is unseeded (gate then no-ops — every entity reads as untyped)."""

        def _q(conn: psycopg.Connection[Any]) -> dict[str, str]:
            rows = conn.execute("SELECT subtype, supertype FROM kg_entity_types").fetchall()
            return {str(sub): str(sup) for sub, sup in rows}

        return self._run(_q)

    def load_entities(self, group_id: str) -> list[tuple[str, str, str | None]]:
        """(uuid, name, entity_supertype) for every entity in the group (LSH hydrate)."""

        def _q(conn: psycopg.Connection[Any]) -> list[tuple[str, str, str | None]]:
            rows = conn.execute(
                "SELECT uuid, COALESCE(name, ''), entity_supertype "
                "FROM kg_entities WHERE owner_id = %s AND group_id = %s",
                (OWNER, group_id),
            ).fetchall()
            return [(str(u), n, st) for u, n, st in rows]

        return self._run(_q)

    def entity_summaries(self, uuids: list[str], group_id: str) -> dict[str, str]:
        """uuid -> summary of the given entities (LSH candidates handed to the LLM confirm)."""
        if not uuids:
            return {}

        def _q(conn: psycopg.Connection[Any]) -> dict[str, str]:
            rows = conn.execute(
                "SELECT uuid, COALESCE(summary, '') FROM kg_entities "
                "WHERE owner_id = %s AND group_id = %s AND uuid = ANY(%s)",
                (OWNER, group_id, uuids),
            ).fetchall()
            return {str(u): s for u, s in rows}

        return self._run(_q)

    # ------------------------------------------------------------------
    # Edge reads (stage 6 dedup + contradiction pools)
    # ------------------------------------------------------------------

    def find_similar_edges(
        self,
        fact_embedding: list[float],
        group_id: str,
        distance_threshold: float = 0.20,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Live edges within cosine-distance threshold of the fact embedding.

        distance_threshold=0.20 corresponds to cosine similarity >= 0.80.
        """
        emb_s = _vec(fact_embedding)

        def _q(conn: psycopg.Connection[Any]) -> list[dict[str, Any]]:
            with conn.transaction():
                cur = conn.cursor()
                cur.execute("SET LOCAL enable_seqscan = off")
                cur.execute(
                    "SELECT uuid, fact, valid_at, dist FROM ("  # nosec B608 — _EMBED_DIMS is a validated int, not user input
                    "  SELECT uuid, fact, valid_at, owner_id, group_id, "
                    f"         fact_embedding::halfvec({_EMBED_DIMS}) <=> %s::halfvec({_EMBED_DIMS}) AS dist "
                    "  FROM kg_relationships "
                    "  WHERE t_invalid IS NULL AND fact_embedding IS NOT NULL "
                    "  ORDER BY dist LIMIT %s"
                    ") sub WHERE owner_id = %s AND group_id = %s AND dist <= %s LIMIT %s",
                    (emb_s, _OVERFETCH, OWNER, group_id, distance_threshold, limit),
                )
                return [
                    {"uuid": u, "fact": f, "valid_at": _iso(v), "score": float(d)}
                    for u, f, v, d in cur.fetchall()
                ]

        return self._run(_q)

    def nearest_live_edges(
        self,
        fact_embeddings: list[list[float]],
        group_id: str,
        max_distance: float,
        limit: int,
    ) -> list[list[dict[str, Any]]]:
        """Per embedding, up to ``limit`` live edges of the group within ``max_distance``
        (cosine), nearest first. Parallel to ``fact_embeddings``.

        One statement for the whole batch (Stage 6c restatement recheck): ``unnest``
        feeds each embedding into a LATERAL HNSW scan over the bare partial-index
        predicate, and the tenant scope and distance bound filter that scan's short
        over-fetch (same discipline as ``find_similar_edges``)."""
        if not fact_embeddings:
            return []

        def _q(conn: psycopg.Connection[Any]) -> list[list[dict[str, Any]]]:
            with conn.transaction():
                cur = conn.cursor()
                cur.execute("SET LOCAL enable_seqscan = off")
                cur.execute(
                    "SELECT q.i, n.uuid, n.fact, n.dist "  # nosec B608 — _EMBED_DIMS is a validated int, not user input
                    "FROM unnest(%s::text[]) WITH ORDINALITY AS q(emb, i) "
                    "CROSS JOIN LATERAL ("
                    "  SELECT uuid, fact, dist FROM ("
                    "    SELECT uuid, fact, owner_id, group_id, "
                    f"           fact_embedding::halfvec({_EMBED_DIMS}) <=> q.emb::halfvec({_EMBED_DIMS}) AS dist "
                    "    FROM kg_relationships "
                    "    WHERE t_invalid IS NULL AND fact_embedding IS NOT NULL "
                    "    ORDER BY dist LIMIT %s"
                    "  ) c WHERE owner_id = %s AND group_id = %s AND dist <= %s "
                    "  ORDER BY dist LIMIT %s"
                    ") n ORDER BY q.i, n.dist",
                    (
                        [_vec(e) for e in fact_embeddings],
                        _RESTATED_OVERFETCH,
                        OWNER,
                        group_id,
                        max_distance,
                        limit,
                    ),
                )
                out: list[list[dict[str, Any]]] = [[] for _ in fact_embeddings]
                for i, u, f, d in cur.fetchall():
                    out[i - 1].append({"uuid": u, "fact": f, "score": float(d)})
                return out

        return self._run(_q)

    def find_edges_by_pair(
        self,
        source_uuid: str,
        target_uuid: str,
        group_id: str,
        fact_embedding: list[float],
        limit: int,
    ) -> list[dict[str, Any]]:
        """The ``limit`` live edges between a source/target pair nearest to ``fact_embedding``
        (duplicate/contradiction candidates), nearest first; an edge with no embedding
        comes last.

        Live only: already-invalidated edges must NOT re-enter the candidate pool, or
        Stage 6b could re-contradict an edge that was already retired. The pool goes whole
        into the 6b prompt, so it is capped, and an exact duplicate is the nearest edge."""

        def _q(conn: psycopg.Connection[Any]) -> list[dict[str, Any]]:
            rows = conn.execute(
                "SELECT uuid, fact, valid_at FROM kg_relationships "  # nosec B608 — _EMBED_DIMS is a validated int, not user input
                "WHERE owner_id = %s AND group_id = %s "
                "  AND src_uuid = %s AND tgt_uuid = %s AND t_invalid IS NULL "
                f"ORDER BY fact_embedding::halfvec({_EMBED_DIMS}) <=> %s::halfvec({_EMBED_DIMS}) "
                "  NULLS LAST, id "
                "LIMIT %s",
                (OWNER, group_id, source_uuid, target_uuid, _vec(fact_embedding), limit),
            ).fetchall()
            return [{"uuid": u, "fact": f, "valid_at": _iso(v)} for u, f, v in rows]

        return self._run(_q)

    def find_edges_by_fulltext(
        self,
        query: str,
        group_id: str,
        limit: int = 20,
        fact_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """BM25 candidates over live fact text (ParadeDB, OR semantics).

        No score floor: rank-only RRF (k=1) plus the stage-6b LLM confirm gate the pool
        instead. With ``fact_embedding``, each hit also carries ``dist``: the cosine
        distance from its stored embedding to that vector (None when the edge has no
        embedding), so a caller can tell a lexical hit on a shared identifier from one on
        the same topic."""
        safe = bm25_query_text(query)
        if not safe:
            return []
        args: tuple[Any, ...]
        if fact_embedding is None:
            dist_sql, args = "NULL", (safe, OWNER, group_id, limit)
        else:
            dist_sql = f"fact_embedding::halfvec({_EMBED_DIMS}) <=> %s::halfvec({_EMBED_DIMS})"
            args = (_vec(fact_embedding), safe, OWNER, group_id, limit)

        def _q(conn: psycopg.Connection[Any]) -> list[dict[str, Any]]:
            rows = conn.execute(
                f"SELECT uuid, fact, valid_at, paradedb.score(id) AS sc, {dist_sql} AS dist "  # nosec B608 — _EMBED_DIMS is a validated int, not user input
                "FROM kg_relationships "
                "WHERE id @@@ paradedb.match('fact', %s) "
                "  AND owner_id = %s AND group_id = %s AND t_invalid IS NULL "
                "ORDER BY sc DESC LIMIT %s",
                args,
            ).fetchall()
            return [
                {
                    "uuid": u,
                    "fact": f,
                    "valid_at": _iso(v),
                    "score": float(sc),
                    "dist": None if d is None else float(d),
                }
                for u, f, v, sc, d in rows
            ]

        return self._run(_q)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def insert_node(
        self,
        node_uuid: str,
        name: str,
        entity_type: str,
        summary: str,
        group_id: str,
        project: str | None = None,
        embedding: list[float] | None = None,
        supertype: str | None = None,
    ) -> str:
        """Insert a NEW entity. Returns the node UUID.

        ``normalized_name`` (lowercase, whitespace-collapsed, leading/trailing
        punctuation stripped) is what ``ingestion.dedup.NodeDeduper`` serves the
        exact-name short-circuit from, via an indexed lookup. entity_subtype = the raw
        entity_type; supertype is the caller-derived taxonomy rollup. Re-inserting an
        existing uuid is a retry no-op; a mention that matched an existing entity goes
        through :meth:`update_node`, which never touches its identity.
        """
        from ingestion.dedup import _normalize_name

        now = datetime.now(UTC)
        params = (
            node_uuid,
            OWNER,
            group_id,
            project or "",
            strip_nul(name or ""),
            strip_nul(_normalize_name(name) or ""),
            entity_type or "",
            strip_nul(summary or ""),
            _vec(embedding),
            now,
            now,
            now,
            now,
            supertype,
            entity_type or None,
        )

        def _do(conn: psycopg.Connection[Any]) -> None:
            conn.execute(
                "INSERT INTO kg_entities "  # nosec B608 — _EMBED_DIMS is a validated int, not user input
                "  (uuid, owner_id, group_id, project, name, normalized_name, "
                "   entity_type, summary, embedding, created_at, t_created, "
                "   valid_at, t_valid, entity_supertype, entity_subtype) "
                f"VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::vector({_EMBED_DIMS}),%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (uuid) DO NOTHING",
                params,
            )

        self._run(_do)
        return node_uuid

    def update_node(self, node_uuid: str, summary: str = "", supertype: str | None = None) -> None:
        """What a mention that matched an existing entity may change on it: the summary,
        when the mention's is longer, and the supertype, when the taxonomy map is loaded
        (``supertype`` not None; the map is fluid, so it propagates). The name,
        normalized_name, embedding and subtype are whatever the entity was first written
        with: a fuzzy match from "Postgres" must not rename "PostgreSQL", and every
        rewrite of the 8 KB embedding costs a new HNSW tuple. A mention that adds nothing
        writes nothing."""
        new_summary = strip_nul(summary or "").strip()

        def _do(conn: psycopg.Connection[Any]) -> None:
            conn.execute(
                "UPDATE kg_entities SET "
                "  summary = CASE WHEN length(%(s)s::text) > length(COALESCE(summary, '')) "
                "                 THEN %(s)s ELSE summary END, "
                "  entity_supertype = COALESCE(%(st)s, entity_supertype) "
                "WHERE uuid = %(u)s AND owner_id = %(o)s "
                "  AND (length(%(s)s::text) > length(COALESCE(summary, '')) "
                "       OR entity_supertype IS DISTINCT FROM COALESCE(%(st)s, entity_supertype))",
                {"s": new_summary, "st": supertype, "u": node_uuid, "o": OWNER},
            )

        self._run(_do)

    def create_edges_batch(self, rows: list[dict[str, Any]], group_id: str) -> None:
        """Insert relationship edges. One multi-row INSERT, ON CONFLICT (uuid)
        DO NOTHING (re-creating an existing edge is a retry no-op).

        Each row carries: src, tgt, edge_uuid, name, fact, episodes,
        created_at, t_created, valid_at, t_valid, optional emb (fact embedding),
        optional web_artifact_id (web-lane provenance, task #68), and the optional
        schema-058 evidence fields ongoing / last_supported_at / last_supported_by
        (absent or None -> NULL: unknown, never-confirmed).

        Edges are never born invalid: ``t_invalid`` is set only by a later fact
        superseding them (``invalidate_edges_batch``).
        """
        if not rows:
            return

        params: list[tuple[Any, ...]] = []
        for r in rows:
            created = _ts(r.get("created_at")) or _ts(r.get("t_created"))
            valid = _ts(r.get("valid_at")) or created
            t_valid = _ts(r.get("t_valid")) or _ts(r.get("valid_at")) or created
            eps = r.get("episodes")
            params.append(
                (
                    r["edge_uuid"],
                    OWNER,
                    group_id,
                    r["src"],
                    r["tgt"],
                    strip_nul(r.get("name") or ""),
                    strip_nul(r.get("fact") or ""),
                    _vec(r.get("emb")),
                    json.dumps(eps) if eps is not None else None,
                    0,
                    created,
                    _ts(r.get("t_created")) or created,
                    valid,
                    t_valid,
                    r.get("web_artifact_id"),
                    r.get("ongoing"),
                    _ts(r.get("last_supported_at")),
                    r.get("last_supported_by"),
                )
            )

        def _do(conn: psycopg.Connection[Any]) -> None:
            # Constant parameterized statement + executemany: psycopg3 pipelines
            # the rows, and a fixed query string keeps every value bound.
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO kg_relationships "  # nosec B608 — _EMBED_DIMS is a validated int, not user input
                    "(uuid, owner_id, group_id, src_uuid, tgt_uuid, name, fact, "
                    "fact_embedding, episodes, retrieval_count, created_at, t_created, "
                    "valid_at, t_valid, web_artifact_id, ongoing, last_supported_at, "
                    "last_supported_by) VALUES "
                    f"(%s,%s,%s,%s,%s,%s,%s,%s::vector({_EMBED_DIMS}),%s::jsonb,%s,%s,%s,%s,%s,%s,"
                    "%s,%s,%s) "
                    "ON CONFLICT (uuid) DO NOTHING",
                    params,
                )

        self._run(_do)

    def invalidate_edges_batch(
        self,
        items: list[tuple[str, str | None]],
        group_id: str,
        invalidated_by: str | None = None,
    ) -> None:
        """Set ``t_invalid`` (and the legacy ``invalid_at``) on each edge.
        ``items`` is a list of (edge_uuid, invalid_at); a None invalid_at falls back to
        now() per-row, so callers can mix "invalidate as of the fact's evidenced date"
        rows with "as of right now" rows in one call. ``invalidated_by`` (the superseding
        edge's uuid) applies to every item in the call and is set only when provided;
        COALESCE preserves any existing value when None (schema 028).

        The effective time is clamped to no earlier than the edge's own
        ``t_valid``: a correction dated before a stale edge was ever written
        retires it as of the moment it became live, never with a negative
        lifetime. An already-retired edge keeps its earlier ``t_invalid``."""
        if not items:
            return

        def _do(conn: psycopg.Connection[Any]) -> None:
            for edge_uuid, inv in items:
                ts = _ts(inv)
                conn.execute(
                    "UPDATE kg_relationships "
                    "SET t_invalid = COALESCE(t_invalid, GREATEST(t_valid, COALESCE(%s, now()))), "
                    "    invalid_at = COALESCE(t_invalid, GREATEST(t_valid, COALESCE(%s, now()))), "
                    "    invalidated_by = COALESCE(%s, invalidated_by) "
                    "WHERE uuid = %s",
                    (ts, ts, invalidated_by, edge_uuid),
                )

        self._run(_do)

    def reinforce_edges(self, items: list[tuple[Any, ...]], group_id: str) -> None:
        """Capture dedup hits: a newly-extracted fact restated an existing edge.

        For each ``(edge_uuid, source_episode_ids[, support_at, support_episode])``:
        UNION the new source episodes into the edge's ``episodes`` (provenance) and
        bump ``mention_count`` (the clean re-assertion frequency signal). Idempotent
        on re-processing — mention_count only increments when the episodes carry
        genuinely NEW provenance (``episodes`` doesn't already contain them), so
        re-extracting the same chunk can't double-count. Empty episode list is a
        per-item no-op; the DEFAULT-1 on create already counts the first assertion.

        Schema 058: a non-None ``support_at`` (the caller passes it only for a
        USER-attributed re-assertion) moves ``last_supported_at`` forward to it —
        never backward, and only under the same new-provenance condition, so a
        replayed chunk cannot refresh currency. ``last_supported_by`` follows it.
        """
        if not items:
            return

        def _do(conn: psycopg.Connection[Any]) -> None:
            for item in items:
                edge_uuid, eps = item[0], item[1]
                support_at = _ts(item[2]) if len(item) > 2 else None
                support_by = item[3] if len(item) > 3 else None
                if not eps:
                    continue
                eps_json = json.dumps(eps)
                conn.execute(
                    "UPDATE kg_relationships SET "
                    "  episodes = (SELECT jsonb_agg(DISTINCT e) FROM "
                    "      jsonb_array_elements(COALESCE(episodes, '[]'::jsonb) || %s::jsonb) e), "
                    "  mention_count = mention_count + CASE "
                    "      WHEN COALESCE(episodes, '[]'::jsonb) @> %s::jsonb THEN 0 ELSE 1 END, "
                    "  last_supported_by = CASE "
                    "      WHEN %s::timestamptz IS NULL "
                    "        OR COALESCE(episodes, '[]'::jsonb) @> %s::jsonb "
                    "        OR (last_supported_at IS NOT NULL AND last_supported_at >= %s::timestamptz) "
                    "      THEN last_supported_by ELSE %s END, "
                    "  last_supported_at = CASE "
                    "      WHEN %s::timestamptz IS NULL "
                    "        OR COALESCE(episodes, '[]'::jsonb) @> %s::jsonb "
                    "      THEN last_supported_at "
                    "      ELSE GREATEST(last_supported_at, %s::timestamptz) END "
                    "WHERE uuid = %s",
                    (
                        eps_json,
                        eps_json,
                        support_at,
                        eps_json,
                        support_at,
                        support_by,
                        support_at,
                        eps_json,
                        support_at,
                        edge_uuid,
                    ),
                )

        self._run(_do)

    def delete_orphan_entities(self, min_age_hours: int = 24) -> int:
        """Delete entities that no edge, live or retired, references, once they are
        ``min_age_hours`` old. Returns the number deleted.

        Extraction writes a new entity only for a fact Stage 7 creates, but an entity can
        still end up with no edge: its edges were deleted with their episode, the item
        failed between the entity and edge writes, or it predates that rule (until
        2026-10 Stage 5 wrote entities before Stage 6/7 decided their facts). Orphans cost
        serving: the facts leg's seed KNN takes 25 entities and keeps those with live
        degree, so every orphan near the query burns a seed slot (19% of entities on prod
        by 2026-10-10). The age gate keeps a sweep from racing an in-flight extraction.
        The anti-join covers edges in any group (a few cross groups); the re-check at
        delete time is the indexed same-group probe, for an edge written mid-statement."""

        def _do(conn: psycopg.Connection[Any]) -> int:
            cur = conn.execute(
                "DELETE FROM kg_entities e USING ("
                "  SELECT e2.id FROM kg_entities e2 "
                "  LEFT JOIN (SELECT src_uuid AS u FROM kg_relationships "
                "             UNION SELECT tgt_uuid FROM kg_relationships) r ON r.u = e2.uuid "
                "  WHERE r.u IS NULL AND e2.created_at < now() - make_interval(hours => %s)"
                ") o "
                "WHERE e.id = o.id "
                "  AND NOT EXISTS (SELECT 1 FROM kg_relationships r WHERE r.owner_id = e.owner_id "
                "                  AND r.group_id = e.group_id AND r.src_uuid = e.uuid) "
                "  AND NOT EXISTS (SELECT 1 FROM kg_relationships r WHERE r.owner_id = e.owner_id "
                "                  AND r.group_id = e.group_id AND r.tgt_uuid = e.uuid)",
                (min_age_hours,),
            )
            return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0

        return self._run(_do)
