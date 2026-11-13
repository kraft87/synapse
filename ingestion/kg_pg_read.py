"""Postgres reads for the WRITE-side KG pipeline (#67 PR 2; sole backend since PR 3).

The extraction pipeline reads the graph in six places while writing it:

  - stage 3 entity resolution      -> ``find_similar_nodes``
  - stage 4 NodeDeduper            -> exact normalized-name lookup + LSH hydrate scan
  - stage 6a/6b contradiction      -> ``find_edges_by_pair``
  - stage 7 dedup candidate pool   -> ``find_edges_by_pair`` + ``find_similar_edges``
                                      + ``find_edges_by_fulltext``
  - stage 6c restatement recheck   -> ``nearest_live_edges``

This module serves all of them from the ``kg_entities`` / ``kg_relationships``
tables (schema/017), scoped by ``owner_id``, via
``kg_client.KGClient``. FalkorDB was decommissioned in #67 PR 3, so the
``SYNAPSE_KG_READ`` dispatch/fallback seam from PRs #128/#129 is gone: a read
failure raises and the extraction queue item retries.

Return shapes are contracts the callers depend on (they feed prompts and
``rrf_merge``): ``valid_at`` comes back as an ISO string, ``fact_embedding``
as ``list[float]``, vector ``score`` as cosine distance (0 = identical) — the
scale the stage-7 ``distance_threshold=0.20`` gate expects.

The ParadeDB BM25 leg intentionally has NO score floor: the pool is capped by
the caller (``_SEMANTIC_POOL_LIMIT``), rank-only RRF (k=1) does the weighting,
and stage 6b's LLM confirm gates every candidate. The old RediSearch leg's
``min_score=0.6`` floor never transferred (different score scale) — and that
leg returned ZERO results on all multi-word queries in prod (AND/phrase
semantics — see ab_kg_pg_quality.py), so this leg is strictly additive to the
candidate pool.

Vector KNN uses the same planner discipline as ``mcp_server/kg_pg.py``: over-fetch
the GLOBAL HNSW on the bare partial-index predicate (an owner equality filter
pushes the planner off the index onto a bitmap+sort), then filter tenant scope on the
small candidate set. The connection is lazy + autocommit, mirroring
``kg_shadow.PgKgShadowWriter``; per-statement ``hnsw.ef_search`` is set via
``SET LOCAL`` inside a transaction block.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import orjson
from psycopg.rows import class_row

from ingestion.bm25_sql import bm25_filter
from ingestion.embedding import embed_dims

logger = logging.getLogger(__name__)

# Embedding width for the halfvec casts below — must match the provisioned schema
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

#: Which entity answers when several rows share a name: the owner's flagged entity, then the
#: oldest. The id never changes, so the SQL lookup below, the in-memory name index
#: (``load_entities``, read by ``dedup``) and the one-memory merge (schema/067) pick the same
#: row for a name that exists twice. All three key a row the same way: by its stored
#: normalized_name, or by lower(trim(name)) when that is empty.
SURVIVOR_ORDER = "is_self DESC, id"


#: The name lookup, one statement: a stored-name match, or an old row with no stored name matched
#: by lower(trim(name)); ranked together. Both arms are index conditions on ``normalized_name``.
_ENTITY_BY_NAME_SQL = (
    "SELECT uuid FROM kg_entities WHERE owner_id = %s "
    "AND (normalized_name = %s "
    "     OR ((normalized_name IS NULL OR normalized_name = '') AND lower(trim(name)) = %s)) "
    "ORDER BY " + SURVIVOR_ORDER + " LIMIT 1"  # nosec B608 - a module constant
)


def _vec_literal(emb: list[float]) -> str:
    return "[" + ",".join(map(str, emb)) + "]"


def _iso(v: Any) -> str | None:
    """TIMESTAMPTZ -> ISO string, mirroring FalkorDB's string-property reads."""
    return v.isoformat() if v is not None else None


def _emb_list(v: Any) -> list[float] | None:
    """pgvector text ('[a,b,c]') -> list[float]; tolerates None/pre-parsed."""
    if v is None:
        return None
    if isinstance(v, list):
        return [float(x) for x in v]
    try:
        parsed = orjson.loads(str(v))
        return [float(x) for x in parsed]
    except Exception:
        return None


@dataclass(frozen=True, slots=True)
class EntityRow:
    """One entity of the owner, as the dedup name index hydrates it.

    ``normalized_name`` is '' for a row written before that column existed."""

    uuid: str
    name: str
    summary: str
    entity_supertype: str | None
    normalized_name: str


@dataclass(frozen=True, slots=True)
class _NearEntityRow:
    """An entity and its cosine distance to the query embedding."""

    uuid: str
    name: str | None
    dist: float


@dataclass(frozen=True, slots=True)
class _NearEdgeRow:
    """A live edge and its cosine distance to the fact embedding."""

    uuid: str
    fact: str | None
    valid_at: datetime | None
    dist: float


@dataclass(frozen=True, slots=True)
class _BatchEdgeRow:
    """A live edge near the ``i``-th embedding of a batch (``i`` counts from 1)."""

    i: int
    uuid: str
    fact: str | None
    dist: float


@dataclass(frozen=True, slots=True)
class _PairEdgeRow:
    """A live edge between one source and target, its embedding as pgvector text."""

    uuid: str
    fact: str | None
    valid_at: datetime | None
    fact_embedding: str | None


@dataclass(frozen=True, slots=True)
class _FulltextEdgeRow:
    """A BM25 hit: its score, and its distance to the caller's embedding when one was given."""

    uuid: str
    fact: str | None
    valid_at: datetime | None
    sc: float
    dist: float | None


class KGPostgresReader:
    """Lazily-connected reader over the Postgres KG tables.

    Methods RAISE on failure — Postgres is the source of truth, so a read
    failure fails the extraction item and the queue retries it.
    """

    def __init__(self) -> None:
        self._conn: Any | None = None

    def _connection(self) -> Any:
        if self._conn is not None:
            try:
                self._conn.execute("SELECT 1")
                return self._conn
            except Exception:
                self._reset()
        import psycopg

        url = os.environ.get("SYNAPSE_DB_URL")
        if not url:
            raise RuntimeError("SYNAPSE_DB_URL unset")
        self._conn = psycopg.connect(url, autocommit=True)
        return self._conn

    def _reset(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            pass
        self._conn = None

    # -- entity reads ----------------------------------------------------

    def find_similar_nodes(
        self, query_embedding: list[float], limit: int = 20
    ) -> list[dict[str, Any]]:
        """Top-k entities by embedding distance. Mirrors FalkorDB's shape."""
        conn = self._connection()
        emb_s = _vec_literal(query_embedding)
        with conn.transaction():
            cur = conn.cursor(row_factory=class_row(_NearEntityRow))
            cur.execute("SET LOCAL hnsw.ef_search = 200")
            # The planner costs the seq scan as cheap (it can't see the
            # per-row detoast + halfvec-cast + distance work) and skips the
            # HNSW index: ~2.5s/lookup at 44K entities vs ~10ms indexed.
            # This scope only ever runs the ANN lookup, so force the index.
            cur.execute("SET LOCAL enable_seqscan = off")
            cur.execute(
                "SELECT uuid, name, dist FROM ("  # nosec B608 — _EMBED_DIMS is a validated int, not user input
                "  SELECT uuid, name, owner_id, "
                f"         embedding::halfvec({_EMBED_DIMS}) <=> %s::halfvec({_EMBED_DIMS}) AS dist "
                "  FROM kg_entities WHERE embedding IS NOT NULL "
                "  ORDER BY dist LIMIT %s"
                ") sub WHERE owner_id = %s LIMIT %s",
                (emb_s, _OVERFETCH, OWNER, limit),
            )
            return [
                {"uuid": r.uuid, "name": r.name, "score": float(r.dist)} for r in cur.fetchall()
            ]

    def entity_uuid_by_normalized_name(self, normalized: str) -> str | None:
        """Exact normalized-name lookup (NodeDeduper short-circuit).

        One ranked match on the row's key: its stored ``normalized_name``, or lower(trim(name))
        for a row written before that column existed (empty). Not a stored-name lookup with a
        fallback behind it: a fallback is reached only when the first step finds nothing, so a
        name held by an old row and a newer one would resolve to the newer, while the merge
        (schema/067) keeps the oldest. The empty-name arm is a few dozen rows, so the btree on
        ``normalized_name`` still answers the lookup."""
        if not normalized:
            return None
        conn = self._connection()
        cur = conn.execute(_ENTITY_BY_NAME_SQL, (OWNER, normalized, normalized))
        row = cur.fetchone()
        return str(row[0]) if row else None

    def load_type_map(self) -> dict[str, str]:
        """subtype -> supertype from kg_entity_types (the editable taxonomy map).

        Used by entity dedup to map a new entity's extracted type to its
        supertype for the type-compatibility candidate gate. Empty dict if the
        table is unseeded (gate then no-ops — every entity reads as untyped)."""
        conn = self._connection()
        cur = conn.execute("SELECT subtype, supertype FROM kg_entity_types")
        return {str(sub): str(sup) for sub, sup in cur.fetchall()}

    def load_entities(self) -> list[EntityRow]:
        """Every entity of the owner (LSH hydrate).

        In ``SURVIVOR_ORDER``: the caller's name index keeps the first row it sees for a key
        (``dict.setdefault``), so it must be the row ``entity_uuid_by_normalized_name`` returns.
        The key is the stored normalized_name, or lower(trim(name)) when that is empty."""
        conn = self._connection()
        cur = conn.cursor(row_factory=class_row(EntityRow)).execute(
            "SELECT uuid, COALESCE(name, '') AS name, COALESCE(summary, '') AS summary, "
            "entity_supertype, COALESCE(normalized_name, '') AS normalized_name "
            "FROM kg_entities WHERE owner_id = %s "
            "ORDER BY " + SURVIVOR_ORDER,  # nosec B608 - a module constant
            (OWNER,),
        )
        rows: list[EntityRow] = cur.fetchall()
        return rows

    # -- edge reads ------------------------------------------------------

    def find_similar_edges(
        self,
        fact_embedding: list[float],
        distance_threshold: float = 0.20,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Live edges within cosine-distance threshold of the fact embedding."""
        conn = self._connection()
        emb_s = _vec_literal(fact_embedding)
        with conn.transaction():
            cur = conn.cursor(row_factory=class_row(_NearEdgeRow))
            cur.execute("SET LOCAL hnsw.ef_search = 200")
            # The planner costs the seq scan as cheap (it can't see the
            # per-row detoast + halfvec-cast + distance work) and skips the
            # HNSW index: ~2.5s/lookup at 44K entities vs ~10ms indexed.
            # This scope only ever runs the ANN lookup, so force the index.
            cur.execute("SET LOCAL enable_seqscan = off")
            cur.execute(
                "SELECT uuid, fact, valid_at, dist FROM ("  # nosec B608 — _EMBED_DIMS is a validated int, not user input
                "  SELECT uuid, fact, valid_at, owner_id, "
                f"         fact_embedding::halfvec({_EMBED_DIMS}) <=> %s::halfvec({_EMBED_DIMS}) AS dist "
                "  FROM kg_relationships "
                "  WHERE t_invalid IS NULL AND fact_embedding IS NOT NULL "
                "  ORDER BY dist LIMIT %s"
                ") sub WHERE owner_id = %s AND dist <= %s LIMIT %s",
                (emb_s, _OVERFETCH, OWNER, distance_threshold, limit),
            )
            return [
                {
                    "uuid": r.uuid,
                    "fact": r.fact,
                    "valid_at": _iso(r.valid_at),
                    "score": float(r.dist),
                }
                for r in cur.fetchall()
            ]

    def nearest_live_edges(
        self,
        fact_embeddings: list[list[float]],
        max_distance: float,
        limit: int,
    ) -> list[list[dict[str, Any]]]:
        """Per embedding, up to ``limit`` live edges of the owner within ``max_distance``
        (cosine), nearest first. Parallel to ``fact_embeddings``.

        One statement for the whole batch: ``unnest`` feeds each embedding into a LATERAL
        HNSW scan over the bare partial-index predicate, and the tenant scope and distance
        bound filter that scan's short over-fetch (same discipline as
        ``find_similar_edges``)."""
        if not fact_embeddings:
            return []
        conn = self._connection()
        with conn.transaction():
            cur = conn.cursor(row_factory=class_row(_BatchEdgeRow))
            cur.execute("SET LOCAL hnsw.ef_search = 200")
            cur.execute("SET LOCAL enable_seqscan = off")
            cur.execute(
                "SELECT q.i, n.uuid, n.fact, n.dist "  # nosec B608 — _EMBED_DIMS is a validated int, not user input
                "FROM unnest(%s::text[]) WITH ORDINALITY AS q(emb, i) "
                "CROSS JOIN LATERAL ("
                "  SELECT uuid, fact, dist FROM ("
                "    SELECT uuid, fact, owner_id, "
                f"           fact_embedding::halfvec({_EMBED_DIMS}) <=> q.emb::halfvec({_EMBED_DIMS}) AS dist "
                "    FROM kg_relationships "
                "    WHERE t_invalid IS NULL AND fact_embedding IS NOT NULL "
                "    ORDER BY dist LIMIT %s"
                "  ) c WHERE owner_id = %s AND dist <= %s "
                "  ORDER BY dist LIMIT %s"
                ") n ORDER BY q.i, n.dist",
                (
                    [_vec_literal(e) for e in fact_embeddings],
                    _RESTATED_OVERFETCH,
                    OWNER,
                    max_distance,
                    limit,
                ),
            )
            out: list[list[dict[str, Any]]] = [[] for _ in fact_embeddings]
            for r in cur.fetchall():
                out[r.i - 1].append({"uuid": r.uuid, "fact": r.fact, "score": float(r.dist)})
            return out

    def find_edges_by_pair(self, source_uuid: str, target_uuid: str) -> list[dict[str, Any]]:
        """Live edges between a source/target pair (contradiction candidates)."""
        conn = self._connection()
        cur = conn.cursor(row_factory=class_row(_PairEdgeRow)).execute(
            "SELECT uuid, fact, valid_at, fact_embedding::text AS fact_embedding "
            "FROM kg_relationships WHERE owner_id = %s "
            "  AND src_uuid = %s AND tgt_uuid = %s AND t_invalid IS NULL",
            (OWNER, source_uuid, target_uuid),
        )
        return [
            {
                "uuid": r.uuid,
                "fact": r.fact,
                "valid_at": _iso(r.valid_at),
                "fact_embedding": _emb_list(r.fact_embedding),
            }
            for r in cur.fetchall()
        ]

    def find_edges_by_fulltext(
        self,
        query: str,
        limit: int = 20,
        fact_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """BM25 candidates over live fact text (ParadeDB, OR semantics).

        With ``fact_embedding``, each hit also carries ``dist``: the cosine distance from
        its stored embedding to that vector (None when the edge has no embedding), so a
        caller can tell a lexical hit on a shared identifier from one on the same topic."""
        safe = "".join(c if (c.isalnum() or c.isspace()) else " " for c in query).strip()
        if not safe:
            return []
        conn = self._connection()
        args: tuple[Any, ...]
        if fact_embedding is None:
            dist_sql, args = "NULL", (safe, OWNER, limit)
        else:
            dist_sql = f"fact_embedding::halfvec({_EMBED_DIMS}) <=> %s::halfvec({_EMBED_DIMS})"
            args = (_vec_literal(fact_embedding), safe, OWNER, limit)
        cur = conn.cursor(row_factory=class_row(_FulltextEdgeRow)).execute(
            f"SELECT uuid, fact, valid_at, paradedb.score(id) AS sc, {dist_sql} AS dist "  # nosec B608 — _EMBED_DIMS is a validated int, not user input
            "FROM kg_relationships "
            "WHERE id @@@ paradedb.match('fact', %s)"
            f"{bm25_filter(['owner_id = %s', 't_invalid IS NULL'])} "
            "ORDER BY sc DESC LIMIT %s",
            args,
        )
        return [
            {
                "uuid": r.uuid,
                "fact": r.fact,
                "valid_at": _iso(r.valid_at),
                "score": float(r.sc),
                "dist": None if r.dist is None else float(r.dist),
            }
            for r in cur.fetchall()
        ]
