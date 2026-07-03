"""Postgres reads for the WRITE-side KG pipeline (#67 PR 2; sole backend since PR 3).

The extraction pipeline reads the graph in six places while writing it:

  - stage 3 entity resolution      -> ``find_similar_nodes``
  - stage 4 NodeDeduper            -> exact normalized-name lookup + LSH hydrate scan
  - stage 6a/6b contradiction      -> ``find_edges_by_pair``
  - stage 7 dedup candidate pool   -> ``find_edges_by_pair`` + ``find_similar_edges``
                                      + ``find_edges_by_fulltext``

This module serves all of them from the ``kg_entities`` / ``kg_relationships``
tables (schema/017), scoped by ``owner_id`` + ``group_id``, via
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
the GLOBAL HNSW on the bare partial-index predicate (owner/group equality filters
push the planner off the index onto a bitmap+sort), then filter tenant scope on the
small candidate set. The connection is lazy + autocommit, mirroring
``kg_shadow.PgKgShadowWriter``; per-statement ``hnsw.ef_search`` is set via
``SET LOCAL`` inside a transaction block.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import orjson

from ingestion.embedding import embed_dims

logger = logging.getLogger(__name__)

# Embedding width for the halfvec casts below — must match the provisioned schema
# (and its HNSW index expressions) verbatim. Default 2048 (Voyage prod, unchanged).
_EMBED_DIMS = embed_dims()

# Single-owner constant today; #49 owner axis threads a real owner through env.
OWNER = os.environ.get("SYNAPSE_KG_OWNER_ID", "default")

# Over-fetch pool for global-HNSW-then-filter (see module docstring / kg_pg.py).
_OVERFETCH = 200


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
        self, query_embedding: list[float], group_id: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Top-k entities by embedding distance. Mirrors FalkorDB's shape."""
        conn = self._connection()
        emb_s = _vec_literal(query_embedding)
        with conn.transaction():
            cur = conn.cursor()
            cur.execute("SET LOCAL hnsw.ef_search = 200")
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

    def entity_uuid_by_normalized_name(self, normalized: str, group_id: str) -> str | None:
        """Exact normalized-name lookup (NodeDeduper short-circuit)."""
        conn = self._connection()
        cur = conn.execute(
            "SELECT uuid FROM kg_entities "
            "WHERE owner_id = %s AND group_id = %s AND normalized_name = %s LIMIT 1",
            (OWNER, group_id, normalized),
        )
        row = cur.fetchone()
        if row:
            return str(row[0])
        # Pre-migration nodes without the property: compute on the fly (rare;
        # mirrors the FalkorDB fallback).
        cur = conn.execute(
            "SELECT uuid FROM kg_entities "
            "WHERE owner_id = %s AND group_id = %s AND lower(trim(name)) = %s LIMIT 1",
            (OWNER, group_id, normalized),
        )
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

    def load_entities(self, group_id: str) -> list[tuple[str, str, str, str | None]]:
        """(uuid, name, summary, entity_supertype) for every entity in the group (LSH hydrate)."""
        conn = self._connection()
        cur = conn.execute(
            "SELECT uuid, COALESCE(name, ''), COALESCE(summary, ''), entity_supertype "
            "FROM kg_entities WHERE owner_id = %s AND group_id = %s",
            (OWNER, group_id),
        )
        return [(str(u), n, s, st) for u, n, s, st in cur.fetchall()]

    # -- edge reads ------------------------------------------------------

    def find_similar_edges(
        self,
        fact_embedding: list[float],
        group_id: str,
        distance_threshold: float = 0.20,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Live edges within cosine-distance threshold of the fact embedding."""
        conn = self._connection()
        emb_s = _vec_literal(fact_embedding)
        with conn.transaction():
            cur = conn.cursor()
            cur.execute("SET LOCAL hnsw.ef_search = 200")
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

    def find_edges_by_pair(
        self, source_uuid: str, target_uuid: str, group_id: str
    ) -> list[dict[str, Any]]:
        """Live edges between a source/target pair (contradiction candidates)."""
        conn = self._connection()
        cur = conn.execute(
            "SELECT uuid, fact, valid_at, fact_embedding::text FROM kg_relationships "
            "WHERE owner_id = %s AND group_id = %s "
            "  AND src_uuid = %s AND tgt_uuid = %s AND t_invalid IS NULL",
            (OWNER, group_id, source_uuid, target_uuid),
        )
        return [
            {"uuid": u, "fact": f, "valid_at": _iso(v), "fact_embedding": _emb_list(e)}
            for u, f, v, e in cur.fetchall()
        ]

    def find_edges_by_fulltext(
        self, query: str, group_id: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        """BM25 candidates over live fact text (ParadeDB, OR semantics)."""
        safe = "".join(c if (c.isalnum() or c.isspace()) else " " for c in query).strip()
        if not safe:
            return []
        conn = self._connection()
        cur = conn.execute(
            "SELECT uuid, fact, valid_at, paradedb.score(id) AS sc FROM kg_relationships "
            "WHERE id @@@ paradedb.match('fact', %s) "
            "  AND owner_id = %s AND group_id = %s AND t_invalid IS NULL "
            "ORDER BY sc DESC LIMIT %s",
            (safe, OWNER, group_id, limit),
        )
        return [
            {"uuid": u, "fact": f, "valid_at": _iso(v), "score": float(sc)}
            for u, f, v, sc in cur.fetchall()
        ]
