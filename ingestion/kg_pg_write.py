"""Primary Postgres writer for the KG (#67 PR 3 — dual-write retired).

Promoted from the dual-write shadow (``kg_shadow.PgKgShadowWriter``): the SQL
is unchanged, the failure semantics are inverted. The shadow swallowed every
exception because FalkorDB was the source of truth and a mirror hiccup must
never break a graph write. Postgres IS the source of truth now, so every
method RAISES on failure — the extraction queue item fails and retries
instead of silently dropping facts.

All statements are idempotent (``ON CONFLICT`` upsert/no-op, ``UPDATE`` by
uuid), so a retry after a partial failure is safe.

``owner_id`` is written as the single-owner constant ``'default'``; the #49
owner axis can thread a real owner through ``SYNAPSE_KG_OWNER_ID`` without
touching any call site.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

from ingestion.embedding import embed_dims
from ingestion.textsafe import strip_nul

logger = logging.getLogger(__name__)

# Embedding width for the vector casts below — matches the provisioned schema.
# Default 2048 (Voyage prod, unchanged).
_EMBED_DIMS = embed_dims()

# Single-owner constant for the live graph today; overridable for multi-tenant.
OWNER = os.environ.get("SYNAPSE_KG_OWNER_ID", "default")


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


def _vec(v: Any) -> str | None:
    """Format an embedding list for pgvector text input ('[a,b,c]'), or None."""
    if not v:
        return None
    return "[" + ",".join(map(str, v)) + "]"


class KGPostgresWriter:
    """Lazily-connected writer over the ``kg_entities`` / ``kg_relationships``
    tables (schema/017). Raises on failure — see module docstring."""

    def __init__(self) -> None:
        self._conn: Any | None = None

    # -- connection management ------------------------------------------------
    def _cursor(self) -> Any:
        """Return a live cursor, (re)connecting if needed. Raises when the
        database is unreachable or ``SYNAPSE_DB_URL`` is unset."""
        if self._conn is not None:
            try:
                return self._conn.cursor()
            except Exception:
                self._reset()
        import psycopg

        url = os.environ.get("SYNAPSE_DB_URL")
        if not url:
            raise RuntimeError("SYNAPSE_DB_URL unset")
        self._conn = psycopg.connect(url, autocommit=True)
        return self._conn.cursor()

    def _reset(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            pass
        self._conn = None

    def _run(self, fn: Any) -> None:
        """Execute ``fn(cursor)``; drop the connection on failure and re-raise
        so the next call reconnects cleanly."""
        cur = self._cursor()
        try:
            fn(cur)
        except Exception:
            self._reset()
            raise

    # -- mutations ------------------------------------------------------------
    def upsert_node(
        self,
        *,
        uuid: str,
        name: str,
        normalized_name: str,
        entity_type: str,
        summary: str,
        group_id: str,
        project: str | None,
        created_at: str | None,
        valid_at: str | None,
        embedding: list[float] | None,
        supertype: str | None = None,
    ) -> None:
        """Insert-or-update an entity. On conflict updates only the fields an
        update is allowed to change (name/summary/normalized_name,
        embedding-if-present, entity_supertype/subtype); created_at/entity_type/
        valid_at are insert-only. entity_subtype = the raw entity_type; supertype
        is the caller-derived taxonomy rollup (updatable so the fluid map propagates)."""
        created = _ts(created_at)
        valid = _ts(valid_at) or created

        def _do(cur: Any) -> None:
            cur.execute(
                "INSERT INTO kg_entities "  # nosec B608 — _EMBED_DIMS is a validated int, not user input
                "  (uuid, owner_id, group_id, project, name, normalized_name, "
                "   entity_type, summary, embedding, created_at, t_created, "
                "   valid_at, t_valid, entity_supertype, entity_subtype) "
                f"VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::vector({_EMBED_DIMS}),%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (uuid) DO UPDATE SET "
                "   name = EXCLUDED.name, "
                "   summary = EXCLUDED.summary, "
                "   normalized_name = EXCLUDED.normalized_name, "
                "   embedding = COALESCE(EXCLUDED.embedding, kg_entities.embedding), "
                "   entity_supertype = EXCLUDED.entity_supertype, "
                "   entity_subtype = EXCLUDED.entity_subtype",
                (
                    uuid,
                    OWNER,
                    group_id,
                    project or "",
                    strip_nul(name or ""),
                    strip_nul(normalized_name or ""),
                    entity_type or "",
                    strip_nul(summary or ""),
                    _vec(embedding),
                    created,
                    created,
                    valid,
                    valid,
                    supertype,
                    entity_type or None,
                ),
            )

        self._run(_do)

    def create_edges(self, rows: list[dict[str, Any]], group_id: str) -> None:
        """Insert relationship edges. One multi-row INSERT, ON CONFLICT (uuid)
        DO NOTHING (re-creating an existing edge is a retry no-op).

        Each row carries: src, tgt, edge_uuid, name, fact, episodes,
        created_at, t_created, valid_at, t_valid, optional emb (fact embedding),
        optional web_artifact_id (web-lane provenance, task #68), and the optional
        schema-058 evidence fields ongoing / last_supported_at / last_supported_by
        (absent or None -> NULL: unknown, never-confirmed).
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

        def _do(cur: Any) -> None:
            # Constant parameterized statement + executemany: psycopg3 pipelines
            # the rows, and a fixed query string keeps every value bound.
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

    def invalidate_edges(
        self,
        items: list[tuple[str, str | None]],
        group_id: str,
        invalidated_by: str | None = None,
    ) -> None:
        """Set ``t_invalid`` (and the legacy ``invalid_at``) on each edge.
        ``items`` is a list of (edge_uuid, invalid_at); invalid_at is already
        resolved to a concrete timestamp by the caller, None falls back to
        now() per-row. ``invalidated_by`` (the superseding edge's uuid) applies
        to every item in the call and is set only when provided (contradiction
        path); COALESCE preserves any existing value when None (schema 028).

        The effective time is clamped to no earlier than the edge's own
        ``t_valid``: a correction dated before a stale edge was ever written
        retires it as of the moment it became live, never with a negative
        lifetime. An already-retired edge keeps its earlier ``t_invalid``."""
        if not items:
            return

        def _do(cur: Any) -> None:
            for edge_uuid, inv in items:
                ts = _ts(inv)
                cur.execute(
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

        def _do(cur: Any) -> None:
            for item in items:
                edge_uuid, eps = item[0], item[1]
                support_at = _ts(item[2]) if len(item) > 2 else None
                support_by = item[3] if len(item) > 3 else None
                if not eps:
                    continue
                eps_json = json.dumps(eps)
                cur.execute(
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
