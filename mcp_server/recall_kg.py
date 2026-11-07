"""The KG facts leg of recall(): the requested group's search plus, for full trust, the other
group's BM25 leg, which recall_overview runs beside it."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from psycopg.rows import tuple_row

from mcp_server.kg_pg import CROSS_GROUP_EXTRAS, FactRow, cross_bm25_rows, search_kg
from mcp_server.recall_warnings import error_brief as _err_brief
from mcp_server.recall_warnings import warn as _warn

logger = logging.getLogger(__name__)


class RecallKGMixin:
    def _search_kg(
        self,
        query: str,
        query_emb: list[float],
        group_id: str | list[str],
        session_focus: list[str],
        fact_limit: int,
        allowed_projects: list[str] | None = None,
        cross_bm25: Callable[[], list[FactRow]] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """KG facts leg over kg_entities / kg_relationships (task #67).

        ``allowed_projects`` is a restricted surface's allowlist: only facts whose whole
        provenance lies inside it are served (schema 056, ``kg_pg.scope_predicate``).
        Without schema 056 the restricted leg returns nothing and logs one warning. It
        never adds a response warning, matching the pre-056 skip. ``None`` = full trust.
        ``group_id``: one group or a list (``kg_pg.read_groups``; ``kg_pg.search_kg``).
        ``cross_bm25``: the other group's BM25 rows when a caller is already fetching them
        on another connection (:meth:`_cross_bm25_rows`); it is called once, after the
        requested group is read, and a failure in it fails this leg as one in the group's
        own legs does.

        Runs inside one transaction so the planner GUCs are SET LOCAL — scoped
        to this query, not the thread-local connection that other legs may
        reuse later (enable_seqscan=off session-wide would be a foot-gun for
        any future unindexed query). hnsw.ef_search is already set
        on every connection by ingestion/pg_pool. A failure degrades to an empty facts
        bucket rather than failing the whole recall.
        """
        settings = self._settings()
        if self._kg_scope_blocked(allowed_projects):
            return [], []
        try:
            with self._pg() as conn:
                with conn.transaction():
                    # kg_pg indexes rows positionally; recall borrows dict-row
                    # connections, so override at the cursor.
                    cur = conn.cursor(row_factory=tuple_row)
                    cur.execute("SET LOCAL enable_seqscan = off")
                    cur.execute("SET LOCAL max_parallel_workers_per_gather = 0")
                    return search_kg(
                        cur,
                        query,
                        query_emb,
                        settings._KG_OWNER,
                        group_id,
                        session_focus,
                        fact_limit,
                        allowed_projects=allowed_projects,
                        cross_bm25=cross_bm25,
                    )
        except Exception as e:
            if self._kg_scope_error(allowed_projects, e):
                return [], []
            logger.warning("KG search failed: %s", e)
            _warn(f"KG facts leg failed ({_err_brief(e)}): no facts served.")
            return [], []

    def _cross_bm25_rows(self, query: str, groups: list[str]) -> list[FactRow]:
        """The BM25 leg of the KG groups a full-trust recall did not ask for
        (``kg_pg.cross_bm25_rows``), on a connection of its own, so ``recall_overview`` can
        start it with the first wave of legs and the requested group's search never waits
        for it. It needs only the query text. Same transaction and planner settings as
        :meth:`_search_kg`, which the leg's statement was always sent under. Raises on a
        failure: the consumer (:meth:`_search_kg`) decides what that costs the bucket."""
        with self._pg() as conn, conn.transaction():
            cur = conn.cursor(row_factory=tuple_row)
            cur.execute("SET LOCAL enable_seqscan = off")
            cur.execute("SET LOCAL max_parallel_workers_per_gather = 0")
            return cross_bm25_rows(
                cur, query, self._settings()._KG_OWNER, groups, CROSS_GROUP_EXTRAS
            )
