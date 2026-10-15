"""The engine surface the recall mixins use of one another, for the type checker.

``Recall`` is assembled from mixins that call each other's methods and read the engine's
connection and config through ``self``. Each mixin inherits ``RecallEngine`` so mypy sees
those members. It exists only while type checking (at runtime the base is ``object``), and
each signature is checked against the real definition where ``Recall`` combines the mixins.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import AbstractContextManager

    from ingestion.surfaces import SurfaceTrust
    from mcp_server.recall_config import RecallConfig
    from mcp_server.recall_episode_search import EpisodeLegs, EpisodePool

    class RecallEngine:
        _db_url: str
        _config: RecallConfig
        _leg_executor: ThreadPoolExecutor
        _async_executor: ThreadPoolExecutor

        def _pg(self) -> AbstractContextManager[Any]: ...
        def _ensure_embedder(self) -> Any: ...
        def _ensure_reranker(self) -> Any: ...
        @staticmethod
        def _echo_overlap(content: str, query: str) -> int: ...
        @staticmethod
        def _exclude_self(
            ranked_eps: list[dict[str, Any]], self_session: str
        ) -> list[dict[str, Any]]: ...
        def _floor_shadow(
            self, served_ids: dict[str, Any], rerank_top: float, emb_ok: bool
        ) -> None: ...
        def _resolve_trust(
            self, surface: str | None, trust: SurfaceTrust | None
        ) -> SurfaceTrust: ...
        def _own_surface(self, st: SurfaceTrust) -> str | None: ...
        def _start_episode_legs(
            self,
            query: str,
            project: str | None,
            session_id: str | None = None,
            allowed_projects: list[str] | None = None,
            own_surface: str | None = None,
        ) -> EpisodeLegs: ...
        def _join_episode_pool(
            self, legs: EpisodeLegs, self_session: str | None = None
        ) -> EpisodePool: ...
        def _rerank_pool_scored(
            self, query: str, pool: list[dict[str, Any]]
        ) -> list[tuple[int, float]]: ...
        def _rank_episodes(
            self,
            query: str,
            pool: list[dict[str, Any]],
            scored: list[tuple[int, float]],
            need: int,
        ) -> tuple[list[dict[str, Any]], float, int]: ...
        def _select_episodes(
            self, query: str, pool: list[dict[str, Any]], limit: int
        ) -> tuple[list[dict[str, Any]], int, float]: ...
        def _floor_by_rerank(
            self,
            query: str,
            items: list[dict[str, Any]],
            floor: float,
            *,
            text_key: str = "fact",
        ) -> list[dict[str, Any]]: ...
        def _compact_to_passages(
            self,
            query: str,
            episodes: list[dict[str, Any]],
            n: int,
            stats: dict[str, int] | None = None,
        ) -> list[dict[str, Any]]: ...
        def _search_kg(
            self,
            query: str,
            query_emb: list[float],
            group_id: str,
            session_focus: list[str],
            fact_limit: int,
            allowed_projects: list[str] | None = None,
        ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]: ...
        def _search_web_reranked(
            self,
            query: str,
            query_emb: list[float],
            allowed_projects: list[str] | None = None,
            own_surface: str | None = None,
        ) -> list[dict[str, Any]]: ...
        def _search_notes(
            self,
            query: str,
            query_emb: list[float],
            project: str | None,
            audience: str | None = None,
        ) -> list[dict[str, Any]]: ...
        def _surface_supersessions(
            self,
            query_emb: list[float] | None,
            group_id: str,
            allowed_projects: list[str] | None = None,
        ) -> list[dict[str, Any]]: ...
        def _self_session_facts(self, uuids: list[str], self_session: str) -> set[str]: ...
        def _fetch_superseded_pairs_pg(
            self,
            group_id: str,
            active_edge_uuids: list[str],
            cap: int,
            allowed_projects: list[str] | None = None,
        ) -> list[dict[str, Any]]: ...
        def _episode_supersessions(
            self,
            episode_ids: list[int],
            group_id: str,
            cap: int = 6,
            allowed_projects: list[str] | None = None,
            self_session: str | None = None,
            served_text: dict[int, str] | None = None,
        ) -> dict[int, list[str]]: ...
        def _increment_retrieval_counts(self, episode_ids: list[int]) -> None: ...
        def _increment_fact_retrieval_counts(
            self, edge_uuids: list[str], group_id: str
        ) -> None: ...
        def _record_metrics(self, metrics: dict[str, Any]) -> None: ...

else:
    RecallEngine = object
