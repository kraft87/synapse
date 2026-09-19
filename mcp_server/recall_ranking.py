"""Pure ranking and response-sizing policy used by the recall engine."""

from __future__ import annotations

import math
import time
from datetime import UTC, datetime
from typing import Any


def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank + 1)


def recency_multiplier(created_at: Any, half_life_days: float = 30) -> float:
    """Exponential decay: 1.0 today, 0.5 at ``half_life_days``."""
    if created_at is None:
        return 1.0
    try:
        ts = (
            datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            if isinstance(created_at, str)
            else created_at
        )
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        age_days = (datetime.now(UTC) - ts).total_seconds() / 86400
        return math.exp(-age_days * math.log(2) / half_life_days)
    except Exception:
        return 1.0


def feedback_multiplier(retrieval_count: Any) -> float:
    """Boost frequently retrieved items, capped at 2x."""
    try:
        return min(2.0, 1.0 + math.log1p(int(retrieval_count or 0)) * 0.3)
    except Exception:
        return 1.0


def rrf_fuse(ranked_lists: list[list[str]], k: int = 1) -> dict[str, float]:
    """Fuse ranked identifier lists into RRF scores."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, identifier in enumerate(ranked):
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (rank + k + 1)
    return scores


def merge_rrf(
    *ranked_lists: list[dict[str, Any]],
    id_key: str = "id",
    apply_recency: bool = True,
) -> list[dict[str, Any]]:
    """Merge ranked documents with RRF, recency decay, and feedback boost."""
    scores: dict[Any, float] = {}
    items: dict[Any, dict[str, Any]] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            item_id = item[id_key]
            score = rrf_score(rank)
            if apply_recency:
                score *= recency_multiplier(item.get("created_at"))
            score *= feedback_multiplier(item.get("retrieval_count"))
            scores[item_id] = scores.get(item_id, 0.0) + score
            items.setdefault(item_id, item)
    return sorted(items.values(), key=lambda item: scores[item[id_key]], reverse=True)


def timed(fn: Any, *args: Any) -> tuple[Any, float]:
    """Run a recall leg and return its result and wall time in milliseconds."""
    started = time.perf_counter()
    result = fn(*args)
    return result, (time.perf_counter() - started) * 1000.0


def served_chars(out: dict[str, Any]) -> int:
    """Count answer-bearing characters in a recall response."""
    size = len(out.get("query") or "")
    for fact in out.get("facts", []):
        size += len(fact.get("fact") or "")
    for episode in out.get("episodes", []):
        size += len(episode.get("content") or "") + len(str(episode.get("date") or ""))
    for entity in out.get("entities", []):
        size += len(entity.get("name") or "") + len(entity.get("summary") or "")
    for web in out.get("web", []):
        size += (
            len(web.get("context") or "")
            + len(web.get("excerpt") or "")
            + len(web.get("title") or "")
        )
    for historical in out.get("superseded_facts", []):
        size += len(str(historical.get("fact") or "")) + len(
            str(historical.get("superseded_by") or "")
        )
    return size


def cutoff_k(scores: list[float], tau: float, min_k: int, max_k: int) -> int:
    """Return the bounded count whose score is at least ``tau`` times the top score."""
    if not scores:
        return 0
    top = scores[0]
    if top <= 0:
        return min(len(scores), max_k)
    keep = sum(1 for score in scores if score >= tau * top)
    return min(max(keep, min_k), max_k, len(scores))
