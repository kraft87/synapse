"""Shared embedding test scaffolding: a deterministic one-hot vector builder and
the default test group id, used across the notes / preferences / KG DB-backed
tests to assert KNN ordering without a real embedder."""

from __future__ import annotations

from ingestion.embedding import embed_dims

GROUP = "technical"

_DIMS = embed_dims()


def onehot(slot: int) -> list[float]:
    """A one-hot unit vector: identical slots -> cosine sim 1, distinct -> 0."""
    v = [0.0] * _DIMS
    v[slot % _DIMS] = 1.0
    return v


class OneHotEmbedder:
    """Embeds every text to one slot, so a note seeded on that slot is an exact match."""

    model_name = "test"

    def __init__(self, slot: int) -> None:
        self._slot = slot

    def embed(self, texts: list[str], task: str) -> list[list[float]]:
        return [onehot(self._slot) for _ in texts]
