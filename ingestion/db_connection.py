from __future__ import annotations

from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg.rows import dict_row

from ingestion.embedding import embed_dims

# Embedding width for the vector casts below — matches the provisioned schema.
# Default 2048 (Voyage prod, unchanged).
_EMBED_DIMS = embed_dims()


def _vector_literal(embedding: Sequence[float] | None) -> str | None:
    """Format an embedding as a pgvector text literal. The ``.6f`` precision is a
    correctness invariant: the write literal must match the query-cast literal
    byte-for-byte, or vector comparisons drift. Returns ``None`` for a NULL embedding."""
    if embedding is None:
        return None
    return "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"


class DatabaseConnection:
    def __init__(self, url: str) -> None:
        self._url = url
        # Simple synchronous connection — ONE thread per instance. The commit/rollback
        # in _conn() spans the whole connection, so concurrent users would entangle
        # transactions; the poller's concurrent drain gives each worker thread its own
        # Database (see Poller._thread_worker) rather than sharing this one.
        self._connection: psycopg.Connection[Any] | None = None

    @contextmanager
    def _conn(self) -> Generator[psycopg.Connection[Any], None, None]:
        if self._connection is None or self._connection.closed:
            self._connection = psycopg.connect(self._url, row_factory=dict_row, autocommit=False)
        try:
            yield self._connection
        except Exception:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

    def close(self) -> None:
        if self._connection and not self._connection.closed:
            self._connection.close()
