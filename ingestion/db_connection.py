from __future__ import annotations

from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg.rows import dict_row

from ingestion import pg_pool
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
        # In a process with a pool open (the MCP server), the connection is borrowed for
        # the outermost _conn() block only and goes back when it exits; nested blocks
        # reuse it and commit on exit exactly as below. Without a pool the instance keeps
        # one connection until close(), which the poller and the backfills rely on.
        self._borrowed = False

    @contextmanager
    def _conn(self) -> Generator[psycopg.Connection[Any], None, None]:
        if self._connection is None and pg_pool.pooled(self._url):
            with pg_pool.connection(self._url, autocommit=False, row_factory=dict_row) as conn:
                self._connection, self._borrowed = conn, True
                try:
                    with self._conn() as same:
                        yield same
                finally:
                    self._connection, self._borrowed = None, False
            return
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
        if self._connection and not self._connection.closed and not self._borrowed:
            self._connection.close()
