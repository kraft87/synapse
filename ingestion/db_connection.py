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
    """Format an embedding as a pgvector text literal, six decimals per component.
    Returns ``None`` for a NULL embedding."""
    if embedding is None:
        return None
    return "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"


def bm25_query_text(query: str) -> str:
    """A free-text query made safe for ParadeDB's ``paradedb.match``: alphanumerics and
    whitespace only, so identifiers and error strings survive while the query-language
    metacharacters that would raise a parse error become spaces. May return ''."""
    return "".join(c if (c.isalnum() or c.isspace()) else " " for c in query).strip()


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
        # Depth of atomic() blocks: while > 0 a nested _conn() exit leaves the commit and
        # the rollback to the block, so everything inside commits together or not at all.
        self._atomic = 0

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
            if not self._atomic:
                self._connection.rollback()
            raise
        else:
            if not self._atomic:
                self._connection.commit()

    @contextmanager
    def atomic(self) -> Generator[psycopg.Connection[Any], None, None]:
        """One borrow and one transaction for the block. Store methods called inside reuse
        the connection but neither commit nor roll back, so the block's writes land together
        on a clean exit and any exception rolls all of them back (an episode never commits
        without its queue row). Code inside that catches a failed statement and carries on
        must wrap it in ``conn.transaction()`` (a savepoint here), or the aborted
        transaction fails the next statement."""
        with self._conn() as conn, conn.transaction():
            self._atomic += 1
            try:
                yield conn
            finally:
                self._atomic -= 1

    def close(self) -> None:
        if self._connection and not self._connection.closed and not self._borrowed:
            self._connection.close()
