"""The MCP server's shared Postgres connection pool, and how call sites borrow from it.

The server's HTTP lanes used to open a fresh connection for every read. One /context
request did five: the auth gate, the trust lookup, notes, timeline, banner. Each one
costs a TCP connect plus a SCRAM handshake, more than the query it carried. The server
now holds one pool open from startup to shutdown (:func:`held_open` is its lifespan),
and the per-request call sites borrow through :func:`connection`.

A process that never opens a pool gets exactly the old behaviour from
:func:`connection`: one autocommit connection, closed when the block exits. That is
every process except the HTTP server (the poller, dream, scripts, tests), and a
stdio or dev server with no database.

Borrowed connections are autocommit with tuple rows and must go back that way. A
caller that wants dict rows sets them on its cursor (``conn.cursor(row_factory=
dict_row)``), never on the connection. Never borrow a second connection while holding
one: with a small pool, nested borrows deadlock under load.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import psycopg
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)

# Postgres has 100 connection slots, shared with the poller's workers, dream, and the
# recall engine's per-thread connections. The routes hold a connection for a query or
# two, so a handful covers the server's request concurrency.
_MIN_SIZE = 2
_MAX_SIZE = 8
# Every borrower already fails closed or soft on a database error. Waiting the
# library's default 30 s for a free connection would only hold the request open.
_BORROW_TIMEOUT_S = 10.0

_POOLS: dict[str, ConnectionPool[psycopg.Connection[Any]]] = {}
_LOCK = threading.Lock()


def open_pool(db_url: str) -> None:
    """Open the pool for ``db_url`` if it isn't open yet. No-op without a URL.

    Returns at once: the pool fills in the background, so a slow or unreachable database
    does not hold up server start. Until a connection exists, a borrower waits up to the
    borrow timeout and then fails the way a failed connect always did.
    """
    if not db_url:
        return
    with _LOCK:
        if db_url in _POOLS:
            return
        pool: ConnectionPool[psycopg.Connection[Any]] = ConnectionPool(
            db_url,
            min_size=_MIN_SIZE,
            max_size=_MAX_SIZE,
            kwargs={"autocommit": True, "connect_timeout": 5},
            # A Postgres restart (an image update) kills every idle pooled connection.
            # Checking on borrow costs one round trip and swaps in a live connection
            # instead of failing the request that drew a dead one.
            check=ConnectionPool.check_connection,
            timeout=_BORROW_TIMEOUT_S,
            name="synapse-http",
            open=False,
        )
        pool.open(wait=False)
        _POOLS[db_url] = pool
    logger.info("postgres pool opened (min %d, max %d)", _MIN_SIZE, _MAX_SIZE)


def close_pool(db_url: str) -> None:
    """Close and forget the pool for ``db_url``. No-op if none is open."""
    with _LOCK:
        pool = _POOLS.pop(db_url, None)
    if pool is not None:
        pool.close()
        logger.info("postgres pool closed")


@asynccontextmanager
async def held_open(db_url: str) -> AsyncIterator[dict[str, Any]]:
    """Hold the pool for ``db_url`` open for the length of the block: the MCP server's
    lifespan, which FastMCP enters once per process, not per stateless request. Yields
    the (empty) lifespan context. No-op without a URL, so a stdio or dev boot with no
    database opens nothing."""
    open_pool(db_url)
    try:
        yield {}
    finally:
        await asyncio.to_thread(close_pool, db_url)


@contextmanager
def connection(
    db_url: str, connect_timeout: int | None = None
) -> Iterator[psycopg.Connection[Any]]:
    """An autocommit connection to ``db_url`` for the length of the block: borrowed from
    the pool when one is open, otherwise opened here and closed on exit.

    ``connect_timeout`` applies to that one-off connect only; pooled connections use the
    pool's own.
    """
    pool = _POOLS.get(db_url)
    if pool is not None:
        with pool.connection() as conn:
            yield conn
        return
    kwargs: dict[str, Any] = {} if connect_timeout is None else {"connect_timeout": connect_timeout}
    conn = psycopg.connect(db_url, autocommit=True, **kwargs)
    try:
        yield conn
    finally:
        conn.close()
