"""The MCP server's one Postgres connection pool, and how everything in it borrows.

The server used to open a fresh connection for nearly every read and write: a /context
request did five, and each one costs a TCP connect plus a SCRAM handshake, more than
the query it carried. The server now holds one pool open from startup to shutdown
(:func:`held_open` is its lifespan), and its routes, tools, stores and the recall engine
all borrow through :func:`connection`.

A process that never opens a pool gets the old behaviour from the same call: one
connection opened for the block and closed when it exits. That is every process except
the HTTP server (the poller, dream, scripts, tests), and a stdio or dev server with no
database. Either way the connection comes out the same shape: autocommit unless asked
otherwise, tuple rows unless asked otherwise, and ``hnsw.ef_search = 200``.

Every borrow sets its own ``autocommit`` and ``row_factory``, so one borrower's never
reach the next. Session-level settings are not scrubbed on return:
a ``RESET ALL`` on every return measured 0.5 ms a borrow, and running it in the pool's
worker made back-to-back borrows three times slower. So nothing that borrows may issue
a session-level ``SET``: use ``SET LOCAL`` or ``set_config(..., true)`` inside a
transaction (tests/test_pg_pool.py scans for it). ``hnsw.ef_search`` is a startup
option of every connection.

Never borrow a second connection while holding one: with a bounded pool, nested borrows
deadlock under load. Two connections never borrow: the dashboard's LISTEN connection
(dedicated for its lifetime by design) and the startup schema check (it runs before the
pool opens).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import psycopg
from psycopg.conninfo import conninfo_to_dict
from psycopg.pq import TransactionStatus
from psycopg.rows import RowFactory, tuple_row
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)

# Postgres has 100 connection slots; prod uses about 18 without the server's. Borrowers
# hold a connection for a statement or a short transaction. The busiest moment is a
# recall (its caller plus up to 5 parallel search legs) overlapping a few HTTP requests,
# so 16 covers it with room to spare while bounding what the server can ever take.
_MIN_SIZE = 2
_MAX_SIZE = 16
# Every borrower already fails closed or soft on a database error. Waiting the
# library's default 30 s for a free connection would only hold the request open.
_BORROW_TIMEOUT_S = 10.0
# HNSW search breadth for the halfvec vector indexes. Default 40 under-recalls a 100-deep
# fetch; 200 gives recall@100 0.981 / recall@10 1.000 vs exact (validated). Harmless on
# queries that don't use an HNSW index. Set at connect time, for every connection.
_SESSION_OPTIONS = "-c hnsw.ef_search=200"

_POOLS: dict[str, ConnectionPool[psycopg.Connection[Any]]] = {}
_HOLDERS: dict[str, int] = {}
_LOCK = threading.Lock()


def _options(db_url: str) -> str:
    """The URL's own ``options``, if any, plus ours: a kwarg would replace them."""
    own = conninfo_to_dict(db_url).get("options")
    return f"{own} {_SESSION_OPTIONS}" if own else _SESSION_OPTIONS


def open_pool(db_url: str) -> None:
    """Open the pool for ``db_url``, or add a holder to the open one. No-op without a URL.

    Each call needs a matching :func:`close_pool`; the pool closes when the last holder
    lets go. Returns at once: the pool fills in the background, so a slow or unreachable
    database does not hold up server start. Until a connection exists, a borrower waits
    up to the borrow timeout and then fails the way a failed connect always did.
    """
    if not db_url:
        return
    with _LOCK:
        _HOLDERS[db_url] = _HOLDERS.get(db_url, 0) + 1
        if db_url in _POOLS:
            return
        pool: ConnectionPool[psycopg.Connection[Any]] = ConnectionPool(
            db_url,
            min_size=_MIN_SIZE,
            max_size=_MAX_SIZE,
            kwargs={"autocommit": True, "connect_timeout": 5, "options": _options(db_url)},
            # A Postgres restart (an image update) kills every idle pooled connection.
            # Checking on borrow costs one round trip and swaps in a live connection
            # instead of failing the request that drew a dead one.
            check=ConnectionPool.check_connection,
            timeout=_BORROW_TIMEOUT_S,
            name="synapse",
            open=False,
        )
        pool.open(wait=False)
        _POOLS[db_url] = pool
    logger.info("postgres pool opened (min %d, max %d)", _MIN_SIZE, _MAX_SIZE)


def close_pool(db_url: str) -> None:
    """Let go of the pool for ``db_url``; the last holder closes it. No-op if none."""
    with _LOCK:
        holders = _HOLDERS.get(db_url, 0) - 1
        if holders > 0:
            _HOLDERS[db_url] = holders
            return
        _HOLDERS.pop(db_url, None)
        pool = _POOLS.pop(db_url, None)
    if pool is not None:
        pool.close()
        logger.info("postgres pool closed")


def pooled(db_url: str) -> bool:
    """True when a pool is open for ``db_url`` in this process."""
    return db_url in _POOLS


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
    db_url: str,
    *,
    autocommit: bool = True,
    row_factory: RowFactory[Any] | None = None,
    connect_timeout: int | None = None,
) -> Iterator[psycopg.Connection[Any]]:
    """A connection to ``db_url`` for the length of the block: borrowed from the pool
    when one is open, otherwise opened here and closed on exit.

    With ``autocommit=False`` the block is one transaction, committed when it exits
    cleanly and rolled back when it raises, pooled or not (``psycopg.connect`` used as a
    context manager behaves the same way). ``row_factory`` applies to this borrow only.
    ``connect_timeout`` applies to the one-off connect; pooled connections use the pool's.
    """
    pool = _POOLS.get(db_url)
    if pool is not None:
        conn = pool.getconn()
        try:
            conn.autocommit = autocommit
            conn.row_factory = row_factory if row_factory is not None else tuple_row
            try:
                yield conn
            except BaseException:
                if not autocommit:
                    with contextlib.suppress(Exception):  # a broken connection is discarded
                        conn.rollback()
                raise
            else:
                if not autocommit:
                    conn.commit()
        finally:
            if conn.info.transaction_status != TransactionStatus.IDLE:
                conn.close()  # broken, or a transaction left open: never hand it on
            pool.putconn(conn)
        return
    kwargs: dict[str, Any] = {"options": _options(db_url)}
    if row_factory is not None:
        kwargs["row_factory"] = row_factory
    if connect_timeout is not None:
        kwargs["connect_timeout"] = connect_timeout
    with psycopg.connect(db_url, autocommit=autocommit, **kwargs) as conn:
        yield conn
