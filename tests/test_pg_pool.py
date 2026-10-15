"""The server's shared Postgres pool (ingestion/pg_pool.py) and its lifespan wiring.

Covers: the no-pool fallback (every process but the HTTP server), connection reuse
once a pool is open, borrowed connections going back state-neutral (dict rows set on a
cursor never leak to the next borrower; a failed statement doesn't poison one), a
Postgres restart's dead connections being replaced on borrow, and the server's
lifespan opening the pool once and closing it, with a DB-less boot opening none.
"""

from __future__ import annotations

import importlib
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import psycopg
import pytest
from fastmcp import Client
from starlette.testclient import TestClient

from ingestion import pg_pool
from ingestion.surfaces import resolve_caller, token_hash
from tests.helpers.db import DB_URL as _DB_URL

pytestmark = pytest.mark.db


def _keyed(name: str) -> str:
    """The test DB under a URL of its own: pools are keyed by URL, so these tests never
    touch the session-wide pool the pooled DB-suite run opens (SYNAPSE_TEST_POOL=1)."""
    return f"{_DB_URL}{'&' if '?' in _DB_URL else '?'}application_name={name}"


_URL = _keyed("pg_pool_tests")
_LIFESPAN_URL = _keyed("pg_pool_lifespan")


@pytest.fixture()
def pool():
    pg_pool.open_pool(_URL)
    pg_pool._POOLS[_URL].wait(timeout=10)
    yield pg_pool._POOLS[_URL]
    pg_pool.close_pool(_URL)


def _backend_pid() -> int:
    with pg_pool.connection(_URL) as conn:
        row = conn.execute("SELECT pg_backend_pid()").fetchone()
        assert row is not None
        return int(row[0])


def test_without_a_pool_each_block_gets_its_own_connection_and_closes_it():
    assert _URL not in pg_pool._POOLS
    with pg_pool.connection(_URL) as conn:
        assert conn.autocommit
        held = conn
    assert held.closed
    assert _backend_pid() != _backend_pid()


def test_a_pool_reuses_its_connections(pool):
    pids = {_backend_pid() for _ in range(20)}
    assert len(pids) <= 2  # min_size; sequential borrows never grow the pool
    assert pool.get_stats()["pool_size"] <= 2


def test_dict_rows_from_resolve_caller_do_not_leak_to_the_next_borrower(pool, conn):
    conn.execute("DELETE FROM surfaces")
    conn.execute(
        "INSERT INTO surfaces (surface_id, trust, status, token_hash) "
        "VALUES ('dev:pool-test', 'full', 'approved', %s)",
        (token_hash("pool-test-token"),),
    )
    for _ in range(4):  # touch every pooled connection
        assert resolve_caller(_URL, token_hash_hex=token_hash("pool-test-token")).known
    for _ in range(4):
        with pg_pool.connection(_URL) as c:
            assert c.execute("SELECT 1, 2").fetchone() == (1, 2)
    conn.execute("DELETE FROM surfaces")


def test_a_failed_statement_does_not_poison_the_connection(pool):
    for _ in range(4):
        with pytest.raises(psycopg.errors.UndefinedTable):
            with pg_pool.connection(_URL) as c:
                c.execute("SELECT 1 FROM no_such_table_pg_pool_test")
    for _ in range(4):
        with pg_pool.connection(_URL) as c:
            assert c.execute("SELECT 1").fetchone() == (1,)


def test_connections_killed_by_a_postgres_restart_are_replaced_on_borrow(pool, conn):
    pids = {_backend_pid() for _ in range(6)}
    for pid in pids:
        conn.execute("SELECT pg_terminate_backend(%s)", (pid,))
    for _ in range(6):
        assert _backend_pid() not in pids


def test_open_pool_is_reference_counted():
    url = _keyed("pg_pool_refcount")
    pg_pool.open_pool(url)
    pg_pool.open_pool(url)
    first = pg_pool._POOLS[url]
    pg_pool.close_pool(url)
    assert pg_pool._POOLS.get(url) is first  # one holder left
    pg_pool.close_pool(url)
    assert url not in pg_pool._POOLS
    assert first.closed


def _private_row_count(conn, session_id: str) -> int:
    row = conn.execute(
        "SELECT count(*) FROM private_sessions WHERE session_id = %s", (session_id,)
    ).fetchone()
    return int(row[0])


def test_a_non_autocommit_borrow_commits_on_exit_and_rolls_back_on_error(pool, conn):
    conn.execute("DELETE FROM private_sessions WHERE session_id LIKE 'pg-pool-%'")
    with pg_pool.connection(_URL, autocommit=False) as c:
        c.execute("INSERT INTO private_sessions (session_id) VALUES ('pg-pool-kept')")
        assert _private_row_count(conn, "pg-pool-kept") == 0  # not yet committed
    assert _private_row_count(conn, "pg-pool-kept") == 1
    with pytest.raises(RuntimeError):
        with pg_pool.connection(_URL, autocommit=False) as c:
            c.execute("INSERT INTO private_sessions (session_id) VALUES ('pg-pool-dropped')")
            raise RuntimeError("boom")
    assert _private_row_count(conn, "pg-pool-dropped") == 0
    conn.execute("DELETE FROM private_sessions WHERE session_id LIKE 'pg-pool-%'")


def test_a_borrows_settings_never_reach_the_next_borrower(pool):
    """Each borrow sets its own autocommit and row_factory, so a dict-row transaction
    borrow leaves nothing behind, and every connection carries hnsw.ef_search = 200
    from its startup options."""
    from psycopg.rows import dict_row

    for _ in range(4):  # both pooled connections, at least once each
        with pg_pool.connection(_URL, autocommit=False, row_factory=dict_row) as c:
            assert not c.autocommit
            assert c.execute("SELECT 1 AS one").fetchone() == {"one": 1}
    for _ in range(4):
        with pg_pool.connection(_URL) as c:
            assert c.autocommit
            assert c.execute("SHOW hnsw.ef_search").fetchone() == ("200",)


def test_a_connection_left_mid_transaction_is_not_handed_on(pool):
    """A borrower that leaves a transaction open on an autocommit borrow (an explicit
    BEGIN) gets its connection closed, not returned to the next borrower mid-transaction."""
    with pg_pool.connection(_URL) as c:
        c.execute("BEGIN")
        held = c
    assert held.closed
    with pg_pool.connection(_URL) as c:
        assert c.execute("SELECT 1").fetchone() == (1,)


def test_nothing_issues_a_session_level_set():
    """The pool does not scrub session settings on return (a RESET ALL per borrow cost
    0.5 ms; run in the pool's worker it made back-to-back borrows 3x slower), so a
    session-level SET from one borrower would reach the next. Use SET LOCAL or
    set_config(..., true) inside a transaction instead."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    session_set = re.compile(r"""execute\(\s*f?["']SET\s+(?!LOCAL\b)""", re.I)
    session_config = re.compile(r"set_config\([^)]*,\s*false\s*\)", re.I)
    offenders = [
        f"{path.relative_to(root)}:{n}"
        for folder in ("mcp_server", "ingestion")
        for path in sorted((root / folder).rglob("*.py"))
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if session_set.search(line) or session_config.search(line)
    ]
    assert offenders == []


def test_more_borrowers_than_connections_wait_their_turn(pool):
    """A burst wider than the pool (a recall's legs plus HTTP requests) queues for a
    connection instead of failing with PoolTimeout."""
    from concurrent.futures import ThreadPoolExecutor

    def borrow(_):
        with pg_pool.connection(_URL) as c:
            c.execute("SELECT pg_sleep(0.05)")
            return True

    with ThreadPoolExecutor(40) as ex:
        assert all(ex.map(borrow, range(40)))
    assert pool.get_stats()["pool_size"] <= pg_pool._MAX_SIZE


def test_a_database_store_borrows_for_each_outermost_block(pool, conn):
    """With a pool open, Database borrows per outermost _conn() block: nested blocks
    share the connection, each block still commits on exit, and an error rolls back."""
    from ingestion.db import Database

    conn.execute("DELETE FROM private_sessions WHERE session_id LIKE 'pg-pool-%'")
    db = Database(_URL)
    with db._conn() as outer:
        with db._conn() as inner:
            assert inner is outer
            inner.execute("INSERT INTO private_sessions (session_id) VALUES ('pg-pool-db')")
        assert _private_row_count(conn, "pg-pool-db") == 1  # the inner block committed
    assert db._connection is None
    with pytest.raises(RuntimeError):
        with db._conn() as c:
            c.execute("INSERT INTO private_sessions (session_id) VALUES ('pg-pool-db-x')")
            raise RuntimeError("boom")
    assert _private_row_count(conn, "pg-pool-db-x") == 0
    assert db._connection is None
    db.close()  # nothing held: a no-op, not a closed pool connection
    with pg_pool.connection(_URL) as c:
        assert c.execute("SELECT 1").fetchone() == (1,)
    conn.execute("DELETE FROM private_sessions WHERE session_id LIKE 'pg-pool-%'")


# ---------------------------------------------------------------------------
# Server lifespan
# ---------------------------------------------------------------------------


@pytest.fixture()
def server_with(monkeypatch):
    """Reload mcp_server.server with SYNAPSE_DB_URL set to the given value (and the
    dotenv fallback neutralized), then restore the original module afterwards."""
    import mcp_server.server as s

    def load(db_url: str):
        monkeypatch.setenv("SYNAPSE_ENV_FILE", "/nonexistent/synapse-test.env")
        monkeypatch.setenv("SYNAPSE_DB_URL", db_url)
        monkeypatch.setenv("SYNAPSE_MACHINE_TOKEN", "")
        return importlib.reload(s)

    yield load
    monkeypatch.undo()
    importlib.reload(s)


async def test_lifespan_opens_the_pool_and_closes_it(server_with):
    s = server_with(_LIFESPAN_URL)
    assert _LIFESPAN_URL not in pg_pool._POOLS
    async with Client(s.mcp) as client:
        assert _LIFESPAN_URL in pg_pool._POOLS
        await client.list_tools()
    assert _LIFESPAN_URL not in pg_pool._POOLS


def test_http_app_opens_one_pool_across_stateless_requests(server_with):
    s = server_with(_LIFESPAN_URL)
    app = s.mcp.http_app(path="/mcp", stateless_http=True)
    with TestClient(app) as http:
        opened = pg_pool._POOLS[_LIFESPAN_URL]
        for _ in range(3):
            assert http.get("/health").status_code == 200
            http.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                headers={"Accept": "application/json, text/event-stream"},
            )
        assert pg_pool._POOLS[_LIFESPAN_URL] is opened
    assert _LIFESPAN_URL not in pg_pool._POOLS


async def test_a_server_without_a_database_boots_and_opens_no_pool(server_with):
    s = server_with("")
    assert s.DB_URL == ""
    before = dict(pg_pool._POOLS)
    async with Client(s.mcp) as client:
        assert [t.name for t in await client.list_tools()]
        assert pg_pool._POOLS == before and "" not in pg_pool._POOLS


def test_sigterm_closes_the_pool_and_exits_cleanly():
    """SIGTERM must reach the lifespan teardown. uvicorn drains and then re-raises it, and
    for any process but PID 1 Python's default SIGTERM killed it right there (exit 143),
    before the pool was closed. The real server, as an ordinary subprocess (as a local
    run or a container with an init runs it)."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {
        **os.environ,
        "SYNAPSE_ENV_FILE": "/nonexistent/synapse-test.env",
        "SYNAPSE_DB_URL": _DB_URL,
        "SYNAPSE_MACHINE_TOKEN": "pool-test-root-token",
        "MCP_PORT": str(port),
        "VOYAGE_API_KEY": "",
    }
    env.pop("LOGFIRE_TOKEN", None)
    proc = subprocess.Popen(
        [sys.executable, "-m", "mcp_server.server"],
        cwd=Path(__file__).resolve().parent.parent,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1).read()
                break
            except OSError:
                assert proc.poll() is None and time.monotonic() < deadline, "server didn't start"
                time.sleep(0.2)
        proc.send_signal(signal.SIGTERM)
        out, _ = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert "postgres pool opened" in out
    assert "postgres pool closed" in out, out[-2000:]
    assert proc.returncode == 0 and "Traceback" not in out, out[-2000:]


@pytest.mark.skipif(os.environ.get("SYNAPSE_TEST_POOL") != "1", reason="pooled run only")
def test_the_pooled_run_really_borrows():
    """Guards the pooled DB-suite run against silently testing the fallback instead."""
    from ingestion.db import Database

    assert pg_pool.pooled(_DB_URL)
    pool = pg_pool._POOLS[_DB_URL]
    before = pool.get_stats().get("requests_num", 0)
    db = Database(_DB_URL)
    db.known_project_slugs()
    resolve_caller(_DB_URL, token_hash_hex="0" * 64)
    assert pool.get_stats().get("requests_num", 0) >= before + 2
