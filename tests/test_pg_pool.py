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

_DB_URL = os.environ.get(
    "SYNAPSE_TEST_URL", "postgresql://synapse:synapse@127.0.0.1:5432/synapse_test"
)

try:
    _probe = psycopg.connect(_DB_URL, connect_timeout=2)
    _probe.close()
except Exception:  # pragma: no cover - environment dependent
    pytest.skip("no test DB reachable", allow_module_level=True)

from fastmcp import Client  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from ingestion import pg_pool  # noqa: E402
from ingestion.surfaces import resolve_caller, token_hash  # noqa: E402


@pytest.fixture()
def pool():
    pg_pool.open_pool(_DB_URL)
    pg_pool._POOLS[_DB_URL].wait(timeout=10)
    yield pg_pool._POOLS[_DB_URL]
    pg_pool.close_pool(_DB_URL)


def _backend_pid() -> int:
    with pg_pool.connection(_DB_URL) as conn:
        row = conn.execute("SELECT pg_backend_pid()").fetchone()
        assert row is not None
        return int(row[0])


def test_without_a_pool_each_block_gets_its_own_connection_and_closes_it():
    assert _DB_URL not in pg_pool._POOLS
    with pg_pool.connection(_DB_URL) as conn:
        assert conn.autocommit
        held = conn
    assert held.closed
    assert _backend_pid() != _backend_pid()


def test_a_pool_reuses_its_connections(pool):
    pids = {_backend_pid() for _ in range(10)}
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
        assert resolve_caller(_DB_URL, token_hash_hex=token_hash("pool-test-token")).known
    for _ in range(4):
        with pg_pool.connection(_DB_URL) as c:
            assert c.execute("SELECT 1, 2").fetchone() == (1, 2)
    conn.execute("DELETE FROM surfaces")


def test_a_failed_statement_does_not_poison_the_connection(pool):
    for _ in range(4):
        with pytest.raises(psycopg.errors.UndefinedTable):
            with pg_pool.connection(_DB_URL) as c:
                c.execute("SELECT 1 FROM no_such_table_pg_pool_test")
    for _ in range(4):
        with pg_pool.connection(_DB_URL) as c:
            assert c.execute("SELECT 1").fetchone() == (1,)


def test_connections_killed_by_a_postgres_restart_are_replaced_on_borrow(pool, conn):
    pids = {_backend_pid() for _ in range(6)}
    for pid in pids:
        conn.execute("SELECT pg_terminate_backend(%s)", (pid,))
    for _ in range(6):
        assert _backend_pid() not in pids


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
    s = server_with(_DB_URL)
    assert _DB_URL not in pg_pool._POOLS
    async with Client(s.mcp) as client:
        assert _DB_URL in pg_pool._POOLS
        await client.list_tools()
    assert _DB_URL not in pg_pool._POOLS


def test_http_app_opens_one_pool_across_stateless_requests(server_with):
    s = server_with(_DB_URL)
    app = s.mcp.http_app(path="/mcp", stateless_http=True)
    with TestClient(app) as http:
        opened = pg_pool._POOLS[_DB_URL]
        for _ in range(3):
            assert http.get("/health").status_code == 200
            http.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                headers={"Accept": "application/json, text/event-stream"},
            )
        assert pg_pool._POOLS[_DB_URL] is opened
    assert _DB_URL not in pg_pool._POOLS


async def test_a_server_without_a_database_boots_and_opens_no_pool(server_with):
    s = server_with("")
    assert s.DB_URL == ""
    async with Client(s.mcp) as client:
        assert [t.name for t in await client.list_tools()]
        assert not pg_pool._POOLS
    assert not pg_pool._POOLS


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
