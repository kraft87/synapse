"""A DSN onto a database with none of Synapse's tables, for "migration missing" tests.

The routes no longer hide a missing table behind an empty answer or a canned 503; they
let the database's own error through. Proving that needs a connection on which every
table is absent, without touching the shared test tables: an empty throwaway schema as
the whole search_path does it.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg


@contextmanager
def schemaless_dsn(conn: psycopg.Connection, db_url: str) -> Iterator[str]:
    """Yield a DSN whose search_path is one empty schema; drop the schema on exit."""
    schema = f"empty_{uuid.uuid4().hex[:8]}"
    conn.execute(f"CREATE SCHEMA {schema}")
    try:
        sep = "&" if "?" in db_url else "?"
        yield f"{db_url}{sep}options=-csearch_path%3D{schema}"
    finally:
        conn.execute(f"DROP SCHEMA {schema} CASCADE")
