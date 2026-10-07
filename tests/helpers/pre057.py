"""A database view from before schema 057 (no ``surface_id`` on episodes/timeline_events).

The code ships before 057 is applied by hand, so the deploy-order tests need a database
without the provenance columns. The shared test tables cannot be altered (other test
files run against them, possibly concurrently), so this builds column-less copies of the
tables 057 touches in a throwaway schema and returns a DSN whose search_path puts that
schema first. Everything else (surfaces, notes, the KG, config_lane) still resolves to the
shared tables.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg

from ingestion.surfaces import OWN_SURFACE_PROBE


@contextmanager
def pre057_schema(conn: psycopg.Connection, db_url: str) -> Iterator[tuple[str, str]]:
    """Yield ``(dsn, schema)``. The schema is dropped and the probe cache cleared on exit."""
    schema = f"pre057_{uuid.uuid4().hex[:8]}"
    conn.execute(f"CREATE SCHEMA {schema}")
    try:
        for table in ("episodes", "timeline_events", "extraction_queue"):
            conn.execute(
                f"CREATE TABLE {schema}.{table} "
                f"(LIKE public.{table} INCLUDING ALL EXCLUDING INDEXES)"
            )
        conn.execute(f"ALTER TABLE {schema}.episodes DROP COLUMN surface_id")
        conn.execute(f"ALTER TABLE {schema}.timeline_events DROP COLUMN surface_id")
        conn.execute(f"ALTER TABLE {schema}.episodes ADD PRIMARY KEY (id)")
        conn.execute(f"ALTER TABLE {schema}.episodes ADD UNIQUE (session_id, sequence)")
        conn.execute(
            f"CREATE INDEX ON {schema}.episodes USING bm25 (id, content, session_id, project) "
            "WITH (key_field = 'id')"
        )
        conn.execute(f"ALTER TABLE {schema}.timeline_events ADD UNIQUE (source, source_ref)")
        sep = "&" if "?" in db_url else "?"
        yield f"{db_url}{sep}options=-csearch_path%3D{schema},public", schema
    finally:
        conn.execute(f"DROP SCHEMA {schema} CASCADE")
        OWN_SURFACE_PROBE.reset()
