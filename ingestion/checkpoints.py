"""Shared ingestion_state checkpoint helpers.

`WebArtifactsIngester` and `ResearchArchiveCapture` each track a per-source
high-water mark in the ``ingestion_state`` table through an injected raw
psycopg connection. The get/set pair (same table, same tz-normalization to
UTC, same dict-row/tuple-row defensiveness) lives here once; both classes
delegate. Distinct from ``Database.get_watermark``/``set_watermark`` in
``ingestion/db.py``, which owns its own pooled connection — different layer,
deliberately not unified.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import psycopg


def checkpoint_get(conn: psycopg.Connection[Any], source: str) -> datetime | None:
    row = conn.execute(
        "SELECT last_ingested_at FROM ingestion_state WHERE source = %s",
        (source,),
    ).fetchone()
    if not row:
        return None
    # dict_row vs tuple-row
    val = row["last_ingested_at"] if isinstance(row, dict) else row[0]
    if not isinstance(val, datetime):
        return None
    return val.replace(tzinfo=UTC) if val.tzinfo is None else val


def checkpoint_set(conn: psycopg.Connection[Any], source: str, ts: datetime) -> None:
    conn.execute(
        """
        INSERT INTO ingestion_state (source, last_ingested_at)
        VALUES (%s, %s)
        ON CONFLICT (source) DO UPDATE SET last_ingested_at = EXCLUDED.last_ingested_at
        """,
        (source, ts),
    )
