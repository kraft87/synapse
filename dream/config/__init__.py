"""dream->config lane. This module only holds the schema probe, so the review routes can ask it
without importing the lane (and the whole skills lane under it)."""

from __future__ import annotations

from ingestion.schema_check import ColumnProbe

#: Schema 065 adds the four columns below. The image ships before it is applied, so the lane
#: asks this probe first and idles with one log line until all four exist; the review routes
#: leave the recommendation and the days pending out until they do.
IDENTITY_PROBE = ColumnProbe(
    "SELECT 1 FROM pg_attribute "
    "WHERE attrelid = to_regclass('config_lane.config_proposals') "
    "  AND attname IN ('embedding', 'proposed_at', 'recommendation', 'coverage') "
    "  AND NOT attisdropped HAVING count(*) = 4",
    "schema 065 (config_proposals.embedding, proposed_at, recommendation, coverage)",
)


def identity_ready(conn) -> bool:
    """Schema 065 is applied on `conn`'s database."""
    return IDENTITY_PROBE.ready(conn, conn.info.dsn)
