"""Boot-time schema-version guard.

``scripts/apply_schema.sh`` stamps ``synapse_meta.schema_version`` (table
from migration 034) after a full run; every long-lived service (poller,
MCP server, dream) calls :func:`check_schema_version` at startup. A
database that is missing the stamp or behind the code's schema fails FAST
with upgrade instructions, instead of surfacing as confusing runtime
errors mid-request.

Fail-open cases — the guard never blocks on its own infrastructure:
  * ``SYNAPSE_SCHEMA_CHECK=0`` skips the check entirely (escape hatch).
  * schema/ directory not found (unusual dev layout) -> skip.
  * database unreachable (compose start-up race) -> warn and continue; the
    service's own connection handling owns that failure mode.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

import psycopg

logger = logging.getLogger(__name__)

_SCHEMA_FILE_RE = re.compile(r"^(\d{3})_.*\.sql$")

# repo checkout: <repo>/schema ; container image: /app/ingestion/../schema
# = /app/schema (Dockerfile ships it); /schema is the first-boot init mount.
_SCHEMA_DIR_CANDIDATES = (
    Path(__file__).resolve().parent.parent / "schema",
    Path("/schema"),
)


def expected_schema_version(schema_dir: Path | None = None) -> str | None:
    """Highest ``NNN`` prefix among the schema files shipped with this build.

    Returns None when no schema directory can be found — callers treat
    that as "cannot verify, don't block".
    """
    candidates = (schema_dir,) if schema_dir is not None else _SCHEMA_DIR_CANDIDATES
    for d in candidates:
        if not d.is_dir():
            continue
        nums = sorted(m.group(1) for f in d.glob("*.sql") if (m := _SCHEMA_FILE_RE.match(f.name)))
        if nums:
            return nums[-1]
    return None


def applied_schema_version(db_url: str) -> str | None:
    """The database's ``schema_version`` stamp, or None if never stamped.

    None covers both "synapse_meta doesn't exist" (database predates 039)
    and "table exists but no stamp row" (partial init).
    Raises on connection failure — the caller decides what that means.
    """
    with psycopg.connect(db_url, connect_timeout=10) as conn:
        try:
            row = conn.execute(
                "SELECT value FROM synapse_meta WHERE key = 'schema_version'"
            ).fetchone()
        except psycopg.errors.UndefinedTable:
            return None
        return None if row is None else str(row[0])


def check_schema_version(db_url: str, schema_dir: Path | None = None) -> None:
    """Exit the process when the database schema doesn't match this build."""
    if os.environ.get("SYNAPSE_SCHEMA_CHECK", "1") == "0":
        return
    expected = expected_schema_version(schema_dir)
    if expected is None:
        logger.warning("schema check skipped: no schema directory found in this layout")
        return
    try:
        applied = applied_schema_version(db_url)
    except Exception as e:
        logger.warning("schema check skipped (database not reachable yet): %s", e)
        return
    if applied == expected:
        return
    state = (
        "has no schema_version stamp (it predates the stamp, or init was interrupted)"
        if applied is None
        else f"is at schema {applied}"
    )
    logger.critical(
        "Database %s but this build expects schema %s. "
        "Run scripts/apply_schema.sh against your database to upgrade it "
        "(see README > Upgrading), then restart. "
        "To skip this check: SYNAPSE_SCHEMA_CHECK=0.",
        state,
        expected,
    )
    sys.exit(1)
