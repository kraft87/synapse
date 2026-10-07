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
  * database AHEAD of this build -> warn and continue. Migrations are
    additive, so an older build runs fine on a newer schema; exiting here
    turned "migration applied before the new image landed" (or an older image
    redeployed by watchtower after the stamp) into a full outage.
  * database behind this build ONLY by migrations marked optional -> warn and
    continue. A migration carries the :data:`OPTIONAL_MARKER` header line when
    the code that ships with it probes for what it adds and keeps its
    pre-migration behaviour until it is applied. That lets the image land
    before the migration is applied by hand, without a crash loop. Any
    unmarked migration in the gap still exits.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import psycopg

logger = logging.getLogger(__name__)

_SCHEMA_FILE_RE = re.compile(r"^(\d{3})_.*\.sql$")

# repo checkout: <repo>/schema ; container image: /app/ingestion/../schema
# = /app/schema (Dockerfile ships it); /schema is the first-boot init mount.
_SCHEMA_DIR_CANDIDATES = (
    Path(__file__).resolve().parent.parent / "schema",
    Path("/schema"),
)


#: A migration header line declaring that this build runs correctly without it.
OPTIONAL_MARKER = "-- schema-check: optional"


def _migrations(schema_dir: Path | None) -> list[tuple[str, Path]]:
    """``(NNN, path)`` for every migration in the first schema dir that has any,
    sorted by number. Empty when no schema directory can be found."""
    candidates = (schema_dir,) if schema_dir is not None else _SCHEMA_DIR_CANDIDATES
    for d in candidates:
        if not d.is_dir():
            continue
        found = sorted(
            (m.group(1), f) for f in d.glob("*.sql") if (m := _SCHEMA_FILE_RE.match(f.name))
        )
        if found:
            return found
    return []


def is_optional_migration(path: Path) -> bool:
    """True when the migration's text carries :data:`OPTIONAL_MARKER` on a line of its own."""
    try:
        return any(line.strip() == OPTIONAL_MARKER for line in path.read_text().splitlines())
    except OSError:
        return False


def expected_schema_version(schema_dir: Path | None = None) -> str | None:
    """Highest ``NNN`` prefix among the schema files shipped with this build.

    Returns None when no schema directory can be found — callers treat
    that as "cannot verify, don't block".
    """
    found = _migrations(schema_dir)
    return found[-1][0] if found else None


def required_schema_version(schema_dir: Path | None = None) -> str | None:
    """Highest ``NNN`` this build cannot run without: the newest migration NOT marked
    optional. An optional migration only relaxes the check when it sits in the tail;
    one followed by a required migration is covered by that migration's number."""
    required = [num for num, path in _migrations(schema_dir) if not is_optional_migration(path)]
    return required[-1] if required else None


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
    """Exit the process when the database schema is unstamped or behind a migration this
    build requires (an unmarked one; see :func:`required_schema_version`)."""
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
    if applied is not None and applied.isdigit() and int(applied) > int(expected):
        logger.warning(
            "Database is at schema %s, ahead of this build's schema %s; continuing "
            "(migrations are additive). Deploy the newer image to clear this.",
            applied,
            expected,
        )
        return
    required = required_schema_version(schema_dir)
    if (
        applied is not None
        and applied.isdigit()
        and required is not None
        and int(applied) >= int(required)
    ):
        logger.warning(
            "Database is at schema %s, behind this build's schema %s, but every newer "
            "migration is marked optional (this build keeps its pre-migration behaviour "
            "without them); continuing. Run scripts/apply_schema.sh to apply them.",
            applied,
            expected,
        )
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


class ColumnProbe:
    """Cached runtime answer to "has this optional migration been applied here yet?".

    Code that ships with an optional migration (:data:`OPTIONAL_MARKER`) keeps its
    pre-migration behaviour until what the migration adds exists. ``sql`` must return
    a row exactly when it does, and no row otherwise (one form works for dict and tuple
    rows alike). Answers are cached per ``key`` (normally the database URL): a positive
    one for the life of the process, a negative one for ``reprobe_s`` only, so applying
    the migration under a running service takes effect without a restart.

    The probe runs on the CALLER's connection, inside a transaction block (a savepoint
    when one is already open), so a probe failure never poisons the caller's
    transaction. A failure answers False without caching. State is plain dicts: threads
    race only to write the same answer, and the worst case is one extra probe.
    """

    def __init__(self, sql: str, what: str, reprobe_s: float = 60.0) -> None:
        self._sql = sql
        self._what = what
        self.reprobe_s = reprobe_s
        self._ok: dict[str, bool] = {}
        self._probed_at: dict[str, float] = {}
        self._warned: set[str] = set()

    def ready(self, conn: Any, key: str = "") -> bool:
        """True when the migration's columns exist on ``conn``'s database."""
        if self._ok.get(key):
            return True
        last = self._probed_at.get(key)
        if last is not None and time.monotonic() - last < self.reprobe_s:
            return False
        self._probed_at[key] = time.monotonic()
        try:
            with conn.transaction():
                row = conn.execute(self._sql).fetchone()
        except Exception as e:
            logger.warning("%s probe failed (%s); treating it as not applied", self._what, e)
            return False
        if row is None:
            self.missing(key)
            return False
        self._ok[key] = True
        return True

    def cached(self, key: str = "") -> bool | None:
        """The cached answer for ``key`` without touching the database: True (applied),
        False (absent, and not yet due a re-probe), or None (never probed, or due)."""
        if self._ok.get(key):
            return True
        last = self._probed_at.get(key)
        if last is not None and time.monotonic() - last < self.reprobe_s:
            return False
        return None

    def missing(self, key: str = "", err: BaseException | None = None) -> None:
        """Record that the columns are absent (a negative probe, or a query that hit
        ``UndefinedColumn`` after a positive one). One log line per key, not per call."""
        self._ok[key] = False
        self._probed_at[key] = time.monotonic()
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(
                "%s is not applied%s: keeping the pre-migration behaviour until it is",
                self._what,
                f" ({err})" if err is not None else "",
            )

    def note_error(self, err: BaseException, key: str = "") -> bool:
        """True (and the cache flipped to "missing") when ``err`` is an UndefinedColumn,
        i.e. the migration was rolled back under a running service."""
        if isinstance(err, psycopg.errors.UndefinedColumn):
            self.missing(key, err)
            return True
        return False

    def reset(self) -> None:
        """Forget every cached answer (tests)."""
        self._ok.clear()
        self._probed_at.clear()
        self._warned.clear()
