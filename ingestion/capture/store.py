"""Store what :func:`ingestion.capture.detect.decide` accepts: the caps, the dedup, the insert.

:func:`capture_calls` runs after the episodes are committed, in a transaction of its own, so
nothing here can fail or hold up the episode write; each call has its own savepoint, so one
bad row costs one artifact. Called with ``dry_run=True`` it does every step but the insert, so
the report of ``POST /capture/dry-run`` is exactly what a live capture would do.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from ingestion.capture.calls import ToolCall
from ingestion.capture.detect import Capture, Skip, decide
from ingestion.db import Database
from ingestion.web_artifacts import (
    artifact_stored,
    build_row,
    count_fetched_between,
    insert_artifact,
)

logger = logging.getLogger(__name__)

#: Captures one call to :func:`capture_calls` stores, and captures per UTC day of the call.
#: Runaway guards (a loop re-running a search; a backfill of a huge history), not quotas: a
#: person doing research does not get near them. The rest are skipped, counted as
#: ``cap_post`` / ``cap_day``.
MAX_CAPTURES_PER_POST = 200
MAX_CAPTURES_PER_DAY = 1000


def enabled() -> bool:
    """``SYNAPSE_WEB_CAPTURE=on`` turns server-side capture on; off (the default) stores
    nothing. The dry run works either way."""
    value = os.environ.get("SYNAPSE_WEB_CAPTURE", "").strip().lower()
    if value not in ("", "on", "off"):
        raise ValueError(f"SYNAPSE_WEB_CAPTURE must be 'on' or 'off', not {value!r}")
    return value == "on"


@dataclass
class CaptureResult:
    #: every call given, in order, with what was (or, dry-run, would be) done with it
    outcomes: list[tuple[ToolCall, Capture | Skip]] = field(default_factory=list)
    #: calls that raised; the rest of the batch is unaffected
    errors: int = 0

    @property
    def captured(self) -> int:
        return sum(isinstance(d, Capture) for _, d in self.outcomes)

    @property
    def skipped(self) -> Counter[str]:
        return Counter(d.reason for _, d in self.outcomes if isinstance(d, Skip))


def capture_calls(
    db: Database,
    calls: Iterable[ToolCall],
    *,
    is_private: Callable[[str | None], bool],
    dry_run: bool = False,
) -> CaptureResult:
    """Capture the web research in ``calls``.

    A call from a private session is never captured (``is_private`` fails closed: a lookup
    that raises skips the call). A call is stored once per ``tool_use_id`` and once per
    (canonical URL, content hash)."""
    calls = list(calls)
    result = CaptureResult()
    if not calls:
        return result
    # What this call has stored (a dry run: would have stored), for the dedup and the caps.
    seen: set[str] = set()
    per_day: dict[datetime, int] = {}
    stored = 0
    unlinked = {c.span_id for c in calls if c.span_id and c.episode_id is None}
    episode_ids = {} if dry_run else db.episode_ids_by_span(unlinked)

    with db.atomic() as conn:

        def capture(call: ToolCall) -> Capture | Skip:
            if is_private(call.session_id):
                return Skip("private")
            decision = decide(call)
            if isinstance(decision, Skip):
                return decision
            fetched_at = call.timestamp or datetime.now(UTC)
            if fetched_at.tzinfo is None:
                fetched_at = fetched_at.replace(tzinfo=UTC)
            row = build_row(
                decision.parsed,
                tool_name=call.tool_name,
                tool_use_id=call.tool_use_id,
                session_id=call.session_id,
                fetched_at=fetched_at,
                parent_episode_id=call.episode_id or episode_ids.get(call.span_id or ""),
                metadata={
                    "capture": decision.source,
                    "project": call.project,
                    "surface_id": call.surface_id,
                    # Generic captures are searchable, not KG-extracted; a rule opts in.
                    "kg_extract": False,
                },
            )
            url, digest = row.get("url_canonical"), row.get("content_hash")
            keys = {f"id:{call.tool_use_id}"} | ({f"content:{url}|{digest}"} if digest else set())
            if keys & seen or artifact_stored(conn, call.tool_use_id, url, digest):
                return Skip("duplicate")
            if stored >= MAX_CAPTURES_PER_POST:
                return Skip("cap_post")
            day = fetched_at.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
            if day not in per_day:
                per_day[day] = count_fetched_between(conn, day, day + timedelta(days=1))
            if per_day[day] >= MAX_CAPTURES_PER_DAY:
                return Skip("cap_day")
            if not dry_run:
                insert_artifact(conn, row)
            seen.update(keys)
            per_day[day] += 1
            return decision

        for call in calls:
            try:
                with conn.transaction():  # a savepoint: one bad row costs one artifact
                    outcome = capture(call)
            except Exception:
                logger.warning("web capture failed for %s", call.tool_name, exc_info=True)
                result.errors += 1
                continue
            stored += isinstance(outcome, Capture)
            result.outcomes.append((call, outcome))
    return result
