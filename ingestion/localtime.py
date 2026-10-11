"""The user's local timezone (``SYNAPSE_TIMEZONE``) and the calendar day of an instant.

Timestamps are stored as instants (``timestamptz``), and the Postgres session runs in
UTC, so ``str(created_at)[:10]`` is the UTC calendar day. For a user in Toronto that is
the NEXT day for every turn after 20:00 EDT / 19:00 EST: "Got my first ride in finally"
said at 20:30 on 2026-10-10 was anchored to 2026-10-11. Anything that turns a
conversation timestamp into a calendar day (the extraction prompt's session date, the
timeline gate's turn date, the date recall serves on an episode) goes through
:func:`local_date`, and the reference time an LLM resolves "yesterday" against goes
through :func:`local_iso`.

Transcripts carry UTC timestamps only (no client offset), so the zone is a deployment
setting. The env var is read per call, like ``SYNAPSE_PERSONAL_SCOPE``, so a process
can be tested both ways. Only instants are converted: a bare ``YYYY-MM-DD`` (a
date-precision stamp) is returned as is.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, tzinfo
from functools import lru_cache
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

logger = logging.getLogger(__name__)

#: IANA zone used when ``SYNAPSE_TIMEZONE`` is unset or empty.
DEFAULT_TIMEZONE = "America/Toronto"


@lru_cache(maxsize=8)
def _zone(name: str) -> tzinfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        # Logged once per bad name (cached). UTC is the pre-setting behavior.
        logger.warning("SYNAPSE_TIMEZONE=%r is not a known IANA zone; using UTC", name)
        return UTC


def user_tz() -> tzinfo:
    """The zone the user lives in: ``SYNAPSE_TIMEZONE``, default America/Toronto."""
    return _zone((os.environ.get("SYNAPSE_TIMEZONE") or "").strip() or DEFAULT_TIMEZONE)


def _to_local(ts: Any) -> datetime | None:
    """``ts`` (a datetime or an ISO string) in the user's zone; None when it is not a
    parseable instant. A naive value is taken as UTC, the zone instants are stored in."""
    if isinstance(ts, datetime):
        dt = ts
    else:
        s = str(ts).strip()
        if len(s) <= 10:  # bare date: no time of day to shift
            return None
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(user_tz())


def local_date(ts: Any) -> str:
    """The user's calendar day (YYYY-MM-DD) of instant ``ts``. Anything that is not an
    instant falls back to its first 10 characters, the previous behavior."""
    dt = _to_local(ts)
    return dt.date().isoformat() if dt is not None else str(ts)[:10]


def local_iso(ts: Any) -> str:
    """``ts`` as ISO 8601 with the user's UTC offset: the same instant, local wall clock."""
    dt = _to_local(ts)
    return dt.isoformat() if dt is not None else str(ts)
