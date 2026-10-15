"""Small helpers shared by the dashboard route modules: gist and provenance shaping, the
filter and keyset-cursor builders, the flag set, the limit/offset parsers and the in-process
cache the two aggregate routes use.
"""

from __future__ import annotations

import base64
import re
import time
from collections.abc import Callable
from typing import Any

import psycopg
from starlette.requests import Request

from mcp_server.http_helpers import RouteError

_GIST_MAX = 200
# Episode text read for a gist: _gist keeps the first non-empty line, so the head of
# the content is enough, and a 100 KB turn never leaves Postgres to make 200 chars.
_GIST_SOURCE_MAX = 2000
_SNIPPET_MAX = 240


# Coarse timeline salience (0/1/2) → the 0.3/0.6/0.9 readout the feed + type ramp use.
_SAL_MAP = {0: 0.3, 1: 0.6, 2: 0.9}


def _gist(text: str | None) -> str:
    """First non-empty line, internal whitespace collapsed, hard-capped at 200 chars."""
    if not text:
        return ""
    for line in text.splitlines():
        collapsed = " ".join(line.split())
        if collapsed:
            return collapsed[:_GIST_MAX]
    return ""


def _provenance(episodes: Any) -> int | None:
    """First element of a fact's ``episodes`` jsonb array as an int, else None.

    psycopg parses jsonb to a Python list; the array holds source episode ids as
    numbers or (defensively) numeric strings."""
    if isinstance(episodes, list) and episodes:
        first = episodes[0]
        if isinstance(first, bool):  # bool is an int subclass — exclude it
            return None
        if isinstance(first, int):
            return first
        if isinstance(first, str) and first.isdigit():
            return int(first)
    return None


_EP_REF_RE = re.compile(r"ep:(\d+)(?:#\d+)?")


def _episode_id_from_ref(source_ref: Any) -> int | None:
    """Resolve a timeline event's source_ref to an episode id, else None. A turn's first
    event carries 'ep:<id>' and its k-th 'ep:<id>#<k>' (ingestion/timeline_gate.py);
    git-sourced refs are SHAs, which resolve to null."""
    if isinstance(source_ref, str):
        m = _EP_REF_RE.fullmatch(source_ref.strip())
        if m:
            return int(m.group(1))
    return None


def _label_filter(col: str, value: str) -> tuple[str, list[Any]]:
    """WHERE fragment for a project/source filter. 'untagged' is the catalog's bucket for
    NULL and '' (a literal 'untagged' label lands in it too); a real label is a plain
    equality, so the column's index can serve it. ``col`` is a fixed identifier from
    this module, never user input."""
    if value == "untagged":
        return f"({col} IS NULL OR {col} = '' OR {col} = 'untagged')", []
    return f"{col} = %s", [value]


def _domain_filter(group_id: str) -> tuple[str, list[Any]]:
    """WHERE fragment for the group filter on timeline_events. ``domain`` is NULL for an
    unlabelled event (legacy rows, a gate parse miss, a pushed non-git event) and schema 038
    makes a NULL FAIL OPEN at read, so it stays visible under either group."""
    return "(domain = %s OR domain IS NULL)", [group_id]


def _like_contains(q: str) -> str:
    """ILIKE pattern for "name contains q" with q taken literally: '%' and '_' in user text
    are not wildcards. Backslash is Postgres's default LIKE escape, so it is escaped too."""
    escaped = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _encode_cursor(ts_iso: str, type_: str, id_str: str) -> str:
    return base64.urlsafe_b64encode(f"{ts_iso}|{type_}|{id_str}".encode()).decode()


def _decode_cursor(cursor: str) -> tuple[str, str] | None:
    """Return (ts_iso, id_str) from an opaque cursor, or None if unparseable.

    The keyset predicate is (ts, id) per the contract; the type segment rides in the
    cursor string for debuggability but is not used in the SQL comparison."""
    try:
        raw = base64.urlsafe_b64decode(cursor.encode()).decode()
        ts_iso, _type, id_str = raw.split("|", 2)
        return ts_iso, id_str
    except Exception:
        return None


def _flag_set(conn: psycopg.Connection[dict[str, Any]]) -> set[tuple[str, str]]:
    """The active (kind, item_id) flag pairs — loaded once per request to mark items."""
    rows = conn.execute(
        "SELECT kind, item_id FROM dashboard_flags WHERE removed_at IS NULL"
    ).fetchall()
    return {(r["kind"], r["item_id"]) for r in rows}


def _count(conn: psycopg.Connection[dict[str, Any]], sql: str, params: Any) -> int:
    """Scalar COUNT for a query aliased AS c (0 if — impossibly — no row comes back)."""
    row = conn.execute(sql, params).fetchone()
    return int(row["c"]) if row else 0


def _limit(raw: str | None, default: int, cap: int) -> int:
    try:
        return max(1, min(int(raw), cap)) if raw is not None else default
    except (TypeError, ValueError):
        return default


def _offset(raw: str | None) -> int:
    try:
        return max(0, int(raw)) if raw is not None else 0
    except (TypeError, ValueError):
        return 0


def _episode_id(request: Request) -> int:
    """The ``{id}`` path parameter as an episode id (the 400 when it is not digits)."""
    raw = str(request.path_params["id"])
    if not raw.isdigit():
        raise RouteError("bad episode id")
    return int(raw)


class _TtlCache:
    """One computed value kept for ``ttl`` seconds (monotonic) in this process. Owned by a
    ``register()`` call, not a module global, so a fresh register() in tests starts empty and
    one test's inserts never serve stale counts to another's."""

    def __init__(self, ttl: float) -> None:
        self._ttl = ttl
        self._ts = 0.0
        self._data: Any = None

    def get(self, work: Callable[[], Any]) -> Any:
        now = time.monotonic()
        if self._data is not None and now - self._ts < self._ttl:
            return self._data
        data = work()
        self._ts = now
        self._data = data
        return data
