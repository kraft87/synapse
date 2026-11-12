"""The compaction cut on self-exclusion (``self_since``), pure-logic half.

With ``self_session`` alone the calling session is hidden whole, as it always was. With
``self_since`` too (the time of the caller's last compaction, sent by the client hook)
only its turns at or after that time are hidden: the ones before it are gone from the
caller's context, so they serve again. Anything unusable means no cut. The database half
(facts and the supersession overlay) is tests/test_self_session_facts.py; the telemetry
envelope is tests/test_telemetry_kinds.py.
"""

from __future__ import annotations

from contextlib import nullcontext
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest

from mcp_server.recall import Recall
from mcp_server.recall_sources import _self_only_sql

_ME = "MY-SESSION"
_CUT = datetime(2026, 11, 12, 14, 0, tzinfo=UTC)

# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-11-12T14:48:54.738Z", datetime(2026, 11, 12, 14, 48, 54, 738000, tzinfo=UTC)),
        ("2026-11-12T09:48:54-05:00", datetime(2026, 11, 12, 14, 48, 54, tzinfo=UTC)),
        ("2026-11-12T16:48:54+02:00", datetime(2026, 11, 12, 14, 48, 54, tzinfo=UTC)),
        ("2026-11-12T14:48:54", datetime(2026, 11, 12, 14, 48, 54, tzinfo=UTC)),  # naive = UTC
        (" 2026-11-12T14:48:54Z ", datetime(2026, 11, 12, 14, 48, 54, tzinfo=UTC)),
    ],
)
def test_the_cut_parses_to_an_aware_utc_time(raw, expected):
    cut = Recall._self_cutoff(_ME, raw)
    assert cut == expected
    assert cut is not None and cut.utcoffset() == timedelta(0)


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "   ",
        "yesterday",
        "2026-13-01T00:00:00Z",
        "2026-11-12T25:00:00Z",
        "0001-01-01T00:00:00+05:00",  # before the earliest UTC time there is
        1763000000,
        1763000000.5,
        ["2026-11-12T14:48:54Z"],
        {"at": "2026-11-12T14:48:54Z"},
    ],
    ids=repr,
)
def test_an_unusable_cut_is_no_cut(raw):
    assert Recall._self_cutoff(_ME, raw) is None


@pytest.mark.parametrize("self_session", [None, ""])
def test_a_cut_without_a_session_is_ignored(self_session):
    assert Recall._self_cutoff(self_session, "2026-11-12T14:48:54Z") is None


# ---------------------------------------------------------------------------
# _exclude_self: the episode half
# ---------------------------------------------------------------------------


def _ep(eid: str, session: str | None, at: Any = None, **extra: Any) -> dict[str, Any]:
    return {"id": eid, "session_id": session, "created_at": at, **extra}


_POOL = [
    _ep("e:1", _ME, _CUT - timedelta(hours=1)),  # before the cut
    _ep("e:2", _ME, _CUT),  # at the cut: still in context
    _ep("e:3", _ME, _CUT + timedelta(hours=1)),
    _ep("e:4", "other", _CUT + timedelta(hours=1)),
    _ep("e:5", "other", None),
    _ep("e:6", None, None),
]


def _ids(rows: list[dict[str, Any]]) -> list[str]:
    return [r["id"] for r in rows]


def test_without_a_cut_the_whole_session_is_hidden():
    assert _ids(Recall._exclude_self(_POOL, _ME)) == ["e:4", "e:5", "e:6"]
    assert _ids(Recall._exclude_self(_POOL, _ME, since=None)) == ["e:4", "e:5", "e:6"]


def test_with_a_cut_only_the_turns_before_it_serve_again():
    assert _ids(Recall._exclude_self(_POOL, _ME, since=_CUT)) == ["e:1", "e:4", "e:5", "e:6"]


def test_the_cut_compares_instants_not_wall_clocks():
    east = timezone(timedelta(hours=5))
    pool = [
        _ep("e:1", _ME, datetime(2026, 11, 12, 18, 59, tzinfo=east)),  # 13:59 UTC
        _ep("e:2", _ME, datetime(2026, 11, 12, 19, 1, tzinfo=east)),  # 14:01 UTC
    ]
    assert _ids(Recall._exclude_self(pool, _ME, since=_CUT)) == ["e:1"]


def test_a_naive_created_at_is_utc():
    pool = [
        _ep("e:1", _ME, datetime(2026, 11, 12, 13, 59)),
        _ep("e:2", _ME, datetime(2026, 11, 12, 14, 1)),
    ]
    assert _ids(Recall._exclude_self(pool, _ME, since=_CUT)) == ["e:1"]


@pytest.mark.parametrize("at", [None, "2026-11-12T13:00:00+00:00", 0, "missing"])
def test_an_own_turn_with_no_usable_time_stays_hidden(at):
    row = {"id": "e:1", "session_id": _ME} if at == "missing" else _ep("e:1", _ME, at)
    assert Recall._exclude_self([row, _ep("e:2", "other")], _ME, since=_CUT) == [
        _ep("e:2", "other")
    ]


def test_rows_come_back_unchanged_and_in_order():
    out = Recall._exclude_self(_POOL, _ME, since=_CUT)
    assert all(a is b for a, b in zip(out, [_POOL[0], *_POOL[3:]], strict=True))


# ---------------------------------------------------------------------------
# The facts half: no cut is exactly the predicate it always was
# ---------------------------------------------------------------------------

_TODAY = (
    "(jsonb_typeof(r.episodes) = 'array' AND jsonb_array_length(r.episodes) > 0 "
    "AND NOT EXISTS (SELECT 1 FROM jsonb_array_elements_text(r.episodes) AS src(eid) "
    "WHERE NOT EXISTS (SELECT 1 FROM episodes e "
    "WHERE e.id = CASE WHEN src.eid ~ '^[0-9]+$' THEN src.eid::bigint END "
    "AND e.session_id = %s)))"
)


def test_without_a_cut_the_predicate_is_unchanged():
    assert _self_only_sql("r") == _TODAY
    assert _self_only_sql("r", False) == _TODAY


def test_the_cut_binds_inside_the_source_lookup_after_the_session():
    sql = _self_only_sql("r", True)
    assert sql == _TODAY.replace("e.session_id = %s", "e.session_id = %s AND e.created_at >= %s")
    assert sql.count("%s") == 2


class _Rows:
    def fetchall(self) -> list[Any]:
        return []


class _CapturePG:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def execute(self, sql: str, params: Any = ()) -> _Rows:
        self.calls.append((sql, tuple(params)))
        return _Rows()


def _captured(**kwargs: Any) -> tuple[str, tuple[Any, ...]]:
    r = Recall("", "")
    pg = _CapturePG()
    r._pg = lambda: nullcontext(pg)  # type: ignore[method-assign]
    r._self_session_facts(["f-1"], _ME, **kwargs)
    (call,) = pg.calls
    return call


def test_the_fact_lookup_binds_the_cut_only_when_there_is_one():
    sql, params = _captured()
    assert params == (["f-1"], _ME) and "created_at" not in sql
    sql, params = _captured(since=_CUT)
    assert params == (["f-1"], _ME, _CUT) and "e.created_at >= %s" in sql
