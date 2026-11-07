"""The board's per-type quotas and the timeline trim (mcp_server/board.py ``_keep_order``,
``_fit_notes``, ``_one_per_day``). Pure Python, no DB: the DB-backed tests in
test_board.py run the same policy through ``build_board``, and test_board_cap.py pins the
binary-searched cap to a one-drop-at-a-time loop in this order."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

from mcp_server.board import _EVENT_DAYS, _fit_notes, _keep_order, _one_per_day

_T0 = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
_TYPES = ("feedback", "user", "project", "reference")


def _note(nid: int, type_: str, *, age_days: int = 0, helpful: int | None = None, hook="h"):
    note: dict[str, Any] = {
        "id": nid,
        "type": type_,
        "hook": hook,
        "updated_at": _T0 - timedelta(days=age_days),
    }
    if helpful is not None:
        note["helpful"] = helpful
    return note


def _pool(per_type: dict[str, int], hook: str = "h" * 100) -> list[dict[str, Any]]:
    """``per_type`` notes of each type, newest first inside a type, ids unique."""
    out, nid = [], 0
    for t, n in per_type.items():
        for age in range(n):
            nid += 1
            out.append(_note(nid, t, age_days=age, hook=hook))
    return out


def _mix(notes: list[dict[str, Any]], order: list[int], k: int) -> Counter[str]:
    return Counter(notes[i]["type"] for i in order[:k])


def test_every_prefix_follows_the_weights():
    notes = _pool({"feedback": 40, "user": 40, "project": 40, "reference": 40})
    order = _keep_order(notes)
    assert sorted(order) == list(range(len(notes)))  # a permutation: nothing lost or doubled
    # Weights 4:3:2:1 -> every ten picks hold 4/3/2/1, and any prefix is within one of
    # its proportional share.
    for k in (10, 20, 30, 40):
        mix = _mix(notes, order, k)
        assert [mix[t] for t in _TYPES] == [k * w // 10 for w in (4, 3, 2, 1)]
    for k in range(1, 61):
        mix = _mix(notes, order, k)
        for t, w in zip(_TYPES, (4, 3, 2, 1), strict=True):
            assert abs(mix[t] - k * w / 10) <= 1, (k, t, mix)


def test_a_type_with_few_notes_hands_its_share_to_the_others():
    notes = _pool({"feedback": 60, "user": 3, "project": 0, "reference": 60})
    order = _keep_order(notes)
    mix = _mix(notes, order, 40)
    assert mix["user"] == 3 and mix["project"] == 0
    assert mix["feedback"] + mix["reference"] == 37
    assert mix["feedback"] > 3 * mix["reference"]  # 4:1 between the two that remain
    # One type alone gets the whole budget.
    only = _pool({"reference": 7})
    assert _keep_order(only) == [0, 1, 2, 3, 4, 5, 6]


def test_within_a_type_helpful_then_newest_then_highest_id():
    notes = [
        _note(1, "user", age_days=0, helpful=0),
        _note(2, "user", age_days=30, helpful=2),
        _note(3, "user", age_days=60, helpful=5),
        _note(4, "user", age_days=1, helpful=0),
        _note(5, "user", age_days=1, helpful=0),  # same age as 4: higher id first
        _note(6, "user", age_days=90),  # no helpful key at all counts as 0
    ]
    assert [notes[i]["id"] for i in _keep_order(notes)] == [3, 2, 1, 5, 4, 6]


def test_ties_between_types_go_to_the_earlier_type_and_the_order_is_deterministic():
    notes = _pool({"reference": 2, "feedback": 2, "user": 2, "project": 2})  # ids 1..8
    first = _keep_order(notes)
    assert first == _keep_order(list(notes))
    # Reference (weight 1) puts its first note at virtual time 1/2, the same instant as
    # user's (weight 3) second; the earlier type wins the tie.
    ids = [notes[i]["id"] for i in first]
    assert ids.index(6) < ids.index(1)  # user's second note ahead of reference's first


def test_unknown_type_weighs_one_and_sorts_last():
    notes = [*_pool({"feedback": 2, "reference": 1}), _note(99, "mystery")]
    order = _keep_order(notes)
    assert sorted(order) == [0, 1, 2, 3]
    # It ties reference at 1/2, and an unlisted type loses ties.
    assert notes[order[-1]]["id"] == 99


def test_a_feedback_flood_leaves_room_for_every_other_type():
    """The shape that starved the board for eight weeks: 65 feedback hooks alone overflow
    the cap, and user, project and reference notes went first. Through the real fit."""
    notes = _pool({"feedback": 65, "user": 119, "project": 30, "reference": 70}, hook="w" * 140)
    text, kept, dropped = _fit_notes("alpha", 50000, ["alpha"], notes, [], None)
    shown = Counter(n["type"] for n in kept)
    assert dropped > 0 and len(text) // 4 <= 2000
    assert all(shown[t] > 0 for t in _TYPES), shown
    # What fits (about 40 notes) splits roughly 4:3:2:1.
    assert shown["feedback"] > shown["user"] > shown["project"] > shown["reference"], shown
    assert shown["feedback"] < 65


def test_everything_fits_so_nothing_is_dropped_and_order_is_the_lists():
    notes = [_note(3, "user", helpful=9), _note(2, "feedback"), _note(1, "reference", helpful=1)]
    _, kept, dropped = _fit_notes(None, 10, [], notes, [], None)
    assert dropped == 0 and [n["id"] for n in kept] == [3, 2, 1]  # list order, not rank order


def test_the_survivors_of_an_overflow_keep_the_lists_section_order():
    notes = _pool({"feedback": 30, "user": 30, "project": 30, "reference": 30}, hook="w" * 140)
    _, kept, dropped = _fit_notes("alpha", 50000, ["alpha"], notes, [], None)
    assert dropped > 0
    positions = [notes.index(n) for n in kept]
    assert positions == sorted(positions)


def _event(day: int, fact: str, salience: int = 2, project: str | None = "alpha"):
    return {
        "id": hash(fact) % 1000,
        "date": f"2026-10-{day:02d}",
        "fact": fact,
        "project": project,
        "salience": salience,
    }


def test_one_per_day_picks_the_best_salience_and_ties_go_to_the_newest():
    # _recent_events returns newest first, so within a day the first row is the newest.
    events = [
        _event(7, "newest, plain"),
        _event(7, "older but important", salience=4),
        _event(7, "oldest important", salience=4),
        _event(6, "first of day six"),
        _event(6, "second of day six"),
    ]
    got = _one_per_day(events)
    assert [e["fact"] for e in got] == ["older but important", "first of day six"]


def test_one_per_day_keeps_the_five_newest_days_newest_first():
    events = [_event(d, f"day {d}") for d in (9, 8, 8, 7, 5, 4, 3, 2)]
    got = _one_per_day(events)
    assert _EVENT_DAYS == 5
    assert [e["date"][-2:] for e in got] == ["09", "08", "07", "05", "04"]
    assert _one_per_day([]) == []
    assert len(_one_per_day(events, max_days=2)) == 2


def test_one_per_day_tolerates_a_missing_salience():
    events = [
        _event(3, "a"),
        {**_event(3, "b"), "salience": None},
        {"date": "2026-10-02", "fact": "c"},
    ]
    got = _one_per_day(events)
    assert [e["fact"] for e in got] == ["a", "c"]
