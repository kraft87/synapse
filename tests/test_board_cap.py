"""The board's cap (mcp_server/board.py ``_fit_notes``) against the loop it replaced.

The old cap loop dropped one note at a time and re-rendered after each drop: exact by
construction, one render per dropped note. ``_fit_notes`` binary-searches the drop count
instead. These tests pin the two to the same ``(text, kept ids, dropped)`` on randomized
note sets and on padding sweeps that walk the cap boundary across every drop count, and
bound the number of renders. Pure Python, no DB.
"""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from mcp_server import board
from mcp_server.board import _DROP_CLASS, _fit_notes, _fits, _render

_TYPES = ("feedback", "user", "project", "reference")
_T0 = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


def _old_cap_loop(project, n_episodes, project_names, notes, events, notice, n_projects=None):
    """The cap loop as it was before ``_fit_notes``, verbatim apart from the return and
    the banner's project count passed through to ``_render``."""
    kept = list(notes)
    dropped = 0
    floor = _render(project, n_episodes, project_names, [], len(notes), events, notice, n_projects)
    can_reach_cap = _fits(floor)
    while True:
        text = _render(
            project, n_episodes, project_names, kept, dropped, events, notice, n_projects
        )
        if not kept or not can_reach_cap or _fits(text):
            break
        victim = min(kept, key=lambda n: (_DROP_CLASS.get(n["type"], 0), n["updated_at"], n["id"]))
        kept.remove(victim)
        dropped += 1
    return text, kept, dropped


def _assert_same(project, n_episodes, names, notes, events, notice) -> int:
    old_text, old_kept, old_dropped = _old_cap_loop(
        project, n_episodes, names, notes, events, notice
    )
    text, kept, dropped = _fit_notes(project, n_episodes, names, notes, events, notice)
    assert text == old_text
    assert [n["id"] for n in kept] == [n["id"] for n in old_kept]
    assert dropped == old_dropped
    return dropped


def _notes(rng: random.Random, n: int, hook_len: tuple[int, int], newline_rate: float = 0.0):
    """``n`` notes with unique ids in shuffled order, so list order, id order and drop
    order all differ, and some updated_at ties so the id tie-break matters."""
    ids = rng.sample(range(1, 10 * n + 10), n)
    out: list[dict[str, Any]] = []
    for nid in ids:
        hook = "h" * rng.randint(*hook_len)
        if hook and rng.random() < newline_rate:
            cut = rng.randrange(len(hook))
            hook = hook[:cut] + "\n" + hook[cut:]
        out.append(
            {
                "id": nid,
                "type": rng.choice(_TYPES),
                "hook": hook,
                "updated_at": _T0 - timedelta(days=rng.randint(0, 40)),
            }
        )
    return out


def _events(rng: random.Random, n: int) -> list[dict[str, Any]]:
    return [
        {
            "id": rng.choice([None, i]),
            "date": "2026-10-0" + str(1 + i % 9),
            "project": rng.choice([None, "alpha", "beta"]),
            "fact": "f" * rng.randint(1, 320),  # past the 200-char render clamp sometimes
        }
        for i in range(n)
    ]


def _db_ordered(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The order select_board_notes serves: type rank, then newest-updated first."""
    rank = {"feedback": 0, "user": 1, "project": 2, "reference": 3}
    return sorted(notes, key=lambda n: (rank[n["type"]], -n["updated_at"].timestamp()))


@pytest.mark.parametrize("seed", range(40))
def test_matches_old_loop_on_random_boards(seed):
    rng = random.Random(seed)
    for _ in range(10):
        shape = rng.choice(["short", "mixed", "long", "lines"])
        if shape == "short":
            notes = _notes(rng, rng.randint(0, 120), (0, 12), newline_rate=0.05)
        elif shape == "mixed":
            notes = _notes(rng, rng.randint(0, 120), (0, 300), newline_rate=0.05)
        elif shape == "long":
            notes = _notes(rng, rng.randint(0, 30), (300, 3000))
        else:
            notes = _notes(rng, rng.randint(60, 140), (5, 60))
        if rng.random() < 0.5:
            notes = _db_ordered(notes)
        names = [f"proj{i}" for i in range(rng.randint(0, 12))]
        notice = rng.choice([None, "n" * rng.randint(1, 4000)])
        _assert_same(
            rng.choice([None, "alpha"]),
            rng.randint(0, 60000),
            names,
            notes,
            _events(rng, rng.randint(0, 10)),
            notice,
        )


def _sweep(notes, pads, *, events=(), project="alpha") -> list[int]:
    """Old vs new for each notice length in ``pads``: the notice line is padding that
    slides the token cap's boundary along the drop order one char at a time."""
    seen = []
    for pad in pads:
        seen.append(_assert_same(project, 1234, ["alpha", "beta"], notes, list(events), "w" * pad))
    return seen


def test_token_cap_boundary_walks_every_drop_count():
    """Fifteen notes with empty hooks (the shortest possible line): sweeping the padding
    char by char puts the boundary at every k from 0 to 15, through the overflow
    count's 9 -> 10 rollover and every section emptying out. Except k=1: with lines
    this short the first drop lengthens the board, so no padding makes k=1 the answer
    (the next test covers that case)."""
    rng = random.Random(7)
    notes = _db_ordered(_notes(rng, 15, (0, 0)))
    seen = _sweep(notes, range(7300, 7950))
    assert set(range(16)) - {1} <= set(seen)


def test_k0_fitting_while_k1_does_not_is_kept_at_k0():
    """The case a naive binary search over [0, n] would get wrong. Dropping one short note
    adds the 27-char overflow line and removes a 20-char note line, so a board can fit
    with no drops and not fit with one. The sweep must hit that case and agree with the
    old loop on it."""
    rng = random.Random(11)
    notes = _db_ordered(_notes(rng, 15, (0, 0)))
    hit = False
    for pad in range(7300, 7950):
        notice = "w" * pad
        if _assert_same("alpha", 1234, ["alpha", "beta"], notes, [], notice) != 0:
            continue
        old_kept = _old_cap_loop("alpha", 1234, ["alpha", "beta"], notes, [], notice)[1]
        if old_kept:  # k=0 fit; now render the board one drop later
            victim = min(notes, key=lambda n: (_DROP_CLASS[n["type"]], n["updated_at"], n["id"]))
            one_drop = [n for n in notes if n is not victim]
            if not _fits(_render("alpha", 1234, ["alpha", "beta"], one_drop, 1, [], notice)):
                hit = True
    assert hit


def test_token_cap_boundary_across_the_99_to_100_rollover():
    """110 short notes: past the line cap, so the line and token caps both bind, and
    the token boundary is swept through k = 99 -> 100, where the overflow count gains
    a digit."""
    rng = random.Random(3)
    notes = _db_ordered(_notes(rng, 110, (0, 3)))

    def k_at(pad: int) -> int:
        return _fit_notes("alpha", 1234, ["alpha", "beta"], notes, [], "w" * pad)[2]

    # Locate the padding where the answer reaches k=99 (cheap: the new search), then
    # walk that window char by char against the old loop.
    coarse = next(p for p in range(5000, 8000, 20) if k_at(p) >= 99)
    p99 = next(p for p in range(coarse - 20, coarse + 1) if k_at(p) >= 99)
    seen = _sweep(notes, range(p99 - 25, p99 + 50))
    assert {98, 99, 100, 101} <= set(seen)


def test_line_cap_boundary_with_events_and_notice():
    """The line cap: 100 notes, the boundary moved by one line per timeline event and
    by the notice line."""
    rng = random.Random(5)
    notes = _db_ordered(_notes(rng, 100, (5, 40)))
    seen = set()
    for n_events in range(11):
        events = _events(random.Random(n_events), n_events)
        for notice in (None, "short notice"):
            seen.add(_assert_same(None, 99, ["alpha"], notes, events, notice))
    assert len(seen) > 10


def test_floor_over_cap_keeps_every_note():
    names = [f"project-{i}-" + "z" * 800 for i in range(12)]
    notes = _notes(random.Random(1), 20, (10, 30))
    assert _assert_same(None, 1, names, notes, [], None) == 0


def test_no_notes_and_one_giant_note():
    assert _assert_same(None, 0, [], [], [], None) == 0
    giant = [{"id": 1, "type": "user", "hook": "Giant " + "x" * 9000, "updated_at": _T0}]
    assert _assert_same(None, 0, [], giant, [], None) == 1


def test_renders_logarithmically_for_a_large_board(monkeypatch):
    """240 notes, about 200 of them over the cap: the old loop rendered once per drop;
    the search renders k=0, the floor, and log2(n) midpoints."""
    rng = random.Random(9)
    notes = _db_ordered(_notes(rng, 240, (30, 110)))
    calls = 0
    real_render = board._render

    def counting_render(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_render(*args, **kwargs)

    monkeypatch.setattr(board, "_render", counting_render)
    _, kept, dropped = _fit_notes("alpha", 50000, ["alpha"], notes, [], None)
    assert dropped > 150 and len(kept) + dropped == 240
    assert calls <= 2 + math.ceil(math.log2(240))
