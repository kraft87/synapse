"""A recall_full_turns / fetch call serves whole turns up to a size budget.

Claude Code cuts an MCP result off past MAX_MCP_OUTPUT_TOKENS (25K tokens by default), and a
tool-heavy turn is 100K+ characters, so five or twenty whole turns could come back
truncated, or fill the caller's context with tool transcripts. Pure-logic tests: the
turns are plain recall items.
"""

from __future__ import annotations

from typing import Any

from mcp_server.recall_presentation import FULL_TURN_BUDGET, fit_turns_to_budget


def _turn(i: int, chars: int, needle_at: int | None = None) -> dict[str, Any]:
    body = ("filler words about nothing in particular " * (chars // 40 + 1))[:chars]
    if needle_at is not None:
        body = body[:needle_at] + " the zebra crossing decision " + body[needle_at:]
    return {"id": f"e:{i}", "content": body, "project": "synapse", "date": "2026-10-01"}


def test_turns_within_the_budget_are_served_whole():
    turns = [_turn(i, 8_000) for i in range(4)]
    before = [dict(t) for t in turns]
    assert fit_turns_to_budget(turns, "zebra") == 0
    assert turns == before


def test_turns_past_the_budget_come_back_as_excerpts_with_their_length():
    turns = [_turn(1, 30_000), _turn(2, 30_000, needle_at=15_000), _turn(3, 5_000)]
    assert fit_turns_to_budget(turns, "zebra crossing") == 1
    whole, excerpt, small = turns
    assert len(whole["content"]) == 30_000 and "full_chars" not in whole
    # The second would pass the budget: its opening plus the stretch around the match.
    assert excerpt["full_chars"] == 30_000 + len(" the zebra crossing decision ")
    assert len(excerpt["content"]) < 4_000
    assert "zebra crossing decision" in excerpt["content"]
    assert excerpt["content"].startswith("filler words about nothing")
    assert (excerpt["id"], excerpt["project"], excerpt["date"]) == ("e:2", "synapse", "2026-10-01")
    # The budget is spent in rank order, but a later small turn still fits what is left.
    assert small["content"] and "full_chars" not in small
    assert sum(len(t["content"]) for t in turns) <= FULL_TURN_BUDGET


def test_the_top_ranked_turn_is_cut_too_when_it_alone_passes_the_budget():
    turns = [_turn(1, 120_000, needle_at=80_000), _turn(2, 3_000)]
    assert fit_turns_to_budget(turns, "zebra") == 1
    assert turns[0]["full_chars"] > 120_000
    assert "zebra" in turns[0]["content"]
    assert "full_chars" not in turns[1]


def test_a_match_inside_the_opening_is_not_served_twice():
    turns = [_turn(1, 50_000, needle_at=200)]
    fit_turns_to_budget(turns, "zebra crossing")
    content = turns[0]["content"]
    assert content.count("zebra crossing decision") == 1
    assert "…" not in content  # one contiguous stretch from the start, no elision mark


def test_without_a_query_an_excerpt_is_just_the_opening():
    turns = [_turn(1, 60_000)]
    assert fit_turns_to_budget(turns, None) == 1
    assert len(turns[0]["content"]) <= 1_000
    assert turns[0]["full_chars"] == 60_000


def test_keep_first_serves_the_requested_turn_whole_whatever_its_size():
    """fetch(ids) is the only way to read a turn whole: the first one asked for is never
    cut, and what follows shares what is left of the budget."""
    turns = [_turn(1, 90_000), _turn(2, 4_000), _turn(3, 4_000)]
    assert fit_turns_to_budget(turns, None, keep_first=True) == 2
    assert len(turns[0]["content"]) == 90_000 and "full_chars" not in turns[0]
    assert [t["full_chars"] for t in turns[1:]] == [4_000, 4_000]
    assert all(len(t["content"]) <= 1_000 for t in turns[1:])


def test_a_turn_with_no_content_is_left_alone():
    turns = [{"id": "e:9", "content": ""}, {"id": "e:10"}]
    assert fit_turns_to_budget(turns, "q") == 0
    assert turns == [{"id": "e:9", "content": ""}, {"id": "e:10"}]
