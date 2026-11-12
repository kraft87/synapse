"""The shared transcript line rules (ingestion/render.py), pinned.

Unit tests for each rule at its exact boundary.
"""

from __future__ import annotations

from ingestion import render
from ingestion.render import (
    assistant_line,
    context_line,
    group_turns,
    join_lines,
    repeated_span,
    result_line,
    strip_nulls,
    title_line,
    tool_line,
    user_line,
)


class TestCaps:
    def test_values(self):
        assert render.CONTEXT_CAP == 300
        assert render.USER_CAP == 3000
        assert render.ASSISTANT_CAP == 3000
        assert render.TOOL_CAP == 300
        assert render.RESULT_CAP == 500
        assert render.RESULT_MIN_CHARS == 20
        assert render.TITLE_CAP == 200
        assert render.LINE_SEP == "\n\n"


class TestStripNulls:
    def test_removes_every_nul(self):
        assert strip_nulls("a\x00b\x00\x00c") == "abc"

    def test_text_without_nul_is_unchanged(self):
        assert strip_nulls("plain text") == "plain text"
        assert strip_nulls("") == ""


class TestContextLine:
    def test_shape(self):
        assert context_line("previous reply") == "[context] previous reply"

    def test_strips_before_capping(self):
        text = "   " + "a" * 299 + "b" + "c" + "  \n"
        assert context_line(text) == "[context] " + "a" * 299 + "b"

    def test_cap_boundary(self):
        assert context_line("x" * 300) == "[context] " + "x" * 300
        assert context_line("x" * 301) == "[context] " + "x" * 300


class TestUserLine:
    def test_shape_keeps_whitespace(self):
        assert user_line(" hi ") == "[user]  hi "

    def test_cap_boundary(self):
        assert user_line("u" * 3000) == "[user] " + "u" * 3000
        assert user_line("u" * 3001) == "[user] " + "u" * 3000


class TestAssistantLine:
    def test_shape(self):
        assert assistant_line("Done.") == "[assistant] Done."

    def test_cap_boundary(self):
        assert assistant_line("a" * 3000) == "[assistant] " + "a" * 3000
        assert assistant_line("a" * 3001) == "[assistant] " + "a" * 3000


class TestToolLine:
    def test_shape(self):
        assert tool_line("Bash", "ls -la") == "[tool:Bash] ls -la"

    def test_cap_boundary(self):
        assert tool_line("t", "d" * 300) == "[tool:t] " + "d" * 300
        assert tool_line("t", "d" * 301) == "[tool:t] " + "d" * 300

    def test_name_is_formatted_as_given(self):
        # A transcript's tool name is not validated: null renders as None.
        assert tool_line(None, "x") == "[tool:None] x"

    def test_empty_detail_still_renders(self):
        # Whether an empty detail gets a line is each parser's own rule.
        assert tool_line("t", "") == "[tool:t] "


class TestResultLine:
    def test_twenty_chars_is_no_line(self):
        assert result_line("r" * 20) is None
        assert result_line("") is None

    def test_twenty_one_chars_is_a_line(self):
        assert result_line("r" * 21) == "[result] " + "r" * 21

    def test_cap_boundary(self):
        assert result_line("r" * 500) == "[result] " + "r" * 500
        assert result_line("r" * 501) == "[result] " + "r" * 500


class TestTitleLine:
    def test_shape_and_cap(self):
        assert title_line("My chat") == "[title] My chat"
        assert title_line("t" * 201) == "[title] " + "t" * 200


class TestJoinLines:
    def test_blank_line_between_lines(self):
        assert join_lines(["[user] a", "[assistant] b"]) == "[user] a\n\n[assistant] b"

    def test_strips_nuls(self):
        assert join_lines(["[user] a\x00", "\x00[assistant] b"]) == "[user] a\n\n[assistant] b"

    def test_empty(self):
        assert join_lines([]) == ""


class TestGroupTurns:
    @staticmethod
    def _starts(item: str) -> bool:
        return item.startswith("U")

    def test_each_starter_opens_a_group(self):
        items = ["U1", "a", "b", "U2", "c", "U3"]
        assert group_turns(items, self._starts) == [["U1", "a", "b"], ["U2", "c"], ["U3"]]

    def test_leading_items_form_their_own_group(self):
        assert group_turns(["a", "b", "U1", "c"], self._starts) == [["a", "b"], ["U1", "c"]]

    def test_first_item_starter_opens_no_empty_group(self):
        assert group_turns(["U1", "U2"], self._starts) == [["U1"], ["U2"]]

    def test_no_starters_is_one_group(self):
        assert group_turns(["a", "b"], self._starts) == [["a", "b"]]

    def test_empty(self):
        assert group_turns([], self._starts) == []

    def test_predicate_sees_every_item_once_in_order(self):
        seen: list[str] = []

        def starts(item: str) -> bool:
            seen.append(item)
            return item.startswith("U")

        group_turns(["a", "U1", "b"], starts)
        assert seen == ["a", "U1", "b"]


class TestRepeatedSpan:
    def test_first_sighting_is_recorded(self):
        seen: set[str] = set()
        assert repeated_span(seen, "s1") is False
        assert seen == {"s1"}
        assert repeated_span(seen, "s1") is True
        assert repeated_span(seen, "s2") is False

    def test_no_span_id_is_never_a_repeat(self):
        seen: set[str] = set()
        assert repeated_span(seen, None) is False
        assert repeated_span(seen, None) is False
        assert seen == set()
