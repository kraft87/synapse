"""Tests for the EdgeDateExtractor: the temporal-marker pre-filter (task #46) and the dates.

The pre-filter short-circuits the LLM date-extraction call for facts whose
text carries no resolvable temporal content. Those facts resolve to the
caller's fallback anyway, so skipping the call is behaviour-preserving
while removing the most expensive per-item stage on the common case. What the call
returns is a valid-from per fact, a date with no time stored at noon UTC.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ingestion.edge_dates import EdgeDateExtractor, _has_temporal_markers, _noon_if_date_only
from ingestion.llm_client import LLMUnavailableError


def _mock_llm(text: str) -> MagicMock:
    llm = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    llm.messages.create.return_value = msg
    return llm


class TestHasTemporalMarkers:
    def test_structural_code_facts_are_non_temporal(self):
        for fact in [
            "config_loader.py defines EligibilityConfig class",
            "ace_processing.py contains parse_case_detections() function",
            "test_ace_processing.py tests ace_processing.py",
            "get_eligibility() implements single detection early return pattern",
            "dedup_map maps normalized detection IDs to original IDs using dict[str, list[str]] pattern",
            "single detection early return pattern is intentionally placed before ignore filtering logic",
        ]:
            assert not _has_temporal_markers(fact), fact

    def test_explicit_dates_and_times_are_temporal(self):
        for fact in [
            "Alex joined Acme in 2020",
            "The migration ran on 2026-05-12",
            "Standup is at 09:30 every day",
            "Released 5/12/2025",
            "Contract started in January",
            "He left on Monday",
        ]:
            assert _has_temporal_markers(fact), fact

    def test_relative_and_duration_expressions_are_temporal(self):
        for fact in [
            "User uses cannabis daily for 8-year period",
            "During cannabis withdrawal week 4 he improved",
            "Alex quit 3 months ago",
            "He has worked there since 2019",
            "The role lasted until last year",
            "User takes Vitamin D 5000 IU daily",
        ]:
            assert _has_temporal_markers(fact), fact

    def test_the_modal_may_is_not_temporal_but_the_month_is(self):
        assert not _has_temporal_markers("The poller may skip a cycle")
        assert not _has_temporal_markers("This may fail when the queue is empty")
        for fact in [
            "Contract started in May",
            "The audit is on May 3",
            "Alex has been on leave since May",
            "May 2026 was the launch",
        ]:
            assert _has_temporal_markers(fact), fact

    def test_validity_flip_words_are_temporal(self):
        for fact in [
            "Alex previously worked at Initech",
            "The endpoint is no longer supported",
            "She used to manage the team",
            "The feature was deprecated",
        ]:
            assert _has_temporal_markers(fact), fact


class TestExtractBatchPreFilter:
    def test_all_non_temporal_skips_llm(self):
        llm = _mock_llm('{"results": []}')
        ext = EdgeDateExtractor(llm)
        facts = [
            "config_loader.py defines EligibilityConfig class",
            "ace_processing.py contains parse_case_detections() function",
        ]
        out = ext.extract_batch(facts)
        assert out == [None, None]
        llm.messages.create.assert_not_called()

    def test_only_temporal_facts_sent_to_llm(self):
        # idx 1 is temporal; idx 0 and 2 are structural.
        llm = _mock_llm('{"results": [{"id": 1, "valid_at": "2020-01-01T09:30:00Z"}]}')
        ext = EdgeDateExtractor(llm)
        facts = [
            "module.py defines Thing",
            "Alex joined Acme in 2020",
            "module.py contains helper()",
        ]
        out = ext.extract_batch(facts)
        assert llm.messages.create.call_count == 1
        # Only the temporal fact was put in the LLM payload, under its original idx.
        sent = llm.messages.create.call_args.kwargs["messages"]
        payload = sent[-1]["content"]
        assert '"id": 1' in payload
        assert "Alex joined Acme" in payload
        assert "module.py defines Thing" not in payload
        # Result mapped back to the right index; non-temporal stay None.
        assert out[0] is None
        assert out[1] == "2020-01-01T09:30:00+00:00"
        assert out[2] is None

    def test_blank_and_non_temporal_both_skipped(self):
        llm = _mock_llm('{"results": []}')
        ext = EdgeDateExtractor(llm)
        out = ext.extract_batch(["", "module.py defines Thing", "  "])
        assert out == [None, None, None]
        llm.messages.create.assert_not_called()

    def test_empty_list_returns_empty(self):
        llm = _mock_llm('{"results": []}')
        ext = EdgeDateExtractor(llm)
        assert ext.extract_batch([]) == []
        llm.messages.create.assert_not_called()


class TestExtractBatchFailures:
    """A response that never validates costs the batch its dates; an outage fails the item."""

    def test_unparseable_response_falls_back_to_none_and_is_logged(self, caplog):
        llm = _mock_llm("not json at all")
        ext = EdgeDateExtractor(llm)
        with caplog.at_level("WARNING", logger="ingestion.edge_dates"):
            out = ext.extract_batch(["Alex joined Acme in 2020"])
        assert out == [None]
        assert "edge dates" in caplog.text

    def test_transport_failure_propagates(self):
        llm = _mock_llm("{}")
        llm.messages.create.side_effect = LLMUnavailableError("claude CLI unreachable")
        ext = EdgeDateExtractor(llm)
        with pytest.raises(LLMUnavailableError):
            ext.extract_batch(["Alex joined Acme in 2020"])


class TestExtractBatchResults:
    """Batched extract_batch collapses N per-fact ~30s claude-CLI subprocess
    calls into ONE Haiku call. Each fact gets its OWN valid_at back, indexed by
    position in the input list.
    """

    def _llm_with_results(self, results: list[dict]) -> object:
        import json as _json

        msg = MagicMock()
        msg.content = [MagicMock(text=_json.dumps({"results": results}))]
        llm = MagicMock()
        llm.messages.create.return_value = msg
        return llm

    def test_one_call_for_many_facts(self):
        llm = self._llm_with_results(
            [
                {"id": 0, "valid_at": "2026-05-10T08:15:00Z"},
                {"id": 1, "valid_at": None},
            ]
        )
        ext = EdgeDateExtractor(llm)
        out = ext.extract_batch(
            ["Alex joined Acme last week", "Old contract ended in April"],
            reference_time="2026-05-17T00:00:00Z",
        )
        assert out == ["2026-05-10T08:15:00+00:00", None]
        # ONE LLM call regardless of input length.
        llm.messages.create.assert_called_once()

    def test_empty_input_short_circuits(self):
        llm = MagicMock()
        ext = EdgeDateExtractor(llm)
        assert ext.extract_batch([]) == []
        llm.messages.create.assert_not_called()

    def test_blank_facts_skipped_index_preserved(self):
        """Blank entries skip the LLM but keep the position so the caller
        can map results back 1:1 with the input list."""
        llm = self._llm_with_results(
            [
                {"id": 0, "valid_at": "2026-05-10T08:15:00Z"},
                {"id": 2, "valid_at": "2026-05-12T08:15:00Z"},
            ]
        )
        ext = EdgeDateExtractor(llm)
        out = ext.extract_batch(["Alex joined Acme", "", "Alex promoted today"])
        # Index alignment preserved — blank slot stays None.
        assert out == ["2026-05-10T08:15:00+00:00", None, "2026-05-12T08:15:00+00:00"]
        # Only the 2 non-blank facts went to the LLM.
        prompt = "\n".join(m["content"] for m in llm.messages.create.call_args.kwargs["messages"])
        assert "Alex joined Acme" in prompt
        assert "Alex promoted today" in prompt

    def test_tolerates_trailing_prose(self):
        msg = MagicMock()
        msg.content = [
            MagicMock(
                text=(
                    '{"results": [{"id": 0, "valid_at": "2026-05-10T08:15:00Z"}]}'
                    "\n\nHere is my reasoning..."
                )
            )
        ]
        llm = MagicMock()
        llm.messages.create.return_value = msg

        ext = EdgeDateExtractor(llm)
        out = ext.extract_batch(["Alex joined Acme last week"])
        assert out == ["2026-05-10T08:15:00+00:00"]

    def test_missing_id_in_response_treated_as_none(self):
        # LLM returns results only for ids 0 and 2, omits id 1.
        llm = self._llm_with_results(
            [
                {"id": 0, "valid_at": "2026-05-10T08:15:00Z"},
                {"id": 2, "valid_at": "2026-05-12T08:15:00Z"},
            ]
        )
        ext = EdgeDateExtractor(llm)
        # Temporal facts so all three pass the pre-filter and reach the LLM;
        # the test exercises missing-id handling, not the pre-filter.
        out = ext.extract_batch(
            ["Alex joined in 2020", "She left last week", "Released on 2026-05-12"]
        )
        assert out == ["2026-05-10T08:15:00+00:00", None, "2026-05-12T08:15:00+00:00"]

    def test_the_prompt_asks_for_a_valid_from_only_with_the_rules_first(self):
        llm = self._llm_with_results([])
        EdgeDateExtractor(llm).extract_batch(["Alex joined Acme in 2020"], reference_time="REF-T")
        prompt = llm.messages.create.call_args.kwargs["messages"][-1]["content"]
        assert "invalid_at" not in prompt
        assert "assume 12:00:00" in prompt
        # Per-call content last, so a shared rule prefix can be cached by the provider.
        assert prompt.index("Rules") < prompt.index("REF-T") < prompt.index("Alex joined Acme")


class TestDateOnlyIsStoredAtNoonUtc:
    """A date with no time is a calendar day, not an instant. At UTC midnight it reads as the
    previous evening for a viewer west of UTC, so it is stored at noon UTC, as the timeline
    gate already stores the dates it resolves."""

    @pytest.mark.parametrize(
        "given",
        [
            "2026-01-31",
            "2026-01-31T00:00:00Z",
            "2026-01-31T00:00:00",
            "2026-01-31T00:00:00+00:00",
            "2026-01-31T05:00:00+05:00",  # UTC midnight, written with an offset
        ],
    )
    def test_date_only_values_become_noon_utc(self, given):
        assert _noon_if_date_only(given) == "2026-01-31T12:00:00+00:00"

    @pytest.mark.parametrize(
        ("given", "stored"),
        [
            ("2026-01-31T08:15:00Z", "2026-01-31T08:15:00+00:00"),
            ("2026-01-31T00:00:00-05:00", "2026-01-31T05:00:00+00:00"),  # a real local midnight
            ("2026-01-31T12:00:00Z", "2026-01-31T12:00:00+00:00"),
        ],
    )
    def test_an_instant_is_kept(self, given, stored):
        assert _noon_if_date_only(given) == stored

    @pytest.mark.parametrize("given", [None, "", "  ", "last week", "-4599999974-05-27T00:00:00Z"])
    def test_junk_is_no_date(self, given):
        assert _noon_if_date_only(given) is None
