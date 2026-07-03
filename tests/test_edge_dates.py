"""Tests for the EdgeDateExtractor temporal-marker pre-filter (task #46).

The pre-filter short-circuits the LLM date-extraction call for facts whose
text carries no resolvable temporal content. Those facts resolve to the
caller's now() fallback anyway, so skipping the call is behaviour-preserving
while removing the single most expensive per-item stage on the common case.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from ingestion.edge_dates import EdgeDateExtractor, _has_temporal_markers


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
            "Kyle joined Acme in 2020",
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
            "Kyle quit 3 months ago",
            "He has worked there since 2019",
            "The role lasted until last year",
            "User takes Vitamin D 5000 IU daily",
        ]:
            assert _has_temporal_markers(fact), fact

    def test_validity_flip_words_are_temporal(self):
        for fact in [
            "Kyle previously worked at RBC",
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
        assert out == [(None, None), (None, None)]
        llm.messages.create.assert_not_called()

    def test_only_temporal_facts_sent_to_llm(self):
        # idx 1 is temporal; idx 0 and 2 are structural.
        llm = _mock_llm(
            '{"results": [{"id": 1, "valid_at": "2020-01-01T00:00:00Z", "invalid_at": null}]}'
        )
        ext = EdgeDateExtractor(llm)
        facts = [
            "module.py defines Thing",
            "Kyle joined Acme in 2020",
            "module.py contains helper()",
        ]
        out = ext.extract_batch(facts)
        assert llm.messages.create.call_count == 1
        # Only the temporal fact was put in the LLM payload, under its original idx.
        sent = llm.messages.create.call_args.kwargs["messages"]
        payload = sent[-1]["content"]
        assert '"id": 1' in payload
        assert "Kyle joined Acme" in payload
        assert "module.py defines Thing" not in payload
        # Result mapped back to the right index; non-temporal stay (None, None).
        assert out[0] == (None, None)
        assert out[1] == ("2020-01-01T00:00:00Z", None)
        assert out[2] == (None, None)

    def test_blank_and_non_temporal_both_skipped(self):
        llm = _mock_llm('{"results": []}')
        ext = EdgeDateExtractor(llm)
        out = ext.extract_batch(["", "module.py defines Thing", "  "])
        assert out == [(None, None), (None, None), (None, None)]
        llm.messages.create.assert_not_called()

    def test_empty_list_returns_empty(self):
        llm = _mock_llm('{"results": []}')
        ext = EdgeDateExtractor(llm)
        assert ext.extract_batch([]) == []
        llm.messages.create.assert_not_called()


class TestExtractSinglePreFilter:
    def test_non_temporal_skips_llm(self):
        llm = _mock_llm('{"valid_at": null, "invalid_at": null}')
        ext = EdgeDateExtractor(llm)
        assert ext.extract("config_loader.py defines EligibilityConfig class") == (None, None)
        llm.messages.create.assert_not_called()

    def test_temporal_calls_llm(self):
        llm = _mock_llm('{"valid_at": "2021-06-01T00:00:00Z", "invalid_at": null}')
        ext = EdgeDateExtractor(llm)
        assert ext.extract("Kyle joined in 2021") == ("2021-06-01T00:00:00Z", None)
        llm.messages.create.assert_called_once()
