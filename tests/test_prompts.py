"""Tests for the verbatim-ported Graphiti prompt modules.

Phase 4. Each test asserts two things at minimum:
1. The prompt template renders without raising and produces a list of
   ``{"role": ..., "content": ...}`` dicts the LLM client can consume.
2. The matching Pydantic response model parses well-formed LLM output and
   raises ``ValidationError`` on malformed shapes.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ingestion.prompts.dedupe_edges import (
    EdgeDedupResponse,
)
from ingestion.prompts.dedupe_edges import (
    build_prompt as build_dedupe_edges_prompt,
)
from ingestion.prompts.dedupe_nodes import (
    NodeDedupResponse,
)
from ingestion.prompts.dedupe_nodes import (
    build_prompt as build_dedupe_nodes_prompt,
)
from ingestion.prompts.extract_edge_dates import (
    EdgeDatesResponse,
)
from ingestion.prompts.extract_edge_dates import (
    build_prompt as build_extract_edge_dates_prompt,
)
from ingestion.prompts.invalidate_edges import (
    EdgeContradictionResponse,
)
from ingestion.prompts.invalidate_edges import (
    build_prompt as build_invalidate_edges_prompt,
)

# ---------------------------------------------------------------------------
# dedupe_nodes
# ---------------------------------------------------------------------------


class TestDedupeNodesPrompt:
    def test_dedupe_nodes_prompt_renders(self):
        """Given a mock candidate + new entity, the prompt formats cleanly
        and the returned response model parses a well-formed LLM reply."""
        context = {
            "previous_episodes": [{"role": "user", "content": "Hi from Sam"}],
            "episode_content": "Sam joined the team today.",
            "extracted_node": {"name": "Sam", "summary": "New teammate"},
            "entity_type_description": "Person",
            "existing_nodes": [
                {
                    "candidate_id": 0,
                    "name": "Sam",
                    "entity_types": ["Person"],
                    "summary": "Sam enjoys hiking and photography",
                }
            ],
        }
        messages = build_dedupe_nodes_prompt(context)

        assert isinstance(messages, list)
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        # Verbatim Graphiti language is preserved.
        assert "entity deduplication assistant" in messages[0]["content"]
        # Required context substituted into the user prompt.
        assert "Sam joined the team today." in messages[1]["content"]
        assert "candidate_id" in messages[1]["content"]
        assert "duplicate_candidate_id" in messages[1]["content"]

    def test_parses_well_formed_response(self):
        parsed = NodeDedupResponse.model_validate(
            {"id": 0, "name": "Sam", "duplicate_candidate_id": 0}
        )
        assert parsed.duplicate_candidate_id == 0
        assert parsed.name == "Sam"

    def test_parses_no_match_response(self):
        parsed = NodeDedupResponse.model_validate(
            {"id": 0, "name": "Java", "duplicate_candidate_id": -1}
        )
        assert parsed.duplicate_candidate_id == -1

    def test_rejects_missing_field(self):
        with pytest.raises(ValidationError):
            NodeDedupResponse.model_validate({"id": 0, "name": "Sam"})

    def test_rejects_non_int_candidate_id(self):
        with pytest.raises(ValidationError):
            NodeDedupResponse.model_validate(
                {"id": 0, "name": "Sam", "duplicate_candidate_id": "zero"}
            )


# ---------------------------------------------------------------------------
# invalidate_edges (contradiction-only carve-out)
# ---------------------------------------------------------------------------


class TestInvalidateEdgesPrompt:
    def test_invalidate_edges_prompt_renders(self):
        context = {
            "new_fact": "Synapse uses FalkorDB on port 7778",
            "existing_facts": (
                "[0] Synapse uses FalkorDB on port 7777\n[1] Synapse uses Redis for the queue"
            ),
        }
        messages = build_invalidate_edges_prompt(context)

        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        # Verbatim language preserved.
        assert "fact deduplication assistant" in messages[0]["content"]
        # The user prompt must surface the new fact + the [idx] list verbatim.
        assert "port 7778" in messages[1]["content"]
        assert "[0] Synapse uses FalkorDB on port 7777" in messages[1]["content"]
        # And it must reference the response field name.
        assert "contradicted_facts" in messages[1]["content"]

    def test_parses_well_formed_response(self):
        parsed = EdgeContradictionResponse.model_validate({"contradicted_facts": [0, 3]})
        assert parsed.contradicted_facts == [0, 3]

    def test_parses_empty_response(self):
        parsed = EdgeContradictionResponse.model_validate({"contradicted_facts": []})
        assert parsed.contradicted_facts == []

    def test_rejects_missing_field(self):
        with pytest.raises(ValidationError):
            EdgeContradictionResponse.model_validate({})

    def test_rejects_non_list(self):
        with pytest.raises(ValidationError):
            EdgeContradictionResponse.model_validate({"contradicted_facts": "0,1"})


# ---------------------------------------------------------------------------
# dedupe_edges (full duplicate + contradiction sweep — Graphiti's combined call)
# ---------------------------------------------------------------------------


class TestDedupeEdgesPrompt:
    def test_dedupe_edges_prompt_renders(self):
        context = {
            "existing_edges": "[0] Alice joined Acme Corp in 2020",
            "edge_invalidation_candidates": "[1] Alice works at Acme Corp as a SE",
            "new_edge": "Alice works at Acme Corp as a senior engineer",
        }
        messages = build_dedupe_edges_prompt(context)
        assert messages[0]["role"] == "system"
        assert "duplicate_facts" in messages[1]["content"]
        assert "contradicted_facts" in messages[1]["content"]
        # Verbatim Graphiti example is present (catches accidental edits).
        assert "Bob ran 5 miles on Tuesday" in messages[1]["content"]

    def test_parses_well_formed_response(self):
        parsed = EdgeDedupResponse.model_validate(
            {"duplicate_facts": [0], "contradicted_facts": [1]}
        )
        assert parsed.duplicate_facts == [0]
        assert parsed.contradicted_facts == [1]


# ---------------------------------------------------------------------------
# extract_edge_dates
# ---------------------------------------------------------------------------


class TestExtractEdgeDatesPrompt:
    def test_extract_edge_dates_prompt_renders(self):
        context = {
            "fact": "Alex joined Acme Corp last week",
            "reference_time": "2026-05-17T00:00:00Z",
        }
        messages = build_extract_edge_dates_prompt(context)
        assert messages[0]["role"] == "system"
        assert "Alex joined Acme Corp last week" in messages[1]["content"]
        assert "2026-05-17T00:00:00Z" in messages[1]["content"]
        # Verbatim Graphiti rule wording preserved.
        assert "Resolve relative expressions" in messages[1]["content"]
        assert "ISO 8601 with Z suffix" in messages[1]["content"]

    def test_extract_edge_dates_returns_iso8601(self):
        # Mock LLM responds with a clean ISO 8601 valid_at and null invalid_at.
        parsed = EdgeDatesResponse.model_validate(
            {"valid_at": "2026-05-10T00:00:00Z", "invalid_at": None}
        )
        assert parsed.valid_at == "2026-05-10T00:00:00Z"
        assert parsed.invalid_at is None

    def test_extract_edge_dates_both_null(self):
        parsed = EdgeDatesResponse.model_validate({"valid_at": None, "invalid_at": None})
        assert parsed.valid_at is None and parsed.invalid_at is None

    def test_rejects_non_string_valid_at(self):
        with pytest.raises(ValidationError):
            EdgeDatesResponse.model_validate({"valid_at": 12345, "invalid_at": None})

    def test_response_model_rejects_bad_shape(self):
        # Graphiti's EdgeTimestamps defaults both fields to None, so a
        # totally-empty dict is valid. The "bad shape" we care about is a
        # non-dict payload (e.g. the LLM returned a list, a bare string,
        # or an integer instead of an object).
        with pytest.raises(ValidationError):
            EdgeDatesResponse.model_validate(["2026-05-10T00:00:00Z"])
        with pytest.raises(ValidationError):
            EdgeDatesResponse.model_validate(42)


# ---------------------------------------------------------------------------
# EdgeDateExtractor integration — confirms the prompt round-trips through
# the structured-response code path in `ingestion.edge_dates`.
# ---------------------------------------------------------------------------


class TestEdgeDateExtractor:
    def test_extracts_dates_from_mock_llm(self):
        from unittest.mock import MagicMock

        from ingestion.edge_dates import EdgeDateExtractor

        llm = MagicMock()
        response = MagicMock()
        response.content = [
            MagicMock(text='{"valid_at": "2026-05-10T00:00:00Z", "invalid_at": null}')
        ]
        llm.messages.create.return_value = response

        ext = EdgeDateExtractor(llm)
        valid, invalid = ext.extract(
            "Alex joined Acme Corp last week",
            reference_time="2026-05-17T00:00:00Z",
        )
        assert valid == "2026-05-10T00:00:00Z"
        assert invalid is None
        # The prompt the LLM saw must reference the verbatim rule wording.
        # System-role prompt parts now arrive via the ``system`` kwarg
        # (structured_call flattens roles); the rule text lives there.
        kwargs = llm.messages.create.call_args.kwargs
        prompt = (kwargs.get("system") or "") + "\n".join(m["content"] for m in kwargs["messages"])
        assert "NEVER hallucinate dates" in prompt
        assert "Alex joined Acme Corp last week" in prompt

    def test_extractor_returns_none_on_llm_failure(self):
        from unittest.mock import MagicMock

        from ingestion.edge_dates import EdgeDateExtractor

        llm = MagicMock()
        llm.messages.create.side_effect = RuntimeError("upstream 500")
        ext = EdgeDateExtractor(llm)
        # Graceful degradation: extractor must never raise so create_edge
        # can fall through to its now() default.
        assert ext.extract("Alex works at Acme") == (None, None)

    def test_extractor_returns_none_on_malformed_json(self):
        from unittest.mock import MagicMock

        from ingestion.edge_dates import EdgeDateExtractor

        llm = MagicMock()
        response = MagicMock()
        response.content = [MagicMock(text="not json at all")]
        llm.messages.create.return_value = response

        ext = EdgeDateExtractor(llm)
        assert ext.extract("Alex works at Acme") == (None, None)

    def test_extractor_skips_empty_fact(self):
        from unittest.mock import MagicMock

        from ingestion.edge_dates import EdgeDateExtractor

        llm = MagicMock()
        ext = EdgeDateExtractor(llm)
        assert ext.extract("") == (None, None)
        assert ext.extract("   ") == (None, None)
        # LLM should never have been called.
        llm.messages.create.assert_not_called()


class TestEdgeDateExtractorBatch:
    """Batched extract_batch collapses N per-fact ~30s claude-CLI subprocess
    calls into ONE Haiku call. Each fact gets its OWN (valid_at, invalid_at)
    back, indexed by position in the input list.
    """

    def _llm_with_results(self, results: list[dict]) -> object:
        import json as _json
        from unittest.mock import MagicMock

        msg = MagicMock()
        msg.content = [MagicMock(text=_json.dumps({"results": results}))]
        llm = MagicMock()
        llm.messages.create.return_value = msg
        return llm

    def test_one_call_for_many_facts(self):
        from ingestion.edge_dates import EdgeDateExtractor

        llm = self._llm_with_results(
            [
                {"id": 0, "valid_at": "2026-05-10T00:00:00Z", "invalid_at": None},
                {"id": 1, "valid_at": None, "invalid_at": "2026-04-01T00:00:00Z"},
            ]
        )
        ext = EdgeDateExtractor(llm)
        out = ext.extract_batch(
            ["Alex joined Acme last week", "Old contract ended in April"],
            reference_time="2026-05-17T00:00:00Z",
        )
        assert len(out) == 2
        assert out[0] == ("2026-05-10T00:00:00Z", None)
        assert out[1] == (None, "2026-04-01T00:00:00Z")
        # ONE LLM call regardless of input length.
        llm.messages.create.assert_called_once()

    def test_empty_input_short_circuits(self):
        from unittest.mock import MagicMock

        from ingestion.edge_dates import EdgeDateExtractor

        llm = MagicMock()
        ext = EdgeDateExtractor(llm)
        assert ext.extract_batch([]) == []
        llm.messages.create.assert_not_called()

    def test_blank_facts_skipped_index_preserved(self):
        """Blank entries skip the LLM but keep the position so the caller
        can map results back 1:1 with the input list."""
        from ingestion.edge_dates import EdgeDateExtractor

        llm = self._llm_with_results(
            [
                {"id": 0, "valid_at": "2026-05-10T00:00:00Z", "invalid_at": None},
                {"id": 2, "valid_at": "2026-05-12T00:00:00Z", "invalid_at": None},
            ]
        )
        ext = EdgeDateExtractor(llm)
        out = ext.extract_batch(["Alex joined Acme", "", "Alex promoted today"])
        # Index alignment preserved — blank slot stays (None, None).
        assert out[0] == ("2026-05-10T00:00:00Z", None)
        assert out[1] == (None, None)
        assert out[2] == ("2026-05-12T00:00:00Z", None)
        # Only the 2 non-blank facts went to the LLM.
        prompt = "\n".join(m["content"] for m in llm.messages.create.call_args.kwargs["messages"])
        assert "Alex joined Acme" in prompt
        assert "Alex promoted today" in prompt

    def test_tolerates_trailing_prose(self):
        from unittest.mock import MagicMock

        from ingestion.edge_dates import EdgeDateExtractor

        msg = MagicMock()
        msg.content = [
            MagicMock(
                text=(
                    '{"results": [{"id": 0, "valid_at": "2026-05-10T00:00:00Z",'
                    ' "invalid_at": null}]}\n\nHere is my reasoning...'
                )
            )
        ]
        llm = MagicMock()
        llm.messages.create.return_value = msg

        ext = EdgeDateExtractor(llm)
        out = ext.extract_batch(["Alex joined Acme last week"])
        assert out[0] == ("2026-05-10T00:00:00Z", None)

    def test_llm_exception_returns_all_none(self):
        from unittest.mock import MagicMock

        from ingestion.edge_dates import EdgeDateExtractor

        llm = MagicMock()
        llm.messages.create.side_effect = RuntimeError("upstream 500")
        ext = EdgeDateExtractor(llm)
        # Temporal facts so they pass the pre-filter and actually reach the
        # (failing) LLM — otherwise the pre-filter would short-circuit and we
        # wouldn't be testing the exception path at all.
        out = ext.extract_batch(
            ["Alex joined in 2020", "She left last week", "Released on 2026-05-10"]
        )
        # Conservative posture: an LLM failure must never block writes.
        # Each fact falls back to (None, None) so create_edge uses now().
        assert out == [(None, None), (None, None), (None, None)]
        llm.messages.create.assert_called_once()

    def test_missing_id_in_response_treated_as_none(self):
        from ingestion.edge_dates import EdgeDateExtractor

        # LLM returns results only for ids 0 and 2, omits id 1.
        llm = self._llm_with_results(
            [
                {"id": 0, "valid_at": "2026-05-10T00:00:00Z", "invalid_at": None},
                {"id": 2, "valid_at": "2026-05-12T00:00:00Z", "invalid_at": None},
            ]
        )
        ext = EdgeDateExtractor(llm)
        # Temporal facts so all three pass the pre-filter and reach the LLM;
        # the test exercises missing-id handling, not the pre-filter.
        out = ext.extract_batch(
            ["Alex joined in 2020", "She left last week", "Released on 2026-05-12"]
        )
        assert out[0] == ("2026-05-10T00:00:00Z", None)
        assert out[1] == (None, None)
        assert out[2] == ("2026-05-12T00:00:00Z", None)
