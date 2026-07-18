"""Tests for ingestion/llm_schemas.py — stage output models + schema helpers.

Covers:

* strict wire schemas — OpenAI-strict shape: every object closed
  (``additionalProperties: false``), every property required, ``$defs``
  inlined, Optional fields nullable-but-required.
* The tolerant index validator (deepseek-v4-flash wraps indices in
  ``{"index": N}`` objects / digit strings; junk is dropped, never fatal).
* Batch models dropping invalid items instead of failing the whole batch.
* ``TimelineGateEvents`` — legacy single-event shape, cap, per-field
  degradation mirroring the old ``_parse_gate`` behavior.
* ``ExtractionOutput`` — bad rows skipped, defaults applied.
* ``first_json_object`` prose tolerance.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from ingestion.llm_schemas import (
    BatchEdgeDatesResult,
    BatchNodeDedupResult,
    BatchResolutionResult,
    ContradictionVerdict,
    EdgeDatesResult,
    ExtractionOutput,
    NodeDedupVerdict,
    ResolutionResult,
    TimelineGateEvents,
    first_json_object,
)


def _walk_objects(schema: dict) -> list[dict]:
    """Every dict with type=object anywhere in the schema."""
    found = []
    stack = [schema]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if node.get("type") == "object":
                found.append(node)
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return found


def _strict_schema(model_cls):
    """The strict transform the OpenAI path applies natively (NativeOutput
    strict=True) — reproduced here to pin the wire-schema invariants."""
    from pydantic_ai import InlineDefsJsonSchemaTransformer
    from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer

    inlined = InlineDefsJsonSchemaTransformer(model_cls.model_json_schema()).walk()
    return OpenAIJsonSchemaTransformer(inlined, strict=True).walk()


class TestStrictSchema:
    def test_every_object_is_closed_and_fully_required(self):
        for model in (
            ResolutionResult,
            BatchResolutionResult,
            ContradictionVerdict,
            EdgeDatesResult,
            BatchEdgeDatesResult,
            NodeDedupVerdict,
            BatchNodeDedupResult,
            TimelineGateEvents,
            ExtractionOutput,
        ):
            schema = _strict_schema(model)
            objects = _walk_objects(schema)
            assert objects, f"{model.__name__}: no object schemas found"
            for obj in objects:
                assert obj.get("additionalProperties") is False, model.__name__
                props = list(obj.get("properties", {}))
                # OpenAI strict mode: every property must be required.
                assert sorted(obj.get("required", [])) == sorted(props), model.__name__

    def test_defs_are_inlined(self):
        schema = _strict_schema(BatchResolutionResult)
        assert "$defs" not in json.dumps(schema)
        assert "$ref" not in json.dumps(schema)

    def test_optional_fields_are_nullable_but_required(self):
        schema = _strict_schema(EdgeDatesResult)
        assert sorted(schema["required"]) == ["invalid_at", "valid_at"]
        valid_at = schema["properties"]["valid_at"]
        rendered = json.dumps(valid_at)
        assert "null" in rendered and "string" in rendered

    def test_index_lists_are_plain_integer_arrays(self):
        # The wire schema asks for the CLEAN shape; tolerance is python-side.
        schema = _strict_schema(ResolutionResult)
        assert schema["properties"]["duplicate_facts"]["items"] == {"type": "integer"}


class TestIndexCoercion:
    def test_wrapped_and_stringified_indices(self):
        v = ResolutionResult.model_validate(
            {
                "duplicate_facts": [1, "2", {"index": 3}, {"id": 4}, {"fact_index": 5}],
                "contradicted_facts": [],
            }
        )
        assert v.duplicate_facts == [1, 2, 3, 4, 5]

    def test_junk_dropped_not_fatal(self):
        v = ContradictionVerdict.model_validate(
            {"contradicted_facts": [0, None, True, {"bogus": 1}, "x", [], "  7 "]}
        )
        assert v.contradicted_facts == [0, 7]

    def test_non_list_collapses_to_empty(self):
        v = ContradictionVerdict.model_validate({"contradicted_facts": "3"})
        assert v.contradicted_facts == []

    def test_missing_key_defaults_empty(self):
        assert ResolutionResult.model_validate({}).duplicate_facts == []


class TestBatchTolerance:
    def test_invalid_items_dropped_others_kept(self):
        v = BatchResolutionResult.model_validate(
            {
                "results": [
                    {"id": 0, "duplicate_facts": [1], "contradicted_facts": []},
                    {"id": {"nope": 1}},  # unparseable id — dropped
                    "garbage",
                    {"id": 2, "duplicate_facts": [], "contradicted_facts": ["0"]},
                ]
            }
        )
        assert [(r.id, r.duplicate_facts, r.contradicted_facts) for r in v.results] == [
            (0, [1], []),
            (2, [], [0]),
        ]

    def test_non_list_results_collapse_to_empty(self):
        assert BatchNodeDedupResult.model_validate({"results": "nope"}).results == []


class TestEdgeDates:
    def test_empty_strings_normalize_to_none(self):
        v = EdgeDatesResult.model_validate({"valid_at": "", "invalid_at": None})
        assert v.valid_at is None and v.invalid_at is None

    def test_junk_typed_dates_degrade_to_none(self):
        v = EdgeDatesResult.model_validate({"valid_at": 20260101, "invalid_at": {"x": 1}})
        assert v.valid_at is None and v.invalid_at is None

    def test_batch_items_keep_ids(self):
        v = BatchEdgeDatesResult.model_validate(
            {"results": [{"id": 3, "valid_at": "2026-01-01T00:00:00Z", "invalid_at": ""}]}
        )
        assert v.results[0].id == 3
        assert v.results[0].valid_at == "2026-01-01T00:00:00Z"
        assert v.results[0].invalid_at is None


class TestNodeDedup:
    def test_partial_response_parses_with_defaults(self):
        # The legacy parser only ever read duplicate_candidate_id; a partial
        # response must not fail the confirm.
        v = NodeDedupVerdict.model_validate({"duplicate_candidate_id": 0})
        assert v.duplicate_candidate_id == 0
        assert v.id == 0 and v.name == ""

    def test_missing_candidate_id_defaults_to_no_match(self):
        assert NodeDedupVerdict.model_validate({}).duplicate_candidate_id == -1


class TestTimelineGateEvents:
    def test_legacy_single_event_shape(self):
        v = TimelineGateEvents.model_validate({"event": "did a thing"})
        assert len(v.events) == 1
        assert v.events[0].event == "did a thing"

    def test_legacy_null_event_is_empty(self):
        assert TimelineGateEvents.model_validate({"event": None}).events == []

    def test_events_capped_at_three(self):
        v = TimelineGateEvents.model_validate({"events": [{"event": f"e{i}"} for i in range(5)]})
        assert len(v.events) == TimelineGateEvents.MAX_EVENTS

    def test_non_list_events_raises(self):
        with pytest.raises(ValidationError):
            TimelineGateEvents.model_validate({"events": "nope"})

    def test_field_degradation_matches_legacy_parser(self):
        v = TimelineGateEvents.model_validate(
            {
                "events": [
                    {
                        "event": " ran the benchmark ",
                        "salience": 9,
                        "event_type": "vibe",
                        "domain": "work",
                        "date": 20260620,
                    }
                ]
            }
        )
        e = v.events[0]
        assert e.event == "ran the benchmark"
        assert e.salience == 1
        assert e.event_type is None
        assert e.domain is None
        assert e.date is None

    def test_junk_items_dropped(self):
        v = TimelineGateEvents.model_validate(
            {"events": ["garbage", {"no_event": 1}, {"event": "kept"}]}
        )
        assert [e.event for e in v.events] == ["kept"]


class TestExtractionOutput:
    def test_bad_rows_skipped(self):
        v = ExtractionOutput.model_validate(
            {
                "entities": [{"name": "X"}, {"summary": "no name"}, "junk"],
                "facts": [
                    {"source": "X", "target": "Y", "relationship": "R", "fact": "f"},
                    {"source": "X"},  # missing keys — dropped
                ],
            }
        )
        assert [e.name for e in v.entities] == ["X"]
        assert v.entities[0].type == "Topic"
        assert len(v.facts) == 1

    def test_null_type_and_summary_default(self):
        v = ExtractionOutput.model_validate(
            {"entities": [{"name": "X", "type": None, "summary": None}], "facts": []}
        )
        assert v.entities[0].type == "Topic"
        assert v.entities[0].summary == ""


class TestJsonExtraction:
    def test_first_json_object_tolerates_prose(self):
        raw = 'Sure! Here you go: {"a": 1} hope that helps'
        assert json.loads(first_json_object(raw)) == {"a": 1}

    def test_no_object_raises(self):
        with pytest.raises(ValueError):
            first_json_object("nothing here")

    def test_prose_wrapped_model_validation_end_to_end(self):
        raw = 'prefix {"contradicted_facts": [{"index": 2}]} suffix'
        v = ContradictionVerdict.model_validate_json(first_json_object(raw))
        assert v.contradicted_facts == [2]
