"""Schema-058 evidence fields on the extraction wire + model (ongoing / attribution).

The prompt asks for both on every fact; the wire schema tolerates their absence and the
model coerces sloppy values. The invariant that matters: nothing a model omits or
mangles can ever pass for a USER-attributed, ongoing claim by accident.
"""

from __future__ import annotations

import pytest

from ingestion.llm_schemas import ExtractionOutput
from ingestion.models import ExtractedFact


def _row(**over) -> dict:
    row = {"source": "A", "target": "B", "relationship": "R", "fact": "A r B"}
    row.update(over)
    return row


class TestWire:
    def test_fields_optional_on_the_wire(self):
        out = ExtractionOutput.model_validate({"entities": [], "facts": [_row()]})
        assert out.facts[0].ongoing is None
        assert out.facts[0].attribution is None

    def test_fields_pass_through(self):
        out = ExtractionOutput.model_validate(
            {"entities": [], "facts": [_row(ongoing=True, attribution="user")]}
        )
        assert out.facts[0].ongoing is True
        assert out.facts[0].attribution == "user"


class TestModelCoercion:
    def test_defaults_are_the_conservative_reading(self):
        f = ExtractedFact(**_row())
        assert f.ongoing is False
        assert f.attribution == "unknown"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("user", "user"),
            ("User ", "user"),
            ("assistant", "assistant"),
            ("third-party", "third_party"),
            ("document", "third_party"),
            ("other", "third_party"),
            ("", "unknown"),
            (None, "unknown"),
            ("the user", "unknown"),  # anything unrecognised is NOT user
        ],
    )
    def test_attribution_coercion(self, raw, expected):
        assert ExtractedFact(**_row(attribution=raw)).attribution == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (True, True),
            ("true", True),
            ("yes", True),
            ("false", False),
            ("", False),
            (None, False),
            (0, False),
        ],
    )
    def test_ongoing_coercion(self, raw, expected):
        assert ExtractedFact(**_row(ongoing=raw)).ongoing is expected
