"""Tests for the web_chunk extraction lane (task #68): extract_web prompt
selection and the date-precedence threading into stage 7 rows."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

from ingestion.extractor import LLMExtractor
from ingestion.models import ExtractionResult


def _mock_response(text: str) -> Any:
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


_GOOD_JSON = (
    '{"entities": [{"name": "Trafilatura", "type": "Technology"}],'
    ' "facts": [{"source": "Trafilatura", "target": "Trafilatura",'
    ' "relationship": "ACHIEVES", "fact": "Trafilatura achieves F1 0.93"}]}'
)


class TestExtractWeb:
    def _client_and_prompt(self, provenance: dict[str, Any]) -> str:
        client = MagicMock()
        client.messages.create.return_value = _mock_response(_GOOD_JSON)
        extractor = LLMExtractor(llm_client=client, model="claude-haiku-4-5")
        result = extractor.extract_web("some chunk text", [], provenance)
        assert isinstance(result, ExtractionResult)
        assert [e.name for e in result.entities] == ["Trafilatura"]
        return client.messages.create.call_args.kwargs["messages"][0]["content"]

    def test_raw_scrape_prompt(self):
        prompt = self._client_and_prompt(
            {
                "web_artifact_id": 7,
                "url": "https://example.com/post",
                "title": "Boilerplate removal",
                "kind": "web_scrape",
                "synthesized": False,
                "fetched_at": datetime(2026, 6, 1, tzinfo=UTC),
                "published_at": None,
            }
        )
        assert "a scraped web page" in prompt
        assert "https://example.com/post" in prompt
        assert "ATTRIBUTION FIREWALL" in prompt
        # closed vocabulary present
        assert "Person, Organization, Product, Technology" in prompt

    def test_synthesized_prompt(self):
        prompt = self._client_and_prompt(
            {
                "web_artifact_id": 7,
                "url": "https://example.com/x",
                "title": None,
                "kind": "web_scrape",
                "synthesized": True,
                "fetched_at": None,
                "published_at": None,
            }
        )
        assert "AI-generated answer" in prompt

    def test_research_brief_prompt(self):
        prompt = self._client_and_prompt(
            {
                "web_artifact_id": 9,
                "url": "research://web-ingestion",
                "title": "Web ingestion brief",
                "kind": "research_brief",
                "synthesized": True,
                "fetched_at": None,
                "published_at": None,
            }
        )
        assert "research brief" in prompt
