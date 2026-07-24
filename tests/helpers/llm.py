"""Shared Anthropic-shaped LLM mocks for the extraction/dedup/contradiction tests.

``mock_llm_message`` builds a single response object shaped like the SDK's
``messages.create`` return (``.content[0].text``); ``mock_llm_client`` wraps it
in a full client mock whose ``messages.create`` returns that message."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock


def mock_llm_message(text: str) -> Any:
    """A MagicMock shaped like an Anthropic message: ``.content[0].text == text``."""
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


def mock_llm_client(text: str) -> MagicMock:
    """A MagicMock LLM client whose ``messages.create`` returns ``mock_llm_message(text)``."""
    client = MagicMock()
    client.messages.create.return_value = mock_llm_message(text)
    return client
