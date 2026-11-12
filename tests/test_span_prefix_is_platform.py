"""A turn's span id starts with the platform that produced it (schema 068).

One name per client: the platform a parser stamps is also the prefix of the span ids it
writes. That is what lets the messages format reserve only platform names, and what the
068 trigger assumes when it rewrites the three spellings older builds wrote.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ingestion.messages_client import RESERVED_CLIENTS, MessagesParser

_INGESTION = Path(__file__).resolve().parents[1] / "ingestion"
_PARSERS = (
    "jsonl_client.py",
    "codex_client.py",
    "openai_client.py",
    "cursor_sqlite_client.py",
    "claude_ai_client.py",
)
_PLATFORM = re.compile(r'platform="([a-z_]+)"')
_SPAN_PREFIX = re.compile(r'span_id\s*=\s*f"([a-z_-]+):')


@pytest.mark.parametrize("module", _PARSERS)
def test_a_parser_prefixes_its_span_ids_with_its_platform(module: str) -> None:
    source = (_INGESTION / module).read_text()
    platforms = set(_PLATFORM.findall(source))
    prefixes = set(_SPAN_PREFIX.findall(source))
    assert len(platforms) == 1, f"{module}: expected one platform literal, found {platforms}"
    assert prefixes == platforms, f"{module}: span-id prefix {prefixes} != platform {platforms}"


def test_the_reserved_client_names_are_exactly_the_other_parsers_platforms() -> None:
    stamped = {
        platform
        for module in _PARSERS
        for platform in _PLATFORM.findall((_INGESTION / module).read_text())
    }
    assert stamped == RESERVED_CLIENTS


def test_the_messages_parser_follows_the_same_rule() -> None:
    (episode,) = MessagesParser().parse_records(
        [{"role": "user", "content": "a question worth storing", "id": "m1"}],
        "test",
        session_id="s1",
        client="hermes",
    )
    assert episode.platform == "hermes"
    assert episode.span_id == "hermes:m1"


def test_the_migration_renames_exactly_the_three_format_named_prefixes() -> None:
    sql = (_INGESTION.parent / "schema" / "068_span_id_prefix_is_platform.sql").read_text()
    function = sql[
        sql.index("FUNCTION canonical_span_id") : sql.index("FUNCTION episodes_canonical")
    ]
    renames = dict(re.findall(r"LIKE '([a-z-]+):%'\s+THEN '([a-z_]+):'", function))
    assert renames == {"jsonl": "claude_code", "openai": "chatgpt", "cursor-sqlite": "cursor"}
    assert set(renames.values()) <= RESERVED_CLIENTS
