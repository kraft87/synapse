"""strip_nul: NUL bytes must never reach Postgres (TEXT and jsonb both reject them)."""

from __future__ import annotations

from datetime import UTC, datetime

from ingestion.textsafe import strip_nul


def test_strips_nul_from_strings():
    assert strip_nul("clean") == "clean"
    assert strip_nul("a\x00b") == "ab"
    assert strip_nul("\x00\x00") == ""


def test_recurses_into_containers():
    nested = {"k": "v\x00w", "list": ["ok\x00", {"deep": "x\x00y"}], "tup": ("a\x00",)}
    assert strip_nul(nested) == {"k": "vw", "list": ["ok", {"deep": "xy"}], "tup": ("a",)}


def test_non_strings_pass_through():
    when = datetime(2026, 7, 23, tzinfo=UTC)
    assert strip_nul(None) is None
    assert strip_nul(42) == 42
    assert strip_nul(when) is when
