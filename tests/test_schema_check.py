"""Boot-time schema-version guard (ingestion/schema_check.py).

Pure unit tests — the database side is monkeypatched; the real stamp is
written by scripts/apply_schema.sh, which CI exercises when provisioning
the ephemeral test database.
"""

from pathlib import Path

import pytest

from ingestion import schema_check
from ingestion.schema_check import check_schema_version, expected_schema_version


def _make_schema_dir(tmp_path: Path, names: list[str]) -> Path:
    d = tmp_path / "schema"
    d.mkdir()
    for n in names:
        (d / n).write_text("-- test")
    return d


def test_expected_version_is_highest_numeric_prefix(tmp_path):
    d = _make_schema_dir(
        tmp_path, ["001_init.sql", "014_hnsw.sql", "039_schema_meta.sql", "notes.txt"]
    )
    assert expected_schema_version(d) == "039"


def test_expected_version_ignores_non_migration_files(tmp_path):
    d = _make_schema_dir(tmp_path, ["README.sql", "no_prefix.sql"])
    assert expected_schema_version(d) is None


def test_expected_version_missing_dir_returns_none(tmp_path):
    assert expected_schema_version(tmp_path / "nope") is None


def test_repo_schema_dir_is_discovered():
    # The default candidate list must find the real checkout's schema/.
    v = expected_schema_version()
    assert v is not None and v >= "038"


def test_check_passes_on_match(tmp_path, monkeypatch):
    d = _make_schema_dir(tmp_path, ["039_schema_meta.sql"])
    monkeypatch.setattr(schema_check, "applied_schema_version", lambda url: "039")
    check_schema_version("postgresql://x", schema_dir=d)  # no exit


def test_check_exits_when_database_is_behind(tmp_path, monkeypatch):
    d = _make_schema_dir(tmp_path, ["039_schema_meta.sql"])
    monkeypatch.setattr(schema_check, "applied_schema_version", lambda url: "037")
    with pytest.raises(SystemExit):
        check_schema_version("postgresql://x", schema_dir=d)


def test_check_exits_when_stamp_is_missing(tmp_path, monkeypatch):
    d = _make_schema_dir(tmp_path, ["039_schema_meta.sql"])
    monkeypatch.setattr(schema_check, "applied_schema_version", lambda url: None)
    with pytest.raises(SystemExit):
        check_schema_version("postgresql://x", schema_dir=d)


def test_kill_switch_skips_check(tmp_path, monkeypatch):
    d = _make_schema_dir(tmp_path, ["039_schema_meta.sql"])
    monkeypatch.setenv("SYNAPSE_SCHEMA_CHECK", "0")
    monkeypatch.setattr(schema_check, "applied_schema_version", lambda url: None)
    check_schema_version("postgresql://x", schema_dir=d)  # no exit


def test_unreachable_database_fails_open(tmp_path, monkeypatch):
    d = _make_schema_dir(tmp_path, ["039_schema_meta.sql"])

    def _boom(url):
        raise ConnectionError("refused")

    monkeypatch.setattr(schema_check, "applied_schema_version", _boom)
    check_schema_version("postgresql://x", schema_dir=d)  # warn + continue


def test_no_schema_dir_fails_open(tmp_path, monkeypatch):
    monkeypatch.setattr(
        schema_check, "applied_schema_version", lambda url: pytest.fail("should not query")
    )
    check_schema_version("postgresql://x", schema_dir=tmp_path / "nope")
