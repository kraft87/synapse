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


def test_check_continues_when_database_is_ahead(tmp_path, monkeypatch, caplog):
    # A migration stamped before the new image lands, or an older image redeployed
    # after the stamp, must not take the service down.
    d = _make_schema_dir(tmp_path, ["039_schema_meta.sql"])
    monkeypatch.setattr(schema_check, "applied_schema_version", lambda url: "041")
    with caplog.at_level("WARNING"):
        check_schema_version("postgresql://x", schema_dir=d)  # no exit
    assert "ahead of this build" in caplog.text


def _make_schema_dir_with(tmp_path: Path, files: dict[str, bool]) -> Path:
    """{filename: optional?} -> a schema dir whose optional files carry the marker."""
    d = tmp_path / "schema"
    d.mkdir()
    for name, optional in files.items():
        marker = f"{schema_check.OPTIONAL_MARKER}\n" if optional else ""
        (d / name).write_text(f"-- header\n{marker}SELECT 1;\n")
    return d


def test_required_version_skips_an_optional_tail(tmp_path):
    d = _make_schema_dir_with(tmp_path, {"039_a.sql": False, "040_b.sql": True, "041_c.sql": True})
    assert expected_schema_version(d) == "041"
    assert schema_check.required_schema_version(d) == "039"


def test_an_optional_migration_followed_by_a_required_one_relaxes_nothing(tmp_path):
    d = _make_schema_dir_with(tmp_path, {"039_a.sql": False, "040_b.sql": True, "041_c.sql": False})
    assert schema_check.required_schema_version(d) == "041"


def test_the_marker_must_be_a_line_of_its_own(tmp_path):
    d = tmp_path / "schema"
    d.mkdir()
    (d / "040_b.sql").write_text(f"-- mentions {schema_check.OPTIONAL_MARKER} in prose\n")
    assert not schema_check.is_optional_migration(d / "040_b.sql")


def test_check_continues_when_behind_only_by_optional_migrations(tmp_path, monkeypatch, caplog):
    # The image lands before an optional migration is applied by hand: boot, warn.
    d = _make_schema_dir_with(tmp_path, {"039_a.sql": False, "040_b.sql": True})
    monkeypatch.setattr(schema_check, "applied_schema_version", lambda url: "039")
    with caplog.at_level("WARNING"):
        check_schema_version("postgresql://x", schema_dir=d)  # no exit
    assert "marked optional" in caplog.text


def test_check_exits_when_behind_a_required_migration_despite_an_optional_tail(
    tmp_path, monkeypatch
):
    d = _make_schema_dir_with(tmp_path, {"038_a.sql": False, "039_b.sql": False, "040_c.sql": True})
    monkeypatch.setattr(schema_check, "applied_schema_version", lambda url: "038")
    with pytest.raises(SystemExit):
        check_schema_version("postgresql://x", schema_dir=d)


def test_an_optional_tail_does_not_excuse_a_missing_stamp(tmp_path, monkeypatch):
    d = _make_schema_dir_with(tmp_path, {"039_a.sql": False, "040_b.sql": True})
    monkeypatch.setattr(schema_check, "applied_schema_version", lambda url: None)
    with pytest.raises(SystemExit):
        check_schema_version("postgresql://x", schema_dir=d)


def test_check_exits_on_unparseable_stamp(tmp_path, monkeypatch):
    d = _make_schema_dir(tmp_path, ["039_schema_meta.sql"])
    monkeypatch.setattr(schema_check, "applied_schema_version", lambda url: "garbage")
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
