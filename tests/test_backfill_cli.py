"""``python -m ingestion.backfill``: one driver for every source (no database needed here)."""

from __future__ import annotations

import sys

import pytest

from ingestion import backfill
from ingestion.models import Episode


def _run(monkeypatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["backfill", *argv])
    return backfill.main()


def test_every_source_has_a_parser():
    assert sorted(backfill.SOURCES) == ["chatgpt", "claude-ai", "claude-code", "codex", "cursor"]


def test_a_source_without_a_default_root_needs_a_path(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, "--source", "cursor")
    assert exc.value.code == 2
    assert "needs a path" in capsys.readouterr().err


def test_directory_flags_are_refused_for_a_single_export(monkeypatch, tmp_path, capsys):
    export = tmp_path / "conversations.json"
    export.write_text("[]")
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, "--source", "claude-ai", "--since-days", "3", str(export))
    assert exc.value.code == 2
    assert "one export" in capsys.readouterr().err


def test_a_missing_path_is_an_error_not_an_empty_run(monkeypatch, tmp_path, capsys):
    assert _run(monkeypatch, str(tmp_path / "nowhere")) == 2
    assert "does not exist" in capsys.readouterr().err


def test_the_codex_root_defaults_to_the_codex_sessions_dir(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setitem(
        backfill.SOURCES,
        "codex",
        backfill.Source("rollout-*.jsonl", lambda p, project: [], default_root=tmp_path),
    )
    monkeypatch.setattr(backfill, "get_settings", lambda: type("S", (), {"db_url": "x"})())
    monkeypatch.setattr(
        backfill,
        "backfill",
        lambda source, path, db, **kw: seen.update(source=source, path=path) or {},
    )
    assert _run(monkeypatch, "--source", "codex", "--dry-run") == 0
    assert seen == {"source": "codex", "path": tmp_path}


def test_a_malformed_single_export_is_an_error_not_an_empty_run(monkeypatch, tmp_path):
    """The export is the whole run: one that does not parse must fail the command, not print
    'files: 0' and exit 0 (a directory sweep warns and carries on, below)."""
    monkeypatch.setattr(backfill, "get_settings", lambda: type("S", (), {"db_url": "x"})())
    export = tmp_path / "conversations.json"
    export.write_text("{not json")
    with pytest.raises(ValueError):
        _run(monkeypatch, "--source", "claude-ai", str(export))
    export.write_text('{"not": "a list"}')
    with pytest.raises(ValueError, match="Expected list"):
        _run(monkeypatch, "--source", "claude-ai", str(export))


def test_a_directory_sweep_skips_a_file_that_does_not_parse(monkeypatch, tmp_path, caplog):
    def parse(path, project):
        if path.name == "bad.jsonl":
            raise ValueError("garbled")
        return [Episode(session_id="s", sequence=1, content="[user] hi", span_id="x:1")]

    monkeypatch.setitem(backfill.SOURCES, "codex", backfill.Source("*.jsonl", parse))
    for name in ("bad.jsonl", "good.jsonl"):
        (tmp_path / name).write_text("")
    stats = backfill.backfill("codex", tmp_path, db=None, dry_run=True)
    assert (stats["files"], stats["episodes_parsed"]) == (1, 1)
    assert "Failed to parse" in caplog.text and "garbled" in caplog.text
