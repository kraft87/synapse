"""Pure tests for the write-back the config review CLI does on accept and apply
(plugin config_review._apply_rule, cmd_accept, cmd_apply).

No DB, no network (the server is a stub). Guards the on-disk behavior: a missing file is created
with a header, a rule is appended as one bullet line, re-applying the same rule is idempotent (no
duplicate line, but a longer line containing it is not the same rule), a path outside the config
dir is never written, and a rule approved elsewhere (the dashboard) is written by `apply`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "plugin", "scripts"))
import config
import config_review


def test_apply_creates_appends_and_dedups(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", Path(tmp_path))
    target = Path(tmp_path) / "rules" / "learned.md"

    path, wrote = config_review._apply_rule("rules/learned.md", "Do not use em-dashes")
    assert wrote and Path(path) == target
    body = target.read_text()
    assert body.startswith("# Learned rules")  # header seeded on first write
    assert "- Do not use em-dashes" in body

    # a second, distinct rule appends below the first
    _, wrote2 = config_review._apply_rule("rules/learned.md", "Filter out contract jobs")
    assert wrote2
    body2 = target.read_text()
    assert "- Do not use em-dashes" in body2 and "- Filter out contract jobs" in body2

    # re-applying an existing rule is a no-op (no duplicate line)
    _, wrote3 = config_review._apply_rule("rules/learned.md", "Do not use em-dashes")
    assert wrote3 is False
    assert target.read_text().count("- Do not use em-dashes") == 1


def test_a_rule_that_is_only_part_of_a_longer_line_is_still_written(tmp_path, monkeypatch):
    # "- Use uv" is a substring of "- Use uv run for scripts", not the same rule
    monkeypatch.setattr(config, "CONFIG_DIR", Path(tmp_path))
    target = Path(tmp_path) / "rules" / "learned.md"
    target.parent.mkdir()
    target.write_text("# Learned rules\n\n- Use uv run for scripts\n")

    _, wrote = config_review._apply_rule("rules/learned.md", "Use uv")

    assert wrote
    assert target.read_text().splitlines()[-2:] == ["- Use uv run for scripts", "- Use uv"]
    assert config_review._apply_rule("rules/learned.md", "  Use uv ")[1] is False  # exact line


def test_a_rule_is_written_as_one_line(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_DIR", Path(tmp_path))
    config_review._apply_rule("rules/learned.md", "Never summarize\n\n## Injected heading\n- x")
    body = (Path(tmp_path) / "rules" / "learned.md").read_text()
    assert body.splitlines()[-1] == "- Never summarize ## Injected heading - x"
    with pytest.raises(ValueError, match="no rule text"):
        config_review._apply_rule("rules/learned.md", " \n ")


@pytest.mark.parametrize(
    "key",
    [
        "/etc/cron.d/x",
        "../outside.md",
        "rules/../../outside.md",
        "rules//x.md",
        "a\\b.md",
        "c:x",
        "",
    ],
)
def test_a_file_key_outside_the_config_dir_is_refused(tmp_path, monkeypatch, key):
    root = Path(tmp_path) / "claude"
    monkeypatch.setattr(config, "CONFIG_DIR", root)
    with pytest.raises(ValueError, match="refusing to write"):
        config_review._apply_rule(key, "a rule")
    assert not root.exists() and not (Path(tmp_path) / "outside.md").exists()


class _Server:
    """Stands in for config.post_json: lists two approved proposals and one still to decide."""

    def __init__(self, rows):
        self.rows, self.applied = rows, []

    def __call__(self, path, payload):
        if path == "/config/proposals":
            return {"proposals": self.rows}
        assert path == "/config/proposals/act" and payload["action"] == "apply"
        self.applied.append(payload["id"])
        return {"status": "applied"}


def _row(cid, status, rule, file_key="rules/learned.md"):
    return {
        "id": cid, "kind": "add", "file_key": file_key, "scope": "general", "status": status,
        "rule": rule, "summary": rule, "sessions": 2,
    }  # fmt: skip


def test_apply_writes_every_approved_rule_and_only_those(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "CONFIG_DIR", Path(tmp_path))
    server = _Server(
        [
            _row(1, "accepted", "Do not use em-dashes"),
            _row(2, "proposed", "Still to be decided"),
            _row(3, "accepted", "Escapes", file_key="../x.md"),  # refused, stays accepted
            _row(4, "accepted", "Filter out contract jobs"),
        ]
    )
    monkeypatch.setattr(config, "post_json", server)

    config_review.cmd_apply()

    body = (Path(tmp_path) / "rules" / "learned.md").read_text()
    assert "- Do not use em-dashes" in body and "- Filter out contract jobs" in body
    assert "Still to be decided" not in body
    assert server.applied == [1, 4]  # 3 was not written, so it is not marked applied
    assert "accepted [3], not written: refusing to write" in capsys.readouterr().out


def test_list_marks_the_rules_waiting_for_a_disk_write(monkeypatch, capsys):
    monkeypatch.setattr(
        config,
        "post_json",
        _Server(
            [_row(1, "proposed", "Decide me"), _row(2, "accepted", "Approved on the dashboard")]
        ),
    )
    config_review.cmd_list()
    lines = capsys.readouterr().out.splitlines()
    assert "APPROVED" not in lines[0] and "[2]" in lines[2] and "APPROVED" in lines[2]


def test_accept_does_not_mark_applied_a_rule_it_could_not_write(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "CONFIG_DIR", Path(tmp_path))
    accepted = _row(7, "accepted", "Escapes", file_key="/etc/x.md")
    monkeypatch.setattr(config, "act", lambda *a, **kw: accepted)
    server = _Server([])
    monkeypatch.setattr(config, "post_json", server)

    config_review.cmd_accept(7, local=False)

    assert server.applied == []
    assert "accepted [7], not written" in capsys.readouterr().out
