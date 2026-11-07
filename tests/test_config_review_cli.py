"""plugin/scripts/config_review.py list/show: what the reviewer sees of the nightly judge's
recommendation and the days a proposal has waited. No DB, no network: config.post_json is
stubbed with canned server replies. All rules are synthetic.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "plugin", "scripts"))
import config
import config_review


@pytest.fixture()
def reply(monkeypatch):
    box: dict = {}
    monkeypatch.setattr(config, "post_json", lambda path, payload, timeout=30.0: box["r"])
    return box


def _row(**over):
    row = {
        "id": 3,
        "kind": "add",
        "file_key": "rules/learned.md",
        "scope": "general",
        "summary": "Write short answers",
        "sessions": 2,
    }
    row.update(over)
    return row


def test_list_shows_days_pending_and_the_recommendation(reply, capsys):
    reply["r"] = {
        "proposals": [
            _row(
                days_pending=9,
                recommendation={"verdict": "accept", "reason": "A standing preference."},
            )
        ]
    }
    config_review.cmd_list()
    out = capsys.readouterr().out
    assert "[3] add -> rules/learned.md (general)  2 session(s)  9d pending" in out
    assert "recommends accept: A standing preference." in out


def test_list_from_a_server_before_schema_065(reply, capsys):
    reply["r"] = {"proposals": [_row()]}
    config_review.cmd_list()
    out = capsys.readouterr().out
    assert "[3] add -> rules/learned.md (general)  2 session(s)\n" in out
    assert "pending" not in out and "recommends" not in out


def test_list_with_no_recommendation_yet(reply, capsys):
    reply["r"] = {"proposals": [_row(days_pending=0, recommendation=None)]}
    config_review.cmd_list()
    out = capsys.readouterr().out
    assert "0d pending" in out and "recommends" not in out


def test_show_prints_the_recommendation(reply, capsys):
    reply["r"] = {
        "found": True,
        "id": 3,
        "kind": "add",
        "file_key": "rules/learned.md",
        "scope": "general",
        "status": "proposed",
        "diff": "Write short answers",
        "recommendation": {"verdict": "reject", "reason": "Two quotes conflict."},
        "evidence": [{"session_id": "abcdef123456", "quote": "keep it short"}],
    }
    config_review.cmd_show(3)
    out = capsys.readouterr().out
    assert "rule: Write short answers" in out
    assert "recommends reject: Two quotes conflict." in out
    assert "sess=abcdef12 keep it short" in out
