"""Pure test for the write-back the config review CLI does on accept (plugin config_review._apply_rule).

No DB, no network. Guards the on-disk behavior: a missing file is created with a header, a rule is
appended as a bullet, and re-applying the same rule is idempotent (no duplicate line).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

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
