"""PostToolUse hook (plugin/scripts/recall_feedback_nudge.py): opt-in gating.

recall_feedback labels are offline tuning data — useful only to whoever tunes the
retrieval stack, noise for everyone else. The nudge therefore ships OFF and turns
on only with SYNAPSE_RECALL_FEEDBACK_NUDGE=1 (flipped from default-on 2026-08-26).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_HOOK_PY = (
    Path(__file__).resolve().parents[1] / "plugin" / "scripts" / "recall_feedback_nudge.py"
)


def _run(tmp_path, env: dict | None = None) -> str:
    full_env = {
        **{k: v for k, v in os.environ.items() if k != "SYNAPSE_RECALL_FEEDBACK_NUDGE"},
        "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
        "SYNAPSE_DATA_DIR": str(tmp_path / "data"),
        **(env or {}),
    }
    out = subprocess.run(
        [sys.executable, str(_HOOK_PY)],
        capture_output=True,
        text=True,
        env=full_env,
        cwd=tmp_path,
        check=True,
    )
    return out.stdout


def test_default_off_prints_nothing(tmp_path):
    assert _run(tmp_path) == ""


def test_explicit_zero_prints_nothing(tmp_path):
    assert _run(tmp_path, {"SYNAPSE_RECALL_FEEDBACK_NUDGE": "0"}) == ""


def test_opt_in_emits_posttooluse_context(tmp_path):
    out = _run(tmp_path, {"SYNAPSE_RECALL_FEEDBACK_NUDGE": "1"})
    payload = json.loads(out)
    ctx = payload["hookSpecificOutput"]["additionalContext"]
    assert payload["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
    assert "recall_feedback" in ctx
