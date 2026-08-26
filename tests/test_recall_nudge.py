"""UserPromptSubmit hook (plugin/scripts/recall_nudge.py): timestamp + nudge lines.

The hook injects a host-local timestamp line (weekday included — models misderive
day-of-week from a bare date) followed by the recall/remember nudge. Each line has
its own kill switch; both off means no output at all (not a blank line).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

_NUDGE_PY = Path(__file__).resolve().parents[1] / "plugin" / "scripts" / "recall_nudge.py"
_TS_RE = re.compile(r"^\[[A-Z][a-z]{2} \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} [^\]]+\]$")


def _run(tmp_path, env: dict | None = None) -> str:
    full_env = {
        **os.environ,
        "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
        "SYNAPSE_DATA_DIR": str(tmp_path / "data"),
        "SYNAPSE_RECALL_NUDGE": "",
        "SYNAPSE_PROMPT_TIMESTAMP": "",
        **(env or {}),
    }
    full_env = {k: v for k, v in full_env.items() if v != ""}
    out = subprocess.run(
        [sys.executable, str(_NUDGE_PY)],
        capture_output=True,
        text=True,
        env=full_env,
        cwd=tmp_path,
        check=True,
    )
    return out.stdout


def test_default_emits_timestamp_then_nudge(tmp_path):
    lines = _run(tmp_path).splitlines()
    assert len(lines) == 2
    assert _TS_RE.match(lines[0]), lines[0]
    assert lines[1].startswith("[Synapse]")


def test_timestamp_kill_switch(tmp_path):
    lines = _run(tmp_path, {"SYNAPSE_PROMPT_TIMESTAMP": "0"}).splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("[Synapse]")


def test_nudge_kill_switch_keeps_timestamp(tmp_path):
    lines = _run(tmp_path, {"SYNAPSE_RECALL_NUDGE": "0"}).splitlines()
    assert len(lines) == 1
    assert _TS_RE.match(lines[0]), lines[0]


def test_both_off_prints_nothing(tmp_path):
    assert _run(tmp_path, {"SYNAPSE_RECALL_NUDGE": "0", "SYNAPSE_PROMPT_TIMESTAMP": "0"}) == ""
