"""PostToolUse hook (plugin/scripts/recall_feedback_nudge.py): opt-in gating.

recall_feedback labels are offline tuning data — useful only to whoever tunes the
retrieval stack, noise for everyone else. The nudge therefore ships OFF and turns
on only with SYNAPSE_RECALL_FEEDBACK_NUDGE=1 (flipped from default-on 2026-08-26) or `synapse
config recall_feedback_nudge on`. Codex had kept the old default-on: both harnesses share
the text and the switch now.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_HOOK_PY = Path(__file__).resolve().parents[1] / "plugin" / "scripts" / "recall_feedback_nudge.py"


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


_CODEX_HOOK_PY = Path(__file__).resolve().parents[1] / "plugin-codex" / "hooks" / "post_tool_use.py"


def _run_codex(tmp_path, tool: str, env: dict | None = None, config: dict | None = None) -> str:
    home = tmp_path / "synapse-home"
    home.mkdir(exist_ok=True)
    if config is not None:
        (home / "config.json").write_text(json.dumps(config))
    full_env = {
        **{k: v for k, v in os.environ.items() if not k.startswith("SYNAPSE_")},
        "HOME": str(tmp_path),
        "SYNAPSE_HOME": str(home),
        **(env or {}),
    }
    return subprocess.run(
        [sys.executable, str(_CODEX_HOOK_PY)],
        input=json.dumps({"tool_name": tool}),
        capture_output=True,
        text=True,
        env=full_env,
        cwd=tmp_path,
        check=True,
    ).stdout


def test_codex_default_off_prints_nothing(tmp_path):
    assert _run_codex(tmp_path, "mcp__synapse__recall") == ""


def test_codex_opt_in_by_env_or_config_nudges_after_a_recall(tmp_path):
    for out in (
        _run_codex(tmp_path, "mcp__synapse__recall", {"SYNAPSE_RECALL_FEEDBACK_NUDGE": "1"}),
        _run_codex(
            tmp_path, "mcp__synapse__recall_full_turns", config={"recall_feedback_nudge": True}
        ),
    ):
        payload = json.loads(out)
        assert payload["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
        assert "recall_feedback" in payload["hookSpecificOutput"]["additionalContext"]


def test_codex_opt_in_skips_tools_that_serve_no_rateable_ids(tmp_path):
    env = {"SYNAPSE_RECALL_FEEDBACK_NUDGE": "1"}
    assert _run_codex(tmp_path, "mcp__synapse__recall_feedback", env) == ""
    assert _run_codex(tmp_path, "mcp__synapse__fetch", env) == ""


def test_both_harnesses_print_the_same_text(tmp_path):
    claude = _run(tmp_path, {"SYNAPSE_RECALL_FEEDBACK_NUDGE": "1"})
    codex = _run_codex(tmp_path, "mcp__synapse__recall", {"SYNAPSE_RECALL_FEEDBACK_NUDGE": "1"})
    assert json.loads(claude) == json.loads(codex)
