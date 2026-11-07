# mypy: ignore-errors
"""The one stripped `claude -p` call the skills autopilot's reviewer makes, live and in the
calibration alike (scripts/calibrate_skills_autoreview.py), so the gate measures exactly what
runs.

A default `claude -p` loads ~16k tokens of tool definitions and makes a ~5.5k side call; with
only `--tools ""` it can still pull in the claude.ai connector schemas (~76k). This call
carries none of that: no tools, no settings, no MCP, no hooks, no memory files, no transcript,
no telemetry or side calls, so the input is the prompt plus a ~1.6k-token system preamble.

    claude -p <prompt> --model <model> --no-session-persistence --setting-sources ""
           --strict-mcp-config --tools ""

    env: ENABLE_CLAUDEAI_MCP_SERVERS=false  DISABLE_NON_ESSENTIAL_MODEL_CALLS=1
         CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
         CLAUDE_CODE_DISABLE_CLAUDE_MDS=1  CLAUDE_CODE_DISABLE_AUTO_MEMORY=1
         (the last two keep a workstation's CLAUDE.md and memory out of the verdict)
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile

STRIP_ENV = {
    "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
    "DISABLE_NON_ESSENTIAL_MODEL_CALLS": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
}


def argv(prompt: str, model: str, *, as_json: bool = False) -> list[str]:
    cmd = [
        "claude",
        "-p",
        prompt,
        "--model",
        model,
        "--no-session-persistence",
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--tools",
        "",
    ]
    return [*cmd, "--output-format", "json"] if as_json else cmd


def env() -> dict[str, str]:
    return {**os.environ, **STRIP_ENV}


def run(prompt: str, model: str, *, timeout: int = 300, cwd: str | None = None) -> str:
    """The model's text answer ('' when the CLI is missing, not logged in, or times out).
    cwd defaults to a fresh empty directory so no project files are in reach."""
    try:
        with tempfile.TemporaryDirectory(prefix="synapse-review-") as scratch:
            r = subprocess.run(
                argv(prompt, model),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env(),
                cwd=cwd or scratch,
            )
    except (OSError, subprocess.SubprocessError) as e:
        print(f"  claude cli unavailable: {e}")
        return ""
    return r.stdout or ""


def run_with_usage(prompt: str, model: str, *, timeout: int = 300) -> dict:
    """{'text', 'usage', 'model', 'cost_usd'} from `--output-format json`: how the call's
    token use was checked against the prompt's size."""
    with tempfile.TemporaryDirectory(prefix="synapse-review-") as scratch:
        r = subprocess.run(
            argv(prompt, model, as_json=True),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env(),
            cwd=scratch,
        )
    d = json.loads(r.stdout)
    return {
        "text": d.get("result") or "",
        "usage": d.get("usage") or {},
        "model": sorted((d.get("modelUsage") or {}).keys()),
        "cost_usd": d.get("total_cost_usd"),
    }
