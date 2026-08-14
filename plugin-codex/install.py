#!/usr/bin/env python3
"""Install the Synapse hook + MCP server into a local Codex CLI.

Codex v0.147 plugins cannot bundle hooks (the plugin_hooks feature is off in
shipped builds), so this installer wires the two halves directly:

  1. Appends a ``[[hooks.Stop]]`` block to ~/.codex/config.toml pointing at
     hooks/synapse_stop_hook.py (absolute path, resolved from this repo).
  2. Registers the Synapse MCP server via ``codex mcp add synapse``.

Idempotent: re-running skips anything already present. Nothing else in
config.toml is touched — the hook block is appended, not merged.

Usage:
    python3 plugin-codex/install.py [--synapse-url URL] [--dry-run]

After installing, run ``codex`` once and trust the new hook via /hooks —
Codex refuses to run unreviewed hooks.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

CONFIG_PATH = Path(os.path.expanduser("~/.codex/config.toml"))
HOOK_SCRIPT = (Path(__file__).parent / "hooks" / "synapse_stop_hook.py").resolve()
MARKER = "synapse_stop_hook.py"


def install_hook(dry_run: bool) -> None:
    existing = CONFIG_PATH.read_text(encoding="utf-8") if CONFIG_PATH.exists() else ""
    if MARKER in existing:
        print(f"hook: already present in {CONFIG_PATH} — skipping")
        return
    block = (
        "\n# Synapse ingest hook (installed by synapse plugin-codex/install.py)\n"
        "[[hooks.Stop]]\n"
        "[[hooks.Stop.hooks]]\n"
        'type = "command"\n'
        f'command = "python3 {HOOK_SCRIPT}"\n'
        "timeout = 30\n"
        'statusMessage = "Syncing to Synapse"\n'
    )
    if dry_run:
        print(f"hook: would append to {CONFIG_PATH}:\n{block}")
        return
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "a", encoding="utf-8") as f:
        f.write(block)
    print(f"hook: appended Stop hook to {CONFIG_PATH}")


def install_mcp(synapse_url: str, dry_run: bool) -> None:
    codex = shutil.which("codex")
    if not codex:
        print("mcp: codex binary not on PATH — skipping MCP registration", file=sys.stderr)
        return
    have = subprocess.run([codex, "mcp", "get", "synapse"], capture_output=True, text=True)
    if have.returncode == 0:
        print("mcp: server 'synapse' already registered — skipping")
        return
    cmd = [
        codex,
        "mcp",
        "add",
        "synapse",
        "--url",
        synapse_url.rstrip("/") + "/mcp",
        "--bearer-token-env-var",
        "SYNAPSE_INGEST_TOKEN",
    ]
    if dry_run:
        print(f"mcp: would run: {' '.join(cmd)}")
        return
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"mcp: codex mcp add failed: {r.stderr.strip()}", file=sys.stderr)
    else:
        print("mcp: registered 'synapse' MCP server")


def main() -> int:
    parser = argparse.ArgumentParser(description="Install Synapse into Codex CLI")
    parser.add_argument(
        "--synapse-url",
        default=os.environ.get("SYNAPSE_URL", "http://localhost:8765"),
        help="Synapse base URL (default: $SYNAPSE_URL or http://localhost:8765)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not HOOK_SCRIPT.exists():
        print(f"error: {HOOK_SCRIPT} not found", file=sys.stderr)
        return 2

    install_hook(args.dry_run)
    install_mcp(args.synapse_url, args.dry_run)
    print(
        "\nDone. Next: start `codex`, run /hooks, and trust the Synapse Stop hook.\n"
        "Set SYNAPSE_INGEST_TOKEN in the environment Codex runs under if your\n"
        "server is auth-gated."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
