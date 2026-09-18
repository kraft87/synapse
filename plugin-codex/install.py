#!/usr/bin/env python3
"""Install the Synapse hook + MCP server into a local Codex CLI.

Codex v0.147 plugins cannot bundle hooks (the plugin_hooks feature is off in
shipped builds), so this installer wires the two halves directly:

  1. Appends a ``[[hooks.Stop]]`` block to ~/.codex/config.toml pointing at
     hooks/synapse_stop_hook.py (absolute path, resolved from this repo).
  2. Registers Synapse MCP with a saved-credential HTTP header helper.

Idempotent: existing Synapse auth and legacy matchers are upgraded in place.
Unrelated configuration and hook trust records are preserved.

Usage:
    python3 plugin-codex/install.py [--synapse-url URL] [--dry-run]

After installing, run ``codex`` once and trust the new hook via /hooks —
Codex refuses to run unreviewed hooks.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
import tomllib
from pathlib import Path

CONFIG_PATH = Path(os.path.expanduser("~/.codex/config.toml"))
HOOKS_DIR = (Path(__file__).parent / "hooks").resolve()
SCRIPTS_DIR = (Path(__file__).parent / "scripts").resolve()

# One entry per hook block; the marker (the script filename) doubles as the
# idempotency check so re-running after an upgrade appends only what's new.
_HOOK_BLOCKS: list[tuple[str, str]] = [
    (
        "synapse_stop_hook.py",
        "[[hooks.Stop]]\n"
        "[[hooks.Stop.hooks]]\n"
        'type = "command"\n'
        f'command = "python3 {HOOKS_DIR}/synapse_stop_hook.py"\n'
        "timeout = 30\n",
    ),
    (
        "session_start.py",
        "[[hooks.SessionStart]]\n"
        "[[hooks.SessionStart.hooks]]\n"
        'type = "command"\n'
        f'command = "python3 {HOOKS_DIR}/session_start.py"\n'
        "timeout = 20\n",
    ),
    (
        "user_prompt_submit.py",
        "[[hooks.UserPromptSubmit]]\n"
        "[[hooks.UserPromptSubmit.hooks]]\n"
        'type = "command"\n'
        f'command = "python3 {HOOKS_DIR}/user_prompt_submit.py"\n'
        "timeout = 5\n",
    ),
    (
        "pre_tool_use.py",
        "[[hooks.PreToolUse]]\n"
        'matcher = "mcp__.*__(recall|recall_full_turns|recall_feedback|fetch_session|remember)$"\n'
        "[[hooks.PreToolUse.hooks]]\n"
        'type = "command"\n'
        f'command = "python3 {HOOKS_DIR}/pre_tool_use.py"\n'
        "timeout = 5\n",
    ),
    (
        "post_tool_use.py",
        "[[hooks.PostToolUse]]\n"
        'matcher = "mcp__.*__(recall|recall_full_turns)$"\n'
        "[[hooks.PostToolUse.hooks]]\n"
        'type = "command"\n'
        f'command = "python3 {HOOKS_DIR}/post_tool_use.py"\n'
        "timeout = 5\n",
    ),
    (
        "private_mode.py --session-end",
        "[[hooks.SessionEnd]]\n"
        "[[hooks.SessionEnd.hooks]]\n"
        'type = "command"\n'
        f'command = "python3 {SCRIPTS_DIR}/private_mode.py --session-end"\n'
        "timeout = 3\n",
    ),
]


def install_hook(dry_run: bool) -> None:
    existing = CONFIG_PATH.read_text(encoding="utf-8") if CONFIG_PATH.exists() else ""
    old_matcher = 'matcher = "mcp__.*__(recall|recall_full_turns|recall_feedback|fetch_session)$"'
    new_matcher = old_matcher.replace("fetch_session)", "fetch_session|remember)")
    hook_blocks = re.compile(
        r"(?ms)^\[\[hooks\.PreToolUse\]\].*?(?=^\[(?!\[hooks\.PreToolUse\.hooks\]\])|\Z)"
    )
    upgraded = hook_blocks.sub(
        lambda match: (
            match[0].replace(old_matcher, new_matcher)
            if "pre_tool_use.py" in match[0]
            else match[0]
        ),
        existing,
    )
    if upgraded != existing:
        tomllib.loads(upgraded)
        if dry_run:
            print("hooks: would upgrade the Synapse session-id matcher to include remember")
        else:
            CONFIG_PATH.write_text(upgraded, encoding="utf-8")
            print("hooks: upgraded the Synapse session-id matcher to include remember")
        existing = upgraded
    missing = [(m, b) for m, b in _HOOK_BLOCKS if m.split()[0] not in existing]
    if not missing:
        print(f"hooks: all {len(_HOOK_BLOCKS)} blocks already present in {CONFIG_PATH} — skipping")
        return
    text = "\n# Synapse hooks (installed by synapse plugin-codex/install.py)\n" + "\n".join(
        b for _, b in missing
    )
    if dry_run:
        print(f"hooks: would append to {CONFIG_PATH}:\n{text}")
        return
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "a", encoding="utf-8") as f:
        f.write(text)
    print(f"hooks: appended {len(missing)} hook block(s) to {CONFIG_PATH}")


def install_mcp(synapse_url: str, dry_run: bool) -> None:
    existing = CONFIG_PATH.read_text(encoding="utf-8") if CONFIG_PATH.exists() else ""
    config = tomllib.loads(existing)
    current = config.get("mcp_servers", {}).get("synapse", {})
    if current.get("command"):
        raise ValueError("Refusing to replace a custom Synapse stdio connection")
    if current.get("http_headers", {}).get("Authorization") or current.get(
        "env_http_headers", {}
    ).get("Authorization"):
        raise ValueError("Remove the custom Synapse Authorization override before migrating")
    url = synapse_url.rstrip("/")
    if not url.endswith("/mcp"):
        url += "/mcp"
    helper = shlex.join(["python3", str(SCRIPTS_DIR / "mcp_headers.py"), "--url", url])
    fields = f"url = {json.dumps(url)}\nhttp_headers_helper = {json.dumps(helper)}\n"
    table = re.search(r"(?m)^\[mcp_servers\.synapse\][ \t]*(?:#.*)?$", existing)
    if table:
        next_table = re.search(r"(?m)^\[", existing[table.end() :])
        end = table.end() + next_table.start() if next_table else len(existing)
        body = existing[table.end() : end]
        body = re.sub(
            r"(?m)^[ \t]*(?:url|bearer_token_env_var|http_headers_helper)[ \t]*=.*\n?",
            "",
            body,
        )
        updated = existing[: table.end()] + "\n" + fields + body.lstrip("\n") + existing[end:]
    elif current:
        raise ValueError("Synapse config uses an unsupported table layout; left unchanged")
    else:
        updated = existing.rstrip() + "\n\n[mcp_servers.synapse]\n" + fields
    tomllib.loads(updated)
    if updated == existing:
        print("mcp: saved-credential helper already configured")
        return
    if dry_run:
        print("mcp: would configure the saved-credential helper (no token copied into config)")
        return
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(updated, encoding="utf-8")
    print("mcp: configured saved-credential helper")


def main() -> int:
    parser = argparse.ArgumentParser(description="Install Synapse into Codex CLI")
    parser.add_argument(
        "--synapse-url",
        default=None,
        help="Synapse base URL (default: existing MCP URL, then hook configuration)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not (HOOKS_DIR / "synapse_stop_hook.py").exists():
        print(f"error: hook scripts not found under {HOOKS_DIR}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(SCRIPTS_DIR))
    from common import BASE_URL

    config = tomllib.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
    url = (
        args.synapse_url or config.get("mcp_servers", {}).get("synapse", {}).get("url") or BASE_URL
    )
    install_mcp(url, args.dry_run)
    install_hook(args.dry_run)
    print(
        "\nDone. Next: start `codex`, run /hooks, and trust the Synapse Stop hook.\n"
        "MCP and hooks now share the saved Synapse device credential; no daemon\n"
        "environment export is needed. Requires Codex with http_headers_helper support."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
