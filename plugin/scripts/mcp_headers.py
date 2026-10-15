#!/usr/bin/env python3
# mypy: ignore-errors
"""MCP auth headers from this machine's saved Synapse credential: the headers helper
for both harnesses.

Claude Code runs it from plugin.json (`headersHelper`, cwd = the plugin root), Codex
from config.toml (`http_headers_helper`, with `--url`). Neither hands it the user's
environment: Claude Code strips every variable whose name contains TOKEN, KEY, SECRET or
AUTH, and Codex clears the environment down to HOME, PATH and a few others. So the
credential comes from the saved config (~/.synapse/config.json, then the legacy Claude
plugin options), resolved exactly as the hooks resolve it (config.py), and an exported
SYNAPSE_INGEST_TOKEN is never a source here.

Stdout is a credential channel the harness consumes, not a diagnostic log: exactly one
JSON object, ``{"Authorization": "Bearer ..."}``, with that casing (Claude Code merges
the helper's headers over its static ones by exact key, so a lowercase key would be sent
alongside the static header rather than replace it). On any failure: nothing on stdout,
exit 1. Claude Code then sends its static header and Codex none. Codex discards stderr,
so a failure is also appended to ~/.synapse/logs/mcp_headers.log (the reason only, never
a token). No network I/O: both harnesses kill the helper after 10 s.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from urllib.parse import urlsplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# The user's own settings only: Codex runs this in the session's directory, and a
# repository's .claude/settings.json must not choose the server or the token.
os.environ["SYNAPSE_USER_SETTINGS_ONLY"] = "1"
import config


def _log(reason: str) -> None:
    config.synapse_home.log("mcp_headers", reason)


def _fail(reason: str) -> int:
    print(reason, file=sys.stderr)
    _log(reason)
    return 1


def _target(arg: str | None) -> str:
    """The URL the harness is about to connect to. Codex passes it as --url. Claude Code
    sets CLAUDE_CODE_MCP_SERVER_URL, but censors any substring equal to the value of a
    credential-named variable in it ("REDACTED"); then the plugin's own SYNAPSE_URL,
    which Claude Code expands into that URL, stands in for it."""
    url = arg or os.environ.get("CLAUDE_CODE_MCP_SERVER_URL") or ""
    if "REDACTED" in url:
        url = str(config._FILE_OPTIONS.get("SYNAPSE_URL") or "")
    return url


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synapse MCP credential helper")
    parser.add_argument("--url", help="the MCP URL being connected to (Codex passes it)")
    args = parser.parse_args(argv)
    target = urlsplit(_target(args.url))
    configured = urlsplit(config.BASE_URL)
    if not target.netloc:
        return _fail("Synapse MCP helper: no target URL (pass --url)")
    if (target.scheme.lower(), target.netloc.lower()) != (
        configured.scheme.lower(),
        configured.netloc.lower(),
    ):
        return _fail(
            f"Synapse MCP URL {target.scheme}://{target.netloc} differs from the saved "
            f"server {configured.scheme}://{configured.netloc}; `synapse install` points "
            "the harnesses at the saved one"
        )
    # Whether the device token may go to this server is config's decision, the same one
    # the hooks make: only to the server that minted it (config._device_origins).
    override = getattr(config, "TOKEN_OVERRIDE", None) or {}
    if override.get("withheld"):
        return _fail(config.withheld_notice(override["withheld"], override.get("server") or ""))
    if not config.INGEST_TOKEN:
        return _fail("Synapse device credential missing: run `synapse login`")
    print(json.dumps({"Authorization": f"Bearer {config.INGEST_TOKEN}"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
