"""Reversible client bootstrap for the gateway pilot (Claude Code and Codex).

    python -m mcp_gateway.client_setup snippets --gateway-url http://127.0.0.1:8766
    python -m mcp_gateway.client_setup install-bootstrap --client claude|codex [--dry-run]
    python -m mcp_gateway.client_setup remove-bootstrap  --client claude|codex [--dry-run]

``snippets`` only PRINTS client configuration; it never edits a client config. The
bootstrap commands write/remove one pointer skill (``synapse-gateway-research``) whose
only job is to make the client fetch the gateway-served workflow at
``skill://gateway-research/SKILL.md`` — so both clients follow the same, single copy.
``remove-bootstrap`` deletes only a folder this tool created (marker line + SKILL.md only).

With Synapse's opt-in two-way skills sync on (``SYNAPSE_SKILLS_SYNC=1``) an installed
pointer is published like any local skill and reaches the user's other machines. It is
inert where the gateway is not connected (it says so and stops).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

STUB_NAME = "synapse-gateway-research"
MARKER = "<!-- installed-by: mcp_gateway.client_setup -->"
_REPO = Path(__file__).resolve().parents[1]

STUB = f"""---
name: {STUB_NAME}
description: Web research through the Synapse MCP gateway. Use when the user asks to research, look up, compare, or verify something on the web and the synapse-gateway MCP server is connected.
---
{MARKER}
# Research via the Synapse gateway

This skill is a pointer; the workflow itself is served by the gateway so every client
follows the same copy.

1. Read the MCP resource `skill://gateway-research/SKILL.md` from the `synapse-gateway`
   server — with your client's MCP resource reader, or by calling that server's
   `read_resource` tool with the URI.
2. Follow it.

If the synapse-gateway server is not connected, or the resource is not available to this
device, say so and stop; do not improvise a substitute workflow.
"""


def _default_dir(client: str) -> Path:
    if client == "claude":
        return Path(os.path.expanduser("~/.claude/skills"))
    # Same default and override as plugin-codex/scripts/skills_sync.py.
    return Path(os.path.expanduser(os.environ.get("SYNAPSE_CODEX_SKILLS_DIR", "~/.agents/skills")))


def install_bootstrap(skills_dir: Path, dry_run: bool) -> int:
    target = skills_dir / STUB_NAME / "SKILL.md"
    if target.parent.exists():
        if _is_ours(target.parent):
            if target.read_text(encoding="utf-8") == STUB:
                print(f"bootstrap: already installed at {target}")
                return 0
        else:
            print(f"bootstrap: {target.parent} exists and was not created here; left unchanged")
            return 1
    if dry_run:
        print(f"bootstrap: would write {target}")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(STUB, encoding="utf-8")
    print(f"bootstrap: wrote {target}")
    return 0


def remove_bootstrap(skills_dir: Path, dry_run: bool) -> int:
    folder = skills_dir / STUB_NAME
    if not folder.exists():
        print(f"bootstrap: nothing at {folder}")
        return 0
    if not _is_ours(folder):
        print(f"bootstrap: {folder} was not created by this tool (or was edited); left unchanged")
        return 1
    if dry_run:
        print(f"bootstrap: would remove {folder}")
        return 0
    (folder / "SKILL.md").unlink()
    folder.rmdir()
    print(f"bootstrap: removed {folder}")
    return 0


def _is_ours(folder: Path) -> bool:
    md = folder / "SKILL.md"
    try:
        entries = {p.name for p in folder.iterdir()}
        return entries == {"SKILL.md"} and MARKER in md.read_text(encoding="utf-8")
    except OSError:
        return False


def snippets(gateway_url: str) -> str:
    base = gateway_url.rstrip("/").removesuffix("/mcp")
    url = base + "/mcp"
    helper = shlex.join(
        ["python3", str(_REPO / "plugin-codex/scripts/mcp_headers.py"), "--url", url, "--gateway"]
    )
    claude_json = json.dumps({"type": "http", "url": url, "headersHelper": helper})
    return f"""\
# Credential: both clients reuse this machine's Synapse DEVICE token through
# plugin-codex/scripts/mcp_headers.py (env SYNAPSE_INGEST_TOKEN, else the Synapse Claude
# plugin's saved options). --gateway admits only a gateway on the Synapse host or loopback.

## Claude Code — add (user scope)
claude mcp add-json --scope user synapse-gateway {shlex.quote(claude_json)}
## Claude Code — alternative without headersHelper: the header keeps a literal
## ${{SYNAPSE_INGEST_TOKEN}} that Claude Code expands from its environment at load time
## (verify on the installed Claude Code version; never paste the token itself here)
claude mcp add --scope user --transport http synapse-gateway {url} \\
  --header 'Authorization: Bearer ${{SYNAPSE_INGEST_TOKEN}}'
## Claude Code — remove
claude mcp remove --scope user synapse-gateway

## Codex — add to ~/.codex/config.toml
[mcp_servers.synapse-gateway]
url = "{url}"
http_headers_helper = {_toml_str(helper)}
## Codex — remove: delete the [mcp_servers.synapse-gateway] table above.

## Bootstrap pointer skill (both clients; reversible)
python -m mcp_gateway.client_setup install-bootstrap --client claude
python -m mcp_gateway.client_setup install-bootstrap --client codex
python -m mcp_gateway.client_setup remove-bootstrap --client claude   # / --client codex
"""


def _toml_str(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sn = sub.add_parser("snippets", help="print client configuration (no changes made)")
    sn.add_argument("--gateway-url", default="http://127.0.0.1:8766")
    for name in ("install-bootstrap", "remove-bootstrap"):
        p = sub.add_parser(name)
        p.add_argument("--client", choices=("claude", "codex"), required=True)
        p.add_argument("--skills-dir", type=Path, default=None)
        p.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.cmd == "snippets":
        sys.stdout.write(snippets(args.gateway_url))
        return 0
    skills_dir = args.skills_dir or _default_dir(args.client)
    if args.cmd == "install-bootstrap":
        return install_bootstrap(skills_dir, args.dry_run)
    return remove_bootstrap(skills_dir, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
