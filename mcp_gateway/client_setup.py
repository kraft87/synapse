"""Reversible client bootstrap for the gateway (Claude Code and Codex).

    python -m mcp_gateway.client_setup snippets --gateway-url http://127.0.0.1:8766
    python -m mcp_gateway.client_setup install-bootstrap --client claude|codex [--dry-run]
    python -m mcp_gateway.client_setup remove-bootstrap  --client claude|codex [--dry-run]

``snippets`` only PRINTS client configuration; it never edits a client config. The
bootstrap commands write/remove one small local skill, ``synapse-gateway``, whose only job
is to make the client discover the user's published skills on the gateway
(``skill://<name>/SKILL.md``) and load the one that fits a task. It names no workflow,
tool or service of its own.

Ownership is by exact content: a folder counts as ours only if it holds nothing but a
SKILL.md byte-identical to a version this tool generated. Any user edit — even one that
keeps the marker line — makes install and remove leave the folder alone. Installing also
retires this tool's earlier pointer (``synapse-gateway-research``) under the same rule.

With Synapse's opt-in two-way skills sync on (``SYNAPSE_SKILLS_SYNC=1``) an installed
pointer is published like any local skill and reaches the user's other machines. It is
inert where the gateway is not connected (it says so and carries on).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path

STUB_NAME = "synapse-gateway"
MARKER = "<!-- installed-by: mcp_gateway.client_setup -->"
_REPO = Path(__file__).resolve().parents[1]

STUB = f"""---
name: {STUB_NAME}
description: Find and use the user's own published workflows through the Synapse MCP gateway. Use when a task may match a procedure, checklist or workflow the user keeps as a skill, or when the user refers to their skills, and the synapse-gateway MCP server is connected.
---
{MARKER}
# Skills via the Synapse gateway

The user's published skills are MCP resources on the `synapse-gateway` server.

1. List that server's resources (your client's MCP resource list, or its `list_resources`
   tool) and scan the `skill://<name>/SKILL.md` entries and their descriptions. Skip this
   discovery skill itself if it appears in the list.
2. If one fits the task, read its SKILL.md (resource reader, or `read_resource` with the
   URI) and follow it. Its other files are listed at `skill://<name>/_manifest`.
3. For tools supplied by the gateway, use its listed names and input schemas: Synapse memory
   tools keep their own names; other configured services use `<namespace>_<tool>`. A skill
   may also use local scripts, CLI tools, or connections outside this gateway.

Reading a skill's script does not run it; only a skill materialized into a local skills
folder can run scripts. If nothing fits, or the server is not connected or serves no skills
to this device, say so if relevant and carry on without it.
"""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


#: Every SKILL.md this tool has ever generated for the CURRENT name. When STUB changes, add
#: the old digest here so an untouched older install still upgrades/removes.
_GENERATED = frozenset(
    {_sha(STUB), "6a5f1313251c4e3c726f22d455a2414ad0ada0e2b89b5b9325cd32c1542cd81c"}
)

#: Earlier pointer names this tool generated, with the exact digests it wrote. Only a
#: byte-identical copy is ever removed; an edited one is the user's.
_LEGACY: dict[str, frozenset[str]] = {
    "synapse-gateway-research": frozenset(
        {"7ccbd41ad6f62013839d0057fe91ca47a58909964dfdbf916556ad7c96e6873e"}
    ),
}


def _default_dir(client: str) -> Path:
    if client == "claude":
        return Path(os.path.expanduser("~/.claude/skills"))
    # Same default and override as plugin-codex/scripts/skills_sync.py.
    return Path(os.path.expanduser(os.environ.get("SYNAPSE_CODEX_SKILLS_DIR", "~/.agents/skills")))


def install_bootstrap(skills_dir: Path, dry_run: bool) -> int:
    target = skills_dir / STUB_NAME / "SKILL.md"
    if target.parent.exists() and not _is_ours(target.parent, _GENERATED):
        print(
            f"bootstrap: {target.parent} exists and is not an unmodified copy from this "
            "tool; left unchanged"
        )
        return 1
    if target.exists() and target.read_text(encoding="utf-8") == STUB:
        print(f"bootstrap: already installed at {target}")
    elif dry_run:
        print(f"bootstrap: would write {target}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(STUB, encoding="utf-8")
        print(f"bootstrap: wrote {target}")
    _retire_legacy(skills_dir, dry_run)
    return 0


def remove_bootstrap(skills_dir: Path, dry_run: bool) -> int:
    rc = _remove(skills_dir / STUB_NAME, _GENERATED, dry_run)
    return max(rc, _retire_legacy(skills_dir, dry_run))


def _retire_legacy(skills_dir: Path, dry_run: bool) -> int:
    rc = 0
    for name, digests in _LEGACY.items():
        folder = skills_dir / name
        if folder.exists():
            rc = max(rc, _remove(folder, digests, dry_run))
    return rc


def _remove(folder: Path, digests: frozenset[str], dry_run: bool) -> int:
    if not folder.exists():
        print(f"bootstrap: nothing at {folder}")
        return 0
    if not _is_ours(folder, digests):
        print(f"bootstrap: {folder} is not an unmodified copy from this tool; left unchanged")
        return 1
    if dry_run:
        print(f"bootstrap: would remove {folder}")
        return 0
    (folder / "SKILL.md").unlink()
    folder.rmdir()
    print(f"bootstrap: removed {folder}")
    return 0


def _is_ours(folder: Path, digests: frozenset[str]) -> bool:
    md = folder / "SKILL.md"
    try:
        entries = {p.name for p in folder.iterdir()}
        return entries == {"SKILL.md"} and _sha(md.read_text(encoding="utf-8")) in digests
    except OSError:
        return False


#: Env vars that override the saved plugin credential in mcp_headers.py's resolution order.
TOKEN_OVERRIDES = ("SYNAPSE_INGEST_TOKEN", "CLAUDE_PLUGIN_OPTION_SYNAPSE_INGEST_TOKEN")


def snippets(gateway_url: str, saved_credential: bool = False) -> str:
    base = gateway_url.rstrip("/").removesuffix("/mcp")
    url = base + "/mcp"
    argv = [
        "python3",
        str(_REPO / "plugin-codex/scripts/mcp_headers.py"),
        "--url",
        url,
        "--gateway",
    ]
    if saved_credential:
        # A shell that exports some OTHER token (e.g. the root enrollment token for admin
        # tooling) would otherwise win over the saved device credential, and the gateway
        # rightly refuses root. Drop the overrides for the helper only.
        argv = ["env", *[a for v in TOKEN_OVERRIDES for a in ("-u", v)], *argv]
    helper = shlex.join(argv)
    claude_json = json.dumps({"type": "http", "url": url, "headersHelper": helper})
    return f"""\
# Credential: both clients reuse this machine's Synapse DEVICE token through
# plugin-codex/scripts/mcp_headers.py (env SYNAPSE_INGEST_TOKEN, else the Synapse Claude
# plugin's saved options). --gateway admits only a gateway on the Synapse host or loopback.
# If your shell exports a token that is NOT this device's (e.g. the root token), rerun with
# --saved-credential so the helper ignores env overrides and uses the saved device token.

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
## Optional: codex exec cannot prompt; pre-approve memory tools like a direct synapse entry:
## [mcp_servers.synapse-gateway.tools.recall]
## approval_mode = "approve"
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
    sn.add_argument(
        "--saved-credential",
        action="store_true",
        help="helper ignores SYNAPSE_INGEST_TOKEN env overrides (use the saved device token)",
    )
    for name in ("install-bootstrap", "remove-bootstrap"):
        p = sub.add_parser(name)
        p.add_argument("--client", choices=("claude", "codex"), required=True)
        p.add_argument("--skills-dir", type=Path, default=None)
        p.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.cmd == "snippets":
        sys.stdout.write(snippets(args.gateway_url, args.saved_credential))
        return 0
    skills_dir = args.skills_dir or _default_dir(args.client)
    if args.cmd == "install-bootstrap":
        return install_bootstrap(skills_dir, args.dry_run)
    return remove_bootstrap(skills_dir, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
