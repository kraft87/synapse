#!/usr/bin/env python3
"""Regenerate plugin/synapse_client/tool_schemas.json from the MCP server's tools.

The client core (plugin/synapse_client/tools.py) hands a harness the server's tool
definitions from this snapshot, with no round trip and no MCP library on the client.
tests/test_client_core_tools.py fails when the snapshot and the server disagree; this
brings it back in line:

    uv run python scripts/gen_client_tool_schemas.py [--out PATH]

It needs no database and sends nothing anywhere. Before the server is imported:

- every variable the server reads at import is dropped from the calling shell's
  environment (SYNAPSE_*, LOGFIRE_*, the GitHub and OIDC login settings, VOYAGE_*), so
  the snapshot does not depend on who ran it and no login provider is configured;
- telemetry is off. The server passes ``send_to_logfire="if-token-present"`` itself, so
  LOGFIRE_SEND_TO_LOGFIRE alone would not do it: what does is that no token is left,
  neither LOGFIRE_TOKEN nor a ``.logfire/`` credentials folder, since the import runs
  with an empty temporary directory as its working directory;
- the server's .env fallback is pointed at a file that does not exist.

The snapshot keeps the server's tool order and, per tool, only what tools/list shows a
client: name, description and inputSchema. Tools hidden from tools/list are left out.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = _REPO / "plugin" / "synapse_client" / "tool_schemas.json"
_FIELDS = ("name", "description", "inputSchema")
#: Prefixes of the variables mcp_server/server.py (and what it imports) reads at import.
_DROP = ("SYNAPSE_", "LOGFIRE_", "GITHUB_", "OIDC_", "ALLOWED_", "VOYAGE_", "MCP_")


def _quiet_environment(env_file: str) -> None:
    for name in list(os.environ):
        if name.startswith(_DROP):
            del os.environ[name]
    os.environ["LOGFIRE_SEND_TO_LOGFIRE"] = "false"
    os.environ["LOGFIRE_CONSOLE"] = "false"
    os.environ["SYNAPSE_ENV_FILE"] = env_file
    os.environ["SYNAPSE_DB_URL"] = ""


def snapshot() -> dict[str, Any]:
    """The tools a client is shown, in the server's order. Call after the environment
    is quiet: importing the server configures telemetry and reads its .env."""
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    from mcp_server import server
    from mcp_server.hidden_tools import HIDDEN_TOOLS

    tools = []
    for tool in asyncio.run(server.mcp.list_tools()):
        if tool.name in HIDDEN_TOOLS:
            continue
        data = tool.to_mcp_tool().model_dump(exclude_none=True, by_alias=True)
        tools.append({key: data[key] for key in _FIELDS if key in data})
    return {"generated_by": "scripts/gen_client_tool_schemas.py", "tools": tools}


def render(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="where to write")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    here = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        _quiet_environment(os.path.join(tmp, "absent.env"))
        os.chdir(tmp)  # no .logfire/ credentials folder where the server looks for one
        try:
            data = snapshot()
        finally:
            os.chdir(here)
    out.write_text(render(data), encoding="utf-8", newline="\n")
    print(f"{len(data['tools'])} tools -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
