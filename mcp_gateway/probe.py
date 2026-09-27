"""Read-only smoke check of a running gateway, as one device sees it.

    SYNAPSE_INGEST_TOKEN=<device token> python -m mcp_gateway.probe --url http://127.0.0.1:8766/mcp

The token is read from the environment (never an argument, so it stays out of shell
history and process listings) and never printed. Default checks are free and read-only:
tools/list, resources/list, and reading the research skill through both the resource path
and the read_resource bridge. ``--search QUERY`` additionally makes ONE real research call
(it spends research credits and sends QUERY to the third-party provider).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from mcp_gateway.app import RESEARCH_SKILL


async def _probe(url: str, token: str, search: str | None) -> int:
    uri = f"skill://{RESEARCH_SKILL}/SKILL.md"
    async with Client(StreamableHttpTransport(url, auth=token), timeout=60) as c:
        listed = {t.name: t for t in await c.list_tools()}
        tools = sorted(listed)
        groups: dict[str, list[str]] = {}
        for name in tools:
            groups.setdefault(name.split("_", 1)[0], []).append(name)
        for prefix, names in sorted(groups.items()):
            print(f"tools[{prefix}]: {', '.join(names)}")
        skills = sorted(
            str(r.uri) for r in await c.list_resources() if str(r.uri).endswith("/SKILL.md")
        )
        print(f"skill resources: {len(skills)}")
        if uri not in skills:
            print(f"{uri}: not served to this device (research not permitted, or disabled)")
        else:
            direct = (await c.read_resource(uri))[0]
            bridged = await c.call_tool("read_resource", {"uri": uri})
            same = getattr(direct, "text", None) == bridged.content[0].text
            print(f"{uri}: readable; resources/read == read_resource tool: {same}")
        if search:
            candidates = [
                t
                for t in tools
                if t.startswith("exa_")
                and "search" in t
                and "query" in (listed[t].inputSchema.get("properties") or {})
            ]
            if not candidates:
                print("search: no exa_ search tool taking `query` is listed for this device")
                return 1
            args: dict[str, Any] = {"query": search}
            result = await c.call_tool(candidates[0], args, raise_on_error=False)
            text = result.content[0].text if result.content else ""
            status = "error" if result.is_error else "ok"
            print(f"search via {candidates[0]}: {status}, {len(text)} chars")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-check a Synapse MCP gateway")
    parser.add_argument("--url", default="http://127.0.0.1:8766/mcp")
    parser.add_argument("--token-env", default="SYNAPSE_INGEST_TOKEN")
    parser.add_argument("--search", default=None, help="make one real research call")
    args = parser.parse_args(argv)
    token = os.environ.get(args.token_env, "")
    if not token:
        print(f"set {args.token_env} to this device's Synapse token", file=sys.stderr)
        return 2
    return asyncio.run(_probe(args.url, token, args.search))


if __name__ == "__main__":
    raise SystemExit(main())
