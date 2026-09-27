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

RESEARCH_PREFIXES = ("exa_", "firecrawl_")


def _search_call(listed: dict[str, Any], text: str) -> tuple[str, dict[str, Any]] | None:
    """First research search tool whose required arguments are all strings, with every
    required argument filled from ``text`` (e.g. Exa wants ``query`` AND ``objective``)."""
    for name in sorted(listed, key=lambda n: (not n.startswith("exa_"), n)):
        if not name.startswith(RESEARCH_PREFIXES) or "search" not in name:
            continue
        schema = listed[name].inputSchema or {}
        props = schema.get("properties") or {}
        required = schema.get("required") or []
        if "query" not in props or any(props.get(r, {}).get("type") != "string" for r in required):
            continue
        return name, {"query": text, **{r: text for r in required}}
    return None


async def _probe(url: str, token: str, search: str | None) -> int:
    uri = f"skill://{RESEARCH_SKILL}/SKILL.md"
    async with Client(StreamableHttpTransport(url, auth=token), timeout=60) as c:
        listed = {t.name: t for t in await c.list_tools()}
        groups: dict[str, list[str]] = {}
        for name in sorted(listed):
            group = next((p[:-1] for p in RESEARCH_PREFIXES if name.startswith(p)), "memory+bridge")
            groups.setdefault(group, []).append(name)
        for group, names in sorted(groups.items()):
            print(f"tools[{group}]: {', '.join(names)}")
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
            call = _search_call(listed, search)
            if call is None:
                print("search: no research search tool with string-only required args is listed")
                return 1
            name, args = call
            result = await c.call_tool(name, args, raise_on_error=False)
            text = result.content[0].text if result.content else ""
            status = "error" if result.is_error else "ok"
            print(
                f"search via {name} (args: {', '.join(sorted(args))}): {status}, {len(text)} chars"
            )
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
    try:
        return asyncio.run(_probe(args.url, token, args.search))
    except Exception as e:
        if "401" in str(e):
            print(
                f"gateway refused the credential in {args.token_env} (401): only approved "
                "Synapse DEVICE tokens are admitted; the root/enrollment token and pending or "
                "revoked devices are refused by design",
                file=sys.stderr,
            )
            return 3
        raise


if __name__ == "__main__":
    raise SystemExit(main())
