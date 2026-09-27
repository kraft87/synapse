"""Read-only structural smoke check of a running gateway, as one device sees it.

    SYNAPSE_INGEST_TOKEN=<device token> python -m mcp_gateway.probe --url http://127.0.0.1:8766/mcp
        [--expect-namespace NS ...] [--call TOOL --args '{"key": "value"}']

The token is read from the environment (never an argument, so it stays out of shell
history and process listings) and never printed. By default the probe only lists: the core
memory and bridge tools, tools grouped by namespace, and skill resources, and it reads one
listed skill both natively and through the read_resource bridge to check they agree.
``--expect-namespace`` fails the probe unless tools with that prefix are listed.
``--call`` makes exactly the one call you specify, with exactly your arguments, and prints
only whether it errored and the result size. Nothing is guessed about what a tool does.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

#: Always present for an approved device: Synapse's memory tools plus the resources bridge.
CORE_TOOLS = frozenset({"recall", "fetch", "remember", "list_resources", "read_resource"})


async def _probe(
    url: str,
    token: str,
    expect: list[str],
    call: tuple[str, dict[str, Any]] | None,
    skill: str | None = None,
) -> int:
    rc = 0
    async with Client(StreamableHttpTransport(url, auth=token), timeout=60) as c:
        names = sorted(t.name for t in await c.list_tools())
        missing = sorted(CORE_TOOLS - set(names))
        print(f"core tools: {'ok' if not missing else 'MISSING ' + ', '.join(missing)}")
        rc |= bool(missing)
        others = [n for n in names if n not in CORE_TOOLS]
        groups: dict[str, list[str]] = {}
        for name in others:
            groups.setdefault(name.split("_", 1)[0] if "_" in name else "", []).append(name)
        for prefix, members in sorted(groups.items()):
            print(f"tools[{prefix or '-'}]: {len(members)}")
        for ns in expect:
            present = any(n.startswith(f"{ns}_") for n in names)
            print(f"namespace {ns}: {'ok' if present else 'NOT LISTED'}")
            rc |= not present

        skills = sorted(
            str(r.uri)
            for r in await c.list_resources()
            if str(r.uri).startswith("skill://") and str(r.uri).endswith("/SKILL.md")
        )
        print(f"skill resources: {len(skills)}")
        target = skill or (skills[0] if skills else None)
        if target:
            direct = (await c.read_resource(target))[0]
            bridged = await c.call_tool("read_resource", {"uri": target}, raise_on_error=False)
            same = not bridged.is_error and getattr(direct, "text", None) == bridged.content[0].text
            print(f"skill read (native == read_resource tool): {same}")
            rc |= not same

        if call:
            tool, args = call
            result = await c.call_tool(tool, args, raise_on_error=False)
            size = sum(len(getattr(x, "text", "") or "") for x in (result.content or []))
            print(f"call {tool}: {'error' if result.is_error else 'ok'}, {size} chars")
            rc |= bool(result.is_error)
    return int(rc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-check a Synapse MCP gateway")
    parser.add_argument("--url", default="http://127.0.0.1:8766/mcp")
    parser.add_argument("--token-env", default="SYNAPSE_INGEST_TOKEN")
    parser.add_argument("--expect-namespace", action="append", default=[], metavar="NS")
    parser.add_argument("--skill", default=None, help="skill:// URI to read (default: first)")
    parser.add_argument("--call", default=None, metavar="TOOL", help="make this one call")
    parser.add_argument("--args", default="{}", help="JSON object of arguments for --call")
    args = parser.parse_args(argv)
    token = os.environ.get(args.token_env, "")
    if not token:
        print(f"set {args.token_env} to this device's Synapse token", file=sys.stderr)
        return 2
    call = None
    if args.call:
        try:
            call_args = json.loads(args.args)
        except json.JSONDecodeError:
            call_args = None
        if not isinstance(call_args, dict):
            print("--args must be a JSON object", file=sys.stderr)
            return 2
        call = (args.call, call_args)
    try:
        return asyncio.run(_probe(args.url, token, args.expect_namespace, call, args.skill))
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
