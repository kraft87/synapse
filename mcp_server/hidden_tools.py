"""Tools that stay callable but are kept out of tools/list."""

from __future__ import annotations

from fastmcp.server.middleware import Middleware

# Tools that stay fully callable via tools/call but never appear in tools/list:
# plumbing that a specific client invokes by name (synapse_login's raw tools/call
# on issue_machine_token) and that a model browsing the tool list must not see.
HIDDEN_TOOLS = {"issue_machine_token"}


class HiddenToolsList(Middleware):
    """Filter hidden tools out of tools/list responses.

    Listing and calling are separate request paths in FastMCP, so dropping a tool
    here leaves tools/call untouched — `synapse login` keeps working while the
    model-facing surface stays the deliberate tools registered in mcp_server/server.py."""

    async def on_list_tools(self, context, call_next):
        tools = await call_next(context)
        return [t for t in tools if t.name not in HIDDEN_TOOLS]
