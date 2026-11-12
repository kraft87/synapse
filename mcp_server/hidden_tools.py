"""Tools that stay callable but are kept out of tools/list, and arguments a client may
still send after a tool dropped them."""

from __future__ import annotations

import logging

from fastmcp.server.middleware import Middleware

logger = logging.getLogger(__name__)

# Tools that stay fully callable via tools/call but never appear in tools/list:
# plumbing that a specific client invokes by name (synapse_login's raw tools/call
# on issue_machine_token) and that a model browsing the tool list must not see.
HIDDEN_TOOLS = {"issue_machine_token"}

# Arguments a tool no longer declares but an older client, or a tool schema it cached,
# may still send. FastMCP rejects an unknown keyword, so each is dropped before validation.
RETIRED_ARGS = {"recall": frozenset({"group_id"})}


class HiddenToolsList(Middleware):
    """Filter hidden tools out of tools/list responses.

    Listing and calling are separate request paths in FastMCP, so dropping a tool
    here leaves tools/call untouched — `synapse login` keeps working while the
    model-facing surface stays the deliberate tools registered in mcp_server/server.py."""

    async def on_list_tools(self, context, call_next):
        tools = await call_next(context)
        return [t for t in tools if t.name not in HIDDEN_TOOLS]


class RetiredToolArgs(Middleware):
    """Accept and ignore an argument a tool has retired.

    The argument is not in the tool's schema, so tools/list never shows it; a client that
    still sends it is served as if it had not, and each dropped argument logs one INFO line."""

    async def on_call_tool(self, context, call_next):
        message = context.message
        sent = RETIRED_ARGS.get(message.name, frozenset()).intersection(message.arguments or ())
        if sent:
            for arg in sorted(sent):
                logger.info("retired arg %s ignored on %s", arg, message.name)
            kept = {k: v for k, v in (message.arguments or {}).items() if k not in sent}
            context = context.copy(message=message.model_copy(update={"arguments": kept}))
        return await call_next(context)
