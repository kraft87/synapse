"""Assemble the gateway server: Synapse + optional research upstreams behind one /mcp.

Public surface (stable; clients and the research skill depend on it):

* ``synapse_<tool>`` — Synapse's own tools, called with the CALLER's device bearer.
* ``exa_<tool>`` / ``firecrawl_<tool>`` — research tools, called with the gateway's
  research keys, listed only to callers whose trust meets ``RESEARCH_TRUST``.
* ``list_resources`` / ``read_resource`` — FastMCP's ResourcesAsTools bridge for clients
  that cannot read MCP resources. It routes through this server's normal resource path,
  so a bridged read is authorized exactly like ``resources/read``.
* ``skill://…`` resources — Synapse's canonical PG skills, URIs unchanged, plus the
  gateway's bundled ``skill://gateway-research/…`` pilot skill.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.server.providers.skills import SkillsDirectoryProvider
from fastmcp.server.transforms import Namespace, ResourcesAsTools, Transform
from starlette.requests import Request
from starlette.responses import JSONResponse

from mcp_gateway.caller import (
    CLIENT_ID,
    SynapseCallerVerifier,
    current_bearer,
    current_token,
    identity_key,
    trust_allows,
)
from mcp_gateway.config import (
    SYNAPSE_NAMESPACE,
    GatewaySettings,
    install_log_redaction,
    load_settings,
    redact_url,
)
from mcp_gateway.upstream import GatedProvider, UpstreamProvider, http_client_factory

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(__file__).resolve().parent / "skills"
RESEARCH_SKILL = "gateway-research"


class ToolNamespace(Namespace):
    """Namespace tools and prompts only; resource URIs pass through untouched.

    ``skill://{name}/SKILL.md`` is the wire contract stock ``sync_skills`` and the Synapse
    plugins rely on. Prefixing it (``skill://synapse/...``) would fork that contract.
    """

    list_resources = Transform.list_resources
    get_resource = Transform.get_resource
    list_resource_templates = Transform.list_resource_templates
    get_resource_template = Transform.get_resource_template


def _authenticated() -> bool:
    tok = current_token()
    return tok is not None and tok.client_id == CLIENT_ID


def _instructions(settings: GatewaySettings) -> str:
    research = ", ".join(f"{u.name}_*" for u in settings.research) or "none configured"
    return (
        "Gateway to the user's Synapse memory plus web-research tools, over one connection. "
        "Memory tools are prefixed synapse_ (synapse_recall, synapse_fetch, synapse_remember, "
        "...): BEFORE answering anything about past work, decisions, devices, projects, people "
        "or preferences, call synapse_recall first; when the user states a durable fact, call "
        "synapse_remember. Research tools: " + research + " (only listed where this "
        "device may use them; names are whatever tools/list returns — never assume). "
        "Skills are MCP resources at skill://<name>/SKILL.md with a skill://<name>/_manifest "
        "file list. Read them with your client's MCP resource reader or the read_resource / "
        "list_resources tools. For web research, read skill://" + RESEARCH_SKILL + "/SKILL.md "
        "and follow it. Reading a skill's bundled script does not run it; scripts only run "
        "after the skill is materialized locally."
    )


def build_gateway(settings: GatewaySettings) -> FastMCP:
    # Part of building the server, not of main(): the mcp client library logs each keyed
    # endpoint URL at DEBUG, so no gateway instance may exist without the scrubber.
    install_log_redaction(settings.secrets())
    auth = SynapseCallerVerifier(
        settings.whoami_url,
        cache_ttl=settings.auth_cache_ttl,
        timeout=settings.whoami_timeout,
        base_url=settings.public_url or None,
    )
    mcp = FastMCP("synapse-gateway", instructions=_instructions(settings), auth=auth)

    synapse = UpstreamProvider(
        "synapse",
        http_client_factory(
            settings.synapse_mcp_url,
            # The caller's OWN bearer, resolved per request. Never a gateway credential.
            headers=lambda: {"Authorization": f"Bearer {current_bearer()}"},
            timeout=settings.synapse_call_timeout,
            init_timeout=settings.synapse_discovery_timeout,
        ),
        identity=lambda: identity_key(current_bearer()),
        allow=_authenticated,
        cache_ttl=settings.synapse_cache_ttl,
        discovery_timeout=settings.synapse_discovery_timeout,
        # Memory is the core service: retry on the next request rather than sit out.
        failure_backoff=0.0,
    )
    synapse.add_transform(ToolNamespace(SYNAPSE_NAMESPACE))
    mcp.add_provider(synapse)

    def research_allowed() -> bool:
        return trust_allows(settings.research_trust)

    for up in settings.research:
        provider = UpstreamProvider(
            up.name,
            http_client_factory(
                up.url(),
                headers=up.headers,
                timeout=up.call_timeout,
                init_timeout=up.discovery_timeout,
            ),
            # Research credentials are the gateway's, identical for every caller, so the
            # component listing is shared. Access is still decided per caller by `allow`.
            identity=lambda: "shared",
            allow=research_allowed,
            cache_ttl=settings.research_cache_ttl,
            discovery_timeout=up.discovery_timeout,
            failure_backoff=settings.failure_backoff,
        )
        provider.add_transform(Namespace(up.name))
        mcp.add_provider(provider)
        logger.info("research upstream %s -> %s", up.name, up.display_url)

    if settings.local_skills:
        mcp.add_provider(
            GatedProvider(SkillsDirectoryProvider(SKILLS_DIR), research_allowed, "local-skills")
        )

    mcp.add_transform(ResourcesAsTools(mcp))

    @mcp.custom_route("/health", methods=["GET"])
    async def health(request: Request) -> JSONResponse:
        """Unauthenticated liveness only: no upstream contact, no configuration details."""
        return JSONResponse({"status": "ok", "service": "synapse-gateway"})

    return mcp


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = load_settings()
    mcp = build_gateway(settings)
    logger.info(
        "synapse-gateway on %s:%d -> synapse %s (research: %s; research trust: %s)",
        settings.host,
        settings.port,
        redact_url(settings.synapse_url),
        ", ".join(u.name for u in settings.research) or "none",
        settings.research_trust,
    )
    mcp.run(
        transport="http",
        host=settings.host,
        port=settings.port,
        path="/mcp",
        stateless_http=True,
        show_banner=False,
    )
