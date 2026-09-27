"""Assemble the gateway server: Synapse + optional research upstreams behind one /mcp.

Public surface (stable; clients, hooks and the research skill depend on it):

* Synapse's own tools under their OWN names (``recall``, ``fetch``, ``remember``, …),
  called with the CALLER's device bearer. Unprefixed on purpose: the Synapse plugins' hooks
  match ``mcp__<any server>__recall`` etc., so the gateway must not rename them.
* ``exa_<tool>`` / ``firecrawl_<tool>`` — research tools, called with the gateway's
  research configuration, listed only to callers whose trust meets ``RESEARCH_TRUST``.
* ``list_resources`` / ``read_resource`` — FastMCP's ResourcesAsTools bridge for clients
  that cannot read MCP resources. It routes through this server's normal resource path,
  so a bridged read is authorized exactly like ``resources/read``.
* ``skill://…`` resources — Synapse's canonical PG skills (URIs unchanged; callers meeting
  ``SKILLS_TRUST`` only), plus the bundled ``skill://gateway-research/…`` pilot skill.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.server.providers.skills import SkillsDirectoryProvider
from fastmcp.server.transforms import Namespace, ResourcesAsTools
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
    GatewaySettings,
    install_log_redaction,
    load_settings,
    redact_url,
)
from mcp_gateway.upstream import GatedProvider, UpstreamProvider, http_client_factory

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(__file__).resolve().parent / "skills"
RESEARCH_SKILL = "gateway-research"


def _authenticated() -> bool:
    tok = current_token()
    return tok is not None and tok.client_id == CLIENT_ID


def _instructions(settings: GatewaySettings) -> str:
    research = ", ".join(f"{u.name}_*" for u in settings.research) or "none configured"
    return (
        "Gateway to the user's Synapse memory plus web-research tools, over one connection. "
        "Memory: BEFORE answering anything that references past work — a prior decision, "
        "device, project, person or preference — call recall first; fetch expands ids from "
        "its results; when the user states a durable fact or correction, call remember. "
        "Research tools: " + research + " (listed only where this device may use them; use "
        "the names and input schemas tools/list returns). Skills are MCP resources at "
        "skill://<name>/SKILL.md (file list: skill://<name>/_manifest); read them with your "
        "client's MCP resource reader or the read_resource / list_resources tools. For web "
        "research read skill://" + RESEARCH_SKILL + "/SKILL.md. Reading a bundled script "
        "does not run it; scripts run only from a skill materialized locally."
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

    def research_allowed() -> bool:
        return trust_allows(settings.research_trust)

    def synapse_allowed(kind: str) -> bool:
        # Memory tools: every approved device (Synapse scopes the data per bearer).
        # Skill resources: Synapse's provider serves every active skill to any bearer, so
        # withhold the class unless the caller meets SKILLS_TRUST (fail closed).
        if kind in ("resources", "templates"):
            return trust_allows(settings.skills_trust)
        return _authenticated()

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
        allow=synapse_allowed,
        cache_ttl=settings.synapse_cache_ttl,
        discovery_timeout=settings.synapse_discovery_timeout,
        # Memory is the core service: retry on the next request rather than sit out.
        failure_backoff=0.0,
    )
    mcp.add_provider(synapse)

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
            allow=lambda kind: research_allowed(),
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
        "synapse-gateway on %s:%d -> synapse %s (research: %s; research trust: %s; "
        "skills trust: %s)",
        settings.host,
        settings.port,
        redact_url(settings.synapse_url),
        ", ".join(u.name for u in settings.research) or "none",
        settings.research_trust,
        settings.skills_trust,
    )
    mcp.run(
        transport="http",
        host=settings.host,
        port=settings.port,
        path="/mcp",
        stateless_http=True,
        show_banner=False,
    )
