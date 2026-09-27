"""Assemble the gateway: Synapse memory + skills, plus the deployment's own MCP upstreams.

Public surface (stable; clients and the Synapse plugins' hooks depend on it):

* Synapse's own tools under their OWN names (``recall``, ``fetch``, ``remember``, …),
  called with the CALLER's device bearer. Unprefixed on purpose: the plugins' hooks match
  ``mcp__<any server>__recall`` etc., so the gateway must not rename them.
* ``<namespace>_<tool>`` for each upstream in the deployment's registry
  (:mod:`mcp_gateway.registry`), called with that upstream's own configured credential and
  listed only to callers meeting its ``min_trust``. None by default.
* ``list_resources`` / ``read_resource`` — FastMCP's ResourcesAsTools bridge for clients
  that cannot read MCP resources. It routes through this server's normal resource path,
  so a bridged read is authorized exactly like ``resources/read``.
* ``skill://…`` resources — Synapse's canonical skills (URIs unchanged; callers meeting
  ``SKILLS_TRUST`` only), plus any skill directories the registry adds.
"""

from __future__ import annotations

import logging
from functools import partial

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
from mcp_gateway.config import GatewaySettings, install_log_redaction, load_settings, redact_url
from mcp_gateway.registry import Upstream
from mcp_gateway.upstream import GatedProvider, UpstreamProvider, http_client_factory

logger = logging.getLogger(__name__)

#: Claude Code keeps at most this much of a server's instructions.
INSTRUCTIONS_CAP = 2048


def _authenticated() -> bool:
    tok = current_token()
    return tok is not None and tok.client_id == CLIENT_ID


def _instructions(settings: GatewaySettings) -> str:
    memory = (
        "One connection to the user's Synapse memory, their published skills, and any MCP "
        "services this deployment adds. Memory: BEFORE answering anything that references "
        "past work — a prior decision, device, project, person or preference — call recall "
        "first; fetch expands ids from its results; when the user states a durable fact or "
        "correction, call remember. "
    )
    skills = (
        "Skills: the user's published workflows are MCP resources at skill://<name>/SKILL.md "
        "(file list: skill://<name>/_manifest). When a task may match one, list resources "
        "(your client's resource list, or the list_resources tool), read the SKILL.md whose "
        "description fits, and follow it; read_resource reads any listed URI. Reading a "
        "bundled script does not run it; scripts run only from a skill materialized locally."
    )
    ups = settings.registry.upstreams
    if not ups:
        return memory + skills

    # Initialization instructions are shared by all callers. Service names and operator
    # descriptions can be private, so discover them only through the authorized listing.
    services = (
        "Additional MCP tools use <namespace>_<tool> names. Discover the tools this device "
        "may access through tools/list, and use their listed input schemas. "
    )
    return memory + services + skills


def _upstream_provider(up: Upstream) -> UpstreamProvider:
    provider = UpstreamProvider(
        up.namespace,
        http_client_factory(
            up.url(),
            headers=up.headers,
            timeout=up.call_timeout,
            init_timeout=up.discovery_timeout,
        ),
        # The upstream's credential is the deployment's, identical for every caller, so its
        # component listing is shared. Access is still decided per caller by `allow`.
        identity=lambda: "shared",
        allow=lambda kind: trust_allows(up.min_trust),
        cache_ttl=up.cache_ttl,
        discovery_timeout=up.discovery_timeout,
        failure_backoff=up.failure_backoff,
    )
    provider.add_transform(Namespace(up.namespace))
    return provider


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

    for up in settings.registry.upstreams:
        mcp.add_provider(_upstream_provider(up))
        logger.info(
            "upstream %s -> %s (auth %s, min trust %s)",
            up.namespace,
            up.display_url,
            up.auth_type,
            up.min_trust,
        )

    for i, skills_dir in enumerate(settings.registry.skills_dirs):
        mcp.add_provider(
            GatedProvider(
                SkillsDirectoryProvider(skills_dir.path),
                partial(trust_allows, skills_dir.min_trust),
                f"skills-dir-{i}",
            )
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
        "synapse-gateway on %s:%d -> synapse %s (upstreams: %s; skills dirs: %d; skills trust: %s)",
        settings.host,
        settings.port,
        redact_url(settings.synapse_url),
        ", ".join(u.namespace for u in settings.registry.upstreams) or "none",
        len(settings.registry.skills_dirs),
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
