"""Register the auxiliary HTTP lanes in their established startup order."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastmcp import FastMCP
from starlette.requests import Request


def register(
    mcp: FastMCP,
    DB_URL: str,
    VOYAGE_API_KEY: str,
    MACHINE_TOKEN: str,
    PUBLIC_URL: str,
    _idp: Any,
    _machine_authorized: Callable[[Request], bool],
    _admin_authorized: Callable[[Request], bool],
    _get_recall: Callable[[], Any],
    _request_trust: Callable[..., Any],
    _issue_dash_token: Callable[[str], str],
) -> None:
    # Skill sync + review over plain HTTP — lets the Claude Code plugin stay DSN-free (it talks to
    # these machine-token-gated routes instead of reaching Postgres directly). No-op without DB_URL.
    from mcp_server.skill_sync_routes import register as _register_skill_routes

    _register_skill_routes(mcp, DB_URL, _machine_authorized)

    # Config lane — the plugin mirrors each surface's opted-in config files here (machine-token gated)
    # so the dream pipeline can read them and propose edits. Same DSN-free seam as skills. No-op w/o DB.
    from mcp_server.config_sync_routes import register as _register_config_routes

    _register_config_routes(mcp, DB_URL, _machine_authorized)

    # Timeline event ingest — feeders (the plugin's git feeder, later calendar) POST plain event
    # rows here; the server embeds + upserts. Same DSN-free machine-token seam. No-op w/o DB.
    from mcp_server.timeline_routes import register as _register_timeline_routes

    _register_timeline_routes(mcp, DB_URL, _machine_authorized, VOYAGE_API_KEY)

    # Preferences read route — the plugin's SessionStart block GETs the top standing user
    # preferences here (schema 035). Same machine-token seam; server owns the DB. No-op w/o DB.
    from mcp_server.preferences_routes import register as _register_preferences_routes

    _register_preferences_routes(mcp, DB_URL, _machine_authorized)

    # Private mode — the plugin's toggle CLI flips a session's "off the record" flag here
    # (schema 050). The flag is what ingest_turns/backfill check, so a session marked private
    # can never be ingested by ANY path, including one that bypasses the plugin's hook.
    # Same machine-token seam. No-op w/o DB.
    from mcp_server.private_session_routes import register as _register_private_routes

    _register_private_routes(mcp, DB_URL, _machine_authorized)

    # Surface enrollment + registration — audience scoping's operator seam (schema 053/054).
    # Which DEVICES get the full corpus and which get only work-safe notes + an allowlisted
    # set of projects. Enrolling is anchored to an allowlisted OAuth/OIDC identity (the same
    # device flow `synapse login` uses), NOT to the shared machine token; minting, listing
    # and revoking demand the admin gate, which the root token also cannot clear. Registered
    # AFTER _idp is built, since enrollment needs it. Doing nothing is already the safe state.
    from mcp_server.surface_routes import register as _register_surface_routes

    _register_surface_routes(mcp, DB_URL, _machine_authorized, _admin_authorized, idp=_idp)

    # Board read route — GET /context?project=X serves the rendered explicit-memory board
    # for the plugin's SessionStart hook (the ONLY serve path — see the Tools comment).
    # Same machine-token seam. No-op w/o DB. The engine callback is resolved lazily
    # at request time (telemetry shares its writer).
    from mcp_server.board import register as _register_board_routes

    _register_board_routes(
        mcp,
        DB_URL,
        _machine_authorized,
        get_recall=lambda: _get_recall(),
        resolve_trust=_request_trust,
    )

    # Operator dashboard — static React bundle at /dash + read/flag API at /dash/api/* (issue #12,
    # contract docs/dashboard-contract.md). Static routes are unauthenticated (public bundle, no
    # data); every api route rides the ADMIN gate, not the client one. That move is required, not
    # cosmetic: /dash/api serves the whole corpus unfiltered and can flag/edit, so leaving it on the
    # root token would let anything holding the enrollment credential read everything the TOFU gate
    # was meant to withhold. No-op w/o DB_URL, like the siblings.
    from mcp_server.dashboard_routes import register as _register_dashboard_routes

    _register_dashboard_routes(mcp, DB_URL, _admin_authorized)

    # Device-login lane — RFC 8628 device flow so `synapse login` works browser-free on servers /
    # headless boxes. Proxies the configured IdP's device flow (GitHub or OIDC) and gates the
    # machine token by the same allowlist as the web leg. No-op without an IdP + machine token.
    from mcp_server.device_routes import register as _register_device_routes

    _register_device_routes(mcp, _idp, MACHINE_TOKEN)

    # Browser-login lane — authorization-code flow for the dashboard login screen (redirect UX;
    # the device flow stays for `synapse login`). Same IdP identity + allowlist gate; return
    # origins restricted via SYNAPSE_DASH_ORIGINS. Same enablement condition as the device flow.
    from mcp_server.web_login_routes import register as _register_web_login_routes

    _register_web_login_routes(
        mcp, _idp, MACHINE_TOKEN, PUBLIC_URL, _issue_dash_token if DB_URL else None
    )
