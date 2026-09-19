"""Authentication-provider construction and persistent OAuth storage."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastmcp.server.auth import MultiAuth
from fastmcp.server.auth.providers.github import GitHubProvider

from mcp_server.auth_tokens import SynapseTokenVerifier

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuthSettings:
    DB_URL: str
    MACHINE_TOKEN: str
    OIDC_SCOPES: list[str]
    OIDC_CONFIG_URL: str
    OIDC_CLIENT_ID: str
    OIDC_CLIENT_SECRET: str
    PUBLIC_URL: str
    OAUTH_SIGNING_KEY: str
    ALLOWED_OIDC_USERS: set[str]
    OIDC_USER_CLAIMS: tuple[str, ...]
    GITHUB_CLIENT_ID: str
    GITHUB_CLIENT_SECRET: str
    ALLOWED_GITHUB_USERS: set[str]


_ALLOWED_CLIENT_REDIRECTS = [
    "https://claude.ai/api/mcp/auth_callback",
    "https://claude.com/api/mcp/auth_callback",
    # `synapse login` (RFC 8252): an ephemeral loopback redirect on a random
    # port. Without these patterns the OAuthProxy 400s the authorize step
    # ("does not match allowed patterns") and the CLI login can never complete.
    "http://localhost:*",
    "http://127.0.0.1:*",
]


def oauth_client_storage(DB_URL: str, OAUTH_SIGNING_KEY: str) -> Any:
    """Where the OAuth proxy keeps its state: DCR client registrations, upstream GitHub
    tokens, and JTI mappings.

    FastMCP's default is a FileTree store under ~/.local/share/fastmcp/oauth-proxy. In a
    container with no volume that path is on the ephemeral layer, so every recreate
    (watchtower redeploy, reboot) wipes it — the claude.ai connector's registered client
    then vanishes and its next token refresh fails, forcing a full re-auth. Persist in
    Postgres instead (the DB already survives on its own volume).

    FastMCP only Fernet-wraps the state in its disk-default branch; a bare backend stores
    the upstream GitHub tokens as plaintext. Since this is the same DB recall() serves, we
    wrap it ourselves with a key deterministically derived from the signing key — matching
    the default's encryption-at-rest. Returns None (=> FastMCP's encrypted disk default)
    when there's no DB or signing key, e.g. dev/stdio.
    """
    if not (DB_URL and OAUTH_SIGNING_KEY):
        return None
    import base64
    import hashlib

    from cryptography.fernet import Fernet
    from key_value.aio.stores.postgresql import PostgreSQLStore
    from key_value.aio.wrappers.encryption import FernetEncryptionWrapper

    # Deterministic 32-byte Fernet key from the (stable) signing key. Self-contained: it
    # does not have to match FastMCP's internal derivation, only be stable across restarts
    # so the same ciphertext decrypts after a redeploy.
    fernet_key = base64.urlsafe_b64encode(
        hashlib.sha256(f"synapse-oauth-store::{OAUTH_SIGNING_KEY}".encode()).digest()
    )
    store = PostgreSQLStore(url=DB_URL, table_name="oauth_proxy_kv")
    return FernetEncryptionWrapper(
        key_value=store, fernet=Fernet(fernet_key), raise_on_decryption_error=False
    )


def build_auth(
    config: AuthSettings,
    _oauth_client_storage: Callable[[], Any],
    _UserAllowlist: Callable[..., Any],
) -> tuple[Any, list[Any]]:
    """(auth_provider, middleware). No machine token => open server (dev/stdio/pre-cutover)."""
    if not config.MACHINE_TOKEN:
        return None, []
    # The bearer's scopes must clear whichever interactive leg is active (MultiAuth
    # applies the server's required scopes to /mcp): "user" for GitHub, the OIDC set
    # otherwise. Carrying both is harmless.
    bearer = SynapseTokenVerifier(
        config.MACHINE_TOKEN, config.DB_URL, ["user", *config.OIDC_SCOPES]
    )
    if config.OIDC_CONFIG_URL and config.OIDC_CLIENT_ID:
        from fastmcp.server.auth.oidc_proxy import OIDCProxy

        oidc = OIDCProxy(
            config_url=config.OIDC_CONFIG_URL,
            client_id=config.OIDC_CLIENT_ID,
            client_secret=config.OIDC_CLIENT_SECRET,
            base_url=config.PUBLIC_URL,
            jwt_signing_key=config.OAUTH_SIGNING_KEY or None,
            client_storage=_oauth_client_storage(),
            # Self-hosted IdPs (Authelia et al.) issue opaque access tokens; the
            # id_token is the JWT the discovery JWKS can verify.
            verify_id_token=True,
            required_scopes=list(config.OIDC_SCOPES),
            allowed_client_redirect_uris=_ALLOWED_CLIENT_REDIRECTS,
        )
        if not config.ALLOWED_OIDC_USERS:
            logger.warning(
                "OIDC auth on but ALLOWED_OIDC_USERS empty -> all human logins DENIED (fail-closed)"
            )
        return (
            MultiAuth(server=oidc, verifiers=[bearer]),
            [_UserAllowlist(config.ALLOWED_OIDC_USERS, config.OIDC_USER_CLAIMS, "oidc")],
        )
    if not config.GITHUB_CLIENT_ID:
        return bearer, []  # bearer-only: hooks + Claude Code --header; no claude.ai-web connector
    github = GitHubProvider(
        client_id=config.GITHUB_CLIENT_ID,
        client_secret=config.GITHUB_CLIENT_SECRET,
        base_url=config.PUBLIC_URL,
        jwt_signing_key=config.OAUTH_SIGNING_KEY or None,
        client_storage=_oauth_client_storage(),
        allowed_client_redirect_uris=_ALLOWED_CLIENT_REDIRECTS,
    )
    if not config.ALLOWED_GITHUB_USERS:
        logger.warning(
            "GitHub OAuth on but ALLOWED_GITHUB_USERS empty -> all human logins DENIED (fail-closed)"
        )
    return (
        MultiAuth(server=github, verifiers=[bearer]),
        [_UserAllowlist(config.ALLOWED_GITHUB_USERS, ("login",), "github")],
    )


def build_idp(config: AuthSettings) -> Any:
    """IdP for the custom login flows (dashboard + device) — same selection as _build_auth."""
    if config.OIDC_CONFIG_URL and config.OIDC_CLIENT_ID:
        from mcp_server.idp import OIDCIdP

        # The custom flows discard the upstream tokens after the identity check, so
        # they never need offline_access/refresh — request the trimmed scope set.
        return OIDCIdP(
            config_url=config.OIDC_CONFIG_URL,
            client_id=config.OIDC_CLIENT_ID,
            client_secret=config.OIDC_CLIENT_SECRET,
            allowed=config.ALLOWED_OIDC_USERS,
            scope=" ".join(s for s in config.OIDC_SCOPES if s != "offline_access"),
            identity_claims=config.OIDC_USER_CLAIMS,
        )
    if config.GITHUB_CLIENT_ID:
        from mcp_server.idp import GitHubIdP

        return GitHubIdP(
            client_id=config.GITHUB_CLIENT_ID,
            client_secret=config.GITHUB_CLIENT_SECRET,
            allowed=config.ALLOWED_GITHUB_USERS,
        )
    return None
