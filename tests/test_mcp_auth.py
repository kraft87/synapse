"""Auth-mode wiring for the MCP server.

The server is env-gated: no machine token => OPEN (dev/pre-cutover); machine token =>
StaticTokenVerifier bearer; + GitHub creds => MultiAuth with a login allowlist. These
tests reload the module under different env so the security boundary can't silently
regress (e.g. a refactor that drops the allowlist or stops gating the custom routes).
"""

from __future__ import annotations

import importlib

_AUTH_KEYS = (
    "SYNAPSE_MACHINE_TOKEN",
    "GITHUB_CLIENT_ID",
    "GITHUB_CLIENT_SECRET",
    "ALLOWED_GITHUB_USERS",
    "SYNAPSE_OAUTH_SIGNING_KEY",
    "OIDC_CONFIG_URL",
    "OIDC_CLIENT_ID",
    "OIDC_CLIENT_SECRET",
    "ALLOWED_OIDC_USERS",
)


def _reload(monkeypatch, env):
    # Point the module's dotenv fallback at a path that doesn't exist. Setting a key to
    # "" is not enough on its own: _cfg reads `os.environ.get(k) or _env.get(k)`, so an
    # empty env var falls through to the repo-root .env and a developer's local config
    # leaks into these assertions (CI passed only because it has no .env).
    monkeypatch.setenv("SYNAPSE_ENV_FILE", "/nonexistent/synapse-test.env")
    # Neutralize any inherited/.env auth config, then apply the case's env.
    for k in _AUTH_KEYS:
        monkeypatch.setenv(k, env.get(k, ""))
    import mcp_server.server as s

    return importlib.reload(s)


class _Req:
    def __init__(self, headers):
        self.headers = headers


def test_open_mode_has_no_auth(monkeypatch):
    s = _reload(monkeypatch, {})
    assert s._auth is None
    assert s._auth_mw == []
    # Open server => custom routes are intentionally ungated.
    assert s._machine_authorized(_Req({})) is True


def test_bearer_only(monkeypatch):
    from fastmcp.server.auth.providers.jwt import StaticTokenVerifier

    s = _reload(monkeypatch, {"SYNAPSE_MACHINE_TOKEN": "tok"})
    assert isinstance(s._auth, StaticTokenVerifier)
    assert s._auth_mw == []  # no GitHub leg => no allowlist middleware


def test_multiauth_with_allowlist(monkeypatch):
    from fastmcp.server.auth import MultiAuth

    s = _reload(
        monkeypatch,
        {
            "SYNAPSE_MACHINE_TOKEN": "tok",
            "GITHUB_CLIENT_ID": "id",
            "GITHUB_CLIENT_SECRET": "sec",
            "ALLOWED_GITHUB_USERS": "Alice, bob",
        },
    )
    assert isinstance(s._auth, MultiAuth)
    assert len(s._auth_mw) == 1
    assert s._auth_mw[0]._allowed == {"alice", "bob"}  # normalized lower, trimmed


def _stub_oidc_discovery(monkeypatch):
    """Make OIDCProxy construction network-free: pin the discovery doc it would fetch."""
    from fastmcp.server.auth import oidc_proxy as op

    cfg = op.OIDCConfiguration.model_validate(
        {
            "strict": False,
            "issuer": "https://idp.example.net",
            "authorization_endpoint": "https://idp.example.net/authorize",
            "token_endpoint": "https://idp.example.net/token",
            "userinfo_endpoint": "https://idp.example.net/userinfo",
            "jwks_uri": "https://idp.example.net/jwks.json",
        }
    )
    monkeypatch.setattr(
        op.OIDCProxy, "get_oidc_configuration", lambda self, url, strict, timeout: cfg
    )


def test_oidc_mode_replaces_github(monkeypatch):
    # OIDC env set => the OIDC leg is the interactive provider, even with GitHub also
    # configured (MCP discovery can only advertise one authorization server).
    from fastmcp.server.auth import MultiAuth

    from mcp_server.idp import OIDCIdP

    _stub_oidc_discovery(monkeypatch)
    s = _reload(
        monkeypatch,
        {
            "SYNAPSE_MACHINE_TOKEN": "tok",
            "GITHUB_CLIENT_ID": "ghid",
            "GITHUB_CLIENT_SECRET": "ghsec",
            "ALLOWED_GITHUB_USERS": "somegithubuser",
            "OIDC_CONFIG_URL": "https://idp.example.net/.well-known/openid-configuration",
            "OIDC_CLIENT_ID": "synapse",
            "OIDC_CLIENT_SECRET": "sec",
            "ALLOWED_OIDC_USERS": "Kyle",
        },
    )
    assert isinstance(s._auth, MultiAuth)
    assert len(s._auth_mw) == 1
    assert s._auth_mw[0]._allowed == {"kyle"}  # OIDC allowlist, not the GitHub one
    assert s._auth_mw[0]._claim_keys == ("preferred_username", "email")
    assert isinstance(s._idp, OIDCIdP)  # custom flows follow the same selection


def test_oidc_scopes_and_claims_are_configurable(monkeypatch):
    # Different IdPs grant different scopes/claims (e.g. Google: no offline_access,
    # identity in "email") — both knobs must come from env, not code.
    _stub_oidc_discovery(monkeypatch)
    monkeypatch.setenv("OIDC_SCOPES", "openid email")
    monkeypatch.setenv("OIDC_USER_CLAIMS", "email")
    s = _reload(
        monkeypatch,
        {
            "SYNAPSE_MACHINE_TOKEN": "tok",
            "OIDC_CONFIG_URL": "https://idp.example.net/.well-known/openid-configuration",
            "OIDC_CLIENT_ID": "synapse",
            "OIDC_CLIENT_SECRET": "sec",
            "ALLOWED_OIDC_USERS": "kyle@example.net",
        },
    )
    assert s.OIDC_SCOPES == ["openid", "email"]
    assert s._auth_mw[0]._claim_keys == ("email",)
    assert s._idp.scope == "openid email"
    assert s._idp.identity_claims == ("email",)


def test_github_mode_builds_github_idp(monkeypatch):
    from mcp_server.idp import GitHubIdP

    s = _reload(
        monkeypatch,
        {
            "SYNAPSE_MACHINE_TOKEN": "tok",
            "GITHUB_CLIENT_ID": "id",
            "GITHUB_CLIENT_SECRET": "sec",
            "ALLOWED_GITHUB_USERS": "alice",
        },
    )
    assert s._auth_mw[0]._claim_keys == ("login",)
    assert isinstance(s._idp, GitHubIdP)
    assert s._idp.allowed == {"alice"}


async def test_allowlist_middleware_claim_fallback(monkeypatch):
    # OIDC leg: preferred_username wins, email is the fallback, unknown users are refused.
    s = _reload(monkeypatch, {"SYNAPSE_MACHINE_TOKEN": "tok"})
    mw = s._UserAllowlist({"kyle", "kyle@example.net"}, ("preferred_username", "email"), "oidc")

    class _Tok:
        def __init__(self, claims):
            self.client_id = "some-oauth-client"
            self.claims = claims

    import fastmcp.exceptions as fe

    async def _next(_ctx):
        return "ok"

    for claims in ({"preferred_username": "Kyle"}, {"email": "Kyle@Example.net"}):
        monkeypatch.setattr(s, "get_access_token", lambda c=claims: _Tok(c))
        assert await mw.on_call_tool(None, _next) == "ok"

    monkeypatch.setattr(s, "get_access_token", lambda: _Tok({"preferred_username": "mallory"}))
    try:
        await mw.on_call_tool(None, _next)
        raise AssertionError("expected AuthorizationError")
    except fe.AuthorizationError as e:
        assert "mallory" in str(e)


def test_identity_claims_track_the_active_leg(monkeypatch):
    """Audience scoping derives an OAuth caller's surface (`oauth:<login>`) from the SAME
    claim keys the allowlist gate reads. If the two ever drift, a login clears the gate
    as one identity and is served as another."""
    _stub_oidc_discovery(monkeypatch)
    s = _reload(
        monkeypatch,
        {
            "SYNAPSE_MACHINE_TOKEN": "tok",
            "OIDC_CONFIG_URL": "https://idp.example.net/.well-known/openid-configuration",
            "OIDC_CLIENT_ID": "synapse",
            "OIDC_CLIENT_SECRET": "sec",
            "ALLOWED_OIDC_USERS": "kyle",
        },
    )
    assert s._IDENTITY_CLAIMS == s._auth_mw[0]._claim_keys == ("preferred_username", "email")

    # Bearer-only / open: no interactive leg means no OAuth callers to identify, so the
    # derivation stays inert and a caller's own `surface` param is left alone.
    s = _reload(monkeypatch, {"SYNAPSE_MACHINE_TOKEN": "tok"})
    assert s._IDENTITY_CLAIMS == ()
    assert s._caller_surface("work-host") == "work-host"


def test_machine_authorized_constant_time_check(monkeypatch):
    s = _reload(monkeypatch, {"SYNAPSE_MACHINE_TOKEN": "tok"})
    assert s._machine_authorized(_Req({"authorization": "Bearer tok"})) is True
    assert s._machine_authorized(_Req({"authorization": "Bearer wrong"})) is False
    assert s._machine_authorized(_Req({"authorization": "tok"})) is False  # no Bearer prefix
    assert s._machine_authorized(_Req({})) is False  # no header


def test_oauth_storage_persists_to_db_when_configured(monkeypatch):
    # DB_URL + signing key => OAuth-proxy state lands in Postgres (survives container
    # recreates), Fernet-wrapped so upstream tokens aren't plaintext in the served DB.
    from key_value.aio.wrappers.encryption import FernetEncryptionWrapper

    monkeypatch.setenv("SYNAPSE_DB_URL", "postgresql://u:p@127.0.0.1:5432/x")
    s = _reload(monkeypatch, {"SYNAPSE_OAUTH_SIGNING_KEY": "k" * 32})
    assert isinstance(s._oauth_client_storage(), FernetEncryptionWrapper)


def test_oauth_storage_none_without_db_or_key(monkeypatch):
    # Missing either => None => FastMCP's encrypted disk default (dev/stdio path).
    monkeypatch.setenv("SYNAPSE_DB_URL", "")
    s = _reload(monkeypatch, {"SYNAPSE_OAUTH_SIGNING_KEY": "k" * 32})
    assert s._oauth_client_storage() is None
