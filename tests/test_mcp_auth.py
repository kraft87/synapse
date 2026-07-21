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
)


def _reload(monkeypatch, env):
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
