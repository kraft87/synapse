"""Unit tests for credential precedence, independent of the FastMCP application.

A caller's scope comes from its device token or its OAuth identity. Nothing else
resolves to a surfaces row: the shared root token, an open server, and an OAuth token
with no identity claim are all UNKNOWN (restricted, empty allowlist), and neither
resolver takes a ``surface`` argument any more, so there is nothing a caller can pass.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass

import pytest

from ingestion.surfaces import FULL_TRUST, UNKNOWN_SURFACE, token_hash
from mcp_server import caller_trust

_MACHINE_IDS = {"synapse-machine", "synapse-device"}


@dataclass
class _Token:
    client_id: str
    claims: dict


def _identity(claims: dict, keys: tuple[str, ...]) -> str:
    return next((str(claims[k]).lower() for k in keys if claims.get(k)), "")


@pytest.fixture()
def lookups(monkeypatch):
    """Record every resolve_caller call; answer FULL_TRUST so a leak would be visible."""
    calls: list[dict] = []

    def _resolve(_db, **kwargs):
        calls.append(kwargs)
        return FULL_TRUST

    monkeypatch.setattr(caller_trust, "resolve_caller", _resolve)
    return calls


def _mcp(token):
    return caller_trust.caller_trust(
        db_url="postgres://unused",
        access_token=token,
        identity_claims=("login",),
        machine_client_ids=_MACHINE_IDS,
        claims_identity=_identity,
    )


def test_neither_resolver_accepts_a_surface_argument():
    """Structural: a self-reported id cannot reach trust resolution because there is no
    parameter for it. Re-adding one would re-open the root-token lane."""
    assert "surface" not in inspect.signature(caller_trust.caller_trust).parameters
    assert "surface" not in inspect.signature(caller_trust.request_trust).parameters


def test_device_claims_decide_without_a_lookup(lookups):
    token = _Token(
        "synapse-device", {"kind": "device", "surface_id": "work", "trust": "restricted"}
    )
    result = _mcp(token)
    assert result.surface_id == "work"
    assert result.restricted and result.known
    assert lookups == []


def test_oauth_identity_resolves_its_server_derived_id(lookups):
    _mcp(_Token("claude-ai", {"login": "Kyle"}))
    assert lookups == [{"legacy_surface_id": "oauth:kyle"}]


def test_a_root_token_caller_is_unknown_and_never_looked_up(lookups):
    """The bug this module closes: the shared root token used to name any surface id
    and inherit that row's trust, full included. Now it names nothing."""
    assert _mcp(_Token("synapse-machine", {"kind": "root"})) == UNKNOWN_SURFACE
    assert lookups == []


def test_no_token_context_is_unknown(lookups):
    """An open server or a call outside a request has no credential to read."""
    assert _mcp(None) == UNKNOWN_SURFACE
    assert lookups == []


def test_an_oauth_token_without_an_identity_claim_is_unknown(lookups):
    assert _mcp(_Token("claude-ai", {})) == UNKNOWN_SURFACE
    assert lookups == []


def _http(bearer, machine_token="root-tok"):
    return caller_trust.request_trust(
        db_url="postgres://unused", machine_token=machine_token, bearer=bearer
    )


def test_http_device_bearer_is_resolved_by_its_hash(lookups):
    assert _http("device-tok") == FULL_TRUST
    assert lookups == [{"token_hash_hex": token_hash("device-tok")}]


@pytest.mark.parametrize(
    "bearer,machine_token",
    [
        ("root-tok", "root-tok"),  # the shared root token
        ("", "root-tok"),  # no bearer at all
        ("anything", ""),  # an open server: no root token configured
    ],
)
def test_http_without_a_device_bearer_is_unknown(lookups, bearer, machine_token):
    assert _http(bearer, machine_token) == UNKNOWN_SURFACE
    assert lookups == []


def test_an_ignored_surface_is_logged_once_per_site_without_its_value(monkeypatch, caplog):
    monkeypatch.setattr(caller_trust, "_IGNORED_SURFACE_SITES", set())
    with caplog.at_level(logging.WARNING, logger="mcp_server.caller_trust"):
        caller_trust.note_ignored_surface(None, "site-a")
        caller_trust.note_ignored_surface("", "site-a")
        assert caplog.records == []
        caller_trust.note_ignored_surface("laptop-1", "site-a")
        caller_trust.note_ignored_surface("laptop-2", "site-a")
        caller_trust.note_ignored_surface("laptop-1", "site-b")
    assert [r.getMessage().split(":")[0] for r in caplog.records] == ["site-a", "site-b"]
    assert not any("laptop" in r.getMessage() for r in caplog.records)
