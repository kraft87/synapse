"""Unit tests for credential precedence, independent of the FastMCP application."""

from __future__ import annotations

from dataclasses import dataclass

from ingestion.surfaces import FULL_TRUST, UNKNOWN_SURFACE
from mcp_server import caller_trust


@dataclass
class _Token:
    client_id: str
    claims: dict


def _identity(claims: dict, keys: tuple[str, ...]) -> str:
    return next((str(claims[k]).lower() for k in keys if claims.get(k)), "")


def test_device_claims_override_a_client_supplied_surface(monkeypatch):
    token = _Token(
        "synapse-device", {"kind": "device", "surface_id": "work", "trust": "restricted"}
    )
    monkeypatch.setattr(caller_trust, "resolve_caller", lambda *_args, **_kwargs: FULL_TRUST)

    result = caller_trust.caller_trust(
        db_url="postgres://unused",
        surface="personal-host",
        access_token=token,
        identity_claims=("login",),
        machine_client_ids={"synapse-machine", "synapse-device"},
        claims_identity=_identity,
    )

    assert result.surface_id == "work"
    assert result.restricted


def test_oauth_identity_overrides_a_client_supplied_surface(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(
        caller_trust,
        "resolve_caller",
        lambda _db, **kwargs: calls.append(kwargs) or UNKNOWN_SURFACE,
    )

    caller_trust.caller_trust(
        db_url="postgres://unused",
        surface="personal-host",
        access_token=_Token("claude-ai", {"login": "Kyle"}),
        identity_claims=("login",),
        machine_client_ids={"synapse-machine", "synapse-device"},
        claims_identity=_identity,
    )

    assert calls == [{"legacy_surface_id": "oauth:kyle"}]


def test_legacy_surface_is_used_only_when_no_credential_identity_exists(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(
        caller_trust,
        "resolve_caller",
        lambda _db, **kwargs: calls.append(kwargs) or UNKNOWN_SURFACE,
    )

    caller_trust.caller_trust(
        db_url="postgres://unused",
        surface="work-host",
        access_token=None,
        identity_claims=("login",),
        machine_client_ids={"synapse-machine", "synapse-device"},
        claims_identity=_identity,
    )

    assert calls == [{"legacy_surface_id": "work-host"}]
