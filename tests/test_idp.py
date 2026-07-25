"""Tests for mcp_server/idp.py (both providers, HTTP stubbed at the _post_form/_get_json
seams) and the device routes running against an OIDC provider — the shape GitHub's
pre-standard dialect and RFC 8628 must both normalize into.
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pytest
from fastmcp import FastMCP
from starlette.testclient import TestClient

import mcp_server.device_routes as dr
import mcp_server.idp as idp_mod
from mcp_server.idp import GitHubIdP, OIDCIdP

_CFG = {
    "authorization_endpoint": "https://idp.example.net/authorize",
    "token_endpoint": "https://idp.example.net/token",
    "userinfo_endpoint": "https://idp.example.net/userinfo",
    "device_authorization_endpoint": "https://idp.example.net/device",
}


@pytest.fixture()
def oidc():
    p = OIDCIdP(
        config_url="https://idp.example.net/.well-known/openid-configuration",
        client_id="synapse",
        client_secret="sec",
        allowed={"kyle"},
    )
    p._config = dict(_CFG)  # discovery pre-cached => no network
    return p


async def test_oidc_authorize_url_shape(oidc):
    url = urlparse(await oidc.authorize_url("https://s.example.net/auth/callback/dash", "st8"))
    assert f"{url.scheme}://{url.netloc}{url.path}" == _CFG["authorization_endpoint"]
    q = parse_qs(url.query)
    assert q["response_type"] == ["code"]
    assert q["client_id"] == ["synapse"]
    assert q["scope"] == ["openid profile email"]
    assert q["state"] == ["st8"]


async def test_oidc_identity_prefers_username_then_email(oidc, monkeypatch):
    claims_box = {}

    async def fake_get(url, token, accept="application/json"):
        assert url == _CFG["userinfo_endpoint"] and token == "at"
        return claims_box

    monkeypatch.setattr(idp_mod, "_get_json", fake_get)

    claims_box.update({"email": "Kyle@Example.net", "preferred_username": "Kyle", "sub": "u-1"})
    assert await oidc.fetch_identity("at") == "kyle"
    claims_box.clear()
    claims_box.update({"email": "Kyle@Example.net", "sub": "u-1"})
    assert await oidc.fetch_identity("at") == "kyle@example.net"
    claims_box.clear()
    assert await oidc.fetch_identity("at") == ""


async def test_oidc_device_calls_use_client_auth(oidc, monkeypatch):
    seen = []

    async def fake_post(url, data, auth=None):
        seen.append((url, data, auth))
        return {"device_code": "dc", "user_code": "UC"}

    monkeypatch.setattr(idp_mod, "_post_form", fake_post)
    await oidc.device_start()
    await oidc.device_poll("dc")
    assert seen[0][0] == _CFG["device_authorization_endpoint"]
    assert seen[0][1]["client_id"] == "synapse"  # body param per RFC 8628 §3.1
    assert seen[0][2] == ("synapse", "sec")
    assert seen[1][0] == _CFG["token_endpoint"]
    assert seen[1][1]["grant_type"] == "urn:ietf:params:oauth:grant-type:device_code"
    assert seen[1][1]["client_id"] == "synapse"
    assert seen[1][2] == ("synapse", "sec")


async def test_oidc_device_start_without_endpoint_is_disabled(oidc):
    del oidc._config["device_authorization_endpoint"]
    assert (await oidc.device_start())["error"] == "device_flow_disabled"


async def test_github_authorize_url_and_device_start(monkeypatch):
    gh = GitHubIdP("cid", "csecret", {"alice"})
    url = urlparse(await gh.authorize_url("https://s.example.net/auth/callback/dash", "st8"))
    assert url.netloc == "github.com"
    q = parse_qs(url.query)
    assert q["client_id"] == ["cid"] and q["allow_signup"] == ["false"]

    seen = []

    async def fake_post(url, data, auth=None):
        seen.append((url, data, auth))
        return {}

    monkeypatch.setattr(idp_mod, "_post_form", fake_post)
    await gh.device_start()
    # GitHub device endpoints authenticate by client_id alone — no basic auth.
    assert seen[0][1]["client_id"] == "cid" and seen[0][2] is None


# ---------------------------------------------------------------------------
# Device routes against an OIDC-shaped provider
# ---------------------------------------------------------------------------

_MT = "test-machine-token"


@pytest.fixture()
def device_client(oidc):
    mcp = FastMCP("t")
    dr.register(mcp, oidc, _MT)
    with TestClient(mcp.http_app()) as c:
        yield c


async def test_device_route_happy_path(device_client, oidc, monkeypatch):
    async def fake_start():
        return {"device_code": "dc", "user_code": "UC-1", "verification_uri": "https://idp/dev"}

    async def fake_poll(device_code):
        assert device_code == "dc"
        return {"access_token": "at"}

    async def fake_identity(access):
        return "kyle"

    monkeypatch.setattr(oidc, "device_start", fake_start)
    monkeypatch.setattr(oidc, "device_poll", fake_poll)
    monkeypatch.setattr(oidc, "fetch_identity", fake_identity)

    start = device_client.post("/device/code", json={}).json()
    assert start["user_code"] == "UC-1" and start["verification_uri"] == "https://idp/dev"
    done = device_client.post("/device/token", json={"device_code": "dc"}).json()
    assert done == {"token": _MT, "login": "kyle"}


async def test_device_route_pending_and_denied(device_client, oidc, monkeypatch):
    responses = [{"error": "authorization_pending"}, {"access_token": "at"}]

    async def fake_poll(device_code):
        return responses.pop(0)

    async def fake_identity(access):
        return "stranger"

    monkeypatch.setattr(oidc, "device_poll", fake_poll)
    monkeypatch.setattr(oidc, "fetch_identity", fake_identity)

    r = device_client.post("/device/token", json={"device_code": "dc"})
    assert r.json()["error"] == "authorization_pending"  # poll on
    r = device_client.post("/device/token", json={"device_code": "dc"})
    assert r.status_code == 403 and "allowlist" in r.json()["error_description"]


async def test_device_route_disabled_hint(device_client, oidc):
    del oidc._config["device_authorization_endpoint"]
    r = device_client.post("/device/code", json={})
    assert r.status_code == 400
    body = r.json()
    assert body["error"] == "device_flow_disabled"
    assert "device_authorization_endpoint" in body["error_description"]
