"""Tests for mcp_server/web_login_routes.py — the dashboard's browser OAuth login.

Pure-Python (no DB): GitHub calls are stubbed at the module seams
(_exchange_code / _fetch_login). Covers the origin allowlist on /dash/oauth/start,
authorize-URL construction, state sign/verify (forged + expired), and the
callback's allowlist gate + fragment handoffs.
"""

from __future__ import annotations

import time
from urllib.parse import parse_qs, urlparse

import pytest
from fastmcp import FastMCP
from starlette.testclient import TestClient

import mcp_server.web_login_routes as wl

_MT = "test-machine-token"
_PUB = "https://synapse.example.net"
_LAN = "https://synapse.lan.example.net"


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("SYNAPSE_DASH_ORIGINS", _LAN)
    mcp = FastMCP("t")
    wl.register(mcp, "cid", "csecret", {"alloweduser"}, _MT, _PUB)
    with TestClient(mcp.http_app(), base_url=_PUB) as c:
        yield c


def _start(client, host="synapse.example.net", proto="https"):
    return client.get(
        "/dash/oauth/start",
        headers={"x-forwarded-host": host, "x-forwarded-proto": proto},
        follow_redirects=False,
    )


def test_start_redirects_to_github_with_signed_state(client):
    r = _start(client)
    assert r.status_code == 302
    url = urlparse(r.headers["location"])
    assert url.netloc == "github.com" and url.path == "/login/oauth/authorize"
    q = parse_qs(url.query)
    assert q["client_id"] == ["cid"]
    assert q["redirect_uri"] == [f"{_PUB}/auth/callback/dash"]
    assert wl._verify_state(_MT, q["state"][0]) == _PUB


def test_start_accepts_configured_lan_origin(client):
    r = _start(client, host="synapse.lan.example.net")
    assert r.status_code == 302
    q = parse_qs(urlparse(r.headers["location"]).query)
    assert wl._verify_state(_MT, q["state"][0]) == _LAN


def test_start_refuses_unlisted_origin(client):
    r = _start(client, host="evil.example.com")
    assert r.status_code == 403


def test_callback_rejects_forged_and_expired_state(client):
    r = client.get("/auth/callback/dash?code=x&state=garbage", follow_redirects=False)
    assert r.status_code == 400
    # tampered signature
    forged = wl._sign_state("wrong-key", _LAN)
    r = client.get(f"/auth/callback/dash?code=x&state={forged}", follow_redirects=False)
    assert r.status_code == 400
    # expired
    old = wl._sign_state(_MT, _LAN)
    payload, _sig = old.split(".", 1)
    import json as _json

    data = _json.loads(wl._unb64(payload))
    data["t"] = int(time.time()) - 999999
    stale_payload = wl._b64(_json.dumps(data).encode())
    import hashlib as _hl
    import hmac as _hm

    stale = (
        stale_payload
        + "."
        + wl._b64(_hm.new(_MT.encode(), stale_payload.encode(), _hl.sha256).digest())
    )
    r = client.get(f"/auth/callback/dash?code=x&state={stale}", follow_redirects=False)
    assert r.status_code == 400


def test_callback_happy_path_hands_token_to_origin(client, monkeypatch):
    async def fake_exchange(cid, cs, code, ruri):
        assert code == "goodcode" and ruri == f"{_PUB}/auth/callback/dash"
        return "gh-access"

    async def fake_login(access):
        assert access == "gh-access"
        return "alloweduser"

    monkeypatch.setattr(wl, "_exchange_code", fake_exchange)
    monkeypatch.setattr(wl, "_fetch_login", fake_login)
    state = wl._sign_state(_MT, _LAN)
    r = client.get(f"/auth/callback/dash?code=goodcode&state={state}", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == f"{_LAN}/dash#token={_MT}"


def test_callback_rejects_unlisted_github_user(client, monkeypatch):
    async def fake_exchange(cid, cs, code, ruri):
        return "gh-access"

    async def fake_login(access):
        return "stranger"

    monkeypatch.setattr(wl, "_exchange_code", fake_exchange)
    monkeypatch.setattr(wl, "_fetch_login", fake_login)
    state = wl._sign_state(_MT, _LAN)
    r = client.get(f"/auth/callback/dash?code=c&state={state}", follow_redirects=False)
    assert r.status_code == 302
    loc = r.headers["location"]
    assert loc.startswith(f"{_LAN}/dash#login_error=") and "allowlist" in loc


def test_callback_user_cancel_surfaces_error_fragment(client):
    state = wl._sign_state(_MT, _LAN)
    r = client.get(f"/auth/callback/dash?error=access_denied&state={state}", follow_redirects=False)
    assert r.status_code == 302
    assert f"{_LAN}/dash#login_error=" in r.headers["location"]
