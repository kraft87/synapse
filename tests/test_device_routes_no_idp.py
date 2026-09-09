"""`POST /device/*` on a server with no identity provider.

The routes used to simply not exist in that configuration, so a client asking to enroll
got a bare 404 and printed "enrollment unavailable: unknown". On a local install that is
not an error at all — there is genuinely nobody to sign in to — and the user needs the
OTHER remedy (`synapse-admin bootstrap`), which a 404 cannot tell them about.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from mcp_server.device_routes import register


def _client(*, idp=None, machine_token="root-tok"):
    from fastmcp import FastMCP

    mcp = FastMCP("test-device-routes")
    register(mcp, idp, machine_token)
    return TestClient(mcp.http_app())


@pytest.mark.parametrize("path", ["/device/code", "/device/token"])
def test_no_idp_is_a_clean_json_error_naming_the_bootstrap(path):
    with _client(idp=None) as client:
        r = client.post(path, json={})
    assert r.status_code == 503
    body = r.json()
    assert body["error"] == "no_idp"
    # The client keys off `error`; the description is what a human reads.
    assert "synapse-admin bootstrap" in body["error_description"]


def test_no_machine_token_is_distinguishable_from_no_idp():
    """Different cause, different remedy: nothing to hand back vs nobody to ask."""
    with _client(idp=object(), machine_token="") as client:
        r = client.post("/device/code", json={})
    assert r.status_code == 503
    assert r.json()["error"] == "no_machine_token"


def test_a_configured_server_still_serves_the_real_flow():
    class _IdP:
        label = "fake"
        device_disabled_hint = "turn it on"

        async def device_start(self):
            return {"device_code": "dc", "user_code": "ABCD", "verification_uri": "https://x"}

    with _client(idp=_IdP()) as client:
        r = client.post("/device/code", json={})
    assert r.status_code == 200 and r.json()["user_code"] == "ABCD"
