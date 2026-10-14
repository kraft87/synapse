"""The plain-HTTP gates resolve a device bearer once per request, off the event loop.

FastMCP's auth middleware runs app-wide, so SynapseTokenVerifier already looks up every
bearer, custom routes included. The gates used to look it up again (client or admin
gate), and the handler a third time for its trust, the last two synchronously inside
async handlers. During a Postgres stall each of those lookups could wait out the pool's
borrow timeout on the loop thread, freezing every other request, /health and the
root-token /ingest lane included. These tests pin the fix without a database: the
verifier's lookup is the only one, it runs in a worker thread, and the gates read its
verdict.
"""

from __future__ import annotations

import threading

import pytest
from fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from ingestion.surfaces import FULL_TRUST, UNKNOWN_SURFACE, SurfaceTrust, token_hash
from mcp_server import auth_tokens, caller_trust, http_auth

_ROOT = "root-tok"
_DB = "postgres://unused"
_WORK = SurfaceTrust(surface_id="dev-work", trust="restricted", allowed_projects=("w",), known=True)
_ROWS = {token_hash("full-dev"): FULL_TRUST, token_hash("work-dev"): _WORK}


@pytest.fixture()
def lookups(monkeypatch):
    """Record the thread of every surfaces lookup, from the verifier or from a gate."""
    calls: list[tuple[str, int]] = []

    def _recorder(where: str):
        def _resolve(_db, *, token_hash_hex=None, **_):
            calls.append((where, threading.get_ident()))
            return _ROWS.get(token_hash_hex, UNKNOWN_SURFACE)

        return _resolve

    monkeypatch.setattr(auth_tokens, "resolve_caller", _recorder("verifier"))
    monkeypatch.setattr(caller_trust, "resolve_caller", _recorder("gate"))
    return calls


async def _probe(request: Request) -> JSONResponse:
    """What every custom route does: clear a gate, then read the caller's trust."""
    client_ok = http_auth.machine_authorized(request, _ROOT, _DB)
    admin_ok = http_auth.admin_authorized(request, _ROOT, _DB)
    st = http_auth.request_trust(request, _ROOT, _DB)
    return JSONResponse(
        {
            "client": client_ok,
            "admin": admin_ok,
            "surface": st.surface_id,
            "projects": list(st.allowed_projects),
            "stamp": st.stamp_surface_id,
            "handler_thread": threading.get_ident(),
        }
    )


def _server() -> TestClient:
    """A credentialed server wired the way server_auth wires it (verifier as the auth)."""
    mcp = FastMCP("t", auth=auth_tokens.SynapseTokenVerifier(_ROOT, _DB, ["user"]))
    mcp.custom_route("/probe", methods=["GET"])(_probe)
    return TestClient(mcp.http_app())


def _get(client: TestClient, token: str | None) -> dict:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return client.get("/probe", headers=headers).json()


def test_a_device_request_costs_one_lookup_made_off_the_event_loop(lookups):
    out = _get(_server(), "work-dev")
    assert (out["client"], out["admin"]) == (True, False)
    # The verifier's verdict carries through whole, provenance stamp included.
    assert (out["surface"], out["projects"], out["stamp"]) == ("dev-work", ["w"], "dev-work")
    assert [where for where, _ in lookups] == ["verifier"]
    assert lookups[0][1] != out["handler_thread"]


def test_a_full_trust_device_clears_the_admin_gate_on_the_verifiers_word(lookups):
    out = _get(_server(), "full-dev")
    assert (out["client"], out["admin"]) == (True, True)
    assert [where for where, _ in lookups] == ["verifier"]


def test_a_rejected_bearer_is_unknown_without_a_second_lookup(lookups):
    """During a Postgres stall the verifier rejects every device token. A gate that then
    retried the lookup itself would block the event loop, which is the outage this
    avoids; the verifier's "no" stands."""
    out = _get(_server(), "garbage")
    assert (out["client"], out["admin"], out["surface"]) == (False, False, None)
    assert [where for where, _ in lookups] == ["verifier"]


def test_the_gates_accept_only_the_bearer_form_they_parse(lookups):
    """The verifier takes "bearer <token>" in any case; the gates read "Bearer " only.
    Reusing the verifier's verdict must not widen what the gates accept."""
    client = _server()
    out = client.get("/probe", headers={"Authorization": "bearer work-dev"}).json()
    assert (out["client"], out["admin"], out["surface"]) == (False, False, None)


def test_root_and_no_bearer_never_touch_the_database(lookups):
    client = _server()
    out = _get(client, _ROOT)
    assert (out["client"], out["admin"], out["surface"]) == (True, False, None)
    out = _get(client, None)
    assert (out["client"], out["admin"], out["surface"]) == (False, False, None)
    assert lookups == []


def test_without_auth_middleware_the_gates_share_one_lookup_per_request(lookups):
    """A bare app (tests, an embedding) has no verifier verdict to read, so the first
    gate looks the bearer up and everything after it reuses that, for this request
    only: a revoke must bite on the very next call."""
    client = TestClient(Starlette(routes=[Route("/probe", _probe)]))
    out = _get(client, "work-dev")
    assert (out["client"], out["admin"], out["surface"]) == (True, False, "dev-work")
    assert [where for where, _ in lookups] == ["gate"]
    _get(client, "work-dev")
    assert [where for where, _ in lookups] == ["gate", "gate"]


async def test_the_verifier_looks_up_a_device_token_off_the_event_loop(lookups):
    v = auth_tokens.SynapseTokenVerifier(_ROOT, _DB, ["user"])
    tok = await v.verify_token("work-dev")
    assert tok is not None and tok.claims["surface_id"] == "dev-work"
    assert lookups and lookups[0][1] != threading.get_ident()
