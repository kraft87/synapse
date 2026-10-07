"""Serve-nothing notices: the server says WHY an authenticated caller gets nothing.

Fail-closed serving is silent by construction. A caller with no device identity (the
shared root token, an id with no row, a revoked row), or a registered restricted surface
with an empty allowlist, gets a healthy 200 with empty buckets and an empty board. The
fix is server-side text, because that is the one channel every client renders, old ones
included: the board comes from /context, and recall results carry ``warnings``.

The properties pinned here:

  * Unknown and restricted-with-no-projects verdicts get a notice, first in ``warnings``
    and at the top of the board.
  * Full trust and restricted-with-projects get nothing: narrower serving is the design.
  * Only an AUTHENTICATED caller is told. An open server has no credential to fix, and an
    unauthenticated request is refused before any verdict exists, so there is no oracle.
  * The text carries no ids or credentials, the caller's own surface id included: it
    lands in model context and transcripts, which are ingested themselves.
"""

from __future__ import annotations

import importlib
import os

import pytest

from ingestion.surfaces import FULL_TRUST, UNKNOWN_SURFACE, SurfaceTrust
from mcp_server.recall_warnings import (
    NO_CREDENTIAL_NOTICE,
    RESTRICTED_EMPTY_NOTICE,
    SIGN_IN_NOTICE,
    serving_notice,
    with_notice,
)

_ALL_NOTICES = (NO_CREDENTIAL_NOTICE, RESTRICTED_EMPTY_NOTICE, SIGN_IN_NOTICE)

# ---------------------------------------------------------------------------
# The verdict -> notice mapping (pure, no DB)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "trust,expected",
    [
        # UNKNOWN: root token with no surface, an id with no row, a revoked row.
        (UNKNOWN_SURFACE, NO_CREDENTIAL_NOTICE),
        (SurfaceTrust(surface_id="some-legacy-host"), NO_CREDENTIAL_NOTICE),
        # A registered restricted surface that was granted no projects.
        (SurfaceTrust("dev-0a1b2c", "restricted", (), known=True), RESTRICTED_EMPTY_NOTICE),
        # The sign-in lane names its own remedy (no device token to re-mint there).
        (SurfaceTrust(surface_id="oauth:someone"), SIGN_IN_NOTICE),
        (SurfaceTrust("oauth:someone", "restricted", (), known=True), SIGN_IN_NOTICE),
        # Restricted WITH projects is narrower serving by design: no notice.
        (SurfaceTrust("dev-0a1b2c", "restricted", ("alpha",), known=True), None),
        (SurfaceTrust("oauth:someone", "restricted", ("alpha",), known=True), None),
        # Full trust never.
        (FULL_TRUST, None),
        (SurfaceTrust("dev-0a1b2c", "full", (), known=True), None),
        (None, None),
    ],
)
def test_serving_notice_fires_only_for_the_serve_nothing_verdicts(trust, expected):
    assert serving_notice(trust) == expected


def test_with_notice_goes_first_and_keeps_leg_warnings():
    leg = "embedding failed (voyage: Unauthorized): vector legs skipped."
    out = with_notice({"query": "q", "facts": [], "warnings": [leg]}, NO_CREDENTIAL_NOTICE)
    assert out["warnings"] == [NO_CREDENTIAL_NOTICE, leg]
    # Idempotent: attaching twice does not duplicate.
    assert with_notice(out, NO_CREDENTIAL_NOTICE)["warnings"] == [NO_CREDENTIAL_NOTICE, leg]


def test_with_notice_leaves_a_healthy_shape_untouched():
    out = {"query": "q", "facts": []}
    assert with_notice(out, None) == {"query": "q", "facts": []}  # no empty warnings key
    assert with_notice(["not", "a", "dict"], NO_CREDENTIAL_NOTICE) == ["not", "a", "dict"]


@pytest.mark.parametrize("notice", _ALL_NOTICES)
def test_notice_text_names_no_ids_or_credentials(notice):
    """The notice is a constant, so nothing from the caller's verdict can leak into it:
    not its surface id, not a project, not a token. One or two lines on the board."""
    for sid in ("dev-0a1b2c", "oauth:someone", "some-legacy-host", "alpha"):
        trust = SurfaceTrust(sid, "restricted", (), known=True)
        assert sid not in (serving_notice(trust) or "")
    assert "\n" not in notice
    assert len(notice) <= 240
    for leak in ("Bearer", "token=", "http://", "https://", "@"):
        assert leak not in notice


# ---------------------------------------------------------------------------
# End to end through the real server wiring
# ---------------------------------------------------------------------------

_TEST_DB = os.environ.get(
    "SYNAPSE_TEST_URL", "postgresql://synapse:synapse@127.0.0.1:5432/synapse_test"
)
_ROOT = "root-tok"


def _db_reachable() -> bool:
    try:
        import psycopg

        psycopg.connect(_TEST_DB, connect_timeout=2).close()
        return True
    except Exception:  # pragma: no cover - environment dependent
        return False


_needs_db = pytest.mark.skipif(not _db_reachable(), reason="no test DB reachable")

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


class _StubEngine:
    """The engine's four read entry points, returning the empty serve an unknown or
    empty-allowlist caller gets. ``leg_warning`` simulates a degraded leg, so the test
    can prove the notice composes with #182's list instead of replacing it."""

    def __init__(self, leg_warning: str | None = None) -> None:
        self._leg = leg_warning

    def _out(self, **body):
        if self._leg:
            body["warnings"] = [self._leg]
        return body

    def recall(self, **kw):
        return self._out(query=kw["query"], facts=[])

    def recall_episodes(self, **kw):
        return self._out(query=kw["query"], episodes=[])

    def fetch(self, ids, **kw):
        return {"episodes": [], "notes": [], "skipped": []}

    def fetch_session(self, **kw):
        return {"session_id": kw["session_id"], "error": "session not indexed"}

    def record_event(self, *a, **k):  # board telemetry
        return None


class _Tok:
    """What FastMCP's verifier stamps on a request (see mcp_server.auth_tokens)."""

    def __init__(self, client_id: str, claims: dict) -> None:
        self.client_id = client_id
        self.claims = claims


def _reload(monkeypatch, machine_token: str):
    """The server module, reloaded against the test DB with (or without) a root token —
    the same isolation test_mcp_auth uses, so a local .env cannot leak in."""
    monkeypatch.setenv("SYNAPSE_ENV_FILE", "/nonexistent/synapse-test.env")
    for k in _AUTH_KEYS:
        monkeypatch.setenv(k, "")
    monkeypatch.setenv("SYNAPSE_MACHINE_TOKEN", machine_token)
    monkeypatch.setenv("SYNAPSE_DB_URL", _TEST_DB)
    import mcp_server.server as s

    s = importlib.reload(s)
    monkeypatch.setattr(s, "_recall_engine", _StubEngine())
    return s


@pytest.fixture()
def devices(conn):
    """One device per verdict that matters: full, restricted with projects, restricted
    with none. Returned as token -> surface id."""
    from tests.helpers.surfaces import clear_surfaces, register_device

    clear_surfaces(conn)
    ids = {
        "full-dev-tok": register_device(conn, "full-dev-tok", trust="full", surface_id="dev-f"),
        "scoped-dev-tok": register_device(
            conn, "scoped-dev-tok", trust="restricted", projects=["alpha"], surface_id="dev-s"
        ),
        "empty-dev-tok": register_device(
            conn, "empty-dev-tok", trust="restricted", projects=[], surface_id="dev-e"
        ),
    }
    yield ids
    clear_surfaces(conn)


@pytest.fixture()
def server(monkeypatch, devices):
    return _reload(monkeypatch, _ROOT)


def _device_claims(conn, token: str) -> dict:
    from ingestion.surfaces import resolve_caller, token_hash

    st = resolve_caller(_TEST_DB, token_hash_hex=token_hash(token))
    return {
        "kind": "device",
        "surface_id": st.surface_id,
        "trust": st.trust,
        "allowed_projects": list(st.allowed_projects),
    }


def _call_all_tools(s) -> list[dict]:
    return [
        s.recall("widget"),
        s.recall_full_turns("widget"),
        s.fetch(["e:1"]),
        s.fetch_session("some-session"),
    ]


@_needs_db
@pytest.mark.parametrize(
    "token,expected",
    [
        (None, NO_CREDENTIAL_NOTICE),  # the shared root token: no device identity
        ("empty-dev-tok", RESTRICTED_EMPTY_NOTICE),
        ("scoped-dev-tok", None),
        ("full-dev-tok", None),
    ],
)
def test_mcp_read_tools_explain_a_serve_nothing_verdict(server, conn, monkeypatch, token, expected):
    if token is None:
        tok = _Tok(server._MACHINE_CLIENT_ID, {"kind": "root"})
    else:
        tok = _Tok(server._DEVICE_CLIENT_ID, _device_claims(conn, token))
    monkeypatch.setattr(server, "get_access_token", lambda: tok)

    for out in _call_all_tools(server):
        if expected is None:
            assert "warnings" not in out, out
        else:
            assert out["warnings"] == [expected], out


@_needs_db
def test_mcp_notice_precedes_leg_degradation_warnings(server, monkeypatch):
    leg = "embedding failed (voyage: Unauthorized): vector legs skipped, results are BM25-only."
    monkeypatch.setattr(server, "_recall_engine", _StubEngine(leg_warning=leg))
    monkeypatch.setattr(
        server, "get_access_token", lambda: _Tok(server._MACHINE_CLIENT_ID, {"kind": "root"})
    )
    assert server.recall("widget")["warnings"] == [NO_CREDENTIAL_NOTICE, leg]
    assert server.recall_full_turns("widget")["warnings"] == [NO_CREDENTIAL_NOTICE, leg]


@_needs_db
def test_mcp_oauth_sign_in_without_a_grant_gets_the_sign_in_remedy(server, monkeypatch):
    monkeypatch.setattr(server, "_IDENTITY_CLAIMS", ("preferred_username",))
    monkeypatch.setattr(
        server, "get_access_token", lambda: _Tok("some-connector", {"preferred_username": "x"})
    )
    assert server.recall("widget")["warnings"] == [SIGN_IN_NOTICE]


@_needs_db
def test_mcp_without_a_verified_credential_stays_silent(server, monkeypatch):
    """No token context at all (open server, stdio): nothing authenticated, so nothing to
    fix with a credential, and the response keeps its pre-notice shape."""
    monkeypatch.setattr(server, "get_access_token", lambda: None)
    for out in _call_all_tools(server):
        assert "warnings" not in out, out


def _client(s):
    from starlette.testclient import TestClient

    return TestClient(s.mcp.http_app())


def _h(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@_needs_db
@pytest.mark.parametrize(
    "token,expected",
    [
        (_ROOT, NO_CREDENTIAL_NOTICE),
        ("empty-dev-tok", RESTRICTED_EMPTY_NOTICE),
        ("scoped-dev-tok", None),
        ("full-dev-tok", None),
    ],
)
def test_http_recall_explains_a_serve_nothing_verdict(server, token, expected):
    with _client(server) as c:
        r = c.post("/recall", json={"query": "widget"}, headers=_h(token))
    assert r.status_code == 200
    body = r.json()
    if expected is None:
        assert "warnings" not in body
    else:
        assert body["warnings"] == [expected]


@_needs_db
@pytest.mark.parametrize(
    "token,expected",
    [
        (_ROOT, NO_CREDENTIAL_NOTICE),
        ("empty-dev-tok", RESTRICTED_EMPTY_NOTICE),
        ("scoped-dev-tok", None),
        ("full-dev-tok", None),
    ],
)
def test_board_puts_the_notice_at_the_top(server, token, expected):
    with _client(server) as c:
        r = c.get("/context", headers=_h(token))
    assert r.status_code == 200
    body = r.json()
    lines = body["text"].splitlines()
    assert lines[0].startswith("[Synapse board")
    if expected is None:
        assert "warnings" not in body
        assert not any(line.startswith("WARNING:") for line in lines)
        assert "episodes across" in lines[1]  # banner keeps its slot
    else:
        assert body["warnings"] == [expected]
        assert lines[1] == f"WARNING: {expected}"  # one line, right under the header
        assert "episodes across" in lines[2]


@_needs_db
def test_unauthenticated_requests_are_refused_before_any_notice(server):
    """No oracle: a request without a valid credential never reaches a verdict."""
    with _client(server) as c:
        for headers in ({}, _h("not-a-real-token")):
            r = c.get("/context", headers=headers)
            assert r.status_code == 401 and "warnings" not in r.json()
            r = c.post("/recall", json={"query": "widget"}, headers=headers)
            assert r.status_code == 401 and "warnings" not in r.json()


@_needs_db
def test_open_server_serves_its_empty_answer_without_a_notice(monkeypatch, devices):
    """An open server authenticates nobody, so every caller is unknown and no credential
    would change that. The notice would be wrong advice there; the startup banner owns
    that message instead."""
    s = _reload(monkeypatch, "")
    with _client(s) as c:
        for headers in ({}, _h(_ROOT)):
            board = c.get("/context", headers=headers).json()
            assert board["trust"] == "restricted"
            assert "warnings" not in board and "WARNING:" not in board["text"]
            assert "warnings" not in c.post("/recall", json={"query": "q"}, headers=headers).json()
