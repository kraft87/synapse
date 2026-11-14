"""What a ``GET /context`` reply holds (mcp_server/board.py): the parts a request names.

A session start is one request: ``?parts=board,preferences`` returns the board as
``text`` and the rendered preferences block as ``preferences``. A client from before the
parameter sends none and gets the board alone, as it always did.

No database: the two builders are replaced, so these are the route's own rules. What the
builders render is in tests/test_board.py and tests/test_preferences_routes.py.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastmcp import FastMCP
from starlette.testclient import TestClient

from ingestion.surfaces import FULL_TRUST, UNKNOWN_SURFACE
from mcp_server import board

_TOKEN = "test-parts-token"
_AUTH = {"Authorization": f"Bearer {_TOKEN}"}
_BOARD = "[Synapse board]\n3 episodes"
_PREFS = "[Synapse preferences]\n  - (rule) User wants short answers"


@pytest.mark.parametrize(
    ("raw", "parts"),
    [
        (None, {"board"}),  # a client from before the parameter
        ("board", {"board"}),
        ("preferences", {"preferences"}),
        ("board,preferences", {"board", "preferences"}),
        (" preferences , board ", {"board", "preferences"}),
        ("board,later", {"board"}),  # a name this server does not know is not an error
        ("later", set()),
        ("", set()),
    ],
)
def test_the_parts_a_request_asks_for(raw, parts):
    assert board.wanted_parts(raw) == parts


class _Built:
    """Stand-ins for the two builders, recording what each was asked."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.boards: list[tuple[Any, ...]] = []
        self.prefs: list[Any] = []
        self.board_error: Exception | None = None
        self.prefs_error: Exception | None = None
        monkeypatch.setattr(board, "build_board", self._board)
        monkeypatch.setattr(board, "top_block", self._prefs)

    def _board(self, db_url, project, surface, trust, notice) -> dict[str, Any]:
        self.boards.append((project, trust))
        if self.board_error is not None:
            raise self.board_error
        return {"status": "ok", "trust": trust.trust, "text": _BOARD, "note_ids": [1, 2]}

    def _prefs(self, db_url, trust) -> str:
        self.prefs.append(trust)
        if self.prefs_error is not None:
            raise self.prefs_error
        return _PREFS


@pytest.fixture()
def built(monkeypatch) -> _Built:
    return _Built(monkeypatch)


def _client(resolve_trust=None) -> TestClient:
    def authorized(request):
        return request.headers.get("authorization", "") == f"Bearer {_TOKEN}"

    app = FastMCP("test-parts")
    board.register(app, "postgresql://unused", authorized, resolve_trust=resolve_trust)
    return TestClient(app.http_app())


def test_without_the_parameter_the_reply_is_the_board_alone(built):
    with _client() as client:
        body = client.get("/context?project=alpha", headers=_AUTH).json()
    assert body == {"status": "ok", "trust": "restricted", "text": _BOARD}
    assert built.boards == [("alpha", UNKNOWN_SURFACE)] and built.prefs == []


def test_both_parts_come_in_one_reply(built):
    with _client(lambda _request: FULL_TRUST) as client:
        body = client.get("/context?project=alpha&parts=board,preferences", headers=_AUTH).json()
    assert body == {"status": "ok", "trust": "full", "text": _BOARD, "preferences": _PREFS}
    # Both are built for the verdict the credential resolved to.
    assert built.boards == [("alpha", FULL_TRUST)] and built.prefs == [FULL_TRUST]


def test_the_preferences_alone_build_no_board(built):
    with _client() as client:
        body = client.get("/context?parts=preferences", headers=_AUTH).json()
    assert body == {"status": "ok", "trust": "restricted", "preferences": _PREFS}
    assert built.boards == [] and built.prefs == [UNKNOWN_SURFACE]


def test_no_known_part_builds_nothing(built):
    with _client() as client:
        body = client.get("/context?parts=later", headers=_AUTH).json()
    assert body == {"status": "ok", "trust": "restricted"}
    assert built.boards == [] and built.prefs == []


def test_preferences_that_cannot_be_built_do_not_cost_the_board(built):
    built.prefs_error = RuntimeError("no such table")
    with _client() as client:
        reply = client.get("/context?parts=board,preferences", headers=_AUTH)
    assert reply.status_code == 200
    assert reply.json() == {
        "status": "ok",
        "trust": "restricted",
        "text": _BOARD,
        "preferences": "",
    }


def test_a_board_that_cannot_be_built_is_an_error_whatever_else_was_asked(built):
    built.board_error = RuntimeError("no such table")
    with _client() as client:
        reply = client.get("/context?parts=board,preferences", headers=_AUTH)
    assert reply.status_code == 500 and built.prefs == []


def test_no_part_is_served_without_the_token(built):
    with _client() as client:
        assert client.get("/context?parts=preferences").status_code == 401
    assert built.boards == [] and built.prefs == []
