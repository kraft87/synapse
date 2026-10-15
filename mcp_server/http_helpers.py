# mypy: ignore-errors
# Deliberately untyped route-handler helper module (mirrors skill_sync_routes/config_sync_routes,
# already in the mypy pre-commit exclude via mcp_server/*).
"""Shared plain-HTTP route helpers for the machine-token custom routes.

One place for the error-envelope + auth/JSON/threadpool boilerplate the route modules used
to hand-roll. The contract's error body is ``{"status": "error", "detail": ...}`` at some
status; ``unauthorized()`` is the 401 special case, ``route`` is the decorator that applies
all of it, and ``guarded_json`` is the skill/config sync routes' lenient-body variant.

Import discipline: this module imports ONLY starlette + stdlib, never any other mcp_server
module, so it can be imported from all of them without a cycle.
"""

from __future__ import annotations

import functools
import inspect
import logging
import re
from collections.abc import Callable
from typing import Any

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

_ORIGIN_RE = re.compile(r"^https?://(?:[A-Za-z0-9.\-]+|\[[0-9A-Fa-f:]+\])(?::\d{1,5})?$")


def err(detail: str, status: int) -> JSONResponse:
    """The shared error envelope: ``{"status": "error", "detail": detail}`` at ``status``."""
    return JSONResponse({"status": "error", "detail": detail}, status_code=status)


def unauthorized() -> JSONResponse:
    """The 401 every machine-token-gated route returns when the token is absent/wrong."""
    return err("unauthorized", 401)


def status_conflict(cid: int, current: str, expected: tuple[str, ...]) -> dict:
    """A proposal decision found its row in a status the decision does not act on (a stale
    pane, a double click, a CLI that got there first). Nothing changed; the dashboard
    answers it with a 409, the CLI prints the detail."""
    wanted = " / ".join(repr(s) for s in expected)
    return {"status": "conflict", "id": cid, "detail": f"proposal is '{current}', not {wanted}"}


def _iso(dt: Any) -> str | None:
    """A datetime's ISO string, or None when it's absent."""
    return dt.isoformat() if dt is not None else None


class RouteError(Exception):
    """Raised inside a ``route`` handler to answer the error envelope: ``detail`` at
    ``status`` (400 unless the handler says otherwise), for a request it refuses."""

    def __init__(self, detail: str, status: int = 400) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status = status


def route(
    gate: Callable[[Request], bool] | None,
    *,
    body: bool = False,
    not_found: str | None = None,
    label: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[[Request], Any]]:
    """Decorator for a ``custom_route`` handler: the gate, the body and the error envelope.

    The wrapper checks ``gate(request)`` once (401; ``None`` for a route that authenticates
    by other means), parses a JSON object body when ``body`` is set (400 for invalid JSON or
    anything but an object) and calls ``fn(request)`` / ``fn(request, body)``. A plain
    function runs in the threadpool, so it may block on the database and resolve a caller's
    trust; an ``async def`` runs on the event loop and offloads its own blocking calls.

    What ``fn`` returns goes out as the 200 JSON body, except a ready ``Response`` (a
    streamed or non-200 answer) which passes through, and ``None`` when ``not_found`` is
    set, which is the 404 with that detail. ``RouteError`` is the handler's own refusal;
    any other exception is logged under ``label`` (the handler's name unless given) and
    answered as a 500 carrying the first 200 characters of its message."""

    def decorate(fn: Callable[..., Any]) -> Callable[[Request], Any]:
        offload = not inspect.iscoroutinefunction(fn)
        name = label or fn.__name__

        @functools.wraps(fn)
        async def handler(request: Request) -> Response:
            if gate is not None and not gate(request):
                return unauthorized()
            args: tuple[Any, ...] = (request,)
            if body:
                try:
                    payload = await request.json()
                except Exception:
                    return err("invalid JSON body", 400)
                if not isinstance(payload, dict):
                    return err("body must be a JSON object", 400)
                args = (request, payload)
            try:
                out = await (run_in_threadpool(fn, *args) if offload else fn(*args))
            except RouteError as e:
                return err(e.detail, e.status)
            except Exception as e:
                logger.exception("%s failed", name)
                return err(str(e)[:200], 500)
            if out is None and not_found is not None:
                return err(not_found, 404)
            return out if isinstance(out, Response) else JSONResponse(out)

        return handler

    return decorate


async def guarded_json(
    request: Request,
    authorized: Callable[[Request], bool],
    work: Callable[[Any], Any],
    *,
    label: str = "route",
) -> JSONResponse:
    """The skill/config sync routes' form of ``route``: gate (401), parse the JSON body
    (400 on failure) and run ``work(body)`` in a threadpool (500 + logged on exception).

    Unlike ``route(body=True)`` the body may be any JSON value: those routes' ``work`` answers
    a malformed body in its own words (a 200 ``status: error`` the CLIs print). ``label``
    names the route in the 500 log line (e.g. 'skill route' / 'config route'). ``work`` may
    return a ready ``JSONResponse`` (e.g. a 403) and it is passed through."""
    if not authorized(request):
        return unauthorized()
    try:
        body = await request.json()
    except Exception:
        return err("invalid JSON body", 400)
    try:
        out = await run_in_threadpool(work, body)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("%s failed", label)
        return err(str(exc)[:200], 500)
    if isinstance(out, JSONResponse):
        return out
    return JSONResponse(out)


def request_origin(request: Request) -> str:
    """The origin this request was addressed to, as the client sees it: the first entry of
    X-Forwarded-Proto / -Host when a proxy set them, validated, else the request's own."""
    proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip()
    host = request.headers.get("x-forwarded-host", "").split(",")[0].strip()
    origin = f"{proto or request.url.scheme}://{host or request.headers.get('host', '')}"
    if _ORIGIN_RE.match(origin):
        return origin
    return str(request.base_url).rstrip("/")
