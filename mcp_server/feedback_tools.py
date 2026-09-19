"""feedback tools registration with explicit runtime dependencies."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

import logfire
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from mcp_server.http_helpers import err, unauthorized

logger = logging.getLogger(__name__)


_FEEDBACK_ID_RE = re.compile(r"^(?:[entw]:\d+|f:[0-9a-fA-F-]{8,})$")


def register(
    mcp: FastMCP,
    db_url: Callable[[], str],
    _machine_authorized: Callable[[Request], bool],
) -> tuple[Any, ...]:
    def _feedback_ids_error(field: str, ids: list[str] | None) -> str | None:
        """Validation error for recall_feedback's helpful/noise lists, or None if valid.

        Accepts None or a list of served-id strings — "e:N" episode, "n:N" note,
        "f:<uuid>" fact, "t:N" timeline, "w:N" web — exactly as recall() serves them.
        Anything else is rejected."""
        if ids is None:
            return None
        if not isinstance(ids, list):
            return f"{field} must be a list of served ids like ['e:123', 'f:<uuid>']"
        bad = [i for i in ids if not (isinstance(i, str) and _FEEDBACK_ID_RE.fullmatch(i))]
        if bad:
            return (
                f"{field} contains invalid ids {bad!r} — expected recall-served ids: "
                '"e:N" episode, "n:N" note, "f:<uuid>" fact, "t:N" timeline, "w:N" web'
            )
        return None

    def _file_recall_feedback(
        query: str,
        helpful: list[str] | None,
        noise: list[str] | None,
        missing: str | None,
        found_via: str | None,
        comment: str | None,
        session_id: str | None,
        project: str | None,
    ) -> dict:
        """Validate + INSERT one recall_feedback row — shared by the MCP tool and
        POST /feedback. Returns the tool-shaped result dict (never raises for bad
        input; DB errors propagate to the caller's boundary)."""
        from ingestion.db import Database

        q = (query or "").strip()
        if not q:
            return {
                "status": "error",
                "detail": "missing 'query' — pass the recall query being rated",
            }
        for field, ids in (("helpful", helpful), ("noise", noise)):
            err = _feedback_ids_error(field, ids)
            if err:
                return {"status": "error", "detail": err}

        db = Database(db_url())
        try:
            feedback_id = db.insert_recall_feedback(
                query=q,
                helpful=helpful or [],
                noise=noise or [],
                missing=missing,
                # One short token ("full_turns", "filesystem", ...) — whitespace-collapsed
                # and capped, free text by design (see schema 048).
                found_via=" ".join((found_via or "").split())[:60] or None,
                # The model-facing name is `comment` (general free-text spot); the
                # column stays `note` from schema 046 — no migration for a rename.
                note=comment,
                session_id=session_id,
                project=project,
            )
        finally:
            db.close()
        return {"status": "ok", "feedback_id": feedback_id}

    @mcp.tool()
    def recall_feedback(
        query: str,
        helpful: list[str] | None = None,
        noise: list[str] | None = None,
        missing: str | None = None,
        found_via: str | None = None,
        comment: str | None = None,
        session_id: str | None = None,
        project: str | None = None,
    ) -> dict:
        """Report retrieval quality after a recall() whose results you used: which
        served ids helped, which were noise, plus free-text comment on the serving.

        AFTER acting on a recall's results, call this ONCE with that recall's
        query string. `helpful` = served ids that were load-bearing; `noise` =
        served ids that were irrelevant or distracting. Everything else goes in
        `comment`, free text: too much dumped at once, load-bearing hit buried,
        wrong granularity or ordering, a misleading passage, an improvement idea.

        `missing` is the exception, not a per-report field — most recalls lack
        nothing; then OMIT it (never file "nothing missing"). Set it only when
        you can name the specific content you needed AND have concrete reason to
        believe memory holds it (a past session covered it, or the user said it
        was stored). Never-discussed content is not a miss; weak results belong
        in `comment`. A `missing` without `found_via` is unusable, so always pair
        them: "full_turns" / "fetch" / "another_recall" (memory HAD it — a
        serving miss, the highest-value signal) vs "filesystem" / "live_system" /
        "web" / "user" (memory never had it) vs "nowhere" (still unresolved).

        Rate any served id verbatim — "e:N" episodes, "f:<uuid>" facts (and
        superseded_facts), "t:N" timeline, "w:N" web — plus
        "n:N" note ids from the session-start board or fetch(): WHEN a board
        note shaped your answer (or misled it), rate it too. A comment-only
        report is still valuable when the serving itself was the problem.

        This is offline labeled data (eval goldens, reranker tuning); it never
        changes live ranking, so honest negatives are safe and wanted. Do NOT
        file more than one report per recall query, do NOT rate results you
        never used, and do NOT invent ids — report only ids actually served to
        you by recall, the board, or fetch.

        Args:
            query: The recall query being rated, verbatim.
            helpful: Served ids that were load-bearing ("e:123", "f:<uuid>", "w:7").
            noise: Served ids that were irrelevant or distracting.
            missing: The specific content you needed but were not served — omit
                entirely unless you can name it AND believe memory holds it.
                Always pair with `found_via`.
            found_via: Where the missing info turned up ("full_turns"/"fetch"/
                "another_recall" = memory had it; "filesystem"/"live_system"/
                "web"/"user" = it never did; "nowhere" = unresolved). Set it
                whenever `missing` is set.
            comment: Free text for anything the other fields can't say — serving
                volume, ordering, presentation, misleading results, improvement ideas.
            session_id: Optional session id for grouping reports.
            project: Optional project slug the recall was scoped to.
        """
        with logfire.span("mcp.recall_feedback {query!r}", query=query[:80], project=project):
            return _file_recall_feedback(
                query, helpful, noise, missing, found_via, comment, session_id, project
            )

    @mcp.custom_route("/feedback", methods=["POST"])
    async def feedback_http(request: Request) -> JSONResponse:
        """Plain-HTTP recall_feedback for non-MCP callers — the /recall sibling.

        Hooks and the dashboard talk to /recall over plain HTTP (no MCP client), so
        the labeled-feedback write needs the same seam or those callers could never
        file a report. Same validation + insert as the recall_feedback tool
        (_file_recall_feedback); same machine-token gate; fail-soft like /ingest.

        Body: {"query": str, "helpful"?: [..], "noise"?: [..], "missing"?: str,
               "found_via"?: str, "comment"?: str, "session_id"?: str, "project"?: str}.
        "note" is accepted as a legacy alias for "comment" (pre-1.0.1 callers).
        """
        if not _machine_authorized(request):
            return unauthorized()

        from starlette.concurrency import run_in_threadpool

        try:
            body = await request.json()
        except Exception:
            return err("invalid JSON body", 400)

        def _work() -> dict:
            return _file_recall_feedback(
                query=body.get("query") or "",
                helpful=body.get("helpful"),
                noise=body.get("noise"),
                missing=body.get("missing"),
                found_via=body.get("found_via"),
                comment=body.get("comment") or body.get("note"),
                session_id=body.get("session_id"),
                project=body.get("project"),
            )

        try:
            with logfire.span("http.feedback"):
                out = await run_in_threadpool(_work)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("http feedback failed")
            return err(str(exc)[:200], 500)
        return JSONResponse(out, status_code=200 if out.get("status") == "ok" else 400)

    return _feedback_ids_error, _file_recall_feedback, recall_feedback, feedback_http
