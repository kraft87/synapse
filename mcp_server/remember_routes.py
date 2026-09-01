"""Spooled-remember replay route — the durable seam for memory writes made while MCP is down.

Background (2026-08-25): the MCP transport's OAuth failed mid-session, so ``remember()``
could not be called at all — a user correction was silently lost for two days while the
poisoned note it was meant to fix kept re-injecting at every session start. The plugin's
plain-HTTP lane (bearer token, /ingest) worked the entire time; only MCP was broken. So the
fix is a local spool in the plugin plus THIS route: a machine-token-gated HTTP path that
performs the exact same write as the MCP tool, over the transport that stayed up.

Route (POST, machine-token gated):
  /remember/spool  {"probe": true}                          -> {"status":"ok","probe":true}
  /remember/spool  {intent_id, hook, body, type?, project?, session_id?, content?}
                                                            -> {"status":"ok", note_id, ...}

``remember_fn`` is ``mcp_server.server.remember`` itself — injected rather than imported so
this module has no cycle with server.py, and so the spooled write CANNOT drift from the tool
it replays (episode archive + extraction enqueue + note reconcile, one code path).

Idempotency (schema 052). The client generates ``intent_id`` once, when the intent is
spooled, and keeps re-posting it until the server confirms; the flush only drops the local
line after that confirm. So a flush that dies in the confirm window re-posts. The route
CLAIMS the id before writing anything:

  * claim wins            -> do the write, mark the row 'done' with its note/episode ids;
  * row already 'done'    -> return the recorded ids with outcome 'duplicate', write nothing;
  * row still 'pending'   -> a previous attempt crashed mid-write; re-run it. reconcile_note
                             is convergent (the identical hook resolves onto the same live
                             note as an update), so re-running cannot fork a second note,
                             whereas skipping could lose the note entirely.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import psycopg
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse

from mcp_server.http_helpers import err, unauthorized

logger = logging.getLogger(__name__)

_MAX_INTENT_ID = 200


class _SchemaMissing(Exception):
    """schema/052 not applied on this deployment — reported as 503, never as success."""


def _claim(db_url: str, intent_id: str, hook: str | None) -> dict[str, Any] | None:
    """Reserve ``intent_id``. Returns None when this caller won the claim (proceed with the
    write), or the existing row when someone already claimed it (see the module docstring)."""
    conn = psycopg.connect(db_url, autocommit=True)
    try:
        row = conn.execute(
            "INSERT INTO remember_intents (intent_id, hook) VALUES (%s, %s) "
            "ON CONFLICT (intent_id) DO NOTHING RETURNING intent_id",
            (intent_id, (hook or "")[:200] or None),
        ).fetchone()
        if row is not None:
            return None
        prior = conn.execute(
            "SELECT status, note_id, episode_id, outcome FROM remember_intents WHERE intent_id = %s",
            (intent_id,),
        ).fetchone()
        if prior is None:  # raced with a delete — treat as ours
            return None
        return {
            "status": prior[0],
            "note_id": prior[1],
            "episode_id": prior[2],
            "outcome": prior[3],
        }
    except psycopg.errors.UndefinedTable as e:
        raise _SchemaMissing from e
    finally:
        conn.close()


def _complete(db_url: str, intent_id: str, result: dict[str, Any]) -> None:
    """Record the write's ids against the claim. Only a 'done' row short-circuits a replay."""
    conn = psycopg.connect(db_url, autocommit=True)
    try:
        conn.execute(
            "UPDATE remember_intents SET status = 'done', note_id = %s, episode_id = %s, "
            "outcome = %s, completed_at = now() WHERE intent_id = %s",
            (result.get("note_id"), result.get("episode_id"), result.get("outcome"), intent_id),
        )
    except psycopg.errors.UndefinedTable as e:  # pragma: no cover - claim would have raised
        raise _SchemaMissing from e
    finally:
        conn.close()


def _validate(body: dict[str, Any]) -> str | None:
    """Reject a payload the route cannot faithfully replay. None means valid."""
    intent_id = str(body.get("intent_id") or "").strip()
    if not intent_id:
        return "intent_id required — it is the idempotency key for the replay"
    if len(intent_id) > _MAX_INTENT_ID:
        return f"intent_id too long (max {_MAX_INTENT_ID})"
    has_pair = bool(str(body.get("hook") or "").strip()) and bool(
        str(body.get("body") or "").strip()
    )
    if not has_pair and not str(body.get("content") or "").strip():
        return "provide hook + body (preferred) or content (legacy) — same forms as remember()"
    return None


def register(
    mcp: Any,
    db_url: str,
    authorized: Callable[[Request], bool],
    remember_fn: Callable[..., Any],
    caller_surface: Callable[[Request], str | None] | None = None,
) -> None:
    """``caller_surface`` resolves the CALLING DEVICE from its credential (schema 054).
    Supplied, it outranks the ``surface`` field in the body — a self-reported hostname
    must not decide a note's audience when the bearer already says who is writing."""
    if not db_url:
        logger.info("remember spool route disabled (no DB_URL)")
        return

    @mcp.custom_route("/remember/spool", methods=["POST"])  # type: ignore[misc]
    async def remember_spool(request: Request) -> JSONResponse:
        """Replay one spooled remember() intent. Idempotent on the client's intent_id.

        Not wrapped in ``guarded_json``: that helper runs its work in a threadpool, and the
        write here is the async ``remember`` coroutine (which manages its own thread hop for
        the blocking DB/LLM legs). Same auth + error envelope, hand-rolled."""
        if not authorized(request):
            return unauthorized()
        try:
            body = await request.json()
        except Exception:
            return err("invalid JSON", 400)
        if not isinstance(body, dict):
            return err("invalid JSON", 400)

        # Cheap liveness+auth probe for the plugin's SessionStart step: it must be able to
        # tell "server reachable and my token works" from "spool locally" WITHOUT writing.
        if body.get("probe"):
            return JSONResponse({"status": "ok", "probe": True})

        detail = _validate(body)
        if detail:
            return err(detail, 400)
        intent_id = str(body["intent_id"]).strip()

        try:
            prior = await run_in_threadpool(_claim, db_url, intent_id, body.get("hook"))
        except _SchemaMissing:
            return err("remember_intents missing (apply schema/052)", 503)
        except Exception as e:
            logger.warning("remember spool claim failed: %s", e)
            return err(str(e)[:200], 500)

        if prior is not None and prior.get("status") == "done":
            return JSONResponse(
                {
                    "status": "ok",
                    "intent_id": intent_id,
                    "outcome": "duplicate",
                    "note_id": prior.get("note_id"),
                    "episode_id": prior.get("episode_id"),
                    "prior_outcome": prior.get("outcome"),
                }
            )

        kwargs: dict[str, Any] = {
            "type": body.get("type") or "project",
            "project": body.get("project") or None,
            "session_id": body.get("session_id") or None,
            # A spooled write must classify the same way the live tool would have: the
            # spool exists because the MCP transport was down, not because the note came
            # from a different device. Prefer the CREDENTIAL's surface; the body field is
            # the pre-054 fallback. Neither -> derived by project rule, still fail-closed.
            "surface": (
                (caller_surface(request) if caller_surface else None) or body.get("surface") or None
            ),
            "audience": body.get("audience") or None,
        }
        if str(body.get("hook") or "").strip() and str(body.get("body") or "").strip():
            kwargs["hook"] = body["hook"]
            kwargs["body"] = body["body"]
        if str(body.get("content") or "").strip():
            kwargs["content"] = body["content"]

        try:
            result = await remember_fn(**kwargs)
        except Exception as e:
            logger.warning("remember spool write failed: %s", e)
            return err(str(e)[:200], 500)

        if not isinstance(result, dict) or result.get("status") != "ok":
            # A rejected payload (bad type, blank body) is the client's problem, not a
            # transport failure: 400 so the flush drops the line instead of retrying forever.
            reason = (result or {}).get("detail") if isinstance(result, dict) else "write refused"
            return err(str(reason)[:200], 400)

        try:
            await run_in_threadpool(_complete, db_url, intent_id, result)
        except Exception as e:  # pragma: no cover - defensive
            # The note IS written; failing here would make the client retry and (thanks to
            # the still-'pending' claim) re-run the convergent write. Log and report success.
            logger.warning("remember spool complete-mark failed for %s: %s", intent_id, e)

        return JSONResponse({"status": "ok", "intent_id": intent_id, **result})
