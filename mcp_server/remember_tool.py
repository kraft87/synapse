"""remember tool registration with explicit runtime dependencies."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from fastmcp import FastMCP

from ingestion.surfaces import SurfaceTrust

logger = logging.getLogger(__name__)


def register(
    mcp: FastMCP,
    db_url: Callable[[], str],
    _get_recall: Callable[[], Any],
    _caller_trust: Callable[[str | None], SurfaceTrust],
    _notes_deps: Callable[[], tuple],
    _derive_hook: Callable[[str], str],
) -> Any:
    @mcp.tool()
    async def remember(
        content: str | None = None,
        hook: str | None = None,
        body: str | None = None,
        type: str = "project",
        project: str | None = None,
        session_id: str | None = None,
        surface: str | None = None,
        audience: str | None = None,
    ) -> dict:
        """Write to the user's curated long-term memory: reconciles a NOTE into the
        explicit notes store (deduped against the live set, superseded on
        contradiction) AND archives the text as an episode with knowledge-graph
        extraction.

        WHEN the user states a durable fact, preference, or decision, corrects
        something you had wrong, or explicitly asks you to remember → call this.
        If you are about to reply "noted" / "got it" / "I'll remember that," call
        this FIRST, then reply. Also bank a 2-3 sentence summary of what was
        decided before a session ends or is cleared.

        Do NOT call for transient task state, to restate something already stored,
        or to save your own speculation or an unconfirmed plan. Routine
        conversation is ingested automatically.

        Form — pass hook + body (+ type). `hook` is the one-line index entry
        (target ~120 chars, hard cap 200, whitespace collapsed); `body` is the
        full note, fetched on demand by id. Passing either selects this form and
        both must then be non-empty — a lone hook never falls back to legacy.
        The legacy content-only form stays for compatibility (hook derived from
        the first sentence, type 'project'); content alongside a full hook + body
        pair becomes the archived episode text.

        Type semantics — user: durable facts about the user, and feedback:
        corrections to agent behavior (both global — every session); project:
        scoped to `project`, staleness-managed (the default);
        reference: pointers to canonical sources.

        The write contract — a good note:
        - States decisions WITH reasons: "chose X because Y" survives; "merged
          the fix" is noise.
        - Is DECLARATIVE, not imperative: "User prefers concise responses",
          never "Always respond concisely" — imperative memory re-reads as a
          standing directive.
        - Has no PR numbers, commit SHAs, "phase N done", file counts — anything
          week-stale belongs in the episode archive (written automatically),
          never in a note.
        - Uses absolute dates ("2026-07-12"), never "yesterday" or "last week".
        - Is self-contained: the body must stand alone months later.

        Args:
            content: Legacy form — full text to remember (hook and type derived).
            hook: Preferred form — one-line note index entry (target ~120 chars).
            body: Preferred form — the full, self-contained note text.
            type: Note type — 'user' | 'feedback' | 'project' (default) | 'reference'.
            project: Optional project slug (scopes 'project' notes and the episode).
            session_id: Optional session ID to attach the episode to.
            surface: DEPRECATED and ignored — the server identifies the calling
                device from its own credential. Never set it.
            audience: 'personal' (default) or 'work-safe' — who may be SERVED this
                note later. Leave unset unless the user says where it may appear;
                the server derives it from the calling host and the project.
        """
        import time as _time
        import uuid as _uuid

        import anyio.to_thread

        from ingestion.db import Database
        from ingestion.models import Episode, ExtractionItem
        from ingestion.notes import _VALID_TYPES, reconcile_note
        from ingestion.surfaces import AUDIENCES

        structured = hook is not None or body is not None
        if not structured and not (content or "").strip():
            return {
                "status": "error",
                "detail": "provide hook + body (preferred) or content (legacy)",
            }
        if type not in _VALID_TYPES:
            return {
                "status": "error",
                "detail": f"invalid type {type!r} — expected one of {_VALID_TYPES}",
            }
        if audience is not None and audience not in AUDIENCES:
            return {
                "status": "error",
                "detail": f"invalid audience {audience!r} — expected one of {AUDIENCES}",
            }

        if structured:
            # Passing either hook or body commits to the structured form — a lone
            # hook (even with content also present) must NOT silently fall back to
            # legacy, which would discard the caller's hook. Board lines are
            # single-line by contract: collapse whitespace runs/newlines BEFORE the
            # hard cap so a multi-line hook can't smuggle newlines under it.
            note_hook = " ".join((hook or "").split())[:200]  # hard cap; targets ~120
            note_body = (body or "").strip()
            if not note_hook or not note_body:
                missing = "hook" if not note_hook else "body"
                return {
                    "status": "error",
                    "detail": f"structured form requires hook + body — {missing} is missing or blank",
                }
            note_type = type
            # content alongside a full pair serves as the archived episode text.
            ep_content = (content or "").strip() or f"{note_hook}\n\n{note_body}"
        else:
            ep_content = (content or "").strip()  # non-empty per the form check above
            note_hook = _derive_hook(ep_content)
            note_body = ep_content
            note_type = "project"

        sid = session_id or str(_uuid.uuid4())
        # Same resolution the serving tools do, so a device (or OAuth identity) registered
        # as a RESTRICTED surface writes notes it can still read back. Resolved here, on the
        # event loop, rather than inside _work(): the token context belongs to the request,
        # and _work runs on a worker thread.
        caller_trust = _caller_trust(surface)

        def _work() -> dict:
            t0 = _time.perf_counter()
            # Use a dedicated short-lived connection for writes — keeps the shared
            # recall engine's read connection clean and avoids transaction leakage.
            db = Database(db_url())
            try:
                existing = db.get_session_episodes(sid)
                seq = (max(e["sequence"] for e in existing) + 1) if existing else 1

                ep = Episode(
                    session_id=sid,
                    sequence=seq,
                    project=project,
                    content=ep_content,
                    source="manual",
                )
                episode_id = db.upsert_episode(ep)

                db.enqueue_extraction(
                    ExtractionItem(
                        episode_id=episode_id,
                        session_id=sid,
                        content=ep_content,
                        content_type="manual",
                        project=project,
                    )
                )

                # Audience precedence rule 2 fires only for a REGISTERED restricted surface
                # (`known`). An unknown surface restricts what this caller READS, but it must
                # never widen a WRITE — defaulting an unrecognised credential's notes to
                # work-safe would turn a fail-closed read rule into a leak.
                caller_restricted = caller_trust.known and caller_trust.restricted

                embedder, llm = _notes_deps()
                res = reconcile_note(
                    db,
                    embedder,
                    llm,
                    hook=note_hook,
                    body=note_body,
                    type=note_type,
                    project=project,
                    source_ref=f"ep:{episode_id}",
                    audience=audience,
                    caller_restricted=caller_restricted,
                )
            finally:
                db.close()

            _get_recall().record_event(
                "remember",
                source="mcp-tool",
                ms_total=(_time.perf_counter() - t0) * 1000.0,
                served_ids={
                    "note": res["note_id"],
                    "outcome": res["outcome"],
                    "type": note_type,
                    "audience": res.get("audience"),
                },
            )
            return {
                "status": "ok",
                "note_id": res["note_id"],
                "outcome": res["outcome"],
                "episode_id": episode_id,
                "session_id": sid,
                # None on a restatement that preserved the stored tier — the caller can tell
                # "this write classified the note" from "this write left it alone".
                "audience": res.get("audience"),
            }

        # reconcile_note does blocking DB I/O + possibly a sync LLM call that runs
        # asyncio.run() internally — asyncio.run() cannot be called from a running
        # event loop, so it must live on a worker thread, never on FastMCP's loop.
        return await anyio.to_thread.run_sync(_work)

    return remember
