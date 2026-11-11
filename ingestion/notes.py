"""Reconcile one explicit note into the notes store (schema 041).

Notes are the EXPLICIT-memory store behind the always-injected board: one row per
curated memory, a short ``hook`` (the board line; also the embed target for dedup
KNN) plus a self-contained ``body`` fetched on demand by id. This module owns the
single write path — the remember() rework and the seed importer both call
:func:`reconcile_note` so dedup/supersession semantics can't drift between callers.

Reconciliation, keyed on cosine similarity of the new HOOK to the live notes of ONE
tier at a time (``notes.audience``: 'personal' or 'work-safe'). The rule across tiers:
a write never edits a live note of another tier, and a restricted caller never sees,
edits or retires a personal note. Which tier is searched:
  - a REGISTERED restricted caller looks only at work-safe notes, whatever ``audience``
    it passed (the personal tier is never queried for it);
  - a full-trust caller looks first in the tier its write will be stored under, then in
    the other tier.
A candidate is the top hit of a tier when its sim >= SYNAPSE_NOTES_SIM_MATCH (default
0.80) AND it has the same type. Then one small LLM confirm call ({"relation":
"same"|"contradicts"}):
  - "same" in the write's own tier -> UPDATE the existing note in place (refresh hook/
    body/embedding, bump updated_at — restatements don't pile up); the stored tier is
    kept
  - "contradicts" -> INSERT the new note at its own tier + supersede the old one
    (lineage kept), whichever tier the old one was in
  - "same" in the OTHER tier -> leave that note alone and INSERT the new one at its own
    tier, so the same wording can hold a twin in each tier; only an explicit
    ``audience`` argument updates the other-tier note in place and moves it to that tier
  - no candidate in either tier -> INSERT a new live note.

Fail-open, always toward "same": the kill switch (SYNAPSE_NOTES_CONFIRM=0) and any
LLM failure both collapse the confirm to "same" — an over-eager update loses a
nuance; a missed update duplicates a board line forever. Across tiers that same answer
means a new note in the write's own tier, never an edit of the other one.

Keyless dev/test (no embedder, or the embed call fails) degrades to a straight insert
with NULL embedding — dedup is silently skipped, never a hard failure (timeline
precedent).

Threading note: this function is synchronous and calls the LLM through the sync
``messages.create`` surface (which may run ``asyncio.run`` internally, e.g. the
Agent-SDK client). Inside an async server (FastMCP handlers) it MUST be run in a
worker thread (``anyio.to_thread.run_sync``), never on the event loop —
``asyncio.run()`` cannot be called from a running event loop.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, cast

from ingestion.llm_client import MalformedResponseError, parse_with_retry, stage_model
from ingestion.surfaces import AUDIENCES, RESTRICTED_AUDIENCE, derive_audience

logger = logging.getLogger(__name__)

# Single-owner constant for the live store today, mirroring preferences_gate._OWNER —
# one env axis (SYNAPSE_KG_OWNER_ID) so a future multi-tenant split threads through here.
_OWNER = os.environ.get("SYNAPSE_KG_OWNER_ID", "default")

_VALID_TYPES = ("user", "feedback", "project", "reference")

_DEFAULT_SIM_MATCH = 0.80


CONFIRM_PROMPT = """Two curated memory notes landed on the same topic. Decide their relationship.

EXISTING (stored earlier):
hook: {existing_hook}
body: {existing_body}

NEW (just asserted):
hook: {new_hook}
body: {new_body}

- "same": the NEW note restates, refines, or extends the EXISTING one — the same underlying fact or stance, possibly with updated wording or added detail. The store will UPDATE the existing note in place.
- "contradicts": the NEW note REVERSES or invalidates the EXISTING one — if the new statement is true, the old one no longer is (a decision flipped, a value changed, a rule inverted). The store will retire the old note and link it to the new one.

Output ONLY JSON: {{"relation": "same"}} or {{"relation": "contradicts"}}"""


def _sim_match_threshold() -> float:
    """Read at call time so tests/ops can tune without a restart."""
    raw = os.environ.get("SYNAPSE_NOTES_SIM_MATCH", "")
    try:
        return float(raw) if raw else _DEFAULT_SIM_MATCH
    except ValueError:
        logger.warning("bad SYNAPSE_NOTES_SIM_MATCH=%r; using %s", raw, _DEFAULT_SIM_MATCH)
        return _DEFAULT_SIM_MATCH


def _parse_relation(text: str) -> str:
    """Parser for parse_with_retry: extract {"relation": "same"|"contradicts"}."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise MalformedResponseError("no JSON object in response", text[:200])
    d = json.loads(text[start : end + 1])
    rel = d.get("relation")
    if rel not in ("same", "contradicts"):
        raise MalformedResponseError("'relation' must be 'same' or 'contradicts'", text[:200])
    return cast(str, rel)


def _confirm_relation(llm: Any, existing: dict[str, Any], hook: str, body: str) -> str:
    """One small LLM call: is the new note a restatement or a contradiction of the
    existing one? Kill switch (SYNAPSE_NOTES_CONFIRM=0) and ANY LLM failure both
    collapse to "same" -> UPDATE (no dup, no dangling lineage)."""
    if os.environ.get("SYNAPSE_NOTES_CONFIRM", "1") == "0":
        logger.info("notes confirm disabled (SYNAPSE_NOTES_CONFIRM=0); treating as 'same'")
        return "same"
    prompt = CONFIRM_PROMPT.format(
        existing_hook=existing["hook"],
        existing_body=str(existing.get("body") or "")[:2000],
        new_hook=hook,
        new_body=body[:2000],
    )
    try:
        return parse_with_retry(
            llm,
            base_prompt=prompt,
            parser=_parse_relation,
            model=stage_model("NOTES_CONFIRM"),
            max_tokens=64,
        )
    except Exception as e:
        logger.warning("notes confirm LLM failed (%s); collapsing to 'same' -> update", e)
        return "same"


def _live_match(db: Any, vec: list[float], type: str, audience: str) -> dict[str, Any] | None:
    """The nearest live note of ONE tier, if it is a restatement candidate: similar
    enough and of the same type. The tier filter runs in SQL, so a row of any other tier
    is never returned."""
    hits = db.find_live_notes(_OWNER, vec, limit=5, audience=audience)
    top = hits[0] if hits else None
    if top is None or top["sim"] < _sim_match_threshold() or top["type"] != type:
        return None
    return cast("dict[str, Any]", top)


def reconcile_note(
    db: Any,
    embedder: Any,
    llm: Any,
    *,
    hook: str,
    body: str,
    type: str,
    project: str | None,
    source_ref: str | None,
    audience: str | None = None,
    caller_restricted: bool = False,
) -> dict[str, Any]:
    """Write one explicit note, reconciling against the live set.

    Returns ``{"outcome": "created"|"updated"|"superseded", "note_id": int,
    "prev_id": int|None, "audience": str|None}`` — ``note_id`` is the live row after
    the write; ``prev_id`` is the retired row on supersession; ``audience`` is the tier
    this write set, or None when an UPDATE preserved the stored one.

    ``embedder=None`` (or an embed failure) degrades to a straight insert with
    NULL embedding — dedup silently skipped, logged, never raised.

    Audience (schema 053) is resolved here because this is the single write chokepoint.
    ``audience`` is the explicit override (precedence 1); ``caller_restricted`` says the
    write came from a REGISTERED restricted surface (precedence 2); everything else
    falls to the project rule in :func:`ingestion.surfaces.derive_audience`.

    A write never edits a live note of another tier, and a restricted caller never sees,
    edits or retires a personal note:

    * A restricted caller matches only work-safe notes, whatever ``audience`` it passed.
    * A full-trust caller matches in the tier its write will be stored under first. If
      nothing there matches, it looks in the other tier. A note found there is retired
      by a contradiction (the new note takes its own tier), is left alone by a
      restatement (the new note is created in its own tier, beside it), and is updated
      in place and moved to the tier asked for only when ``audience`` was given.
    * A restatement within a tier preserves the note's stored tier. Reclassification is
      an explicit act, never a side effect of rephrasing.
    """
    if type not in _VALID_TYPES:
        raise ValueError(f"invalid note type {type!r} — expected one of {_VALID_TYPES}")
    # Raises on an invalid explicit value BEFORE any write — a bad audience must not
    # land a note under the wrong tier and report success.
    new_audience = derive_audience(
        db, explicit=audience, caller_restricted=caller_restricted, project=project
    )

    vec: list[float] | None = None
    embed_model: str | None = None
    if embedder is None:
        logger.warning(
            "no embedder (keyless dev/test); note stored with NULL embedding, dedup skipped"
        )
    else:
        try:
            vec = list(embedder.embed([hook], task="document")[0])
            embed_model = getattr(embedder, "model_name", None) or "voyage-4-large"
        except Exception as e:
            logger.warning("note embed failed (%s); storing with NULL embedding, dedup skipped", e)
            vec, embed_model = None, None

    # `top` is the note this write may restate or contradict. A restricted caller sees
    # only the restricted tier; a full-trust one tries its own tier, then the other.
    top: dict[str, Any] | None = None
    other_tier = False
    if vec is not None:
        if caller_restricted:
            top = _live_match(db, vec, type, RESTRICTED_AUDIENCE)
        else:
            top = _live_match(db, vec, type, new_audience)
            if top is None:
                top = _live_match(db, vec, type, next(a for a in AUDIENCES if a != new_audience))
                other_tier = top is not None

    def _insert() -> int:
        return int(
            db.insert_note(
                owner_id=_OWNER,
                project=project,
                type=type,
                hook=hook,
                body=body,
                embedding=vec,
                embed_model=embed_model,
                source_ref=source_ref,
                audience=new_audience,
            )
        )

    if top is None:
        new_id = _insert()
        logger.info("note created (#%s, %s, %s): %s", new_id, type, new_audience, hook[:80])
        return {
            "outcome": "created",
            "note_id": new_id,
            "prev_id": None,
            "audience": new_audience,
        }

    relation = _confirm_relation(llm, top, hook, body)
    if relation == "contradicts":
        # A contradiction is a NEW assertion, so it derives its own tier rather than
        # inheriting the retired note's: inheriting would let a personal statement
        # arrive work-safe just because it reversed a work-safe one.
        new_id = _insert()
        db.supersede_note(top["id"], new_id)
        logger.info("note superseded #%s -> #%s: %s", top["id"], new_id, hook[:80])
        return {
            "outcome": "superseded",
            "note_id": new_id,
            "prev_id": top["id"],
            "audience": new_audience,
        }

    if other_tier and audience is None:
        # The match is a note of the other tier and nobody asked to move it. Editing it
        # would put this text under that tier's label (a personal statement served to a
        # work device, or the reverse), so it stays as it is and this write gets its own
        # row. This is also where the fail-open "same" lands (kill switch, LLM failure).
        new_id = _insert()
        logger.info(
            "note created (#%s, %s, %s) beside #%s of the other tier: %s",
            new_id,
            type,
            new_audience,
            top["id"],
            hook[:80],
        )
        return {
            "outcome": "created",
            "note_id": new_id,
            "prev_id": None,
            "audience": new_audience,
        }

    # Restatement: pass the tier ONLY when it was explicitly given. `audience=None`
    # reaches update_note as COALESCE(NULL, audience) — the stored tier survives.
    db.update_note(
        top["id"],
        hook=hook,
        body=body,
        embedding=vec,
        embed_model=embed_model,
        audience=audience,
    )
    logger.info("note updated (#%s): %s", top["id"], hook[:80])
    return {"outcome": "updated", "note_id": top["id"], "prev_id": None, "audience": audience}
