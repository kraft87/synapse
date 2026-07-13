"""Reconcile one explicit note into the notes store (schema 041).

Notes are the EXPLICIT-memory store behind the always-injected board: one row per
curated memory, a short ``hook`` (the board line; also the embed target for dedup
KNN) plus a self-contained ``body`` fetched on demand by id. This module owns the
single write path — the remember() rework and the seed importer both call
:func:`reconcile_note` so dedup/supersession semantics can't drift between callers.

Reconciliation, keyed on cosine similarity of the new HOOK to the live set:
  - top sim >= SYNAPSE_NOTES_SIM_MATCH (default 0.80) AND same type
      -> one small LLM confirm call ({"relation": "same"|"contradicts"}):
         "same"        -> UPDATE the existing note in place (refresh hook/body/
                          embedding, bump updated_at — restatements don't pile up)
         "contradicts" -> INSERT the new note + supersede the old (lineage kept)
  - else -> INSERT a new live note.

Fail-open, always toward UPDATE: the kill switch (SYNAPSE_NOTES_CONFIRM=0) and any
LLM failure both collapse the confirm to "same" — an over-eager update loses a
nuance; a missed update duplicates a board line forever. Keyless dev/test (no
embedder, or the embed call fails) degrades to a straight insert with NULL
embedding — dedup is silently skipped, never a hard failure (timeline precedent).

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
from ingestion.preferences_gate import _group_for

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
) -> dict[str, Any]:
    """Write one explicit note, reconciling against the live set.

    Returns ``{"outcome": "created"|"updated"|"superseded", "note_id": int,
    "prev_id": int|None}`` — ``note_id`` is the live row after the write;
    ``prev_id`` is the retired row on supersession, else None.

    ``embedder=None`` (or an embed failure) degrades to a straight insert with
    NULL embedding — dedup silently skipped, logged, never raised.
    """
    if type not in _VALID_TYPES:
        raise ValueError(f"invalid note type {type!r} — expected one of {_VALID_TYPES}")

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

    group_id = _group_for(project)

    candidates: list[dict[str, Any]] = []
    if vec is not None:
        candidates = db.find_live_notes(_OWNER, group_id, vec, limit=5)
    top = candidates[0] if candidates else None

    if top is None or top["sim"] < _sim_match_threshold() or top["type"] != type:
        new_id = db.insert_note(
            owner_id=_OWNER,
            group_id=group_id,
            project=project,
            type=type,
            hook=hook,
            body=body,
            embedding=vec,
            embed_model=embed_model,
            source_ref=source_ref,
        )
        logger.info("note created (#%s, %s): %s", new_id, type, hook[:80])
        return {"outcome": "created", "note_id": new_id, "prev_id": None}

    relation = _confirm_relation(llm, top, hook, body)
    if relation == "contradicts":
        new_id = db.insert_note(
            owner_id=_OWNER,
            group_id=group_id,
            project=project,
            type=type,
            hook=hook,
            body=body,
            embedding=vec,
            embed_model=embed_model,
            source_ref=source_ref,
        )
        db.supersede_note(top["id"], new_id)
        logger.info("note superseded #%s -> #%s: %s", top["id"], new_id, hook[:80])
        return {"outcome": "superseded", "note_id": new_id, "prev_id": top["id"]}

    db.update_note(top["id"], hook=hook, body=body, embedding=vec, embed_model=embed_model)
    logger.info("note updated (#%s): %s", top["id"], hook[:80])
    return {"outcome": "updated", "note_id": top["id"], "prev_id": None}
