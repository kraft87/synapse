"""Reconcile one explicit note into the notes store (schema 041).

Notes are the EXPLICIT-memory store behind the always-injected board: one row per
curated memory, a short ``hook`` (the board line; also the embed target for dedup
KNN) plus a self-contained ``body`` fetched on demand by id. This module owns the
single write path: :func:`reconcile_note` is what remember() calls, so dedup and
supersession semantics can't drift between callers. It also owns the pair judge
(:data:`PAIR_PROMPT`, :func:`judge_pair`), which the nightly notes lane
(dream/notes/nightly.py) shares, so the write-time and the nightly verdict mean
the same thing.

Reconciliation, keyed on cosine similarity of the new HOOK to the live set:
  - the nearest live note of the SAME type (for a project note, the same project) with
    sim >= _SIM_MATCH (0.80) -> one LLM judgement, DUPLICATE / CORRECTION / DISTINCT:
      DUPLICATE                -> UPDATE the existing note in place (refresh hook/body/
                                  embedding/source_ref, bump updated_at; the wording it
                                  replaces goes to notes_history)
      CORRECTION, new survives -> INSERT the new note + supersede the old (lineage kept)
      anything else            -> INSERT the new note beside the old one
  - else -> INSERT a new live note.

When unsure, INSERT: a failed judgement, an unparseable one and DISTINCT all insert. A
wrong overwrite replaces a note the user wrote on purpose, while a missed merge costs one
board line until the nightly lane (which judges every near pair in both orders) merges
it. Keyless dev/test (no embedder, or the embed call fails) degrades to a straight insert
with NULL embedding — dedup is silently skipped, never a hard failure (timeline precedent).

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
from datetime import UTC, datetime
from typing import Any

from ingestion.llm_client import MalformedResponseError, parse_with_retry, stage_model
from ingestion.preferences_gate import _group_for
from ingestion.surfaces import derive_audience

logger = logging.getLogger(__name__)

# Single-owner constant for the live store today, mirroring preferences_gate._OWNER —
# one env axis (SYNAPSE_KG_OWNER_ID) so a future multi-tenant split threads through here.
_OWNER = os.environ.get("SYNAPSE_KG_OWNER_ID", "default")

_VALID_TYPES = ("user", "feedback", "project", "reference")

# Cosine of the new hook to a live note of the same type above which the judge is asked.
_SIM_MATCH = 0.80

_PAIR_VERDICTS = ("DUPLICATE", "CORRECTION", "DISTINCT")
_BODY_CHARS = 2000


PAIR_PROMPT = """Two curated memory notes from a long-term memory store sit on the same topic. Both are live, which means both are injected into every session the agent runs. Decide their relationship.

NOTE 1 (type: {t1}, project: {p1}, last updated: {u1})
hook: {h1}
body: {b1}

NOTE 2 (type: {t2}, project: {p2}, last updated: {u2})
hook: {h2}
body: {b2}

Verdicts:
- DUPLICATE — the same durable content, restated. Different wording, same claim, and neither note carries anything the other lacks. The older note will be retired and linked to the newer one.
- CORRECTION — one note explicitly corrects, retracts, or replaces the content of the other: it is written as a correction to it, it states that the other is wrong, or it records a reversal of the state the other records. The corrected note will be retired and linked to the correcting one. Name the note that SURVIVES (the correcting one) in "keep".
- DISTINCT — anything else. Both notes stay live.

Rules:
- UNDER-REPORT. When unsure, answer DISTINCT. A wrong merge hides a note the user wrote on purpose; a missed merge costs one line on a board. Those costs are not symmetric.
- Evolution of a situation is DISTINCT: a decision and its later revision, a problem and its later fix, a plan and its execution are separate notes UNLESS the newer one explicitly corrects or replaces the older.
- Sharing a subject is not enough. Two notes about the same project, tool, or person are duplicates only if they make the same CLAIM.
- Complementary detail is DISTINCT: if either note carries a fact, reason, constraint, or example the other does not, both stand.
- A general rule and a specific application of that rule are DISTINCT.

Output ONLY JSON, no prose. One of:
{{"verdict": "DUPLICATE"}}
{{"verdict": "CORRECTION", "keep": 1}}
{{"verdict": "CORRECTION", "keep": 2}}
{{"verdict": "DISTINCT"}}"""


def extract_obj(text: str) -> dict[str, Any]:
    """The outermost JSON object in an LLM reply, as a parser for ``parse_with_retry``."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise MalformedResponseError("no JSON object in response", text[:200])
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise MalformedResponseError(f"JSON decode error: {exc}", text[:200]) from exc
    if not isinstance(obj, dict):
        raise MalformedResponseError("expected a JSON object", text[:200])
    return obj


def parse_pair_verdict(text: str) -> dict[str, Any]:
    """Parser for parse_with_retry: ``{"verdict": ..., "keep": 1|2}``.

    ``keep`` is REQUIRED for CORRECTION — without it there is no direction, and
    guessing one would be exactly the destructive coin-flip the judge must not make.
    Raising instead lets the retry loop quote the failure back to the model."""
    obj = extract_obj(text)
    verdict = str(obj.get("verdict", "")).strip().upper()
    if verdict not in _PAIR_VERDICTS:
        raise MalformedResponseError(f"'verdict' must be one of {_PAIR_VERDICTS}", text[:200])
    keep = obj.get("keep")
    if isinstance(keep, str) and keep.strip().isdigit():
        keep = int(keep.strip())
    if keep not in (1, 2):
        keep = None
    if verdict == "CORRECTION" and keep is None:
        raise MalformedResponseError("CORRECTION requires 'keep': 1 or 2", text[:200])
    return {"verdict": verdict, "keep": keep}


def judge_model() -> str:
    """The nightly lane's judge: independently overridable via SYNAPSE_NOTES_CURATE_MODEL
    (stage_model precedence). remember() judges on its own SYNAPSE_NOTES_CONFIRM_MODEL,
    since a caller waits on that one."""
    return stage_model("NOTES_CURATE")


def fmt_note(note: dict[str, Any]) -> dict[str, str]:
    return {
        "hook": str(note.get("hook") or ""),
        "body": str(note.get("body") or "")[:_BODY_CHARS],
        "type": str(note.get("type") or ""),
        "project": str(note.get("project") or "-"),
        "updated": str(note.get("updated_at") or "")[:19],
    }


def judge_pair(
    llm: Any, first: dict[str, Any], second: dict[str, Any], model: str | None = None
) -> dict[str, Any]:
    """One judgement of the pair in ONE ordering (NOTE 1 = first, NOTE 2 = second), on
    ``model`` (default: the lane's :func:`judge_model`)."""
    f, s = fmt_note(first), fmt_note(second)
    prompt = PAIR_PROMPT.format(
        t1=f["type"], p1=f["project"], u1=f["updated"], h1=f["hook"], b1=f["body"],
        t2=s["type"], p2=s["project"], u2=s["updated"], h2=s["hook"], b2=s["body"],
    )  # fmt: skip
    return parse_with_retry(
        llm,
        base_prompt=prompt,
        parser=parse_pair_verdict,
        model=model or judge_model(),
        max_tokens=64,
    )


def _write_action(llm: Any, existing: dict[str, Any], new: dict[str, Any]) -> str:
    """What the store does with ``new`` beside its near neighbour ``existing``:
    ``"update"`` (a restatement), ``"supersede"`` (it corrects the old note) or
    ``"insert"`` (anything else, including every failure — see the module docstring)."""
    if existing["hook"] == new["hook"] and (existing.get("body") or "") == new["body"]:
        return "update"  # an exact restatement (a replayed spool write) needs no judge
    try:
        verdict = judge_pair(llm, existing, new, model=stage_model("NOTES_CONFIRM"))
    except Exception as e:
        logger.warning("notes judge failed (%s); inserting as a new note", e)
        return "insert"
    if verdict["verdict"] == "DUPLICATE":
        return "update"
    # keep=2 names the NEW note as the survivor. keep=1 says the old note corrects the new
    # one, which the user just asserted: leave both live and let the nightly lane decide.
    if verdict["verdict"] == "CORRECTION" and verdict["keep"] == 2:
        return "supersede"
    return "insert"


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
    NULL embedding — dedup silently skipped, logged, never raised. So does a failed
    judge call: see the module docstring for why every doubt inserts.

    Audience (schema 053) is resolved here because this is the single write chokepoint.
    ``audience`` is the explicit override (precedence 1); ``caller_restricted`` says the
    write came from a REGISTERED restricted surface (precedence 2); everything else
    falls to the project rule in :func:`ingestion.surfaces.derive_audience`. A
    restatement (UPDATE) PRESERVES the note's stored tier unless ``audience`` overrides
    it — reclassification is an explicit act, never a side effect of rephrasing.
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

    group_id = _group_for(project)

    # The nearest live note this write could restate: the same type and, for a project
    # note, the same project (a restatement in project B must not rewrite project A's note).
    top = None
    if vec is not None:
        top = next(
            (
                c
                for c in db.find_live_notes(_OWNER, group_id, vec, limit=5)
                if c["type"] == type and (type != "project" or c["project"] == project)
            ),
            None,
        )
    prev_id: int | None = None
    if top is not None and top["sim"] >= _SIM_MATCH:
        new = {
            "hook": hook,
            "body": body,
            "type": type,
            "project": project,
            "updated_at": datetime.now(UTC),
        }
        action = _write_action(llm, top, new)
        if action == "update":
            # Restatement: pass the tier ONLY when it was explicitly given. `audience=None`
            # reaches update_note as COALESCE(NULL, audience) — the stored tier survives.
            db.update_note(
                top["id"],
                hook=hook,
                body=body,
                embedding=vec,
                embed_model=embed_model,
                source_ref=source_ref,
                audience=audience,
            )
            logger.info("note updated (#%s): %s", top["id"], hook[:80])
            return {
                "outcome": "updated",
                "note_id": top["id"],
                "prev_id": None,
                "audience": audience,
            }
        if action == "supersede":
            prev_id = top["id"]

    # A new note, and a superseding one in particular, derives its own tier rather than
    # inheriting the retired note's: inheriting would let a personal statement arrive
    # work-safe just because it reversed a work-safe one.
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
        audience=new_audience,
    )
    if prev_id is not None:
        db.supersede_note(prev_id, new_id)
        logger.info("note superseded #%s -> #%s: %s", prev_id, new_id, hook[:80])
        return {
            "outcome": "superseded",
            "note_id": new_id,
            "prev_id": prev_id,
            "audience": new_audience,
        }
    logger.info("note created (#%s, %s, %s): %s", new_id, type, new_audience, hook[:80])
    return {"outcome": "created", "note_id": new_id, "prev_id": None, "audience": new_audience}
