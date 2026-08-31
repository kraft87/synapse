#!/usr/bin/env python3
"""dream→notes curation lane (server-side in the dream container; entry: run_lane()).

WHY THIS EXISTS
The notes store (schema 041) has one inflow — ``remember()`` — and had no outflow.
Nothing merged paraphrases, retired a note a later correction had already replaced,
or re-scoped project-specific content that was filed as a global rule. Notes are
injected into every session, so each redundant row is a permanent tax on attention,
and the only thing keeping the board bounded was truncation.

WHY IT AUTO-APPLIES
This deliberately has no review queue, no proposals table, and no review command:
curation is meant to happen silently in the background as a self-improvement step,
not to hand the user a second inbox. The safety comes from the shape of the work
instead:

  * Every write is non-destructive and reversible. Merges set ``superseded_by``
    (the row stays, the lineage is explicit); retypes change ``type``/``project``
    and touch nothing else. Nothing is ever deleted or rewritten.
  * Every judgement is made twice, with the two notes in both orders, and only a
    consensus acts. Disagreement collapses to the safe verdict (DISTINCT / GLOBAL).
  * Both prompts carry an explicit under-report bias: a wrong merge hides a note
    the user wrote on purpose, a missed merge costs one board line.
  * Per-night caps bound the blast radius of a bad model day (default: 40 pairs
    judged, 10 scope judgements, 10 applied operations total).
  * Every judgement — applied or not — is written to ``notes_curation`` (schema
    051), which is simultaneously the audit log, the memo that stops the same pair
    being re-judged every night, and the staleness cursor (a note edited after its
    verdict gets re-judged).

STAGES
  A. Candidate pairs — pure SQL self-join of live notes on the cosine similarity of
     their stored HOOK embeddings. No new embedding calls. The 0.60 floor is
     permissive on purpose: a correction often shares little wording with the note
     it corrects. The floor bounds cost; the judge is the precision layer.
  B. Pair judge + apply — DUPLICATE retires the older-updated note, CORRECTION
     retires the note the model names as corrected, DISTINCT does nothing.
  C. Retype — live global notes (user/feedback) judged for project specificity; a
     PROJECT verdict applies only if the slug already exists in the store.

Kill switch: ``SYNAPSE_NOTES_LANE=0``. ENABLED BY DEFAULT.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from dream.skills.config import db_url
from ingestion.llm_client import (
    MalformedResponseError,
    create_llm_client,
    parse_with_retry,
    stage_model,
)
from ingestion.notes import _OWNER
from ingestion.surfaces import (
    DEFAULT_AUDIENCE,
    RESTRICTED_AUDIENCE,
    restricted_project_union,
)

logger = logging.getLogger(__name__)

# Cosine floor for pair candidacy. Low by design — see the module header.
DEFAULT_SIM_FLOOR = 0.60
# Per-run caps. MAX_APPLY spans stages B and C together; when it is spent the lane
# STOPS JUDGING rather than judging on and recording verdict-only rows, so a memo
# row always tells the truth about whether its verdict was acted on.
DEFAULT_MAX_JUDGE = 40
DEFAULT_MAX_RETYPE_JUDGE = 10
DEFAULT_MAX_APPLY = 10

_PAIR_VERDICTS = ("DUPLICATE", "CORRECTION", "DISTINCT")
_BODY_CHARS = 2000  # matches the write-time NOTES_CONFIRM prompt's body budget


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


RETYPE_PROMPT = """A curated memory note is filed as GLOBAL, so it is injected into every session the agent runs, on every project. Decide whether it belongs there or under a single project.

NOTE (type: {t}, last updated: {u})
hook: {h}
body: {b}

Known project ids (the ONLY values you may answer with): {projects}

Answer PROJECT only if the note's content is specific to ONE named codebase or system — how that system is built, configured, deployed, or debugged — such that it would be noise in a session about anything else. The project id must come from the list above; do not invent one.

Answer GLOBAL for everything else. These stay GLOBAL even when they name a project:
- standing behavioral rules for the agent: tone, format, verbosity, workflow, what to do or avoid;
- safety rules, prohibitions, and things the agent must never do;
- facts about the user or the people around them: preferences, hardware they own, how they work, their history;
- anything that would still apply on a different project next week.

UNDER-REPORT. When unsure, answer GLOBAL. Wrongly narrowing a note hides a standing rule the agent needs everywhere; leaving one global costs a line on a board.

Output ONLY JSON, no prose. One of:
{{"scope": "GLOBAL"}}
{{"scope": "PROJECT", "project": "<one of the known project ids>"}}"""


# --------------------------------------------------------------------------- config
def _enabled() -> bool:
    """Kill switch, not an enable gate: the lane ships ON."""
    return os.environ.get("SYNAPSE_NOTES_LANE", "1") != "0"


def _env_num(name: str, default: float, *, cast: type[int] | type[float]) -> Any:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return cast(default)
    try:
        return cast(float(raw))
    except ValueError:
        logger.warning("bad %s=%r; using %s", name, raw, default)
        return cast(default)


# --------------------------------------------------------------------------- parsers
def _extract_obj(text: str) -> dict[str, Any]:
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
    guessing one would be exactly the destructive coin-flip this lane must not make.
    Raising instead lets the retry loop quote the failure back to the model."""
    obj = _extract_obj(text)
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


def parse_scope_verdict(text: str) -> dict[str, Any]:
    """Parser for parse_with_retry: ``{"scope": "GLOBAL"}`` or
    ``{"scope": "PROJECT", "project": "<slug>"}``."""
    obj = _extract_obj(text)
    scope = str(obj.get("scope", "")).strip().upper()
    if scope not in ("GLOBAL", "PROJECT"):
        raise MalformedResponseError("'scope' must be GLOBAL or PROJECT", text[:200])
    project = obj.get("project")
    project = project.strip() if isinstance(project, str) and project.strip() else None
    if scope == "PROJECT" and not project:
        raise MalformedResponseError("PROJECT requires a 'project' slug", text[:200])
    return {"scope": scope, "project": project if scope == "PROJECT" else None}


# --------------------------------------------------------------------------- pure policy
def _older_first(a: dict[str, Any], b: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """(older, newer) by updated_at, id breaking a tie — deterministic either way."""
    pair = sorted((a, b), key=lambda n: (n["updated_at"], n["id"]))
    return pair[0], pair[1]


def _keep_id(verdict: dict[str, Any], first: dict[str, Any], second: dict[str, Any]) -> int | None:
    """Map the model's positional ``keep`` (1 or 2) onto the note id that occupied
    that slot IN THAT ORDERING. This is what makes the both-orders check meaningful:
    an agreeing pair of answers must name the same real id, not the same slot."""
    keep = verdict.get("keep")
    if keep == 1:
        return int(first["id"])
    if keep == 2:
        return int(second["id"])
    return None


def resolve_pair(
    a: dict[str, Any], b: dict[str, Any], v_ab: dict[str, Any], v_ba: dict[str, Any]
) -> dict[str, Any]:
    """Collapse the two orderings into one decision.

    ``v_ab`` is the judgement with NOTE 1 = a, NOTE 2 = b; ``v_ba`` is the same pair
    judged with the contents swapped. Returns
    ``{"verdict", "winner", "loser", "reason"}``; anything short of a clean consensus
    returns DISTINCT with the reason recorded, so the audit row explains the no-op."""
    safe = {"verdict": "DISTINCT", "winner": None, "loser": None}
    if v_ab.get("verdict") != v_ba.get("verdict"):
        return safe | {"reason": "order-disagreement"}
    verdict = v_ab.get("verdict")
    if verdict == "DISTINCT":
        return safe | {"reason": "both-distinct"}

    if verdict == "DUPLICATE":
        # A duplicate across types is really a mis-typing, and merging would silently
        # pick one shelf for content the user filed on two. Leave it to stage C.
        if a["type"] != b["type"]:
            return safe | {"reason": "cross-type-duplicate"}
        older, newer = _older_first(a, b)
        return {
            "verdict": "DUPLICATE",
            "winner": int(newer["id"]),
            "loser": int(older["id"]),
            "reason": "consensus",
        }

    # CORRECTION — both orders must name the SAME surviving note.
    keep_ab = _keep_id(v_ab, a, b)
    keep_ba = _keep_id(v_ba, b, a)
    if keep_ab is None or keep_ba is None:
        return safe | {"reason": "missing-direction"}
    if keep_ab != keep_ba:
        return safe | {"reason": "direction-disagreement"}
    ids = {int(a["id"]), int(b["id"])}
    if keep_ab not in ids:
        return safe | {"reason": "keep-not-in-pair"}
    loser = (ids - {keep_ab}).pop()
    return {"verdict": "CORRECTION", "winner": keep_ab, "loser": loser, "reason": "consensus"}


# --------------------------------------------------------------------------- judges
def _judge_model() -> str:
    """Independently overridable via SYNAPSE_NOTES_CURATE_MODEL (stage_model precedence)."""
    return stage_model("NOTES_CURATE")


def _fmt(note: dict[str, Any]) -> dict[str, str]:
    return {
        "hook": str(note.get("hook") or ""),
        "body": str(note.get("body") or "")[:_BODY_CHARS],
        "type": str(note.get("type") or ""),
        "project": str(note.get("project") or "-"),
        "updated": str(note.get("updated_at") or "")[:19],
    }


def judge_pair(llm: Any, first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """One judgement of the pair in ONE ordering (NOTE 1 = first, NOTE 2 = second)."""
    f, s = _fmt(first), _fmt(second)
    prompt = PAIR_PROMPT.format(
        t1=f["type"], p1=f["project"], u1=f["updated"], h1=f["hook"], b1=f["body"],
        t2=s["type"], p2=s["project"], u2=s["updated"], h2=s["hook"], b2=s["body"],
    )  # fmt: skip
    return parse_with_retry(
        llm, base_prompt=prompt, parser=parse_pair_verdict, model=_judge_model(), max_tokens=64
    )


def judge_scope(llm: Any, note: dict[str, Any], projects: list[str]) -> dict[str, Any]:
    n = _fmt(note)
    prompt = RETYPE_PROMPT.format(
        t=n["type"],
        u=n["updated"],
        h=n["hook"],
        b=n["body"],
        projects=", ".join(projects) if projects else "(none known)",
    )
    return parse_with_retry(
        llm, base_prompt=prompt, parser=parse_scope_verdict, model=_judge_model(), max_tokens=64
    )


# --------------------------------------------------------------------------- orchestrator
def run_lane(
    *,
    db: Any | None = None,
    llm: Any | None = None,
    sim_floor: float | None = None,
    max_judge: int | None = None,
    max_retype_judge: int | None = None,
    max_apply: int | None = None,
) -> dict[str, Any]:
    """Run one nightly curation pass. Returns the counts/samples/errors the dream
    runner records. ``db``/``llm`` are injection points for tests; in production both
    are built from the container env.

    Every LLM failure is per-item and fail-soft: the item is skipped WITHOUT a memo
    row, so a transient outage means "try again tomorrow" rather than a DISTINCT
    verdict frozen in forever."""
    counts = {
        "pairs_judged": 0,
        "retype_judged": 0,
        "applied_supersede": 0,
        "applied_retype": 0,
    }
    samples: list[str] = []
    errors: list[str] = []
    if not _enabled():
        logger.info("dream→notes lane disabled (SYNAPSE_NOTES_LANE=0)")
        return {"enabled": False, "counts": counts, "samples": samples, "errors": errors}

    floor = float(sim_floor if sim_floor is not None else _env_num("NOTES_LANE_SIM_FLOOR", DEFAULT_SIM_FLOOR, cast=float))  # fmt: skip
    cap_judge = int(max_judge if max_judge is not None else _env_num("NOTES_LANE_MAX_JUDGE", DEFAULT_MAX_JUDGE, cast=int))  # fmt: skip
    cap_retype = int(max_retype_judge if max_retype_judge is not None else _env_num("NOTES_LANE_MAX_RETYPE_JUDGE", DEFAULT_MAX_RETYPE_JUDGE, cast=int))  # fmt: skip
    cap_apply = int(max_apply if max_apply is not None else _env_num("NOTES_LANE_MAX_APPLY", DEFAULT_MAX_APPLY, cast=int))  # fmt: skip

    owns_db = db is None
    if db is None:
        from ingestion.db import Database

        url = db_url()
        if not url:
            raise RuntimeError("SYNAPSE_DB_URL not set")
        db = Database(url)
    if llm is None:
        llm = create_llm_client()

    applied = 0
    try:
        # ---- stages A + B: candidate pairs, judged both ways, merged on consensus.
        retired: set[int] = set()
        for cand in db.find_note_pair_candidates(_OWNER, sim_floor=floor, limit=cap_judge):
            if applied >= cap_apply:
                break
            a, b = cand["a"], cand["b"]
            # A note retired earlier in THIS run is no longer live; its remaining
            # candidate pairs are stale and would be judged against a dead row.
            if a["id"] in retired or b["id"] in retired:
                continue
            try:
                v_ab = judge_pair(llm, a, b)
                v_ba = judge_pair(llm, b, a)
            except Exception as e:  # fail-soft, no memo — retry tomorrow
                logger.warning("notes pair judge failed (%s/%s): %s", a["id"], b["id"], e)
                errors.append(f"pair {a['id']}/{b['id']}: {e}")
                continue
            res = resolve_pair(a, b, v_ab, v_ba)
            counts["pairs_judged"] += 1
            act = res["verdict"] in ("DUPLICATE", "CORRECTION")
            if act:
                db.supersede_note(res["loser"], res["winner"])
                retired.add(res["loser"])
                applied += 1
                counts["applied_supersede"] += 1
                samples.append(
                    f"n:{res['loser']} -> superseded by n:{res['winner']} "
                    f"({res['verdict'].lower()})"
                )
            db.record_curation(
                op="pair",
                note_a=int(a["id"]),
                note_b=int(b["id"]),
                verdict=res["verdict"],
                applied=act,
                detail={
                    "sim": round(cand["sim"], 4),
                    "reason": res["reason"],
                    "winner": res["winner"],
                    "loser": res["loser"],
                    "orders": [v_ab, v_ba],
                },
            )

        # ---- stage C: global notes that are really about one project.
        projects: list[str] = db.known_project_slugs()
        # Audience (schema 053) is derived from a note's PROJECT, so a retype changes the
        # right answer. Read the restricted-surface allowlist union ONCE for the batch —
        # it can't change mid-run, and re-deriving per note would be one query each.
        restricted_projects = restricted_project_union(db)
        for note in db.find_retype_candidates(_OWNER, limit=cap_retype):
            if applied >= cap_apply:
                break
            try:
                scope = judge_scope(llm, note, projects)
            except Exception as e:  # fail-soft, no memo — retry tomorrow
                logger.warning("notes scope judge failed (%s): %s", note["id"], e)
                errors.append(f"retype {note['id']}: {e}")
                continue
            counts["retype_judged"] += 1
            slug = scope.get("project")
            detail: dict[str, Any] = {"from_type": note["type"]}
            act = False
            if scope["scope"] == "PROJECT" and slug:
                detail["project"] = slug
                # The slug must already exist in the store. A hallucinated project is
                # recorded and skipped — writing it would file the note on a shelf
                # nothing ever reads from, which is deletion with extra steps.
                if db.project_slug_exists(slug):
                    # Re-derive by the project rule: a note moving onto a work-readable
                    # project becomes work-safe, one moving off it goes back to personal.
                    # Nothing else in this lane touches audience — pair supersession keeps
                    # each note's own tag, which IS the "preserve on update" requirement.
                    new_audience = (
                        RESTRICTED_AUDIENCE if slug in restricted_projects else DEFAULT_AUDIENCE
                    )
                    db.retype_note(
                        int(note["id"]), type="project", project=slug, audience=new_audience
                    )
                    detail["audience"] = new_audience
                    act = True
                    applied += 1
                    counts["applied_retype"] += 1
                    samples.append(f"n:{note['id']} {note['type']} -> project:{slug}")
                else:
                    detail["reason"] = "unknown-project-slug"
            db.record_curation(
                op="retype",
                note_a=int(note["id"]),
                note_b=None,
                verdict=scope["scope"],
                applied=act,
                detail=detail,
            )
    finally:
        if owns_db:
            db.close()

    logger.info(
        "dream→notes: judged %d pairs / %d scopes, applied %d supersede + %d retype "
        "(budget %d), %d errors",
        counts["pairs_judged"],
        counts["retype_judged"],
        counts["applied_supersede"],
        counts["applied_retype"],
        cap_apply,
        len(errors),
    )
    return {"enabled": True, "counts": counts, "samples": samples, "errors": errors}


if __name__ == "__main__":  # pragma: no cover - manual runs
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-judge", type=int, default=None)
    ap.add_argument("--max-retype-judge", type=int, default=None)
    ap.add_argument("--max-apply", type=int, default=None)
    ap.add_argument("--sim-floor", type=float, default=None)
    args = ap.parse_args()
    print(
        run_lane(
            sim_floor=args.sim_floor,
            max_judge=args.max_judge,
            max_retype_judge=args.max_retype_judge,
            max_apply=args.max_apply,
        )
    )
