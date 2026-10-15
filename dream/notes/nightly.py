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
    (the row stays, the lineage is explicit); retypes change ``type``/``project``/
    ``audience`` and touch nothing else, with the old values kept in the audit row.
    Nothing is ever deleted or rewritten. ``--list`` shows what was applied and
    ``--revert ID`` undoes one row (see :func:`revert_curation`).
  * A merge needs a both-orders consensus: the pair is judged with the two notes in
    each order and only agreement acts. Disagreement collapses to the safe verdict
    (DISTINCT / GLOBAL); a first order that already says DISTINCT ends the pair, since
    the second could not change it.
  * A winner must be visible wherever its loser was: a project note never retires a
    global one or another project's, and a personal note never retires a work-safe one.
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
     it corrects. The floor bounds cost; the judge is the precision layer. The same
     judge (ingestion.notes.judge_pair) runs at write time in remember(), which
     inserts whenever it is unsure; this lane is the second pass over what that left.
  B. Pair judge + apply — DUPLICATE retires the older-updated note, CORRECTION
     retires the note the model names as corrected, DISTINCT does nothing.
  C. Retype — live global notes (user/feedback) judged for project specificity; a
     PROJECT verdict applies only if the slug is one of the project ids the judge
     was offered.

Kill switch: ``SYNAPSE_NOTES_LANE=0``, read by dream/__main__ (it does not run the lane).
ENABLED BY DEFAULT.

Operator commands (full trust: they run in the dream container):
  python -m dream.notes.nightly --list [N]    the N newest applied rows
  python -m dream.notes.nightly --revert ID   undo one of them
"""

from __future__ import annotations

import logging
from typing import Any

from ingestion.llm_client import (
    MalformedResponseError,
    create_llm_client,
    extract_json_object,
    parse_with_retry,
)
from ingestion.notes import fmt_note, judge_model, judge_pair
from ingestion.settings import KG_OWNER_ID, require_setting
from ingestion.surfaces import RESTRICTED_AUDIENCE, project_audience, restricted_project_union

logger = logging.getLogger(__name__)

# Cosine floor for pair candidacy. Low by design — see the module header.
DEFAULT_SIM_FLOOR = 0.60
# Per-run caps. MAX_APPLY spans stages B and C together; when it is spent the lane
# STOPS JUDGING rather than judging on and recording verdict-only rows, so a memo
# row always tells the truth about whether its verdict was acted on.
DEFAULT_MAX_JUDGE = 40
DEFAULT_MAX_RETYPE_JUDGE = 10
DEFAULT_MAX_APPLY = 10


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


# --------------------------------------------------------------------------- parsers
def parse_scope_verdict(text: str) -> dict[str, Any]:
    """Parser for parse_with_retry: ``{"scope": "GLOBAL"}`` or
    ``{"scope": "PROJECT", "project": "<slug>"}``."""
    obj = extract_json_object(text)
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


def settled_by_first_order(
    a: dict[str, Any], b: dict[str, Any], v_ab: dict[str, Any]
) -> str | None:
    """The reason the first ordering ends the pair on its own, or None when the swapped
    ordering is still needed. A DISTINCT stays DISTINCT whatever the second order says,
    and so does a DUPLICATE across types, so judging them again would only cost a call."""
    if v_ab.get("verdict") == "DISTINCT":
        return "first-order-distinct"
    # A duplicate across types is really a mis-typing, and merging would silently
    # pick one shelf for content the user filed on two. Leave it to stage C.
    if v_ab.get("verdict") == "DUPLICATE" and a["type"] != b["type"]:
        return "cross-type-duplicate"
    return None


def _cannot_stand_in(winner: dict[str, Any], loser: dict[str, Any]) -> str | None:
    """The reason ``winner`` may not retire ``loser``, or None. A merge must not take a
    note off a board that stays without its replacement: project notes are served only
    on their own project's board, and work-safe ones are the only kind a work host gets.
    The reverse directions (a global note retiring a project one, a work-safe note
    retiring a personal one) lose nothing anywhere."""
    if winner["type"] == "project" and (
        loser["type"] != "project" or loser["project"] != winner["project"]
    ):
        return "winner-narrower-than-loser"
    if loser["audience"] == RESTRICTED_AUDIENCE and winner["audience"] != RESTRICTED_AUDIENCE:
        return "audience-narrowing"
    return None


def resolve_pair(
    a: dict[str, Any], b: dict[str, Any], v_ab: dict[str, Any], v_ba: dict[str, Any] | None
) -> dict[str, Any]:
    """Collapse the orderings into one decision.

    ``v_ab`` is the judgement with NOTE 1 = a, NOTE 2 = b; ``v_ba`` is the same pair
    judged with the contents swapped, or None when :func:`settled_by_first_order` ended
    the pair before it was asked. Returns ``{"verdict", "winner", "loser", "reason"}``;
    anything short of a clean consensus returns DISTINCT with the reason recorded, so
    the audit row explains the no-op."""
    safe = {"verdict": "DISTINCT", "winner": None, "loser": None}
    if (settled := settled_by_first_order(a, b, v_ab)) is not None:
        return safe | {"reason": settled}
    if v_ba is None:
        raise ValueError("this pair needs both orderings judged")
    if v_ab.get("verdict") != v_ba.get("verdict"):
        return safe | {"reason": "order-disagreement"}
    verdict = v_ab["verdict"]

    if verdict == "DUPLICATE":
        loser, winner = _older_first(a, b)
    else:
        # CORRECTION — both orders must name the SAME surviving note.
        keep_ab = _keep_id(v_ab, a, b)
        keep_ba = _keep_id(v_ba, b, a)
        if keep_ab is None or keep_ba is None:
            return safe | {"reason": "missing-direction"}
        if keep_ab != keep_ba:
            return safe | {"reason": "direction-disagreement"}
        winner, loser = (a, b) if keep_ab == int(a["id"]) else (b, a)

    if (refusal := _cannot_stand_in(winner, loser)) is not None:
        return safe | {"reason": refusal}
    return {
        "verdict": verdict,
        "winner": int(winner["id"]),
        "loser": int(loser["id"]),
        "reason": "consensus",
    }


# --------------------------------------------------------------------------- judges
def judge_scope(llm: Any, note: dict[str, Any], projects: list[str]) -> dict[str, Any]:
    n = fmt_note(note)
    prompt = RETYPE_PROMPT.format(
        t=n["type"],
        u=n["updated"],
        h=n["hook"],
        b=n["body"],
        projects=", ".join(projects) if projects else "(none known)",
    )
    return parse_with_retry(
        llm, base_prompt=prompt, parser=parse_scope_verdict, model=judge_model(), max_tokens=64
    )


# --------------------------------------------------------------------------- orchestrator
def run_lane(
    *,
    db: Any | None = None,
    llm: Any | None = None,
    sim_floor: float = DEFAULT_SIM_FLOOR,
    max_judge: int = DEFAULT_MAX_JUDGE,
    max_retype_judge: int = DEFAULT_MAX_RETYPE_JUDGE,
    max_apply: int = DEFAULT_MAX_APPLY,
) -> dict[str, Any]:
    """Run one nightly curation pass. Returns the counts/samples/errors the dream
    runner records. ``db``/``llm`` are injection points for tests; in production both
    are built from the container env. The limits are arguments for manual runs
    (``--max-judge`` and friends) and tests.

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

    owns_db = db is None
    if db is None:
        from ingestion.db import Database

        db = Database(require_setting("SYNAPSE_DB_URL"))
    if llm is None:
        llm = create_llm_client()

    applied = 0
    try:
        # ---- stages A + B: candidate pairs, judged both ways, merged on consensus.
        retired: set[int] = set()
        for cand in db.find_note_pair_candidates(KG_OWNER_ID, sim_floor=sim_floor, limit=max_judge):
            if applied >= max_apply:
                break
            a, b = cand["a"], cand["b"]
            # A note retired earlier in THIS run is no longer live; its remaining
            # candidate pairs are stale and would be judged against a dead row.
            if a["id"] in retired or b["id"] in retired:
                continue
            try:
                v_ab = judge_pair(llm, a, b)
                v_ba = None if settled_by_first_order(a, b, v_ab) else judge_pair(llm, b, a)
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
                    "orders": [v for v in (v_ab, v_ba) if v is not None],
                },
            )

        # ---- stage C: global notes that are really about one project.
        projects: list[str] = db.known_project_slugs()
        stored_casing = {p.lower(): p for p in projects}
        # Audience (schema 053) is derived from a note's PROJECT, so a retype changes the
        # right answer. Read the restricted-surface allowlist union ONCE for the batch —
        # it can't change mid-run, and re-deriving per note would be one query each.
        restricted_projects = restricted_project_union(db)
        for note in db.find_retype_candidates(KG_OWNER_ID, limit=max_retype_judge):
            if applied >= max_apply:
                break
            try:
                scope = judge_scope(llm, note, projects)
            except Exception as e:  # fail-soft, no memo — retry tomorrow
                logger.warning("notes scope judge failed (%s): %s", note["id"], e)
                errors.append(f"retype {note['id']}: {e}")
                continue
            counts["retype_judged"] += 1
            slug = scope.get("project")
            # What a revert needs to put the note back exactly as it was.
            detail: dict[str, Any] = {
                "from_type": note["type"],
                "from_project": note["project"],
                "from_audience": note["audience"],
            }
            act = False
            if scope["scope"] == "PROJECT" and slug:
                # The slug must be one the judge was offered (the closed list above), matched
                # case-folded and filed under the stored casing. A hallucinated project is
                # recorded and skipped — writing it would file the note on a shelf nothing
                # ever reads from, which is deletion with extra steps.
                known = slug if slug in projects else stored_casing.get(slug.lower())
                detail["project"] = known or slug
                if known:
                    # Re-derive by the project rule: a note moving onto a work-readable
                    # project becomes work-safe, one moving off it goes back to personal.
                    # Nothing else in this lane touches audience — pair supersession keeps
                    # each note's own tag, which IS the "preserve on update" requirement.
                    new_audience = project_audience(known, restricted_projects)
                    db.retype_note(
                        int(note["id"]), type="project", project=known, audience=new_audience
                    )
                    detail["audience"] = new_audience
                    act = True
                    applied += 1
                    counts["applied_retype"] += 1
                    samples.append(f"n:{note['id']} {note['type']} -> project:{known}")
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
        max_apply,
        len(errors),
    )
    return {"counts": counts, "samples": samples, "errors": errors}


# --------------------------------------------------------------------------- undo
def list_applied(db: Any, limit: int = 20) -> list[str]:
    """One line per applied curation row, newest first: the id ``revert_curation`` takes,
    what happened, and the hook of each note involved."""
    lines = []
    for row in db.list_applied_curation(limit=limit):
        when = row["judged_at"].strftime("%Y-%m-%d")
        detail = row["detail"] or {}
        if row["op"] == "pair":
            loser = detail.get("loser")
            retired_hook, kept_hook = (
                (row["hook_a"], row["hook_b"])
                if loser == row["note_a"]
                else (row["hook_b"], row["hook_a"])
            )
            lines.append(
                f"#{row['id']} {when} {row['verdict'].lower()}: n:{loser} superseded by "
                f"n:{detail.get('winner')} | retired: {retired_hook} | kept: {kept_hook}"
            )
        else:
            lines.append(
                f"#{row['id']} {when} retype: n:{row['note_a']} {detail.get('from_type')} -> "
                f"project:{detail.get('project')} | {row['hook_a']}"
            )
    return lines


def revert_curation(db: Any, curation_id: int) -> str:
    """Undo one applied curation row and mark it ``REVERTED``, which leaves the memo in
    place: the lane will not judge the same pair or note again until one of them is edited.

    A merge is undone by clearing the loser's ``superseded_by``; a retype by putting back
    the type, project and audience the audit row recorded. Raises ``ValueError`` when the
    row cannot be undone safely (not applied, already changed by someone since, or applied
    before the audit row kept the old values), and changes nothing in that case."""
    row = db.get_curation(curation_id)
    if row is None:
        raise ValueError(f"no curation row #{curation_id}")
    if not row["applied"]:
        raise ValueError(f"curation row #{curation_id} was not applied ({row['verdict']})")
    detail = row["detail"] or {}
    if row["op"] == "pair":
        loser, winner = detail["loser"], detail["winner"]
        if not db.unsupersede_note(loser, winner):
            raise ValueError(f"n:{loser} is no longer superseded by n:{winner}; nothing to undo")
        done = f"n:{loser} is live again"
    else:
        if "from_audience" not in detail:
            raise ValueError(
                f"retype #{curation_id} predates revert support (no prior project or audience "
                "recorded); restore the note by hand"
            )
        (note,) = db.get_notes_by_ids([row["note_a"]])
        if note["type"] != "project" or note["project"] != detail["project"]:
            raise ValueError(
                f"n:{note['id']} is {note['type']}/{note['project']} now, not the project note "
                "the lane made; nothing to undo"
            )
        db.retype_note(
            note["id"],
            type=detail["from_type"],
            project=detail["from_project"],
            audience=detail["from_audience"],
        )
        done = f"n:{note['id']} is back to {detail['from_type']}"
    db.mark_curation_reverted(curation_id)
    return f"reverted #{curation_id}: {done}"


if __name__ == "__main__":  # pragma: no cover - manual runs
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="Run the notes curation lane, or inspect and undo it.")
    ap.add_argument("--max-judge", type=int, default=DEFAULT_MAX_JUDGE)
    ap.add_argument("--max-retype-judge", type=int, default=DEFAULT_MAX_RETYPE_JUDGE)
    ap.add_argument("--max-apply", type=int, default=DEFAULT_MAX_APPLY)
    ap.add_argument("--sim-floor", type=float, default=DEFAULT_SIM_FLOOR)
    ap.add_argument(
        "--list",
        nargs="?",
        type=int,
        const=20,
        metavar="N",
        help="print the N newest applied rows (default 20) instead of running the lane",
    )
    ap.add_argument("--revert", type=int, metavar="ID", help="undo one applied row (see --list)")
    args = ap.parse_args()
    if args.list is not None or args.revert is not None:
        from ingestion.db import Database

        cli_db = Database(require_setting("SYNAPSE_DB_URL"))
        try:
            if args.revert is not None:
                print(revert_curation(cli_db, args.revert))
            else:
                print("\n".join(list_applied(cli_db, args.list)) or "nothing applied yet")
        finally:
            cli_db.close()
    else:
        print(
            run_lane(
                sim_floor=args.sim_floor,
                max_judge=args.max_judge,
                max_retype_judge=args.max_retype_judge,
                max_apply=args.max_apply,
            )
        )
