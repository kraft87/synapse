# mypy: ignore-errors
"""dream->skills autopilot: the lane decides the routine proposals itself (nightly, from
nightly.run_lane after drafting and the #229 auto-reject/expiry; entries: run, reconcile).

Policy (approved 2026-11-06). Every drafted proposal a person has not touched is read by
an Opus REVIEWER (the lane's judge, skill_measure.run_judge) and by an independent VERIFIER
of another model family (ingestion.llm_client, stage SKILLS_VERIFY) that never sees the
review and is asked twice, with the evidence in both orders.

  apply   the reviewer accepts, the verifier passes in BOTH orders, and the change is
          one of:
            new_skill    a derive that met the recurrence gate (skill_ledger._recurrence:
                         2+ sessions on 2+ days), a usable SKILL.md of at most 8 KB
            description  a retune whose draft changes ONLY the frontmatter description
                         (skill_doc.only_description_changed), on a skill not pinned
          At most NIGHT_APPLY_CAP a night, new skills at most NEW_SKILL_NIGHT_CAP a night
          and NEW_SKILL_WEEK_CAP a week. Applied through the review accept's own write
          (skill_sync_routes.apply_candidate, actor='auto'): skill_history keeps the body
          it replaced, two-way sync delivers it, nothing grounded is written.
  reject  the reviewer rejects and the verifier fails it in both orders: 'rejected',
          reject_reason 'auto:review', the 30-day cooldown, no grounded -3 (as #229's
          auto-reject), so new evidence or `skill_review.py revive` can bring it back.
  ask     everything else stays 'proposed' with the reviewer's recommendation (#229's queue
          and report): body edits, pinned skills, a derive without recurrence,
          reviewer/verifier disagreement, a verifier that could not answer.

A row is judged once per (evidence_hash, draft_hash): a re-scan of the same sessions is
not new evidence, so a reopened row is not re-judged on what it was judged on. A stored
'apply' is re-checked against the catalog before it is acted on (a sibling applied since
may cover it now). A row a person has acted on (accept/reject/revive) is never decided here.

Rollback (reconcile, every night before the review): within ROLLBACK_DAYS of an apply,
  - a fire of the changed skill after the apply that was dismissed, or that post_fire
    judged a deviation/fight -> the old body comes back (a new skill is retired), the row
    goes to 'rejected' (auto_rollback);
  - the owner published over it: a description change whose description is gone ->
    rolled back as owner_edit (the owner's version stands); a retired auto-created skill
    -> owner_retired. An owner edit that keeps the change is not a rollback;
  - a machine held an UNPUBLISHED local edit the auto change pulled over (the sync client
    records it in skill_history as 'disk_overwrite'): a description change is re-applied
    on top of the local body when that body kept the old description (both kept), else
    the local body is restored and the row goes back to 'proposed' (local_edit).
Two rollbacks (other than local_edit) in ROLLBACK_DAYS trip the breaker: the stage runs as
shadow until a person runs `skill_review.py autopilot resume`, and the report says so.

Mode: env SKILLS_AUTOREVIEW = off | shadow | live (DEFAULT_MODE when unset). shadow
records what it would do and changes nothing; off skips the stage. Needs schema 064
(skill_ledger.AUTOPILOT_PROBE); until it is applied the stage logs one line and skips.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
from datetime import timedelta

from psycopg.rows import dict_row
from psycopg.types.json import Json

from . import config
from . import skill_doc as D
from . import skill_ledger as L
from . import skill_measure as SM
from .skill_draft import missing_phrasings, target_skill
from .struggle_arc import NEGATIVE_LIST

MODES = ("off", "shadow", "live")
#: The calibration gate decides this (see the PR that introduced the stage): 'live' only
#: when no proposal a person rejected would have been applied.
DEFAULT_MODE = "shadow"
NIGHT_APPLY_CAP = 3
NEW_SKILL_NIGHT_CAP = 1
NEW_SKILL_WEEK_CAP = 3
REVIEW_CAP = 5  # rows judged per night (1 reviewer + 2 verifier calls each)
ROLLBACK_DAYS = 14
BREAKER_ROLLBACKS = 2
MAX_NEW_SKILL_BYTES = 8192
COOLDOWN_DAYS = 30
DIFF_CHARS = 4000
CATALOG_DESC_CHARS = 220
EVIDENCE_CAP = 8

AUTO_TIERS = ("new_skill", "description")
#: Rollback reasons that say nothing about the change's quality: they don't count toward
#: the breaker.
NEUTRAL_ROLLBACKS = ("local_edit",)
_HUMAN_ACTS = ("accept", "reject", "revive")


# ------------------------------------------------------------------ mode
def current_mode() -> str:
    m = os.environ.get("SKILLS_AUTOREVIEW", "").strip().lower() or DEFAULT_MODE
    if m not in MODES:
        print(f"(autoreview: SKILLS_AUTOREVIEW={m!r} is not one of {MODES}; using shadow)")
        return "shadow"
    return m


# ------------------------------------------------------------------ pure helpers
def evidence_hash(evidence) -> str:
    """What the row's evidence says, independent of how often it was re-scanned: the set of
    (session, class, signal) sightings. A re-scan of the same session on a later night adds
    an entry but not new evidence; a new session or a new kind of signal does."""
    keys = sorted(
        {
            (str(e.get("session_id")), str(e.get("class")), str(e.get("signal")))
            for e in evidence or []
            if isinstance(e, dict) and e.get("session_id")
        }
    )
    return hashlib.sha256(json.dumps(keys).encode()).hexdigest()[:32]


def human_touched(evidence) -> bool:
    """A person accepted, rejected or revived this row: it is theirs to decide."""
    return any(
        isinstance(e, dict)
        and e.get("class") == "grounded"
        and not e.get("session_id")
        and e.get("signal") in _HUMAN_ACTS
        for e in evidence or []
    )


def evidence_items(evidence, cap: int = EVIDENCE_CAP) -> list[str]:
    """One line per distinct sighting (quote, missing phrasing, why), newest last."""
    lines: list[str] = []
    for e in evidence or []:
        if not isinstance(e, dict) or e.get("class") == "grounded" and not e.get("session_id"):
            continue
        bits = [str(e.get("signal") or "?")]
        for key in ("quote", "missing_phrasing", "phrasing", "why"):
            if e.get(key):
                bits.append(f"{key}: {str(e[key])[:240]}")
        line = "- " + "; ".join(bits)
        if line not in lines:
            lines.append(line)
    return lines[-cap:] or ["- (no detail recorded)"]


def catalog_hash(catalog: list[tuple[str, str]]) -> str:
    return hashlib.sha256(json.dumps(sorted(catalog)).encode()).hexdigest()[:16]


def catalog_text(catalog: list[tuple[str, str]], exclude: str | None) -> str:
    rows = [
        f"- {n}: {' '.join(str(d or '').split())[:CATALOG_DESC_CHARS]}"
        for n, d in sorted(catalog)
        if n != exclude
    ]
    return "\n".join(rows) or "(none)"


def what(row: dict) -> str:
    if row["kind"] == "derive":
        return f"new skill {row['name']}"
    if row["kind"] == "consolidate":
        return f"merge {row['name']}"
    return f"{target_skill(row)} {row.get('direction') or 'retune'}"


def change_block(row: dict, registry_body: str | None) -> str:
    """The change as the reviewer and verifier read it: the draft's unified diff against
    the current skill (a new skill's whole draft); with no draft yet, the detector's
    proposed patch or what the proposal says it wants."""
    body = row.get("proposal_body") or ""
    if body:
        if row["kind"] == "retune" and registry_body:
            diff = "".join(
                difflib.unified_diff(
                    registry_body.splitlines(keepends=True),
                    body.splitlines(keepends=True),
                    fromfile="current/SKILL.md",
                    tofile="proposed/SKILL.md",
                )
            )
            return "THE CHANGE (unified diff against the current skill):\n" + diff[:DIFF_CHARS]
        return "THE DRAFTED SKILL.md:\n" + body[:DIFF_CHARS]
    parts = ["THE CHANGE (not drafted yet; judge what it proposes):"]
    if row["kind"] == "retune" and registry_body is not None:
        parts.append(f"current description: {SM.skill_description(registry_body)[:600]}")
    if (row.get("proposed_patch") or "").strip():
        parts.append("proposed patch:\n" + row["proposed_patch"].strip()[:1500])
    phrases = missing_phrasings(row.get("evidence"))
    if row.get("direction") == "widen" and phrases:
        parts.append("make the description also cover: " + "; ".join(phrases))
    trig = row.get("trigger_phrasings") or []
    if row["kind"] == "derive" and trig:
        parts.append("trigger phrasings: " + "; ".join(str(t) for t in trig[:8]))
    if row["kind"] == "derive" and row.get("signature"):
        parts.append(f"observed procedure: {str(row['signature'])[:400]}")
    return "\n".join(parts)


REVIEWER_PROMPT = """You review one proposed change to a Claude Code skill (a SKILL.md procedure
document) that a nightly job mined from real sessions. If you accept it, it is applied
without the skill's owner looking at it first, so accept only what the owner would plainly
want.

Accept when the change is grounded in the evidence, reusable beyond one task, and does what
the evidence asks without inventing tools, commands or paths. Reject when it is not: a
one-off incident, a gap the current description or another skill already covers, work
that belongs in a script or a note rather than a skill, evidence about something else, or a
change to a skill that does not exist. Never propose these:
{negative_list}

SKILLS ALREADY INSTALLED (name: description):
{catalog}

PROPOSAL #{id}: {what}
SUMMARY: {summary}
EVIDENCE FROM THE SESSIONS:
{evidence}

{change}

Output ONLY a JSON object, no prose:
{{"verdict": "accept" | "reject", "reason": "one short sentence in plain words, no markdown"}}"""


VERIFIER_PROMPT = """You check one proposed change to an AI agent's skill (a SKILL.md procedure
document). Answer four questions strictly from the material below.

EVIDENCE FROM THE SESSIONS (excerpts):
{evidence}

SKILLS ALREADY INSTALLED (name: description):
{catalog}

{change}

Q1 traces_to_evidence: does everything the change adds or alters follow from the evidence
   above? (When the change is described rather than drafted, judge what it describes.)
Q2 invents: does it introduce a tool, command, file path, flag or capability that appears
   neither in the evidence nor in the current skill?
Q3 steals_from: would it make this skill fire for requests that one of the installed skills
   listed above already covers? Give that skill's exact name, or "" when none does.
Q4 negative_list: is it material that should never become a skill?
{negative_list}

Output ONLY a JSON object, no prose:
{{"traces_to_evidence": true|false, "invents": true|false, "steals_from": "",
 "negative_list": true|false, "note": "one short sentence"}}"""


def reviewer_prompt(row: dict, registry_body: str | None, catalog) -> str:
    return REVIEWER_PROMPT.format(
        negative_list=NEGATIVE_LIST,
        catalog=catalog_text(catalog, target_skill(row) if row["kind"] == "retune" else None),
        id=row["id"],
        what=what(row),
        summary=" ".join(str(row.get("summary") or "(none)").split())[:600],
        evidence="\n".join(evidence_items(row.get("evidence"))),
        change=change_block(row, registry_body),
    )


def verifier_prompt(row: dict, registry_body: str | None, catalog, *, reverse: bool) -> str:
    items = evidence_items(row.get("evidence"))
    return VERIFIER_PROMPT.format(
        evidence="\n".join(reversed(items) if reverse else items),
        catalog=catalog_text(catalog, target_skill(row) if row["kind"] == "retune" else None),
        change=change_block(row, registry_body),
        negative_list=NEGATIVE_LIST,
    )


def _bool(v) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "yes"):
        return True
    if isinstance(v, str) and v.strip().lower() in ("false", "no"):
        return False
    return None


def parse_reviewer(raw: str) -> dict | None:
    obj = SM._extract_json(raw or "") or {}
    verdict = str(obj.get("verdict") or "").strip().lower()
    reason = " ".join(str(obj.get("reason") or "").split())[:300]
    if verdict not in ("accept", "reject") or not reason:
        return None
    return {"verdict": verdict, "reason": reason}


def parse_verifier(raw: str, catalog_names, target: str | None) -> dict | None:
    """{clean, traces_to_evidence, invents, steals_from, negative_list, note}, or None when
    the answer is not usable. steals_from counts only when it names another installed
    skill."""
    obj = SM._extract_json(raw or "") or {}
    traces, invents, neg = (
        _bool(obj.get("traces_to_evidence")),
        _bool(obj.get("invents")),
        _bool(obj.get("negative_list")),
    )
    if traces is None or invents is None or neg is None:
        return None
    steals = str(obj.get("steals_from") or "").strip()
    if steals not in set(catalog_names) or steals == target:
        steals = ""
    return {
        "clean": bool(traces and not invents and not steals and not neg),
        "traces_to_evidence": traces,
        "invents": invents,
        "steals_from": steals,
        "negative_list": neg,
        "note": " ".join(str(obj.get("note") or "").split())[:240],
    }


def combine_verifier(first: dict | None, second: dict | None) -> str:
    """'pass' when clean in both orders, 'fail' when flagged in both, 'split' when the two
    orders disagree, 'error' when either answer is missing."""
    if first is None or second is None:
        return "error"
    if first["clean"] and second["clean"]:
        return "pass"
    if not first["clean"] and not second["clean"]:
        return "fail"
    return "split"


def decide(tier: str, reviewer: str | None, verifier: str) -> str:
    """apply | reject | ask (see the module docstring)."""
    if reviewer == "accept" and verifier == "pass" and tier in AUTO_TIERS:
        return "apply"
    if reviewer == "reject" and verifier == "fail":
        return "reject"
    return "ask"


def verifier_flags(v: dict | None) -> str:
    """What the verifier objected to, in plain words (for the report and the ask)."""
    if not v:
        return "no answer"
    out = []
    if not v.get("traces_to_evidence"):
        out.append("not traced to the evidence")
    if v.get("invents"):
        out.append("invents tools or paths")
    if v.get("steals_from"):
        out.append(f"overlaps {v['steals_from']}")
    if v.get("negative_list"):
        out.append("one-off or preference material")
    return ", ".join(out) or "clean"


# ------------------------------------------------------------------ tiers (preflight)
def tier_for(row: dict, plan: dict, pinned: bool) -> tuple[str, list[str]]:
    """(tier, notes) for a drafted row whose plan has no issues. new_skill and description
    may be applied; every other tier asks."""
    body = row.get("proposal_body") or ""
    if row["kind"] == "derive":
        if not L._recurrence(row.get("evidence") or []):
            return "derive_no_recurrence", ["seen on fewer than 2 days"]
        problem = D.new_skill_problem(body)
        if problem:
            return "derive_unusable", [problem]
        if len(body.encode("utf-8")) > MAX_NEW_SKILL_BYTES:
            return "derive_too_big", [f"{len(body.encode('utf-8'))} bytes"]
        return "new_skill", []
    if row["kind"] == "retune":
        if pinned:
            return "pinned", ["the skill is pinned"]
        if plan.get("registry_body") is not None and D.only_description_changed(
            plan["registry_body"], body
        ):
            return "description", []
        return "body_edit", ["changes the skill's body"]
    return "consolidate", ["merges are done by hand"]


# ------------------------------------------------------------------ LLM seams
def _reviewer_call(prompt: str) -> str:
    """The reviewer (the lane's judge: claude CLI, config.JUDGE_MODEL). Tests stub it."""
    return SM.run_judge(prompt)


def verifier_model() -> str | None:
    """The verifier's model, or None when it would not be independent of the reviewer
    (the extraction client is the claude CLI, or the stage resolves to a Claude model)."""
    from ingestion.llm_client import stage_model

    if os.environ.get("SYNAPSE_LLM_PROVIDER", "").strip().lower() != "openai":
        return None
    model = stage_model("SKILLS_VERIFY")
    if not model or "claude" in model.lower() or model.lower().startswith("anthropic/"):
        return None
    return model


def _verifier_call(prompt: str) -> str:
    """One verifier answer (ingestion.llm_client, OpenAI-compatible). Tests stub it."""
    from ingestion.llm_client import create_llm_client

    resp = create_llm_client().messages.create(
        model=verifier_model(),
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return str(resp.content[0].text)


def judge(row: dict, registry_body: str | None, catalog) -> dict:
    """Run the reviewer once and the verifier in both evidence orders. Never raises:
    a failed call is recorded as missing ('error' for the verifier, None for the reviewer)."""
    names = [n for n, _ in catalog]
    target = target_skill(row) if row["kind"] == "retune" else None
    try:
        rev = parse_reviewer(_reviewer_call(reviewer_prompt(row, registry_body, catalog)))
    except Exception as e:
        print(f"  reviewer failed for cand#{row['id']}: {e}")
        rev = None
    model = verifier_model()
    orders: list[dict | None] = []
    if model is None:
        orders = [None, None]
    else:
        for reverse in (False, True):
            try:
                raw = _verifier_call(verifier_prompt(row, registry_body, catalog, reverse=reverse))
                orders.append(parse_verifier(raw, names, target))
            except Exception as e:
                print(f"  verifier failed for cand#{row['id']}: {e}")
                orders.append(None)
    return {
        "reviewer": ({**rev, "model": config.JUDGE_MODEL} if rev else None),
        "verifier": {
            "result": combine_verifier(orders[0], orders[1])
            if model
            else "error",
            "orders": orders,
            "model": model or "unavailable",
        },
    }
