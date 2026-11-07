# mypy: ignore-errors
"""dream->skills autopilot: the lane decides the routine proposals itself (nightly, from
nightly.run_lane after drafting and the #229 auto-reject/expiry; entries: run, reconcile).

Policy (approved 2026-11-06). Every drafted proposal a person has not touched is read by
an Opus REVIEWER (the stripped claude CLI call, claude_cli.run) and by an independent VERIFIER
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
from contextlib import contextmanager
from datetime import UTC, date, datetime

from psycopg.rows import dict_row
from psycopg.types.json import Json

from . import claude_cli, config
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
        if not isinstance(e, dict) or (e.get("class") == "grounded" and not e.get("session_id")):
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
            if len(SM.skill_description(body)) > D.MAX_DESCRIPTION:
                return "description_unusable", ["the description is over the length cap"]
            return "description", []
        return "body_edit", ["changes the skill's body"]
    return "consolidate", ["merges are done by hand"]


# ------------------------------------------------------------------ LLM seams
def reviewer_model() -> str:
    """The reviewer's model: env SKILLS_AUTOREVIEW_MODEL, else the lane's judge model."""
    return os.environ.get("SKILLS_AUTOREVIEW_MODEL", "").strip() or config.JUDGE_MODEL


def _reviewer_call(prompt: str) -> str:
    """The reviewer: the stripped claude CLI call (claude_cli.run), the same one the
    calibration script makes. Needs the CLI logged in on the machine that runs the stage
    (see run_host; the dream container has none). Tests stub it."""
    return claude_cli.run(prompt, reviewer_model())


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
    if model is None or rev is None:  # no independent verifier, or nothing to verify
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
        "reviewer": ({**rev, "model": reviewer_model()} if rev else None),
        "verifier": {
            "result": combine_verifier(orders[0], orders[1]) if model else "error",
            "orders": orders,
            "model": model or "unavailable",
        },
    }


# ================================================================== the stage
@contextmanager
def dict_rows(conn):
    """`conn` with dict rows for the block (skill_sync_routes' write paths expect them)."""
    prev = conn.row_factory
    conn.row_factory = dict_row
    try:
        yield conn
    finally:
        conn.row_factory = prev


def _routes():
    """Lazy: the dream image ships mcp_server/, and importing it at module load would drag
    the HTTP stack into every lane import (review_queue does the same)."""
    from mcp_server import skill_sync_routes as R

    return R


_LOCK_KEY = 0x5C1A0064  # advisory lock: one autopilot run at a time (host cron and nightly)


def _plain(text, cap: int = 240) -> str:
    t = " ".join(str(text or "").split()).replace("\N{EM DASH}", ",").replace("\N{EN DASH}", "-")
    return t if len(t) <= cap else t[: cap - 3].rstrip() + "..."


def _config(c) -> dict:
    r = c.execute("SELECT config FROM skills_lane.skill_scan_cursor WHERE id=1").fetchone()
    return (r["config"] if r else None) or {}


def _patch_config(c, key: str, value: dict) -> None:
    """Merge `value` into config[key] (the cursor's JSONB parks each stage's own state)."""
    merged = {**(_config(c).get(key) or {}), **value}
    c.execute(
        "UPDATE skills_lane.skill_scan_cursor SET config = COALESCE(config,'{}'::jsonb) || %s::jsonb, "
        "updated_at=now() WHERE id=1",
        (Json({key: merged}),),
    )


def _event(c, review_id: int, ev: dict) -> None:
    c.execute(
        "UPDATE skills_lane.skill_auto_reviews SET events = events || %s::jsonb WHERE id=%s",
        (Json([ev]), review_id),
    )


# ------------------------------------------------------------------ breaker
def breaker_state(c) -> dict:
    """tripped when BREAKER_ROLLBACKS rollbacks (a local_edit does not count) fall in the
    last ROLLBACK_DAYS and after the last `autopilot resume`."""
    resumed = (_config(c).get("autopilot") or {}).get("resumed_at")
    rows = c.execute(
        "SELECT id, candidate_id, skill, rollback_reason FROM skills_lane.skill_auto_reviews "
        "WHERE rolled_back_at IS NOT NULL AND rolled_back_at >= now() - make_interval(days => %s) "
        "AND rollback_reason <> ALL(%s) "
        "AND (%s::timestamptz IS NULL OR rolled_back_at > %s::timestamptz) ORDER BY id",
        (ROLLBACK_DAYS, list(NEUTRAL_ROLLBACKS), resumed, resumed),
    ).fetchall()
    return {"tripped": len(rows) >= BREAKER_ROLLBACKS, "rollbacks": [dict(r) for r in rows]}


def resume(c) -> dict:
    """`skill_review.py autopilot resume`: the breaker counts afresh from now."""
    _patch_config(c, "autopilot", {"resumed_at": datetime.now(UTC).isoformat(), "tripped": False})
    return {"status": "resumed", **breaker_state(c)}


def status(c) -> dict:
    """What `skill_review.py autopilot status` shows."""
    ap = _config(c).get("autopilot") or {}
    recent = c.execute(
        "SELECT r.id, r.candidate_id, r.night, r.tier, r.verdict, r.skill, r.mode, "
        "r.rolled_back_at, r.rollback_reason, c.name AS candidate "
        "FROM skills_lane.skill_auto_reviews r JOIN skills_lane.skill_gap_candidates c "
        "ON c.id = r.candidate_id ORDER BY r.id DESC LIMIT 12"
    ).fetchall()
    counts = c.execute(
        "SELECT verdict, count(*) AS n FROM skills_lane.skill_auto_reviews "
        "WHERE created_at >= now() - make_interval(days => %s) GROUP BY verdict",
        (ROLLBACK_DAYS,),
    ).fetchall()
    return {
        "status": "ok",
        "last_mode": ap.get("last_mode"),
        "last_run_at": ap.get("last_run_at"),
        "resumed_at": ap.get("resumed_at"),
        "breaker": breaker_state(c),
        "counts_14d": {r["verdict"]: r["n"] for r in counts},
        "recent": [{**r, "night": str(r["night"])} for r in recent],
        "last_report": _config(c).get("last_report"),
    }


# ------------------------------------------------------------------ caps
def cap_reason(c, tier: str, night) -> str | None:
    """Why one more auto-apply is not allowed now, or None."""
    n_night = c.execute(
        "SELECT count(*) AS n FROM skills_lane.skill_auto_reviews "
        "WHERE verdict='applied' AND night=%s",
        (night,),
    ).fetchone()["n"]
    if n_night >= NIGHT_APPLY_CAP:
        return f"{NIGHT_APPLY_CAP} auto-applies tonight already"
    if tier == "new_skill":
        n_new = c.execute(
            "SELECT count(*) FILTER (WHERE night=%s) AS night, count(*) AS week "
            "FROM skills_lane.skill_auto_reviews WHERE verdict='applied' AND tier='new_skill' "
            "AND applied_at >= now() - interval '7 days'",
            (night,),
        ).fetchone()
        if n_new["night"] >= NEW_SKILL_NIGHT_CAP:
            return f"{NEW_SKILL_NIGHT_CAP} new skill tonight already"
        if n_new["week"] >= NEW_SKILL_WEEK_CAP:
            return f"{NEW_SKILL_WEEK_CAP} new skills this week already"
    return None


# ------------------------------------------------------------------ reconcile
def _known_hashes(c, skill: str, review: dict, current: str | None) -> set[str]:
    """Hashes of every body the registry has held (or holds): a machine pulling a body
    the registry already had is not a local edit."""
    known = {h for h in (review["previous_body_hash"], review["applied_body_hash"], current) if h}
    for r in c.execute(
        "SELECT body FROM skills_lane.skill_history WHERE name=%s AND op <> 'disk_overwrite'",
        (skill,),
    ).fetchall():
        known.add(D.body_hash(r["body"] or ""))
    return known


def _unpublished_edit(c, review: dict, current: str | None) -> dict | None:
    """The newest local body a pull overwrote after the apply that the registry never held
    (the sync client records it as 'disk_overwrite' just before it overwrites), unless
    reconcile already dealt with it."""
    known = _known_hashes(c, review["skill"], review, current)
    done = {e.get("hash") for e in review["events"] or [] if isinstance(e, dict)}
    for r in c.execute(
        "SELECT id, body, content_modified_at FROM skills_lane.skill_history "
        "WHERE name=%s AND op='disk_overwrite' AND recorded_at >= %s ORDER BY id DESC",
        (review["skill"], review["applied_at"]),
    ).fetchall():
        h = D.body_hash(r["body"] or "")
        if r["body"] and h not in known and h not in done:
            return {**r, "hash": h}
    return None


def _bad_signal(c, review: dict) -> str | None:
    """A rollback reason when the changed skill fired after the apply and the user pushed
    back: a dismissed fire, or a post_fire deviation/fight on a later day."""
    skill, since = review["skill"], review["applied_at"]
    if c.execute(
        "SELECT 1 FROM skills_lane.skill_usage WHERE skill=%s AND fired_at > %s "
        "AND dismissed IS TRUE LIMIT 1",
        (skill, since),
    ).fetchone():
        return "dismissed"
    day = since.date().isoformat()
    for row in c.execute(
        "SELECT evidence FROM skills_lane.skill_gap_candidates "
        "WHERE kind='retune' AND source_detector='post_fire' AND (name=%s OR %s = ANY(target_skills))",
        (skill, skill),
    ).fetchall():
        for e in row["evidence"] or []:
            if (
                isinstance(e, dict)
                and e.get("signal") == "post_fire_deviation"
                and str(e.get("date") or "") > day
            ):
                return "post_fire_deviation"
    return None


def _re_propose(c, review: dict) -> None:
    """The candidate goes back to the queue for a person; the autopilot will not apply this
    evidence + draft again (the memo sees the rolled-back review)."""
    c.execute(
        "UPDATE skills_lane.skill_gap_candidates SET status='proposed', applied_by=NULL, "
        "auto_review_id=NULL, proposed_at=now(), updated_at=now() WHERE id=%s",
        (review["candidate_id"],),
    )


def _repair_local_edit(c, R, review: dict, edit: dict, reg: dict) -> dict:
    """A machine's unpublished edit was pulled over by the auto change. Neither is lost:
    description change + local body that kept the OLD description -> the new description
      is written onto the local body (both kept);
    anything else -> the local body is published back as the newest version and the
      candidate returns to the queue for a person ('local_edit', a neutral rollback)."""
    skill, cid = review["skill"], review["candidate_id"]
    local = edit["body"]
    prev = c.execute(
        "SELECT body FROM skills_lane.skill_history WHERE id=%s", (review["history_id"],)
    ).fetchone()
    cand = c.execute(
        "SELECT proposal_body FROM skills_lane.skill_gap_candidates WHERE id=%s", (cid,)
    ).fetchone()
    merged = None
    if review["tier"] == "description" and prev and cand and cand["proposal_body"]:
        old_desc = SM.skill_description(prev["body"] or "")
        if SM.skill_description(local) == old_desc:
            merged = D.replace_description(local, SM.skill_description(cand["proposal_body"]))
    if merged is not None and D.only_description_changed(local, merged):
        R.lane_write(c, skill, merged)
        _event(c, review["id"], {"type": "merged", "hash": edit["hash"], "history_id": edit["id"]})
        c.commit()
        return {"id": cid, "skill": skill, "what": "merged", "review_id": review["id"]}
    R.lane_write(c, skill, local)
    c.execute(
        "UPDATE skills_lane.skill_auto_reviews SET rolled_back_at=now(), rollback_reason='local_edit' "
        "WHERE id=%s",
        (review["id"],),
    )
    _event(c, review["id"], {"type": "restored", "hash": edit["hash"], "history_id": edit["id"]})
    _re_propose(c, review)
    c.commit()
    return {"id": cid, "skill": skill, "what": "restored", "review_id": review["id"]}


def reconcile(c) -> dict:
    """Look at every auto-apply from the last ROLLBACK_DAYS that is still standing and undo
    what went wrong (module docstring). `c` is a dict-row connection."""
    R = _routes()
    out: dict = {"rolled_back": [], "conflicts": [], "checked": 0}
    reviews = c.execute(
        "SELECT r.*, k.name AS candidate_name, k.kind AS candidate_kind, k.direction "
        "FROM skills_lane.skill_auto_reviews r JOIN skills_lane.skill_gap_candidates k "
        "ON k.id = r.candidate_id WHERE r.verdict='applied' AND r.rolled_back_at IS NULL "
        "AND r.applied_at >= now() - make_interval(days => %s) ORDER BY r.id",
        (ROLLBACK_DAYS,),
    ).fetchall()
    c.commit()
    for rv in reviews:
        out["checked"] += 1
        reg = c.execute(
            "SELECT scope, status, COALESCE(body,'') AS body FROM skills_lane.skill_registry "
            "WHERE name=%s",
            (rv["skill"],),
        ).fetchone()
        c.commit()
        cur_hash = D.body_hash(reg["body"]) if reg else None
        reason = None
        if reg is None or reg["status"] == "retired":
            reason = "owner_retired" if rv["tier"] == "new_skill" else "skill_gone"
        else:
            edit = _unpublished_edit(c, rv, cur_hash)
            c.commit()
            if edit is not None:
                item = _repair_local_edit(c, R, rv, edit, reg)
                out["conflicts"].append({**item, "what_skill": _what_row(rv)})
                continue
            if rv["tier"] == "description" and cur_hash != rv["applied_body_hash"]:
                cand = c.execute(
                    "SELECT proposal_body FROM skills_lane.skill_gap_candidates WHERE id=%s",
                    (rv["candidate_id"],),
                ).fetchone()
                c.commit()
                kept = SM.skill_description(reg["body"]) == SM.skill_description(
                    (cand or {}).get("proposal_body") or ""
                )
                if not kept:
                    reason = "owner_edit"
            if reason is None:
                reason = _bad_signal(c, rv)
                c.commit()
        if reason is None:
            continue
        res = R.revert_applied(c, rv["candidate_id"], reason=reason, actor="auto")
        if res.get("status") == "reverted":
            print(f"  rolled back cand#{rv['candidate_id']} ({rv['skill']}): {reason}")
            out["rolled_back"].append(
                {
                    "id": rv["candidate_id"],
                    "skill": rv["skill"],
                    "reason": reason,
                    "action": res["action"],
                    "what": _what_row(rv),
                }
            )
        else:
            print(f"  rollback of cand#{rv['candidate_id']} not done: {res.get('detail')}")
    return out


def _what_row(rv: dict) -> str:
    if rv["tier"] == "new_skill":
        return f"new skill {rv['skill']}"
    return f"{rv['skill']} description"


# ------------------------------------------------------------------ the review pass
_PROPOSED = """
SELECT {cols}, trigger_phrasings, signature, recommendation, source_detector
  FROM skills_lane.skill_gap_candidates
 WHERE status='proposed' AND kind IN ('derive','retune') AND proposal_body IS NOT NULL
   AND (rejected_until IS NULL OR rejected_until < now())
 ORDER BY salience DESC NULLS LAST, score DESC, id"""


def _latest_review(c, cid: int, eh: str, dh: str | None) -> dict | None:
    return c.execute(
        "SELECT * FROM skills_lane.skill_auto_reviews WHERE candidate_id=%s AND evidence_hash=%s "
        "AND draft_hash IS NOT DISTINCT FROM %s ORDER BY id DESC LIMIT 1",
        (cid, eh, dh),
    ).fetchone()


def _complete(rv: dict | None) -> bool:
    """A stored judgement worth reusing: the reviewer answered and the verifier did not error."""
    return bool(
        rv and rv["reviewer"] and (rv["verifier"] or {}).get("result") in ("pass", "fail", "split")
    )


def _record(c, cid, night, mode, eh, dh, tier, pre, j, verdict, skill) -> int:
    return c.execute(
        "INSERT INTO skills_lane.skill_auto_reviews (candidate_id, night, mode, evidence_hash, "
        "draft_hash, tier, preflight, reviewer, verifier, verdict, skill) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
        (
            cid, night, mode, eh, dh, tier, Json(pre), Json(j.get("reviewer")),
            Json(j.get("verifier")), verdict, skill,
        ),
    ).fetchone()["id"]  # fmt: skip


def _recommendation(j: dict, dh: str | None, note: str) -> dict | None:
    rev = j.get("reviewer")
    if not rev:
        return None
    reason = _plain(rev["reason"], 200)
    if note:
        reason = _plain(f"{reason} ({note})", 280)
    return {
        "verdict": rev["verdict"],
        "reason": reason,
        "draft_hash": dh,
        "model": rev.get("model"),
        "source": "autopilot",
    }


def review_pass(c, mode: str, night=None, cap: int = REVIEW_CAP) -> dict:
    """Judge (or reuse the memo of) each proposed, drafted, untouched row and act on the
    decision per `mode` ('shadow' records only). `c` is a dict-row connection."""
    R = _routes()
    night = night or date.fromisoformat(SM.scan_night())
    out: dict = {"applied": [], "rejected": [], "asked": [], "capped": [], "would": {}, "judged": 0}
    rows = c.execute(_PROPOSED.format(cols=R._CANDIDATE_COLS)).fetchall()
    regs = {
        r["name"]: r
        for r in c.execute(
            "SELECT name, description, status, pinned, lane_body_hash, COALESCE(body,'') AS body "
            "FROM skills_lane.skill_registry"
        ).fetchall()
    }
    catalog = [(n, r["description"] or "") for n, r in regs.items() if r["status"] == "active"]
    c.commit()
    for row in rows:
        if human_touched(row["evidence"]):
            continue
        cid, body = row["id"], row["proposal_body"]
        plan = R._plan_apply(c, row, body, None)
        c.commit()
        if plan["issues"]:  # unappliable as drafted: #229's auto-reject / re-draft owns it
            continue
        skill = plan["skill"]
        reg = regs.get(skill)
        tier, notes = tier_for(row, plan, bool(reg and reg["pinned"]))
        hand = bool(
            reg
            and plan["exists"]
            and (reg["lane_body_hash"] is None or reg["lane_body_hash"] != D.body_hash(reg["body"]))
        )
        pre = {"notes": notes, "hand_written": hand, "scope": plan["scope"]}
        eh, dh = evidence_hash(row["evidence"]), D.body_hash(body)
        prior = _latest_review(c, cid, eh, dh)
        c.commit()
        if _complete(prior):
            j = {"reviewer": prior["reviewer"], "verifier": prior["verifier"]}
        elif out["judged"] >= cap:
            continue
        else:
            j = judge(row, plan["registry_body"], catalog)
            out["judged"] += 1
        rev_verdict = (j["reviewer"] or {}).get("verdict")
        ver = (j["verifier"] or {}).get("result", "error")
        decision = decide(tier, rev_verdict, ver)
        note = ""
        if prior and prior["rolled_back_at"] is not None and decision == "apply":
            decision, note = "ask", "rolled back before; needs a person"
        failed = j["reviewer"] is None or ver == "error"
        if failed:
            decision = "ask"
        if mode != "live":
            verdict = {"apply": "would_apply", "reject": "would_reject"}.get(decision, "ask")
        else:
            verdict = {"apply": "applied", "reject": "rejected"}.get(decision, "ask")
        if verdict == "applied":
            why = cap_reason(c, tier, night)
            c.commit()
            if why:
                verdict, note = "capped", why
        if failed:
            verdict = "failed"
        if verdict == "applied":
            res = R.apply_candidate(
                c,
                cid,
                None,
                None,
                False,
                actor="auto",
                review={
                    "night": night,
                    "mode": mode,
                    "evidence_hash": eh,
                    "draft_hash": dh,
                    "tier": tier,
                    "preflight": pre,
                    "reviewer": j["reviewer"],
                    "verifier": j["verifier"],
                },
            )
            if res.get("status") != "promoted":
                verdict, note = "ask", _plain(res.get("detail") or "refused", 160)
            else:
                out["applied"].append(
                    {
                        "id": cid,
                        "review_id": res["auto_review_id"],
                        "skill": skill,
                        "tier": tier,
                        "what": what(row),
                        "reason": _plain(j["reviewer"]["reason"], 160),
                    }
                )
                print(f"  auto-applied cand#{cid} ({what(row)}), review {res['auto_review_id']}")
                continue
        if prior and prior["verdict"] == verdict and prior["mode"] == mode and not failed:
            continue  # nothing new to record or say
        _record(c, cid, night, mode, eh, dh, tier, pre, j, verdict, skill)
        if verdict == "rejected":
            c.execute(
                "UPDATE skills_lane.skill_gap_candidates SET status='rejected', "
                "reject_reason='auto:review', "
                "rejected_until=now() + make_interval(days => %s), updated_at=now() "
                "WHERE id=%s AND status='proposed'",
                (COOLDOWN_DAYS, cid),
            )
            out["rejected"].append(
                {"id": cid, "what": what(row), "reason": _plain(j["reviewer"]["reason"], 160)}
            )
            print(f"  auto-rejected cand#{cid} ({what(row)})")
        elif (
            verdict in ("ask", "capped")
            and mode == "live"
            and L.queue_ready(c.connection if hasattr(c, "connection") else c)
        ):
            rec = _recommendation(j, dh, note or _ask_note(tier, hand, rev_verdict, ver, j))
            if rec:
                c.execute(
                    "UPDATE skills_lane.skill_gap_candidates SET recommendation = %s::jsonb || "
                    "jsonb_build_object('at', now()) WHERE id=%s AND status='proposed'",
                    (Json(rec), cid),
                )
            out["asked"].append(cid)
            if verdict == "capped":
                out["capped"].append({"id": cid, "what": what(row), "why": note})
        elif verdict in ("would_apply", "would_reject"):
            out["would"][verdict] = out["would"].get(verdict, 0) + 1
        c.commit()
    return out


def _ask_note(tier: str, hand: bool, rev: str | None, ver: str, j: dict) -> str:
    """Why this one is not decided by the lane, in plain words (appended to the reason)."""
    if tier == "body_edit":
        return "a body edit to a hand-written skill" if hand else "a body edit"
    if tier == "pinned":
        return "the skill is pinned"
    if tier != "new_skill" and tier != "description":
        return tier.replace("_", " ")
    if rev == "accept" and ver in ("fail", "split"):
        return f"verifier disagrees: {verifier_flags((j.get('verifier') or {}).get('orders', [None])[0])}"
    if rev == "reject" and ver != "fail":
        return "the verifier did not confirm the rejection"
    return ""


# ------------------------------------------------------------------ entry
def run(conn, run_started=None, *, review: bool = True, mode: str | None = None) -> dict:
    """One autopilot pass: reconcile what it applied before, then (review=True) judge and act
    on the proposed queue. `conn` is a plain psycopg connection. Returns the pass's items
    (the report's auto lines) and counts."""
    mode = mode or current_mode()
    out: dict = {
        "mode": mode, "applied": [], "rejected": [], "asked": [], "capped": [], "would": {},
        "rolled_back": [], "conflicts": [], "breaker": None, "judged": 0, "ran": False,
    }  # fmt: skip
    if mode == "off":
        return out
    if not L.autopilot_ready(conn):
        print("(autoreview: schema 064 not applied; skipping)")
        out["skipped"] = "schema 064"
        return out
    with dict_rows(conn) as c:
        if not c.execute("SELECT pg_try_advisory_lock(%s) AS ok", (_LOCK_KEY,)).fetchone()["ok"]:
            c.commit()
            print("(autoreview: another run holds the lock; skipping)")
            out["skipped"] = "locked"
            return out
        try:
            c.commit()
            rec = reconcile(c)
            out["rolled_back"], out["conflicts"] = rec["rolled_back"], rec["conflicts"]
            br = breaker_state(c)
            effective = mode
            if br["tripped"] and mode == "live":
                effective = "shadow"
                out["breaker"] = {
                    "rollbacks": len(br["rollbacks"]),
                    "skills": [r["skill"] for r in br["rollbacks"]],
                }
                print("(autoreview: circuit breaker tripped; running as shadow)")
            _patch_config(
                c,
                "autopilot",
                {
                    "last_mode": effective,
                    "configured_mode": mode,
                    "last_run_at": datetime.now(UTC).isoformat(),
                    "tripped": br["tripped"],
                },
            )
            c.commit()
            if review:
                out.update({k: v for k, v in review_pass(c, effective).items()})
            out["ran"] = True
            out["effective_mode"] = effective
        finally:
            c.rollback()
            c.execute("SELECT pg_advisory_unlock(%s)", (_LOCK_KEY,))
            c.commit()
    return out


def flat_counts(res: dict) -> dict:
    """The run's counts as flat ints (dream_runs.counts, prefixed autoreview_)."""
    return {
        "applied": len(res.get("applied") or []),
        "rejected": len(res.get("rejected") or []),
        "asked": len(res.get("asked") or []),
        "capped": len(res.get("capped") or []),
        "rolled_back": len(res.get("rolled_back") or []),
        "conflicts": len(res.get("conflicts") or []),
        "judged": res.get("judged", 0),
        "would_apply": (res.get("would") or {}).get("would_apply", 0),
        "would_reject": (res.get("would") or {}).get("would_reject", 0),
        "breaker": 1 if res.get("breaker") else 0,
    }
