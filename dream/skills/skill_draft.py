# mypy: ignore-errors
"""dream->skills: give every proposal a complete, ready-to-apply SKILL.md.

Accept applies proposal_body as-is (mcp_server/skill_review_routes.py), so a proposal without
one can't be accepted. This pass runs once per nightly lane, after the detectors, over
'proposed' rows with no body yet — and retune rows whose draft went stale because the skill
changed underneath it — and stores proposal_body + base_body_hash (the hash of the registry
body the draft was made from; accept refuses when the registry has moved on):

  derive                    a full new SKILL.md (skill_derive.draft_skill); no base hash
  retune / widen            the CURRENT registry body with only the frontmatter description
                            rewritten to cover the evidence's missing_phrasing. The splice is
                            verified byte-identical outside the description, else no draft.
  retune / extend|fix|narrow
                            the current registry body with proposed_patch (or the evidence)
                            applied: frontmatter name unchanged, body non-empty, size sane

A draft that fails its checks is not stored: the row stays body-less, and the reason and the
attempt are recorded on the candidate (the draft-attempts migration; the review CLI shows the
reason). A failed
row waits 1, 2, 4... nights (at most 16 days) before it is tried again and rows with fewer
failures go first, so a few permanently failing rows can't use up the night's budget.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from dream import lane
from ingestion.llm_client import LLM_TRANSPORT_ERRORS

from . import skill_derive as SD
from . import skill_doc as D
from . import skill_ledger as L
from .skill_doc import DraftProblem, skill_description
from .skill_measure import _extract_json

DRAFT_CAP = 12  # drafts per nightly run (each is an Opus call); the rest wait a night
MAX_PHRASINGS = 8
MAX_CHANGE_LINES = 8
MAX_BASH = 20  # shell commands shown to the model for a new skill
MAX_BACKOFF_DAYS = 16  # a draft that keeps failing is retried this often at most

_WIDEN_PROMPT = """You maintain the trigger DESCRIPTION of one Claude Code skill. The model decides whether
to invoke a skill by soft-matching the user's request against this description. Users asked
for this skill's job in words the description does not cover.

SKILL: {name}
CURRENT DESCRIPTION:
{description}

PHRASINGS THE DESCRIPTION MISSES (from real sessions):
{phrasings}

Rewrite the description so it ALSO covers those phrasings. Keep everything it already
covers and its voice; add the missing trigger vocabulary compactly; do not describe new
capabilities. Every session carries every skill's description in its context, so spend few
characters: tighten the existing wording where you need room. One line, at most {max_chars}
characters (it is {current_chars} now), no line breaks.

Output ONLY a JSON object, no prose: {{"description": "..."}}"""

_REVISE_PROMPT = """You revise one Claude Code skill: a SKILL.md with YAML frontmatter, then a markdown
procedure. Real sessions showed it needs this change.

DIRECTION: {direction} (extend = add the missing coverage or steps; fix = correct a wrong step;
narrow = stop it applying where it shouldn't)
SUMMARY: {summary}
CHANGE TO MAKE:
{change}

CURRENT SKILL.md:
<<<
{current}
>>>

Rules:
- Apply ONLY the change above. Leave every other line exactly as it is.
- Keep the frontmatter `name` exactly as it is.
- Do not invent tools, commands or capabilities that are in neither the skill nor the change.
- Output ONLY the complete revised SKILL.md (frontmatter + body), nothing else."""


def target_skill(row: dict) -> str:
    """The registry skill a retune edits (target_skills[0], else the candidate name)."""
    return (row.get("target_skills") or [row["name"]])[0]


# ------------------------------------------------------------------ drafters
def missing_phrasings(evidence) -> list[str]:
    """What a widen must cover: each under_trigger entry's missing_phrasing (the judge's
    stated gap), falling back to the raw user phrasing on legacy entries."""
    out: list[str] = []
    for e in evidence or []:
        if not isinstance(e, dict) or e.get("signal") != "under_trigger":
            continue
        p = str(e.get("missing_phrasing") or e.get("phrasing") or "").strip()
        if p and p not in out:
            out.append(p)
    return out[:MAX_PHRASINGS]


def draft_widen(name: str, current: str, evidence) -> str:
    """`current` with only its description widened. Raises DraftProblem when no gap is stated,
    the skill has no description field, or the result touched anything else."""
    phrasings = missing_phrasings(evidence)
    old = skill_description(current)
    if not phrasings:
        raise DraftProblem("no missing phrasing is recorded in the evidence to widen for")
    if not old:
        raise DraftProblem("the registry body has no frontmatter description to widen")
    raw = lane.draft(
        _WIDEN_PROMPT.format(
            name=name,
            description=old,
            phrasings="\n".join(f"- {p}" for p in phrasings),
            max_chars=D.widen_limit(old),
            current_chars=len(old),
        )
    )
    obj = _extract_json(raw or "") or {}
    new = obj.get("description") if isinstance(obj, dict) else None
    if not isinstance(new, str):
        raise DraftProblem("the model's reply had no description")
    return D.widen_draft(current, new)


def _change_text(row: dict) -> str:
    """The change a revision applies: the detector's proposed_patch, else what the evidence
    says (judge why / verbatim quotes) under the summary."""
    if (row.get("proposed_patch") or "").strip():
        return row["proposed_patch"].strip()
    lines: list[str] = []
    for e in row.get("evidence") or []:
        if not isinstance(e, dict):
            continue
        for bit in (e.get("why"), e.get("quote") and f'user said: "{e["quote"]}"'):
            if bit and bit not in lines:
                lines.append(str(bit))
    lines = lines[-MAX_CHANGE_LINES:] or [row.get("summary") or "(no detail recorded)"]
    return "\n".join(f"- {ln}" for ln in lines)


def draft_revision(row: dict, current: str) -> str:
    """`current` with the extend/fix/narrow change applied. Raises DraftProblem when the
    model's revision fails the sanity checks (renamed, emptied, rewritten wholesale)."""
    raw = lane.draft(
        _REVISE_PROMPT.format(
            direction=row.get("direction") or "fix",
            summary=row.get("summary") or "",
            change=_change_text(row),
            current=current,
        )
    )
    body = D.strip_outer_fence(raw or "") + "\n"
    problem = D.revision_problem(current, body)
    if problem:
        raise DraftProblem(problem)
    return body


def draft_derive(row: dict) -> str:
    """A full new SKILL.md for a derive row from any detector; raises DraftProblem if the
    model's reply is not a usable one."""
    ev = [e for e in row.get("evidence") or [] if isinstance(e, dict)]
    quotes = [str(e["quote"]) for e in ev if e.get("quote")][:6]
    what = row.get("summary") or row["name"]
    bash = list(dict.fromkeys(h for e in ev for h in e.get("bash") or []))[:MAX_BASH]
    if quotes:
        what += "\nEvidence from the sessions:\n" + "\n".join(f"- {q}" for q in quotes)
    body = SD.draft_skill(
        {
            "procedure": row["name"],
            "what": what,
            "trigger_phrasings": row.get("trigger_phrasings") or [],
            "signature": row.get("signature") or "",
            "bash_evidence": bash,
            "n_sessions": len({e["session_id"] for e in ev if e.get("session_id")}),
        }
    )
    problem = D.new_skill_problem(body)
    if problem:
        raise DraftProblem(problem)
    return body


def draft(row: dict, registry_body: str | None) -> tuple[str, str | None]:
    """(proposal_body, base_body_hash) for one candidate row. Raises DraftProblem when it
    can't be drafted."""
    if row["kind"] == "derive":
        return draft_derive(row), None
    if not registry_body:
        raise DraftProblem(f"{target_skill(row)!r} is not in the skill registry")
    if row.get("direction") == "widen":
        body = draft_widen(target_skill(row), registry_body, row.get("evidence"))
    else:
        body = draft_revision(row, registry_body)
    return body, D.body_hash(registry_body)


# ------------------------------------------------------------------ nightly pass
_PENDING_SQL = """
SELECT c.id, c.kind, c.name, c.direction, c.target_skills, c.summary, c.signature,
       c.trigger_phrasings, c.evidence, c.proposed_patch, c.proposal_body, c.base_body_hash,
       r.body AS registry_body
  FROM skills_lane.skill_gap_candidates c
  LEFT JOIN skills_lane.skill_registry r
         ON c.kind = 'retune' AND r.name = COALESCE(c.target_skills[1], c.name)
 WHERE c.status = 'proposed' AND c.kind IN ('derive', 'retune')
 ORDER BY c.salience DESC NULLS LAST, c.score DESC, c.id"""


def needs_draft(row: dict) -> bool:
    """No body yet, or a retune draft pinned to a registry body that has since changed."""
    if not row.get("proposal_body"):
        return True
    base = row.get("base_body_hash")
    return bool(
        row["kind"] == "retune"
        and base
        and row.get("registry_body") is not None
        and base != D.body_hash(row["registry_body"])
    )


def _draft_state(conn, ids: list[int]) -> dict[int, tuple[int, datetime | None]]:
    """{id: (failed drafts, when the last one failed)}."""
    rows = conn.execute(
        "SELECT id, draft_attempts, last_draft_at FROM skills_lane.skill_gap_candidates "
        "WHERE id = ANY(%s)",
        (ids,),
    ).fetchall()
    return {cid: (attempts, at) for cid, attempts, at in rows}


def _backed_off(attempts: int, last_at: datetime | None, today) -> bool:
    """True while a row whose draft failed `attempts` times waits to be tried again: 1, 2, 4...
    days after the last failure, counted in calendar days so a run a few minutes earlier than
    last night's is not skipped for it."""
    if not attempts or last_at is None:
        return False
    wait = timedelta(days=min(2 ** (attempts - 1), MAX_BACKOFF_DAYS))
    return today < last_at.astimezone(UTC).date() + wait


def draft_pending(conn, cap: int = DRAFT_CAP) -> dict:
    """Draft up to `cap` proposed rows that need one, those with the fewest failed drafts
    first and skipping the ones still backing off after a failure. A row whose draft fails for
    its own reasons is counted, has the reason recorded and is skipped; a model that cannot be
    reached raises, since every later row would fail the same way."""
    cur = conn.cursor()
    cur.execute(_PENDING_SQL)
    cols = [d.name for d in cur.description]
    rows = [r for r in (dict(zip(cols, t, strict=True)) for t in cur.fetchall()) if needs_draft(r)]
    has_state = L.DRAFT_STATE.ready(conn, conn.info.dsn)  # the draft-attempts migration
    state = _draft_state(conn, [r["id"] for r in rows]) if has_state else {}
    today = datetime.now(UTC).date()
    due = [r for r in rows if not _backed_off(*state.get(r["id"], (0, None)), today)]
    due.sort(key=lambda r: state.get(r["id"], (0, None))[0])  # stable: salience order within
    stats = {"pending": len(rows), "drafted": 0, "redrafted": 0, "no_draft": 0, "errors": 0}
    for row in due[:cap]:
        try:
            body, base = draft(row, row.get("registry_body"))
        except LLM_TRANSPORT_ERRORS:
            raise
        except DraftProblem as e:
            print(f"  no draft for cand#{row['id']} ({row['name']}): {e}")
            _note_failure(conn, row["id"], str(e), has_state)
            stats["no_draft"] += 1
            continue
        except Exception as e:
            print(f"  draft failed for cand#{row['id']} ({row['name']}): {e}")
            _note_failure(conn, row["id"], f"{type(e).__name__}: {e}", has_state)
            stats["errors"] += 1
            continue
        cur.execute(
            "UPDATE skills_lane.skill_gap_candidates SET proposal_body=%s, base_body_hash=%s, "  # nosec B608 — a fixed column fragment; the values are bound
            "updated_at=now()"
            + (", draft_attempts=0, last_draft_at=NULL, last_draft_error=NULL" if has_state else "")
            + " WHERE id=%s AND status='proposed'",
            (body, base, row["id"]),
        )
        conn.commit()
        stats["redrafted" if row.get("proposal_body") else "drafted"] += 1
    stats["deferred"] = max(0, len(due) - cap)
    stats["backed_off"] = len(rows) - len(due)
    return stats


def _note_failure(conn, cid: int, error: str, has_state: bool) -> None:
    """Record a failed draft on the candidate: one more attempt, when, and why. Before the
    draft-attempts migration there is nowhere to record it."""
    if not has_state:
        return
    conn.execute(
        "UPDATE skills_lane.skill_gap_candidates SET draft_attempts = draft_attempts + 1, "
        "last_draft_at = now(), last_draft_error = %s WHERE id = %s",
        (error[:500], cid),
    )
    conn.commit()
