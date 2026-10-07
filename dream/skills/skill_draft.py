# mypy: ignore-errors
"""dream->skills: give every proposal a complete, ready-to-apply SKILL.md.

Accept applies proposal_body as-is (mcp_server/skill_sync_routes.py), so a proposal without
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
  consolidate               never drafted (a merge is done by hand)

A draft that fails its checks is not stored: the row stays body-less, the review CLI says
"no draft yet", and the next nightly tries again.
"""

from __future__ import annotations

from . import skill_derive as SD
from . import skill_doc as D
from .skill_measure import _extract_json, skill_description

DRAFT_CAP = 12  # drafts per nightly run (each is an Opus call); the rest wait a night
MAX_PHRASINGS = 8
MAX_CHANGE_LINES = 8

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
capabilities. One line, at most {max_chars} characters, no line breaks.

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


def _llm(prompt: str) -> str:
    """The drafting model (Opus by default, SKILL_DERIVE_DRAFT_MODEL to override) — the same
    call skill_derive.draft_skill makes. Resolved at call time so tests can stub it."""
    return SD._draft_call(prompt)


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


def draft_widen(name: str, current: str, evidence) -> str | None:
    """`current` with only its description widened, or None (no gap stated, no description
    field, or the result touched anything else)."""
    phrasings = missing_phrasings(evidence)
    old = skill_description(current)
    if not phrasings or not old:
        return None
    raw = _llm(
        _WIDEN_PROMPT.format(
            name=name,
            description=old,
            phrasings="\n".join(f"- {p}" for p in phrasings),
            max_chars=D.MAX_DESCRIPTION,
        )
    )
    obj = _extract_json(raw or "") or {}
    new = obj.get("description") if isinstance(obj, dict) else None
    if not isinstance(new, str):
        return None
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


def draft_revision(row: dict, current: str) -> str | None:
    """`current` with the extend/fix/narrow change applied, or None when the model's
    revision fails the sanity checks (renamed, emptied, rewritten wholesale)."""
    raw = _llm(
        _REVISE_PROMPT.format(
            direction=row.get("direction") or "fix",
            summary=row.get("summary") or "",
            change=_change_text(row),
            current=current,
        )
    )
    body = D.strip_outer_fence(raw or "") + "\n"
    return None if D.revision_problem(current, body) else body


def draft_derive(row: dict) -> str | None:
    """A full new SKILL.md for a derive row from any detector, or None if unusable."""
    ev = [e for e in row.get("evidence") or [] if isinstance(e, dict)]
    quotes = [str(e["quote"]) for e in ev if e.get("quote")][:6]
    what = row.get("summary") or row["name"]
    if quotes:
        what += "\nEvidence from the sessions:\n" + "\n".join(f"- {q}" for q in quotes)
    body = SD.draft_skill(
        {
            "procedure": row["name"],
            "what": what,
            "trigger_phrasings": row.get("trigger_phrasings") or [],
            "signature": row.get("signature") or "",
            "bash_evidence": [],
            "n_sessions": len({e["session_id"] for e in ev if e.get("session_id")}),
        }
    )
    return None if D.new_skill_problem(body) else body


def draft(row: dict, registry_body: str | None) -> tuple[str, str | None] | None:
    """(proposal_body, base_body_hash) for one candidate row, or None for no draft."""
    if row["kind"] == "derive":
        body = draft_derive(row)
        return (body, None) if body else None
    if row["kind"] != "retune" or not registry_body:
        return None  # consolidate, or a retune of a skill the registry doesn't carry
    if row.get("direction") == "widen":
        body = draft_widen(target_skill(row), registry_body, row.get("evidence"))
    else:
        body = draft_revision(row, registry_body)
    return (body, D.body_hash(registry_body)) if body else None


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


def draft_pending(conn, cap: int = DRAFT_CAP) -> dict:
    """Draft up to `cap` proposed rows that need one. Per-row failures never stop the lane."""
    cur = conn.cursor()
    cur.execute(_PENDING_SQL)
    cols = [d.name for d in cur.description]
    rows = [r for r in (dict(zip(cols, t, strict=True)) for t in cur.fetchall()) if needs_draft(r)]
    stats = {"pending": len(rows), "drafted": 0, "redrafted": 0, "no_draft": 0, "errors": 0}
    for row in rows[:cap]:
        try:
            out = draft(row, row.get("registry_body"))
        except Exception as e:
            print(f"  draft failed for cand#{row['id']} ({row['name']}): {e}")
            stats["errors"] += 1
            continue
        if not out:
            stats["no_draft"] += 1
            continue
        body, base = out
        cur.execute(
            "UPDATE skills_lane.skill_gap_candidates SET proposal_body=%s, base_body_hash=%s, "
            "updated_at=now() WHERE id=%s AND status='proposed'",
            (body, base, row["id"]),
        )
        conn.commit()
        stats["redrafted" if row.get("proposal_body") else "drafted"] += 1
    stats["deferred"] = max(0, len(rows) - cap)
    return stats
