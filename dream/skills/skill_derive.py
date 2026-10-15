# mypy: ignore-errors
"""dream->skills, DERIVE: find skill GAPS in session analyses, draft new skills.

skill_measure audits EXISTING skills (under-trigger = RETUNE fuel). This is the other half:
spot recurring, generalizable multi-step PROCEDURES the agent carried out by hand that NO
existing skill covers, cluster the candidates ACROSS sessions, and DRAFT a SKILL.md for a
survivor. Propose-only: the lane stores a draft on the candidate row (skill_draft.py) and the
review accept applies it to the skill registry.

  1. per-session analysis (skill_measure.analyze_session, shared with the retune judge)
     -> candidate gaps with tool/command signature
  2. cross-session CLUSTER (1 call) -> merge dup candidates, count distinct sessions
  3. the ledger's gates (skill_ledger.merge_candidate) decide which clusters get proposed
  4. per-proposal DRAFT (Opus, quality matters) -> SKILL.md

Models: the analysis and cluster calls use SYNAPSE_SKILLS_JUDGE_MODEL, drafting
SYNAPSE_SKILLS_DRAFT_MODEL (llm_client.stage_model; both default to Opus).
"""

from __future__ import annotations

from collections import defaultdict

from dream import lane

from .skill_measure import _extract_json


# --------------------------------------------------------------------------- gaps
def session_gaps(s: dict, analysis: dict | None) -> list[dict]:
    """The gaps in a session's analysis (skill_measure.analyze_session), each tagged with the
    session it came from."""
    gaps = (analysis or {}).get("gaps") or []
    for g in gaps:
        g["_session"] = s["session"]
        g["_bash_heads"] = s["bash_heads"][:12]
    return gaps


# ------------------------------------------------------------------------ cluster

_CLUSTER_PROMPT = """You are consolidating candidate "skill gaps" mined from many separate agent sessions.
Different sessions describe the SAME underlying procedure with different words. Merge them.

CANDIDATES (each has a source session id):
{candidates}

Group candidates that describe the same underlying repeatable procedure. For each group output a
single merged cluster. Drop singletons that look like one-off tasks. Be conservative — a cluster is
only worth a skill if the procedure is genuinely repeatable.

Output ONLY JSON, no prose:
{{"clusters": [{{"procedure": "short-kebab-name", "what": "one-sentence merged description",
"trigger_phrasings": ["merged, deduped phrasings"], "signature": "merged tool/command pattern",
"sessions": ["session-id", "..."]}}]}}"""


def cluster_gaps(gaps: list[dict]) -> list[dict]:
    if not gaps:
        return []
    lines = []
    for g in gaps:
        lines.append(
            f"- [{g.get('_session', '?')[:8]}] {g.get('procedure', '?')}: {g.get('what', '')} "
            f"| triggers={g.get('trigger_phrasings', [])} | sig={g.get('signature', '')}"
        )
    raw = lane.judge(_CLUSTER_PROMPT.format(candidates="\n".join(lines)))
    obj = _extract_json(raw) or {}
    clusters = obj.get("clusters", [])
    # attach distinct-session count + the observed bash heads from member sessions
    by_session = defaultdict(list)
    for g in gaps:
        by_session[g.get("_session")].append(g)
    for c in clusters:
        sids = {s for s in c.get("sessions", []) if s}
        # tolerate truncated ids from the model
        full = {g["_session"] for g in gaps for sid in sids if g["_session"].startswith(sid)}
        c["sessions"] = sorted(full or sids)
        c["n_sessions"] = len(c["sessions"])
        bash = []
        for sid in c["sessions"]:
            for g in by_session.get(sid, []):
                bash += g.get("_bash_heads", [])
        c["bash_evidence"] = bash[:20]
    return clusters


# -------------------------------------------------------------------------- draft

_DRAFT_PROMPT = """Write a Claude Code SKILL.md for the procedure below, derived from real agent sessions.

A SKILL.md has YAML frontmatter then a markdown body:
---
name: <kebab-name>
description: <ONE line. This is the TRIGGER — the model soft-matches user phrasing against it.
  Pack the natural user phrasings that should invoke this skill. Keep it under ~200 chars. No marketing.>
---
# <name>
<1-2 line purpose, then the numbered procedure: concrete steps with the actual tools/commands observed.>

PROCEDURE TO CAPTURE:
- what: {what}
- trigger phrasings observed: {triggers}
- tool/command signature: {signature}
- real bash commands seen across sessions: {bash}
- seen in {n} distinct sessions

Rules: steps must be concrete and reflect the observed signature; do not invent capabilities not in
evidence; the description must read like real user phrasings, not a feature list. Output ONLY the
SKILL.md file content (frontmatter + body), nothing else."""


def draft_skill(c: dict) -> str:
    raw = lane.draft(
        _DRAFT_PROMPT.format(
            what=c.get("what", ""),
            triggers=c.get("trigger_phrasings", []),
            signature=c.get("signature", ""),
            bash=" | ".join(c.get("bash_evidence", [])) or "(none captured)",
            n=c.get("n_sessions", 0),
        )
    ).strip()
    # strip an OUTER ```fence only if the WHOLE output is wrapped (don't eat inner ```bash blocks)
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3]
    return raw.strip() + "\n"
