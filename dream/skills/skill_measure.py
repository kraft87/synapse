# mypy: ignore-errors
"""dream->skills: what the lane's judges share — the skill catalog, the session analysis
prompt and its parse, and the under-trigger gate.

The lane measures, per skill, how often it FIRED vs how often it was APPLICABLE-but-didn't-fire
(under-trigger = RETUNE fuel) and spots gaps no skill covers (DERIVE fuel). Session views come
from the Synapse episodes table (dream/lane.py); skills come from the skill_registry table.

The judge model is SYNAPSE_SKILLS_JUDGE_MODEL (llm_client.stage_model; default Opus).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

from dream import lane
from ingestion.llm_client import MalformedResponseError, extract_json_object


def load_catalog(conn) -> list[tuple[str, str]]:
    """(name, description) for ACTIVE skills, ordered by name — the lane-wide registry read
    every detector shares. Server-side the registry IS the catalog (the dream container has no
    ~/.claude/skills); the client's skills_sync owns the disk<->registry publish. `description`
    is the trigger surface."""
    cur = conn.cursor()
    cur.execute(
        "SELECT name, COALESCE(description, '') FROM skills_lane.skill_registry "
        "WHERE status = 'active' ORDER BY name"
    )
    return [(name, desc) for name, desc in cur.fetchall()]


def load_scopes(conn) -> dict[str, str]:
    """name -> scope ('global' or 'project:<p>') for ACTIVE skills."""
    cur = conn.cursor()
    cur.execute("SELECT name, scope FROM skills_lane.skill_registry WHERE status = 'active'")
    return dict(cur.fetchall())


def served_catalog(
    catalog: list[tuple[str, str]], scopes: dict[str, str], project: str | None
) -> list[tuple[str, str]]:
    """The (name, description) pairs of the skills a session in `project` is served: the
    global ones and its own project's `project:<p>` ones, the project compared
    case-insensitively like the sync's SkillAccess (a label keeps its folder's case)."""
    mine = f"project:{project.lower()}" if project else None

    def served(name: str) -> bool:
        scope = scopes.get(name, "global")
        return scope == "global" or scope.lower() == mine

    return [(n, d) for n, d in catalog if served(n)]


def format_catalog(items) -> str:
    """Render a skill catalog for a judge prompt: one `- {name}: {desc}` line per entry,
    "(none)" when the catalog is empty."""
    return "\n".join(f"- {n}: {d}" for n, d in items) or "(none)"


def is_substantive(s: dict) -> bool:
    """Filter out cron one-shots / trivial sessions. Needs real back-and-forth
    AND real tool work — that's where a skill could plausibly have helped."""
    if s["user_turns"] < 2:
        return False
    if sum(s["tools"].values()) < 3:
        return False
    return True


_ANALYZE_PROMPT = """You audit AI-agent skill routing and skill coverage. Below are the full skill catalog and one coding-session summary.
Skills auto-fire when the user's phrasing matches a skill's description. Judge two things STRICTLY, each on its own:

1. would_have_helped: skills that did NOT fire this session but CLEARLY should have (the work squarely matches the skill's purpose). Be conservative — empty list if none obviously apply. Do NOT list a skill that did fire.
   For EACH would_have_helped skill, compare the user's wording against that skill's CURRENT catalog description (the text after its name in the catalog — its trigger surface):
   - description_covers: true when the description ALREADY names what the user asked for (the same words, tool, command, product, file or concept). The skill was a match and simply was not invoked; adding trigger words would change nothing.
   - description_covers: false ONLY when there is a real vocabulary gap: the user used wording, a file, a tool, a setting name or a concept that the description does not mention.
   - missing_phrasing: when description_covers is false, the specific user wording or concept the description lacks (a few words, quoted from the user where possible). Use "" when description_covers is true.

2. gaps: a "gap" = a RECURRING, GENERALIZABLE, multi-step PROCEDURE the agent carried out (visible in the tools / bash commands / user requests) that NO catalog skill covers and that a skill SHOULD exist for.
   STRICT — false positives are expensive (a bad skill pollutes every future session):
   - It must be a repeatable PROCEDURE with an OBSERVABLE tool/command signature, not a one-off and not pure deliberation/chat.
   - It must NOT be covered by any catalog skill. If a skill NEARLY covers it, that's a retune of that skill (item 1), NOT a gap -> skip it.
   - It must generalize beyond this exact task (would plausibly recur on a different day or topic).
   - Most sessions have NO gap. Return an empty list rather than reaching.

SKILL CATALOG (name: current description):
{catalog}

SESSION:
- skills that FIRED: {fired}
- tools used (name: count): {tools}
- sample bash commands: {bash}
- USER MESSAGES across the session (the trigger surface — judge under-trigger against THESE phrasings, not just the first):
{user_msgs}

Output ONLY a JSON object, no prose:
{{"would_have_helped": [{{"skill": "name", "why": "one sentence", "description_covers": true or false, "missing_phrasing": "the user wording the description lacks; empty string when covered"}}],
"gaps": [{{"procedure": "short-kebab-name", "what": "one sentence describing the procedure",
"trigger_phrasings": ["how a user would ask for it", "another phrasing"],
"signature": "the tool/command pattern observed (e.g. rg -> Read -> Edit -> Bash pytest)",
"why_generalizable": "one sentence"}}]}}"""


# --- under-trigger gate: does a would_have_helped item show a real description gap? ---
# Whole words dropped before the deterministic coverage match, so filler ("run THIS sql")
# can't break an otherwise literal match. Only standalone words: the parts of a compound
# identifier (pipeline_a.yml -> pipeline, a, yml) are always kept.
_COVER_STOP = frozenset(
    "a an the this that these those it its to of in on at by for from with into and or "
    "is are be me my i we our you your please can could would should some any".split()
)


def _cover_tokens(text: str) -> list[str]:
    """Lowercase alphanumeric tokens in order. Punctuation, quotes and identifier
    separators (_ . - /) all split, so `SYNAPSE_RECALL_FLOOR` -> synapse recall floor and
    `pipeline_c.yml` -> pipeline c yml. Standalone stopwords are dropped."""
    out: list[str] = []
    for word in (text or "").lower().split():
        parts = re.findall(r"[a-z0-9]+", word)
        if len(parts) == 1 and parts[0] in _COVER_STOP:
            continue
        out.extend(parts)
    return out


def description_covers_phrasing(phrasing: str, description: str) -> bool:
    """Deterministic backstop for the under-trigger judge: True when `phrasing` is already
    literally present in the skill `description` — its normalized token sequence appears
    contiguously in the description's (the `Triggers on:` list is part of the description
    text, so a trigger phrase counts). Conservative by design: a phrasing with any word the
    description lacks (`pipeline_c.yml` against a description naming only pipeline_a.yml /
    pipeline_b.yml) is NOT covered, and an empty or all-stopword phrasing is never covered."""
    p = _cover_tokens(phrasing)
    if not p:
        return False
    return f" {' '.join(p)} " in f" {' '.join(_cover_tokens(description))} "


def _judge_bool(v) -> bool | None:
    """A judge-emitted boolean: real bools, or the strings "true"/"false" (LLMs drift).
    None when absent or unparseable."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower() == "true"
    return None


def under_trigger_verdict(item: dict, description: str, phrasing: str = "") -> str:
    """Classify one would_have_helped item against the skill's CURRENT description:
      "gap"      — judge says description_covers=false AND names a non-empty
                   missing_phrasing that the description really lacks -> widen fuel;
      "covered"  — judge says description_covers=true: the skill matched but wasn't invoked;
      "backstop" — judge claims a gap, but its missing_phrasing (or the session `phrasing`)
                   is already literally in the description -> treated as covered;
      "no_gap"   — no stated gap: legacy output without the fields, description_covers
                   missing/unparseable, or false with an empty missing_phrasing.
    Safe default: only "gap" may widen a skill; everything else never does."""
    covers = _judge_bool(item.get("description_covers"))
    if covers is True:
        return "covered"
    missing = str(item.get("missing_phrasing") or "").strip()
    if covers is None or not missing:
        return "no_gap"
    if description_covers_phrasing(missing, description) or description_covers_phrasing(
        phrasing, description
    ):
        return "backstop"
    return "gap"


def _extract_json(raw: str) -> dict | None:
    """Lenient reading of a judge response: its JSON object, None when it has none (or not a
    parseable one). Shared by every dream->skills detector."""
    try:
        return extract_json_object(raw)
    except MalformedResponseError:
        return None


def _excerpt(episodes: list[dict], excerpt_chars: int = 700) -> str:
    """Render a transcript window for a judge prompt: one `[seq N]` block per episode
    (content, else human_turn), each capped at `excerpt_chars`. Shared leaf helper."""
    lines = []
    for ep in episodes:
        text = (ep.get("content") or "").strip() or (ep.get("human_turn") or "").strip()
        if len(text) > excerpt_chars:
            text = text[:excerpt_chars] + " …"
        lines.append(f"[seq {ep['sequence']}]\n{text}")
    return "\n\n".join(lines)


def _clamp_salience(v) -> int | None:
    """Coerce a judge-provided salience to 1..5; None when unparseable. Shared leaf helper."""
    try:
        return max(1, min(5, int(v)))
    except (TypeError, ValueError):
        return None


def scan_night() -> str:
    """The UTC scan-night stamp (YYYY-MM-DD) every detector tags its evidence with."""
    return datetime.now(UTC).date().isoformat()


def _sample(msgs: list[str], cap: int = 40) -> str:
    """All user phrasings if few; an evenly-spaced sample if many (bounds prompt size)."""
    if len(msgs) > cap:
        step = len(msgs) / cap
        msgs = [msgs[int(i * step)] for i in range(cap)]
    return "\n".join(f"  - {m}" for m in msgs)


def analyze_session(s: dict, catalog: str) -> dict | None:
    """The lane's one LLM read of a session: {would_have_helped, gaps} against the skill
    `catalog` (format_catalog). None when the reply carries no JSON object. Raises when the
    model cannot be reached (lane.judge)."""
    prompt = _ANALYZE_PROMPT.format(
        catalog=catalog,
        fired=s["fired"] or "(none)",
        tools=dict(sorted(s["tools"].items(), key=lambda x: -x[1])),
        bash=" | ".join(s["bash_heads"][:12]),
        user_msgs=_sample(s.get("user_msgs", [])),
    )
    return _extract_json(lane.judge(prompt))
