"""Reject transcribe_ai deposition-transcript content at the ingest boundary.

The transcribe_ai Agent-SDK app processes real legal depositions; its production runs
(and pasted transcript payloads) carry third-party PII — deponent names, testimony,
case numbers. Those must never enter Synapse's memory. This is the single chokepoint:
`ingest_turns` skips any turn that looks like deposition payload, and the chunk backfill
uses the same predicate to exclude contaminated windows.

Keyed on the AGENT'S OWN PROMPT SIGNATURES (high precision), NOT loose domain words like
"witness" or "transcript" — the operator's transcribe_ai *dev* conversations legitimately discuss
the legal domain and must keep flowing in. These multi-word signatures appear only in the
agent's actual transcript-processing calls:

    "You are analyzing speech segments from legal discovery proceedings..."
    "Extract follow-up annotations from an Ontario examination for discovery transcript..."
    "Spellings list (correct forms): [...] Transcript batch: ..."
    "You are a certified court reporter doing verbatim transcription."
"""

from __future__ import annotations

import functools

# Distinctive transcribe_ai agent-prompt phrases. Lowercased substring match.
# Kept deliberately LONG/complete: short fragments ("legal discovery proceedings",
# "speech segments") also appear in dev conversations that quote the prompt while
# building it, and those are legit work to keep. Each phrase below is a full agent
# instruction that only appears in an actual transcript-processing call.
_SIGNATURES = (
    "analyzing speech segments from legal discovery",
    "examination for discovery transcript",
    "spellings list (correct forms)",
    "certified court reporter doing verbatim",
)


def is_transcript_contamination(content: str | None) -> bool:
    """True if `content` is a transcribe_ai deposition-processing payload (drop at ingest)."""
    if not content:
        return False
    low = content.lower()
    return any(sig in low for sig in _SIGNATURES)


# ---------------------------------------------------------------------------
# Synapse's own LLM calls (extraction / judges) must not become memories.
# ---------------------------------------------------------------------------
# ClaudeAgentClient / agent_call spawn the real claude CLI. On a host whose user
# settings carry the Stop ingest hook (cortex), each spawned call used to ship
# its own transcript into /ingest — so every judged eval run self-ingested:
# judge prompts (which embed golden-set ground-truth answers) became episodes,
# outranked real history on the episode leg, and BROAD benchmark scores
# collapsed 0.62 -> 0.41. 1,895 such episodes were purged on 2026-06-12.
#
# agent_call now passes setting_sources=[] so the hook never fires; this
# predicate is the boundary guard for every other path (disk sweep, an old
# image, a future lane). PREFIX-anchored on a content that BEGINS with the
# user turn: harness calls are single-turn sessions whose content starts
# "[user] <prompt>". Dev conversations that merely QUOTE these prompts start
# with prose or [context] blocks and keep flowing in.
#
# The prompts under ingestion/ are not listed here: ``_prompt_heads`` reads them, so editing
# one cannot leave this list guarding text that no longer exists (the Stage 3 prefix did
# exactly that). What is listed is code that is not importable from here: scripts and the
# dream lanes, which import this package, not the other way round.
_HARNESS_PREFIXES = (
    # scripts/ab_*.py fact-quality judge
    "Below are candidate facts extracted for a long-term knowledge graph",
    # scripts/retype_untyped_entities.py / retype_pilot.py entity-typing prompt.
    # The BIGGEST self-ingestion leak (24,462 stranded episodes, 2026-06-18): these
    # scripts build their own ClaudeSDKClient and bypass agent_call's setting_sources=[],
    # so every per-entity typing call shipped its transcript into /ingest. Single-turn
    # sessions whose "answer" is often a usage-limit error string.
    "You are retyping an entity in a personal knowledge graph",
    # scripts/survey_cross_graph_leaks.py / typing-pass audit prompt (205 stranded)
    "You are auditing entities in a personal knowledge graph",
    # KG fact-dedup prompt of the pre-batch pipeline (60 stranded)
    "You are a fact deduplication assistant",
    # dream/ lane prompts (skill_measure, skill_derive, skill_draft, struggle_arc, post_fire,
    # procedure_miner, config/nightly). They go through llm_client, which never loads the Stop
    # hook; these cover a lane run by hand on a workstation that has one.
    "You audit AI-agent skill routing and skill coverage.",
    'You are consolidating candidate "skill gaps"',
    "Write a Claude Code SKILL.md for the procedure below",
    "You maintain the trigger DESCRIPTION of one Claude Code skill.",
    "You revise one Claude Code skill: a SKILL.md",
    "You review one WINDOW of an AI coding-agent transcript",
    "You audit ONE firing of an agent SKILL",
    "You audit an AI coding agent's RECURRING SHELL PROCEDURES",
    "You audit how well an AI agent follows its user's BEHAVIORAL preferences.",
)
# scripts/recall_benchmark.py::_JUDGE — starts "QUESTION: " (too generic alone),
# so it must be paired with the template's second stanza nearby.
_JUDGE_MARKER = "REFERENCE ANSWER (ground truth):"


#: How much of a prompt's opening identifies it.
_HEAD_CHARS = 60


@functools.cache
def _prompt_heads() -> tuple[str, ...]:
    """The opening of each of Synapse's own prompts, read from the prompts.

    The CLI path prepends a call's system message to its prompt, so a prompt built from
    messages is identified by its system message. The head stops at the first format field,
    so a template and its rendering share it. Imported here, not at the top: these modules
    import this one's package at start-up."""
    from ingestion.extraction_llm import _EXTRACTION_PROMPT, _WEB_EXTRACTION_PROMPT
    from ingestion.extraction_policy import _BATCH_CONTRADICTION_PROMPT
    from ingestion.notes import PAIR_PROMPT
    from ingestion.preferences_gate import CONFIRM_PROMPT, GATE_PROMPT
    from ingestion.prompts import dedupe_nodes, extract_edge_dates
    from ingestion.timeline_gate import TIMELINE_DEDUP_PROMPT, gate_prompt
    from ingestion.web_context import PROMPT as CONTEXTUALIZE_PROMPT

    texts = (
        _EXTRACTION_PROMPT,
        _WEB_EXTRACTION_PROMPT,
        _BATCH_CONTRADICTION_PROMPT,
        PAIR_PROMPT,
        GATE_PROMPT,
        CONFIRM_PROMPT,
        gate_prompt(),
        TIMELINE_DEDUP_PROMPT,
        CONTEXTUALIZE_PROMPT,
        dedupe_nodes.build_batch_prompt([])[0]["content"],
        extract_edge_dates.build_batch_prompt([], "")[0]["content"],
    )
    return tuple(t.split("{", 1)[0][:_HEAD_CHARS] for t in texts)


def is_harness_call(content: str | None) -> bool:
    """True if `content` is one of Synapse's own LLM-call payloads (drop at ingest)."""
    if not content or not content.startswith("[user] "):
        return False
    text = content[len("[user] ") :]
    if text.startswith("QUESTION: "):
        return _JUDGE_MARKER in text[:2000]
    return text.startswith(_HARNESS_PREFIXES + _prompt_heads())
