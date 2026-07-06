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
# ClaudeCLIClient / agent_call spawn the real claude CLI. On a host whose user
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
_HARNESS_PREFIXES = (
    # ingestion/extractor.py::_EXTRACTION_PROMPT
    "Given the session summary and pre-identified entities below",
    # ingestion/extractor.py::_WEB_EXTRACTION_PROMPT
    "The text below is an excerpt from ",
    # ingestion/extractor.py::_CONTRADICTION_PROMPT + _BATCH_CONTRADICTION_PROMPT
    "You are a knowledge graph deduplication assistant.",
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
    # KG fact-dedup prompt — distinct wording from the "knowledge graph deduplication
    # assistant" above (60 stranded)
    "You are a fact deduplication assistant",
)
# scripts/recall_benchmark.py::_JUDGE — starts "QUESTION: " (too generic alone),
# so it must be paired with the template's second stanza nearby.
_JUDGE_MARKER = "REFERENCE ANSWER (ground truth):"


def is_harness_call(content: str | None) -> bool:
    """True if `content` is one of Synapse's own LLM-call payloads (drop at ingest)."""
    if not content or not content.startswith("[user] "):
        return False
    text = content[len("[user] ") :]
    if text.startswith("QUESTION: "):
        return _JUDGE_MARKER in text[:2000]
    return text.startswith(_HARNESS_PREFIXES)
