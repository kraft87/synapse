"""Frozen snapshot of the pre-Phase-4 stub prompts.

These were the hand-rolled prompts shipped in Phases 2 and 3 (PRs #34 and
#35). They are preserved verbatim so the Phase 4 smoke test
(`scripts/compare_prompts_smoke.py`) can compare LLM behavior between the
stubs and the verbatim Graphiti-ported replacements without re-checking out
the old code.

Do not edit these strings — the snapshot is the test contract.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Phase 2 — NodeDeduper._llm_confirm (ingestion/dedup.py)
# ---------------------------------------------------------------------------

LEGACY_NODE_CONFIRM_PROMPT = (
    "Are '{a_name}' (summary: {a_sum}) and '{b_name}' (summary: {b_sum}) "
    "the same underlying entity? Answer only 'yes', 'no', or 'uncertain'."
)


def render_legacy_node_confirm(
    new_name: str,
    new_summary: str,
    existing_name: str,
    existing_summary: str,
) -> str:
    """Reproduce dedup.py::_llm_confirm's pre-Phase-4 prompt rendering."""
    return LEGACY_NODE_CONFIRM_PROMPT.format(
        a_name=existing_name,
        a_sum=(existing_summary or "(no summary)")[:600],
        b_name=new_name,
        b_sum=(new_summary or "(no summary)")[:600],
    )


# ---------------------------------------------------------------------------
# Phase 3 — ContradictionDetector LLM prompt (ingestion/contradiction.py)
# ---------------------------------------------------------------------------

LEGACY_CONTRADICTION_PROMPT = """\
You are a fact contradiction detector. Decide which of the EXISTING FACTS the NEW FACT contradicts.

A contradiction means: the new fact's claim is incompatible with the existing fact's claim. This INCLUDES drop-in replacements (e.g. "X uses A" -> "X uses B" is a contradiction even though the target differs) and value updates (e.g. "port is 8787" -> "port is 8788").

NOT a contradiction:
- Different events on different days ("ran 5 miles Tuesday" vs "ran 3 miles Wednesday")
- Additive information ("uses Python" vs "uses Python 3.12")
- Unrelated claims about the same entities

NEW FACT: "{new_fact}"

EXISTING FACTS (each line is "[idx] fact"):
{existing_section}

Return JSON only, no prose: {{"contradicted_facts": [list of idx integers]}}
Return an empty list if the new fact contradicts none of them.
"""


def render_legacy_contradiction(new_fact: str, existing_section: str) -> str:
    """Reproduce contradiction.py::detect_contradictions's pre-Phase-4 prompt."""
    return LEGACY_CONTRADICTION_PROMPT.format(
        new_fact=new_fact,
        existing_section=existing_section,
    )


__all__ = [
    "LEGACY_CONTRADICTION_PROMPT",
    "LEGACY_NODE_CONFIRM_PROMPT",
    "render_legacy_contradiction",
    "render_legacy_node_confirm",
]
