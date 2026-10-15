"""The ingest-boundary filter must drop transcribe_ai deposition payloads but keep
the operator's dev conversations about the legal-transcription domain."""

from __future__ import annotations

import re

from ingestion.contamination import is_harness_call, is_transcript_contamination


def test_flags_agent_processing_prompts():
    # Each of the four transcribe_ai agent prompt types carries real testimony/PII.
    assert is_transcript_contamination(
        "You are analyzing speech segments from legal discovery proceedings. Remove filler words."
    )
    assert is_transcript_contamination(
        'Spellings list (correct forms):\n["Julie Jimenez"]\nTranscript batch: how did the bicycle travel?'
    )
    assert is_transcript_contamination(
        "Extract follow-up annotations from an Ontario examination for discovery transcript."
    )
    assert is_transcript_contamination(
        "You are a certified court reporter doing verbatim transcription. Do not rephrase."
    )


def test_flags_embedded_payload_in_larger_turn():
    # A recall/dream artifact that embeds the agent payload deep in a larger turn still flags.
    blob = (
        "Searched the automation window. " * 50
        + 'episodes: [{"content": "Spellings list (correct forms): [\\"Darcy R. Merkur\\"]"}]'
    )
    assert is_transcript_contamination(blob)


def test_keeps_dev_discussion_about_the_domain():
    # Prompt-design / code work that names the legal domain WITHOUT carrying a payload is kept.
    assert not is_transcript_contamination(
        "Help me design a prompt for classifying a legal discovery transcription segment as Q or A."
    )
    assert not is_transcript_contamination(
        "Let's refactor cleanup_lawyer to use match-case; the examination_type field should be an array."
    )
    assert not is_transcript_contamination("how does synapse recall rank episodes")


def test_empty_and_none():
    assert not is_transcript_contamination(None)
    assert not is_transcript_contamination("")


# ---------------------------------------------------------------------------
# is_harness_call — Synapse's own LLM calls must not become memories
# ---------------------------------------------------------------------------


def test_flags_recall_benchmark_judge_call():
    # The judge prompt embeds the golden reference answer — the worst leak.
    assert is_harness_call(
        "[user] QUESTION: What approach did we settle on for the segmenter?\n\n"
        "REFERENCE ANSWER (ground truth):\nWe pivoted to embedding cosine similarity.\n\n"
        "RETRIEVED CONTEXT:\nFACT: ...\n\n[assistant] 0.3"
    )


def test_flags_extraction_and_judge_calls():
    assert is_harness_call(
        "[user] Below are candidate facts extracted for a long-term knowledge graph. "
        "Rate EACH:\n1. ...\n\n[assistant] [1, 0]"
    )


def _prompt_constants() -> dict[str, str]:
    """Every module-level prompt template under ingestion/, found rather than listed, so a
    prompt added tomorrow is covered by this test without anyone remembering to add it."""
    import importlib
    import pkgutil

    import ingestion

    found: dict[str, str] = {}
    for mod_info in pkgutil.iter_modules(ingestion.__path__):
        if mod_info.name == "__main__":
            continue
        mod = importlib.import_module(f"ingestion.{mod_info.name}")
        for name, value in vars(mod).items():
            if isinstance(value, str) and re.fullmatch(r"_?[A-Z_]*PROMPT[A-Z_]*", name):
                found[f"{mod_info.name}.{name}"] = value
    return found


def test_every_prompt_in_ingestion_is_a_harness_call():
    """The guard reads the prompts themselves, so editing one cannot disarm it: Stage 3's
    opening was reworded once and the literal prefix kept matching the old text."""
    prompts = _prompt_constants()
    # The module layout is what this test discovers; a rename must not make it vacuous.
    assert {"extraction_llm._EXTRACTION_PROMPT", "timeline_gate.TIMELINE_DEDUP_PROMPT"} <= set(
        prompts
    )
    for name, template in prompts.items():
        assert is_harness_call("[user] " + template), name


def test_the_web_context_prompt_is_a_harness_call_once_rendered():
    from ingestion.web_context import PROMPT

    rendered = PROMPT.format(
        document="# Page\n\nbody", n_chunks=2, chunks_section="[0]\na\n\n[1]\nb"
    )
    assert is_harness_call("[user] " + rendered)
    # a dev conversation that pastes a <document> is not one
    assert not is_harness_call("[user] <document>\nmy notes\n</document>\nwhat do you think?")


def test_prompts_built_from_messages_are_harness_calls_too():
    """The edge-date and node-dedup prompts lead with a system message, which the CLI
    path prepends to the user turn."""
    from ingestion.prompts import dedupe_nodes, extract_edge_dates

    for messages in (
        dedupe_nodes.build_batch_prompt([]),
        extract_edge_dates.build_batch_prompt([], "2026-01-01T00:00:00Z"),
    ):
        flattened = "\n\n".join(m["content"] for m in messages)
        assert is_harness_call("[user] " + flattened)


def test_flags_entity_pipeline_calls():
    # The biggest self-ingestion leak (2026-06-18): entity-typing/audit/fact-dedup scripts
    # that bypass agent_call's setting_sources=[]. Single-turn sessions, often a usage-error
    # as the "answer".
    assert is_harness_call(
        "[user] You are retyping an entity in a personal knowledge graph.\n\nEntity name: "
        "dream/cursor_parser.py\nCurrent summary: (none)\n\n[assistant] You're out of extra usage"
    )
    assert is_harness_call(
        "[user] You are auditing entities in a personal knowledge graph for a typing pass.\n\n"
        "Entity name: Postgres\n\n[assistant] {}"
    )
    assert is_harness_call(
        "[user] You are a fact deduplication assistant. NEVER mark facts with key differences "
        "as duplicates.\n\n[assistant] []"
    )


def test_keeps_dev_conversation_quoting_the_prompts():
    # Real sessions discussing the harness start with prose or [context] blocks,
    # not with the prompt itself as the leading user turn.
    assert not is_harness_call(
        "[context] The benchmark judge failed overnight.\n\n"
        "[user] QUESTION: why did BROAD drop?\n\nREFERENCE ANSWER (ground truth): is in "
        "the served episodes\n\n[assistant] because the eval self-ingested"
    )
    assert not is_harness_call(
        "[user] can you show me the extraction prompt? I think it starts with 'Given the "
        "session summary and pre-identified entities below'\n\n[assistant] yes — here it is"
    )
    # A bare question that happens to start with QUESTION: but has no golden marker.
    assert not is_harness_call("[user] QUESTION: how do I tune efRuntime?\n\n[assistant] ...")
    # Dev talk that QUOTES the entity-typing prompt (prose lead) is kept, not flagged.
    assert not is_harness_call(
        "[user] the retype script sends 'You are retyping an entity in a personal knowledge "
        "graph' — should we route it through agent_call?\n\n[assistant] yes"
    )


def test_harness_empty_and_none():
    assert not is_harness_call(None)
    assert not is_harness_call("")
    assert not is_harness_call("plain text without a user prefix")


def test_every_agent_options_source_stops_self_ingestion():
    """Regression guard for the 24K self-ingestion leak (2026-06-18).

    Every ``ClaudeAgentOptions`` construction spawns a ``claude`` CLI. If that
    CLI loads the user's settings on a host carrying the Stop ingest hook, the
    call's transcript self-ingests as an episode. ``setting_sources=[]`` is the
    source-stop. The retype/audit scripts forgot it and leaked 24K episodes;
    ``is_harness_call`` is only a prefix backstop. Any NEW code path that builds
    ClaudeAgentOptions without the source-stop re-opens the leak — fail here,
    not in prod.
    """
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    offenders = []
    for sub in ("ingestion", "scripts"):
        for f in (repo / sub).rglob("*.py"):
            txt = f.read_text(encoding="utf-8", errors="ignore")
            if "ClaudeAgentOptions(" in txt and "setting_sources" not in txt:
                offenders.append(str(f.relative_to(repo)))
    assert not offenders, (
        "ClaudeAgentOptions without setting_sources=[] re-opens the self-ingestion leak: "
        + ", ".join(offenders)
    )
