"""Stage 3: Mine behavioral patterns from recent dreams → memory proposals.

Instead of mutating agent config files in place, this stage writes structured
proposals to the `memory_proposals` Postgres table. A client-side processor
picks them up, surfaces them for human review, and on approval writes new
memory files into the agent's memory directory.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)

# How many recent dream/summary docs (across all projects) to scan for patterns
_RECENT_DOC_LIMIT = 12
# Skip if fewer than this — not enough signal
_MIN_DOCS_FOR_MINING = 4
# Cap memory body length to keep entries scannable
_MAX_BODY_LEN = 800


_MINING_PROMPT = """\
You are mining behavioral patterns from an AI coding assistant's recent dream summaries.

The assistant works for one user. Auto-memory files persist the user's
preferences across all of their conversations. They have four kinds:

- **feedback**: corrections or guidance the user gave (with "Why:" reason and "How to apply:" rule)
- **user**: facts about the user (role, tools, knowledge)
- **project**: ongoing initiatives, goals, constraints (date-stamped)
- **reference**: pointers to where information lives in external systems

Look at the dream summaries below. Identify recurring patterns that should be
codified as new auto-memory entries. STRONG signals only:

- the user corrected the same behavior multiple times across sessions
- the user stated the same fact / preference multiple times
- An ongoing project / constraint that's referenced repeatedly
- A specific tool, host, or system the user uses that future sessions should know about

SKIP:
- One-off comments
- Speculation
- Things already covered by existing memory files (listed below)
- Generic AI-assistant advice

Existing memory file names — DO NOT duplicate these:
{existing_memory_files}

Recent dream summaries (most recent first):
{dream_summaries}

Output JSON ONLY (no markdown fences). Empty proposals list is OK if nothing
strong enough surfaced. Each proposal must include the full memory body in
the format the auto-memory system expects (frontmatter + body, with Why: and
How to apply: lines for feedback/project kinds).

Schema:
{schema_json}
"""


_PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["feedback", "user", "project", "reference"],
                    },
                    "filename": {
                        "type": "string",
                        "description": "snake_case .md filename, e.g., feedback_no_emdashes.md",
                    },
                    "name": {
                        "type": "string",
                        "description": "frontmatter name field, short title",
                    },
                    "description": {
                        "type": "string",
                        "description": "frontmatter one-line description",
                    },
                    "body_md": {
                        "type": "string",
                        "description": (
                            "memory body. For feedback/project: include "
                            "**Why:** and **How to apply:** lines."
                        ),
                    },
                    "rationale": {
                        "type": "string",
                        "description": "why dream proposed this — what was observed",
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },
                "required": [
                    "kind",
                    "filename",
                    "name",
                    "description",
                    "body_md",
                    "rationale",
                    "confidence",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["proposals"],
    "additionalProperties": False,
}


def _existing_memory_filenames() -> list[str]:
    """Best-effort list of already-existing memory filenames.

    The dream container has no direct access to the client's memory dir, so
    the deploy passes `MEMORY_FILES_LIST` as a colon-separated env var and the
    LLM avoids re-proposing existing files. If unset, we fall back to checking
    a local path (useful for dry runs).
    """
    env_list = os.environ.get("MEMORY_FILES_LIST", "")
    if env_list:
        return [name for name in env_list.split(":") if name]

    local = os.environ.get("MEMORY_FILES_PATH")
    if local:
        p = Path(local)
        if p.is_dir():
            return sorted(f.name for f in p.glob("*.md") if f.name != "MEMORY.md")
    return []


def _fetch_recent_docs(conn: Any, limit: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT id, project, content, generated_at AS created_at
        FROM synth_documents
        WHERE doc_type IN ('dream', 'summary')
        ORDER BY generated_at DESC
        LIMIT %s
        """,
        (limit,),
    ).fetchall()
    return list(rows)


def _build_prompt(docs: list[dict[str, Any]], existing: list[str]) -> str:
    summaries_text = "\n\n---\n\n".join(
        f"[{d.get('project') or 'untagged'} {d['created_at'].strftime('%Y-%m-%d')}] {d['content']}"
        for d in docs
    )
    existing_text = (
        "\n".join(f"- {f}" for f in existing)
        if existing
        else "(none provided — be especially careful not to re-propose obvious things)"
    )
    return _MINING_PROMPT.format(
        existing_memory_files=existing_text,
        dream_summaries=summaries_text,
        schema_json=json.dumps(_PROPOSAL_SCHEMA, indent=2),
    )


def _parse_response(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse mining output as JSON: %s", e)
        return []
    proposals = payload.get("proposals", [])
    return list(proposals) if isinstance(proposals, list) else []


def _validate_proposal(p: dict[str, Any]) -> dict[str, Any] | None:
    """Defensive validation. Returns a normalized proposal or None to skip."""
    try:
        kind = p["kind"]
        filename = p["filename"]
        name = p["name"]
        description = p["description"]
        body_md = str(p["body_md"])[:_MAX_BODY_LEN]
        rationale = p["rationale"]
        confidence = float(p.get("confidence", 0.5))
    except (KeyError, TypeError, ValueError):
        return None

    if kind not in ("feedback", "user", "project", "reference"):
        return None
    # filename must be a simple basename ending in .md
    if "/" in filename or ".." in filename or not filename.endswith(".md"):
        return None
    if not 0.0 <= confidence <= 1.0:
        confidence = max(0.0, min(1.0, confidence))

    return {
        "kind": kind,
        "filename": filename,
        "name": name,
        "description": description,
        "body_md": body_md,
        "rationale": rationale,
        "confidence": confidence,
    }


def mine_proposals(db_url: str, *, dry_run: bool = False) -> int:
    """Mine recent dreams for memory proposals; insert pending rows into Postgres.

    Returns the number of proposals inserted (or that would be in dry-run).
    """
    from ingestion.llm_client import create_llm_client

    conn = psycopg.connect(db_url, row_factory=dict_row, autocommit=True)
    try:
        docs = _fetch_recent_docs(conn, _RECENT_DOC_LIMIT)
        if len(docs) < _MIN_DOCS_FOR_MINING:
            logger.info(
                "Not enough recent dream/summary docs (%d < %d) — skipping mining",
                len(docs),
                _MIN_DOCS_FOR_MINING,
            )
            return 0

        existing = _existing_memory_filenames()
        prompt = _build_prompt(docs, existing)

        from ingestion.llm_client import stage_model

        dream_model = stage_model("DREAM")
        llm = create_llm_client(model=dream_model)
        response = llm.messages.create(
            model=dream_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
        raw_proposals = _parse_response(text)
        if not raw_proposals:
            logger.info("No new memory proposals from this dream cycle")
            return 0

        # Use the most recent few dream IDs as breadcrumbs back to evidence
        evidence_ids = [int(d["id"]) for d in docs[:5]]

        inserted = 0
        for raw in raw_proposals:
            p = _validate_proposal(raw) if isinstance(raw, dict) else None
            if p is None:
                logger.warning("Skipping malformed proposal: %r", raw)
                continue

            if dry_run:
                logger.info(
                    "DRY RUN proposal kind=%s filename=%s confidence=%.2f",
                    p["kind"],
                    p["filename"],
                    p["confidence"],
                )
                inserted += 1
                continue

            try:
                conn.execute(
                    """
                    INSERT INTO memory_proposals
                        (kind, filename, name, description, body_md,
                         evidence_dream_ids, confidence, rationale)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        p["kind"],
                        p["filename"],
                        p["name"],
                        p["description"],
                        p["body_md"],
                        evidence_ids,
                        p["confidence"],
                        p["rationale"],
                    ),
                )
                inserted += 1
            except Exception as e:
                logger.warning("Failed to insert proposal %s: %s", p["filename"], e)

        logger.info("Mined %d memory proposal(s) from %d dream docs", inserted, len(docs))
        return inserted
    finally:
        conn.close()
