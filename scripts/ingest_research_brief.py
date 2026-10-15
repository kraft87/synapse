#!/usr/bin/env python3
"""
Ingest a research brief as a first-class web_artifact (task #68, Phase 2).

A brief (run-research output, or any deliberately-written markdown doc) is the
highest-quality extraction substrate in the web lane: multi-source, synthesized,
zero chrome. This script only inserts the kind='research_brief' row. The poller's
maintenance thread chunks, contextualizes and embeds it; the row is marked
kg_extract=true, so once its chunks are embedded ingestion.web_enqueue queues them for KG
extraction.

Idempotent via tool_use_id = 'brief:<slug>:<YYYY-MM-DD>' (UNIQUE).

Usage:
    SYNAPSE_DB_URL=postgresql://... \
        uv run python scripts/ingest_research_brief.py \
        --file ~/data/research/2026-06-10-web-ingestion.md \
        --slug web-ingestion [--title "How RAG systems ingest web pages"]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import UTC, datetime
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.settings import require_setting


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest a research brief into web_artifacts.")
    parser.add_argument("--file", required=True, help="Markdown file containing the brief.")
    parser.add_argument("--slug", required=True, help="Topic slug (kebab-case).")
    parser.add_argument("--title", default=None, help="Title (default: first markdown H1).")
    args = parser.parse_args()

    path = Path(args.file).expanduser()
    if not path.exists():
        print(f"error: {path} not found", file=sys.stderr)
        return 2
    content = path.read_text(encoding="utf-8", errors="replace").strip()
    if not content:
        print("error: file is empty", file=sys.stderr)
        return 2

    title = args.title
    if not title:
        for line in content.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
    title = title or args.slug

    fetched_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    day = fetched_at.strftime("%Y-%m-%d")
    tool_use_id = f"brief:{args.slug}:{day}"
    content_hash = hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()

    try:
        db_url = require_setting("SYNAPSE_DB_URL")
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    with psycopg.connect(db_url) as conn:
        cur = conn.execute(
            """
            INSERT INTO web_artifacts
                (kind, tool_name, tool_use_id, url, url_canonical, content_hash,
                 title, content_markdown, synthesized, fetched_at, raw_chars, metadata)
            VALUES
                ('research_brief', 'run-research', %s, %s, %s, %s,
                 %s, %s, true, %s, %s, '{"capture": "research_brief", "kg_extract": true}')
            ON CONFLICT (tool_use_id) DO NOTHING
            RETURNING id
            """,
            (
                tool_use_id,
                f"research://{args.slug}",
                f"research://{args.slug}",
                content_hash,
                title,
                content,
                fetched_at,
                len(content),
            ),
        )
        row = cur.fetchone()
        conn.commit()

    if row:
        print(f"ingested research_brief id={row[0]} ({tool_use_id}, {len(content)} chars)")
    else:
        print(f"already ingested ({tool_use_id}); no-op")
    return 0


if __name__ == "__main__":
    sys.exit(main())
