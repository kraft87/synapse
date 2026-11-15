#!/usr/bin/env python3
"""
Ingest a research brief as a first-class web_artifact (kind='research_brief').

A brief (run-research output, or any deliberately-written markdown doc) is the
highest-quality substrate in the web lane: multi-source, synthesized, zero chrome.
The row is picked up by the web-lane cron (ARCHITECTURE.md, "How it's fed"): chunked,
contextualized and embedded, then served by recall's web bucket. It is NOT
KG-extracted: web-derived facts are kept off the fact legs (they scored as noise).

Who may read it (schema 063, ``web_artifacts.audience``): ``--audience`` sets it
explicitly and wins over every other rule. Without it the row stays unlabelled, which
a restricted surface is never served, until the poller's classifier labels it
(ingestion/web_audience.py). Before 063 is applied the flag is accepted and ignored.

Keyed by tool_use_id = 'brief:<slug>:<YYYY-MM-DD>' (UNIQUE; the date is the file's
mtime). A re-run with the same content is a no-op. A re-run with changed content (a
corrected brief) updates the row and deletes its web_chunks in the same transaction, so
the web-lane cron chunks, contextualizes and embeds the new text on its next run and no
chunk of the old text stays served (extraction_queue rows of those chunks go with them,
ON DELETE CASCADE). A changed brief also loses its audience label, which was decided on
the old text: the poller's classifier labels it again, unless ``--audience`` is given.
Re-running with ``--audience`` on an already-ingested brief updates its label.

Usage:
    SYNAPSE_DB_URL=postgresql://... \
        uv run python scripts/ingest_research_brief.py \
        --file ~/data/research/2026-06-10-web-ingestion.md \
        --slug web-ingestion [--title "How RAG systems ingest web pages"] \
        [--audience work-safe|personal]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.surfaces import AUDIENCES, WEB_AUDIENCE_PROBE


def load_db_url() -> str:
    db_url = os.environ.get("SYNAPSE_DB_URL")
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not db_url and env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("SYNAPSE_DB_URL="):
                db_url = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    if not db_url:
        print("error: SYNAPSE_DB_URL not set", file=sys.stderr)
        sys.exit(2)
    return db_url


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest a research brief into web_artifacts.")
    parser.add_argument("--file", required=True, help="Markdown file containing the brief.")
    parser.add_argument("--slug", required=True, help="Topic slug (kebab-case).")
    parser.add_argument("--title", default=None, help="Title (default: first markdown H1).")
    parser.add_argument(
        "--audience",
        choices=AUDIENCES,
        default=None,
        help="Who may read it: work-safe is served to restricted surfaces, personal never "
        "is. Default: unlabelled until the poller's classifier decides.",
    )
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

    with psycopg.connect(load_db_url()) as conn:
        cur = conn.execute(
            """
            INSERT INTO web_artifacts
                (kind, tool_name, tool_use_id, url, url_canonical, content_hash,
                 title, content_markdown, synthesized, fetched_at, raw_chars)
            VALUES
                ('research_brief', 'run-research', %s, %s, %s, %s,
                 %s, %s, true, %s, %s)
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
        status = "ingested"
        artifact_id = row[0] if row else None
        dropped_chunks = 0
        audience_reset = False
        if row is None:
            artifact_id, dropped_chunks, audience_reset = update_if_changed(
                conn,
                tool_use_id,
                content=content,
                content_hash=content_hash,
                title=title,
                fetched_at=fetched_at,
                keep_audience=args.audience is not None,
            )
            status = "updated" if artifact_id is not None else "unchanged"
        labelled = set_audience(conn, tool_use_id, args.audience) if args.audience else None
        conn.commit()

    if status == "ingested":
        print(f"ingested research_brief id={artifact_id} ({tool_use_id}, {len(content)} chars)")
    elif status == "updated":
        print(
            f"updated research_brief id={artifact_id} ({tool_use_id}, {len(content)} chars; "
            f"content changed, {dropped_chunks} stale web_chunks deleted for re-chunking)"
        )
        if audience_reset:
            print("audience label cleared: the poller's classifier labels the new text")
    else:
        print(f"already ingested ({tool_use_id}); no-op")
    if labelled is False:
        print(
            "warning: schema 063 (web_artifacts.audience) is not applied; "
            f"--audience {args.audience} was not stored",
            file=sys.stderr,
        )
    elif labelled:
        print(f"audience={args.audience}")
    return 0


def update_if_changed(
    conn: psycopg.Connection,
    tool_use_id: str,
    *,
    content: str,
    content_hash: str,
    title: str,
    fetched_at: datetime,
    keep_audience: bool,
) -> tuple[int | None, int, bool]:
    """Replace a stored brief whose content differs from ``content``.

    Returns (id, chunks deleted, audience cleared); id is None when the stored hash
    already matches, and nothing is written. The brief's web_chunks are deleted in the
    caller's transaction: the lane chunks only artifacts that have no chunks
    (scripts/backfill_web_chunks.py), so this is what makes it re-process the row. New
    chunks get new ids, so a contextualize or embed pass still holding the old ids
    updates nothing. The audience label was decided on the old text, so it is cleared
    for the poller's classifier, unless the caller sets one explicitly (``keep_audience``).
    """
    row = conn.execute(
        """
        UPDATE web_artifacts
        SET content_markdown = %s, content_hash = %s, title = %s,
            raw_chars = %s, fetched_at = %s
        WHERE tool_use_id = %s AND content_hash IS DISTINCT FROM %s
        RETURNING id
        """,
        (content, content_hash, title, len(content), fetched_at, tool_use_id, content_hash),
    ).fetchone()
    if row is None:
        return None, 0, False
    artifact_id = int(row[0])
    dropped = conn.execute(
        "DELETE FROM web_chunks WHERE web_artifact_id = %s", (artifact_id,)
    ).rowcount
    audience_reset = False
    if not keep_audience and WEB_AUDIENCE_PROBE.ready(conn, "ingest-brief"):
        audience_reset = (
            conn.execute(
                "UPDATE web_artifacts SET audience = NULL WHERE id = %s AND audience IS NOT NULL",
                (artifact_id,),
            ).rowcount
            > 0
        )
    return artifact_id, dropped, audience_reset


def set_audience(conn: psycopg.Connection, tool_use_id: str, audience: str) -> bool:
    """Store an explicit audience on the brief (rule 1: it overrides any earlier label).
    False when schema 063 is not applied, so nothing could be stored."""
    if not WEB_AUDIENCE_PROBE.ready(conn, "ingest-brief"):
        return False
    conn.execute(
        "UPDATE web_artifacts SET audience = %s WHERE tool_use_id = %s", (audience, tool_use_id)
    )
    return True


if __name__ == "__main__":
    sys.exit(main())
