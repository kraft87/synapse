"""Backfill Episodes into Synapse from local Claude Code JSONL transcripts.

Usage:
    python -m ingestion.backfill <root_dir> [--project NAME] [--limit N] [--dry-run]

Walks <root_dir> for *.jsonl files (recursively), parses each into Episodes,
and writes them to Postgres. Chunks are rebuilt per session at the end.
Embedding, summarization, and KG extraction happen later via the regular poller.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

from ingestion.chunks import rebuild_chunks
from ingestion.config import get_settings
from ingestion.contamination import is_harness_call, is_transcript_contamination
from ingestion.db import Database
from ingestion.jsonl_client import JSONLParser
from ingestion.models import Episode, ExtractionItem

logger = logging.getLogger(__name__)


def _write_session_episodes(
    db: Database,
    session_id: str,
    parsed_eps: list[Episode],
    project_default: str | None,
) -> int:
    """Write episodes for one session, preserving existing sequence offsets."""
    existing = db.get_session_episodes(session_id)
    existing_uuids = {e["span_id"] for e in existing if e.get("span_id", "").startswith("jsonl:")}
    next_seq = (max((e["sequence"] for e in existing), default=0)) + 1
    written = 0

    for ep in sorted(parsed_eps, key=lambda e: e.sequence):
        if ep.span_id and ep.span_id in existing_uuids:
            continue
        if ep.span_id and db.span_id_exists(ep.span_id):
            continue
        # Same boundary guards the live /ingest path applies (mcp_server.server):
        # transcribe_ai deposition payloads and Synapse's own harness calls must
        # not enter memory via the disk sweep either.
        if is_transcript_contamination(ep.content) or is_harness_call(ep.content):
            continue

        ep_to_write = ep.model_copy(
            update={
                "sequence": next_seq + written,
                "project": ep.project or project_default,
            }
        )
        episode_id = db.upsert_episode(ep_to_write)
        written += 1

        if episode_id and ep_to_write.content.strip():
            db.enqueue_extraction(
                ExtractionItem(
                    episode_id=episode_id,
                    session_id=ep_to_write.session_id,
                    content=ep_to_write.content,
                    content_type="episode",
                    project=ep_to_write.project,
                )
            )

    return written


def backfill_directory(
    root: Path,
    db: Database,
    project_override: str | None = None,
    file_limit: int | None = None,
    dry_run: bool = False,
    since_days: float | None = None,
) -> dict[str, int]:
    """Walk root for *.jsonl files and ingest. Returns counts.

    ``since_days`` restricts the sweep to files modified within the last N days
    (by mtime). This is what makes the sweep cheap enough to run as a periodic
    backstop and to backfill a known gap window without re-parsing the whole
    tree (and holding every episode in memory). Ingestion stays idempotent via
    span_id, so re-sweeping overlapping windows is a no-op.
    """
    parser = JSONLParser()
    files = sorted(root.rglob("*.jsonl"))
    if since_days is not None:
        import time

        cutoff = time.time() - since_days * 86400
        files = [f for f in files if f.stat().st_mtime >= cutoff]
    if file_limit:
        files = files[:file_limit]

    logger.info("Scanning %d transcript files under %s", len(files), root)

    by_session: dict[str, list[Episode]] = defaultdict(list)
    parsed_files = 0
    parsed_episodes = 0

    for path in files:
        try:
            eps = parser.parse_file(path, project_override=project_override)
        except Exception as e:
            logger.warning("Failed to parse %s: %s", path, e)
            continue
        parsed_files += 1
        if eps:
            parsed_episodes += len(eps)
            by_session[eps[0].session_id].extend(eps)

    logger.info(
        "Parsed %d files into %d episodes across %d sessions",
        parsed_files,
        parsed_episodes,
        len(by_session),
    )

    if dry_run:
        return {
            "files": parsed_files,
            "episodes_parsed": parsed_episodes,
            "sessions": len(by_session),
            "episodes_written": 0,
        }

    written_total = 0
    for i, (session_id, eps) in enumerate(by_session.items(), start=1):
        written_total += _write_session_episodes(
            db,
            session_id,
            eps,
            project_default=project_override,
        )
        rebuild_chunks(db, session_id)
        if i % 100 == 0:
            logger.info("Progress: %d / %d sessions ingested", i, len(by_session))

    logger.info(
        "Done. %d new episodes written across %d sessions",
        written_total,
        len(by_session),
    )
    return {
        "files": parsed_files,
        "episodes_parsed": parsed_episodes,
        "sessions": len(by_session),
        "episodes_written": written_total,
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(description="Backfill JSONL transcripts into Synapse")
    parser.add_argument("root", type=Path, help="Directory containing .jsonl transcripts")
    parser.add_argument("--project", default=None, help="Override project tag for all episodes")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N files")
    parser.add_argument("--dry-run", action="store_true", help="Parse but do not write")
    parser.add_argument(
        "--since-days",
        type=float,
        default=None,
        help="Only sweep files modified within the last N days (mtime). "
        "Use for gap backfills and periodic backstop sweeps.",
    )
    args = parser.parse_args()

    if not args.root.exists():
        print(f"Error: {args.root} does not exist", file=sys.stderr)
        return 2

    cfg = get_settings()
    db = Database(cfg.db_url)
    try:
        stats = backfill_directory(
            root=args.root,
            db=db,
            project_override=args.project,
            file_limit=args.limit,
            dry_run=args.dry_run,
            since_days=args.since_days,
        )
    finally:
        db.close()

    print(f"\nBackfill complete: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
