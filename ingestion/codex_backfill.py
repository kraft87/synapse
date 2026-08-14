"""Backfill OpenAI Codex CLI rollout files into Synapse.

Usage:
    python -m ingestion.codex_backfill [root] [--project NAME] [--limit N]
                                       [--dry-run] [--since-days N]

Walks root (default ~/.codex/sessions) for rollout-*.jsonl files, parses each
into Episodes via CodexRolloutParser, and appends them through the shared
write_backfill_session path (span_id dedup, private-session gate, extraction
enqueue). Contamination/harness guards are applied here, matching what the
Claude Code JSONL sweep does — a Codex agent's transcript can carry the same
payloads.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

from ingestion.backfill import write_backfill_session
from ingestion.chunks import rebuild_chunks
from ingestion.codex_client import CodexRolloutParser
from ingestion.config import get_settings
from ingestion.contamination import is_harness_call, is_transcript_contamination
from ingestion.db import Database
from ingestion.models import Episode

logger = logging.getLogger(__name__)

DEFAULT_ROOT = Path.home() / ".codex" / "sessions"


def backfill_codex_sessions(
    root: Path,
    db: Database | None,
    project_override: str | None = None,
    file_limit: int | None = None,
    dry_run: bool = False,
    since_days: float | None = None,
) -> dict[str, int]:
    parser = CodexRolloutParser()
    files = sorted(root.rglob("rollout-*.jsonl"))
    if since_days is not None:
        cutoff = time.time() - since_days * 86400
        files = [f for f in files if f.stat().st_mtime >= cutoff]
    if file_limit:
        files = files[:file_limit]

    logger.info("Scanning %d rollout files under %s", len(files), root)

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
        eps = [
            ep
            for ep in eps
            if not (is_transcript_contamination(ep.content) or is_harness_call(ep.content))
        ]
        if eps:
            parsed_episodes += len(eps)
            by_session[eps[0].session_id].extend(eps)

    logger.info(
        "Parsed %d files into %d episodes across %d sessions",
        parsed_files,
        parsed_episodes,
        len(by_session),
    )

    if dry_run or db is None:
        return {
            "files": parsed_files,
            "episodes_parsed": parsed_episodes,
            "sessions": len(by_session),
            "episodes_written": 0,
        }

    written_total = 0
    for i, (session_id, eps) in enumerate(by_session.items(), start=1):
        written_total += write_backfill_session(db, session_id, eps)
        rebuild_chunks(db, session_id)
        if i % 100 == 0:
            logger.info("Progress: %d / %d sessions ingested", i, len(by_session))

    logger.info("Done. %d new episodes written across %d sessions", written_total, len(by_session))
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
    parser = argparse.ArgumentParser(description="Backfill Codex CLI rollouts into Synapse")
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=DEFAULT_ROOT,
        help=f"Directory containing rollout-*.jsonl files (default: {DEFAULT_ROOT})",
    )
    parser.add_argument("--project", default=None, help="Override project tag for all episodes")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N files")
    parser.add_argument("--dry-run", action="store_true", help="Parse but do not write")
    parser.add_argument(
        "--since-days",
        type=float,
        default=None,
        help="Only sweep files modified within the last N days (mtime)",
    )
    args = parser.parse_args()

    if not args.root.exists():
        print(f"Error: {args.root} does not exist", file=sys.stderr)
        return 2

    if args.dry_run:
        stats = backfill_codex_sessions(
            args.root, None, args.project, args.limit, dry_run=True, since_days=args.since_days
        )
        print(f"\nBackfill complete: {stats}")
        return 0

    cfg = get_settings()
    db = Database(cfg.db_url)
    try:
        stats = backfill_codex_sessions(
            root=args.root,
            db=db,
            project_override=args.project,
            file_limit=args.limit,
            dry_run=False,
            since_days=args.since_days,
        )
    finally:
        db.close()

    print(f"\nBackfill complete: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
