"""Backfill Episodes into Synapse from local transcripts and chat exports.

Usage:
    python -m ingestion.backfill [--source SOURCE] [path] [--project NAME] [--limit N]
                                 [--since-days N] [--dry-run]

Sources (``--source``, default ``claude-code``):
    claude-code  a directory of Claude Code *.jsonl transcripts (~/.claude/projects)
    codex        a directory of Codex rollout-*.jsonl files (default ~/.codex/sessions)
    cursor       a directory of Cursor *.vscdb files
    claude-ai    a claude.ai conversations.json export
    chatgpt      a ChatGPT export directory, or one of its conversations JSON files

Walks the path for transcript files (or reads the one export), parses them into Episodes
and groups them by session. Each session is stored through
``ingestion.episode_writer.store_episodes`` — the same guards, in the same order, as
``POST /ingest`` — and its chunks are rebuilt. Embedding, summarization, and KG extraction
happen later via the regular poller.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ingestion.chatgpt_export import parse_export as parse_chatgpt_export
from ingestion.chunks import rebuild_chunks
from ingestion.claude_ai_client import parse_export as parse_claude_ai_export
from ingestion.codex_client import CodexRolloutParser
from ingestion.cursor_sqlite_client import CursorSQLiteParser
from ingestion.db import Database
from ingestion.episode_writer import StoreStats, store_episodes
from ingestion.jsonl_client import JSONLParser
from ingestion.models import Episode
from ingestion.settings import get_settings

logger = logging.getLogger(__name__)


def _parse_chatgpt(path: Path, project: str | None) -> list[Episode]:
    eps, skipped = parse_chatgpt_export(path)
    if skipped:
        logger.info("skipped %d do-not-remember conversation(s)", skipped)
    return eps


@dataclass(frozen=True)
class Source:
    #: Files to walk the path for, or None when the path is one export parsed as a unit.
    glob: str | None
    #: Parses one file (or the export) into Episodes; the second argument is ``--project``,
    #: which only the sources that label from a working directory apply themselves.
    parse: Callable[[Path, str | None], list[Episode]]
    default_root: Path | None = None


SOURCES: dict[str, Source] = {
    "claude-code": Source("*.jsonl", lambda p, project: JSONLParser().parse_file(p, project)),
    "codex": Source(
        "rollout-*.jsonl",
        lambda p, project: CodexRolloutParser().parse_file(p, project),
        default_root=Path.home() / ".codex" / "sessions",
    ),
    "cursor": Source("*.vscdb", lambda p, project: CursorSQLiteParser.parse_file(p)),
    "claude-ai": Source(None, lambda p, project: parse_claude_ai_export(p)),
    "chatgpt": Source(None, _parse_chatgpt),
}


def backfill(
    source: str,
    path: Path,
    db: Database,
    project: str | None = None,
    file_limit: int | None = None,
    dry_run: bool = False,
    since_days: float | None = None,
) -> dict[str, int]:
    """Parse ``path`` as ``source`` and store its sessions. Returns counts.

    ``since_days`` restricts a directory sweep to files modified within the last N days (by
    mtime). That is what makes the sweep cheap enough to run as a periodic backstop and to
    backfill a known gap window without re-parsing the whole tree (and holding every
    episode in memory). Storing stays idempotent via span_id, so re-sweeping overlapping
    windows is a no-op. ``project`` labels the episodes (see :class:`Source`; a parser that
    leaves a turn unlabeled gets it as the default). A file of a directory sweep that fails
    to parse is skipped with a warning; a single export that fails to parse raises.
    """
    src = SOURCES[source]
    if src.glob is None:
        files = [path]
    else:
        files = sorted(path.rglob(src.glob))
        if since_days is not None:
            cutoff = time.time() - since_days * 86400
            files = [f for f in files if f.stat().st_mtime >= cutoff]
        if file_limit:
            files = files[:file_limit]

    logger.info("Scanning %d %s file(s) under %s", len(files), source, path)

    by_session: dict[str, list[Episode]] = defaultdict(list)
    parsed_files = 0
    parsed_episodes = 0

    for f in files:
        try:
            eps = src.parse(f, project)
        except Exception as e:
            if src.glob is None:
                raise  # the export is the whole run; carrying on would report an empty success
            logger.warning("Failed to parse %s: %s", f, e)
            continue
        parsed_files += 1
        parsed_episodes += len(eps)
        for ep in eps:
            by_session[ep.session_id].append(ep)

    logger.info(
        "Parsed %d files into %d episodes across %d sessions",
        parsed_files,
        parsed_episodes,
        len(by_session),
    )

    totals = StoreStats()
    if not dry_run:
        for i, (session_id, eps) in enumerate(by_session.items(), start=1):
            totals.add(store_episodes(db, eps, project_default=project))
            rebuild_chunks(db, session_id)
            if i % 100 == 0:
                logger.info("Progress: %d / %d sessions ingested", i, len(by_session))
        logger.info(
            "Done. %d new episodes written across %d sessions", totals.written, len(by_session)
        )
        totals.log_skipped()
    return {
        "files": parsed_files,
        "episodes_parsed": parsed_episodes,
        "sessions": len(by_session),
        "episodes_written": totals.written,
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(description="Backfill transcripts and chat exports")
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        help="Directory of transcript files, or the export (the codex default is "
        f"{SOURCES['codex'].default_root})",
    )
    parser.add_argument(
        "--source", choices=sorted(SOURCES), default="claude-code", help="What kind of path this is"
    )
    parser.add_argument("--project", default=None, help="Project tag for all episodes")
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

    src = SOURCES[args.source]
    path = args.path or src.default_root
    if path is None:
        parser.error(f"--source {args.source} needs a path")
    if src.glob is None and (args.limit or args.since_days is not None):
        parser.error(f"--limit and --since-days walk directories; {args.source} is one export")
    if not path.exists():
        print(f"Error: {path} does not exist", file=sys.stderr)
        return 2

    db = Database(get_settings().db_url)
    try:
        stats = backfill(
            args.source,
            path,
            db,
            project=args.project,
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
