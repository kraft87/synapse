#!/usr/bin/env python3
"""
Cold-archive research-source tool results from CC transcript JSONLs.

Scans ~/.claude/projects recursively (main sessions AND subagent transcripts
under <session>/subagents/) for `bird thread` / `bsky thread` / yt-transcript
Bash tool results and stores them verbatim in the isolated research_archive
table. See ingestion/research_capture.py for design + kill switch.

Runs from the synapse-web-ingest cron; idempotent (per-file mtime checkpoint
+ per-source upsert where longer content wins).

Usage:
    uv run python scripts/capture_research_archive.py [--root PATH] [--limit N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.research_capture import CaptureStats, ResearchArchiveCapture

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest_research_brief import load_db_url


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(Path.home() / ".claude" / "projects"))
    ap.add_argument("--limit", type=int, default=None, help="stop after N files (testing)")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    files = sorted(root.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if args.limit:
        files = files[-args.limit :]

    stats = CaptureStats()
    with psycopg.connect(load_db_url()) as conn:
        cap = ResearchArchiveCapture(conn)
        for f in files:
            cap.capture_one(f, stats)

    print(f"research-archive capture: {stats.line()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
