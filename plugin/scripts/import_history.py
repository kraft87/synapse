#!/usr/bin/env python3
# mypy: ignore-errors
"""synapse-import — bulk-import your existing Claude Code transcripts into Synapse.

Kills the cold-start problem: a fresh Synapse has an empty recall() until sessions
accumulate, but months of history usually already sit in ~/.claude/projects. This
ships every existing transcript FULL-LENGTH (not the Stop hook's bounded tail) to
the same /ingest endpoint the hook feeds, so a brand-new install can recall past
work immediately.

Safe to interrupt and re-run: the server dedups turns by span_id, so a re-run
skips already-ingested turns and picks up where it left off. Transcripts are only
read — nothing local is modified.

Endpoint + bearer resolve exactly like the Stop hook (via ingest_hook → config.py):
explicit env var → plugin userConfig → the /plugin install answers persisted in
settings.json → the http://localhost:8765 default.

Usage:
    synapse-import                       # interactive: summary, then y/N confirm
    synapse-import --yes                 # skip the confirmation
    synapse-import --projects-dir DIR    # transcript root (default ~/.claude/projects)
    synapse-import --batch-size N        # records per POST (default 500)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import ingest_hook  # shares the /ingest POST + URL/token resolution with the Stop hook

_PEEK_LINES = 5  # leading lines sniffed to decide "is this a session transcript?"


def _looks_like_transcript(path: Path) -> bool:
    """Cheap sniff: a Claude Code session file is JSONL whose records are dicts
    with a "type" key. Skips obvious non-session files (plain text, exports,
    other tools' JSONL) without reading the whole file."""
    try:
        with open(path, "rb") as f:
            for _ in range(_PEEK_LINES):
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    return False
                return isinstance(rec, dict) and "type" in rec
    except OSError:
        return False
    return False


def discover(projects_dir: Path) -> list[Path]:
    """All session transcripts under projects_dir, oldest-first by mtime, so the
    import lands in roughly chronological order."""
    found: list[tuple[float, Path]] = []
    for path in projects_dir.rglob("*.jsonl"):
        if path.name.startswith("."):
            continue
        try:
            st = path.stat()
        except OSError:
            continue
        if st.st_size == 0 or not _looks_like_transcript(path):
            continue
        found.append((st.st_mtime, path))
    found.sort(key=lambda t: t[0])
    return [p for _, p in found]


def _scan(path: Path) -> tuple[int, int]:
    """(records, turns) for one file — a byte scan with a substring pre-filter so
    only candidate user records pay for a JSON parse. A turn is what the hook's
    ``_is_turn_start`` says it is (same boundary rule the server's parser uses)."""
    records = turns = 0
    try:
        with open(path, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records += 1
                if b'"user"' not in line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if isinstance(rec, dict) and ingest_hook._is_turn_start(rec):
                    turns += 1
    except OSError:
        pass
    return records, turns


def batches(records: list[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    """Split records into POST batches that only ever cut at turn boundaries.

    A cut anywhere else would split one turn across two POSTs, and a partial turn
    carries the wrong span_id (identity is the turn's LAST record uuid). So a
    batch may run past batch_size until the next turn start — that's the point.
    """
    batch: list[dict[str, Any]] = []
    for rec in records:
        if len(batch) >= batch_size and ingest_hook._is_turn_start(rec):
            yield batch
            batch = []
        batch.append(rec)
    if batch:
        yield batch


def ship_file(path: Path, batch_size: int) -> int:
    """POST one transcript full-length in turn-aligned batches. Returns how many
    genuinely-new turns the server ingested (dedup silently skips the rest)."""
    with open(path, "rb") as f:
        raw_lines = [line.strip() for line in f if line.strip()]
    records = ingest_hook._parse_lines(raw_lines)
    ingested = 0
    for batch in batches(records, batch_size):
        reply = ingest_hook._post_records(batch, source="import")
        try:
            ingested += int(json.loads(reply).get("ingested", 0))
        except Exception:
            pass  # non-JSON reply — nothing to count, keep shipping
    return ingested


def _fmt_size(n: float) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="synapse-import",
        description="Bulk-import existing Claude Code transcripts into Synapse "
        "(same /ingest endpoint as the Stop hook; span_id dedup makes re-runs safe).",
    )
    ap.add_argument(
        "--projects-dir",
        default=str(config.PROJECTS_DIR),
        help="transcript root to scan (default: %(default)s)",
    )
    ap.add_argument("--batch-size", type=int, default=500, help="records per POST (default 500)")
    ap.add_argument("--yes", "-y", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args(argv)

    projects_dir = Path(os.path.expanduser(args.projects_dir))
    if not projects_dir.is_dir():
        print(f"synapse-import: no such directory: {projects_dir}", file=sys.stderr)
        return 1

    files = discover(projects_dir)
    if not files:
        print(f"No transcripts found under {projects_dir} — nothing to import.")
        return 0

    total_bytes = sum(p.stat().st_size for p in files)
    total_turns = sum(_scan(p)[1] for p in files)
    token_state = "configured" if ingest_hook.INGEST_TOKEN else "none"
    print(f"Found {len(files)} transcript file(s) under {projects_dir}")
    print(f"  total size       {_fmt_size(total_bytes)}")
    print(f"  estimated turns  ~{total_turns}")
    print(f"  target           {ingest_hook.INGEST_URL} (auth token: {token_state})")
    print()
    print("Importing runs KG extraction on the server's configured LLM for every NEW")
    print("turn — that consumes subscription usage or API credits, roughly proportional")
    print("to the turn count above. Already-ingested turns are skipped (dedup by")
    print("span_id), so interrupting with Ctrl-C and re-running is always safe: the")
    print("import resumes where it left off.")
    print()

    if not args.yes:
        try:
            answer = input("Proceed? [y/N] ")
        except EOFError:
            print("No interactive terminal for the confirmation — re-run with --yes.")
            return 1
        if answer.strip().lower() not in ("y", "yes"):
            print("Aborted — nothing was sent.")
            return 1

    ok = failed = ingested_total = 0
    for i, path in enumerate(files, 1):
        try:
            rel: object = path.relative_to(projects_dir)
        except ValueError:
            rel = path.name
        try:
            n = ship_file(path, args.batch_size)
        except Exception as e:
            failed += 1
            print(
                f"[{i}/{len(files)}] {rel} → FAILED ({type(e).__name__}: {str(e)[:120]})",
                flush=True,
            )
            continue
        ok += 1
        ingested_total += n
        print(f"[{i}/{len(files)}] {rel} → ingested {n}", flush=True)

    print()
    print(f"Done: {ingested_total} new turn(s) ingested from {ok}/{len(files)} file(s).")
    if failed:
        print(f"{failed} file(s) failed — re-run to retry; already-ingested turns are skipped.")
    return 1 if ok == 0 else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is saved server-side (dedup by span_id) — re-run to resume.")
        sys.exit(130)
