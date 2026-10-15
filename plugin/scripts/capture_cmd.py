#!/usr/bin/env python3
# mypy: ignore-errors
"""synapse capture — what web capture would store from your history, and the backfill.

    synapse capture --dry-run  [--since 30d]    # report only; nothing is stored
    synapse capture --backfill [--since 30d]    # ship the history for real (needs capture on)

The server captures web research from the transcript records the Stop hook posts (any
tool; `SYNAPSE_WEB_CAPTURE=on` on the server turns it on). `--dry-run` posts your local
transcripts, the Claude Code sessions with their sub-agent transcripts and the Codex
rollouts, to `POST /capture/dry-run`, which runs the exact detector, rules and guardrails
and answers what it would store, by tool, and what it would skip and why. Read it, then
turn capture on. `--backfill` posts the same history to /ingest, where the now-enabled
capture stores it. With capture off it would only ingest turns, so the command refuses unless
the server says capture is on.

`--since` bounds the history by each transcript's last activity: `30d`, `12h`, `2w`, or a
date (`2026-09-01`). The default is everything. A backfill also ingests any turn the server
has not stored yet, as `synapse import` does (and the server extracts it); a turn it has is
skipped by span_id, so a re-run is safe.

The dry run counts a page fetched twice in two batches twice, and applies the server's
per-POST and per-day caps per batch; a live backfill dedups and counts them as it goes.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import urllib.error
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import import_history
import ingest_hook
import synapse_project

_ROLLOUT_ID = re.compile(r"rollout-.*-([0-9a-f-]{36})\.jsonl$")
_SINCE = re.compile(r"(\d+)\s*([hdw])", re.I)
_UNIT_SECONDS = {"h": 3600, "d": 86400, "w": 7 * 86400}
_SAMPLES = 3  # captured calls shown per tool
_TIMEOUT = 120.0


def parse_since(text: str, now: float | None = None) -> float:
    """The epoch second ``text`` names: `30d` / `12h` / `2w` ago, or a `YYYY-MM-DD` date."""
    now = time.time() if now is None else now
    m = _SINCE.fullmatch(text.strip())
    if m:
        return now - int(m.group(1)) * _UNIT_SECONDS[m.group(2).lower()]
    try:
        return datetime.strptime(text.strip(), "%Y-%m-%d").timestamp()
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a span like 30d, 12h or 2w, or a date like 2026-09-01"
        ) from None


# ---------------------------------------------------------------- the local history


def codex_rollouts(sessions_root: Path) -> list[Path]:
    """Codex rollout files under ``sessions_root``, oldest first."""
    found = []
    for path in sessions_root.rglob("rollout-*.jsonl"):
        if _ROLLOUT_ID.search(path.name):
            try:
                found.append((path.stat().st_mtime, path))
            except OSError:
                continue
    return [p for _, p in sorted(found)]


def _codex_cwd(records: list[dict[str, Any]]) -> str | None:
    for rec in records:
        payload = rec.get("payload")
        if rec.get("type") in ("session_meta", "turn_context") and isinstance(payload, dict):
            if payload.get("cwd"):
                return str(payload["cwd"])
    return None


def batches_of(path: Path, batch_size: int) -> Iterator[dict[str, Any]]:
    """The POST bodies for one transcript. A Claude Code transcript goes in turn-aligned
    batches (the history import's cut: a turn's identity is its last record). A Codex
    rollout goes whole, as the Codex hook sends one it has no cursor for: its turn
    boundary is the parser's to find."""
    with open(path, "rb") as f:
        records = ingest_hook._parse_lines([line.strip() for line in f if line.strip()])
    match = _ROLLOUT_ID.search(path.name)
    if match is None:
        for batch in import_history.batches(records, batch_size):
            yield ingest_hook._payload(batch, "import", inline=True)
        return
    body: dict[str, Any] = {
        "records": records,
        "format": "codex",
        "session_id": match.group(1),
        "source": "codex-import",
    }
    cwd = _codex_cwd(records)
    if cwd and (project := synapse_project.project_label(cwd)):
        body["project"] = project
    yield body


# ---------------------------------------------------------------- the report


def new_total() -> dict[str, Any]:
    return {"files": 0, "failed": 0, "calls": 0, "ignored": 0, "errors": 0, "tools": {}}


def merge(total: dict[str, Any], report: dict[str, Any]) -> None:
    """Add one batch's `/capture/dry-run` reply to the running total."""
    for key in ("calls", "ignored", "errors"):
        total[key] += int(report.get(key) or 0)
    for name, part in (report.get("tools") or {}).items():
        tool = total["tools"].setdefault(name, {"captured": 0, "skipped": {}, "samples": []})
        tool["captured"] += int(part.get("captured") or 0)
        for reason, n in (part.get("skipped") or {}).items():
            tool["skipped"][reason] = tool["skipped"].get(reason, 0) + int(n)
        room = _SAMPLES - len(tool["samples"])
        tool["samples"].extend((part.get("samples") or [])[: max(room, 0)])


def render(total: dict[str, Any], enabled: bool) -> str:
    tools = total["tools"]
    captured = sum(t["captured"] for t in tools.values())
    skipped = sum(sum(t["skipped"].values()) for t in tools.values())
    lines = [
        f"{total['calls']} tool call(s) in {total['files']} transcript(s): "
        f"{captured} would be captured, {skipped} skipped, "
        f"{total['ignored']} had nothing to capture (no URL or query)."
        + (f" {total['errors']} could not be read." if total["errors"] else ""),
        "Capture is ON on this server."
        if enabled
        else "Capture is OFF on this server (SYNAPSE_WEB_CAPTURE); nothing has been stored.",
    ]
    for name, tool in sorted(tools.items(), key=lambda kv: (-kv[1]["captured"], kv[0])):
        why = ", ".join(f"{r} {n}" for r, n in sorted(tool["skipped"].items()))
        lines.append("")
        lines.append(f"{name}: {tool['captured']} captured" + (f", skipped: {why}" if why else ""))
        for s in tool["samples"]:
            what = s.get("url") or s.get("title") or "(no title)"
            title = f" - {s['title']}" if s.get("title") and s.get("url") else ""
            lines.append(f"    {s.get('kind')} -> {s.get('destination')}: {what[:100]}{title[:80]}")
    return "\n".join(lines)


# ---------------------------------------------------------------- the commands


def server_state() -> dict[str, Any] | None:
    """What `/capture/dry-run` says of the server (an empty list of records is the question
    alone), or None after saying why not."""
    try:
        return config.post_json("/capture/dry-run", {"records": []}, timeout=30)
    except urllib.error.HTTPError as e:
        if e.code in (404, 405):
            print(
                "synapse capture: this server has no /capture/dry-run; update the server "
                "first (web capture arrived with it).",
                file=sys.stderr,
            )
        else:
            print(f"synapse capture: the server answered HTTP {e.code}.", file=sys.stderr)
    except (urllib.error.URLError, OSError) as e:
        print(f"synapse capture: could not reach the server: {e}", file=sys.stderr)
    return None


def _files(args: argparse.Namespace) -> list[Path]:
    since = args.since
    claude = import_history.discover(Path(os.path.expanduser(args.projects_dir)))
    codex_root = Path(os.path.expanduser(args.codex_dir))
    codex = codex_rollouts(codex_root) if codex_root.is_dir() else []
    files = claude + codex
    if since is not None:
        files = import_history.filter_by_mtime(files, since, None)
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="synapse capture",
        description="See what web capture would store from your history, or backfill it.",
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="report only; nothing is stored")
    mode.add_argument("--backfill", action="store_true", help="ship the history for real")
    ap.add_argument(
        "--since", type=parse_since, metavar="SPAN", help="30d, 12h, 2w or a date (default: all)"
    )
    ap.add_argument("--projects-dir", default=str(config.PROJECTS_DIR), help="Claude Code root")
    ap.add_argument("--codex-dir", default="~/.codex/sessions", help="Codex rollout root")
    ap.add_argument("--batch-size", type=int, default=500, help="records per POST (default 500)")
    ap.add_argument("--yes", "-y", action="store_true", help="--backfill: skip the confirmation")
    args = ap.parse_args(argv)

    state = server_state()
    if state is None:
        return 1
    enabled = bool(state.get("enabled"))
    if args.backfill and not enabled:
        print(
            "synapse capture: web capture is off on this server, so a backfill would store "
            "nothing. Run --dry-run, then set SYNAPSE_WEB_CAPTURE=on on the server.",
            file=sys.stderr,
        )
        return 1

    files = _files(args)
    if not files:
        print("No transcripts found - nothing to do.")
        return 0
    if args.backfill and not args.yes:
        print(f"{len(files)} transcript file(s) will be posted to {ingest_hook.INGEST_URL}.")
        print("Turns the server has not stored are ingested too, and extracted (LLM usage).")
        try:
            answer = input("Proceed? [y/N] ")
        except EOFError:
            print("No interactive terminal for the confirmation - re-run with --yes.")
            return 1
        if answer.strip().lower() not in ("y", "yes"):
            print("Aborted - nothing was sent.")
            return 1

    path = "/capture/dry-run" if args.dry_run else "/ingest"
    total = new_total()
    stored = 0
    for i, file in enumerate(files, 1):
        try:
            for body in batches_of(file, args.batch_size):
                reply = config.post_json(path, body, timeout=_TIMEOUT)
                if args.dry_run:
                    merge(total, reply)
                else:
                    stored += int(reply.get("captured") or 0)
            total["files"] += 1
        except Exception as e:  # one unreadable file or a failed POST costs that file
            total["failed"] += 1
            print(f"[{i}/{len(files)}] {file.name} -> FAILED ({type(e).__name__}: {str(e)[:120]})")
        if i % 25 == 0:
            print(f"[{i}/{len(files)}] transcripts", file=sys.stderr, flush=True)

    print()
    if args.dry_run:
        print(render(total, enabled))
    else:
        print(f"Done: {stored} page(s) captured from {total['files']}/{len(files)} transcript(s).")
    if total["failed"]:
        print(f"{total['failed']} transcript(s) failed - re-run to retry them.")
    return 1 if total["files"] == 0 else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. A backfill resumes where it left off; the server dedups.")
        sys.exit(130)
