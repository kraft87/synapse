#!/usr/bin/env python3
# mypy: ignore-errors
"""Claude Code ``SessionStart`` hook → push git commits to the Synapse timeline.

The timeline's git feeder: one naked event per commit ("committed to <project>:
<subject>", author-dated), POSTed to the machine-token-gated ``/timeline/events``
route. The SERVER embeds and upserts — this script holds no DSN and no Voyage key,
same thin-client seam as ingest_hook / skills_sync / config_sync.

OFF until repos are configured: set ``SYNAPSE_TIMELINE_REPOS`` (env or plugin option)
to a comma/space-separated list of local checkout paths. Each session start re-reads
a bounded window (``SYNAPSE_TIMELINE_SINCE_DAYS``, default 30) — re-pushes are free
because the server filters existing (source, source_ref) pairs before embedding, so
idempotency needs no local cursor state. For a first-time full-history backfill, run
once with SYNAPSE_TIMELINE_SINCE_DAYS=0 (no --since bound).

Salience is coarse (0=low 1=med 2=high), heuristic, decided here where the numstat
churn is cheap: merges/chore/ci/docs/bumps -> low; feat/fix/perf with big churn or a
PR ref -> high; else med.

Fail-open: any error (repo missing, server down) exits 0 so session start never breaks.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import _cfg, post_json

_US = "\x1f"
_LOW_PREFIX = ("chore", "ci", "docs", "style", "test", "build", "revert")
_HI_PREFIX = ("feat", "fix", "perf")
_MAX_BATCH = 1000  # matches the route's per-call cap


def _repos() -> list[Path]:
    raw = _cfg("SYNAPSE_TIMELINE_REPOS", "")
    return [Path(os.path.expanduser(p)) for p in re.split(r"[,\s]+", raw) if p]


def _salience(subject: str, is_merge: bool, churn: int) -> int:
    s = subject.lower()
    if is_merge or s.startswith(("merge ", "bump ")) or "dependabot" in s:
        return 0
    prefix = subject.split(":", 1)[0].strip().lower().split("(", 1)[0]
    if prefix in _LOW_PREFIX:
        return 0
    if prefix in _HI_PREFIX and (churn > 200 or "#" in subject):
        return 2
    return 1


def _read_commits(repo: Path, since_days: int) -> list[dict]:
    """One `git log --numstat` subprocess for the whole window. %aI author-date
    (rebases mutate commit-date), %P parents (merge = >1), \\x1f field delimiter
    (subjects can contain anything except control chars)."""
    fmt = f"@@C@@{_US}%H{_US}%aI{_US}%P{_US}%s"
    cmd = ["git", "-C", str(repo), "log", "--numstat", f"--pretty=format:{fmt}"]
    if since_days > 0:
        cmd.append(f"--since={since_days} days ago")
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    commits: list[dict] = []
    cur: dict | None = None
    for line in out.splitlines():
        if line.startswith("@@C@@"):
            if cur:
                commits.append(cur)
            _, sha, adate, parents, subject = line.split(_US, 4)
            cur = {
                "sha": sha,
                "t_valid": adate,
                "is_merge": len(parents.split()) > 1,
                "subject": subject,
                "churn": 0,
            }
        elif cur and "\t" in line:
            a, d, *_ = line.split("\t")
            cur["churn"] += (int(a) if a.isdigit() else 0) + (int(d) if d.isdigit() else 0)
    if cur:
        commits.append(cur)
    return commits


def _push_repo(repo: Path, since_days: int) -> tuple[int, int]:
    project = repo.name
    commits = _read_commits(repo, since_days)
    if not commits:
        return 0, 0
    events = [
        {
            "t_valid": c["t_valid"],
            "fact": f"committed to {project}: {c['subject']}",
            "source": f"git:{project}",
            "source_ref": c["sha"],
            "project": project,
            "salience": _salience(c["subject"], c["is_merge"], c["churn"]),
            # a commit IS an action; decisions/findings live in chat events
            "event_type": "action",
        }
        for c in commits
    ]
    inserted = skipped = 0
    for i in range(0, len(events), _MAX_BATCH):
        r = post_json("/timeline/events", {"events": events[i : i + _MAX_BATCH]}, timeout=60)
        inserted += int(r.get("inserted", 0))
        skipped += int(r.get("skipped", 0))
    return inserted, skipped


def main() -> None:
    repos = _repos()
    if not repos:
        return  # feature off until repos are configured
    since_days = int(_cfg("SYNAPSE_TIMELINE_SINCE_DAYS", "30") or "30")
    for repo in repos:
        if not (repo / ".git").exists():
            continue  # quiet skip — hooks stay silent in the user's session
        try:
            _push_repo(repo, since_days)
        except Exception:  # fail-open and quiet — never break or spam session start
            continue


if __name__ == "__main__":
    main()
