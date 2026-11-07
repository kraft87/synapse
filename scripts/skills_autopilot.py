#!/usr/bin/env python3
# mypy: ignore-errors
"""Run the dream->skills autopilot (dream/skills/autoreview.py) from a machine that has a
logged-in `claude` CLI. The dream container has none, so its nightly lane only reconciles
(rolls back what went wrong); this is the half that needs the Opus reviewer.

    SYNAPSE_DB_URL=postgresql://... \\
    SYNAPSE_LLM_PROVIDER=openai SYNAPSE_LLM_API_KEY=... SYNAPSE_SKILLS_VERIFY_MODEL=<non-Claude> \\
      uv run python scripts/skills_autopilot.py [--notify-cmd 'discord-send.sh infra'] [--shadow]

From cron, use scripts/skills_autopilot_cron.sh: it reads SYNAPSE_DB_URL and the verifier's key
from env files at run time, so the crontab carries no password.

What it does, in order: reconcile (roll back auto-applies that went wrong, repair a local edit
a sync pulled over), then judge each proposed, drafted, untouched row (reviewer: the stripped
claude CLI call; verifier: an independent non-Claude model, asked in both evidence orders) and
apply / auto-reject / leave it for a person per the policy in autoreview's docstring. Mode
comes from SKILLS_AUTOREVIEW (off | shadow | live; --shadow forces shadow for one run). Safe
to run any time and to run twice: a row is judged once per evidence+draft, and an advisory
lock keeps it from overlapping the nightly lane.

The report (auto-applied first, each with `undo: revert N`) is stored in
skill_scan_cursor.config.last_report and, with --notify-cmd, handed to that command as its
last argument (e.g. ~/scripts/discord-send.sh infra). Exit 2: the claude CLI is not usable.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dream.skills import autoreview, claude_cli, review_queue
from dream.skills import skill_ledger as L


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--notify-cmd", help="command that receives the report text as last argument")
    ap.add_argument("--shadow", action="store_true", help="record what it would do; change nothing")
    ap.add_argument(
        "--reconcile-only", action="store_true", help="roll back / repair only, no reviews"
    )
    a = ap.parse_args()

    mode = "shadow" if a.shadow else autoreview.current_mode()
    if mode != "off" and not a.reconcile_only:
        if not claude_cli.run(
            "Reply with the single word OK.", autoreview.reviewer_model()
        ).strip():
            print("claude CLI is not usable here (missing or not logged in); nothing done.")
            return 2
    conn = L.connect()
    cur = conn.cursor()
    cur.execute("SELECT now()")
    run_started = cur.fetchone()[0]
    res = autoreview.run(conn, run_started, review=not a.reconcile_only, mode=mode)
    print("autopilot:", autoreview.flat_counts(res), "mode:", res.get("effective_mode", mode))
    changes = {"auto_rejected": [], "expired": [], "autopilot": res}
    rep = review_queue.report(conn, run_started, changes, deliver=False)
    text = rep.get("text")
    if text:
        print(text)
        if a.notify_cmd:
            ok = subprocess.run([*shlex.split(a.notify_cmd), text]).returncode == 0
            if ok:
                review_queue.mark_sent(conn, run_started)
            else:
                print(
                    "(notify command failed; the report is in skill_scan_cursor.config.last_report)"
                )
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
