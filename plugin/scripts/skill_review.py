#!/usr/bin/env python3
# mypy: ignore-errors
"""dream->skills review CLI — the GROUNDED accept/reject gate (stdlib only, DSN-free).

    skill_review.py list                 # proposed candidates
    skill_review.py show <id>            # full evidence + drafted SKILL.md
    skill_review.py accept <id>          # grounded accept -> status=accepted (writes the draft locally)
    skill_review.py reject <id> [reason] # grounded reject -> rejected + 30d cooldown
    skill_review.py promote <id>         # ONLY after you mv the draft into ~/.claude/skills (sets promoted)

accept/reject are the grounded signals that gate apply (the LLM judge can only nominate).
For RETUNE accepts, a server-side routing-eval runs first (advisory — shown, not blocking).
promote is the human-confirmed "it's live in ~/.claude/skills" transition — never automatic.

Talks to the server's /skills/proposals* HTTP routes (machine-token gated); needs no DB access.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def cmd_list() -> None:
    rows = config.post_json("/skills/proposals", {}).get("proposals", [])
    if not rows:
        print("no proposals awaiting review.")
        return
    for r in rows:
        d = f"/{r['direction']}" if r.get("direction") else ""
        print(
            f"[{r['id']}] {r['kind']}{d} {r['name']}  score={r['score']:.1f} "
            f"(g{r['grounded_sessions']}/j{r['judge_sessions']})"
        )
        print(f"      {(r.get('summary') or '')[:110]}")
        if r.get("proposal_path"):
            print(f"      draft: {r['proposal_path']}")


def cmd_show(cid: int) -> None:
    r = config.post_json("/skills/proposals", {"id": cid})
    if not r.get("found"):
        print("not found")
        return
    print(
        f"[{r['id']}] {r['kind']} {r['name']} {r.get('direction') or ''}  "
        f"status={r['status']} score={r['score']:.2f}"
    )
    print(f"summary: {r.get('summary')}\ntargets: {r.get('target_skills')}")
    ev = r.get("evidence") or []
    print(f"evidence ({len(ev)}):")
    for e in ev[:20]:
        print(
            f"  - {e.get('class')}/{e.get('signal')} sess={str(e.get('session_id'))[:8]} "
            f"{e.get('skill') or ''} {e.get('why') or e.get('phrasing') or ''}".rstrip()
        )
    body = r.get("proposal_body")
    if body:
        print(f"\n--- drafted SKILL.md ---\n{body[:1500]}")


def _write_draft(name: str, body: str) -> str | None:
    """Materialize the accepted draft locally so the human can edit + mv it into ~/.claude/skills."""
    if not body:
        return None
    d = config.PROPOSALS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(body, encoding="utf-8", newline="\n")
    return str(d / "SKILL.md")


def cmd_accept(cid: int) -> None:
    r = config.post_json("/skills/proposals/act", {"id": cid, "action": "accept"})
    if not r.get("found", True) or r.get("status") != "accepted":
        print(r.get("detail") or "not found")
        return
    if r.get("routing_eval"):
        print(r["routing_eval"])
    path = _write_draft(r.get("name", f"cand{cid}"), r.get("proposal_body") or "")
    print(f"accepted [{cid}].")
    if path:
        print(f"Draft written to: {path}")
    print(
        f"To go live: review/edit the draft, mv it into ~/.claude/skills/<name>/SKILL.md, "
        f"then `skill_review.py promote {cid}`."
    )


def cmd_reject(cid: int, reason: str | None) -> None:
    r = config.post_json("/skills/proposals/act", {"id": cid, "action": "reject", "reason": reason})
    if r.get("status") != "rejected":
        print(r.get("detail") or "not found")
        return
    print(f"rejected [{cid}] ({r.get('reason')}); suppressed 30d.")


def cmd_promote(cid: int) -> None:
    r = config.post_json("/skills/proposals/act", {"id": cid, "action": "promote"})
    if r.get("status") == "refused":
        print(f"refusing: {r.get('detail')}. Accept it first.")
        return
    if r.get("status") != "promoted":
        print(r.get("detail") or "not found")
        return
    print(
        f"promoted [{cid}] {r.get('name')}. (post-change firing will now be tracked via skill_usage.)"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for c in ("show", "accept", "reject", "promote"):
        p = sub.add_parser(c)
        p.add_argument("id", type=int)
        if c == "reject":
            p.add_argument("reason", nargs="?", default=None)
    args = ap.parse_args()
    if args.cmd == "list":
        cmd_list()
    elif args.cmd == "show":
        cmd_show(args.id)
    elif args.cmd == "accept":
        cmd_accept(args.id)
    elif args.cmd == "reject":
        cmd_reject(args.id, args.reason)
    elif args.cmd == "promote":
        cmd_promote(args.id)


if __name__ == "__main__":
    main()
