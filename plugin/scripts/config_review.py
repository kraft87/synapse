#!/usr/bin/env python3
# mypy: ignore-errors
"""dream->config review CLI — accept/reject the behavioral rules the dream lane proposed, and write
accepted rules onto disk (stdlib only, DSN-free).

    config_review.py list                   # proposed rule edits awaiting review
    config_review.py show <id>              # the rule + the corrections that evidence it
    config_review.py accept <id> [--local] # write the rule to its file + mark applied
    config_review.py reject <id> [reason]  # drop it

accept is the whole loop: it appends the rule to its config file on THIS surface (e.g.
~/.claude/rules/learned.md) and flips the proposal to 'applied'. The human review IS the gate —
read `show <id>` first, since accept edits a live config file. --local records the edit as
surface-local; default 'general' = applies across the user's surfaces (cross-surface fan-out is a
later phase, so today both write here). Talks to /config/proposals* (machine-token gated); no DB.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

_HEADER = (
    "# Learned rules\n\n"
    "_Behavioral rules the Synapse dream->config lane mined from your corrections and you accepted "
    "via `/synapse:config-review`. Edit or prune freely._\n"
)


def cmd_list() -> None:
    rows = config.post_json("/config/proposals", {}).get("proposals", [])
    if not rows:
        print("no config proposals awaiting review.")
        return
    for r in rows:
        print(
            f"[{r['id']}] {r['kind']} -> {r['file_key']} ({r['scope']})  {r['sessions']} session(s)"
        )
        print(f"      {(r.get('summary') or '')[:110]}")


def cmd_show(cid: int) -> None:
    r = config.post_json("/config/proposals", {"id": cid})
    if not r.get("found"):
        print("not found")
        return
    print(f"[{r['id']}] {r['kind']} -> {r['file_key']} ({r['scope']})  status={r['status']}")
    print(f"rule: {r.get('diff') or r.get('summary')}")
    ev = r.get("evidence") or []
    print(f"evidence ({len(ev)} correction(s)):")
    for e in ev[:20]:
        sess = str(e.get("session_id") or "")[:8]
        quote = (e.get("quote") or "").strip()
        print(f"  - sess={sess} {quote}".rstrip())


def _apply_rule(file_key: str, rule: str) -> tuple[str, bool]:
    """Append the rule (as a bullet) to CONFIG_DIR/file_key on this surface. Returns (path, wrote).
    Idempotent: if the exact rule line is already present, nothing is written. Creates the file with
    a header if it doesn't exist yet."""
    path = config.CONFIG_DIR / file_key
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    line = f"- {rule.strip()}"
    if line in existing:
        return str(path), False
    base = existing.rstrip() if existing.strip() else _HEADER.rstrip()
    path.write_text(base + "\n" + line + "\n", encoding="utf-8", newline="\n")
    return str(path), True


def cmd_accept(cid: int, local: bool) -> None:
    scope = "local" if local else None  # None -> keep the proposal's stored scope (general)
    r = config.post_json("/config/proposals/act", {"id": cid, "action": "accept", "scope": scope})
    if r.get("status") != "accepted":
        print(r.get("detail") or "not found")
        return
    path, wrote = _apply_rule(r["file_key"], r.get("rule") or r.get("summary") or "")
    done = config.post_json("/config/proposals/act", {"id": cid, "action": "apply"})
    if done.get("status") != "applied":
        print(f"accepted [{cid}], wrote {path}, but server apply failed: {done.get('detail')}")
        return
    where = "appended to" if wrote else "already in"
    fan = (
        ""
        if r.get("scope") == "local"
        else "  (scope=general; cross-surface fan-out is a later phase)"
    )
    print(f"applied [{cid}] -> {where} {path}{fan}")


def cmd_reject(cid: int, reason: str | None) -> None:
    r = config.post_json("/config/proposals/act", {"id": cid, "action": "reject", "reason": reason})
    if r.get("status") != "rejected":
        print(r.get("detail") or "not found")
        return
    print(f"rejected [{cid}] ({r.get('reason')}).")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    s = sub.add_parser("show")
    s.add_argument("id", type=int)
    a = sub.add_parser("accept")
    a.add_argument("id", type=int)
    a.add_argument("--local", action="store_true")
    j = sub.add_parser("reject")
    j.add_argument("id", type=int)
    j.add_argument("reason", nargs="?", default=None)
    args = ap.parse_args()
    if args.cmd == "list":
        cmd_list()
    elif args.cmd == "show":
        cmd_show(args.id)
    elif args.cmd == "accept":
        cmd_accept(args.id, args.local)
    elif args.cmd == "reject":
        cmd_reject(args.id, args.reason)


if __name__ == "__main__":
    main()
