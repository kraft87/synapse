#!/usr/bin/env python3
# mypy: ignore-errors
"""dream->config review CLI — accept/reject the behavioral rules the dream lane proposed, and write
accepted rules onto disk (stdlib only, DSN-free).

    config_review.py list                   # rule edits awaiting a decision or a disk write
    config_review.py show <id>              # the rule + the corrections that evidence it
    config_review.py accept <id> [--local] # write the rule to its file + mark applied
    config_review.py reject <id> [reason]  # drop it
    config_review.py apply                  # write every rule approved elsewhere (the dashboard)

accept is the whole loop: it appends the rule to its config file on THIS surface (e.g.
~/.claude/rules/learned.md) and flips the proposal to 'applied'. The human review IS the gate —
read `show <id>` first, since accept edits a live config file. A rule approved on the dashboard
is only 'accepted' there: the disk write belongs to this CLI, so `list` shows it and `apply`
writes them all. --local records the edit as surface-local; default 'general' = applies across
the user's surfaces (cross-surface fan-out is a later phase, so today both write here). Talks to
/config/proposals* (machine-token gated); no DB.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")

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
        # An older server lists only the proposed rows and sends no status.
        state = "  APPROVED, not yet written: `apply`" if r.get("status") == "accepted" else ""
        print(
            f"[{r['id']}] {r['kind']} -> {r['file_key']} ({r['scope']})  "
            f"{r['sessions']} session(s){state}"
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


def _safe_file_key(file_key) -> bool:
    """A server-supplied config file path that stays under CONFIG_DIR: relative, no `..`, no
    drive or backslash tricks, no control characters (skills_sync._safe_path's rule)."""
    if not isinstance(file_key, str) or not 0 < len(file_key) <= 512 or file_key.startswith("/"):
        return False
    if any(ch in file_key for ch in "\\:") or _CONTROL.search(file_key):
        return False
    return all(seg and seg not in (".", "..") and seg == seg.strip() for seg in file_key.split("/"))


def _apply_rule(file_key: str, rule: str) -> tuple[str, bool]:
    """Append the rule (as a bullet) to CONFIG_DIR/file_key on this surface. Returns (path, wrote).
    Idempotent: if a line that is exactly this rule is already present, nothing is written (a
    longer line that merely contains it does not count). Creates the file with a header if it
    doesn't exist yet. The rule is one line (its whitespace collapsed). Raises ValueError for a
    file_key outside CONFIG_DIR or an empty rule."""
    if not _safe_file_key(file_key):
        raise ValueError(f"refusing to write outside {config.CONFIG_DIR}: {file_key!r}")
    rule = " ".join(rule.split())
    if not rule:
        raise ValueError("the proposal has no rule text")
    path = config.CONFIG_DIR / file_key
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    line = f"- {rule}"
    if line in {ln.strip() for ln in existing.splitlines()}:
        return str(path), False
    base = existing.rstrip() if existing.strip() else _HEADER.rstrip()
    path.write_text(base + "\n" + line + "\n", encoding="utf-8", newline="\n")
    return str(path), True


def _write_accepted(cid: int, file_key: str, rule: str, scope: str) -> None:
    """Write accepted proposal `cid` to its file on this surface, then mark it applied. A write
    that is refused or fails leaves it 'accepted', so `apply` (or `accept <id>`) can retry."""
    try:
        path, wrote = _apply_rule(file_key, rule)
    except (ValueError, OSError) as e:
        print(f"accepted [{cid}], not written: {e}")
        return
    done = config.post_json("/config/proposals/act", {"id": cid, "action": "apply"})
    if done.get("status") != "applied":
        print(f"accepted [{cid}], wrote {path}, but server apply failed: {done.get('detail')}")
        return
    where = "appended to" if wrote else "already in"
    fan = "" if scope == "local" else "  (scope=general; cross-surface fan-out is a later phase)"
    print(f"applied [{cid}] -> {where} {path}{fan}")


def cmd_accept(cid: int, local: bool) -> None:
    scope = "local" if local else None  # None -> keep the proposal's stored scope (general)
    r = config.act("/config", cid, "accept", expect="accepted", scope=scope)
    if r is None:
        return
    _write_accepted(cid, r["file_key"], r.get("rule") or r.get("summary") or "", r.get("scope"))


def cmd_apply() -> None:
    rows = config.post_json("/config/proposals", {}).get("proposals", [])
    accepted = [r for r in rows if r.get("status") == "accepted"]
    if not accepted:
        print("no accepted config proposals to write.")
        return
    for r in accepted:
        _write_accepted(r["id"], r["file_key"], r.get("rule") or r.get("summary") or "", r["scope"])


def cmd_reject(cid: int, reason: str | None) -> None:
    r = config.act("/config", cid, "reject", expect="rejected", reason=reason)
    if r is None:
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
    sub.add_parser("apply")
    args = ap.parse_args()
    if args.cmd == "list":
        cmd_list()
    elif args.cmd == "show":
        cmd_show(args.id)
    elif args.cmd == "accept":
        cmd_accept(args.id, args.local)
    elif args.cmd == "reject":
        cmd_reject(args.id, args.reason)
    elif args.cmd == "apply":
        cmd_apply()


if __name__ == "__main__":
    main()
