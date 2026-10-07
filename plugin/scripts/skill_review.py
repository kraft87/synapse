#!/usr/bin/env python3
# mypy: ignore-errors
"""dream->skills review CLI — the GROUNDED accept/reject gate (stdlib only, DSN-free).

    skill_review.py list                    # proposed candidates
    skill_review.py show <id>               # evidence + the proposal as a unified diff against
                                            # the registry body (the full body for a new skill)
    skill_review.py accept <id> [--body-file PATH] [--scope SCOPE] [--force]
                                            # APPLIES it: registry write + status=promoted
    skill_review.py reject <id> [reason]    # grounded reject -> rejected + 30d cooldown
    skill_review.py promote <id>            # no-op kept for compatibility: accept applied it

accept/reject are the grounded signals (the LLM judge can only nominate). accept writes the
drafted SKILL.md — or --body-file, a version you edited — into the server's skill registry in
one transaction, through the same write a client publish makes. The machine that owns the
skill pulls it at its next session start via the two-way skills sync; when this machine syncs
the destination folder, the sync runs right away so the change is live here now.

Scope: a retune keeps the skill's scope; a new skill is global, or project:<name> when all its
evidence came from one project. --scope overrides either.

accept refuses, and says why, when there is no draft yet, when the registry body changed
since the draft was made, or when a new skill's name is already taken (--force overrides the
last two). For RETUNE accepts a server-side routing-eval runs first (advisory, never blocks).

Talks to the server's /skills/proposals* HTTP routes (machine-token gated); needs no DB access.
"""

from __future__ import annotations

import argparse
import difflib
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

_KIND_BADGE = {"retune": "RETUNE", "consolidate": "MERGE", "derive": "DERIVE"}
_NO_DRAFT = "no draft yet; re-drafts on the next nightly run, or pass --body-file to accept"


def cmd_list() -> None:
    rows = config.post_json("/skills/proposals", {}).get("proposals", [])
    if not rows:
        print("no proposals awaiting review.")
        return
    for r in rows:
        d = f"/{r['direction']}" if r.get("direction") else ""
        badge = _KIND_BADGE.get(r["kind"], r["kind"].upper())
        sal = f"  sal={r['salience']}" if r.get("salience") is not None else ""
        det = f"  via={r['source_detector']}" if r.get("source_detector") else ""
        draft = ""
        if r["kind"] != "consolidate" and "has_draft" in r:
            draft = "  [draft ready]" if r["has_draft"] else "  [no draft yet]"
        print(
            f"[{r['id']}] [{badge}{d}] {r['name']}  score={r['score']:.1f} "
            f"(g{r['grounded_sessions']}/j{r['judge_sessions']}){sal}{det}{draft}"
        )
        print(f"      {(r.get('summary') or '')[:110]}")


def _diff(old: str, new: str, name: str) -> str:
    """Unified diff of the registry body -> the proposed body ('' when identical)."""

    def lines(text: str) -> list[str]:
        return [ln if ln.endswith("\n") else ln + "\n" for ln in text.splitlines(keepends=True)]

    return "".join(
        difflib.unified_diff(
            lines(old),
            lines(new),
            fromfile=f"registry/{name}/SKILL.md",
            tofile=f"proposal/{name}/SKILL.md",
        )
    )


def cmd_show(cid: int) -> None:
    r = config.post_json("/skills/proposals", {"id": cid})
    if not r.get("found"):
        print("not found")
        return
    sal = f" salience={r['salience']}" if r.get("salience") is not None else ""
    det = f" via={r['source_detector']}" if r.get("source_detector") else ""
    print(
        f"[{r['id']}] {r['kind']} {r['name']} {r.get('direction') or ''}  "
        f"status={r['status']} score={r['score']:.2f}{sal}{det}"
    )
    print(f"summary: {r.get('summary')}\ntargets: {r.get('target_skills')}")
    ev = r.get("evidence") or []
    nights = sorted({e.get("scan_night") for e in ev if e.get("scan_night")})
    if nights:
        print(f"scan nights ({len(nights)}): {', '.join(nights)}")
    print(f"evidence ({len(ev)}):")
    for e in ev[:20]:
        night = f" night={e['scan_night']}" if e.get("scan_night") else ""
        print(
            f"  - {e.get('class')}/{e.get('signal')} sess={str(e.get('session_id'))[:8]}{night} "
            f"{e.get('skill') or ''} {e.get('why') or e.get('phrasing') or ''}".rstrip()
        )
        if e.get("quote"):
            print(f'      "{e["quote"][:200]}"')
    patch = r.get("proposed_patch")
    if patch:
        print(f"\n--- proposed patch ---\n{patch[:1500]}")

    target = r.get("apply_to")
    body = r.get("proposal_body")
    if not target:  # consolidate: merged by hand, never applied
        if body:
            print(f"\n--- drafted SKILL.md ---\n{body}")
        return
    what = "updates the existing skill" if target["exists"] else "creates a new skill"
    print(f"\naccept applies to: '{target['skill']}' ({target['scope']}) — {what}")
    for issue in r.get("apply_issues") or []:
        if issue["code"] != "no_draft":
            force = " [--force overrides]" if issue.get("force") else ""
            print(f"  ! {issue['detail']}{force}")
    if not body:
        print(f"\n{_NO_DRAFT}")
        return
    reg = r.get("registry_body")
    if reg is None:
        print(f"\n--- new skill: {target['skill']}/SKILL.md ---\n{body}")
        return
    print("\n--- proposed change (unified diff against the registry body) ---")
    print(_diff(reg, body, target["skill"]) or "(identical to the current registry body)")


def _local_dir(scope: str) -> Path | None:
    """The folder THIS machine syncs for `scope`, or None when it doesn't sync that scope.
    Mirrors skills_sync.main(): global <-> SKILLS_DIR; project:<name> <-> the session's
    project dir when its basename is <name> (and it isn't the global folder)."""
    if scope == "global":
        return config.SKILLS_DIR
    proj = os.environ.get("CLAUDE_PROJECT_DIR")
    if scope.startswith("project:") and proj and Path(proj).name == scope.split(":", 1)[1]:
        d = Path(proj) / ".claude" / "skills"
        if d.resolve() != config.SKILLS_DIR.resolve():
            return d
    return None


def _deliver(scope: str) -> None:
    """Say where the applied skill goes, and sync it here now if this machine is a
    destination. Reuses the SessionStart sync engine; fail-soft (the hook retries)."""
    if scope == "global":
        print("Every machine with skills sync on pulls it at its next session start.")
    else:
        print(
            f"The machine working in project '{scope.split(':', 1)[1]}' pulls it at its next "
            "session start (skills sync)."
        )
    d = _local_dir(scope) if config.SKILLS_SYNC else None
    if d is None:
        return
    try:
        import skills_sync

        pulled, pushed = skills_sync._sync(scope, d)
        print(f"Synced here now: pulled {pulled}, pushed {pushed} -> {d}")
    except Exception as e:
        print(f"Local sync failed ({e}); it syncs here at the next session start.")


def cmd_accept(cid: int, body_file: str | None, scope: str | None, force: bool) -> None:
    extra: dict = {}
    if body_file:
        try:
            extra["body"] = Path(body_file).read_text(encoding="utf-8")
        except OSError as e:
            sys.exit(f"can't read --body-file: {e}")
    if scope:
        extra["scope"] = scope
    if force:
        extra["force"] = True
    r = config.post_json("/skills/proposals/act", {"id": cid, "action": "accept", **extra})
    if r.get("routing_eval"):
        print(r["routing_eval"])
    status = r.get("status")
    if status == "promoted":
        verb = "created" if r.get("created") else "updated"
        print(
            f"accepted and applied [{cid}]: {verb} skill '{r['skill']}' ({r['scope']}) "
            "in the skill registry."
        )
        if r.get("forced"):
            print(f"  --force overrode: {', '.join(r['forced'])}")
        _deliver(r["scope"])
        return
    if status == "accepted":  # consolidate: the one kind accept records but doesn't apply
        print(
            f"accepted [{cid}]. Merge proposals aren't applied automatically: merge the skills "
            f"by hand, then `skill_review.py promote {cid}` to record it."
        )
        return
    if status == "refused":
        sys.exit(f"refusing [{cid}]: {r.get('detail')}")
    sys.exit(r.get("detail") or "not found")


def cmd_reject(cid: int, reason: str | None) -> None:
    r = config.act("/skills", cid, "reject", expect="rejected", reason=reason)
    if r is None:
        return
    print(f"rejected [{cid}] ({r.get('reason')}); suppressed 30d.")


def cmd_promote(cid: int) -> None:
    """Kept so old habits and scripts don't break: accept now applies the change, so this
    only reports that (or records a hand-merged consolidate)."""
    r = config.post_json("/skills/proposals/act", {"id": cid, "action": "promote"})
    if r.get("status") == "promoted":
        print(r.get("detail") or f"promoted [{cid}] {r.get('name')}.")
        return
    print(r.get("detail") or "not found")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for c in ("show", "accept", "reject", "promote"):
        p = sub.add_parser(c)
        p.add_argument("id", type=int)
        if c == "reject":
            p.add_argument("reason", nargs="?", default=None)
        if c == "accept":
            p.add_argument("--body-file", help="apply this SKILL.md instead of the draft")
            p.add_argument("--scope", help="'global' or 'project:<name>' (overrides the default)")
            p.add_argument(
                "--force",
                action="store_true",
                help="apply even if the skill changed since drafting, or the name is taken",
            )
    args = ap.parse_args()
    if args.cmd == "list":
        cmd_list()
    elif args.cmd == "show":
        cmd_show(args.id)
    elif args.cmd == "accept":
        cmd_accept(args.id, args.body_file, args.scope, args.force)
    elif args.cmd == "reject":
        cmd_reject(args.id, args.reason)
    elif args.cmd == "promote":
        cmd_promote(args.id)


if __name__ == "__main__":
    main()
