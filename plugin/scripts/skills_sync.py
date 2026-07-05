#!/usr/bin/env python3
# mypy: ignore-errors
"""SessionStart hook — two-way sync of skills between this machine and Synapse.

Per skill, compares a whole-skill content hash (body + bundled files) on disk vs on the
server. If they differ, the side with the newer ``content_modified_at`` wins (newest EDIT,
not newest sync — pulls os.utime the files to match, so a late sync can't masquerade as a
late edit). New-on-one-side flows to the other. Deletes never auto-propagate: a folder
missing on disk is treated as "not pulled yet", not "deleted" — removal is an explicit
server-side status='retired' (this sync just stops pulling it; it never deletes a whole local
skill). Within a skill being pulled, though, the folder is mirrored to the server's file set
(stale files removed) so the whole-skill hash can converge — and a push is never allowed to
shrink the server's file set, so a partial/degraded local copy can't wipe a complete one.

Scope routes the target dir:
  * 'global'          <-> CLAUDE_SKILLS_DIR (~/.claude/skills)
  * 'project:<name>'  <-> $CLAUDE_PROJECT_DIR/.claude/skills, only when the session is in a
                         project whose dir basename matches <name>.

DSN-FREE: talks to the server's /skills/* HTTP routes (machine-token gated), never Postgres.
Without a reachable server it is a silent no-op. Fail-open: never blocks/fails session start;
emits reloadSkills only when something actually synced. OPT-IN: enable with
SYNAPSE_SKILLS_SYNC=1 (default off — a hook that writes into ~/.claude/skills should
never be a surprise, issue #9).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def _skill_description(text: str) -> str:
    """Frontmatter `description` (YAML block-scalar / quoted / plain) — the trigger surface the
    server lane reads as its catalog."""
    m = re.search(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return ""
    lines = m.group(1).split("\n")
    for i, line in enumerate(lines):
        dm = re.match(r"^(\s*)description:\s*(.*)$", line)
        if not dm:
            continue
        base_indent, val = len(dm.group(1)), dm.group(2).strip()
        if val[:1] in ("|", ">"):  # block scalar — collect more-indented following lines
            collected = []
            for nxt in lines[i + 1 :]:
                if not nxt.strip():
                    continue
                if len(nxt) - len(nxt.lstrip()) <= base_indent:
                    break
                collected.append(nxt.strip())
            return " ".join(collected).strip()
        return val.strip("\"'")
    return ""


def _whole_hash(body: str, files) -> str:
    """Stable hash over body + sorted (relpath, sha256). Identical formula for disk and server."""
    h = hashlib.sha256()
    h.update(body.encode("utf-8"))
    for path, sha in sorted(files):
        h.update(b"\0")
        h.update(path.encode("utf-8"))
        h.update(b"\0")
        h.update(sha.encode())
    return h.hexdigest()


def _disk_skill(d: Path):
    """(body, files=[(rel, sha256, bytes, is_exec)], newest_mtime) for a skill dir, or None."""
    md = d / "SKILL.md"
    if not md.is_file():
        return None
    body = md.read_text(encoding="utf-8", errors="ignore")
    newest = md.stat().st_mtime
    files = []
    for f in sorted(d.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(d).as_posix()
        if rel == "SKILL.md":
            continue
        b = f.read_bytes()
        files.append((rel, hashlib.sha256(b).hexdigest(), b, os.access(f, os.X_OK)))
        newest = max(newest, f.stat().st_mtime)
    return body, files, newest


def _push(scope: str, name: str, disk) -> None:
    body, files, mtime = disk
    config.post_json(
        "/skills/publish",
        {
            "name": name,
            "scope": scope,
            "body": body,
            "description": _skill_description(body),
            "content_modified_at": datetime.fromtimestamp(mtime, tz=UTC).isoformat(),
            "files": [
                {
                    "path": rel,
                    "content_b64": base64.b64encode(content).decode(),
                    "sha256": sha,
                    "size": len(content),
                    "is_executable": is_exec,
                }
                for rel, sha, content, is_exec in files
            ],
        },
    )


def _pull(scope: str, target_dir: Path, name: str, cmod_iso: str | None) -> None:
    remote = config.post_json("/skills/fetch", {"name": name})
    if not remote.get("found"):
        return
    body = remote.get("body", "")
    sd = target_dir / name
    sd.mkdir(parents=True, exist_ok=True)
    md = sd / "SKILL.md"
    # Preserve a local edit we're about to overwrite that never reached the server -- the one
    # data-loss case the server-side history trigger can't see.
    if md.is_file():
        old = md.read_text(encoding="utf-8", errors="ignore")
        if old != body:
            config.post_json(
                "/skills/overwrite",
                {
                    "name": name,
                    "scope": scope,
                    "body": old,
                    "content_modified_at": datetime.fromtimestamp(
                        md.stat().st_mtime, tz=UTC
                    ).isoformat(),
                },
            )
    md.write_text(body, encoding="utf-8", newline="\n")
    when = None
    if cmod_iso:
        try:
            when = datetime.fromisoformat(cmod_iso).timestamp()
        except ValueError:
            when = None
    server_files = remote.get("files", [])
    for f in server_files:
        fp = sd / f["path"]
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_bytes(base64.b64decode(f["content_b64"]))
        if f.get("is_executable"):
            os.chmod(fp, os.stat(fp).st_mode | 0o111)
        if when:
            os.utime(fp, (when, when))
    if when:
        os.utime(md, (when, when))
    # Mirror the server's file set: a pull means the server's version of THIS skill wins
    # wholesale, so drop local files it no longer carries. Without this the whole-skill hash
    # can never match when disk has extra files, and the sync oscillates forever (pull, still
    # differs, pull, ...). Within-skill reconcile only — a whole missing skill is still treated
    # as not-yet-pulled, never as a delete.
    keep = {"SKILL.md", *(f["path"] for f in server_files)}
    for fp in sd.rglob("*"):
        if fp.is_file() and fp.relative_to(sd).as_posix() not in keep:
            fp.unlink()
    for sub in sorted(
        (p for p in sd.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True
    ):
        try:
            sub.rmdir()  # prune dirs left empty by the removals above
        except OSError:
            pass


def _sync(scope: str, target_dir: Path) -> tuple[int, int]:
    target_dir.mkdir(parents=True, exist_ok=True)

    listing = config.post_json("/skills/list", {"scope": scope}).get("skills", [])
    server = {
        s["name"]: (
            s.get("body", ""),
            [(f["path"], f["sha256"]) for f in s.get("files", [])],
            s.get("content_modified_at"),
        )
        for s in listing
    }

    disk = {}
    for md in sorted(target_dir.glob("*/SKILL.md")):
        r = _disk_skill(md.parent)
        if r:
            disk[md.parent.name] = r

    pulled = pushed = 0
    for name in sorted(set(server) | set(disk)):
        d, g = disk.get(name), server.get(name)
        if g and not d:  # only on server -> materialize
            _pull(scope, target_dir, name, g[2])
            pulled += 1
            continue
        if d and not g:  # only on disk -> publish
            _push(scope, name, d)
            pushed += 1
            continue
        body_d, files_d, mt_d = d
        body_g, fmeta_g, cmod_g = g
        if _whole_hash(body_d, [(f[0], f[1]) for f in files_d]) == _whole_hash(body_g, fmeta_g):
            continue  # identical content
        gt = 0.0
        if cmod_g:
            try:
                gt = datetime.fromisoformat(cmod_g).timestamp()
            except ValueError:
                gt = 0.0
        disk_files = {f[0] for f in files_d}
        srv_files = {f[0] for f in fmeta_g}
        # Newest edit wins — but treat a sub-second timestamp gap as a tie: content_modified_at
        # round-trips through a float on push/pull and loses precision, so an exact copy can come
        # back a hair "newer". On a tie, the more-complete (superset) copy wins.
        EPS = 2.0
        if mt_d > gt + EPS:
            winner = "push"
        elif gt > mt_d + EPS:
            winner = "pull"
        elif disk_files > srv_files:
            winner = "push"
        elif srv_files > disk_files:
            winner = "pull"
        else:
            winner = "pull"  # genuine tie, neither a superset -> server is the canonical copy
        # Never let a push REMOVE files from the server: the server fans out to every machine, so
        # a degraded/partial local copy (fewer files, e.g. from an interrupted pull) must not
        # overwrite a complete one. Push only when disk has all the server's files; else pull
        # (consistent with "deletes don't auto-propagate" — the missing files come back).
        if winner == "push" and not disk_files >= srv_files:
            winner = "pull"
        if winner == "push":
            _push(scope, name, d)
            pushed += 1
        else:
            _pull(scope, target_dir, name, cmod_g)
            pulled += 1
    return pulled, pushed


def main() -> None:
    if not config.SKILLS_SYNC:
        return  # opt-in feature; off unless SYNAPSE_SKILLS_SYNC=1
    try:
        n = _sync("global", config.SKILLS_DIR)
        proj = os.environ.get("CLAUDE_PROJECT_DIR")
        if proj:
            p = Path(proj)
            m = _sync(f"project:{p.name}", p / ".claude" / "skills")
            n = (n[0] + m[0], n[1] + m[1])
    except Exception:
        return  # fail-open: server/token/permission issues never break session start
    if n != (0, 0):
        print(
            json.dumps(
                {"hookSpecificOutput": {"hookEventName": "SessionStart", "reloadSkills": True}}
            )
        )


if __name__ == "__main__":
    main()
