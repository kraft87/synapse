#!/usr/bin/env python3
# mypy: ignore-errors
"""SessionStart hook — two-way sync of skills between this machine and Synapse.

Per skill, compares a whole-skill content hash (body + bundled files) on disk vs on the
server. If they differ, the side with the newer ``content_modified_at`` wins (newest EDIT,
not newest sync — pulls os.utime the files to match, so a late sync can't masquerade as a
late edit). New-on-one-side flows to the other. A local delete never propagates: a folder
missing on disk is treated as "not pulled yet", not "deleted". Removal is an explicit
server-side retirement (`synapse skill retire`), and it follows the same newest-edit rule:
a local copy of a retired skill is deleted when it is still exactly a copy Synapse delivered
(whatever its timestamp: a clone, a restore or a copy renews an mtime, not the content),
pushed back (reactivating the skill) when Synapse doesn't recognise it and it was edited
after the retirement, and otherwise left alone, as is any folder Synapse doesn't
recognise. A copy in another scope's folder is recognised only by the exact retired
content: that folder's own history is of whatever skill it held. Within a skill
being pulled, the folder is mirrored to the server's file set (stale files removed) so the
whole-skill hash can converge — and a push is never allowed to shrink the server's file
set, so a partial/degraded local copy can't wipe a complete one.

"Delivered" is remembered per target folder in ~/.synapse/skills/: every whole-skill hash
this machine's copy had while it agreed with the server (after a pull, after a push, or
found identical), so an older copy that missed later edits is still recognised.

Scope routes the target dir:
  * 'global'          <-> CLAUDE_SKILLS_DIR (~/.claude/skills)
  * 'project:<name>'  <-> $CLAUDE_PROJECT_DIR/.claude/skills, only when the session is in a
                         project whose dir basename matches <name>.

DSN-FREE: talks to the server's /skills/* HTTP routes (machine-token gated), never Postgres.
Without a reachable server it is a silent no-op. Fail-open: never blocks/fails session start;
emits reloadSkills only when something actually synced. OPT-IN: enable with
SYNAPSE_SKILLS_SYNC=1 (default off — a hook that writes into ~/.claude/skills should
never be a surprise, issue #9).

The HTTP client is injectable (``client=``, anything with ``post_json``) so plugin-codex
reuses this engine against its own config instead of carrying a second copy of the
merge rules.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
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


def _push(scope: str, name: str, disk, client=config) -> bool:
    """Publish the disk copy. False when the server refused it: the skill was retired
    after this copy's last edit (and this client's listing predates that)."""
    body, files, mtime = disk
    r = client.post_json(
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
    return not (isinstance(r, dict) and r.get("status") == "retired")


def _pull(scope: str, target_dir: Path, name: str, cmod_iso: str | None, client=config) -> None:
    remote = client.post_json("/skills/fetch", {"name": name})
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
            client.post_json(
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


#: Newest edit wins, but a sub-second gap is a tie: content_modified_at round-trips through
#: a float on push/pull and loses precision, so an exact copy can come back a hair "newer".
EPS = 2.0
_DELIVERED_KEEP = 20  # hashes remembered per skill


def _ts(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso).timestamp()
    except (TypeError, ValueError):
        return None


def _ledger_path(target_dir: Path) -> Path:
    key = hashlib.sha256(str(target_dir.resolve()).encode("utf-8")).hexdigest()[:16]
    return config.synapse_home.home() / "skills" / f"delivered-{key}.json"


def _load_ledger(target_dir: Path) -> dict:
    try:
        data = json.loads(_ledger_path(target_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    # Only lists of hashes count: anything else reads as nothing remembered.
    return {k: [h for h in v if isinstance(h, str)] for k, v in data.items() if isinstance(v, list)}


def _save_ledger(target_dir: Path, ledger: dict) -> None:
    path = _ledger_path(target_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(ledger, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass  # bookkeeping: without it a retired copy is left alone, never wrongly deleted


def _remember(ledger: dict, name: str, digest: str) -> None:
    seen = [h for h in ledger.get(name, []) if h != digest]
    ledger[name] = [*seen, digest][-_DELIVERED_KEEP:]


def _is_link(path: Path) -> bool:
    """A symlink, or a Windows junction (which is_symlink() doesn't report before 3.12,
    and which rmtree refuses)."""
    if path.is_symlink():
        return True
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return bool(getattr(st, "st_file_attributes", 0) & 0x400)  # FILE_ATTRIBUTE_REPARSE_POINT


def _retired_copy(scope, target_dir: Path, name: str, disk, entry: dict, ledger: dict, client):
    """Settle a local copy of a skill the server retired: "removed" (still exactly a copy
    Synapse delivered, whatever its timestamp: a clone, a restore or a copy renews that),
    "pushed" (edited after the retirement: newest edit wins, the push reactivates it), or
    "kept" (anything else: never touch what Synapse doesn't recognise, and never push it
    back either).

    A name is one registry row, retired whatever scope it was published from last, so
    this folder may be another scope's. Its ledger then describes whatever skill this
    folder holds under that name, which may be a different one: only the exact retired
    content is recognised there."""
    body, files, mtime = disk
    digest = _whole_hash(body, [(f[0], f[1]) for f in files])
    retired_at = _ts(entry.get("retired_at"))
    if retired_at is None:
        return "kept"
    folder = target_dir / name
    own_scope = entry.get("scope") in (None, scope)
    if digest == entry.get("content_hash") or (own_scope and digest in ledger.get(name, [])):
        if _is_link(folder):
            return "kept"  # linked in from somewhere else: not a folder Synapse made
        try:
            # SKILL.md first: a failure then leaves the copy untouched (the next session
            # retries), and once it is gone the rest is never read as a skill again.
            (folder / "SKILL.md").unlink()
        except OSError:
            return "kept"
        shutil.rmtree(folder, ignore_errors=True)
        # The ledger stays: a restored copy of any version this folder held is still
        # recognised, and removed again rather than pushed.
        _remember(ledger, name, digest)
        return "removed"
    if mtime > retired_at + EPS:
        return "pushed" if _push(scope, name, disk, client=client) else "kept"
    return "kept"


def _sync(scope: str, target_dir: Path, client=config) -> tuple[int, int, int]:
    """One two-way pass over ``target_dir``. Returns (pulled, pushed, removed)."""
    target_dir.mkdir(parents=True, exist_ok=True)

    reply = client.post_json("/skills/list", {"scope": scope})
    server = {
        s["name"]: (
            s.get("body", ""),
            [(f["path"], f["sha256"]) for f in s.get("files", [])],
            s.get("content_modified_at"),
        )
        for s in reply.get("skills", [])
    }
    # Retired skills come in their own key, which older clients don't read: listed with the
    # active ones, an old client would pull (nothing) or push them back every session.
    retired = {
        r["name"]: r for r in reply.get("retired") or [] if isinstance(r, dict) and r.get("name")
    }

    disk = {}
    for md in sorted(target_dir.glob("*/SKILL.md")):
        r = _disk_skill(md.parent)
        if r:
            disk[md.parent.name] = r

    ledger = _load_ledger(target_dir)
    pulled = pushed = removed = 0
    for name in sorted(set(server) | set(disk)):
        d, g = disk.get(name), server.get(name)
        if g and not d:  # only on server -> materialize
            _pull(scope, target_dir, name, g[2], client=client)
            _remember(ledger, name, _whole_hash(g[0], g[1]))
            pulled += 1
            continue
        if d and not g and name in retired:
            outcome = _retired_copy(scope, target_dir, name, d, retired[name], ledger, client)
            if outcome == "pushed":  # now the server's copy again, like any push
                _remember(ledger, name, _whole_hash(d[0], [(f[0], f[1]) for f in d[1]]))
            pushed += outcome == "pushed"
            removed += outcome == "removed"
            continue
        if d and not g:  # only on disk -> publish
            if _push(scope, name, d, client=client):
                _remember(ledger, name, _whole_hash(d[0], [(f[0], f[1]) for f in d[1]]))
                pushed += 1
            continue
        body_d, files_d, mt_d = d
        body_g, fmeta_g, cmod_g = g
        disk_hash = _whole_hash(body_d, [(f[0], f[1]) for f in files_d])
        if disk_hash == _whole_hash(body_g, fmeta_g):
            _remember(ledger, name, disk_hash)
            continue  # identical content
        gt = _ts(cmod_g) or 0.0
        disk_files = {f[0] for f in files_d}
        srv_files = {f[0] for f in fmeta_g}
        # On a tie, the more-complete (superset) copy wins.
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
            if _push(scope, name, d, client=client):
                _remember(ledger, name, disk_hash)
                pushed += 1
        else:
            _pull(scope, target_dir, name, cmod_g, client=client)
            _remember(ledger, name, _whole_hash(body_g, fmeta_g))
            pulled += 1
    _save_ledger(target_dir, ledger)
    return pulled, pushed, removed


def main() -> None:
    if not config.SKILLS_SYNC:
        return  # opt-in feature; off unless SYNAPSE_SKILLS_SYNC=1
    try:
        n = _sync("global", config.SKILLS_DIR)
        proj = os.environ.get("CLAUDE_PROJECT_DIR")
        proj_dir = Path(proj) / ".claude" / "skills" if proj else None
        # A session whose project dir IS the home dir (cwd=~) resolves the project skills
        # folder to the global one. Syncing that folder twice under two scopes flips every
        # skill's scope on each session start (the registry is unique on name), so the global
        # list is empty between sessions and no other host or plugin ever sees the skills.
        if proj_dir and proj_dir.resolve() != config.SKILLS_DIR.resolve():
            m = _sync(f"project:{Path(proj).name}", proj_dir)
            n = tuple(a + b for a, b in zip(n, m, strict=True))
    except Exception:
        return  # fail-open: server/token/permission issues never break session start
    if any(n):
        print(
            json.dumps(
                {"hookSpecificOutput": {"hookEventName": "SessionStart", "reloadSkills": True}}
            )
        )


if __name__ == "__main__":
    main()
