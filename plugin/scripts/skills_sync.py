#!/usr/bin/env python3
# mypy: ignore-errors
"""SessionStart hook — two-way sync of skills between this machine and Synapse.

Per skill, compares a whole-skill content hash (body + bundled files) on disk vs on the
server. If they differ, the side with the newer ``content_modified_at`` wins (newest EDIT,
not newest sync — pulls os.utime the files to match, so a late sync can't masquerade as a
late edit). New-on-one-side flows to the other. A local delete never propagates: a folder
missing on disk is treated as "not pulled yet", not "deleted". Removal is decided by the
server, through ONE path (``_settle_unserved``) for every way a skill stops being this
folder's: a retirement (`synapse skill retire`), a move to another scope (`synapse skill
scope`), or a skill this device may no longer read (trust: a restricted device is served
only its allowlisted projects' skills). The listing names those ("retired", and
"withdrawn" for the held names it no longer serves), and a local copy of one is deleted
when it is still exactly a version Synapse delivered (whatever its timestamp: a clone, a
restore or a copy renews an mtime, not the content), left alone when Synapse doesn't
recognise it, and, for a retirement only, pushed back (reactivating the skill) when it was
edited after the retirement. A withdrawn copy is never pushed back: the server refuses a
push into another scope's skill, and one from a device that may not write the scope. A
copy in another scope's folder is recognised only by the exact delivered contents: that
folder's own history is of whatever skill it held. A folder git tracks, or one linked in
from elsewhere, is never deleted. Within a skill
being pulled, the folder is mirrored to the server's file set (stale files removed) so the
whole-skill hash can converge — and a push is never allowed to shrink the server's file
set, so a partial/degraded local copy can't wipe a complete one.

"Delivered" is remembered per target folder in ~/.synapse/skills/: every whole-skill hash
this machine's copy had while it agreed with the server (after a pull, after a push, or
found identical), so an older copy that missed later edits is still recognised.

Scope routes the target dir:
  * 'global'          <-> CLAUDE_SKILLS_DIR (~/.claude/skills)
  * 'project:<label>' <-> <project root>/.claude/skills, where <label> is the session's
                         project as every hook labels it and <project root> the folder it
                         belongs to (synapse_project: a .synapse-project marker's folder,
                         else the git toplevel, else $CLAUDE_PROJECT_DIR itself). Claude Code
                         reads .claude/skills from the start folder up to the repository
                         root, so a session started in a subfolder sees them, and a
                         repository has one copy rather than one per subfolder.

A project skill pulled into a folder inside a git repository is added to that repository's
.git/info/exclude (not .gitignore): it comes from Synapse, not from the repository.

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
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import synapse_project


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


#: SKILL.md's temp name while a pull writes it: never part of a skill (a pull cut short
#: can leave it; the next pull of that skill removes it).
_PULL_TMP = ".SKILL.md.synapse-tmp"


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
        if rel in ("SKILL.md", _PULL_TMP):
            continue
        b = f.read_bytes()
        files.append((rel, hashlib.sha256(b).hexdigest(), b, os.access(f, os.X_OK)))
        newest = max(newest, f.stat().st_mtime)
    return body, files, newest


#: Publish replies that wrote nothing: retired after this copy's last edit, a skill that
#: lives in another scope, or a scope this device may not write.
_REFUSED = ("retired", "scoped", "forbidden")


def _push(scope: str, name: str, disk, client=config) -> bool:
    """Publish the disk copy. False when the server refused it (``_REFUSED``)."""
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
    return not (isinstance(r, dict) and r.get("status") in _REFUSED)


_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def _safe_name(name) -> bool:
    """A skill name usable as one folder name here (the server's safe_skill_name rule):
    anything else from a server is never written, so no name can escape the folder."""
    return (
        isinstance(name, str)
        and 0 < len(name) <= 128
        and name == name.strip()
        and name not in (".", "..")
        and not any(ch in name for ch in "/\\:")
        and not _CONTROL.search(name)
    )


def _safe_path(path) -> bool:
    """A bundled file path that stays inside the skill's folder (server: safe_file_path)."""
    if not isinstance(path, str) or not 0 < len(path) <= 512 or path.startswith("/"):
        return False
    if any(ch in path for ch in "\\:") or _CONTROL.search(path):
        return False
    return all(seg and seg not in (".", "..") and seg == seg.strip() for seg in path.split("/"))


def _pull(scope: str, target_dir: Path, name: str, cmod_iso: str | None, client=config) -> None:
    remote = client.post_json("/skills/fetch", {"name": name})
    if not remote.get("found"):
        return
    if not _safe_name(name) or not all(_safe_path(f.get("path")) for f in remote.get("files", [])):
        return  # never write outside the folder, whatever a server sends
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
    server_files = remote.get("files", [])
    # Mirror the server's file set first: a pull means the server's version of THIS skill wins
    # wholesale, so drop local files it no longer carries. Without this the whole-skill hash
    # can never match when disk has extra files, and the sync oscillates forever (pull, still
    # differs, pull, ...). Within-skill reconcile only — a whole missing skill is still treated
    # as not-yet-pulled, never as a delete. Removing before writing means a pull cut short
    # never leaves a local file the server dropped beside newly stamped ones (on an mtime
    # tie the copy with more files wins, and would be pushed).
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
    when = None
    if cmod_iso:
        try:
            when = datetime.fromisoformat(cmod_iso).timestamp()
        except ValueError:
            when = None
    for f in server_files:
        fp = sd / f["path"]
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_bytes(base64.b64decode(f["content_b64"]))
        if f.get("is_executable"):
            os.chmod(fp, os.stat(fp).st_mode | 0o111)
        if when:
            os.utime(fp, (when, when))
    # SKILL.md last, whole and already carrying the server's time: a pull cut short (a
    # session hook's timeout kills the sync) leaves the old SKILL.md with its old mtime, so
    # the next sync pulls again. Written first, a fresh mtime on the new body would make the
    # half-pulled copy look like a newer local edit, and its stale bundled files would be
    # pushed over the server's. The temp file is never read as part of the skill.
    tmp = sd / _PULL_TMP
    tmp.write_text(body, encoding="utf-8", newline="\n")
    if when:
        os.utime(tmp, (when, when))
    os.replace(tmp, md)


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


def _ledger_path(target_dir: Path, kind: str = "delivered") -> Path:
    """One of this folder's ledgers. "delivered": every version it held while it agreed
    with the server (pulled, pushed, or seen identical; clients before 0.22 also counted
    a push a newer server refused). "pulled": only the versions a pull wrote here, so
    exactly what Synapse put in this folder, never the folder's own content."""
    key = hashlib.sha256(str(target_dir.resolve()).encode("utf-8")).hexdigest()[:16]
    return config.synapse_home.home() / "skills" / f"{kind}-{key}.json"


def _load_ledger(target_dir: Path, kind: str = "delivered") -> dict:
    try:
        data = json.loads(_ledger_path(target_dir, kind).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    # Only lists of hashes count: anything else reads as nothing remembered.
    return {k: [h for h in v if isinstance(h, str)] for k, v in data.items() if isinstance(v, list)}


def _save_ledger(target_dir: Path, ledger: dict, kind: str = "delivered") -> None:
    path = _ledger_path(target_dir, kind)
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


def _git_tracked(folder: Path) -> bool | None:
    """True when git tracks a file under ``folder``, False when it tracks none (or the
    folder is in no repository), None when that can't be told (no git, git refused)."""
    try:
        here = folder.resolve()
    except OSError:
        return None
    top = next((d for d in (here, *here.parents) if (d / ".git").exists()), None)
    if top is None:
        return False
    try:
        r = subprocess.run(
            ["git", "-C", str(top), "ls-files", "-z", "--", here.relative_to(top).as_posix()],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return bool(r.stdout) if r.returncode == 0 else None


def _settle_unserved(
    scope,
    target_dir: Path,
    name: str,
    disk,
    entry: dict,
    ledger: dict,
    client,
    *,
    retired: bool,
    may_publish: bool = True,
    pulled: dict | None = None,
):
    """Settle a local copy of a skill this folder is no longer served. The one removal
    path (issue #222 part 6's retire flow, generalised), for a retirement (``retired``)
    and for a skill withdrawn from this folder: moved to another scope, or one this device
    may not read. Returns:

    * "removed" — still exactly a version Synapse delivered (whatever its timestamp: a
      clone, a restore or a copy renews that). The entry vouches for the versions
      Synapse held ("content_hash", "content_hashes"); ``pulled`` for what a pull wrote
      into this folder; the folder's "delivered" ledger (what it held while it agreed
      with the server) only for a retirement in this folder's own scope.
    * "pushed" — retirement only: edited after the retirement, so newest edit wins and the
      push reactivates it.
    * "kept" — anything else: never touch what Synapse doesn't recognise, never delete a
      folder git tracks or one linked in from elsewhere, and never push a withdrawn copy
      back (the server would refuse it anyway).

    A name is one registry row, in whatever scope it lives now, so this folder may be
    another scope's (the entry's "scope"; "" = not this folder's). Its "delivered" ledger
    then may describe a different skill of the same name (two folders each pushed their
    own to an older server), so only what a pull wrote here counts there."""
    body, files, mtime = disk
    digest = _whole_hash(body, [(f[0], f[1]) for f in files])
    retired_at = _ts(entry.get("retired_at"))
    if retired and retired_at is None:
        return "kept"
    folder = target_dir / name
    entry_scope = entry.get("scope")
    own_scope = entry_scope is None or str(entry_scope).lower() == scope.lower()
    vouched = {entry.get("content_hash"), *(entry.get("content_hashes") or [])} - {None, ""}
    # What a pull wrote here is the server's content wherever the row lives now, and it
    # still counts after the server changed the bundled files (they keep no history). The
    # "delivered" ledger also holds the folder's own pushes (and, from older clients,
    # pushes a newer server refused), so it vouches only for a retirement in this scope.
    if (
        digest in vouched
        or digest in (pulled or {}).get(name, [])
        or (retired and own_scope and digest in ledger.get(name, []))
    ):
        if _is_link(folder):
            return "kept"  # linked in from somewhere else: not a folder Synapse made
        if _git_tracked(folder) is not False:
            return "kept"  # the repository's own copy (or can't tell): its owner decides
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
    if retired and may_publish and mtime > retired_at + EPS:
        return "pushed" if _push(scope, name, disk, client=client) else "kept"
    return "kept"


def _entries(reply: dict, key: str) -> dict:
    return {r["name"]: r for r in reply.get(key) or [] if isinstance(r, dict) and r.get("name")}


def _sync(
    scope: str, target_dir: Path, client=config, *, git_exclude: bool = False
) -> tuple[int, int, int]:
    """One two-way pass over ``target_dir``. Returns (pulled, pushed, removed).

    ``git_exclude`` (project folders): add each skill folder this pass pulls to its
    repository's .git/info/exclude. A skill folder the repository tracks is its own: git
    distributes it, so no pull writes into it (or lists it) and it is only published. The
    folder is created only when a skill is pulled into it: a project with no project
    skills gets no empty skills folder."""
    disk = {}
    for md in sorted(target_dir.glob("*/SKILL.md")):
        r = _disk_skill(md.parent)
        if r:
            disk[md.parent.name] = r

    # "held" asks the server which of these it no longer serves here ("withdrawn").
    reply = client.post_json("/skills/list", {"scope": scope, "held": sorted(disk)})
    server = {}
    for s in reply.get("skills", []):
        if not _safe_name(s.get("name")):
            continue
        files = [(f["path"], f["sha256"]) for f in s.get("files", [])]
        # The server names each skill's whole-skill hash; one that predates the field sends
        # the body, hashed here (a pull refetches the body, so the listing needs no copy).
        digest = s.get("content_hash") or _whole_hash(s.get("body", ""), files)
        server[s["name"]] = (digest, files, s.get("content_modified_at"))
    # Retired and withdrawn skills come in their own keys, which older clients don't read:
    # listed with the active ones, an old client would pull (nothing) or push them back
    # every session.
    retired = _entries(reply, "retired")
    withdrawn = _entries(reply, "withdrawn")
    # False when this device may not write this scope (a restricted device's global
    # folder): its own skills stay local instead of being refused every session.
    may_publish = reply.get("publish") is not False

    def pulled_here(name: str) -> None:
        if git_exclude:
            synapse_project.git_exclude(target_dir / name)

    ledger = _load_ledger(target_dir)
    pulled_ledger = _load_ledger(target_dir, "pulled")

    def repository_owns(name: str, digest: str | None = None) -> bool:
        """An existing project skill folder git tracks: never pulled over. When git can't
        answer (no git, a refused repository), only a copy Synapse delivered here (its
        ``digest`` is in the ledger) or a folder with no SKILL.md is updated."""
        folder = target_dir / name
        if not git_exclude or not folder.exists():
            return False
        tracked = _git_tracked(folder)
        if tracked is None:
            return digest is not None and digest not in ledger.get(name, [])
        return tracked

    pulled = pushed = removed = 0
    for name in sorted(set(server) | set(disk)):
        d, g = disk.get(name), server.get(name)
        if g and not d:  # only on server -> materialize
            if repository_owns(name):
                continue
            _pull(scope, target_dir, name, g[2], client=client)
            _remember(ledger, name, g[0])
            _remember(pulled_ledger, name, g[0])
            pulled_here(name)
            pulled += 1
            continue
        if d and not g and (name in retired or name in withdrawn):
            is_retired = name in retired
            entry = retired[name] if is_retired else withdrawn[name]
            outcome = _settle_unserved(
                scope,
                target_dir,
                name,
                d,
                entry,
                ledger,
                client,
                retired=is_retired,
                may_publish=may_publish,
                pulled=pulled_ledger,
            )
            if outcome == "pushed":  # now the server's copy again, like any push
                _remember(ledger, name, _whole_hash(d[0], [(f[0], f[1]) for f in d[1]]))
            pushed += outcome == "pushed"
            removed += outcome == "removed"
            continue
        if d and not g and not may_publish:
            continue  # a local skill this device keeps to itself
        if d and not g:  # only on disk -> publish
            if _push(scope, name, d, client=client):
                _remember(ledger, name, _whole_hash(d[0], [(f[0], f[1]) for f in d[1]]))
                pushed += 1
            continue
        body_d, files_d, mt_d = d
        hash_g, fmeta_g, cmod_g = g
        disk_hash = _whole_hash(body_d, [(f[0], f[1]) for f in files_d])
        if disk_hash == hash_g:
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
        if winner == "push" and not may_publish:
            continue  # a newer local edit this device may not publish: left as it is
        if winner == "pull" and repository_owns(name, disk_hash):
            continue  # the repository's copy: git, not Synapse, updates it here
        if winner == "push":
            if _push(scope, name, d, client=client):
                _remember(ledger, name, disk_hash)
                pushed += 1
        else:
            _pull(scope, target_dir, name, cmod_g, client=client)
            _remember(ledger, name, hash_g)
            _remember(pulled_ledger, name, hash_g)
            pulled_here(name)
            pulled += 1
    _save_ledger(target_dir, ledger)
    _save_ledger(target_dir, pulled_ledger, "pulled")
    return pulled, pushed, removed


def main() -> None:
    if not config.SKILLS_SYNC:
        return  # opt-in feature; off unless SYNAPSE_SKILLS_SYNC=1
    try:
        n = _sync("global", config.SKILLS_DIR)
        proj = os.environ.get("CLAUDE_PROJECT_DIR")
        label, root = synapse_project.resolve(proj) if proj else (None, None)
        proj_dir = (root or Path(proj)) / ".claude" / "skills" if proj else None
        # A session whose project dir IS the home dir (cwd=~) resolves the project skills
        # folder to the global one. Syncing that folder twice under two scopes flips every
        # skill's scope on each session start (the registry is unique on name), so the global
        # list is empty between sessions and no other host or plugin ever sees the skills.
        if label and proj_dir and proj_dir.resolve() != config.SKILLS_DIR.resolve():
            m = _sync(f"project:{label}", proj_dir, git_exclude=True)
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
