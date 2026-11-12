#!/usr/bin/env python3
"""Client releases: the client version, the files it covers, and its release notes.

Stdlib only, so CI runs it with the runner's python3 and no install. Two jobs use it, so
the rules they enforce are written once, here:

* ``release`` (.github/workflows/release.yml: a push to main that touches the client, or
  run by hand) tags ``v<version>`` and publishes a GitHub release whose notes are that
  version's HANDOFF.md entries.
* ``check-pr`` (the Version job in .github/workflows/ci.yml) holds a pull request that
  changes the client to a new, untagged version.

THE CLIENT is every file git tracks under plugin/ and plugin-codex/, minus bytecode and
tool directories: the tree the Claude marketplace copies from git (./plugin) and the one
the server's /install bundles (mcp_server/install_routes.py, whose file rule
``is_client_file`` mirrors; a test pins the two together). Markdown is in it:
plugin/commands/*.md are slash commands a machine runs, and the READMEs are in the
bundle. A machine updates only when the version number moves, so any change in this set
needs a new version.

``release`` decides, for the version at --rev:

  version malformed, or plugin.json and __version__ disagree  -> fail before any call
  no tag v<version>           -> create the tag at the newest commit that changed the
                                 client (an atomic create: it fails if the tag appeared
                                 meanwhile), then publish the release
  tag exists, client differs  -> fail: the client changed but the version is still X
  tag exists, client the same -> already released, green; a tag left without a release
                                 (made by hand, or its release deleted) gets one

``check-pr`` decides, from --base (the PR's base commit) to --head (its merge commit):

  no client file changed        -> pass; the version is not looked at
  client changed, version not raised above the base's -> fail
  client changed, v<new> already tagged               -> fail
  otherwise                     -> pass

Notes are every HANDOFF.md entry (a ``## `` heading and the lines under it) whose heading
names ``plugin <version>``, heading included, in file order, cut to GitHub's body limit.
With none, the notes say so and quote the commit subject.

    python3 scripts/release_notes.py version [--rev REV]
    python3 scripts/release_notes.py notes VERSION [--rev REV]
    python3 scripts/release_notes.py release [--rev REV]          # needs gh + GH_TOKEN
    python3 scripts/release_notes.py check-pr [--base REV] [--head REV]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath

MANIFEST = "plugin/.claude-plugin/plugin.json"
PACKAGE_INIT = "plugin/synapse_client/__init__.py"
HANDOFF = "HANDOFF.md"
CLIENT_TREES = ("plugin", "plugin-codex")
#: Directories whose contents are never client code: install_routes._SKIP_PARTS.
SKIP_PARTS = frozenset(
    {
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".venv",
        "venv",
        "node_modules",
        ".git",
    }
)
#: GitHub refuses a release body over 125,000 characters. Counting UTF-8 bytes instead
#: keeps any count of characters (or UTF-16 units) under it.
MAX_NOTES_BYTES = 125_000

# ASCII digits only ([0-9], not \d, which takes any Unicode digit), no leading zeros.
_VERSION_RE = re.compile(r"(0|[1-9][0-9]{0,8})\.(0|[1-9][0-9]{0,8})\.(0|[1-9][0-9]{0,8})")
_INIT_VERSION_RE = re.compile(r"""^__version__\s*(?::[^=\n]*)?=\s*["']([^"'\n]*)["']""", re.M)
# "plugin 0.24.4" anywhere in a heading: "(plugin 0.24.4)", "(plugin 0.23.0, schema 064)",
# "(#222, plugin 0.20.0, schema 060)". Not "plugin-codex", not 0.24.4 inside 0.24.41.
_HEADING_VERSION_RE = re.compile(
    r"(?<![\w-])plugin v?([0-9]+\.[0-9]+\.[0-9]+)(?![0-9]|\.[0-9])", re.IGNORECASE
)
_LISTED = 10  # changed files named in a failure message

GhRunner = Callable[[Sequence[str], bool], subprocess.CompletedProcess[str]]


class ReleaseError(Exception):
    """A refusal, worded for the person reading the CI log."""


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------


def check_version(value: object, where: str = "the version") -> str:
    """``value`` if it is a plain MAJOR.MINOR.PATCH string, else ReleaseError. It is the
    only thing from the repository that reaches a tag name, a shell or an API call."""
    if isinstance(value, str) and _VERSION_RE.fullmatch(value):
        return value
    shown = repr(value)
    if len(shown) > 60:
        shown = shown[:57] + "..."
    raise ReleaseError(f"{where}: {shown} is not a version (digits.digits.digits)")


def parse_version(value: str) -> tuple[int, int, int]:
    major, minor, patch = (int(part) for part in check_version(value).split("."))
    return major, minor, patch


# ---------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        encoding="utf-8",
        errors="surrogateescape",
        check=check,
    )


def _rev(repo: Path, spec: str) -> str | None:
    out = _git(repo, "rev-parse", "--verify", "--quiet", "--end-of-options", spec, check=False)
    return out.stdout.strip() if out.returncode == 0 else None


def resolve(repo: Path, rev: str) -> str:
    """The commit ``rev`` names, as a full hash."""
    commit = _rev(repo, f"{rev}^{{commit}}")
    if commit is None:
        raise ReleaseError(f"{rev!r} is not a commit in {repo}")
    return commit


def tag_commit(repo: Path, tag: str) -> str | None:
    """The commit tag ``tag`` points at, or None when there is no such tag."""
    return _rev(repo, f"refs/tags/{tag}^{{commit}}")


def read_file(repo: Path, commit: str, path: str) -> str | None:
    out = _git(repo, "show", f"{commit}:{path}", check=False)
    return out.stdout if out.returncode == 0 else None


def version_at(repo: Path, commit: str) -> str:
    """The client version at ``commit``: plugin.json's, validated, and equal to
    synapse_client.__version__ when the package is there."""
    raw = read_file(repo, commit, MANIFEST)
    if raw is None:
        raise ReleaseError(f"{MANIFEST} is missing at {commit[:12]}")
    try:
        manifest = json.loads(raw)
    except ValueError as exc:
        raise ReleaseError(f"{MANIFEST} at {commit[:12]} is not JSON: {exc}") from None
    value = manifest.get("version") if isinstance(manifest, dict) else None
    version = check_version(value, f"{MANIFEST} at {commit[:12]}")
    init = read_file(repo, commit, PACKAGE_INIT)
    if init is not None:
        found = _INIT_VERSION_RE.search(init)
        if found is None or found.group(1) != version:
            other = repr(found.group(1)) if found else "no __version__"
            raise ReleaseError(
                f"{MANIFEST} says {version} but {PACKAGE_INIT} has {other} at "
                f"{commit[:12]}: the two must agree"
            )
    return version


# ---------------------------------------------------------------------------
# The client file set
# ---------------------------------------------------------------------------


def is_client_file(path: str, mode: str = "100644") -> bool:
    """Whether a tracked path (repo-relative, '/'-separated) is part of the client.

    The same rule as install_routes._bundled for regular files: under plugin/ or
    plugin-codex/, not bytecode, under no tool directory. A symlink counts (the
    marketplace's git copy carries it, though /install leaves it out); a submodule does
    not (no one receives its contents from this repository)."""
    parts = path.split("/")
    if len(parts) < 2 or parts[0] not in CLIENT_TREES or mode == "160000":
        return False
    return PurePosixPath(path).suffix != ".pyc" and not SKIP_PARTS.intersection(parts[1:])


def client_tree(repo: Path, commit: str) -> dict[str, tuple[str, str]]:
    """path -> (mode, object id) for every client file at ``commit``."""
    out = _git(repo, "ls-tree", "-r", "-z", commit, "--", *CLIENT_TREES).stdout
    tree: dict[str, tuple[str, str]] = {}
    for item in filter(None, out.split("\0")):
        meta, path = item.split("\t", 1)
        mode, _kind, oid = meta.split(" ")
        if is_client_file(path, mode):
            tree[path] = (mode, oid)
    return tree


def client_changes(repo: Path, old: str, new: str) -> list[str]:
    """Client files added, removed, edited or re-moded between two commits."""
    before, after = client_tree(repo, old), client_tree(repo, new)
    return sorted(
        path for path in before.keys() | after.keys() if before.get(path) != after.get(path)
    )


def last_client_commit(repo: Path, commit: str) -> str:
    """The newest commit on ``commit``'s first-parent line (main's own history, never a
    commit inside a merged branch) that changed the client, so a release made later (by
    hand) still points where its client landed. ``commit`` itself when git's answer does
    not carry exactly the same client."""
    found = _git(
        repo, "log", "-1", "--first-parent", "--format=%H", commit, "--", *CLIENT_TREES
    ).stdout.strip()
    if found and not client_changes(repo, found, commit):
        return found
    return commit


def _listing(paths: list[str]) -> str:
    shown = "\n".join(f"  {path}" for path in paths[:_LISTED])
    more = f"\n  ... and {len(paths) - _LISTED} more" if len(paths) > _LISTED else ""
    return f"Changed client files ({len(paths)}):\n{shown}{more}"


def next_free(repo: Path, version: str) -> str:
    """The first patch version above ``version`` with no tag."""
    major, minor, patch = parse_version(version)
    while True:
        patch += 1
        candidate = f"{major}.{minor}.{patch}"
        if tag_commit(repo, f"v{candidate}") is None:
            return candidate


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def handoff_entries(text: str) -> list[str]:
    """Each entry of HANDOFF.md, heading line included, in file order. An entry is a
    ``## `` heading and every line up to the next ``## `` or ``# `` heading; a heading
    inside a fenced code block is part of the entry around it."""
    entries: list[list[str]] = []
    current: list[str] | None = None
    fence = ""
    for line in text.splitlines():
        stripped = line.lstrip()
        if fence:
            if stripped.startswith(fence):
                fence = ""
        elif stripped.startswith(("```", "~~~")):
            fence = stripped[:3]
        elif line.startswith("## "):
            current = [line]
            entries.append(current)
            continue
        elif line.startswith("# "):
            current = None
            continue
        if current is not None:
            current.append(line)
    return ["\n".join(lines).rstrip() for lines in entries]


def heading_versions(heading: str) -> set[str]:
    """The plugin versions a heading line names."""
    return set(_HEADING_VERSION_RE.findall(heading))


def cap_notes(body: str, max_bytes: int, tag: str) -> str:
    """``body`` cut at a line boundary to at most ``max_bytes`` UTF-8 bytes, with a line
    saying so; unchanged when it fits."""
    if len(body.encode("utf-8")) <= max_bytes:
        return body
    tail = f"\n\n(Cut to fit GitHub's release size limit: the full text is HANDOFF.md at {tag}.)\n"
    budget = max_bytes - len(tail.encode("utf-8"))
    kept: list[str] = []
    used = 0
    for line in body.splitlines(keepends=True):
        used += len(line.encode("utf-8"))
        if used > budget:
            break
        kept.append(line)
    return "".join(kept).rstrip() + tail


def notes_for(text: str, version: str, max_bytes: int = MAX_NOTES_BYTES) -> str | None:
    """Every HANDOFF.md entry whose heading names plugin ``version``, joined in file
    order and capped; None when there is none."""
    picked = [
        entry
        for entry in handoff_entries(text)
        if version in heading_versions(entry.split("\n", 1)[0])
    ]
    if not picked:
        return None
    return cap_notes("\n\n".join(picked) + "\n", max_bytes, f"v{version}")


def fallback_notes(version: str, commit: str, subject: str) -> str:
    """The honest body when HANDOFF.md has nothing for ``version``: never empty."""
    subject = subject.strip() or "(no commit subject)"
    return (
        f"No HANDOFF.md entry names plugin {version}. "
        f"This release is commit {commit[:12]}:\n\n> {subject}\n"
    )


def release_notes_at(repo: Path, commit: str, version: str, target: str) -> str:
    notes = notes_for(read_file(repo, commit, HANDOFF) or "", version)
    if notes is not None:
        return notes
    subject = _git(repo, "log", "-1", "--format=%s", target).stdout
    return fallback_notes(version, target, subject)


# ---------------------------------------------------------------------------
# The two jobs
# ---------------------------------------------------------------------------


def run_gh(args: Sequence[str], capture: bool) -> subprocess.CompletedProcess[str]:
    """gh, as the workflow has it: GH_TOKEN and GH_REPO come from the environment."""
    return subprocess.run(["gh", *args], capture_output=capture, text=True, check=False)


def release(repo: Path, rev: str, gh: GhRunner = run_gh) -> str:
    """Tag and publish the client version at ``rev`` if it is not released yet (see the
    module docstring for the table). Returns what happened; raises ReleaseError."""
    head = resolve(repo, rev)
    version = version_at(repo, head)
    tag = f"v{version}"
    tagged = tag_commit(repo, tag)
    if tagged is not None:
        changed = client_changes(repo, tagged, head)
        if changed:
            raise ReleaseError(
                f"Client changed but the version is still {version}: bump it. {tag} is "
                f"{tagged[:12]}, and {head[:12]} differs from it.\n{_listing(changed)}"
            )
        if gh(["release", "view", tag, "--json", "tagName"], True).returncode == 0:
            return f"{tag} is already released ({tagged[:12]}); the client is unchanged since."
        target = tagged
    else:
        target = last_client_commit(repo, head)
    notes = release_notes_at(repo, head, version, target)
    fd, notes_path = tempfile.mkstemp(prefix="release-notes-", suffix=".md")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(notes)
        if tagged is None:
            created = gh(
                [
                    "api",
                    "--method",
                    "POST",
                    "repos/{owner}/{repo}/git/refs",
                    "-f",
                    f"ref=refs/tags/{tag}",
                    "-f",
                    f"sha={target}",
                ],
                True,
            )
            if created.returncode != 0:
                raise ReleaseError(
                    f"Could not create tag {tag} at {target[:12]} (if it appeared since "
                    f"this run's checkout, re-run): {created.stderr.strip()}"
                )
        published = gh(
            ["release", "create", tag, "--verify-tag", "--title", tag, "--notes-file", notes_path],
            False,
        )
        if published.returncode != 0:
            raise ReleaseError(
                f"Tag {tag} is at {target[:12]} but its release was not published; "
                "re-run this workflow to publish it."
            )
    finally:
        os.unlink(notes_path)
    return f"Released {tag} at {target[:12]}" + ("" if tagged is None else " (existing tag)")


def check_pr(repo: Path, base_rev: str, head_rev: str) -> str:
    """The Version check: a pull request that changes the client raises the version to
    one with no tag. Returns the verdict; raises ReleaseError on a failure."""
    base, head = resolve(repo, base_rev), resolve(repo, head_rev)
    changed = client_changes(repo, base, head)
    if not changed:
        return "No client file changed; the client version is not checked."
    old, new = version_at(repo, base), version_at(repo, head)
    if parse_version(new) <= parse_version(old):
        moved = f"is still {old}" if new == old else f"went from {old} down to {new}"
        raise ReleaseError(
            f"Client changed but the version {moved}: bump it in {MANIFEST} and "
            f"{PACKAGE_INIT} (next free: {next_free(repo, old)}).\n{_listing(changed)}"
        )
    tagged = tag_commit(repo, f"v{new}")
    if tagged is not None:
        raise ReleaseError(
            f"Client {new} is already released (tag v{new} is {tagged[:12]}): pick a "
            f"version above it (next free: {next_free(repo, new)}).\n{_listing(changed)}"
        )
    return f"Client changed and its version goes {old} -> {new}; merging releases v{new}."


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _report(level: str, title: str, message: str) -> None:
    """A GitHub annotation inside Actions, a plain line elsewhere."""
    if os.environ.get("GITHUB_ACTIONS") == "true":
        data = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::{level} title={title}::{data}", flush=True)
    else:
        print(f"{level}: {message}", file=sys.stderr if level == "error" else sys.stdout)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--repo", type=Path, default=Path("."), help="repository root")
    sub = parser.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("version", help="print the client version at a commit")
    cmd.add_argument("--rev", default="HEAD")
    cmd = sub.add_parser("notes", help="print a version's HANDOFF.md entries")
    cmd.add_argument("version")
    cmd.add_argument("--rev", default="HEAD", help="read HANDOFF.md at this commit")
    cmd = sub.add_parser("release", help="tag and publish the version at --rev (uses gh)")
    cmd.add_argument("--rev", default="HEAD")
    cmd = sub.add_parser("check-pr", help="a client change must carry a new version")
    cmd.add_argument("--base", default="HEAD^1", help="default: the merge commit's base")
    cmd.add_argument("--head", default="HEAD")
    args = parser.parse_args(argv)
    repo = args.repo
    title = "Version" if args.command == "check-pr" else "Release"
    try:
        if args.command == "version":
            print(version_at(repo, resolve(repo, args.rev)))
        elif args.command == "notes":
            version = check_version(args.version)
            text = read_file(repo, resolve(repo, args.rev), HANDOFF) or ""
            notes = notes_for(text, version)
            if notes is None:
                raise ReleaseError(f"No HANDOFF.md entry names plugin {version}")
            sys.stdout.write(notes)
        elif args.command == "release":
            _report("notice", title, release(repo, args.rev))
        else:
            print(check_pr(repo, args.base, args.head))
    except ReleaseError as exc:
        _report("error", title, str(exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
