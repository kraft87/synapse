#!/usr/bin/env python3
# mypy: ignore-errors
"""The one place a client turns a working directory into a Synapse project label.

Every hook that names a project resolves it here: the ingest hook (the ``project`` it
sends with each /ingest body), the board hook (the board's project scope), the skills
sync (the ``project:<label>`` scope and the folder it syncs), and the Codex plugin's
session-start and Stop hooks, which load this file from ``plugin/scripts``. Labels used
to be derived four separate ways, so one repository showed up under eight names (a
full Windows path, a lowercase-drive variant, three casings, its subfolders), and a
restricted device's allowlist needed an entry per variant.

Resolution order for a directory:

1. A ``.synapse-project`` marker file: the first one found walking up from the
   directory to the git toplevel, inclusive (only the directory itself when it is in
   no repository); in a linked worktree with none, the main checkout's root marker
   (it may be deliberately left uncommitted). Its first non-empty, non-``#`` line is
   the label (UTF-8, with or without a BOM, or UTF-16 as Windows PowerShell 5.1's
   ``>`` writes it). The marker is how a repository whose folder names differ between
   machines (or clones, or worktrees) keeps one identity.
2. The basename of the git toplevel. A linked worktree (``git worktree add``, including
   the ones Claude Code makes under ``.claude/worktrees``) resolves to its main
   checkout, so a worktree is the same project as the checkout it came from; one of a
   bare repository to the repository's name without ``.git`` (or to the folder holding
   a hidden one, e.g. ``acme/.bare``). A repository at the home directory itself
   (dotfiles) is not the project of the folders under it.
3. The basename of the directory itself.

Every label is normalised the same way: both path separators are treated alike,
trailing separators are dropped, the last component is kept with its case, and a
bare drive (``C:``), ``.`` or ``..`` is no label at all. The server applies the same
rule (``ingestion.project_label``) to transcripts that arrive without a label: catch-up
sweeps and older plugins.

Stdlib only and no subprocess: the hooks run under whatever bare Python the CLI has,
and ``git`` may not be on its PATH (or may refuse a repo owned by another user), so
the repository is found by reading ``.git`` the way git lays it out on disk.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

MARKER = ".synapse-project"
_MAX_MARKER_BYTES = 4096
_MAX_LABEL = 200
_DRIVE = re.compile(r"^[A-Za-z]:$")


def label_from_path(path) -> str | None:
    """The last component of ``path`` as a label, either separator, case kept."""
    if not path:
        return None
    name = str(path).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].strip()
    if not name or name in (".", "..") or _DRIVE.match(name):
        return None
    return name[:_MAX_LABEL]


def _read_marker(path: Path) -> str | None:
    """The label a marker file names, or None when it names none (empty, unreadable)."""
    try:
        with open(path, "rb") as f:
            raw = f.read(_MAX_MARKER_BYTES)
    except OSError:
        return None
    # utf-8-sig: Notepad writes a BOM, which would otherwise become part of the label.
    # Windows PowerShell 5.1's `'name' > .synapse-project` writes UTF-16 with a BOM.
    codec = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
    for line in raw.decode(codec, errors="replace").replace("\x00", "").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return label_from_path(line)
    return None


def _gitdir_of(top: Path) -> Path | None:
    """The git directory ``top/.git`` stands for: the folder itself, or the one a
    ``gitdir:`` file (linked worktree, submodule) points at."""
    dot = top / ".git"
    if dot.is_dir():
        return dot
    if not dot.is_file():
        return None  # a FIFO or device named .git: never read (it could block forever)
    try:
        with open(dot, "rb") as f:
            first = f.read(4096).decode("utf-8", errors="replace").splitlines()[0]
    except (OSError, IndexError):
        return None
    if not first.startswith("gitdir:"):
        return None
    target = Path(first[len("gitdir:") :].strip())
    return target if target.is_absolute() else (top / target)


def _main_checkout(top: Path) -> Path:
    """The main working tree a linked worktree belongs to; ``top`` for anything else.

    A linked worktree's git dir is ``<main>/.git/worktrees/<name>`` and holds a
    ``commondir`` file pointing back at ``<main>/.git``. A submodule's git dir has no
    ``commondir``, so it stays itself. A bare repository has no main working tree: its
    worktrees are named for it (``acme.git`` -> ``acme``), and so are a submodule's
    (``<super>/.git/modules/<name>``). A hidden bare repository inside the project
    folder (``acme/.bare``, the bare-plus-worktrees layout) is named for that folder."""
    gitdir = _gitdir_of(top)
    if gitdir is None or (top / ".git").is_dir():
        return top
    try:
        common = (gitdir / "commondir").read_text(encoding="utf-8").strip()
    except OSError:
        return top
    common_dir = Path(common) if Path(common).is_absolute() else gitdir / common
    try:
        common_dir = common_dir.resolve()
    except OSError:
        return top
    if common_dir.name.startswith("."):  # .git, or a hidden bare repo such as .bare
        return common_dir.parent
    if common_dir.name.endswith(".git") and len(common_dir.name) > 4:
        return common_dir.parent / common_dir.name[:-4]
    return common_dir


def _home() -> Path | None:
    try:
        return Path.home().resolve()
    except (OSError, RuntimeError, KeyError):
        return None


def _toplevel(start: Path, *, projects_only: bool = True) -> Path | None:
    """The nearest directory at or above ``start`` holding a ``.git`` entry. With
    ``projects_only``, a repository at the home directory itself (dotfiles) is no project
    for the folders under it: it counts only for a session in the home directory."""
    home = _home() if projects_only else None
    for d in (start, *start.parents):
        try:
            if (d / ".git").exists():
                return None if (d == home and start != home) else d
        except OSError:
            return None
    return None


def resolve(cwd=None) -> tuple[str | None, Path | None]:
    """``(label, root)`` for ``cwd`` (default: the process cwd).

    ``root`` is the directory the label belongs to: the marker's folder, the git
    toplevel (of this worktree), or ``cwd`` itself; None when ``cwd`` does not exist
    here. A folder that is gone (a removed worktree, a deleted subfolder, which a
    catch-up sweep still ships transcripts from) takes the label of the repository its
    nearest surviving parent is in; anything else that is not here (a transcript from
    another machine) is just the normalised basename of the string."""
    raw = cwd if cwd else os.getcwd()
    try:
        start = Path(raw).resolve()
        exists = start.is_dir()
    except (OSError, RuntimeError, ValueError):
        exists = False
    if not exists:
        return _gone_label(raw), None
    top = _toplevel(start)
    stop = top if top is not None else start
    for d in (start, *start.parents):
        marker = d / MARKER
        if marker.is_file():
            label = _read_marker(marker)
            if label:
                return label, d
        if d == stop:
            break
    if top is None:
        return label_from_path(start), start
    main = _main_checkout(top)
    if main != top and (main / MARKER).is_file():
        label = _read_marker(main / MARKER)
        if label:
            return label, top
    return label_from_path(main), top


def _gone_label(raw) -> str | None:
    """The label for a folder that is not on this disk (see resolve)."""
    try:
        path = Path(raw)
        if path.is_absolute():
            for parent in path.parents:
                if parent.is_dir():
                    top = _toplevel(parent.resolve())
                    # A dotfiles repository at ~ is not the project of a folder under it,
                    # gone or not (see _toplevel).
                    if top is not None and top != _home():
                        return resolve(parent)[0]
                    break
    except (OSError, RuntimeError, ValueError):
        pass
    return label_from_path(raw)


def project_label(cwd=None) -> str | None:
    """Just the label for ``cwd``. Never raises: a hook must not fail over a label."""
    try:
        return resolve(cwd)[0]
    except Exception:
        try:
            return label_from_path(cwd or os.getcwd())
        except Exception:
            return None
