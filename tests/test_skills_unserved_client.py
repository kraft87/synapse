"""Skills a folder is no longer served, on the client: one removal path for every reason.

The server's /skills/list now answers, besides the active skills of the scope asked
about ("skills") and the retired ones ("retired"):

  * "withdrawn" — of the names the client says it holds ("held", the skill folders on
    disk), the active skills it no longer serves in this scope: moved to another scope
    (`synapse skill scope`), or filtered out by trust (a restricted device is served only
    its allowlisted projects' skills). Each entry carries the whole-skill hash of the
    current version ("content_hash") and of recent past versions ("content_hashes"), and
    a "scope" ("" = another scope's skill, as a restricted device is told).
  * "publish" — False when this device may not write the scope at all.

The sync engine (plugin/scripts/skills_sync.py) settles a withdrawn copy through the same
function as a retired one (_settle_unserved):

  * still exactly a version Synapse delivered -> deleted: the entry's hashes, or (only in
    the skill's own scope's folder) this folder's delivered ledger
  * anything else -> kept and NEVER pushed back (a retired copy edited after the
    retirement is still pushed, which reactivates it)
  * never deleted, whatever its content: a symlinked folder, a folder git tracks, or one
    where git can't say (git missing, git refusing the repository)

With "publish": false the engine keeps local skills local: no push for a local-only skill
or a newer local edit, while pulls still happen. A push the server refuses ("retired",
"scoped", "forbidden") is neither counted nor remembered.

Project folders: every skill folder a pass pulls with git_exclude=True is listed in the
repository's .git/info/exclude (synapse_project.git_exclude), not in .gitignore; the
Claude hook's project pass and the Codex plugin's new project pass (<root>/.agents/skills,
from the session's --cwd) both run that way. `synapse skill scope NAME SCOPE` moves a skill.

Stdlib-only scripts loaded by path against an in-memory /skills/* server, and real git
repositories (git init, commit, git worktree add) wherever git decides the outcome.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import urllib.error
from datetime import datetime
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "plugin" / "scripts"
_CODEX = _REPO / "plugin-codex"
_MODULES = (
    "config",
    "enroll",
    "synapse_login",
    "harness",
    "synapse_cli",
    "synapse_home",
    "synapse_project",
    "synapse_filelock",
    "common",
)

_SETTINGS = (
    "SYNAPSE_URL",
    "SYNAPSE_INGEST_URL",
    "SYNAPSE_INGEST_TOKEN",
    "SYNAPSE_MCP_URL",
    "SYNAPSE_MACHINE_ROLE",
    "SYNAPSE_SKILLS_SYNC",
    "SYNAPSE_CONFIG_SYNC",
    "SYNAPSE_CONFIG_PATHS",
    "SYNAPSE_SURFACE",
)
_ENV_VARS = (
    *_SETTINGS,
    *(f"CLAUDE_PLUGIN_OPTION_{k}" for k in _SETTINGS),
    "SYNAPSE_CODEX_SKILLS_DIR",
    "CLAUDE_PROJECT_DIR",
)

_V1 = "---\ndescription: rotate the service keys\n---\nstep one\n"
_V2 = "---\ndescription: rotate the service keys\n---\nstep one\nstep two\n"
_LOCAL = "---\ndescription: rotate the service keys\n---\nstep one, my way\n"
_OTHER = "---\ndescription: tidy the build cache\n---\nclean\n"
_REF = [("ref/notes.md", b"reference notes\n")]

_V1_AT = "2026-09-18T12:00:00+00:00"
_V2_AT = "2026-09-20T12:00:00+00:00"
_RETIRED_AT = "2026-09-25T12:00:00+00:00"
_RETIRED_TS = datetime.fromisoformat(_RETIRED_AT).timestamp()
_BEFORE = datetime.fromisoformat(_V1_AT).timestamp() - 3600
_AFTER = _RETIRED_TS + 3600
_FAR_LATER = _RETIRED_TS + 86400 * 365

_URL = "https://synapse.example.net"
_NO_KEY = object()


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _forget() -> None:
    for mod in _MODULES:
        sys.modules.pop(mod, None)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Scratch HOME, Synapse home, Claude config/data/skills dirs; no Synapse env; a cwd
    with no project; sys.path and sys.argv restored afterwards; no cached modules on
    either side of the test."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("SYNAPSE_HOME", str(tmp_path / "synapse-home"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("SYNAPSE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CLAUDE_SKILLS_DIR", str(tmp_path / "skills"))
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    (tmp_path / "home").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", [*sys.path])
    monkeypatch.setattr(sys, "argv", [*sys.argv])
    _forget()
    yield
    _forget()


def _load(name: str, path: Path) -> ModuleType:
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def engine() -> ModuleType:
    return _load("synapse_unserved_engine_t", _SCRIPTS / "skills_sync.py")


@pytest.fixture()
def project() -> ModuleType:
    return _load("synapse_project_unserved_t", _SCRIPTS / "synapse_project.py")


@pytest.fixture()
def codex() -> ModuleType:
    return _load("synapse_codex_unserved_t", _CODEX / "scripts" / "skills_sync.py")


@pytest.fixture()
def target(tmp_path) -> Path:
    return tmp_path / "skills"


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _hash(engine, body: str, files=()) -> str:
    return engine._whole_hash(body, [(rel, _sha(content)) for rel, content in files])


class FakeServer:
    """In-memory /skills/* routes. Skills carry their scope and the listing serves only
    the scope asked about (case-insensitively, as the server does). ``withdrawable`` is
    what the server would report under "withdrawn": answered only for names the client
    says it holds and the listing did not serve, and only when "held" was sent at all."""

    def __init__(self):
        self.skills: dict[str, dict] = {}
        self.retired: list[dict] | None = []
        self.withdrawable: dict[str, dict] = {}
        self.send_withdrawn = True
        self.publish: bool | None = None  # None: no "publish" key, as an older server
        self.publish_reply: dict = {"status": "ok"}
        self.calls: list[tuple[str, dict]] = []
        self.on_call = None

    def put(self, name, body, files=(), *, at=_V1_AT, scope="global") -> None:
        self.skills[name] = {
            "scope": scope,
            "body": body,
            "files": [
                {
                    "path": rel,
                    "content_b64": base64.b64encode(content).decode(),
                    "sha256": _sha(content),
                    "size": len(content),
                    "is_executable": False,
                }
                for rel, content in files
            ],
            "content_modified_at": at,
        }

    def withdraw(self, name, content_hash, *, scope="global", past=()) -> None:
        """Stop serving ``name`` here (moved away, or filtered by trust)."""
        self.skills.pop(name, None)
        entry = {"name": name, "content_hash": content_hash, "content_hashes": [content_hash]}
        entry["content_hashes"] += [h for h in past if h != content_hash]
        if scope is not _NO_KEY:
            entry["scope"] = scope
        self.withdrawable[name] = entry

    def retire(self, name, content_hash, *, past=(), **entry) -> None:
        self.skills.pop(name, None)
        self.retired = [
            *(self.retired or []),
            {
                "name": name,
                "scope": "global",
                "content_hash": content_hash,
                "content_hashes": [content_hash, *past],
                "retired_at": _RETIRED_AT,
            }
            | entry,
        ]

    def listed(self) -> list[dict]:
        return [p for path, p in self.calls if path == "/skills/list"]

    def published(self) -> list[dict]:
        return [p for path, p in self.calls if path == "/skills/publish"]

    def post_json(self, path, payload, timeout=30.0):
        self.calls.append((path, payload))
        if self.on_call:
            self.on_call(path, payload)
        if path == "/skills/list":
            asked = payload["scope"].lower()
            served = {n: s for n, s in sorted(self.skills.items()) if s["scope"].lower() == asked}
            reply: dict = {
                "skills": [
                    {
                        "name": name,
                        "scope": s["scope"],
                        "body": s["body"],
                        "files": [{"path": f["path"], "sha256": f["sha256"]} for f in s["files"]],
                        "content_modified_at": s["content_modified_at"],
                    }
                    for name, s in served.items()
                ]
            }
            if self.publish is not None:
                reply["publish"] = self.publish
            if self.retired is not None:
                reply["retired"] = self.retired
            if self.send_withdrawn and "held" in payload:
                reply["withdrawn"] = [
                    self.withdrawable[n]
                    for n in payload["held"]
                    if n not in served and n in self.withdrawable
                ]
            return reply
        if path == "/skills/fetch":
            s = self.skills.get(payload["name"])
            return {"found": True, **s} if s else {"found": False}
        if path == "/skills/publish":
            if self.publish_reply.get("status") == "ok":
                self.skills[payload["name"]] = {
                    "scope": payload["scope"],
                    "body": payload["body"],
                    "files": payload["files"],
                    "content_modified_at": payload["content_modified_at"],
                }
            return self.publish_reply
        if path == "/skills/overwrite":
            return {"status": "ok"}
        raise AssertionError(f"unexpected route {path}")


def _write_skill(root: Path, name: str, body: str, files=(), *, mtime: float = _BEFORE) -> Path:
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(body, encoding="utf-8", newline="\n")
    for rel, content in files:
        (folder / rel).parent.mkdir(parents=True, exist_ok=True)
        (folder / rel).write_bytes(content)
    for fp in folder.rglob("*"):
        if fp.is_file():
            os.utime(fp, (mtime, mtime))
    return folder


def _ledger(engine, folder: Path) -> dict:
    path = engine._ledger_path(folder)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# --- real git ---------------------------------------------------------------

_GIT = (
    "git",
    "-c",
    "user.name=Test User",
    "-c",
    "user.email=test@example.invalid",
    "-c",
    "init.defaultBranch=main",
    "-c",
    "commit.gpgsign=false",
)


@pytest.fixture()
def git():
    if shutil.which("git") is None:
        pytest.skip("git is not installed here")

    def run(cwd: Path, *args: str) -> str:
        return subprocess.run(
            [*_GIT, "-C", str(cwd), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout

    return run


def _init_repo(git, path: Path, *, commit: bool = True) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q")
    if commit:
        (path / "README.md").write_text("a repository\n", encoding="utf-8")
        git(path, "add", "README.md")
        git(path, "commit", "-q", "-m", "init")
    return path


def _exclude(repo: Path) -> str:
    path = repo / ".git" / "info" / "exclude"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _untracked(git, repo: Path) -> list[str]:
    """What `git status` would show as new: the check that an exclude line really works."""
    out = git(repo, "status", "--porcelain", "--untracked-files=all", "-z")
    return sorted(e[3:] for e in out.split("\0") if e.startswith("?? "))


# ---------------------------------------------------------------------------
# What the client sends
# ---------------------------------------------------------------------------


def test_the_listing_request_names_every_skill_folder_held(engine, target):
    _write_skill(target, "zeta", _V1)
    _write_skill(target, "alpha", _V2)
    (target / "notes-only").mkdir(parents=True)
    (target / "notes-only" / "README.md").write_text("not a skill\n", encoding="utf-8")
    (target / "loose.md").write_text("not a skill either\n", encoding="utf-8")
    server = FakeServer()

    engine._sync("global", target, client=server)

    assert server.listed() == [{"scope": "global", "held": ["alpha", "zeta"]}]


def test_an_empty_folder_holds_nothing(engine, target):
    server = FakeServer()

    assert engine._sync("project:widget", target, client=server) == (0, 0, 0)

    assert server.listed() == [{"scope": "project:widget", "held": []}]


def test_held_lists_the_folders_before_this_pass_pulls_anything(engine, target):
    _write_skill(target, "mine", _LOCAL)
    server = FakeServer()
    server.put("theirs", _V1)

    assert engine._sync("global", target, client=server) == (1, 1, 0)
    assert server.listed()[0]["held"] == ["mine"]


# ---------------------------------------------------------------------------
# A withdrawn copy: removed when Synapse delivered exactly it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mtime", [_BEFORE, _FAR_LATER], ids=["older", "renewed-mtime"])
def test_a_withdrawn_copy_equal_to_the_current_hash_is_removed(engine, target, mtime):
    folder = _write_skill(target, "alpha", _V1, _REF, mtime=mtime)
    server = FakeServer()
    server.withdraw("alpha", _hash(engine, _V1, _REF), scope="project:widget")

    assert engine._sync("global", target, client=server) == (0, 0, 1)

    assert not folder.exists()
    assert server.published() == []
    # Kept, so a restored copy is recognised and removed again.
    assert _ledger(engine, target)["alpha"] == [_hash(engine, _V1, _REF)]


@pytest.mark.parametrize("scope", ["global", "", "project:widget"])
def test_a_withdrawn_copy_equal_to_a_past_version_is_removed_in_any_folder(engine, target, scope):
    """The server vouches for its own recent versions, whichever folder asks."""
    folder = _write_skill(target, "alpha", _V1, _REF, mtime=_FAR_LATER)
    server = FakeServer()
    server.withdraw("alpha", _hash(engine, _V2, _REF), scope=scope, past=[_hash(engine, _V1, _REF)])
    assert _ledger(engine, target) == {}  # nothing remembered: only the entry vouches

    assert engine._sync("global", target, client=server) == (0, 0, 1)

    assert not folder.exists() and server.published() == []


@pytest.mark.parametrize(
    ("folder_scope", "entry_scope"),
    [
        pytest.param("global", "global", id="own-scope"),
        pytest.param("global", _NO_KEY, id="scope-key-missing"),
        pytest.param("global", None, id="scope-null"),
        pytest.param("project:Widget", "project:widget", id="own-scope-other-case"),
        pytest.param("global", "", id="another-scope-hidden"),
        pytest.param("global", "project:widget", id="another-scope-named"),
        pytest.param("project:widget", "project:gadget", id="another-project"),
    ],
)
def test_a_copy_a_pull_wrote_here_is_removed_whatever_the_scope(
    engine, target, folder_scope, entry_scope
):
    """This folder pulled v1; the server moved on to v2 (a bundled-file change leaves no
    history), then stopped serving it here. A pull wrote v1 here, so it is the server's
    content wherever the row lives now: a skill that drops out of a folder's served list
    leaves it."""
    server = FakeServer()
    server.put("alpha", _V1, _REF, scope=folder_scope)
    assert engine._sync(folder_scope, target, client=server) == (1, 0, 0)

    server.withdraw("alpha", _hash(engine, _V2, _REF), scope=entry_scope)
    assert engine._sync(folder_scope, target, client=server) == (0, 0, 1)

    assert not (target / "alpha").exists()
    assert server.published() == []


def test_a_copy_whose_bundled_file_changed_on_the_server_still_leaves_after_a_move(engine, target):
    """Pulled with run.sh v1; another machine changed only run.sh (no history: the body is
    the same); then the skill moved to another scope. The withdrawn entry vouches for
    today's files only; the pull that wrote this copy vouches for it."""
    server = FakeServer()
    server.put("alpha", _V1, [("run.sh", b"v1\n")], scope="project:widget")
    assert engine._sync("project:widget", target, client=server) == (1, 0, 0)
    server.put("alpha", _V1, [("run.sh", b"v2\n")], at=_V2_AT, scope="project:widget")
    server.withdraw("alpha", _hash(engine, _V1, [("run.sh", b"v2\n")]), scope="global")

    assert engine._sync("project:widget", target, client=server) == (0, 0, 1)
    assert not (target / "alpha").exists()


def _delivered_ledger(engine, target: Path, ledger: dict) -> None:
    """Seed the folder's "delivered" ledger (pulls, pushes, identical sightings; an older
    client also recorded pushes a newer server refused there)."""
    path = engine._ledger_path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger), encoding="utf-8")


@pytest.mark.parametrize(
    ("entry_scope", "retired", "removed"),
    [
        pytest.param("global", True, True, id="retired-own-scope"),
        pytest.param("project:widget", True, False, id="retired-another-scope"),
        pytest.param("global", False, False, id="withdrawn"),
        pytest.param("", False, False, id="withdrawn-hidden-scope"),
    ],
)
def test_the_delivered_ledger_vouches_only_for_a_retirement_in_its_own_scope(
    engine, target, entry_scope, retired, removed
):
    """A version this folder only pushed (or an older client recorded) may be the
    folder's own skill: it counts for a retirement here, never for a withdrawal."""
    folder = _write_skill(target, "alpha", _V1, _REF, mtime=_BEFORE)
    _delivered_ledger(engine, target, {"alpha": [_hash(engine, _V1, _REF)]})
    server = FakeServer()
    if retired:
        server.retire("alpha", _hash(engine, _V2, _REF), scope=entry_scope)
    else:
        server.withdraw("alpha", _hash(engine, _V2, _REF), scope=entry_scope)

    assert engine._sync("global", target, client=server) == (0, 0, int(removed))
    assert folder.exists() is not removed
    assert server.published() == []


def test_a_pushed_copy_is_not_recorded_as_pulled(engine, target):
    folder = _write_skill(target, "alpha", _LOCAL)
    server = FakeServer()
    server.publish = True
    assert engine._sync("project:widget", target, client=server) == (0, 1, 0)
    server.withdraw("alpha", _hash(engine, _V2), scope="global")

    assert engine._sync("project:widget", target, client=server) == (0, 0, 0)
    assert folder.is_dir()
    assert "alpha" not in engine._load_ledger(target, "pulled")


# ---------------------------------------------------------------------------
# A withdrawn copy: everything else is kept, and never pushed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("publish", [None, True], ids=["no-publish-key", "publish-true"])
@pytest.mark.parametrize("mtime", [_BEFORE, _AFTER, _FAR_LATER])
def test_an_edited_withdrawn_copy_is_kept_and_never_pushed(engine, target, publish, mtime):
    """Unlike a retirement, a withdrawal has no time to be newer than: the skill lives on
    elsewhere, and a push would be refused (another scope's skill, or a scope this device
    may not write)."""
    folder = _write_skill(target, "alpha", _LOCAL, _REF, mtime=mtime)
    server = FakeServer()
    server.publish = publish
    server.withdraw("alpha", _hash(engine, _V1, _REF), past=[_hash(engine, _V2, _REF)])

    for _ in range(2):
        assert engine._sync("global", target, client=server) == (0, 0, 0)

    assert (folder / "SKILL.md").read_text(encoding="utf-8") == _LOCAL
    assert (folder / "ref" / "notes.md").read_bytes() == b"reference notes\n"
    assert server.published() == []
    assert "alpha" not in _ledger(engine, target)


def test_a_withdrawn_copy_with_an_extra_local_file_is_kept(engine, target):
    folder = _write_skill(target, "alpha", _V1, [*_REF, ("mine.md", b"my notes\n")])
    server = FakeServer()
    server.withdraw("alpha", _hash(engine, _V1, _REF))

    assert engine._sync("global", target, client=server) == (0, 0, 0)

    assert (folder / "mine.md").read_bytes() == b"my notes\n"
    assert server.published() == []


def test_a_symlinked_withdrawn_folder_is_never_deleted(engine, target, tmp_path):
    real = _write_skill(tmp_path / "dotfiles", "alpha", _V1, _REF)
    target.mkdir()
    link = target / "alpha"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable here")
    server = FakeServer()
    server.withdraw("alpha", _hash(engine, _V1, _REF))

    assert engine._sync("global", target, client=server) == (0, 0, 0)

    assert server.listed()[0]["held"] == ["alpha"]  # it was considered, and matched
    assert link.is_symlink()
    assert (real / "SKILL.md").read_text(encoding="utf-8") == _V1
    assert server.published() == []


def test_a_withdrawn_name_not_on_disk_is_neither_pulled_nor_published(engine, target):
    server = FakeServer()
    server.withdrawable["ghost"] = {"name": "ghost", "scope": "", "content_hash": "0" * 64}
    server.send_withdrawn = True

    assert engine._sync("global", target, client=server) == (0, 0, 0)

    assert [path for path, _ in server.calls] == ["/skills/list"]
    assert not (target / "ghost").exists()


@pytest.mark.parametrize(
    "withdrawn",
    [
        pytest.param([None, "alpha", 3, {"scope": "global"}, {"name": ""}], id="garbage"),
        pytest.param(None, id="null"),
        pytest.param({}, id="wrong-type-empty"),
    ],
)
def test_malformed_withdrawn_entries_are_ignored(engine, target, withdrawn):
    folder = _write_skill(target, "alpha", _V1)
    server = FakeServer()
    server.send_withdrawn = False
    real = server.post_json

    def post_json(path, payload, timeout=30.0):
        reply = real(path, payload, timeout)
        if path == "/skills/list":
            reply["withdrawn"] = withdrawn
        return reply

    server.post_json = post_json

    assert engine._sync("global", target, client=server) == (0, 1, 0)
    assert folder.is_dir()


def test_an_older_server_without_withdrawn_publishes_a_local_only_skill_as_before(engine, target):
    _write_skill(target, "alpha", _V1)
    server = FakeServer()
    server.send_withdrawn = False

    assert engine._sync("global", target, client=server) == (0, 1, 0)

    assert [p["name"] for p in server.published()] == ["alpha"]


# ---------------------------------------------------------------------------
# Git decides: a folder git tracks is the repository's
# ---------------------------------------------------------------------------


def test_a_withdrawn_copy_git_tracks_is_kept(engine, git, tmp_path):
    repo = _init_repo(git, tmp_path / "widget")
    skills = repo / ".claude" / "skills"
    folder = _write_skill(skills, "alpha", _V1, _REF)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "add the skill")
    server = FakeServer()
    server.withdraw("alpha", _hash(engine, _V1, _REF), scope="project:widget")

    assert engine._git_tracked(folder) is True
    assert engine._sync("project:widget", skills, client=server) == (0, 0, 0)

    assert (folder / "SKILL.md").read_text(encoding="utf-8") == _V1
    assert git(repo, "status", "--porcelain") == ""
    assert server.published() == []


def test_a_withdrawn_copy_git_only_has_staged_is_kept(engine, git, tmp_path):
    repo = _init_repo(git, tmp_path / "widget")
    skills = repo / ".claude" / "skills"
    folder = _write_skill(skills, "alpha", _V1)
    git(repo, "add", "-A")
    server = FakeServer()
    server.withdraw("alpha", _hash(engine, _V1), scope="project:widget")

    assert engine._sync("project:widget", skills, client=server) == (0, 0, 0)
    assert folder.is_dir()


def test_an_untracked_withdrawn_copy_in_a_repository_is_removed(engine, git, tmp_path):
    repo = _init_repo(git, tmp_path / "widget")
    skills = repo / ".claude" / "skills"
    folder = _write_skill(skills, "alpha", _V1, _REF)
    server = FakeServer()
    server.withdraw("alpha", _hash(engine, _V1, _REF), scope="project:widget")

    assert engine._git_tracked(folder) is False
    assert engine._sync("project:widget", skills, client=server) == (0, 0, 1)

    assert not folder.exists()
    assert git(repo, "status", "--porcelain") == ""


def test_an_ignored_withdrawn_copy_in_a_repository_is_removed(engine, git, tmp_path):
    repo = _init_repo(git, tmp_path / "widget")
    (repo / ".gitignore").write_text(".claude/\n", encoding="utf-8")
    git(repo, "add", ".gitignore")
    git(repo, "commit", "-q", "-m", "ignore")
    skills = repo / ".claude" / "skills"
    folder = _write_skill(skills, "alpha", _V1)
    server = FakeServer()
    server.withdraw("alpha", _hash(engine, _V1), scope="project:widget")

    assert engine._sync("project:widget", skills, client=server) == (0, 0, 1)
    assert not folder.exists()


def test_a_tracked_retired_copy_is_kept_too(engine, git, tmp_path):
    """One removal path: the git guard covers a retirement the same way."""
    repo = _init_repo(git, tmp_path / "widget")
    skills = repo / ".claude" / "skills"
    folder = _write_skill(skills, "alpha", _V1)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "add the skill")
    server = FakeServer()
    server.retire("alpha", _hash(engine, _V1), scope="project:widget")

    assert engine._sync("project:widget", skills, client=server) == (0, 0, 0)
    assert folder.is_dir() and server.published() == []


def test_git_tracked_is_true_for_a_skill_folder_that_is_its_own_repository(engine, git, tmp_path):
    folder = _init_repo(git, tmp_path / "skills" / "alpha", commit=False)
    _write_skill(tmp_path / "skills", "alpha", _V1)
    git(folder, "add", "SKILL.md")
    git(folder, "commit", "-q", "-m", "skill")

    assert engine._git_tracked(folder) is True


def test_git_tracked_is_false_outside_any_repository_without_running_git(
    engine, tmp_path, monkeypatch
):
    folder = _write_skill(tmp_path / "plain", "alpha", _V1)

    def no_git(*a, **kw):
        raise AssertionError("git must not run outside a repository")

    monkeypatch.setattr(engine.subprocess, "run", no_git)
    assert engine._git_tracked(folder) is False


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(FileNotFoundError(2, "No such file or directory: 'git'"), id="git-missing"),
        pytest.param(PermissionError(13, "Permission denied: 'git'"), id="git-not-runnable"),
        pytest.param(subprocess.TimeoutExpired(["git"], 10), id="git-hangs"),
    ],
)
def test_git_tracked_is_none_when_git_cannot_answer(engine, git, tmp_path, monkeypatch, error):
    repo = _init_repo(git, tmp_path / "widget")
    folder = _write_skill(repo / ".claude" / "skills", "alpha", _V1)

    def broken(*a, **kw):
        raise error

    monkeypatch.setattr(engine.subprocess, "run", broken)
    assert engine._git_tracked(folder) is None


def test_git_tracked_is_none_when_git_refuses_the_repository(engine, git, tmp_path):
    """A .git file pointing nowhere (a worktree whose main checkout was deleted): git
    exits non-zero, so whether the folder is the repository's can't be told."""
    top = tmp_path / "stale-worktree"
    top.mkdir()
    (top / ".git").write_text(f"gitdir: {tmp_path / 'gone' / '.git' / 'worktrees' / 'x'}\n")
    folder = _write_skill(top / ".claude" / "skills", "alpha", _V1)

    assert engine._git_tracked(folder) is None


def test_a_delivered_copy_is_kept_when_git_is_missing_inside_a_repository(
    engine, git, tmp_path, monkeypatch
):
    repo = _init_repo(git, tmp_path / "widget")
    skills = repo / ".claude" / "skills"
    folder = _write_skill(skills, "alpha", _V1)
    server = FakeServer()
    server.withdraw("alpha", _hash(engine, _V1), scope="project:widget")

    def no_git(*a, **kw):
        raise FileNotFoundError(2, "No such file or directory: 'git'")

    monkeypatch.setattr(engine.subprocess, "run", no_git)
    assert engine._sync("project:widget", skills, client=server) == (0, 0, 0)

    assert folder.is_dir()


def test_a_delivered_copy_is_kept_when_git_refuses_the_repository(engine, tmp_path):
    top = tmp_path / "stale-worktree"
    top.mkdir()
    (top / ".git").write_text(f"gitdir: {tmp_path / 'gone' / '.git' / 'worktrees' / 'x'}\n")
    skills = top / ".claude" / "skills"
    folder = _write_skill(skills, "alpha", _V1)
    server = FakeServer()
    server.withdraw("alpha", _hash(engine, _V1))

    assert engine._sync("global", skills, client=server) == (0, 0, 0)
    assert folder.is_dir()


def test_a_delivered_copy_outside_any_repository_is_removed_without_git(
    engine, target, monkeypatch
):
    folder = _write_skill(target, "alpha", _V1)
    server = FakeServer()
    server.withdraw("alpha", _hash(engine, _V1))

    def no_git(*a, **kw):
        raise FileNotFoundError(2, "No such file or directory: 'git'")

    monkeypatch.setattr(engine.subprocess, "run", no_git)
    assert engine._sync("global", target, client=server) == (0, 0, 1)
    assert not folder.exists()


# ---------------------------------------------------------------------------
# The retirement path, through the same function
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mtime", [_AFTER, _FAR_LATER])
def test_a_retired_copy_edited_after_the_retirement_is_still_pushed(engine, target, mtime):
    folder = _write_skill(target, "alpha", _LOCAL, _REF, mtime=mtime)
    server = FakeServer()
    server.retire("alpha", _hash(engine, _V2, _REF), past=[_hash(engine, _V1, _REF)])

    assert engine._sync("global", target, client=server) == (0, 1, 0)

    (push,) = server.published()
    assert (push["name"], push["scope"], push["body"]) == ("alpha", "global", _LOCAL)
    assert folder.is_dir()
    assert _ledger(engine, target)["alpha"] == [_hash(engine, _LOCAL, _REF)]


@pytest.mark.parametrize("scope", ["global", ""], ids=["own-scope", "another-scope"])
def test_a_retired_copy_of_a_past_version_is_removed_through_content_hashes(engine, target, scope):
    """Never pulled through this folder (no ledger), renewed mtime: the past version the
    server vouches for is still a copy Synapse delivered, so it goes, never pushed."""
    folder = _write_skill(target, "alpha", _V1, _REF, mtime=_FAR_LATER)
    server = FakeServer()
    server.retire("alpha", _hash(engine, _V2, _REF), past=[_hash(engine, _V1, _REF)], scope=scope)

    assert engine._sync("global", target, client=server) == (0, 0, 1)

    assert not folder.exists() and server.published() == []


def test_a_retired_copy_older_than_the_retirement_and_unrecognised_is_kept(engine, target):
    folder = _write_skill(target, "alpha", _LOCAL, mtime=_BEFORE)
    server = FakeServer()
    server.retire("alpha", _hash(engine, _V2), past=[_hash(engine, _V1)])

    assert engine._sync("global", target, client=server) == (0, 0, 0)
    assert folder.is_dir() and server.published() == []


@pytest.mark.parametrize("retired", [True, False], ids=["retired", "withdrawn"])
def test_settle_unserved_is_the_one_removal_path(engine, target, monkeypatch, retired):
    _write_skill(target, "alpha", _V1)
    server = FakeServer()
    if retired:
        server.retire("alpha", _hash(engine, _V1))
    else:
        server.withdraw("alpha", _hash(engine, _V1))
    seen = []
    real = engine._settle_unserved

    def spy(*a, **kw):
        seen.append((a[2], kw["retired"]))
        return real(*a, **kw)

    monkeypatch.setattr(engine, "_settle_unserved", spy)

    assert engine._sync("global", target, client=server) == (0, 0, 1)
    assert seen == [("alpha", retired)]


# ---------------------------------------------------------------------------
# "publish": false
# ---------------------------------------------------------------------------


def _mixed_folder(server: FakeServer, target: Path) -> None:
    server.put("theirs", _OTHER)  # only on the server: pulled
    server.put("shared", _V1, at=_V1_AT)  # newer local edit: would be pushed
    _write_skill(target, "shared", _V2, mtime=_AFTER)
    server.put("stale", _V2, at=_RETIRED_AT)  # newer server edit: pulled
    _write_skill(target, "stale", _V1, mtime=_BEFORE)
    _write_skill(target, "mine", _LOCAL)  # only on disk: would be published


def test_publish_false_keeps_local_skills_local_and_still_pulls(engine, target):
    server = FakeServer()
    server.publish = False
    _mixed_folder(server, target)

    for expected in ((2, 0, 0), (0, 0, 0)):
        assert engine._sync("global", target, client=server) == expected

    assert server.published() == []
    assert (target / "theirs" / "SKILL.md").read_text(encoding="utf-8") == _OTHER
    assert (target / "stale" / "SKILL.md").read_text(encoding="utf-8") == _V2
    # The newer local edit is left exactly as it is: neither pushed nor overwritten.
    assert (target / "shared" / "SKILL.md").read_text(encoding="utf-8") == _V2
    assert (target / "mine" / "SKILL.md").read_text(encoding="utf-8") == _LOCAL
    assert server.skills["shared"]["body"] == _V1
    ledger = _ledger(engine, target)
    assert "mine" not in ledger and "shared" not in ledger
    assert [p["held"] for p in server.listed()] == [
        ["mine", "shared", "stale"],
        ["mine", "shared", "stale", "theirs"],
    ]


@pytest.mark.parametrize("publish", [None, True], ids=["no-publish-key", "publish-true"])
def test_publish_true_or_absent_pushes_as_before(engine, target, publish):
    server = FakeServer()
    server.publish = publish
    _mixed_folder(server, target)

    assert engine._sync("global", target, client=server) == (2, 2, 0)

    assert sorted(p["name"] for p in server.published()) == ["mine", "shared"]


def test_publish_false_does_not_push_an_edited_retired_copy_either(engine, target):
    """A restricted device's global folder holds a skill it pulled while it was trusted;
    the skill is retired (the device is told, it holds the name) and the copy was edited
    after that. This device may not write the scope, so the push can only be refused."""
    folder = _write_skill(target, "alpha", _LOCAL, mtime=_AFTER)
    server = FakeServer()
    server.publish = False
    server.publish_reply = {"status": "forbidden"}
    server.retire("alpha", _hash(engine, _V1))

    assert engine._sync("global", target, client=server) == (0, 0, 0)

    assert folder.is_dir()
    assert server.published() == []


# ---------------------------------------------------------------------------
# A push the server refuses
# ---------------------------------------------------------------------------

_REFUSALS = [
    pytest.param({"status": "retired", "retired_at": _RETIRED_AT}, id="retired"),
    pytest.param({"status": "scoped", "scope": "project:widget"}, id="scoped"),
    pytest.param({"status": "scoped", "scope": ""}, id="scoped-hidden"),
    pytest.param({"status": "forbidden"}, id="forbidden"),
]


@pytest.mark.parametrize("reply", _REFUSALS)
def test_push_returns_false_on_a_refusal(engine, reply):
    server = FakeServer()
    server.publish_reply = reply
    disk = (_V1, [], _AFTER)

    assert engine._push("global", "alpha", disk, client=server) is False
    assert len(server.published()) == 1


@pytest.mark.parametrize(
    "reply", [{"status": "ok"}, {"status": "created"}, {}], ids=["ok", "other", "empty"]
)
def test_push_returns_true_otherwise(engine, reply):
    server = FakeServer()
    server.publish_reply = reply

    assert engine._push("global", "alpha", (_V1, [], _AFTER), client=server) is True


@pytest.mark.parametrize("reply", _REFUSALS)
def test_a_refused_publish_of_a_local_only_skill_is_neither_counted_nor_remembered(
    engine, target, reply
):
    folder = _write_skill(target, "alpha", _V1)
    server = FakeServer()
    server.publish_reply = reply

    assert engine._sync("global", target, client=server) == (0, 0, 0)

    assert "alpha" not in _ledger(engine, target)
    assert folder.is_dir()
    assert "alpha" not in server.skills


@pytest.mark.parametrize("reply", _REFUSALS)
def test_a_refused_push_of_a_newer_local_edit_is_neither_counted_nor_remembered(
    engine, target, reply
):
    server = FakeServer()
    server.put("alpha", _V1)
    assert engine._sync("global", target, client=server) == (1, 0, 0)
    _write_skill(target, "alpha", _V2, mtime=_AFTER)
    server.publish_reply = reply

    assert engine._sync("global", target, client=server) == (0, 0, 0)

    assert _ledger(engine, target)["alpha"] == [_hash(engine, _V1)]
    assert (target / "alpha" / "SKILL.md").read_text(encoding="utf-8") == _V2
    assert server.skills["alpha"]["body"] == _V1


@pytest.mark.parametrize("reply", _REFUSALS)
def test_a_refused_reactivating_push_keeps_the_retired_copy(engine, target, reply):
    folder = _write_skill(target, "alpha", _LOCAL, mtime=_AFTER)
    server = FakeServer()
    server.retire("alpha", _hash(engine, _V1))
    server.publish_reply = reply

    assert engine._sync("global", target, client=server) == (0, 0, 0)

    assert folder.is_dir() and "alpha" not in _ledger(engine, target)


# ---------------------------------------------------------------------------
# git_exclude=True: pulled project skills stay out of the repository
# ---------------------------------------------------------------------------


def test_pulled_project_skills_are_listed_in_info_exclude_once(engine, git, tmp_path):
    repo = _init_repo(git, tmp_path / "widget")
    skills = repo / ".claude" / "skills"
    _write_skill(skills, "mine", _LOCAL)  # the repository's own, pushed: not excluded
    server = FakeServer()
    server.put("alpha", _V1, _REF, scope="project:widget")
    server.put("beta", _OTHER, scope="project:widget")
    original = _exclude(repo)

    assert engine._sync("project:widget", skills, client=server, git_exclude=True) == (2, 1, 0)
    # A second pull of an updated version lists nothing twice.
    server.put("alpha", _V2, _REF, at=_RETIRED_AT, scope="project:widget")
    assert engine._sync("project:widget", skills, client=server, git_exclude=True) == (1, 0, 0)
    assert engine._sync("project:widget", skills, client=server, git_exclude=True) == (0, 0, 0)

    text = _exclude(repo)
    assert text.startswith(original)  # git's own template lines are kept
    added = text[len(original) :].splitlines()
    assert added == [
        "# Skills synced by Synapse (synapse_project.git_exclude): not part of this repo",
        "/.claude/skills/alpha/",
        "/.claude/skills/beta/",
    ]
    assert _untracked(git, repo) == [".claude/skills/mine/SKILL.md"]
    assert not (repo / ".gitignore").exists()


def test_a_newer_server_version_pulled_over_an_existing_folder_is_excluded_too(
    engine, git, tmp_path
):
    """A folder an older client pulled (so never excluded) is listed once a pull rewrites
    it: every pull, not only a first one, keeps the repository clean."""
    repo = _init_repo(git, tmp_path / "widget")
    skills = repo / ".claude" / "skills"
    _write_skill(skills, "alpha", _V1, mtime=_BEFORE)
    server = FakeServer()
    server.put("alpha", _V2, at=_RETIRED_AT, scope="project:widget")
    assert _untracked(git, repo) == [".claude/skills/alpha/SKILL.md"]

    assert engine._sync("project:widget", skills, client=server, git_exclude=True) == (1, 0, 0)

    assert (skills / "alpha" / "SKILL.md").read_text(encoding="utf-8") == _V2
    assert _exclude(repo).splitlines().count("/.claude/skills/alpha/") == 1
    assert _untracked(git, repo) == []


def _commit_skill(git, repo: Path, rel: str, body: str, files=()) -> Path:
    folder = _write_skill(repo / Path(rel).parent, Path(rel).name, body, files)
    git(repo, "add", "--", rel)
    git(repo, "commit", "-q", "-m", f"add {rel}")
    return folder


def _status(git, repo: Path) -> list[str]:
    return sorted(git(repo, "status", "--porcelain", "--untracked-files=all").splitlines())


def test_a_project_pass_never_pulls_over_a_skill_folder_the_repository_tracks(
    engine, git, tmp_path
):
    """A repository that commits one skill name for each tool (.claude/skills/x and
    .agents/skills/x, different files) syncs both folders against one project scope. Each
    pass may publish its copy, but neither ever rewrites the other's tracked files: git,
    not Synapse, distributes a skill the repository tracks."""
    repo = _init_repo(git, tmp_path / "widget")
    claude = _commit_skill(
        git, repo, ".claude/skills/review", _V1, [("claude-only.md", b"claude flavor\n")]
    )
    codex = _commit_skill(
        git, repo, ".agents/skills/review", _OTHER, [("agents-only.md", b"codex flavor\n")]
    )
    server = FakeServer()
    original = _exclude(repo)

    for _round in range(3):
        engine._sync("project:widget", claude.parent, client=server, git_exclude=True)
        engine._sync("project:widget", codex.parent, client=server, git_exclude=True)

    assert _status(git, repo) == []
    assert (claude / "SKILL.md").read_text(encoding="utf-8") == _V1
    assert (codex / "agents-only.md").read_bytes() == b"codex flavor\n"
    assert not (codex / "claude-only.md").exists()
    assert _exclude(repo) == original
    assert len(server.published()) >= 1  # the repository's copy is still published


def test_a_newer_server_version_is_not_pulled_over_a_tracked_project_skill(engine, git, tmp_path):
    repo = _init_repo(git, tmp_path / "widget")
    folder = _commit_skill(git, repo, ".claude/skills/alpha", _V1, [("run.sh", b"repo\n")])
    server = FakeServer()
    server.put("alpha", _V2, [("run.sh", b"server\n")], at=_RETIRED_AT, scope="project:widget")

    assert engine._sync("project:widget", folder.parent, client=server, git_exclude=True) == (
        0,
        0,
        0,
    )
    assert (folder / "run.sh").read_bytes() == b"repo\n"
    assert _status(git, repo) == []
    # The same folder outside a project pass (no git_exclude) is synced as before.
    assert engine._sync("project:widget", folder.parent, client=server) == (1, 0, 0)


def test_when_git_cannot_answer_only_a_delivered_copy_is_pulled_over(
    engine, git, tmp_path, monkeypatch
):
    repo = _init_repo(git, tmp_path / "widget")
    skills = repo / ".claude" / "skills"
    server = FakeServer()
    server.put("alpha", _V1, scope="project:widget")
    assert engine._sync("project:widget", skills, client=server, git_exclude=True) == (1, 0, 0)
    _write_skill(skills, "mine", _LOCAL)  # a copy Synapse never delivered here
    server.put("mine", _OTHER, at=_RETIRED_AT, scope="project:widget")
    server.put("alpha", _V2, at=_RETIRED_AT, scope="project:widget")
    server.put("fresh", _V1, scope="project:widget")
    monkeypatch.setattr(engine, "_git_tracked", lambda folder: None)

    assert engine._sync("project:widget", skills, client=server, git_exclude=True) == (2, 0, 0)

    assert (skills / "alpha" / "SKILL.md").read_text(encoding="utf-8") == _V2
    assert (skills / "fresh" / "SKILL.md").read_text(encoding="utf-8") == _V1
    assert (skills / "mine" / "SKILL.md").read_text(encoding="utf-8") == _LOCAL


def test_a_pull_into_a_skill_folder_the_user_created_untracked_still_lists_it(
    engine, git, tmp_path
):
    """Untracked, the folder is Synapse's copy once a pull rewrites it (a committed one is
    the repository's and is never pulled over): listed like any other pulled folder."""
    repo = _init_repo(git, tmp_path / "widget")
    skills = repo / ".claude" / "skills"
    _write_skill(skills, "alpha", _LOCAL)
    server = FakeServer()
    server.put("alpha", _V2, at=_RETIRED_AT, scope="project:widget")

    assert engine._sync("project:widget", skills, client=server, git_exclude=True) == (1, 0, 0)
    assert "/.claude/skills/alpha/" in _exclude(repo).splitlines()


class _Killed(BaseException):
    """A session hook's timeout killing the sync mid-pull: nothing after it runs."""


def _kill_at(monkeypatch, obj, attr):
    def killed(*_a, **_k):
        raise _Killed

    monkeypatch.setattr(obj, attr, killed)


@pytest.mark.parametrize(
    "kill",
    ["before-bundled-files", "before-skill-md"],
)
def test_a_pull_cut_short_is_pulled_again_never_pushed_over_the_server(
    engine, target, monkeypatch, kill
):
    """Killed part-way through a pull (Codex's session hook kills the sync after its time
    budget), the next session finishes the pull. Before, SKILL.md was written first with a
    fresh mtime, so the half-pulled copy looked like a newer local edit and its stale
    bundled files were pushed over the server's, for every machine to pull."""
    server = FakeServer()
    server.put("alpha", _V1, [("run.sh", b"OLD script\n"), ("gone.md", b"dropped\n")])
    assert engine._sync("global", target, client=server) == (1, 0, 0)
    server.put("alpha", _V2, [("run.sh", b"NEW script, fixed\n")], at=_V2_AT)

    with monkeypatch.context() as m:
        if kill == "before-bundled-files":
            _kill_at(m, Path, "write_bytes")
        else:
            _kill_at(m, engine.os, "replace")
        with pytest.raises(_Killed):
            engine._sync("global", target, client=server)

    assert engine._sync("global", target, client=server) == (1, 0, 0)
    assert server.published() == []
    folder = target / "alpha"
    assert (folder / "SKILL.md").read_text(encoding="utf-8") == _V2
    assert sorted(p.name for p in folder.iterdir()) == ["SKILL.md", "run.sh"]
    assert (folder / "run.sh").read_bytes() == b"NEW script, fixed\n"
    assert engine._sync("global", target, client=server) == (0, 0, 0)


def test_a_pull_leftover_temp_file_is_never_part_of_the_skill(engine, target):
    folder = _write_skill(target, "alpha", _V1)
    (folder / engine._PULL_TMP).write_text("half a body", encoding="utf-8")
    server = FakeServer()
    server.put("alpha", _V1)

    assert engine._sync("global", target, client=server) == (0, 0, 0)
    assert server.published() == []


def test_a_global_pass_inside_a_repository_leaves_info_exclude_alone(engine, git, tmp_path):
    repo = _init_repo(git, tmp_path / "dotfiles")
    skills = repo / "claude" / "skills"
    server = FakeServer()
    server.put("alpha", _V1)
    original = _exclude(repo)

    assert engine._sync("global", skills, client=server) == (1, 0, 0)

    assert _exclude(repo) == original
    assert _untracked(git, repo) == ["claude/skills/alpha/SKILL.md"]


def test_exclude_lines_escape_gitignore_wildcards_and_match_only_their_folder(
    engine, git, tmp_path
):
    repo = _init_repo(git, tmp_path / "widget")
    skills = repo / ".claude" / "skills"
    names = ["deploy[v2]", "a*b", "what?"]
    server = FakeServer()
    for name in names:
        server.put(name, _V1, scope="project:widget")
    # Folders the patterns must NOT match if they were left unescaped.
    for decoy in ("axb", "deployv", "whatx"):
        (skills / decoy).mkdir(parents=True)
        (skills / decoy / "notes.txt").write_text("not a skill\n", encoding="utf-8")

    assert engine._sync("project:widget", skills, client=server, git_exclude=True) == (3, 0, 0)

    lines = _exclude(repo).splitlines()
    for line in (
        "/.claude/skills/deploy\\[v2\\]/",
        "/.claude/skills/a\\*b/",
        "/.claude/skills/what\\?/",
    ):
        assert lines.count(line) == 1
    assert _untracked(git, repo) == [
        ".claude/skills/axb/notes.txt",
        ".claude/skills/deployv/notes.txt",
        ".claude/skills/whatx/notes.txt",
    ]


def test_a_linked_worktrees_exclude_goes_to_the_main_repository(engine, git, tmp_path):
    main = _init_repo(git, tmp_path / "widget")
    worktree = tmp_path / "widget-feature"
    git(main, "worktree", "add", "-q", "-b", "feature", str(worktree))
    skills = worktree / ".claude" / "skills"
    server = FakeServer()
    server.put("alpha", _V1, scope="project:widget")

    assert engine._sync("project:widget", skills, client=server, git_exclude=True) == (1, 0, 0)

    assert "/.claude/skills/alpha/" in _exclude(main).splitlines()
    assert not (main / ".git" / "worktrees" / "widget-feature" / "info").exists()
    assert not (worktree / ".git").is_dir()
    assert _untracked(git, worktree) == []


def test_a_target_outside_any_repository_writes_no_exclude_file(engine, tmp_path):
    skills = tmp_path / "plain" / ".claude" / "skills"
    server = FakeServer()
    server.put("alpha", _V1, scope="project:plain")

    assert engine._sync("project:plain", skills, client=server, git_exclude=True) == (1, 0, 0)

    assert (skills / "alpha" / "SKILL.md").is_file()
    assert list(tmp_path.rglob("exclude")) == []
    assert list(tmp_path.rglob(".git")) == []


def test_a_failing_exclude_never_breaks_the_pull(engine, git, tmp_path):
    repo = _init_repo(git, tmp_path / "widget")
    exclude = repo / ".git" / "info" / "exclude"
    exclude.unlink()
    exclude.mkdir()  # unreadable as a file
    skills = repo / ".claude" / "skills"
    server = FakeServer()
    server.put("alpha", _V1, scope="project:widget")

    assert engine._sync("project:widget", skills, client=server, git_exclude=True) == (1, 0, 0)
    assert (skills / "alpha" / "SKILL.md").is_file()


# ---------------------------------------------------------------------------
# synapse_project.git_exclude
# ---------------------------------------------------------------------------

_HEADER = "# Skills synced by Synapse (synapse_project.git_exclude): not part of this repo"


def test_git_exclude_appends_the_header_and_line_and_is_idempotent(project, git, tmp_path):
    repo = _init_repo(git, tmp_path / "widget")
    first = repo / ".claude" / "skills" / "alpha"
    second = repo / "tools" / ".claude" / "skills" / "beta"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    original = _exclude(repo)

    assert project.git_exclude(first) is True
    assert project.git_exclude(first) is True
    assert project.git_exclude(str(second)) is True
    assert project.git_exclude(second) is True

    assert _exclude(repo) == original + (
        f"{_HEADER}\n/.claude/skills/alpha/\n/tools/.claude/skills/beta/\n"
    )


def test_git_exclude_starts_a_new_line_after_a_file_without_one(project, git, tmp_path):
    repo = _init_repo(git, tmp_path / "widget")
    exclude = repo / ".git" / "info" / "exclude"
    exclude.write_text("*.log", encoding="utf-8")
    folder = repo / ".claude" / "skills" / "alpha"

    assert project.git_exclude(folder) is True

    assert exclude.read_text(encoding="utf-8") == f"*.log\n{_HEADER}\n/.claude/skills/alpha/\n"


def test_git_exclude_creates_a_missing_exclude_file(project, git, tmp_path):
    repo = _init_repo(git, tmp_path / "widget")
    shutil.rmtree(repo / ".git" / "info")

    assert project.git_exclude(repo / ".claude" / "skills" / "alpha") is True

    assert _exclude(repo) == f"{_HEADER}\n/.claude/skills/alpha/\n"


def test_git_exclude_keeps_a_header_the_user_moved_and_adds_no_second_one(project, git, tmp_path):
    repo = _init_repo(git, tmp_path / "widget")
    exclude = repo / ".git" / "info" / "exclude"
    exclude.write_text(f"{_HEADER}\n/.claude/skills/alpha/\n*.tmp\n", encoding="utf-8")

    assert project.git_exclude(repo / ".claude" / "skills" / "beta") is True

    text = exclude.read_text(encoding="utf-8")
    assert text.count(_HEADER) == 1
    assert text.endswith("*.tmp\n/.claude/skills/beta/\n")


def test_git_exclude_in_a_linked_worktree_writes_the_main_repositorys_file(project, git, tmp_path):
    main = _init_repo(git, tmp_path / "widget")
    worktree = tmp_path / "wt"
    git(main, "worktree", "add", "-q", "--detach", str(worktree))
    folder = worktree / ".agents" / "skills" / "alpha"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(_V1, encoding="utf-8")

    assert project.git_exclude(folder) is True

    assert _exclude(main).splitlines()[-1] == "/.agents/skills/alpha/"
    assert _untracked(git, worktree) == []
    assert _untracked(git, main) == []


def test_git_exclude_refuses_the_toplevel_itself(project, git, tmp_path):
    repo = _init_repo(git, tmp_path / "widget")
    original = _exclude(repo)

    assert project.git_exclude(repo) is False
    assert _exclude(repo) == original


def test_git_exclude_leaves_a_folder_in_no_repository_alone(project, tmp_path):
    folder = tmp_path / "plain" / ".claude" / "skills" / "alpha"
    folder.mkdir(parents=True)

    assert project.git_exclude(folder) is False
    assert list(tmp_path.rglob("exclude")) == []


@pytest.mark.parametrize(
    "breakage",
    ["exclude-is-a-directory", "info-is-a-file", "dot-git-garbage", "read-denied", "write-denied"],
)
def test_git_exclude_never_raises(project, git, tmp_path, monkeypatch, breakage):
    repo = _init_repo(git, tmp_path / "widget")
    info = repo / ".git" / "info"
    folder = repo / ".claude" / "skills" / "alpha"
    folder.mkdir(parents=True)
    if breakage == "exclude-is-a-directory":
        (info / "exclude").unlink()
        (info / "exclude").mkdir()
    elif breakage == "info-is-a-file":
        shutil.rmtree(info)
        info.write_text("not a directory\n", encoding="utf-8")
    elif breakage == "dot-git-garbage":
        shutil.rmtree(repo / ".git")
        (repo / ".git").write_text("this is not a gitdir line\n", encoding="utf-8")
    elif breakage == "read-denied":
        real = Path.read_text

        def denied(self, *a, **kw):
            if self.name == "exclude":
                raise PermissionError(13, "Permission denied", str(self))
            return real(self, *a, **kw)

        monkeypatch.setattr(Path, "read_text", denied)
    else:

        def denied_open(*a, **kw):
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(project, "open", denied_open, raising=False)

    assert project.git_exclude(folder) is False


def test_git_exclude_writes_nothing_for_a_worktree_whose_main_checkout_is_gone(project, tmp_path):
    gone = tmp_path / "moved-away" / "widget"
    worktree = tmp_path / "widget-feature"
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {gone / '.git' / 'worktrees' / 'feature'}\n")
    folder = worktree / ".agents" / "skills" / "alpha"
    folder.mkdir(parents=True)

    assert project.git_exclude(folder) is False
    assert not (tmp_path / "moved-away").exists()


# ---------------------------------------------------------------------------
# The Claude hook's project pass
# ---------------------------------------------------------------------------


def _record_sync(engine, monkeypatch) -> list[tuple]:
    seen: list[tuple] = []

    def fake_sync(scope, folder, client=None, **kw):
        seen.append((scope, Path(folder), kw))
        return (0, 0, 0)

    monkeypatch.setattr(engine, "_sync", fake_sync)
    monkeypatch.setattr(engine.config, "SKILLS_SYNC", True)
    return seen


def test_the_project_pass_syncs_the_repository_roots_folder_with_git_exclude(
    engine, git, tmp_path, monkeypatch
):
    repo = _init_repo(git, tmp_path / "widget")
    session = repo / "services" / "api"
    session.mkdir(parents=True)
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(session))
    seen = _record_sync(engine, monkeypatch)

    engine.main()

    assert seen == [
        ("global", engine.config.SKILLS_DIR, {}),
        ("project:widget", repo.resolve() / ".claude" / "skills", {"git_exclude": True}),
    ]


def test_the_project_pass_follows_a_marker_to_its_folder(engine, git, tmp_path, monkeypatch):
    repo = _init_repo(git, tmp_path / "monorepo")
    package = repo / "packages" / "gadget"
    (package / "src").mkdir(parents=True)
    (package / ".synapse-project").write_text("Gadget\n", encoding="utf-8")
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(package / "src"))
    seen = _record_sync(engine, monkeypatch)

    engine.main()

    assert seen[1] == (
        "project:Gadget",
        package.resolve() / ".claude" / "skills",
        {"git_exclude": True},
    )


def test_the_project_pass_outside_a_repository_uses_the_folder_itself(
    engine, tmp_path, monkeypatch
):
    folder = tmp_path / "scratch-notes"
    folder.mkdir()
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(folder))
    seen = _record_sync(engine, monkeypatch)

    engine.main()

    assert seen[1] == (
        "project:scratch-notes",
        folder.resolve() / ".claude" / "skills",
        {"git_exclude": True},
    )


def test_no_project_pass_when_the_project_folder_is_the_global_one(engine, tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(home))
    seen = _record_sync(engine, monkeypatch)
    monkeypatch.setattr(engine.config, "SKILLS_DIR", home / ".claude" / "skills")

    engine.main()

    assert [s[0] for s in seen] == ["global"]


def test_the_project_pass_end_to_end_excludes_what_it_pulls(engine, git, tmp_path, monkeypatch):
    repo = _init_repo(git, tmp_path / "widget")
    (repo / "docs").mkdir()
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(repo / "docs"))
    monkeypatch.setattr(engine.config, "SKILLS_SYNC", True)
    server = FakeServer()
    server.put("alpha", _V1, scope="project:widget")
    server.put("tool", _OTHER, scope="global")
    monkeypatch.setattr(engine.config, "post_json", server.post_json)

    engine.main()

    assert (repo / ".claude" / "skills" / "alpha" / "SKILL.md").is_file()
    assert not (repo / "docs" / ".claude").exists()
    assert (engine.config.SKILLS_DIR / "tool" / "SKILL.md").is_file()
    assert "/.claude/skills/alpha/" in _exclude(repo).splitlines()
    assert _untracked(git, repo) == []
    assert [p["scope"] for p in server.listed()] == ["global", "project:widget"]


# ---------------------------------------------------------------------------
# Codex: the project pass
# ---------------------------------------------------------------------------


def test_codex_project_pass_for_a_repository_subfolder(codex, git, tmp_path):
    repo = _init_repo(git, tmp_path / "widget")
    sub = repo / "src" / "pkg"
    sub.mkdir(parents=True)
    user_dir = tmp_path / "home" / ".agents" / "skills"

    assert codex.project_pass(str(sub), user_dir) == (
        "project:widget",
        repo.resolve() / ".agents" / "skills",
    )


def test_codex_project_pass_reads_the_marker(codex, git, tmp_path):
    repo = _init_repo(git, tmp_path / "checkout-2")
    (repo / ".synapse-project").write_text("Widget\n", encoding="utf-8")
    (repo / "lib").mkdir()

    assert codex.project_pass(str(repo / "lib"), tmp_path / "home" / ".agents" / "skills") == (
        "project:Widget",
        repo.resolve() / ".agents" / "skills",
    )


def test_codex_project_pass_in_a_linked_worktree_uses_the_main_checkouts_label(
    codex, git, tmp_path
):
    main = _init_repo(git, tmp_path / "widget")
    worktree = tmp_path / "widget-wt"
    git(main, "worktree", "add", "-q", "--detach", str(worktree))

    assert codex.project_pass(str(worktree), tmp_path / "home" / ".agents" / "skills") == (
        "project:widget",
        worktree.resolve() / ".agents" / "skills",
    )


def test_codex_project_pass_outside_a_repository_uses_the_folder(codex, tmp_path):
    folder = tmp_path / "notes"
    folder.mkdir()

    assert codex.project_pass(str(folder), tmp_path / "home" / ".agents" / "skills") == (
        "project:notes",
        folder.resolve() / ".agents" / "skills",
    )


@pytest.mark.parametrize("cwd", ["none", "empty", "missing", "home"])
def test_codex_project_pass_is_none(codex, tmp_path, cwd):
    home = tmp_path / "home"
    user_dir = home / ".agents" / "skills"
    value = {
        "none": None,
        "empty": "",
        "missing": str(tmp_path / "deleted" / "repo"),
        "home": str(home),
    }[cwd]

    assert codex.project_pass(value, user_dir) is None


def test_codex_run_syncs_global_then_the_project_each_under_its_lock(codex, git, tmp_path):
    repo = _init_repo(git, tmp_path / "widget")
    user_dir = tmp_path / "home" / ".agents" / "skills"
    project_dir = repo / ".agents" / "skills"
    _write_skill(project_dir, "mine", _LOCAL)
    engine = codex._engine()
    locks = {
        "global": engine._lock_path(user_dir),
        "project:widget": engine._lock_path(project_dir),
    }
    server = FakeServer()
    server.put("tool", _OTHER, scope="global")
    server.put("alpha", _V1, scope="project:widget")
    server.put("beta", _V2, scope="project:widget")
    filelock = _load("synapse_filelock_unserved_t", _SCRIPTS / "synapse_filelock.py")
    held_during: list[bool] = []

    def lock_is_held(path, payload):
        if path != "/skills/list":
            return
        with open(locks[payload["scope"]], "a") as other:
            try:
                filelock.lock_exclusive(other, blocking=False)
            except OSError:
                held_during.append(True)
            else:
                held_during.append(False)

    server.on_call = lock_is_held

    result = codex.run(server, user_dir, ("project:widget", project_dir))

    assert result == (3, 1, 0)
    assert [p["scope"] for p in server.listed()] == ["global", "project:widget"]
    assert held_during == [True, True]
    assert sorted(p.name for p in user_dir.iterdir()) == ["tool"]
    assert sorted(p.name for p in project_dir.iterdir()) == ["alpha", "beta", "mine"]
    assert [(p["name"], p["scope"]) for p in server.published()] == [("mine", "project:widget")]
    lines = _exclude(repo).splitlines()
    assert "/.agents/skills/alpha/" in lines and "/.agents/skills/beta/" in lines
    assert "/.agents/skills/mine/" not in lines
    assert _untracked(git, repo) == [".agents/skills/mine/SKILL.md"]
    # Released afterwards.
    for lock_path in locks.values():
        with open(lock_path, "a") as again:
            filelock.lock_exclusive(again, blocking=False)


def test_codex_run_without_a_project_is_the_global_pass_alone(codex, tmp_path):
    server = FakeServer()
    server.put("tool", _OTHER)
    server.put("alpha", _V1, scope="project:widget")

    assert codex.run(server, tmp_path / "agents") == (1, 0, 0)
    assert [p["scope"] for p in server.listed()] == ["global"]


def test_codex_run_skips_the_folders_whose_lock_is_taken(codex, tmp_path):
    agents, project = tmp_path / "agents", tmp_path / "w" / "skills"
    engine = codex._engine()
    engine._lock_path(agents).parent.mkdir(parents=True, exist_ok=True)
    server = FakeServer()
    with (
        open(engine._lock_path(agents), "w") as held_global,
        open(engine._lock_path(project), "w") as held_project,
    ):
        engine.lock_exclusive(held_global, blocking=False)
        engine.lock_exclusive(held_project, blocking=False)
        result = codex.run(server, agents, ("project:w", project))

    assert result == (0, 0, 0) and server.calls == []


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        pytest.param(["skills_sync.py"], None, id="no-flag"),
        pytest.param(["skills_sync.py", "--cwd"], None, id="dangling-flag"),
        pytest.param(["skills_sync.py", "--cwd", "/w/x y"], "/w/x y", id="value"),
        pytest.param(["skills_sync.py", "--other", "--cwd", "/w"], "/w", id="after-another"),
    ],
)
def test_codex_cwd_arg(codex, argv, expected):
    assert codex._cwd_arg(argv) == expected


def test_codex_main_passes_the_cwd_project_to_run(codex, git, tmp_path, monkeypatch):
    repo = _init_repo(git, tmp_path / "widget")
    (repo / "src").mkdir()
    user_dir = tmp_path / "home" / ".agents" / "skills"
    monkeypatch.setattr(codex, "SKILLS_SYNC", True)
    monkeypatch.setattr(codex, "SKILLS_DIR", user_dir)
    monkeypatch.setattr(sys, "argv", ["skills_sync.py", "--cwd", str(repo / "src")])
    seen = []
    monkeypatch.setattr(codex, "run", lambda *a: seen.append(a) or (1, 0, 0))

    assert codex.main() == 0

    ((_client, target_dir, project),) = seen
    assert target_dir == user_dir
    assert project == ("project:widget", repo.resolve() / ".agents" / "skills")


def test_codex_main_without_cwd_runs_no_project_pass(codex, tmp_path, monkeypatch):
    monkeypatch.setattr(codex, "SKILLS_SYNC", True)
    monkeypatch.setattr(codex, "SKILLS_DIR", tmp_path / "home" / ".agents" / "skills")
    monkeypatch.setattr(sys, "argv", ["skills_sync.py"])
    seen = []
    monkeypatch.setattr(codex, "run", lambda *a: seen.append(a) or (0, 0, 0))

    assert codex.main() == 0
    assert seen[0][2] is None


# ---------------------------------------------------------------------------
# Codex: the session-start hook hands the cwd over
# ---------------------------------------------------------------------------


@pytest.fixture()
def hook(monkeypatch) -> ModuleType:
    mod = _load("synapse_codex_session_start_unserved_t", _CODEX / "hooks" / "session_start.py")
    seen: dict = {}

    def fake_run(cmd, **kw):
        seen["cmd"], seen["kw"] = cmd, kw

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    mod.seen = seen
    return mod


@pytest.mark.parametrize("cwd", ["/work/widget", "/work/my widget", "C:\\work\\widget"])
def test_sync_skills_passes_the_cwd(hook, monkeypatch, cwd):
    monkeypatch.setenv("SYNAPSE_SKILLS_SYNC", "1")

    hook._sync_skills(cwd)

    cmd = hook.seen["cmd"]
    assert cmd[1].endswith("skills_sync.py")
    assert cmd[2:] == ["--cwd", cwd]
    assert "shell" not in hook.seen["kw"]


@pytest.mark.parametrize("cwd", [None, ""])
def test_sync_skills_without_a_cwd_passes_no_flag(hook, monkeypatch, cwd):
    monkeypatch.setenv("SYNAPSE_SKILLS_SYNC", "1")

    hook._sync_skills(cwd)

    assert hook.seen["cmd"][-1].endswith("skills_sync.py")
    assert "--cwd" not in hook.seen["cmd"]


def test_sync_skills_stays_off_unless_opted_in(hook, monkeypatch):
    monkeypatch.setenv("SYNAPSE_SKILLS_SYNC", "0")

    hook._sync_skills("/work/widget")

    assert hook.seen == {}


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        pytest.param({"cwd": "/work/widget/src"}, "/work/widget/src", id="payload-cwd"),
        pytest.param({}, "PROCESS_CWD", id="no-cwd"),
        pytest.param({"cwd": ""}, "PROCESS_CWD", id="empty-cwd"),
    ],
)
def test_session_start_main_syncs_the_sessions_cwd(hook, monkeypatch, tmp_path, payload, expected):
    seen = []
    monkeypatch.setattr(hook, "_sync_skills", lambda cwd=None: seen.append(cwd))
    for name in ("_upload_warning", "_prefs_text", "token_override_notice"):
        monkeypatch.setattr(hook, name, lambda: None)
    monkeypatch.setattr(hook, "_board_text", lambda project: None)
    monkeypatch.setattr(hook, "_spawn_catchup", lambda skip="": None)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))

    hook.main()

    assert seen == [os.getcwd() if expected == "PROCESS_CWD" else expected]


# ---------------------------------------------------------------------------
# synapse skill scope
# ---------------------------------------------------------------------------


@pytest.fixture()
def cli(monkeypatch):
    """synapse_cli loaded as the `synapse` command loads it, with config.post_json
    replaced by a recorder answering ``reply`` (an exception is raised instead)."""
    sys.path.insert(0, str(_SCRIPTS))
    spec = importlib.util.spec_from_file_location("synapse_cli", _SCRIPTS / "synapse_cli.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["synapse_cli"] = mod
    spec.loader.exec_module(mod)
    calls: list[tuple] = []
    state: dict = {"reply": {}}

    def post_json(path, payload, timeout=30.0):
        calls.append((path, payload))
        if isinstance(state["reply"], Exception):
            raise state["reply"]
        return state["reply"]

    monkeypatch.setattr(mod.config, "post_json", post_json)

    def run(reply, *argv):
        state["reply"] = reply
        return mod.main(list(argv))

    return run, calls


def test_skill_scope_posts_the_move_and_prints_it(cli, capsys):
    run, calls = cli
    reply = {"status": "rescoped", "name": "alpha", "scope": "project:Widget", "from": "global"}

    assert run(reply, "skill", "scope", "alpha", "project:Widget") == 0

    assert calls == [("/skills/rescope", {"name": "alpha", "scope": "project:Widget"})]
    out = capsys.readouterr().out
    assert "Moved alpha from global to project:Widget." in out


def test_skill_scope_back_to_global(cli, capsys):
    run, calls = cli
    reply = {"status": "rescoped", "name": "alpha", "scope": "global", "from": "project:widget"}

    assert run(reply, "skill", "scope", "alpha", "global") == 0

    assert calls == [("/skills/rescope", {"name": "alpha", "scope": "global"})]
    assert "Moved alpha from project:widget to global." in capsys.readouterr().out


@pytest.mark.parametrize(
    ("reply", "code", "stream", "text"),
    [
        pytest.param(
            {"status": "unchanged", "name": "alpha", "scope": "project:widget"},
            0,
            "out",
            "alpha is already project:widget.",
            id="unchanged",
        ),
        pytest.param(
            {"status": "not_found", "name": "alpha"},
            1,
            "err",
            "synapse: no skill named 'alpha' in Synapse",
            id="not-found",
        ),
        pytest.param(
            {"status": "error", "detail": "alpha is retired, not active"},
            1,
            "err",
            "synapse: alpha is retired, not active",
            id="error-detail",
        ),
        pytest.param(
            {"status": "surprise"},
            1,
            "err",
            "scope change failed: {'status': 'surprise'}",
            id="unknown-status",
        ),
    ],
)
def test_skill_scope_maps_the_server_status(cli, capsys, reply, code, stream, text):
    run, calls = cli

    assert run(reply, "skill", "scope", "alpha", "project:widget") == code

    assert calls == [("/skills/rescope", {"name": "alpha", "scope": "project:widget"})]
    assert text in getattr(capsys.readouterr(), stream)


def test_skill_scope_403_says_only_full_trust_can_move_a_skill(cli, capsys):
    run, _ = cli
    error = urllib.error.HTTPError(_URL, 403, "Forbidden", None, None)

    assert run(error, "skill", "scope", "alpha", "project:widget") == 1

    err = capsys.readouterr().err
    assert "scope change failed: HTTP 403" in err and "full-trust" in err


@pytest.mark.parametrize(
    ("error", "text"),
    [
        pytest.param(
            urllib.error.HTTPError(_URL, 500, "Server Error", None, None),
            "scope change failed: HTTP 500",
            id="http-500",
        ),
        pytest.param(
            urllib.error.URLError("connection refused"),
            "scope change failed: <urlopen error connection refused>",
            id="network",
        ),
    ],
)
def test_skill_scope_other_failures(cli, capsys, error, text):
    run, _ = cli

    assert run(error, "skill", "scope", "alpha", "global") == 1

    err = capsys.readouterr().err
    assert text in err and "full-trust" not in err


def test_skill_scope_needs_a_scope(cli, capsys):
    run, calls = cli

    with pytest.raises(SystemExit) as exc:
        run({}, "skill", "scope", "alpha")

    assert exc.value.code == 2 and calls == []


def test_skill_retire_still_posts_to_retire(cli):
    run, calls = cli

    assert (
        run({"status": "retired", "name": "alpha", "scope": None}, "skill", "retire", "alpha") == 0
    )
    assert calls == [("/skills/retire", {"name": "alpha"})]


# ---------------------------------------------------------------------------
# Nothing from a server is written outside the folder; refused pushes don't vouch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["..", "a/../../escape", "back\\slash", "C:"])
def test_a_skill_name_that_is_not_one_folder_is_never_pulled(engine, tmp_path, name):
    target = tmp_path / "home" / ".claude" / "skills"
    server = FakeServer()
    server.put(name, _V1)
    engine._sync("global", target, client=server)
    assert not [p for p in tmp_path.rglob("*") if p.is_file() and "SKILL" in p.name]


@pytest.mark.parametrize("path", ["../escape.md", "/abs.md", "a\\b.md", "a/../../b.md"])
def test_a_file_path_outside_the_skill_folder_is_never_written(engine, tmp_path, path):
    target = tmp_path / "home" / ".claude" / "skills"
    server = FakeServer()
    server.put("alpha", _V1, [(path, b"payload")])
    engine._sync("global", target, client=server)
    assert not [p for p in tmp_path.rglob("*") if p.is_file() and p.read_bytes() == b"payload"]


def test_an_edit_an_older_client_was_refused_is_not_deleted_as_delivered(engine, target):
    """A restricted device's older client pushed its edit of a global skill; the server
    answered 'forbidden', which that client treated as success and recorded in the
    ledger. Once the skill is withdrawn from the device, that edit is not a version
    Synapse delivered: kept, not deleted."""
    server = FakeServer()
    server.put("alpha", _V1)
    assert engine._sync("global", target, client=server) == (1, 0, 0)
    (target / "alpha" / "SKILL.md").write_text(_V2, encoding="utf-8")
    ledger = engine._load_ledger(target)
    engine._remember(ledger, "alpha", _hash(engine, _V2))  # what the older client did
    engine._save_ledger(target, ledger)
    server.withdraw("alpha", _hash(engine, _V1), scope="")
    assert engine._sync("global", target, client=server) == (0, 0, 0)
    assert (target / "alpha" / "SKILL.md").read_text(encoding="utf-8") == _V2
