"""Skill retirement on the client (#222), and the MCP headers helper both harnesses run.

`synapse skill retire` marks a skill retired on the server. /skills/list then reports it
under its own "retired" key, with the whole-skill hash it was retired with and the time.
The sync engine (plugin/scripts/skills_sync.py) settles a local copy of a retired skill
by the same newest-edit rule as every other conflict:

  * still exactly a copy Synapse delivered (the retired hash, or any hash in this
    folder's delivered ledger; in another scope's folder only the retired hash),
    whatever its timestamp -> the folder is deleted, its ledger kept
  * an unrecognised copy edited after the retirement -> pushed, which reactivates the
    skill; if the server refuses the push, the folder is kept and nothing is counted
  * anything else (an unrecognised copy, a symlinked folder, an entry with no usable
    retired_at) -> left alone and never pushed back
  * an old server that sends no "retired" key -> the sync behaves exactly as before

Deleting files from the user's machine is the one irreversible thing the sync does, so
the ledger that says what was "delivered" (~/.synapse/skills/delivered-<folder>.json) is
covered too: one per target folder, written after pull/push/identical, capped per skill,
and a corrupt file reads as empty (nothing recognised, so nothing deleted).

Both hooks report a removal like any other sync result: the Claude hook asks for a
skills reload, the Codex wrapper prints "removed N".

mcp_headers.py is a credential channel, not a log: exactly one JSON object with the
"Authorization" key on stdout, and on failure nothing there, exit 1 and a token-free
reason in ~/.synapse/logs/mcp_headers.log. It must work from config.json alone, because
Codex runs it with the environment cleared down to HOME and PATH.

Stdlib-only scripts loaded by path against an in-memory /skills/* server. No network.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "plugin" / "scripts"
_CODEX_SCRIPTS = _REPO / "plugin-codex" / "scripts"
_MODULES = ("config", "enroll", "synapse_login", "harness", "synapse_cli", "synapse_home", "common")

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
    "CLAUDE_CODE_MCP_SERVER_URL",
)

_V1 = "---\ndescription: back up the database\n---\nstep one\n"
_V2 = "---\ndescription: back up the database\n---\nstep one\nstep two\n"
_LOCAL = "---\ndescription: back up the database\n---\nstep one, my way\n"
_REF = [("ref/notes.md", b"reference notes\n")]

_V1_AT = "2026-09-18T12:00:00+00:00"
_RETIRED_AT = "2026-09-25T12:00:00+00:00"
_RETIRED_TS = datetime.fromisoformat(_RETIRED_AT).timestamp()
_BEFORE = _RETIRED_TS - 3600
_AFTER = _RETIRED_TS + 3600

_URL = "https://synapse.example.net"
_TOKEN = "tkHDR7Qm2Zp9Lw4Xv1"


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _forget() -> None:
    for mod in _MODULES:
        sys.modules.pop(mod, None)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Scratch HOME, Synapse home, Claude config and data dirs; no Synapse env; a cwd
    with no project .claude; sys.path restored afterwards (every script prepends its own
    directory); no cached config modules on either side of the test."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("SYNAPSE_HOME", str(tmp_path / "synapse-home"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("SYNAPSE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CLAUDE_SKILLS_DIR", str(tmp_path / "skills"))
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", [*sys.path])
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
    return _load("synapse_retire_engine_t", _SCRIPTS / "skills_sync.py")


@pytest.fixture()
def target(tmp_path) -> Path:
    return tmp_path / "skills"


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _hash(engine, body: str, files=()) -> str:
    return engine._whole_hash(body, [(rel, _sha(content)) for rel, content in files])


class FakeServer:
    """In-memory /skills/* routes for one scope: the active skills, the retired entries
    (``None`` = an old server that sends no "retired" key) and every call made."""

    def __init__(self):
        self.skills: dict[str, dict] = {}
        self.retired: list[dict] | None = []
        self.publish_reply: dict = {"status": "ok"}
        self.calls: list[tuple[str, dict]] = []

    def put(self, name: str, body: str, files=(), at: str = _V1_AT) -> None:
        self.skills[name] = {
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

    def retire(self, name: str, content_hash: str, **entry) -> None:
        self.skills.pop(name, None)
        self.retired = [
            *(self.retired or []),
            {"name": name, "scope": "global", "content_hash": content_hash}
            | ({"retired_at": _RETIRED_AT} | entry),
        ]

    def published(self) -> list[dict]:
        return [p for path, p in self.calls if path == "/skills/publish"]

    def post_json(self, path, payload, timeout=30.0):
        self.calls.append((path, payload))
        if path == "/skills/list":
            reply: dict = {
                "skills": [
                    {
                        "name": name,
                        "body": s["body"],
                        "files": [{"path": f["path"], "sha256": f["sha256"]} for f in s["files"]],
                        "content_modified_at": s["content_modified_at"],
                    }
                    for name, s in sorted(self.skills.items())
                ]
            }
            if self.retired is not None:
                reply["retired"] = self.retired
            return reply
        if path == "/skills/fetch":
            s = self.skills.get(payload["name"])
            return {"found": True, **s} if s else {"found": False}
        if path == "/skills/publish":
            if self.publish_reply.get("status") == "ok":
                self.skills[payload["name"]] = {
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


# ---------------------------------------------------------------------------
# A local copy of a retired skill
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mtime",
    [
        pytest.param(_BEFORE, id="older"),
        # content_modified_at round-trips through a float: a sub-EPS gap is a tie, not an edit
        pytest.param(_RETIRED_TS + 1.0, id="within-eps"),
    ],
)
def test_a_copy_matching_the_retired_hash_and_no_newer_is_deleted(engine, target, mtime):
    folder = _write_skill(target, "alpha", _V1, _REF, mtime=mtime)
    server = FakeServer()
    server.retire("alpha", _hash(engine, _V1, _REF))

    assert engine._sync("global", target, client=server) == (0, 0, 1)

    assert not folder.exists()
    assert server.published() == []
    # Kept: a restored copy is recognised, and removed again rather than pushed.
    assert _ledger(engine, target)["alpha"] == [_hash(engine, _V1, _REF)]


@pytest.mark.parametrize("mtime", [_AFTER, _RETIRED_TS + 86400 * 30])
def test_a_delivered_copy_with_a_newer_timestamp_is_still_removed_never_pushed(
    engine, target, mtime
):
    """A fresh git clone, a restore or a copy without -p renews the mtime of exactly the
    content Synapse delivered: that is not an edit, and pushing it would bring the
    retired skill back on every machine."""
    folder = _write_skill(target, "alpha", _V1, _REF, mtime=mtime)
    server = FakeServer()
    server.retire("alpha", _hash(engine, _V1, _REF))

    assert engine._sync("global", target, client=server) == (0, 0, 1)

    assert not folder.exists() and server.published() == []


def test_a_ledger_recognised_copy_with_a_newer_timestamp_is_removed(engine, target):
    server = FakeServer()
    server.put("alpha", _V1, _REF)
    assert engine._sync("global", target, client=server) == (1, 0, 0)
    for fp in (target / "alpha").rglob("*"):
        os.utime(fp, (_AFTER, _AFTER))

    server.retire("alpha", _hash(engine, _V2, _REF))
    assert engine._sync("global", target, client=server) == (0, 0, 1)
    assert server.published() == []


def test_a_restored_older_version_after_a_removal_is_removed_again(engine, target):
    """Committed v1, pulled v2 as a working-tree change, retired at v2: the removal shows
    as a deletion in `git status`, and `git restore` brings back v1 with a new mtime."""
    server = FakeServer()
    server.put("alpha", _V1, _REF)
    assert engine._sync("global", target, client=server) == (1, 0, 0)
    server.put("alpha", _V2, _REF, at="2026-09-20T12:00:00+00:00")
    assert engine._sync("global", target, client=server) == (1, 0, 0)
    server.retire("alpha", _hash(engine, _V2, _REF))
    assert engine._sync("global", target, client=server) == (0, 0, 1)

    _write_skill(target, "alpha", _V1, _REF, mtime=_AFTER)
    assert engine._sync("global", target, client=server) == (0, 0, 1)
    assert server.published() == [] and not (target / "alpha").exists()


def test_another_scopes_folder_keeps_a_different_skill_of_the_same_name(engine, tmp_path):
    """Two repositories each hold their own `deploy`; both were pushed, so each is in its
    own folder's ledger. Retiring the one the row last named must not delete the other."""
    repo_a, repo_b = tmp_path / "a" / "skills", tmp_path / "b" / "skills"
    server = FakeServer()
    _write_skill(repo_b, "alpha", _LOCAL, _REF)
    assert engine._sync("project:b", repo_b, client=server) == (0, 1, 0)
    a_files = [*_REF, ("manifests/k.yaml", b"kind: x\n")]
    _write_skill(repo_a, "alpha", _V1, a_files, mtime=_RETIRED_TS - 60)
    assert engine._sync("project:a", repo_a, client=server) == (0, 1, 0)
    server.retire("alpha", _hash(engine, _V1, a_files), scope="project:a")

    assert engine._sync("project:a", repo_a, client=server) == (0, 0, 1)
    assert engine._sync("project:b", repo_b, client=server) == (0, 0, 0)
    assert (repo_b / "alpha" / "SKILL.md").read_text(encoding="utf-8") == _LOCAL


def test_the_exact_retired_content_in_another_scopes_folder_is_removed(engine, tmp_path):
    folder = _write_skill(tmp_path / "b", "alpha", _V1, _REF)
    server = FakeServer()
    server.retire("alpha", _hash(engine, _V1, _REF), scope="project:a")
    assert engine._sync("project:b", tmp_path / "b", client=server) == (0, 0, 1)
    assert not folder.exists()


def test_skill_md_goes_first_so_a_failure_leaves_the_copy_whole(engine, target, monkeypatch):
    """Windows' rmtree deletes bottom-up and stops at the first locked file: a skill left
    with SKILL.md and half its files would hash as unrecognised, and be pushed back."""
    folder = _write_skill(target, "alpha", _V1, _REF)
    server = FakeServer()
    server.retire("alpha", _hash(engine, _V1, _REF))
    real_unlink = Path.unlink

    def locked(self, *a, **kw):
        if self.name == "SKILL.md":
            raise PermissionError(32, "in use", str(self))
        return real_unlink(self, *a, **kw)

    monkeypatch.setattr(Path, "unlink", locked)
    assert engine._sync("global", target, client=server) == (0, 0, 0)
    assert (folder / "ref" / "notes.md").exists()
    monkeypatch.setattr(Path, "unlink", real_unlink)
    assert engine._sync("global", target, client=server) == (0, 0, 1)


def test_a_junction_is_kept_like_a_symlink(engine, target, monkeypatch):
    """Windows: is_symlink() misses a directory junction before 3.12 and rmtree refuses
    one; its reparse-point attribute gives it away."""
    folder = _write_skill(target, "alpha", _V1, _REF)
    real_lstat = os.lstat

    def lstat(path, *a, **kw):
        st = real_lstat(path, *a, **kw)
        if Path(path) == folder:
            return SimpleNamespace(st_file_attributes=0x400, st_mode=st.st_mode)
        return st

    monkeypatch.setattr(engine.os, "lstat", lstat)
    assert engine._is_link(folder)
    removed = []
    monkeypatch.setattr(engine.shutil, "rmtree", lambda p, *a, **k: removed.append(p))
    server = FakeServer()
    server.retire("alpha", _hash(engine, _V1, _REF))

    assert engine._sync("global", target, client=server) == (0, 0, 0)
    assert removed == [] and server.published() == []


def test_a_folder_that_cannot_be_removed_is_kept_and_the_pass_goes_on(engine, target, monkeypatch):
    _write_skill(target, "alpha", _V1, _REF)
    server = FakeServer()
    server.retire("alpha", _hash(engine, _V1, _REF))
    server.put("zeta", _V2)
    real_unlink = Path.unlink

    def refuse(self, *a, **kw):
        if self.parent.name == "alpha":
            raise PermissionError(13, "file in use", str(self))
        return real_unlink(self, *a, **kw)

    monkeypatch.setattr(Path, "unlink", refuse)

    assert engine._sync("global", target, client=server) == (1, 0, 0)
    assert (target / "zeta" / "SKILL.md").exists()
    assert (target / "alpha" / "SKILL.md").exists()


def test_an_older_delivered_version_is_recognised_through_the_ledger(engine, target):
    """This machine pulled v1; the server moved on to v2 and retired that. The copy here
    never matches the retired hash, but it is still exactly something Synapse delivered."""
    server = FakeServer()
    server.put("alpha", _V1, _REF)
    assert engine._sync("global", target, client=server) == (1, 0, 0)

    server.retire("alpha", _hash(engine, _V2, _REF))
    assert engine._sync("global", target, client=server) == (0, 0, 1)

    assert not (target / "alpha").exists()
    assert server.published() == []


@pytest.mark.parametrize(
    "mtime",
    [pytest.param(_AFTER, id="an-hour-later"), pytest.param(_RETIRED_TS + 2.5, id="just-past-eps")],
)
def test_a_copy_edited_after_the_retirement_is_pushed_to_reactivate_it(engine, target, mtime):
    folder = _write_skill(target, "alpha", _LOCAL, _REF, mtime=mtime)
    server = FakeServer()
    server.retire("alpha", _hash(engine, _V1, _REF))

    assert engine._sync("global", target, client=server) == (0, 1, 0)

    (push,) = server.published()
    assert (push["name"], push["scope"], push["body"]) == ("alpha", "global", _LOCAL)
    assert datetime.fromisoformat(push["content_modified_at"]).timestamp() == pytest.approx(mtime)
    assert [f["path"] for f in push["files"]] == ["ref/notes.md"]
    assert (folder / "SKILL.md").read_text(encoding="utf-8") == _LOCAL


def test_a_reactivating_push_the_server_refuses_is_not_counted_and_keeps_the_folder(engine, target):
    """The listing can predate a newer retirement; the server then answers 'retired'."""
    folder = _write_skill(target, "alpha", _LOCAL, mtime=_AFTER)
    server = FakeServer()
    server.retire("alpha", _hash(engine, _V1))
    server.publish_reply = {"status": "retired", "name": "alpha", "retired_at": _RETIRED_AT}

    assert engine._sync("global", target, client=server) == (0, 0, 0)

    assert len(server.published()) == 1
    assert (folder / "SKILL.md").read_text(encoding="utf-8") == _LOCAL
    assert "alpha" not in server.skills


@pytest.mark.parametrize(
    ("body", "files"),
    [
        pytest.param(_V1, _REF, id="never-delivered-version"),
        pytest.param(_V2, [*_REF, ("mine.md", b"my own notes\n")], id="extra-local-file"),
    ],
)
def test_an_unrecognised_copy_older_than_the_retirement_is_left_alone(engine, target, body, files):
    folder = _write_skill(target, "alpha", body, files)
    server = FakeServer()
    server.retire("alpha", _hash(engine, _V2, _REF))

    assert engine._sync("global", target, client=server) == (0, 0, 0)

    assert (folder / "SKILL.md").read_text(encoding="utf-8") == body
    assert sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()) == (
        sorted(["SKILL.md", *(rel for rel, _ in files)])
    )
    assert server.published() == []


def test_a_symlinked_skill_folder_is_never_deleted(engine, target, tmp_path):
    real = _write_skill(tmp_path / "dotfiles", "alpha", _V1, _REF)
    target.mkdir()
    link = target / "alpha"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable here")
    server = FakeServer()
    server.retire("alpha", _hash(engine, _V1, _REF))
    # The scan does reach the linked copy, and it matches the retired hash exactly: only
    # the symlink guard keeps it.
    assert [md.parent.name for md in target.glob("*/SKILL.md")] == ["alpha"]

    assert engine._sync("global", target, client=server) == (0, 0, 0)

    assert link.is_symlink()
    assert (real / "SKILL.md").read_text(encoding="utf-8") == _V1
    assert (real / "ref" / "notes.md").read_bytes() == b"reference notes\n"
    assert server.published() == []


_MISSING = object()


@pytest.mark.parametrize(
    "retired_at",
    [
        pytest.param(None, id="null"),
        pytest.param("not-a-date", id="unparseable"),
        pytest.param(_MISSING, id="missing"),
    ],
)
@pytest.mark.parametrize(
    "mtime", [pytest.param(_BEFORE, id="older"), pytest.param(_AFTER, id="newer")]
)
def test_a_retired_entry_without_a_usable_retired_at_is_left_alone(
    engine, target, retired_at, mtime
):
    folder = _write_skill(target, "alpha", _V1, mtime=mtime)
    server = FakeServer()
    server.retire("alpha", _hash(engine, _V1))
    if retired_at is _MISSING:
        del server.retired[0]["retired_at"]
    else:
        server.retired[0]["retired_at"] = retired_at

    assert engine._sync("global", target, client=server) == (0, 0, 0)

    assert (folder / "SKILL.md").read_text(encoding="utf-8") == _V1
    assert server.published() == []


def test_a_retired_skill_missing_on_disk_is_not_pulled(engine, target):
    server = FakeServer()
    server.retire("alpha", _hash(engine, _V1))

    assert engine._sync("global", target, client=server) == (0, 0, 0)

    assert not (target / "alpha").exists()
    assert [path for path, _ in server.calls] == ["/skills/list"]


@pytest.mark.parametrize(
    "retired",
    [
        pytest.param(None, id="old-server-no-key"),
        pytest.param([], id="nothing-retired"),
        pytest.param(
            [{"name": "other", "content_hash": "0" * 64, "retired_at": _RETIRED_AT}],
            id="another-skill-retired",
        ),
    ],
)
def test_a_skill_only_on_disk_and_not_retired_is_published_as_before(engine, target, retired):
    folder = _write_skill(target, "alpha", _V1)
    server = FakeServer()
    server.retired = retired

    assert engine._sync("global", target, client=server) == (0, 1, 0)

    assert [p["name"] for p in server.published()] == ["alpha"]
    assert folder.is_dir()
    assert server.skills["alpha"]["body"] == _V1


# ---------------------------------------------------------------------------
# The delivered ledger
# ---------------------------------------------------------------------------


def test_the_ledger_lives_under_synapse_home_not_in_the_skills_folder(engine, target, tmp_path):
    server = FakeServer()
    server.put("alpha", _V1, _REF)

    engine._sync("global", target, client=server)

    ledgers = list((tmp_path / "synapse-home" / "skills").glob("delivered-*.json"))
    assert len(ledgers) == 1
    assert json.loads(ledgers[0].read_text(encoding="utf-8")) == {
        "alpha": [_hash(engine, _V1, _REF)]
    }
    assert [p.name for p in target.iterdir()] == ["alpha"]
    assert not (tmp_path / "home" / ".synapse").exists()


def test_each_target_folder_keeps_its_own_ledger(engine, tmp_path):
    """A copy pulled into one folder must not make a hand-placed copy elsewhere look
    delivered."""
    pulled_into, placed_in = tmp_path / "claude-skills", tmp_path / "agents-skills"
    server = FakeServer()
    server.put("alpha", _V1)
    assert engine._sync("global", pulled_into, client=server) == (1, 0, 0)

    server.retire("alpha", _hash(engine, _V2))
    _write_skill(placed_in, "alpha", _V1)
    assert engine._sync("global", placed_in, client=server) == (0, 0, 0)
    assert engine._sync("global", pulled_into, client=server) == (0, 0, 1)

    assert (placed_in / "alpha" / "SKILL.md").read_text(encoding="utf-8") == _V1
    assert not (pulled_into / "alpha").exists()
    assert engine._ledger_path(pulled_into) != engine._ledger_path(placed_in)
    assert engine._ledger_path(pulled_into).exists() and engine._ledger_path(placed_in).exists()


def _only_on_server(server, target):
    server.put("alpha", _V1)


def _only_on_disk(server, target):
    _write_skill(target, "alpha", _V1)


def _identical(server, target):
    server.put("alpha", _V1)
    _write_skill(target, "alpha", _V1)


def _local_edit_newer(server, target):
    server.put("alpha", _V1, at=_V1_AT)
    _write_skill(target, "alpha", _V2, mtime=_AFTER)


def _server_edit_newer(server, target):
    server.put("alpha", _V2, at=_RETIRED_AT)
    _write_skill(target, "alpha", _V1, mtime=_BEFORE - 86400 * 30)


@pytest.mark.parametrize(
    ("setup", "counts", "agreed"),
    [
        pytest.param(_only_on_server, (1, 0, 0), _V1, id="pull"),
        pytest.param(_only_on_disk, (0, 1, 0), _V1, id="push"),
        pytest.param(_identical, (0, 0, 0), _V1, id="identical"),
        pytest.param(_local_edit_newer, (0, 1, 0), _V2, id="newer-local-push"),
        pytest.param(_server_edit_newer, (1, 0, 0), _V2, id="newer-server-pull"),
    ],
)
def test_the_ledger_records_the_hash_this_folder_agreed_with_the_server_on(
    engine, target, setup, counts, agreed
):
    server = FakeServer()
    setup(server, target)

    assert engine._sync("global", target, client=server) == counts

    assert _ledger(engine, target) == {"alpha": [_hash(engine, agreed)]}


def test_a_push_the_server_refuses_is_neither_counted_nor_remembered(engine, target):
    folder = _write_skill(target, "alpha", _V1)
    server = FakeServer()
    server.publish_reply = {"status": "retired", "name": "alpha", "retired_at": _RETIRED_AT}

    assert engine._sync("global", target, client=server) == (0, 0, 0)

    assert "alpha" not in _ledger(engine, target)
    assert folder.is_dir()


def test_the_ledger_keeps_the_newest_twenty_hashes_per_skill(engine, target):
    path = engine._ledger_path(target)
    path.parent.mkdir(parents=True)
    old = [f"{i:064x}" for i in range(20)]
    path.write_text(json.dumps({"alpha": old, "beta": ["b" * 64]}), encoding="utf-8")
    server = FakeServer()
    server.put("alpha", _V1)

    engine._sync("global", target, client=server)

    assert _ledger(engine, target) == {"alpha": [*old[1:], _hash(engine, _V1)], "beta": ["b" * 64]}


def test_remembering_a_known_hash_moves_it_to_the_newest_end_without_duplicating(engine):
    ledger = {"alpha": ["h1", "h2", "h3"]}
    engine._remember(ledger, "alpha", "h1")
    assert ledger == {"alpha": ["h2", "h3", "h1"]}


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(b"{not json", id="garbage"),
        pytest.param(b"\xff\xfe\x00binary", id="not-utf8"),
        pytest.param(b"", id="empty"),
        pytest.param(b"[1, 2]", id="a-list"),
        pytest.param(b"null", id="null"),
    ],
)
def test_a_corrupt_ledger_reads_as_empty_and_is_rewritten(engine, target, content):
    path = engine._ledger_path(target)
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    kept = _write_skill(target, "alpha", _V1)
    server = FakeServer()
    server.put("beta", _V2)
    server.retire("alpha", _hash(engine, _V2))

    assert engine._sync("global", target, client=server) == (1, 0, 0)

    assert kept.is_dir()
    assert _ledger(engine, target) == {"beta": [_hash(engine, _V2)]}


def test_a_ledger_entry_of_the_wrong_shape_reads_as_empty(engine, target):
    path = engine._ledger_path(target)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"alpha": 5}), encoding="utf-8")
    server = FakeServer()
    server.put("alpha", _V1)

    assert engine._sync("global", target, client=server) == (1, 0, 0)
    assert _ledger(engine, target) == {"alpha": [_hash(engine, _V1)]}


def test_a_reactivating_push_is_remembered_like_any_other_push(engine, target):
    folder = _write_skill(target, "alpha", _LOCAL, mtime=_AFTER)
    server = FakeServer()
    server.retire("alpha", _hash(engine, _V1))
    assert engine._sync("global", target, client=server) == (0, 1, 0)

    # Another machine edits the reactivated skill, then it is retired again, before
    # this folder syncs a second time.
    second = datetime.fromtimestamp(_AFTER + 86400, tz=UTC).isoformat()
    server.retired = []
    server.retire("alpha", _hash(engine, _V2), retired_at=second)

    assert engine._sync("global", target, client=server) == (0, 0, 1)
    assert not folder.exists()


# ---------------------------------------------------------------------------
# The hooks report a removal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("result", "reload"),
    [
        pytest.param((0, 0, 1), True, id="only-a-removal"),
        pytest.param((0, 0, 0), False, id="nothing"),
    ],
)
def test_the_claude_hook_reloads_skills_when_only_a_removal_happened(
    engine, monkeypatch, capsys, result, reload
):
    monkeypatch.setattr(engine.config, "SKILLS_SYNC", True)
    monkeypatch.setattr(engine, "_sync", lambda scope, folder, **_kw: result)

    engine.main()

    out = capsys.readouterr().out
    if reload:
        assert json.loads(out) == {
            "hookSpecificOutput": {"hookEventName": "SessionStart", "reloadSkills": True}
        }
    else:
        assert out == ""


def test_a_removal_in_the_project_folder_alone_also_reloads(engine, monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path / "repo"))
    monkeypatch.setattr(engine.config, "SKILLS_SYNC", True)
    seen = []

    def fake_sync(scope, folder, **_kw):
        seen.append(scope)
        return (0, 0, 1) if scope.startswith("project:") else (0, 0, 0)

    monkeypatch.setattr(engine, "_sync", fake_sync)

    engine.main()

    assert seen == ["global", "project:repo"]
    assert json.loads(capsys.readouterr().out)["hookSpecificOutput"]["reloadSkills"] is True


def test_the_codex_run_returns_pulled_pushed_removed(engine, tmp_path):
    codex = _load("synapse_codex_retire_t", _CODEX_SCRIPTS / "skills_sync.py")
    folder = tmp_path / "agents-skills"
    _write_skill(folder, "retired-one", _V1)
    _write_skill(folder, "mine", _LOCAL)
    server = FakeServer()
    server.put("theirs", _V2)
    server.retire("retired-one", _hash(engine, _V1))

    assert codex.run(server, folder) == (1, 1, 1)

    assert sorted(p.name for p in folder.iterdir()) == ["mine", "theirs"]
    assert [p["name"] for p in server.published()] == ["mine"]


def test_the_codex_main_reports_removals(monkeypatch, capsys, tmp_path):
    codex = _load("synapse_codex_retire_t", _CODEX_SCRIPTS / "skills_sync.py")
    monkeypatch.setattr(codex, "SKILLS_SYNC", True)
    monkeypatch.setattr(codex, "SKILLS_DIR", tmp_path / "agents-skills")
    monkeypatch.setattr(codex, "run", lambda *_a: (0, 0, 2))

    assert codex.main() == 0

    assert capsys.readouterr().out == (
        f"skills: pulled 0, pushed 0, removed 2 -> {tmp_path / 'agents-skills'}\n"
    )


# ---------------------------------------------------------------------------
# mcp_headers.py
# ---------------------------------------------------------------------------


def _home_config(tmp_path, **values) -> None:
    home = tmp_path / "synapse-home"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.json").write_text(json.dumps(values), encoding="utf-8")


def _legacy_options(tmp_path, **options) -> None:
    cfg = tmp_path / "claude"
    cfg.mkdir(parents=True, exist_ok=True)
    settings = {"pluginConfigs": {"synapse@synapse": {"options": options}}}
    (cfg / "settings.json").write_text(json.dumps(settings), encoding="utf-8")


def _helper() -> ModuleType:
    _forget()  # a fresh config.py, so it reads the files this test just wrote
    return _load("synapse_mcp_headers_t", _SCRIPTS / "mcp_headers.py")


def _log(tmp_path) -> Path:
    return tmp_path / "synapse-home" / "logs" / "mcp_headers.log"


_HEADER_LINE = '{"Authorization": "Bearer ' + _TOKEN + '"}\n'


@pytest.mark.parametrize("source", ["config.json", "legacy-settings.json"])
def test_the_helper_prints_exactly_one_authorization_header(tmp_path, capsys, source):
    if source == "config.json":
        _home_config(tmp_path, url=_URL, token=_TOKEN)
    else:
        _legacy_options(tmp_path, SYNAPSE_URL=_URL, SYNAPSE_INGEST_TOKEN=_TOKEN)

    assert _helper().main(["--url", f"{_URL}/mcp"]) == 0

    captured = capsys.readouterr()
    # Claude Code merges helper headers over its static ones by exact key: the casing matters.
    assert captured.out == _HEADER_LINE
    assert captured.err == ""
    assert not _log(tmp_path).exists()


def test_the_target_comes_from_claude_codes_server_url_when_no_url_is_passed(
    tmp_path, capsys, monkeypatch
):
    _home_config(tmp_path, url=_URL, token=_TOKEN)
    monkeypatch.setenv("CLAUDE_CODE_MCP_SERVER_URL", "HTTPS://Synapse.Example.NET/mcp")

    assert _helper().main([]) == 0

    assert capsys.readouterr().out == _HEADER_LINE


def test_an_explicit_url_wins_over_claude_codes_server_url(tmp_path, capsys, monkeypatch):
    _home_config(tmp_path, url=_URL, token=_TOKEN)
    monkeypatch.setenv("CLAUDE_CODE_MCP_SERVER_URL", "https://elsewhere.example/mcp")

    assert _helper().main(["--url", f"{_URL}/mcp"]) == 0

    assert capsys.readouterr().out == _HEADER_LINE


def test_a_redacted_server_url_falls_back_to_the_plugins_url_option(tmp_path, capsys, monkeypatch):
    """Claude Code censors credential values inside CLAUDE_CODE_MCP_SERVER_URL; the URL
    it expanded from the plugin's SYNAPSE_URL option is what it was connecting to."""
    _home_config(tmp_path, url=_URL, token=_TOKEN)
    _legacy_options(tmp_path, SYNAPSE_URL=_URL)
    monkeypatch.setenv("CLAUDE_CODE_MCP_SERVER_URL", "https://REDACTED/mcp")

    assert _helper().main([]) == 0

    assert capsys.readouterr().out == _HEADER_LINE


@pytest.mark.parametrize(
    ("home", "options", "server_url", "argv", "reason"),
    [
        pytest.param(
            {"url": _URL, "token": _TOKEN},
            None,
            None,
            ["--url", "https://evil.example/mcp"],
            "differs from the saved server",
            id="foreign-host",
        ),
        pytest.param(
            {"url": _URL, "token": _TOKEN},
            None,
            None,
            ["--url", "http://synapse.example.net/mcp"],
            "differs from the saved server",
            id="foreign-scheme",
        ),
        pytest.param(
            {"url": _URL, "token": _TOKEN},
            None,
            None,
            ["--url", "https://synapse.example.net:8443/mcp"],
            "differs from the saved server",
            id="foreign-port",
        ),
        pytest.param(
            {"url": _URL, "token": _TOKEN},
            {"SYNAPSE_URL": "https://evil.example"},
            "https://REDACTED/mcp",
            [],
            "differs from the saved server",
            id="redacted-and-plugin-option-elsewhere",
        ),
        pytest.param(
            {"url": _URL},
            None,
            None,
            ["--url", f"{_URL}/mcp"],
            "credential missing",
            id="no-token",
        ),
        pytest.param({"url": _URL, "token": _TOKEN}, None, None, [], "no target URL", id="no-url"),
        pytest.param(
            {"url": _URL, "token": _TOKEN},
            None,
            "https://REDACTED/mcp",
            [],
            "no target URL",
            id="redacted-without-plugin-option",
        ),
    ],
)
def test_the_helper_fails_closed_and_logs_the_reason_never_the_token(
    tmp_path, capsys, monkeypatch, home, options, server_url, argv, reason
):
    _home_config(tmp_path, **home)
    if options:
        _legacy_options(tmp_path, **options)
    if server_url:
        monkeypatch.setenv("CLAUDE_CODE_MCP_SERVER_URL", server_url)

    assert _helper().main(argv) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    log = _log(tmp_path).read_text(encoding="utf-8")
    assert reason in log and reason in captured.err
    for text in (log, captured.err):
        assert _TOKEN not in text and _TOKEN[:6] not in text and _TOKEN[-6:] not in text


def test_the_failure_log_is_restarted_once_it_passes_its_cap(tmp_path, capsys):
    _home_config(tmp_path, url=_URL, token=_TOKEN)
    _log(tmp_path).parent.mkdir(parents=True)
    _log(tmp_path).write_text("x" * (64 * 1024 + 1), encoding="utf-8")

    assert _helper().main([]) == 1

    lines = _log(tmp_path).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and "no target URL" in lines[0]


def _run_cleared(script: Path, env: dict, *args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), *args],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


@pytest.mark.parametrize("where", ["SYNAPSE_HOME", "default-home"])
def test_config_json_alone_works_in_codexs_cleared_environment(tmp_path, where):
    """A real subprocess whose environment holds only HOME, PATH (and SYNAPSE_HOME): no
    token, no URL, no Claude settings anywhere."""
    home = tmp_path / "home"
    env = {"HOME": str(home), "PATH": os.environ.get("PATH", "")}
    if where == "SYNAPSE_HOME":
        env["SYNAPSE_HOME"] = str(tmp_path / "synapse-home")
        _home_config(tmp_path, url=_URL, token=_TOKEN)
    else:
        (home / ".synapse").mkdir(parents=True)
        (home / ".synapse" / "config.json").write_text(
            json.dumps({"url": _URL, "token": _TOKEN}), encoding="utf-8"
        )

    proc = _run_cleared(_SCRIPTS / "mcp_headers.py", env, "--url", f"{_URL}/mcp", cwd=tmp_path)

    assert (proc.returncode, proc.stdout, proc.stderr) == (0, _HEADER_LINE, "")


@pytest.mark.parametrize(
    ("url", "code", "out"),
    [
        pytest.param(f"{_URL}/mcp", 0, _HEADER_LINE, id="match"),
        pytest.param("https://evil.example/mcp", 1, "", id="foreign"),
    ],
)
def test_the_legacy_codex_shim_behaves_like_the_shared_helper(tmp_path, url, code, out):
    _home_config(tmp_path, url=_URL, token=_TOKEN)
    env = {
        "HOME": str(tmp_path / "home"),
        "PATH": os.environ.get("PATH", ""),
        "SYNAPSE_HOME": str(tmp_path / "synapse-home"),
    }

    proc = _run_cleared(_CODEX_SCRIPTS / "mcp_headers.py", env, "--url", url, cwd=tmp_path)

    assert (proc.returncode, proc.stdout) == (code, out)
    assert _TOKEN not in proc.stderr
