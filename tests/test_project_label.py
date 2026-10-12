"""One project identity: the client resolver and the server fallback agree.

Before this, a project's label was derived four ways, and the two server parsers split
the transcript's cwd on "/" only, so a Windows session (``C:\\Users\\pat\\Acme\\``) was
filed under its whole path, a lowercase-drive variant under another, and every
subfolder became a project of its own. Three things are pinned here:

1. ``label_from_path`` — the client copy (``plugin/scripts/synapse_project.py``) and the
   server copy (``ingestion/project_label.py``) give the same answer for ANY string:
   either separator, trailing separators dropped, the last component with its case,
   nothing for a bare drive, ``.``, ``..`` or an empty path, capped at 200 characters.
   ``clean_label`` (what /ingest makes of a client-sent ``project``) is the same rule
   after dropping control characters, and refuses anything that is not a string.
2. ``synapse_project.resolve`` against REAL git layouts built with ``git init`` and
   ``git worktree add`` in tmp_path: a ``.synapse-project`` marker (walking up to the git
   toplevel, never above it; only the directory itself outside a repo), else the git
   toplevel's name (a linked worktree is its main checkout, a submodule is itself),
   else the directory's name; a path that does not exist here (a Windows path, a
   deleted folder) is just its normalised basename; ``project_label`` never raises.
3. The two server parsers label a Windows cwd by its basename, and an explicit
   ``project_override`` (the label the client sent) still wins.

Pure: no DB, no network. The client module is stdlib-only and loaded by path.
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import shutil
import subprocess
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import ModuleType

import pytest

from ingestion import codex_client, jsonl_client
from ingestion import project_label as server
from ingestion.codex_client import CodexRolloutParser
from ingestion.jsonl_client import JSONLParser

_REPO = Path(__file__).resolve().parents[1]
_CLIENT_PATH = _REPO / "plugin" / "scripts" / "synapse_project.py"


def _load_client() -> ModuleType:
    """The client resolver, loaded by path under a private name so it never shadows the
    ``synapse_project`` the hooks import off their own sys.path."""
    spec = importlib.util.spec_from_file_location("_synapse_project_under_test", _CLIENT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


client = _load_client()

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


# ---------------------------------------------------------------------------
# label_from_path: one rule, two copies
# ---------------------------------------------------------------------------

_LONG = "a" * 300


def _short_id(value: object) -> str:
    text = repr(value)
    return text if len(text) <= 40 else f"{text[:30]}...len={len(str(value))}"


_LONG_UNICODE = "é" * 250

# (input, expected label). Every row is also checked client == server.
_LABEL_TABLE: list[tuple[object, str | None]] = [
    # posix
    ("/home/dev/acme", "acme"),
    ("/home/dev/acme/", "acme"),
    ("/home/dev/acme///", "acme"),
    ("acme", "acme"),
    ("MyRepo", "MyRepo"),
    ("/home/dev/MyRepo", "MyRepo"),
    ("/home/dev/myrepo", "myrepo"),
    ("/home/dev/acme.v2", "acme.v2"),
    ("/home/dev/.dotfiles", ".dotfiles"),
    # Windows: the separator the old parsers did not know about
    ("C:\\Users\\pat\\Acme", "Acme"),
    ("C:\\Users\\pat\\Acme\\", "Acme"),
    ("c:\\Users\\pat\\Acme\\", "Acme"),
    ("c:\\users\\pat\\Acme", "Acme"),
    ("C:/Users/pat/Acme", "Acme"),
    ("D:\\src\\acme-api\\\\", "acme-api"),
    ("\\\\fileserver\\share\\Acme", "Acme"),
    ("\\\\?\\C:\\Users\\pat\\Acme", "Acme"),
    # mixed separators
    ("C:\\Users/pat\\Acme/", "Acme"),
    ("C:\\Users\\pat\\Acme\\/\\", "Acme"),
    ("/home/dev\\acme", "acme"),
    # nothing to name
    (None, None),
    ("", None),
    ("/", None),
    ("//", None),
    ("\\", None),
    ("\\\\", None),
    ("C:", None),
    ("c:", None),
    ("Z:", None),
    ("C:\\", None),
    ("C:/", None),
    ("c:\\\\", None),
    (".", None),
    ("..", None),
    ("./", None),
    ("../", None),
    ("/home/dev/..", None),
    ("C:\\work\\.", None),
    ("C:\\work\\..\\", None),
    ("   ", None),
    ("/home/dev/   ", None),
    # spaces are kept inside a name, trimmed around it
    ("/home/dev/My Repo", "My Repo"),
    ("C:\\Users\\pat\\My Repo\\", "My Repo"),
    ("  /home/dev/acme  ", "acme"),
    ("C:\\Users\\pat\\Acme \\", "Acme"),
    # unicode, case kept
    ("/home/dev/Projekt-Ä", "Projekt-Ä"),
    ("/home/dev/日本語のリポ/", "日本語のリポ"),
    ("C:\\Users\\Zoë\\café", "café"),
    ("/home/dev/rocket-🚀", "rocket-🚀"),
    # capped at 200 characters (not bytes)
    (_LONG, "a" * 200),
    (f"/home/dev/{_LONG}/", "a" * 200),
    (f"C:\\Users\\pat\\{_LONG_UNICODE}\\", "é" * 200),
    ("b" * 200, "b" * 200),
    # path objects (the client accepts them; the server stringifies the same way)
    (PurePosixPath("/home/dev/Acme"), "Acme"),
    (PureWindowsPath("C:/Users/pat/Acme"), "Acme"),
]


@pytest.mark.parametrize(("raw", "expected"), _LABEL_TABLE, ids=_short_id)
def test_client_and_server_label_from_path_agree(raw, expected):
    assert client.label_from_path(raw) == expected
    assert server.label_from_path(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), _LABEL_TABLE, ids=_short_id)
def test_both_server_parsers_use_the_same_fallback(raw, expected):
    """The fallback for a body without ``project`` (catch-up, old plugins) is the same
    function in both parsers, so a Claude Code and a Codex session in one folder land
    under one label."""
    assert jsonl_client._cwd_to_project(raw) == expected
    assert codex_client._cwd_to_project(raw) == expected


def test_the_old_split_is_what_this_replaces():
    """Reproduces the production failure: splitting on "/" alone leaves a Windows cwd
    whole, and the lowercase-drive variant is a different label again."""
    upper, lower = "C:\\Users\\pat\\Acme\\", "c:\\Users\\pat\\Acme"

    def old(cwd):
        return cwd.rstrip("/").rsplit("/", 1)[-1] or None

    assert old(upper) == upper and old(lower) == lower and old(upper) != old(lower)
    assert server.label_from_path(upper) == server.label_from_path(lower) == "Acme"


def test_drive_pattern_and_cap_are_the_same_constants():
    assert client._MAX_LABEL == server._MAX_LABEL == 200
    assert client._DRIVE.pattern == server._DRIVE.pattern


# Characters chosen to hit every branch: both separators, drive colons, dots, spaces,
# a comment sign, non-ASCII letters, a no-break space and a control character.
_FUZZ_ALPHABET = ["/", "\\", ":", ".", " ", "C", "c", "a", "Z", "é", "日", "#", "-", "\t", "\u00a0"]


def _fuzz_strings(n: int = 4000, seed: int = 20261012) -> list[str]:
    rng = random.Random(seed)
    return ["".join(rng.choices(_FUZZ_ALPHABET, k=rng.randint(0, 12))) for _ in range(n)]


def test_client_and_server_agree_on_random_strings():
    disagreements = [
        s for s in _fuzz_strings() if client.label_from_path(s) != server.label_from_path(s)
    ]
    assert disagreements == []


def test_a_label_is_always_a_bare_trimmed_name():
    """Whatever comes in, a label is None or one path component: no separator, not a
    bare drive, not ``.``/``..``, no surrounding whitespace. That is what lets skill
    scopes be ``project:<label>`` and allowlists hold bare names."""
    for s in _fuzz_strings():
        label = server.label_from_path(s)
        if label is None:
            continue
        assert label, s
        assert "/" not in label and "\\" not in label, (s, label)
        assert label not in (".", "..") and not server._DRIVE.match(label), (s, label)
        assert label == label.strip(), (s, label)


# ---------------------------------------------------------------------------
# clean_label: what /ingest makes of a client-sent "project"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [None, 42, 3.5, True, b"Acme", bytearray(b"Acme"), ["Acme"], {"project": "Acme"}, object()],
    ids=lambda v: type(v).__name__,
)
def test_clean_label_refuses_non_strings(raw):
    assert server.clean_label(raw) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Acme", "Acme"),
        ("acme", "acme"),
        ("ACME", "ACME"),
        ("  Acme  ", "Acme"),
        ("Acme\n", "Acme"),
        ("Acme\r\n", "Acme"),
        ("\x00Acme\x00", "Acme"),
        ("Ac\x1bme", "Acme"),
        ("Ac\tme", "Acme"),
        ("\x7fAcme", "Acme"),
        ("C:\\Users\\pat\\Ac\x00me", "Acme"),
        # a path from an older or hand-rolled client is cut to its last component
        ("/home/dev/Acme", "Acme"),
        ("/home/dev/Acme/", "Acme"),
        ("C:\\Users\\pat\\Acme\\", "Acme"),
        ("c:\\users\\pat\\Acme", "Acme"),
        ("C:/Users/pat/My Repo", "My Repo"),
        ("/home/dev/日本語", "日本語"),
        # empty after cleaning
        ("", None),
        ("   ", None),
        ("\x00\x01\x1f", None),
        ("\n\t", None),
        ("C:", None),
        ("C:\\", None),
        ("/", None),
        (".", None),
        ("..", None),
        # capped like any label
        (_LONG, "a" * 200),
    ],
    ids=repr,
)
def test_clean_label(raw, expected):
    assert server.clean_label(raw) == expected


def test_clean_label_is_label_from_path_after_dropping_control_characters():
    for s in _fuzz_strings():
        stripped = "".join(ch for ch in s if ch >= " " and ch != "\x7f")
        assert server.clean_label(s) == client.label_from_path(stripped), repr(s)
        label = server.clean_label(s)
        if label is not None:
            assert not any(ch < " " or ch == "\x7f" for ch in label), repr(label)


# ---------------------------------------------------------------------------
# synapse_project.resolve on real layouts
# ---------------------------------------------------------------------------


def _git_env() -> dict[str, str]:
    """A git that sees none of the developer's (or CI's) config, hooks or GIT_* vars."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_NOSYSTEM="1",
        GIT_AUTHOR_NAME="Test Author",
        GIT_AUTHOR_EMAIL="author@example.invalid",
        GIT_COMMITTER_NAME="Test Author",
        GIT_COMMITTER_EMAIL="author@example.invalid",
    )
    return env


def _git(cwd: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-c", "init.defaultBranch=main", *args],
        cwd=cwd,
        env=_git_env(),
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def _init_repo(path: Path, files: dict[str, str] | None = None) -> Path:
    """``git init`` + one commit. ``files`` (relative path -> text) are committed."""
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    for rel, text in {"README.md": "x\n", **(files or {})}.items():
        (path / rel).parent.mkdir(parents=True, exist_ok=True)
        (path / rel).write_text(text, encoding="utf-8")
        _git(path, "add", rel)
    _git(path, "commit", "-q", "-m", "init")
    return path


@pytest.fixture()
def ws(tmp_path: Path) -> Path:
    """A scratch workspace that is in no repository (the "no repo" cases depend on it)."""
    root = tmp_path.resolve()
    if any((d / ".git").exists() for d in (root, *root.parents)):
        pytest.skip(f"{root} is inside a git repository; the no-repo cases cannot run here")
    return root


def _marker(folder: Path, text: str | bytes) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / client.MARKER
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


@needs_git
class TestGitToplevel:
    def test_repo_root_is_its_own_name(self, ws):
        repo = _init_repo(ws / "MyRepo")
        assert client.resolve(repo) == ("MyRepo", repo)
        assert client.resolve(str(repo)) == ("MyRepo", repo)
        assert client.project_label(str(repo)) == "MyRepo"

    def test_nested_subfolder_is_the_repo_not_a_project_of_its_own(self, ws):
        repo = _init_repo(ws / "MyRepo")
        deep = repo / "tickets" / "dev" / "src"
        deep.mkdir(parents=True)
        assert client.resolve(deep) == ("MyRepo", repo)
        assert client.resolve(str(deep) + "/") == ("MyRepo", repo)

    def test_case_of_the_folder_is_kept(self, ws):
        repo = _init_repo(ws / "acme_API")
        assert client.project_label(repo / ".") == "acme_API"

    def test_relative_and_default_cwd_use_the_process_cwd(self, ws, monkeypatch):
        repo = _init_repo(ws / "alpha")
        (repo / "pkg").mkdir()
        monkeypatch.chdir(repo)
        assert client.resolve() == ("alpha", repo)
        assert client.resolve(None) == ("alpha", repo)
        assert client.resolve("pkg") == ("alpha", repo)
        assert client.project_label() == "alpha"

    def test_symlink_into_a_repo_is_the_repo(self, ws):
        repo = _init_repo(ws / "alpha")
        (repo / "src").mkdir()
        link = ws / "shortcut"
        link.symlink_to(repo / "src", target_is_directory=True)
        assert client.resolve(link) == ("alpha", repo)

    def test_nested_repo_is_its_own_project(self, ws):
        outer = _init_repo(ws / "outer")
        inner = _init_repo(outer / "vendor" / "inner")
        assert client.resolve(inner / ".") == ("inner", inner)
        assert client.resolve(outer / "vendor") == ("outer", outer)


@needs_git
class TestMarker:
    def test_marker_at_repo_root_is_picked_up_from_a_subfolder(self, ws):
        repo = _init_repo(ws / "acme-checkout-2")
        _marker(repo, "Acme\n")
        sub = repo / "src" / "lib"
        sub.mkdir(parents=True)
        assert client.resolve(sub) == ("Acme", repo)
        assert client.resolve(repo) == ("Acme", repo)

    def test_marker_case_is_kept(self, ws):
        repo = _init_repo(ws / "acme")
        _marker(repo, "ACME")
        assert client.project_label(repo) == "ACME"

    def test_nearest_marker_wins_over_one_higher_up(self, ws):
        repo = _init_repo(ws / "mono")
        _marker(repo, "Mono")
        pkg = repo / "packages" / "billing"
        _marker(pkg, "Billing")
        deep = pkg / "src"
        deep.mkdir()
        assert client.resolve(deep) == ("Billing", pkg)
        assert client.resolve(pkg) == ("Billing", pkg)
        assert client.resolve(repo / "packages") == ("Mono", repo)

    def test_marker_below_cwd_is_not_used(self, ws):
        repo = _init_repo(ws / "mono")
        _marker(repo / "packages" / "billing", "Billing")
        assert client.resolve(repo) == ("mono", repo)

    def test_marker_above_the_git_toplevel_is_not_used(self, ws):
        _marker(ws / "workspace", "Workspace")
        repo = _init_repo(ws / "workspace" / "alpha")
        (repo / "sub").mkdir()
        assert client.resolve(repo) == ("alpha", repo)
        assert client.resolve(repo / "sub") == ("alpha", repo)
        # ...but it does name the non-repo folder it sits in
        assert client.resolve(ws / "workspace") == ("Workspace", ws / "workspace")

    def test_marker_with_bom_comments_blank_lines_and_a_path(self, ws):
        repo = _init_repo(ws / "checkout")
        text = "# Synapse project id for this checkout\n\n   \n  # indented comment\n  C:\\work\\Acme\\  \nOther\n"
        _marker(repo, text.encode("utf-8-sig"))
        assert client.resolve(repo / ".") == ("Acme", repo)

    def test_bom_is_not_part_of_the_label(self, ws):
        repo = _init_repo(ws / "checkout")
        _marker(repo, "\ufeffAcme\r\n".encode())
        label = client.project_label(repo)
        assert label == "Acme"
        assert "\ufeff" not in label

    def test_posix_path_in_a_marker_is_its_basename(self, ws):
        repo = _init_repo(ws / "checkout")
        _marker(repo, "/srv/projects/Acme/\n")
        assert client.project_label(repo) == "Acme"

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "\n",
            "   \n\t\n",
            "# only a comment\n",
            "#a\n#b\n\n",
            ".\n",
            "/\n",
            "C:\\\n",
            "\ufeff\n",
        ],
        ids=repr,
    )
    def test_empty_marker_is_ignored_and_falls_through_to_git(self, ws, text):
        repo = _init_repo(ws / "alpha")
        (repo / "sub").mkdir()
        _marker(repo, text.encode("utf-8"))
        assert client.resolve(repo / "sub") == ("alpha", repo)

    def test_empty_nearer_marker_falls_through_to_the_one_higher_up(self, ws):
        repo = _init_repo(ws / "mono")
        _marker(repo, "Mono")
        _marker(repo / "pkg", "   \n# nothing here\n")
        assert client.resolve(repo / "pkg") == ("Mono", repo)

    def test_a_directory_named_like_the_marker_is_not_a_marker(self, ws):
        repo = _init_repo(ws / "alpha")
        (repo / client.MARKER).mkdir()
        assert client.resolve(repo) == ("alpha", repo)

    def test_committed_marker_names_every_worktree_too(self, ws):
        main = _init_repo(ws / "acme-main", files={client.MARKER: "Acme\n"})
        wt = ws / "acme-feature"
        _git(main, "worktree", "add", "-q", str(wt))
        assert client.resolve(main) == ("Acme", main)
        assert client.resolve(wt) == ("Acme", wt)


class TestMarkerOutsideAnyRepo:
    def test_marker_applies_in_the_folder_itself(self, ws):
        outer = ws / "outer"
        _marker(outer, "Outer")
        assert client.resolve(outer) == ("Outer", outer)

    def test_marker_in_a_parent_is_not_used_outside_a_repo(self, ws):
        """Without a repository there is no boundary to walk up to, so a marker in
        some parent (a home directory, a Downloads folder) must not swallow every folder
        beneath it."""
        _marker(ws / "outer", "Outer")
        inner = ws / "outer" / "inner"
        inner.mkdir()
        assert client.resolve(inner) == ("inner", inner)


class TestNoRepo:
    def test_plain_folder_is_its_own_name(self, ws):
        folder = ws / "notes" / "Scratch Pad"
        folder.mkdir(parents=True)
        assert client.resolve(folder) == ("Scratch Pad", folder)
        assert client.resolve(str(folder) + "//") == ("Scratch Pad", folder)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("/no/such/dir/Acme/", "Acme"),
            ("C:\\Users\\pat\\Acme\\", "Acme"),
            ("c:\\users\\pat\\Acme", "Acme"),
            ("C:\\Users/pat\\Acme/", "Acme"),
            ("C:\\", None),
        ],
        ids=repr,
    )
    def test_path_that_does_not_exist_here_is_its_basename(self, ws, monkeypatch, raw, expected):
        """A transcript from another machine (or a Windows path on this one): there is
        nothing to look at, so the label is the string's basename, root None — the same
        answer the server's fallback gives."""
        monkeypatch.chdir(ws)  # a relative Windows "path" must not resolve to anything here
        assert client.resolve(raw) == (expected, None)
        assert client.project_label(raw) == expected == server.label_from_path(raw)


@needs_git
class TestWorktrees:
    def test_sibling_worktree_is_the_main_checkout(self, ws):
        main = _init_repo(ws / "alpha")
        wt = ws / "alpha-feature-x"
        _git(main, "worktree", "add", "-q", str(wt))
        assert (wt / ".git").is_file()  # a real linked worktree, made by git
        (wt / "src").mkdir()
        assert client.resolve(wt) == ("alpha", wt)
        assert client.resolve(wt / "src") == ("alpha", wt)

    def test_claude_code_worktree_under_dot_claude_is_the_main_checkout(self, ws):
        main = _init_repo(ws / "MyRepo")
        wt = main / ".claude" / "worktrees" / "agent-7f3c2a"
        _git(main, "worktree", "add", "-q", str(wt))
        assert (wt / ".git").is_file()
        (wt / "tests").mkdir()
        assert client.resolve(wt) == ("MyRepo", wt)
        assert client.resolve(wt / "tests") == ("MyRepo", wt)
        # and the main checkout itself is unchanged
        assert client.resolve(main) == ("MyRepo", main)

    def test_worktree_on_a_detached_head_is_the_main_checkout(self, ws):
        main = _init_repo(ws / "alpha")
        wt = ws / "detached"
        _git(main, "worktree", "add", "-q", "--detach", str(wt))
        assert client.project_label(wt) == "alpha"

    def test_hand_made_relative_gitdir_and_commondir(self, ws):
        """``worktree.useRelativePaths`` (newer git) writes both pointers relative."""
        main = ws / "acme"
        wtdir = main / ".git" / "worktrees" / "wt1"
        wtdir.mkdir(parents=True)
        (wtdir / "commondir").write_text("../..\n")
        wt = ws / "elsewhere" / "wt1"
        wt.mkdir(parents=True)
        (wt / ".git").write_text("gitdir: ../../acme/.git/worktrees/wt1\n")
        assert client.resolve(wt) == ("acme", wt)

    def test_hand_made_absolute_commondir(self, ws):
        main = ws / "acme"
        wtdir = ws / "gitdirs" / "wt1"
        wtdir.mkdir(parents=True)
        (main / ".git").mkdir(parents=True)
        (wtdir / "commondir").write_text(f"{main / '.git'}\n")
        wt = ws / "wt1"
        wt.mkdir()
        (wt / ".git").write_text(f"gitdir: {wtdir}\n")
        assert client.resolve(wt) == ("acme", wt)

    @pytest.mark.parametrize("where", ["dot-claude", "sibling"])
    def test_untracked_marker_in_the_main_checkout_names_its_worktrees(self, ws, where):
        main = _init_repo(ws / "acme-main")
        _marker(main, "Acme\n")  # untracked: not in the worktree's checkout
        wt = (
            main / ".claude" / "worktrees" / "agent-1"
            if where == "dot-claude"
            else ws / "acme-feature"
        )
        _git(main, "worktree", "add", "-q", str(wt))
        assert not (wt / client.MARKER).exists()
        assert client.project_label(main) == "Acme"
        assert client.project_label(wt) == client.project_label(main)


@needs_git
class TestSubmodules:
    def test_real_submodule_is_its_own_project(self, ws):
        lib = _init_repo(ws / "upstream-lib")
        main = _init_repo(ws / "app")
        _git(
            main,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            str(lib),
            "vendor/lib",
        )
        sub = main / "vendor" / "lib"
        assert (sub / ".git").is_file()
        assert client.resolve(sub) == ("lib", sub)
        assert client.resolve(main / "vendor") == ("app", main)

    def test_hand_made_submodule_gitdir_without_commondir_is_itself(self, ws):
        main = _init_repo(ws / "app")
        modules = main / ".git" / "modules" / "widgets"
        modules.mkdir(parents=True)
        (modules / "HEAD").write_text("ref: refs/heads/main\n")
        sub = main / "third_party" / "widgets"
        sub.mkdir(parents=True)
        (sub / ".git").write_text("gitdir: ../../.git/modules/widgets\n")
        (sub / "src").mkdir()
        assert client.resolve(sub / "src") == ("widgets", sub)


class TestBrokenGitFiles:
    def test_dot_git_file_that_is_not_a_gitdir_pointer(self, ws):
        folder = ws / "odd"
        folder.mkdir()
        (folder / ".git").write_text("not a pointer\n")
        assert client.resolve(folder) == ("odd", folder)

    def test_empty_dot_git_file(self, ws):
        folder = ws / "odd"
        folder.mkdir()
        (folder / ".git").write_text("")
        assert client.resolve(folder) == ("odd", folder)

    def test_gitdir_pointing_nowhere(self, ws):
        folder = ws / "odd"
        folder.mkdir()
        (folder / ".git").write_text("gitdir: /no/such/place/.git/worktrees/x\n")
        assert client.resolve(folder) == ("odd", folder)


class TestNeverRaises:
    @pytest.mark.parametrize(
        "raw",
        [5, 3.5, object(), b"/home/dev/Acme", ["a"], {"cwd": "/x"}, "a\x00b", "\x00"],
        ids=["int", "float", "object", "bytes", "list", "dict", "nul-inside", "nul-only"],
    )
    def test_odd_inputs(self, raw):
        label = client.project_label(raw)
        assert label is None or isinstance(label, str)

    def test_falsy_input_means_the_process_cwd(self, ws, monkeypatch):
        folder = ws / "alpha"
        folder.mkdir()
        monkeypatch.chdir(folder)
        for raw in (None, "", 0):
            assert client.project_label(raw) == "alpha"

    def test_deleted_process_cwd(self, ws, monkeypatch):
        gone = ws / "gone"
        gone.mkdir()
        monkeypatch.chdir(gone)
        gone.rmdir()
        assert client.project_label() is None
        assert client.project_label(None) is None
        assert client.project_label(str(gone)) == "gone"
        assert client.project_label("relative-name") == "relative-name"
        assert client.resolve(str(gone)) == ("gone", None)

    @needs_git
    def test_folder_deleted_inside_a_repo(self, ws):
        repo = _init_repo(ws / "alpha")
        sub = repo / "tmpdir"
        sub.mkdir()
        sub.rmdir()
        # it does not exist any more: the repository its surviving parent is in names it
        assert client.resolve(sub) == ("alpha", None)


@pytest.mark.parametrize(
    ("encoding", "bom"),
    [("utf-16-le", b"\xff\xfe"), ("utf-16-be", b"\xfe\xff")],
    ids=["utf-16-le", "utf-16-be"],
)
def test_utf16_marker_written_by_windows_powershell(ws, encoding, bom):
    folder = ws / "acme-checkout"
    data = bom + "Acme\r\n".encode(encoding)
    _marker(folder, data)
    label = client.project_label(folder)
    assert label in ("Acme", "acme-checkout"), repr(label)


# ---------------------------------------------------------------------------
# The server parsers (fallback for bodies without a "project")
# ---------------------------------------------------------------------------

_WINDOWS_CWDS = [
    "C:\\Users\\pat\\Acme\\",
    "C:\\Users\\pat\\Acme",
    "c:\\Users\\pat\\Acme",
    "c:\\users\\pat\\Acme\\",
    "C:/Users/pat/Acme/",
]


def _claude_records(cwd: str | None) -> list[dict]:
    def rec(rtype: str, uuid: str, message: dict, ts: str) -> dict:
        r = {
            "type": rtype,
            "message": message,
            "uuid": uuid,
            "sessionId": "sess-win",
            "timestamp": ts,
        }
        if cwd is not None:
            r["cwd"] = cwd
        return r

    return [
        rec(
            "user", "u1", {"role": "user", "content": "list the files"}, "2026-10-01T10:00:00.000Z"
        ),
        rec(
            "assistant",
            "a1",
            {"model": "claude-sonnet-4-6", "content": [{"type": "text", "text": "Here they are."}]},
            "2026-10-01T10:00:01.000Z",
        ),
        rec(
            "user",
            "u2",
            {"role": "user", "content": "now run the tests"},
            "2026-10-01T10:01:00.000Z",
        ),
        rec(
            "assistant",
            "a2",
            {"model": "claude-sonnet-4-6", "content": [{"type": "text", "text": "All green."}]},
            "2026-10-01T10:01:05.000Z",
        ),
    ]


_CODEX_SID = "019ffdbe-ff9a-7b41-892a-ddc6221d1a64"


def _codex_rec(rtype: str, payload: dict, ts: str = "2026-10-01T10:00:00.000Z") -> dict:
    return {"timestamp": ts, "type": rtype, "payload": payload}


def _codex_msg(role: str, text: str, msg_id: str, ts: str = "2026-10-01T10:00:00.000Z") -> dict:
    block = "output_text" if role == "assistant" else "input_text"
    return _codex_rec(
        "response_item",
        {"type": "message", "id": msg_id, "role": role, "content": [{"type": block, "text": text}]},
        ts,
    )


def _codex_records(meta_cwd: str, turn_cwd: str) -> list[dict]:
    return [
        _codex_rec(
            "session_meta",
            {
                "session_id": _CODEX_SID,
                "cwd": meta_cwd,
                "originator": "codex-tui",
                "cli_version": "0.147.0",
            },
        ),
        _codex_msg("user", "hi", "msg_u1"),
        _codex_msg("assistant", "Hello! How can I help?", "msg_a1"),
        _codex_rec("turn_context", {"turn_id": "t2", "cwd": turn_cwd}),
        _codex_msg("user", "run the tests", "msg_u2", ts="2026-10-01T10:01:00.000Z"),
        _codex_msg("assistant", "All green.", "msg_a2", ts="2026-10-01T10:01:05.000Z"),
    ]


class TestJSONLParser:
    @pytest.mark.parametrize("cwd", _WINDOWS_CWDS, ids=repr)
    def test_windows_cwd_is_labelled_by_its_basename(self, cwd):
        eps = JSONLParser().parse_records(_claude_records(cwd), "push")
        assert len(eps) == 2
        assert {e.project for e in eps} == {"Acme"}

    def test_explicit_project_override_wins(self):
        eps = JSONLParser().parse_records(
            _claude_records("C:\\Users\\pat\\Acme\\"), "push", project_override="acme-platform"
        )
        assert {e.project for e in eps} == {"acme-platform"}

    def test_override_wins_without_a_cwd_and_none_without_either(self):
        assert {e.project for e in JSONLParser().parse_records(_claude_records(None), "p")} == {
            None
        }
        eps = JSONLParser().parse_records(_claude_records(None), "p", project_override="Acme")
        assert {e.project for e in eps} == {"Acme"}

    def test_parse_file_labels_the_same_way(self, tmp_path):
        path = tmp_path / "session.jsonl"
        path.write_text(
            "".join(json.dumps(r) + "\n" for r in _claude_records("c:\\Users\\pat\\Acme"))
        )
        assert {e.project for e in JSONLParser().parse_file(path)} == {"Acme"}


class TestCodexParser:
    @pytest.mark.parametrize("cwd", _WINDOWS_CWDS, ids=repr)
    def test_windows_cwd_is_labelled_by_its_basename(self, cwd):
        eps = CodexRolloutParser().parse_records(_codex_records(cwd, cwd), "push")
        assert len(eps) == 2
        assert {e.project for e in eps} == {"Acme"}
        # the raw cwd is still kept for provenance
        assert {e.metadata["cwd"] for e in eps} == {cwd}

    def test_upper_and_lower_drive_in_one_rollout_are_one_label(self):
        records = _codex_records("C:\\Users\\pat\\Acme\\", "c:\\Users\\pat\\Acme")
        eps = CodexRolloutParser().parse_records(records, "push")
        assert {e.project for e in eps} == {"Acme"}

    def test_explicit_project_override_wins(self):
        records = _codex_records("C:\\Users\\pat\\Acme\\", "c:\\Users\\pat\\Acme")
        eps = CodexRolloutParser().parse_records(records, "push", project_override="acme-platform")
        assert {e.project for e in eps} == {"acme-platform"}

    def test_pushed_tail_without_session_meta(self):
        """A pushed tail usually has only a turn_context: it still carries the cwd."""
        tail = _codex_records("C:\\Users\\pat\\Acme\\", "c:\\Users\\pat\\Acme\\")[1:]
        eps = CodexRolloutParser().parse_records(tail, "push", session_id_hint=_CODEX_SID)
        assert eps and {e.project for e in eps} == {"Acme"}


def test_worktree_of_a_bare_repository_is_named_for_the_repository(ws):
    """A bare repository has no main checkout to name its worktrees, so they are all
    named for the repository itself (``acme.git`` -> ``acme``), not for their folders."""
    src = _init_repo(ws / "src")
    _git(ws, "clone", "-q", "--bare", str(src), str(ws / "acme.git"))
    _git(ws / "acme.git", "worktree", "add", "-q", str(ws / "feature-x"))
    _git(ws / "acme.git", "worktree", "add", "-q", "--detach", str(ws / "other"))
    assert client.resolve(ws / "feature-x") == ("acme", ws / "feature-x")
    assert client.project_label(ws / "other") == "acme"


def test_a_folder_that_is_gone_takes_its_surviving_repository_label(ws):
    """The catch-up sweep ships transcripts after their folder is gone: a removed linked
    worktree or a deleted subfolder is labelled by the repository its nearest surviving
    parent is in, not by its own (often throwaway) folder name."""
    main = _init_repo(ws / "acme-main")
    wt = main / ".claude" / "worktrees" / "feat-1"
    _git(main, "worktree", "add", "-q", str(wt))
    (wt / "sub").mkdir()
    assert client.project_label(wt / "sub") == "acme-main"
    _git(main, "worktree", "remove", "--force", str(wt))
    assert not wt.exists()
    assert client.resolve(wt / "sub") == ("acme-main", None)
    assert client.project_label(main / "deleted" / "deeper") == "acme-main"
    # Outside any repository nothing is borrowed from a parent: just the folder's name.
    assert client.project_label(ws / "loose" / "gone") == "gone"
    # A path from another machine (relative here: a Windows path on POSIX) is its basename.
    assert client.project_label("C:\\Users\\pat\\acme-main\\gone") == "gone"
