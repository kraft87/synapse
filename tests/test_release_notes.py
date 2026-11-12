"""Client releases (scripts/release_notes.py): the version, the client file set, the
notes, and the two decision tables CI runs (release.yml's release, ci.yml's Version).

* versions: only a plain MAJOR.MINOR.PATCH of ASCII digits gets through, and plugin.json
  and synapse_client.__version__ must agree
* notes: pinned against the real HANDOFF.md for each heading form it holds (first entry,
  a two-item parenthetical, an entry in the middle, the last entry), plus synthetic cases
  for what the real file does not have (two entries for one version, schema-first order,
  a fenced ``## ``); entries whose heading names no plugin version are never picked; the
  body is capped to GitHub's limit
* the client file set is the one /install bundles (pinned against install_routes on the
  real tree), Markdown included: plugin/commands/*.md are slash commands
* release and check-pr against real scratch git repositories, with gh replaced by a
  recorder: what is called, with which tag, target and notes, and what is refused before
  any call

Stdlib + pytest + git. Nothing touches the network or GitHub.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "release_notes", _REPO / "scripts" / "release_notes.py"
)
rn = importlib.util.module_from_spec(_SPEC)
sys.modules["release_notes"] = rn
_SPEC.loader.exec_module(rn)

_REAL = (_REPO / "HANDOFF.md").read_text(encoding="utf-8")

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


@pytest.fixture(autouse=True)
def _plain_git(monkeypatch):
    """The script's git sees none of the developer's (or CI's) config."""
    for key in [k for k in os.environ if k.startswith("GIT_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["0.24.4", "0.0.0", "10.20.30", "123456789.0.1"])
def test_a_plain_version_passes(value):
    assert rn.check_version(value) == value


@pytest.mark.parametrize(
    "value",
    [
        "",
        "0.24",
        "0.24.4.1",
        "v0.24.4",
        "0.24.4 ",
        " 0.24.4",
        "0.24.4\n",
        "0.24.4-rc1",
        "0.24.4+build",
        "01.2.3",
        "0.24.04",
        "0.24.x",
        "1234567890.0.0",
        chr(0x661) + "." + chr(0x662) + "." + chr(0x663),  # Arabic-Indic: \d takes them
        "0.24.1" + chr(0x663),
        "0.24.4;rm -rf /",
        "$(id)",
        "0.24.4`id`",
        24,
        0.24,
        None,
        ["0.24.4"],
    ],
)
def test_anything_else_is_refused(value):
    with pytest.raises(rn.ReleaseError, match="is not a version"):
        rn.check_version(value)


def test_a_hostile_value_is_shortened_in_the_message():
    with pytest.raises(rn.ReleaseError) as err:
        rn.check_version("9" * 500)
    assert len(str(err.value)) < 120


def test_versions_compare_as_numbers():
    assert rn.parse_version("0.10.0") > rn.parse_version("0.9.9")
    assert rn.parse_version("0.24.10") > rn.parse_version("0.24.9")


# ---------------------------------------------------------------------------
# The real HANDOFF.md
# ---------------------------------------------------------------------------

_H_0244 = "## 2026-11-12 — plugin-codex: the Codex hooks run on the client core (plugin 0.24.4)"
_H_0241 = (
    "## 2026-11-12 — guards: only a full-trust device takes a session back on the record; "
    "appends to one session take turns (plugin 0.24.1)"
)
_H_0230 = (
    "## 2026-11-06 — dream→skills autopilot: the lane decides the routine proposals itself "
    "(plugin 0.23.0, schema 064)"
)
_H_0200 = (
    "## 2026-10-12 — `synapse login` is the single setup step (#222, plugin 0.20.0, schema 060)"
)


def _slice(heading: str) -> str:
    """The real file from ``heading`` to the next ``## `` heading (or the end), found
    without the script's parser."""
    start = _REAL.index("\n" + heading + "\n") + 1
    end = _REAL.find("\n## ", start)
    return _REAL[start : end if end != -1 else len(_REAL)].rstrip() + "\n"


def _headings() -> list[str]:
    return [line for line in _REAL.splitlines() if line.startswith("## ")]


def test_real_first_entry_0_24_4():
    notes = rn.notes_for(_REAL, "0.24.4")
    assert notes == _slice(_H_0244)
    assert notes.splitlines()[0] == _H_0244
    assert "(plugin 0.24.3)" not in notes


def test_real_two_item_parenthetical_0_23_0():
    notes = rn.notes_for(_REAL, "0.23.0")
    assert notes == _slice(_H_0230)
    assert notes.splitlines()[0] == _H_0230
    # the server-only entry right below it is not swept in
    assert "repair numbered-entity magnets" not in notes


def test_real_entry_in_the_middle_0_24_1():
    """0.24.1 sits between a server-only entry (schema 068) and 0.24.0."""
    notes = rn.notes_for(_REAL, "0.24.1")
    assert notes == _slice(_H_0241)
    assert "(schema 068)" not in notes and "(plugin 0.24.0)" not in notes


def test_real_last_entry_runs_to_the_end_of_the_file():
    assert _headings()[-1] == _H_0200
    notes = rn.notes_for(_REAL, "0.20.0")
    assert notes == _REAL[_REAL.index(_H_0200) :].rstrip() + "\n"


@pytest.mark.parametrize("version", ["0.9.1", "0.24.5", "0.2.4", "1.0.0"])
def test_real_version_with_no_entry(version):
    assert rn.notes_for(_REAL, version) is None


def test_real_entries_without_a_plugin_version_are_never_picked():
    server_only = [h for h in _headings() if not rn.heading_versions(h)]
    assert len(server_only) >= 3  # schema 068, schema 067, the magnet repair script
    versions = set().union(*(rn.heading_versions(h) for h in _headings()))
    assert {"0.24.4", "0.24.1", "0.23.0", "0.20.0"} <= versions
    for version in versions:
        notes = rn.notes_for(_REAL, version)
        assert notes is not None
        assert not any(h in notes for h in server_only), version
        # exactly the entries whose heading names it
        assert [h for h in _headings() if h in notes] == [
            h for h in _headings() if version in rn.heading_versions(h)
        ]


# ---------------------------------------------------------------------------
# Synthetic notes
# ---------------------------------------------------------------------------

_SYNTH = """# Handoff log

Newest first. Mentions plugin 1.0.1 in the preamble.

## 2026-11-14 — client: a follow-up (plugin 1.0.1)

second
## 2026-11-13 — server: a migration (schema 070)

Needs plugin 1.0.1 on every machine.

## 2026-11-12 — client: the fix (schema 069, plugin 1.0.1)

first

```
## a heading inside a fence stays in the entry
```

### a subheading stays too

## 2026-11-11 — client: unrelated (plugin 1.0.10)

no
"""


def test_two_entries_for_one_version_in_file_order():
    notes = rn.notes_for(_SYNTH, "1.0.1")
    assert notes.splitlines()[0] == "## 2026-11-14 — client: a follow-up (plugin 1.0.1)"
    assert notes.index("a follow-up") < notes.index("the fix (schema 069, plugin 1.0.1)")
    assert "## a heading inside a fence stays in the entry" in notes
    assert "### a subheading stays too" in notes
    assert "schema 070" not in notes and "Needs plugin" not in notes
    assert "preamble" not in notes and "1.0.10" not in notes
    assert notes.endswith("### a subheading stays too\n")


def test_a_version_is_matched_whole():
    assert rn.notes_for(_SYNTH, "1.0.10").splitlines()[0].endswith("(plugin 1.0.10)")
    assert rn.notes_for(_SYNTH, "1.0.0") is None


@pytest.mark.parametrize(
    ("heading", "versions"),
    [
        ("## d — x (plugin 0.24.4)", {"0.24.4"}),
        ("## d — x (plugin 0.23.1, schema 065)", {"0.23.1"}),
        ("## d — x (schema 064, plugin 0.23.0)", {"0.23.0"}),
        ("## d — x (#222, plugin 0.20.0, schema 060)", {"0.20.0"}),
        ("## d — x (plugin 0.22.0, no schema change)", {"0.22.0"}),
        ("## d — plugin-codex: x (plugin 0.24.4)", {"0.24.4"}),
        ("## d — x (schema 068)", set()),
        ("## d — x (script only, no schema or serving change)", set()),
        ("## d — plugin-codex 1.2.3 is not the plugin", set()),
        ("## d — x (plugin 0.24.41)", {"0.24.41"}),
    ],
)
def test_heading_forms(heading, versions):
    assert rn.heading_versions(heading) == versions


def test_notes_under_the_cap_are_unchanged():
    body = rn.notes_for(_SYNTH, "1.0.1")
    assert rn.cap_notes(body, len(body.encode()), "v1.0.1") == body


def test_notes_are_cut_at_a_line_to_the_byte_limit():
    entry = "## 2026-11-12 — big (plugin 1.2.3)\n" + "".join(
        f"- line {i} ééé —\n" for i in range(5000)
    )
    full = rn.notes_for(entry, "1.2.3", max_bytes=10**9)
    capped = rn.notes_for(entry, "1.2.3", max_bytes=5000)
    assert len(capped.encode("utf-8")) <= 5000
    assert capped.startswith("## 2026-11-12 — big (plugin 1.2.3)\n- line 0")
    head, marker = capped.rsplit("\n\n", 1)
    assert full.startswith(head + "\n")  # whole lines only
    assert "HANDOFF.md at v1.2.3" in marker


def test_the_default_cap_is_githubs_limit():
    assert rn.MAX_NOTES_BYTES == 125_000
    huge = "## d — x (plugin 1.2.3)\n" + "y" * 80 + "\n"
    notes = rn.notes_for(huge * 3000, "1.2.3")
    assert 120_000 < len(notes.encode()) <= 125_000


def test_fallback_notes_are_never_empty():
    body = rn.fallback_notes("1.2.3", "a" * 40, "")
    assert "No HANDOFF.md entry names plugin 1.2.3" in body and "aaaaaaaaaaaa" in body
    assert body.strip()


# ---------------------------------------------------------------------------
# The client file set
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "mode", "client"),
    [
        ("plugin/.claude-plugin/plugin.json", "100644", True),
        ("plugin/commands/config-review.md", "100644", True),
        ("plugin/README.md", "100644", True),
        ("plugin-codex/README.md", "100644", True),
        ("plugin/bin/synapse", "100755", True),
        ("plugin/bin/link", "120000", True),
        ("plugin-codex/hooks/stop.py", "100644", True),
        ("plugin/scripts/__pycache__/x.cpython-312.pyc", "100644", False),
        ("plugin-codex/stray.pyc", "100644", False),
        ("plugin/node_modules/x/index.js", "100644", False),
        ("plugin/.venv/lib/x.py", "100644", False),
        ("plugin/vendored", "160000", False),
        ("plugin", "100644", False),
        ("plugin-codexx/a.py", "100644", False),
        ("plugins/a.py", "100644", False),
        ("HANDOFF.md", "100644", False),
        ("mcp_server/install_routes.py", "100644", False),
        ("docs/plugin/x.md", "100644", False),
    ],
)
def test_client_file_rule(path, mode, client):
    assert rn.is_client_file(path, mode) is client


@needs_git
def test_the_client_set_is_what_install_bundles():
    """Over every tracked regular file in the real client trees, the release rule and
    /install's bundle rule agree, and the skipped directories are the same list."""
    from mcp_server import install_routes

    assert rn.SKIP_PARTS == install_routes._SKIP_PARTS
    assert rn.CLIENT_TREES == install_routes._TREES
    listed = subprocess.run(
        ["git", "-C", str(_REPO), "ls-files", "-s", "-z", "--", *rn.CLIENT_TREES],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    seen = 0
    for item in filter(None, listed.split("\0")):
        meta, path = item.split("\t", 1)
        mode = meta.split(" ")[0]
        if mode not in ("100644", "100755"):
            continue
        tree = path.split("/")[0]
        bundled = install_routes._bundled(_REPO / tree, _REPO / path)
        assert rn.is_client_file(path, mode) is bundled, path
        seen += 1
    assert seen > 50


# ---------------------------------------------------------------------------
# Scratch repositories and a gh recorder
# ---------------------------------------------------------------------------

_HANDOFF = """# Handoff log

## 2026-11-12 — client: the fix (plugin 1.0.1)

**What changed** the fix.

## 2026-11-11 — server: a migration (schema 070)

Needs plugin 1.0.0.

## 2026-11-10 — client: the first one (plugin 1.0.0)

**What changed** everything.
"""


def _client(version: str, init_version: str | None = None) -> dict[str, str]:
    return {
        rn.MANIFEST: json.dumps({"name": "synapse", "version": version}),
        rn.PACKAGE_INIT: f'"""core"""\n\n__version__ = "{init_version or version}"\n',
    }


class Repo:
    def __init__(self, path: Path):
        self.path = path
        path.mkdir()
        self.git("init", "-q")

    def git(self, *args: str) -> str:
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        env.update(
            GIT_CONFIG_GLOBAL=os.devnull,
            GIT_CONFIG_NOSYSTEM="1",
            GIT_AUTHOR_NAME="Test Author",
            GIT_AUTHOR_EMAIL="author@example.invalid",
            GIT_COMMITTER_NAME="Test Author",
            GIT_COMMITTER_EMAIL="author@example.invalid",
        )
        return subprocess.run(
            ["git", "-c", "init.defaultBranch=main", *args],
            cwd=self.path,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def commit(self, files: dict[str, str], message: str = "change") -> str:
        for rel, text in files.items():
            target = self.path / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        self.git("add", "-A")
        self.git("commit", "-q", "--allow-empty", "-m", message)
        return self.git("rev-parse", "HEAD")


@pytest.fixture()
def repo(tmp_path) -> Repo:
    """1.0.0 is committed and tagged (annotated, like the hand-made tags); then 1.0.1 is
    committed with its HANDOFF entry, followed by a server-only commit. HEAD is 1.0.1."""
    r = Repo(tmp_path / "repo")
    r.v100 = r.commit(
        {
            **_client("1.0.0"),
            "plugin/scripts/hook.py": "print('hook')\n",
            "plugin/commands/review.md": "Run the review.\n",
            "plugin/README.md": "# Synapse plugin\n",
            "plugin-codex/install.py": "print('codex')\n",
            "mcp_server/app.py": "app = 1\n",
            "HANDOFF.md": _HANDOFF,
        },
        "client 1.0.0",
    )
    r.git("tag", "-a", "-m", "v1.0.0", "v1.0.0", r.v100)
    r.v101 = r.commit(
        {**_client("1.0.1"), "plugin/scripts/hook.py": "print('hook 2')\n"},
        "client: the fix (plugin 1.0.1) (#7)",
    )
    r.server = r.commit({"mcp_server/app.py": "app = 2\n"}, "server: a migration")
    return r


class Gh:
    """Stands in for gh: records each call, keeps the notes file's text, answers with the
    exit code set per subcommand."""

    def __init__(self, view: int = 1, api: int = 0, create: int = 0):
        self.codes = {"view": view, "api": api, "create": create}
        self.calls: list[list[str]] = []
        self.notes: list[str] = []

    def __call__(self, args: Sequence[str], capture: bool) -> subprocess.CompletedProcess[str]:
        args = list(args)
        self.calls.append(args)
        if args[:2] == ["release", "create"]:
            path = args[args.index("--notes-file") + 1]
            self.notes.append(Path(path).read_text(encoding="utf-8"))
            code = self.codes["create"]
        elif args[:2] == ["release", "view"]:
            code = self.codes["view"]
        elif args[0] == "api":
            code = self.codes["api"]
        else:
            raise AssertionError(f"unexpected gh call {args}")
        return subprocess.CompletedProcess(["gh", *args], code, "", "gh: HTTP 422" if code else "")

    def kinds(self) -> list[str]:
        return [" ".join(call[:2]) if call[0] == "release" else call[0] for call in self.calls]


def _create_ref(call: list[str]) -> tuple[str, str]:
    assert call[:4] == ["api", "--method", "POST", "repos/{owner}/{repo}/git/refs"]
    fields = dict(call[i + 1].split("=", 1) for i, arg in enumerate(call) if arg == "-f")
    return fields["ref"], fields["sha"]


# ---------------------------------------------------------------------------
# release
# ---------------------------------------------------------------------------


@needs_git
def test_release_creates_the_tag_at_the_newest_client_commit_then_publishes(repo):
    gh = Gh()
    message = rn.release(repo.path, "HEAD", gh)
    assert gh.kinds() == ["api", "release create"]
    # the server-only commit on top is not the target: the commit that made 1.0.1 is
    assert _create_ref(gh.calls[0]) == ("refs/tags/v1.0.1", repo.v101)
    create = gh.calls[1]
    assert create[:3] == ["release", "create", "v1.0.1"]
    assert "--verify-tag" in create and "--target" not in create
    assert create[create.index("--title") + 1] == "v1.0.1"
    assert gh.notes == [
        "## 2026-11-12 — client: the fix (plugin 1.0.1)\n\n**What changed** the fix.\n"
    ]
    assert not Path(create[create.index("--notes-file") + 1]).exists()  # cleaned up
    assert message == f"Released v1.0.1 at {repo.v101[:12]}"


@needs_git
def test_release_when_tagged_and_released_calls_nothing_that_writes(repo):
    repo.git("tag", "v1.0.1", repo.v101)
    gh = Gh(view=0)
    message = rn.release(repo.path, "HEAD", gh)
    assert gh.calls == [["release", "view", "v1.0.1", "--json", "tagName"]]
    assert "already released" in message


@needs_git
def test_release_publishes_on_a_tag_that_has_no_release(repo):
    repo.git("tag", "v1.0.1", repo.v101)
    gh = Gh(view=1)
    message = rn.release(repo.path, "HEAD", gh)
    assert gh.kinds() == ["release view", "release create"]  # no new tag
    assert "--verify-tag" in gh.calls[1]
    assert gh.notes[0].startswith("## 2026-11-12 — client: the fix (plugin 1.0.1)")
    assert message.endswith("(existing tag)")


@needs_git
@pytest.mark.parametrize(
    "change",
    [
        {"plugin/scripts/hook.py": "print('hook 3')\n"},
        {"plugin/commands/review.md": "Run the review twice.\n"},
        {"plugin/README.md": "# Synapse plugin, reworded\n"},
        {"plugin-codex/new.py": "x = 1\n"},
    ],
)
def test_release_fails_when_the_client_changed_under_a_released_version(repo, change):
    repo.git("tag", "v1.0.1", repo.v101)
    repo.commit(change, "client change without a bump")
    gh = Gh()
    with pytest.raises(rn.ReleaseError, match=r"version is still 1\.0\.1: bump it") as err:
        rn.release(repo.path, "HEAD", gh)
    assert next(iter(change)) in str(err.value)
    assert gh.calls == []


@needs_git
@pytest.mark.parametrize(
    ("files", "match"),
    [
        (_client("1.0.2;id"), "is not a version"),
        (_client("v1.0.2"), "is not a version"),
        ({rn.MANIFEST: json.dumps({"version": 102})}, "is not a version"),
        ({rn.MANIFEST: "{not json"}, "is not JSON"),
        (_client("1.0.2", init_version="1.0.3"), "must agree"),
    ],
)
def test_release_refuses_a_bad_version_before_any_call(repo, files, match):
    repo.commit(files, "bad version")
    gh = Gh()
    with pytest.raises(rn.ReleaseError, match=match):
        rn.release(repo.path, "HEAD", gh)
    assert gh.calls == []


@needs_git
def test_release_without_a_handoff_entry_quotes_the_commit(repo):
    sha = repo.commit({**_client("1.0.2"), "plugin/x.py": "x\n"}, "client: quiet fix (#9)")
    gh = Gh()
    rn.release(repo.path, "HEAD", gh)
    assert _create_ref(gh.calls[0]) == ("refs/tags/v1.0.2", sha)
    assert gh.notes == [
        f"No HANDOFF.md entry names plugin 1.0.2. This release is commit {sha[:12]}:\n\n"
        "> client: quiet fix (#9)\n"
    ]


@needs_git
def test_release_stops_when_the_tag_cannot_be_created(repo):
    gh = Gh(api=1)
    with pytest.raises(rn.ReleaseError, match=r"Could not create tag v1\.0\.1"):
        rn.release(repo.path, "HEAD", gh)
    assert gh.kinds() == ["api"]


@needs_git
def test_release_reports_a_failed_publish(repo):
    gh = Gh(create=1)
    with pytest.raises(rn.ReleaseError, match="re-run this workflow"):
        rn.release(repo.path, "HEAD", gh)


@needs_git
def test_last_client_commit_skips_server_only_commits(repo):
    assert rn.last_client_commit(repo.path, repo.server) == repo.v101
    assert rn.last_client_commit(repo.path, repo.v100) == repo.v100


@needs_git
def test_last_client_commit_stays_on_mains_own_line(repo):
    """A PR merged with a merge commit: the tag goes on the merge, not on the branch
    commit that never stood alone on main."""
    repo.git("checkout", "-q", "-b", "topic", repo.v101)
    inside = repo.commit({**_client("1.0.2"), "plugin/x.py": "x\n"}, "topic")
    repo.git("checkout", "-q", "main")
    repo.git("merge", "-q", "--no-ff", "-m", "merge topic", "topic")
    merge = repo.git("rev-parse", "HEAD")
    after = repo.commit({"mcp_server/app.py": "app = 9\n"}, "server after the merge")
    assert rn.last_client_commit(repo.path, after) == merge != inside


# ---------------------------------------------------------------------------
# check-pr
# ---------------------------------------------------------------------------


@needs_git
def test_pr_without_a_client_change_passes_whatever_the_version(repo):
    repo.git("tag", "v1.0.1", repo.v101)  # even with the version already released
    head = repo.commit({"mcp_server/app.py": "app = 3\n", "HANDOFF.md": "# x\n"})
    assert "No client file changed" in rn.check_pr(repo.path, repo.server, head)


@needs_git
@pytest.mark.parametrize(
    "change",
    [{"plugin/scripts/hook.py": "print('hook 3')\n"}, {"plugin/commands/review.md": "new\n"}],
)
def test_pr_changing_the_client_under_the_same_version_fails(repo, change):
    head = repo.commit(change)
    with pytest.raises(rn.ReleaseError, match=r"version is still 1\.0\.1: bump it") as err:
        rn.check_pr(repo.path, repo.server, head)
    assert "next free: 1.0.2" in str(err.value) and next(iter(change)) in str(err.value)


@needs_git
def test_pr_raising_the_version_passes(repo):
    head = repo.commit({**_client("1.1.0"), "plugin/x.py": "x\n"})
    assert "1.0.1 -> 1.1.0" in rn.check_pr(repo.path, repo.server, head)


@needs_git
def test_pr_lowering_the_version_fails(repo):
    head = repo.commit({**_client("0.9.9"), "plugin/x.py": "x\n"})
    with pytest.raises(rn.ReleaseError, match=r"went from 1\.0\.1 down to 0\.9\.9"):
        rn.check_pr(repo.path, repo.server, head)


@needs_git
def test_pr_taking_a_released_version_fails(repo):
    repo.git("tag", "v1.0.2", repo.v100)  # someone already released 1.0.2
    head = repo.commit({**_client("1.0.2"), "plugin/x.py": "x\n"})
    with pytest.raises(rn.ReleaseError, match=r"1\.0\.2 is already released") as err:
        rn.check_pr(repo.path, repo.server, head)
    assert "next free: 1.0.3" in str(err.value)


@needs_git
def test_pr_with_a_malformed_version_fails(repo):
    head = repo.commit(_client("1.0.2-rc1"))
    with pytest.raises(rn.ReleaseError, match="is not a version"):
        rn.check_pr(repo.path, repo.server, head)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@needs_git
def test_cli_version_and_notes(repo, capsys):
    assert rn.main(["--repo", str(repo.path), "version"]) == 0
    assert capsys.readouterr().out == "1.0.1\n"
    assert rn.main(["--repo", str(repo.path), "notes", "1.0.0"]) == 0
    assert capsys.readouterr().out.startswith("## 2026-11-10 — client: the first one")
    assert rn.main(["--repo", str(repo.path), "notes", "1.0.9"]) == 1
    assert rn.main(["--repo", str(repo.path), "notes", "1.0;id"]) == 1


@needs_git
def test_cli_failure_is_one_escaped_annotation_in_actions(repo, capsys, monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    head = repo.commit({"plugin/scripts/hook.py": "100%\n"})
    code = rn.main(["--repo", str(repo.path), "check-pr", "--base", repo.server, "--head", head])
    assert code == 1
    out = capsys.readouterr().out
    assert out.startswith("::error title=Version::Client changed but the version is still")
    assert out.count("\n") == 1 and "%0A" in out


@needs_git
def test_cli_check_pr_defaults_to_the_merge_commit_and_its_first_parent(repo, capsys):
    repo.git("checkout", "-q", "-b", "topic", repo.v101)
    repo.commit({**_client("1.0.2"), "plugin/x.py": "x\n"}, "topic")
    repo.git("checkout", "-q", "main")
    repo.git("merge", "-q", "--no-ff", "-m", "merge", "topic")
    assert rn.main(["--repo", str(repo.path), "check-pr"]) == 0
    assert "1.0.1 -> 1.0.2" in capsys.readouterr().out
