"""The relabel planner (scripts/relabel_projects.py) — pure logic, no database.

Until the one-resolver change a project's label was its session's working directory cut
on "/" only, so one project was stored under many names: a whole Windows path, the same
path with a lowercase drive, several casings, one label per subfolder or worktree a
session started in, and a hyphen/underscore twin from another machine. The planner maps
every stored (label, cwd) group to one canonical name; a human reads the dry run before
anything is written. What is pinned here is the mapping itself, on generic data shaped
like a real deployment:

* a Windows checkout's path label, its lowercase-drive and trailing-separator twins, its
  subfolders (one of them two levels deep), a cursor-style ``/C:/...`` cwd for the bare
  name, the bare name with no cwd, a case twin and an alias all become ONE name;
* a posix repo's subfolder, scratch and worktree sessions fold into the repo;
* two machines' checkouts with the same basename both keep it;
* home directories and folders that hold projects (``services``, ``workspace``, ``data``,
  ``.claude/skills``, a OneDrive root, a ``container:`` line) never absorb anything;
* a label set explicitly (it differs from its cwd's basename) is never re-derived;
* re-planning the relabelled data plans nothing (the --apply re-run is a no-op);
* the alias file grammar, the allowlist/scope mappers and the dry-run report.

DB reads/writes and the trigger-maintained KG provenance are covered end to end in
tests/test_relabel_projects_db.py.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "relabel_projects",
    Path(__file__).resolve().parent.parent / "scripts" / "relabel_projects.py",
)
rp = importlib.util.module_from_spec(_SPEC)
sys.modules["relabel_projects"] = rp
_SPEC.loader.exec_module(rp)

Groups = dict[tuple[str, str | None], int]

# A Windows checkout, as the old server split stored it: the whole path.
WIN = r"C:\Users\pat\OneDrive - Acme Ltd\projects\BigProj"
WIN_ROOT = "C:/Users/pat/OneDrive - Acme Ltd/projects/BigProj"  # normalised display
WIN_LC = "c" + WIN[1:]  # the same folder with a lowercase drive letter
CURSOR = "/" + WIN_ROOT  # how a cursor-style client reports the same cwd
WIN_SUBS = (r"\tickets", r"\tickets\lab\out", r"\src", r"\dev")

HOME = "/home/u"
REPO = f"{HOME}/code/acme"
REPO_SUBS = ("/scripts", "/.claude/worktrees/agent-x", "/scratchpad/probe")

ALIASES = "# generic test aliases\nbig-proj = BigProj\ncontainer: /home/u/clients\n"


def deployment() -> Groups:
    """(label, metadata.cwd) -> episode count, one entry per shape seen in the wild."""
    g: Groups = {
        (WIN, WIN): 40,
        (WIN_LC, WIN_LC): 7,
        (WIN + "\\", WIN + "\\"): 1,
        ("BigProj", CURSOR): 11,
        ("BigProj", None): 4,
        ("bigproj", None): 6,
        ("big-proj", f"{HOME}/code/big-proj"): 9,
        ("acme", REPO): 50,
        ("alpha", f"{REPO}/sub"): 3,  # explicit .synapse-project override
        ("widget", f"{HOME}/code/widget"): 5,
        ("lib", f"{HOME}/code/widget/lib"): 1,
        ("widget", "/Users/v/work/widget"): 6,
        ("u", HOME): 3,
        ("notes", f"{HOME}/notes"): 2,
        (r"C:\Users\pat", r"C:\Users\pat"): 2,
        (r"C:\Users\pat\scratch-tool", r"C:\Users\pat\scratch-tool"): 2,
        (r"C:\Users\pat\OneDrive - Acme Ltd", r"C:\Users\pat\OneDrive - Acme Ltd"): 1,
        (
            r"C:\Users\pat\OneDrive - Acme Ltd\projects",
            r"C:\Users\pat\OneDrive - Acme Ltd\projects",
        ): 1,
        ("services", f"{HOME}/services"): 1,
        ("api", f"{HOME}/services/api"): 4,
        ("workspace", f"{HOME}/workspace"): 1,
        ("thing", f"{HOME}/workspace/thing"): 1,
        ("data", f"{HOME}/data"): 1,
        ("set1", f"{HOME}/data/set1"): 1,
        ("skills", f"{HOME}/.claude/skills"): 1,
        ("my-skill", f"{HOME}/.claude/skills/my-skill"): 1,
        ("clients", f"{HOME}/clients"): 1,
        ("client-a", f"{HOME}/clients/client-a"): 2,
        ("client-b", f"{HOME}/clients/client-b"): 2,
    }
    for sub, n in zip(WIN_SUBS, (5, 2, 3, 1), strict=True):
        g[(WIN + sub, WIN + sub)] = n
    for sub, n in zip(REPO_SUBS, (3, 2, 1), strict=True):
        g[(sub.rsplit("/", 1)[-1], REPO + sub)] = n
    return g


EXPECTED: dict[tuple[str, str | None], str] = {
    (WIN, WIN): "BigProj",
    (WIN_LC, WIN_LC): "BigProj",
    (WIN + "\\", WIN + "\\"): "BigProj",
    **{(WIN + s, WIN + s): "BigProj" for s in WIN_SUBS},
    ("BigProj", CURSOR): "BigProj",
    ("BigProj", None): "BigProj",
    ("bigproj", None): "BigProj",
    ("big-proj", f"{HOME}/code/big-proj"): "BigProj",
    ("acme", REPO): "acme",
    **{(s.rsplit("/", 1)[-1], REPO + s): "acme" for s in REPO_SUBS},
    ("alpha", f"{REPO}/sub"): "alpha",
    ("widget", f"{HOME}/code/widget"): "widget",
    ("lib", f"{HOME}/code/widget/lib"): "widget",
    ("widget", "/Users/v/work/widget"): "widget",
    ("u", HOME): "u",
    ("notes", f"{HOME}/notes"): "notes",
    (r"C:\Users\pat", r"C:\Users\pat"): "pat",
    (r"C:\Users\pat\scratch-tool", r"C:\Users\pat\scratch-tool"): "scratch-tool",
    (r"C:\Users\pat\OneDrive - Acme Ltd", r"C:\Users\pat\OneDrive - Acme Ltd"): (
        "OneDrive - Acme Ltd"
    ),
    (
        r"C:\Users\pat\OneDrive - Acme Ltd\projects",
        r"C:\Users\pat\OneDrive - Acme Ltd\projects",
    ): "projects",
    ("services", f"{HOME}/services"): "services",
    ("api", f"{HOME}/services/api"): "api",
    ("workspace", f"{HOME}/workspace"): "workspace",
    ("thing", f"{HOME}/workspace/thing"): "thing",
    ("data", f"{HOME}/data"): "data",
    ("set1", f"{HOME}/data/set1"): "set1",
    ("skills", f"{HOME}/.claude/skills"): "skills",
    ("my-skill", f"{HOME}/.claude/skills/my-skill"): "my-skill",
    ("clients", f"{HOME}/clients"): "clients",
    ("client-a", f"{HOME}/clients/client-a"): "client-a",
    ("client-b", f"{HOME}/clients/client-b"): "client-b",
}


def _plan(groups: Groups | None = None, aliases: str = ALIASES, other=()):
    return rp.Planner(rp.AliasConfig.parse(aliases)).plan(
        deployment() if groups is None else groups, other
    )


def _changes(plan, label: str) -> list:
    return [c for c in plan.changes if c.label == label]


def _relabelled(groups: Groups, plan) -> Groups:
    """The episode groups as they would be stored after --apply."""
    out: Groups = defaultdict(int)
    for (label, cwd), n in groups.items():
        out[(plan.groups[(label, cwd)], cwd)] += n
    return dict(out)


# ---------------------------------------------------------------------------
# The whole deployment
# ---------------------------------------------------------------------------


def test_every_group_is_planned_and_mapped_as_expected():
    """Plan.groups covers every input group, unchanged ones included: --apply walks it,
    so a group missing here would silently keep its old label."""
    plan = _plan()
    assert set(plan.groups) == set(deployment())
    assert plan.groups == EXPECTED


def test_the_windows_checkout_collapses_into_one_name():
    plan = _plan()
    big = {k for k, v in plan.groups.items() if v == "BigProj"}
    assert len(big) == 3 + len(WIN_SUBS) + 4  # paths, subfolders, bare/case/alias
    assert sum(deployment()[k] for k in big) == 40 + 7 + 1 + 11 + 11 + 4 + 6 + 9


def test_path_labels_become_their_name_whatever_the_drive_case_or_trailing_separator():
    plan = _plan()
    for label in (WIN, WIN_LC, WIN + "\\"):
        (change,) = _changes(plan, label)
        assert change.canonical == "BigProj"
        assert change.reason == "path → its name"
        assert change.cwd_root == WIN_ROOT


@pytest.mark.parametrize("sub", WIN_SUBS, ids=lambda s: s.strip("\\").replace("\\", "/"))
def test_windows_subfolders_fold_into_the_checkout(sub):
    """``src`` and ``dev`` are container names, so they are never roots themselves, but
    a session started in one still belongs to the checkout above it; ``tickets\\lab\\out``
    folds past ``tickets`` (also a candidate) straight to the top."""
    (change,) = _changes(_plan(), WIN + sub)
    assert (change.canonical, change.cwd_root) == ("BigProj", WIN_ROOT)
    assert change.reason == "subfolder → project root"


def test_a_cursor_style_cwd_is_the_same_folder_as_the_windows_path():
    """``/C:/Users/...`` and ``C:\\Users\\...`` name one directory: the bare label it
    produced is already canonical, and a subfolder session reported that way folds into
    the same root."""
    plan = _plan()
    assert _changes(plan, "BigProj") == []
    assert list(plan.roots.values()).count(WIN_ROOT) == 1
    only_cursor = {
        ("tickets", CURSOR + "/tickets"): 3,
        ("BigProj", "/c:/Users/pat/OneDrive - Acme Ltd/projects/BigProj"): 2,
    }
    p = _plan(only_cursor, aliases="")
    assert p.groups == {k: "BigProj" for k in only_cursor}
    (change,) = p.changes
    assert (change.label, change.cwd_root, change.rows) == ("tickets", WIN_ROOT, 3)


def test_a_case_twin_merges_into_the_majority_casing():
    plan = _plan()
    (change,) = _changes(plan, "bigproj")
    assert (change.canonical, change.reason, change.cwd_root) == ("BigProj", "case merge", None)
    assert plan.groups[("BigProj", None)] == "BigProj"


def test_an_alias_maps_what_no_path_rule_can_see():
    (change,) = _changes(_plan(), "big-proj")
    assert (change.canonical, change.reason, change.rows) == ("BigProj", "alias", 9)
    assert change.cwd_root == f"{HOME}/code/big-proj"
    without = _plan(aliases="")
    assert without.groups[("big-proj", f"{HOME}/code/big-proj")] == "big-proj"


@pytest.mark.parametrize("sub", REPO_SUBS)
def test_posix_subfolder_scratch_and_worktree_sessions_fold_into_the_repo(sub):
    (change,) = _changes(_plan(), sub.rsplit("/", 1)[-1])
    assert (change.canonical, change.cwd_root) == ("acme", REPO)
    assert change.reason == "subfolder → project root"


def test_two_machines_checkouts_with_one_basename_both_keep_it():
    """Same name, different folders: two roots, no change, and each subfolder joins the
    root it is actually under."""
    plan = _plan()
    assert {f"{HOME}/code/widget", "/Users/v/work/widget"} <= set(plan.roots.values())
    assert _changes(plan, "widget") == []
    (lib,) = _changes(plan, "lib")
    assert (lib.canonical, lib.cwd_root) == ("widget", f"{HOME}/code/widget")


@pytest.mark.parametrize(
    ("home", "child"),
    [
        (HOME, f"{HOME}/notes"),
        ("/Users/v", "/Users/v/notes"),
        ("/root", "/root/notes"),
        (r"C:\Users\pat", r"C:\Users\pat\notes"),
        ("c:/users/PAT", "c:/users/PAT/notes"),
    ],
)
def test_a_home_directory_never_absorbs_anything(home, child):
    groups = {(rp.label_from_path(home), home): 5, ("notes", child): 1}
    plan = _plan(groups, aliases="")
    assert plan.groups[("notes", child)] == "notes"
    assert rp.norm_path(home) not in plan.roots.values()


def test_the_home_directory_path_label_is_only_renamed():
    plan = _plan()
    (pat,) = _changes(plan, r"C:\Users\pat")
    assert (pat.canonical, pat.cwd_root, pat.reason) == ("pat", None, "path → its name")
    (tool,) = _changes(plan, r"C:\Users\pat\scratch-tool")
    assert (tool.canonical, tool.cwd_root) == ("scratch-tool", "C:/Users/pat/scratch-tool")


@pytest.mark.parametrize(
    ("container", "child"),
    [
        ("services", "api"),
        ("workspace", "thing"),
        ("data", "set1"),
        ("skills", "my-skill"),
    ],
)
def test_a_folder_that_holds_projects_never_absorbs_them(container, child):
    plan = _plan()
    assert _changes(plan, container) == []
    assert _changes(plan, child) == []
    assert not any(r.rsplit("/", 1)[-1] == container for r in plan.roots.values())


def test_the_onedrive_and_projects_folders_hold_the_checkout_without_absorbing_it():
    plan = _plan()
    assert plan.groups[(WIN, WIN)] == "BigProj"
    od = r"C:\Users\pat\OneDrive - Acme Ltd"
    assert plan.groups[(od, od)] == "OneDrive - Acme Ltd"
    assert plan.groups[(od + r"\projects", od + r"\projects")] == "projects"


def test_a_container_line_stops_a_folder_absorbing_its_children():
    """``clients`` is not a built-in container name. One session started in it would
    otherwise make it the root of every client repo below it, merging them all; the
    alias file's ``container:`` line is how the reviewer says so."""
    groups = {
        ("clients", f"{HOME}/clients"): 1,
        ("client-a", f"{HOME}/clients/client-a"): 2,
        ("client-b", f"{HOME}/clients/client-b"): 2,
    }
    merged = _plan(groups, aliases="")
    assert set(merged.groups.values()) == {"clients"}
    kept = _plan(groups, aliases="container: /home/u/clients/\n")
    assert kept.groups == {k: k[0] for k in groups}
    assert kept.changes == []


def test_a_windows_container_line_matches_any_spelling_of_the_path():
    groups = {
        ("Work", r"D:\Work"): 1,
        ("proj-a", r"d:\work\proj-a"): 2,
        ("proj-b", "/D:/Work/proj-b"): 2,
    }
    assert set(_plan(groups, aliases="").groups.values()) == {"Work"}
    kept = _plan(groups, aliases=r"container: d:\WORK")
    assert kept.groups == {k: k[0] for k in groups}


def test_an_explicit_label_is_never_re_derived_from_its_cwd():
    """A ``.synapse-project`` label differs from its folder's name: the folder did not
    produce it, so neither the folder nor the repo above it may rename it, and the
    folder does not become a root either."""
    plan = _plan()
    assert _changes(plan, "alpha") == []
    assert plan.groups[("alpha", f"{REPO}/sub")] == "alpha"
    assert f"{REPO}/sub" not in plan.roots.values()
    inside_windows = {(WIN, WIN): 3, ("alpha", WIN + r"\tickets"): 2}
    assert _plan(inside_windows, aliases="").groups[("alpha", WIN + r"\tickets")] == "alpha"


def test_the_deepest_subfolder_joins_the_topmost_root():
    groups = {
        ("acme", REPO): 5,
        ("tickets", f"{REPO}/tickets"): 2,
        ("out", f"{REPO}/tickets/lab/out"): 1,
    }
    plan = _plan(groups, aliases="")
    assert plan.roots == {REPO: REPO}
    assert set(plan.groups.values()) == {"acme"}


@pytest.mark.parametrize(
    ("parent", "a", "b"),
    [
        ("/home", "/home/u/code/proj-a", "/home/v/code/proj-b"),
        ("/Users", "/Users/u/code/proj-a", "/Users/v/code/proj-b"),
        (r"C:\Users", r"C:\Users\pat\code\proj-a", r"C:\Users\kim\code\proj-b"),
        ("/mnt/c/Users/pat", "/mnt/c/Users/pat/code/proj-a", "/mnt/c/Users/pat/code/proj-b"),
    ],
    ids=["home", "Users", "C-Users", "wsl-home"],
)
def test_the_folder_holding_home_directories_never_absorbs_anything(parent, a, b):
    groups = {
        (rp.label_from_path(parent), parent): 1,
        ("proj-a", a): 3,
        ("proj-b", b): 3,
    }
    plan = _plan(groups, aliases="")
    assert plan.groups[("proj-a", a)] == "proj-a"
    assert plan.groups[("proj-b", b)] == "proj-b"


# ---------------------------------------------------------------------------
# Casing
# ---------------------------------------------------------------------------


def test_case_merge_picks_the_casing_with_the_most_episodes():
    groups = {("Alpha", None): 2, ("alpha", None): 5, ("ALPHA", None): 1}
    plan = _plan(groups, aliases="")
    assert set(plan.groups.values()) == {"alpha"}
    assert {c.label: c.reason for c in plan.changes} == {
        "Alpha": "case merge",
        "ALPHA": "case merge",
    }


def test_a_case_merge_tie_is_broken_deterministically():
    a = {("Alpha", None): 3, ("alpha", None): 3}
    b = dict(reversed(list(a.items())))
    assert (
        _plan(a, aliases="").groups
        == _plan(b, aliases="").groups
        == {
            ("Alpha", None): "Alpha",
            ("alpha", None): "Alpha",
        }
    )


def test_an_alias_target_forces_its_casing_over_the_majority():
    groups = {("BigProj", None): 2, ("bigproj", None): 9, ("big-proj", None): 1}
    assert set(_plan(groups, aliases="").groups.values()) == {"bigproj", "big-proj"}
    forced = _plan(groups, aliases="big-proj = BigProj")
    assert set(forced.groups.values()) == {"BigProj"}
    assert {c.label: c.reason for c in forced.changes} == {
        "bigproj": "case merge",
        "big-proj": "alias",
    }
    # The alias may name the casing alone.
    assert set(_plan(groups, aliases="BIGPROJ = BigProj").groups.values()) == {
        "BigProj",
        "big-proj",
    }


def test_an_alias_combines_with_a_path_rule_in_the_reason():
    groups = {(r"C:\x\transcribe-ai", r"C:\x\transcribe-ai"): 4, ("transcribe_ai", None): 9}
    plan = _plan(groups, aliases="Transcribe-AI = transcribe_ai")
    (change,) = plan.changes
    assert (change.canonical, change.reason) == ("transcribe_ai", "path → its name, alias")


def test_the_canonical_casing_does_not_depend_on_group_order():
    groups = {
        ("bigproj", WIN.lower()): 1,
        ("BigProj", WIN): 100,
        (WIN + r"\tickets", WIN + r"\tickets"): 5,
    }
    reordered = dict(reversed(list(groups.items())))
    assert _plan(groups, aliases="").groups == _plan(reordered, aliases="").groups
    assert set(_plan(groups, aliases="").groups.values()) == {"BigProj"}


def test_every_change_names_why_it_changes():
    """The dry-run table is what a person approves: a row that changes must say why."""
    groups = {**deployment(), ("bigproj", WIN.lower()): 1}
    plan = _plan(groups)
    assert plan.groups[("bigproj", WIN.lower())] == "BigProj"
    assert [c for c in plan.changes if c.reason == "kept"] == []


# ---------------------------------------------------------------------------
# Re-running
# ---------------------------------------------------------------------------


def test_re_planning_the_relabelled_data_plans_nothing():
    """--apply re-plans from the live data, so after one run the next must be a no-op:
    every relabelled group is either its root (derived, unchanged) or an explicit label
    whose cwd basename differs (kept)."""
    first = _plan()
    after = _relabelled(deployment(), first)
    other = {first.map_label(lb) for lb in first.labels}
    second = _plan(after, other=other)
    assert second.changes == []
    assert all(second.groups[k] == k[0] for k in after)
    assert all(second.map_label(lb) == lb for lb in other)


def _stale_inventory(episode_groups: Groups) -> rp.Inventory:
    """Allowlists and scopes still on the old labels: what --apply leaves when it is
    interrupted after the episode commits, before the surfaces/skills transaction."""
    return rp.Inventory(
        episode_groups=episode_groups,
        tables={'"public"."notes"': {"BigProj": 2}},
        surfaces=[
            {"surface_id": "dev-laptop", "allowed_projects": [WIN, WIN_LC, "bigproj", "acme"]},
            {"surface_id": "dev-other", "allowed_projects": ["big-proj"]},
        ],
        skills=[{"name": "triage", "scope": f"project:{WIN}\\tickets"}],
        config_scopes={f"project:{WIN_LC}": 2, "project:big-proj": 1},
    )


def test_other_labels_include_allowlists_and_every_kind_of_scope():
    inv = _stale_inventory({})
    assert inv.other_labels() == {
        "BigProj",
        WIN,
        WIN_LC,
        "bigproj",
        "acme",
        "big-proj",
        WIN + r"\tickets",
    }


def test_a_re_run_after_an_interruption_still_maps_allowlists_and_scopes():
    """The episodes already carry canonical names, so no table holds the old path, case
    or alias labels any more; the allowlists and scopes that do must still map."""
    first = _plan()
    inv = _stale_inventory(_relabelled(deployment(), first))
    plan = _plan(inv.episode_groups, other=inv.other_labels())
    assert plan.changes == []
    assert rp.map_allowlist(plan, inv.surfaces[0]["allowed_projects"]) == ["BigProj", "acme"]
    assert rp.map_allowlist(plan, inv.surfaces[1]["allowed_projects"]) == ["BigProj"]
    assert rp.map_scope(plan, inv.skills[0]["scope"]) == "project:BigProj"
    assert {rp.map_scope(plan, s) for s in inv.config_scopes} == {"project:BigProj"}


# ---------------------------------------------------------------------------
# Labels without a cwd: other tables, allowlists, skill scopes
# ---------------------------------------------------------------------------


def test_labels_cover_every_episode_label_and_every_other_table_label():
    other = {WIN + r"\tickets", "transcribe-ai", "only-in-notes"}
    plan = _plan(other=other)
    assert set(plan.labels) == {lb for lb, _ in deployment()} | other
    assert plan.map_label(WIN + r"\tickets") == "BigProj"
    assert plan.map_label("bigproj") == "BigProj"
    assert plan.map_label("big-proj") == "BigProj"
    assert plan.map_label("only-in-notes") == "only-in-notes"
    assert plan.map_label("never-seen") == "never-seen"
    assert plan.map_label(None) is None


def test_a_path_label_seen_only_outside_episodes_still_becomes_a_name():
    plan = _plan({}, aliases="", other={WIN + r"\tickets", "/home/u/code/beta/"})
    assert plan.labels == {WIN + r"\tickets": "tickets", "/home/u/code/beta/": "beta"}


def test_a_bare_subfolder_label_is_not_widened_without_its_cwd():
    """Without a cwd, ``scripts`` names nothing provable (every repo has one), so the
    label map keeps it; only the episode group that carries the cwd moves."""
    plan = _plan()
    assert plan.groups[("scripts", REPO + "/scripts")] == "acme"
    assert plan.map_label("scripts") == "scripts"


def test_map_allowlist_maps_and_dedupes_case_insensitively_in_order():
    plan = _plan(other={"Gamma"})
    entries = [WIN, "acme", WIN_LC, "bigproj", "ACME", "big-proj", "gamma", "Gamma", "zeta"]
    assert rp.map_allowlist(plan, entries) == ["BigProj", "acme", "gamma", "zeta"]
    assert rp.map_allowlist(plan, []) == []
    assert rp.map_allowlist(plan, ["acme", "BigProj"]) == ["acme", "BigProj"]


@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        (f"project:{WIN}", "project:BigProj"),
        (f"project:{WIN}\\tickets", "project:BigProj"),
        ("project:bigproj", "project:BigProj"),
        ("project:big-proj", "project:BigProj"),
        ("project:acme", "project:acme"),
        ("project:unknown", "project:unknown"),
        ("global", "global"),
        ("project-ish", "project-ish"),
    ],
)
def test_map_scope(scope, expected):
    plan = _plan(other={WIN + r"\tickets"})
    assert rp.map_scope(plan, scope) == expected


# ---------------------------------------------------------------------------
# The alias file
# ---------------------------------------------------------------------------


def test_alias_file_grammar():
    cfg = rp.AliasConfig.parse(
        "# comment\n\n   transcribe-ai   =   transcribe_ai  \n"
        "Big-Proj = BigProj\n"
        "CONTAINER: /home/u/clients/\n"
        "container: c:\\Users\\pat\\Clients\n"
        "container:\n"
    )
    assert cfg.aliases == {"transcribe-ai": "transcribe_ai", "big-proj": "BigProj"}
    assert cfg.containers == {"/home/u/clients", "c:/users/pat/clients"}


@pytest.mark.parametrize(
    ("text", "line"),
    [
        ("just-a-name", 1),
        ("# ok\nold =", 2),
        ("= new", 1),
        ("a = b\nold = some/path", 2),
        ("old = C:\\x\\new", 1),
    ],
    ids=["no-equals", "empty-right", "empty-left", "slash-path-right", "windows-path-right"],
)
def test_alias_file_errors_name_the_line(text, line):
    with pytest.raises(ValueError, match=rf"line {line}:"):
        rp.AliasConfig.parse(text)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", None),
        (None, None),
        ("/", "/"),
        ("//home//u//", "/home/u"),
        ("c:\\", "C:/"),
        ("C:", "C:/"),
        ("/c:/Users/pat/", "C:/Users/pat"),
        ("c:\\Users\\pat\\", "C:/Users/pat"),
        ("relative\\dir", "relative/dir"),
    ],
)
def test_norm_path(raw, expected):
    assert rp.norm_path(raw) == expected


def test_path_key_ignores_case_only_on_windows():
    assert rp.path_key("C:/Users/Pat/X") == "c:/users/pat/x"
    assert rp.path_key("/home/U/X") == "/home/U/X"


# ---------------------------------------------------------------------------
# The dry-run report
# ---------------------------------------------------------------------------


def test_report_counts_the_change_table_and_every_section():
    groups = deployment()
    plan = _plan(other={WIN + r"\tickets", "bigproj", "zeta"})
    inv = rp.Inventory(
        episode_groups=groups,
        tables={
            '"public"."notes"': {WIN + r"\tickets": 2, "bigproj": 1, "zeta": 4},
            '"public"."preferences"': {"zeta": 3},
        },
        surfaces=[
            {"surface_id": "dev-laptop", "allowed_projects": [WIN, WIN_LC, "BigProj", "acme"]},
            {"surface_id": "dev-full", "allowed_projects": []},
        ],
        skills=[
            {"name": "triage", "scope": f"project:{WIN}"},
            {"name": "deploy", "scope": "project:acme"},
        ],
    )
    text = rp.report(inv, plan)
    lines = text.splitlines()

    before, after = len({lb for lb, _ in groups}), len(set(EXPECTED.values()))
    moved = sum(c.rows for c in plan.changes)
    assert lines[0] == (
        f"episodes: {before} labels -> {after}; {moved} of {sum(groups.values())} "
        "labelled rows change"
    )
    assert moved == sum(n for k, n in groups.items() if EXPECTED[k] != k[0])

    assert "| label | decided by folder | canonical | rows | why |" in lines
    table = [ln for ln in lines if ln.startswith("| ") and not ln.startswith(("| label", "| ---"))]
    assert len(table) == len(plan.changes)
    assert f"| {WIN} | {WIN_ROOT} | BigProj | 40 | path → its name |" in table
    assert (
        f"| {WIN}\\tickets\\lab\\out | {WIN_ROOT} | BigProj | 2 | subfolder → project root |"
        in (table)
    )
    assert "| bigproj |  | BigProj | 6 | case merge |" in table
    assert f"| big-proj | {HOME}/code/big-proj | BigProj | 9 | alias |" in table
    assert f"| agent-x | {REPO} | acme | 2 | subfolder → project root |" in table
    assert not any(ln.startswith(("| alpha |", "| widget |")) for ln in table)

    big_total = sum(n for k, n in groups.items() if EXPECTED[k] == "BigProj")
    acme_total = sum(n for k, n in groups.items() if EXPECTED[k] == "acme")
    after_block = lines[lines.index("canonical labels after (episodes):") + 1 :]
    assert after_block[0] == f"  {big_total:>7}  BigProj"
    assert f"  {acme_total:>7}  acme" in after_block

    assert '"public"."notes": 3 rows change' in lines
    assert f"        2  {WIN}\\tickets -> BigProj" in lines
    assert "        1  bigproj -> BigProj" in lines
    assert '"public"."preferences": no change (3 labelled rows)' in lines

    assert f"  dev-laptop: {[WIN, WIN_LC, 'BigProj', 'acme']} -> ['BigProj', 'acme']" in lines
    assert "  dev-full: [] -> []  (unchanged)" in lines
    assert f"  triage: project:{WIN} -> project:BigProj" in lines
    assert "  deploy: project:acme -> project:acme  (unchanged)" in lines


def test_report_of_a_clean_store_lists_no_changes():
    groups = {("acme", REPO): 3, ("widget", f"{HOME}/code/widget"): 2}
    plan = _plan(groups, aliases="")
    inv = rp.Inventory(groups, {}, [], [])
    text = rp.report(inv, plan)
    assert text.startswith("episodes: 2 labels -> 2; 0 of 5 labelled rows change\n")
    assert [ln for ln in text.splitlines() if ln.startswith("| ") and "label" not in ln] == [
        "| --- | --- | --- | ---: | --- |"
    ]


def test_report_lists_config_lane_scopes_with_their_row_counts():
    inv = _stale_inventory(_relabelled(deployment(), _plan()))
    inv.config_scopes["project:acme"] = 4
    plan = _plan(inv.episode_groups, other=inv.other_labels())
    lines = rp.report(inv, plan).splitlines()
    block = lines[lines.index("config_lane.config_registry.scope:") + 1 :]
    assert block[:3] == [
        "        4  project:acme -> project:acme  (unchanged)",
        "        1  project:big-proj -> project:BigProj",
        f"        2  project:{WIN_LC} -> project:BigProj",
    ]
    assert lines.index("config_lane.config_registry.scope:") < lines.index(
        "skills_lane.skill_registry.scope:"
    )


# ---------------------------------------------------------------------------
# The offline inventory (--inventory-sql, then --dry-run --inventory)
# ---------------------------------------------------------------------------

#: What ``psql -tA < inv.sql`` prints: the command tags around one JSON line.
_PSQL_OUTPUT = (
    "BEGIN\n"
    '{"episodes" : [["acme", "/home/u/code/acme", 3], ["scripts", "/home/u/code/acme/scripts", 2]'
    ', ["bigproj", null, 1], ["BigProj", null, 5]], '
    '"tables" : { "\\"public\\".\\"notes\\"" : {"scripts" : 1}, "\\"public\\".\\"chunks\\"" : {} }, '
    '"surfaces" : [{"surface_id" : "dev-laptop", "allowed_projects" : ["scripts", "acme", "bigproj"]}'
    ', {"surface_id" : "dev-full", "allowed_projects" : []}], '
    '"skills" : [{"name" : "triage", "scope" : "project:bigproj"}], '
    '"config_scopes" : {"project:bigproj" : 2}}\n'
    "ROLLBACK\n"
)


def test_inventory_from_psql_output():
    inv = rp.inventory_from_json(_PSQL_OUTPUT)
    assert inv.episode_groups == {
        ("acme", "/home/u/code/acme"): 3,
        ("scripts", "/home/u/code/acme/scripts"): 2,
        ("bigproj", None): 1,
        ("BigProj", None): 5,
    }
    assert inv.tables == {'"public"."notes"': {"scripts": 1}, '"public"."chunks"': {}}
    assert inv.surfaces == [
        {"surface_id": "dev-laptop", "allowed_projects": ["scripts", "acme", "bigproj"]},
        {"surface_id": "dev-full", "allowed_projects": []},
    ]
    assert inv.skills == [{"name": "triage", "scope": "project:bigproj"}]
    assert inv.config_scopes == {"project:bigproj": 2}


def test_inventory_from_json_tolerates_missing_and_null_sections():
    inv = rp.inventory_from_json('{"episodes": null, "surfaces": [{"surface_id": "s"}]}')
    assert (inv.episode_groups, inv.tables, inv.skills, inv.config_scopes) == ({}, {}, [], {})
    assert inv.surfaces == [{"surface_id": "s", "allowed_projects": []}]


def _no_connect(monkeypatch) -> None:
    def refuse(*_a, **_k):  # pragma: no cover - reaching it is the failure
        raise AssertionError("connected to a database")

    monkeypatch.setattr("psycopg.connect", refuse)
    monkeypatch.delenv("SYNAPSE_DB_URL", raising=False)


def test_inventory_sql_is_printed_without_a_database(monkeypatch, capsys):
    _no_connect(monkeypatch)
    assert rp.main(["--inventory-sql"]) == 0
    out = capsys.readouterr().out
    assert out == rp.INVENTORY_SQL.strip() + "\n"
    statements = [s.strip() for s in out.split(";") if s.strip()]
    assert statements[0] == "BEGIN READ ONLY"
    assert statements[-1] == "ROLLBACK"
    body = " ".join(statements[1:-1]).upper()
    assert not any(f" {verb} " in f" {body} " for verb in ("INSERT", "UPDATE", "DELETE", "DROP"))


def test_an_offline_dry_run_plans_from_the_inventory_file(tmp_path, monkeypatch, capsys):
    _no_connect(monkeypatch)
    inv_file = tmp_path / "inv.json"
    inv_file.write_text(_PSQL_OUTPUT, encoding="utf-8")
    out = tmp_path / "dry-run.md"
    aliases = tmp_path / "aliases.txt"
    aliases.write_text("bigproj = BigProj\n", encoding="utf-8")
    argv = ["--dry-run", "--inventory", str(inv_file), "--aliases", str(aliases)]
    assert rp.main([*argv, "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert out.read_text(encoding="utf-8") == text
    inv = rp.inventory_from_json(_PSQL_OUTPUT)
    cfg = rp.AliasConfig.parse(aliases.read_text(encoding="utf-8"))
    assert text == rp.report(inv, rp.Planner(cfg).plan(inv.episode_groups, inv.other_labels()))
    assert "| scripts | /home/u/code/acme | acme | 2 | subfolder → project root |" in text
    assert "| bigproj |  | BigProj | 1 | alias |" in text
    assert "  dev-laptop: ['scripts', 'acme', 'bigproj'] -> ['scripts', 'acme', 'BigProj']" in text
    assert "        2  project:bigproj -> project:BigProj" in text


def test_an_inventory_file_never_goes_with_apply(tmp_path, monkeypatch, capsys):
    _no_connect(monkeypatch)
    inv_file = tmp_path / "inv.json"
    inv_file.write_text(_PSQL_OUTPUT, encoding="utf-8")
    assert rp.main(["--apply", "--inventory", str(inv_file)]) == 2
    assert "only goes with --dry-run" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The command line, short of a database
# ---------------------------------------------------------------------------


def test_main_refuses_to_run_without_a_database_url(monkeypatch, capsys):
    monkeypatch.delenv("SYNAPSE_DB_URL", raising=False)
    assert rp.main(["--dry-run"]) == 2
    assert "no database URL" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [[], ["--dry-run", "--apply"]], ids=["no-mode", "both-modes"])
def test_main_needs_exactly_one_mode(argv):
    with pytest.raises(SystemExit) as exc:
        rp.main([*argv, "--db-url", "postgresql://unused/never_test"])
    assert exc.value.code == 2


def test_a_bad_alias_file_stops_before_touching_the_database(tmp_path, monkeypatch):
    bad = tmp_path / "aliases.txt"
    bad.write_text("old = new/path\n", encoding="utf-8")

    def _no_connect(*_a, **_k):  # pragma: no cover - reaching it is the failure
        raise AssertionError("connected despite a bad alias file")

    monkeypatch.setattr("psycopg.connect", _no_connect)
    with pytest.raises(ValueError, match="line 1"):
        rp.main(["--apply", "--aliases", str(bad), "--db-url", "postgresql://unused/x_test"])


# ---------------------------------------------------------------------------
# Allowlists never widen; the report shows roots and skipped tables
# ---------------------------------------------------------------------------


def test_an_allowlisted_subfolder_is_never_widened_to_its_whole_project():
    """A device allowed only `<repo>\\tickets` must not end up allowed the whole repo: the
    entry is kept as it is (after the relabel it matches nothing: fail-closed) and the
    dry run says so. Allowed through another entry, the subfolder entry just merges."""
    plan = _plan({(WIN, WIN): 10, (WIN + r"\tickets", WIN + r"\tickets"): 3}, aliases="")
    only_sub = [WIN + r"\tickets", "other-proj"]
    assert rp.map_allowlist(plan, only_sub) == only_sub
    assert rp.widening_entries(plan, only_sub) == [(WIN + r"\tickets", "BigProj")]
    both = [WIN + r"\tickets", WIN, "bigproj"]
    assert rp.map_allowlist(plan, both) == ["BigProj"]
    assert rp.widening_entries(plan, both) == []
    # a path, a case variant or an alias of the project itself is the same project (an
    # allowlist's entries are planned too: Inventory.other_labels includes them)
    plan = _plan({(WIN, WIN): 10}, aliases="", other=[WIN_LC, "BIGPROJ"])
    assert rp.map_allowlist(plan, [WIN_LC, "BIGPROJ"]) == ["BigProj"]


def test_the_report_flags_a_widening_allowlist_entry_and_lists_absorbing_roots():
    groups = {(WIN, WIN): 10, (WIN + r"\tickets", WIN + r"\tickets"): 3}
    plan = _plan(groups, aliases="")
    surfaces = [{"surface_id": "dev-sub-only", "allowed_projects": [WIN + r"\tickets"]}]
    text = rp.report(rp.Inventory(groups, {}, surfaces, []), plan)
    assert "dev-sub-only" in text and "is part of 'BigProj'" in text
    assert "project roots that absorb other labels" in text
    assert f"{WIN_ROOT}  <-  1 label(s): {WIN}\\tickets" in text


def test_the_report_marks_a_table_apply_will_skip():
    groups = {(WIN, WIN): 1}
    inv = rp.Inventory(groups, {'"x"."uniq"': {WIN: 2}}, [], [], {}, {'"x"."uniq"'})
    text = rp.report(inv, _plan(groups, aliases=""))
    assert '"x"."uniq": 2 rows WOULD change, but --apply SKIPS this table' in text
