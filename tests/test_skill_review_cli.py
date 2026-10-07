"""plugin/scripts/skill_review.py — the review CLI over the accept-applies routes.

No DB, no network: config.post_json is stubbed with canned server replies. Guards what the
reviewer sees and sends: show renders a unified diff against the registry body (the full
body for a new skill), accept forwards --body-file/--scope/--force and syncs this machine
right away when it is the destination, refusals exit non-zero, promote is a no-op.
All skills are synthetic.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "plugin", "scripts"))
import skill_review

config = skill_review.config  # patch the module object the CLI actually holds

V1 = "---\nname: demo-backup\ndescription: Back up the demo db.\n---\n1. Dump the db.\n"
V2 = "---\nname: demo-backup\ndescription: Back up the demo db.\n---\n1. Dump and verify.\n"


def _detail(**over):
    d = {
        "found": True,
        "id": 7,
        "kind": "retune",
        "name": "demo-backup",
        "direction": "fix",
        "status": "proposed",
        "score": 1.5,
        "summary": "skips verification",
        "target_skills": ["demo-backup"],
        "evidence": [],
        "proposed_patch": "- verify the dump",
        "proposal_body": V2,
        "registry_body": V1,
        "apply_to": {"skill": "demo-backup", "scope": "global", "exists": True},
        "apply_issues": [],
    }
    d.update(over)
    return d


@pytest.fixture()
def server(monkeypatch):
    """Records every POST; replies come from `server.replies[path]` (dict or callable)."""
    sent: list[tuple[str, dict]] = []
    replies: dict = {}

    def post_json(path, payload, timeout=30.0):
        sent.append((path, payload))
        r = replies[path]
        return r(payload) if callable(r) else r

    monkeypatch.setattr(config, "post_json", post_json)
    monkeypatch.setattr(config, "SKILLS_SYNC", False)
    return types.SimpleNamespace(sent=sent, replies=replies)


@pytest.fixture()
def fake_sync(monkeypatch):
    calls: list[tuple[str, Path]] = []
    mod = types.ModuleType("skills_sync")
    mod._sync = lambda scope, target: calls.append((scope, target)) or (1, 0)
    monkeypatch.setitem(sys.modules, "skills_sync", mod)
    return calls


def test_show_prints_a_unified_diff_against_the_registry(server, capsys):
    server.replies["/skills/proposals"] = _detail()
    skill_review.cmd_show(7)
    out = capsys.readouterr().out
    assert "accept applies to: 'demo-backup' (global) — updates the existing skill" in out
    assert "--- registry/demo-backup/SKILL.md" in out
    assert "+++ proposal/demo-backup/SKILL.md" in out
    assert "-1. Dump the db." in out and "+1. Dump and verify." in out
    assert " description: Back up the demo db." in out  # unchanged line as diff context


def test_show_prints_the_full_body_for_a_new_skill(server, capsys):
    server.replies["/skills/proposals"] = _detail(
        kind="derive",
        registry_body=None,
        apply_to={"skill": "demo-backup", "scope": "project:demo", "exists": False},
    )
    skill_review.cmd_show(7)
    out = capsys.readouterr().out
    assert "creates a new skill" in out and "(project:demo)" in out
    assert "--- new skill: demo-backup/SKILL.md ---\n" + V2 in out


def test_show_without_a_draft_and_with_warnings(server, capsys):
    server.replies["/skills/proposals"] = _detail(
        proposal_body=None,
        apply_issues=[
            {"code": "no_draft", "detail": "no draft yet", "force": False},
            {"code": "stale", "detail": "the registry body changed", "force": True},
        ],
    )
    skill_review.cmd_show(7)
    out = capsys.readouterr().out
    assert "no draft yet; re-drafts on the next nightly run" in out
    assert "! the registry body changed [--force overrides]" in out
    assert "--- registry/" not in out


def test_accept_forwards_body_file_scope_and_force(server, tmp_path, capsys):
    f = tmp_path / "SKILL.md"
    f.write_text(V2, encoding="utf-8")
    server.replies["/skills/proposals/act"] = {
        "status": "promoted",
        "skill": "demo-backup",
        "scope": "project:demo",
        "created": False,
        "forced": ["stale"],
    }
    skill_review.cmd_accept(7, str(f), "project:demo", True)
    path, payload = server.sent[-1]
    assert path == "/skills/proposals/act"
    assert payload == {
        "id": 7,
        "action": "accept",
        "body": V2,
        "scope": "project:demo",
        "force": True,
    }
    out = capsys.readouterr().out
    assert "updated skill 'demo-backup' (project:demo)" in out
    assert "--force overrode: stale" in out


def test_accept_global_syncs_this_machine_now(server, fake_sync, monkeypatch, capsys):
    monkeypatch.setattr(config, "SKILLS_SYNC", True)
    server.replies["/skills/proposals/act"] = {
        "status": "promoted",
        "skill": "demo-backup",
        "scope": "global",
        "created": True,
    }
    skill_review.cmd_accept(7, None, None, False)
    assert server.sent[-1][1] == {"id": 7, "action": "accept"}
    assert fake_sync == [("global", config.SKILLS_DIR)]
    assert "Synced here now: pulled 1, pushed 0" in capsys.readouterr().out


def test_accept_project_scope_syncs_only_inside_that_project(
    server, fake_sync, monkeypatch, tmp_path
):
    monkeypatch.setattr(config, "SKILLS_SYNC", True)
    monkeypatch.setattr(config, "SKILLS_DIR", tmp_path / "home" / ".claude" / "skills")
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(tmp_path / "demo"))
    reply = {"status": "promoted", "skill": "demo-backup", "created": False}

    server.replies["/skills/proposals/act"] = {**reply, "scope": "project:elsewhere"}
    skill_review.cmd_accept(7, None, None, False)
    assert fake_sync == []  # another machine's project: it pulls at its next session start

    server.replies["/skills/proposals/act"] = {**reply, "scope": "project:demo"}
    skill_review.cmd_accept(7, None, None, False)
    assert fake_sync == [("project:demo", tmp_path / "demo" / ".claude" / "skills")]


def test_accept_with_sync_off_never_syncs(server, fake_sync, capsys):
    server.replies["/skills/proposals/act"] = {
        "status": "promoted",
        "skill": "demo-backup",
        "scope": "global",
        "created": False,
    }
    skill_review.cmd_accept(7, None, None, False)
    assert fake_sync == []
    assert "next session start" in capsys.readouterr().out


def test_accept_refusal_exits_nonzero_with_the_reason(server):
    server.replies["/skills/proposals/act"] = {
        "status": "refused",
        "detail": "no draft yet; re-drafts on the next nightly run, or pass --body-file",
    }
    with pytest.raises(SystemExit) as e:
        skill_review.cmd_accept(7, None, None, False)
    assert "refusing [7]: no draft yet" in str(e.value.code)


def test_promote_is_a_noop(server, capsys):
    server.replies["/skills/proposals/act"] = {
        "status": "promoted",
        "noop": True,
        "detail": "[7] demo-backup was applied at accept; nothing to promote",
    }
    skill_review.cmd_promote(7)
    assert server.sent == [("/skills/proposals/act", {"id": 7, "action": "promote"})]
    assert "applied at accept" in capsys.readouterr().out
