"""Nightly drafting of ready-to-apply SKILL.md bodies (dream/skills/skill_doc + skill_draft).

No DB, no LLM: the drafting model is monkeypatched. Guards the contract accept relies on:
a widen draft is the current body with ONLY the frontmatter description changed (verified
line by line, else no draft), a revision keeps the skill's name and a sane size, a new-skill
draft is a usable SKILL.md, and a stale retune draft is picked up for re-drafting.
All skills are synthetic.
"""

from __future__ import annotations

import json

import pytest

import dream.skills.skill_derive as SD
import dream.skills.skill_doc as D
import dream.skills.skill_draft as DR
from dream.skills.skill_measure import skill_description

_BODY = (
    "---\n"
    "name: demo-backup\n"
    "description: Run the nightly backup of the demo database to the archive bucket.\n"
    "allowed-tools: Bash\n"
    "---\n"
    "# demo-backup\n"
    "\n"
    "1. Dump the database with `pg_dump`.\n"
    "2. Upload the dump to the archive bucket.\n"
)

_WIDEN_EV = [
    {
        "session_id": "s-1",
        "class": "judge",
        "signal": "under_trigger",
        "skill": "demo-backup",
        "phrasing": "can you snapshot the db before the migration",
        "missing_phrasing": "snapshot the db",
    }
]


def _stub_llm(monkeypatch, reply):
    calls: list[str] = []

    def fake(prompt):
        calls.append(prompt)
        return reply(prompt) if callable(reply) else reply

    monkeypatch.setattr(SD, "_draft_call", fake)
    return calls


# ------------------------------------------------------------------ skill_doc


def test_body_hash_is_sha256_and_none_for_no_body():
    assert D.body_hash(None) is None
    assert D.body_hash("x") == D.body_hash("x") != D.body_hash("y")
    assert len(D.body_hash("")) == 64


def test_frontmatter_name_and_body_text():
    assert D.frontmatter_name(_BODY) == "demo-backup"
    assert D.body_text(_BODY).startswith("# demo-backup")
    assert D.frontmatter_name("# no frontmatter\n") == ""


@pytest.mark.parametrize(
    "old_line,new_value",
    [
        ("description: Plain old trigger text.", "Plain new trigger text, snapshot the db."),
        ('description: "Quoted old text."', "Quoted new text, snapshot the db."),
        ("description: 'Single quoted old.'", 'Says "snapshot the db" now.'),
    ],
)
def test_widen_draft_only_touches_description(old_line, new_value):
    old = _BODY.replace(
        "description: Run the nightly backup of the demo database to the archive bucket.",
        old_line,
    )
    draft = D.widen_draft(old, new_value)
    assert draft is not None
    assert skill_description(draft) == new_value
    assert D.only_description_changed(old, draft)
    # every non-description line is byte-identical
    keep = lambda t: [ln for ln in t.splitlines() if not ln.startswith("description:")]  # noqa: E731
    assert keep(draft) == keep(old)


def test_widen_draft_handles_block_scalar_descriptions():
    old = _BODY.replace(
        "description: Run the nightly backup of the demo database to the archive bucket.\n",
        "description: >-\n  Run the nightly backup\n  of the demo database.\n",
    )
    draft = D.widen_draft(old, "Run or snapshot the demo database backup.")
    assert draft is not None
    assert skill_description(draft) == "Run or snapshot the demo database backup."
    assert "allowed-tools: Bash\n" in draft and D.body_text(draft) == D.body_text(old)


@pytest.mark.parametrize(
    "new_value",
    [
        "",  # nothing
        "two\nlines",  # not one line
        "x" * (D.MAX_DESCRIPTION + 1),  # too long
        "Run the nightly backup of the demo database to the archive bucket.",  # unchanged
        '"starts with a quote',  # can't be written plain without changing what reads back
    ],
)
def test_widen_draft_refuses_unsafe_descriptions(new_value):
    assert D.widen_draft(_BODY, new_value) is None


def test_only_description_changed_rejects_any_other_edit():
    edited = _BODY.replace("description: Run", "description: Snapshot or run").replace(
        "1. Dump", "1. Carefully dump"
    )
    assert not D.only_description_changed(_BODY, edited)
    renamed = _BODY.replace("description: Run", "description: Snapshot or run").replace(
        "name: demo-backup", "name: demo-backups"
    )
    assert not D.only_description_changed(_BODY, renamed)
    assert not D.only_description_changed(_BODY, _BODY)  # description must actually change
    assert not D.only_description_changed("# no frontmatter\n", "# no frontmatter\n")


def test_revision_problem_guards_name_body_and_size():
    good = _BODY + "3. Verify the upload checksum.\n"
    assert D.revision_problem(_BODY, good) is None
    assert "name changed" in D.revision_problem(
        _BODY, good.replace("name: demo-backup", "name: other")
    )
    assert "no body" in D.revision_problem(_BODY, _BODY.split("# demo-backup")[0])
    assert "identical" in D.revision_problem(_BODY, _BODY)
    assert "chars against" in D.revision_problem(_BODY, _BODY + "x" * 10_000)
    assert "chars against" in D.revision_problem(_BODY + "y" * 2000, _BODY)


def test_new_skill_problem():
    assert D.new_skill_problem(_BODY) is None
    assert D.new_skill_problem("# no frontmatter at all\n")
    assert D.new_skill_problem(_BODY.replace("name: demo-backup", "name: Demo Backup"))


# ------------------------------------------------------------------ skill_draft


def test_draft_widen_rewrites_only_the_description(monkeypatch):
    new_desc = "Run or snapshot the demo database backup to the archive bucket."
    calls = _stub_llm(monkeypatch, json.dumps({"description": new_desc}))
    row = {"id": 1, "kind": "retune", "direction": "widen", "name": "demo-backup"}
    row["target_skills"], row["evidence"] = ["demo-backup"], _WIDEN_EV
    out = DR.draft(row, _BODY)
    assert out is not None
    body, base = out
    assert base == D.body_hash(_BODY)
    assert skill_description(body) == new_desc
    assert D.only_description_changed(_BODY, body)
    assert "snapshot the db" in calls[0]  # the judge's missing_phrasing reached the prompt


@pytest.mark.parametrize(
    "reply",
    [
        "not json at all",
        json.dumps({"description": ""}),
        json.dumps({"description": "line one\nline two"}),
        # a model that rewrites the whole file instead of the description gets no draft
        json.dumps({"description": _BODY}),
    ],
)
def test_draft_widen_falls_back_to_no_draft(monkeypatch, reply):
    _stub_llm(monkeypatch, reply)
    assert DR.draft_widen("demo-backup", _BODY, _WIDEN_EV) is None


def test_draft_widen_needs_a_stated_gap(monkeypatch):
    calls = _stub_llm(monkeypatch, json.dumps({"description": "whatever"}))
    assert DR.draft_widen("demo-backup", _BODY, [{"signal": "dismissal"}]) is None
    assert calls == []  # no gap, no LLM call


def test_draft_revision_applies_patch_and_checks_result(monkeypatch):
    revised = _BODY + "3. Verify the upload checksum.\n"
    calls = _stub_llm(monkeypatch, f"```markdown\n{revised}```")
    row = {
        "id": 2,
        "kind": "retune",
        "direction": "extend",
        "name": "demo-backup",
        "target_skills": ["demo-backup"],
        "summary": "skips verification",
        "proposed_patch": "- add a checksum verification step",
        "evidence": [],
    }
    body, base = DR.draft(row, _BODY)
    assert body == revised and base == D.body_hash(_BODY)
    assert "add a checksum verification step" in calls[0] and _BODY in calls[0]

    # a revision that renames the skill is not offered
    _stub_llm(monkeypatch, revised.replace("name: demo-backup", "name: renamed"))
    assert DR.draft(row, _BODY) is None


def test_draft_revision_falls_back_to_evidence_without_a_patch(monkeypatch):
    calls = _stub_llm(monkeypatch, _BODY + "Do not run this for ad-hoc exports.\n")
    row = {
        "id": 3,
        "kind": "retune",
        "direction": "narrow",
        "name": "demo-backup",
        "target_skills": ["demo-backup"],
        "summary": "fired on ad-hoc exports",
        "proposed_patch": None,
        "evidence": [{"signal": "dismissal", "why": "fired on an ad-hoc export"}],
    }
    assert DR.draft(row, _BODY) is not None
    assert "fired on an ad-hoc export" in calls[0]


def test_draft_derive_and_unusable_reply(monkeypatch):
    _stub_llm(monkeypatch, _BODY)
    row = {
        "id": 4,
        "kind": "derive",
        "name": "explicit:abcd1234",
        "summary": "back up the demo db",
        "evidence": [{"session_id": "s-1", "quote": "make this a skill"}],
    }
    assert DR.draft(row, None) == (_BODY, None)
    _stub_llm(monkeypatch, "Sorry, I can't help with that.")
    assert DR.draft(row, None) is None


def test_consolidate_and_missing_registry_body_are_never_drafted(monkeypatch):
    calls = _stub_llm(monkeypatch, _BODY)
    assert DR.draft({"id": 5, "kind": "consolidate", "name": "a+b"}, None) is None
    row = {"id": 6, "kind": "retune", "direction": "fix", "name": "gone", "target_skills": []}
    assert DR.draft(row, None) is None
    assert calls == []


def test_needs_draft_covers_bodyless_and_stale_retunes():
    base = D.body_hash(_BODY)
    assert DR.needs_draft({"kind": "derive", "proposal_body": None})
    assert not DR.needs_draft({"kind": "derive", "proposal_body": _BODY})
    fresh = {"kind": "retune", "proposal_body": "x", "base_body_hash": base, "registry_body": _BODY}
    assert not DR.needs_draft(fresh)
    assert DR.needs_draft({**fresh, "registry_body": _BODY + "edited\n"})
    assert not DR.needs_draft({**fresh, "base_body_hash": None})  # never pinned: leave it
