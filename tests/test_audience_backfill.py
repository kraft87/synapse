"""The one-time backfill tooling (scripts/audience_backfill.py) — pure logic only.

Every pre-053 note is 'personal'. That is safe and useless, so someone has to say which
notes are work-safe. The script's whole design is that a HUMAN makes that call: the
heuristic writes a draft table, a person edits it, and a second pass applies what the
file says. What's worth pinning is therefore the two halves that could silently get it
wrong — the draft's conservatism, and the parser that reads the edited file back.

No DB here: --propose and --apply are never run against a real store in tests (the real
backfill is a manual, reviewed, one-time operation).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "audience_backfill", Path(__file__).resolve().parent.parent / "scripts" / "audience_backfill.py"
)
backfill = importlib.util.module_from_spec(_SPEC)
sys.modules["audience_backfill"] = backfill
_SPEC.loader.exec_module(backfill)


# ---------------------------------------------------------------------------
# The heuristic draft
# ---------------------------------------------------------------------------


def test_project_notes_on_work_readable_projects_are_proposed_work_safe():
    note = {"type": "project", "project": "alpha", "hook": "Alpha runs on Postgres"}
    assert backfill._propose_audience(note, {"alpha"}) == "work-safe"


@pytest.mark.parametrize(
    "note",
    [
        {"type": "project", "project": "gamma", "hook": "Gamma uses SQLite"},
        {"type": "project", "project": None, "hook": "Unscoped project note"},
        {"type": "user", "project": "alpha", "hook": "User prefers tabs"},
        {"type": "feedback", "project": "alpha", "hook": "User dislikes tables"},
        {"type": "reference", "project": "alpha", "hook": "See the runbook"},
    ],
    ids=["off-allowlist", "no-project", "user", "feedback", "reference"],
)
def test_everything_else_is_proposed_personal(note):
    """Deliberately conservative. A GLOBAL note stays personal even when it reads as
    technical: nothing in the row distinguishes 'User prefers tabs' from a note about
    the user's health, and the cost of guessing wrong is one-directional."""
    assert backfill._propose_audience(note, {"alpha"}) == "personal"


def test_cells_cannot_break_the_table_grammar():
    """A hook containing a pipe or a newline would otherwise split a row in two and
    silently re-key every column after it."""
    assert backfill._cell("a | b") == "a \\| b"
    assert backfill._cell("line one\nline two") == "line one line two"
    assert backfill._cell(None) == ""


# ---------------------------------------------------------------------------
# Reading the reviewed file back
# ---------------------------------------------------------------------------


_REVIEWED = """# Audience review

Prose above the table is ignored, including | pipes | in it.

| id | audience | type | project | hook |
| --- | --- | --- | --- | --- |
| 1 | work-safe | project | alpha | Alpha runs on Postgres |
| 2 | personal | user | | User keeps a journal |
| 30 |work-safe| project | beta | Beta ships weekly |
"""


def test_parse_review_reads_edited_rows():
    assert backfill.parse_review(_REVIEWED) == [
        (1, "work-safe"),
        (2, "personal"),
        (30, "work-safe"),
    ]


def test_parse_review_ignores_the_separator_and_header_rows():
    assert backfill.parse_review("| id | audience |\n| --- | --- |\n") == []


def test_parse_review_rejects_an_unknown_tier():
    """A typo must stop the run. Falling back to a default would apply a tag the
    reviewer never chose — in whichever direction the default happens to point."""
    bad = "| id | audience |\n| --- | --- |\n| 7 | worksafe | project | alpha | H |\n"
    with pytest.raises(ValueError, match="invalid audience 'worksafe'"):
        backfill.parse_review(bad)
