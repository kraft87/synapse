"""scripts/rename_owner_in_facts.py: which words it may touch, and what the classifier decides.

No database. The script numbers each "user" word of a fact, a classifier names the numbers
that mean the memory's owner, and exactly those are replaced by the owner's name. These
tests pin the three parts that keep a wrong rewrite out: which words are numbered at all,
that nothing but a picked word changes, and that a reply which cannot be lined up with its
facts renames nothing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from ingestion.llm_client import UsageLimitError

_SPEC = importlib.util.spec_from_file_location(
    "rename_owner_in_facts",
    Path(__file__).resolve().parent.parent / "scripts" / "rename_owner_in_facts.py",
)
ro = importlib.util.module_from_spec(_SPEC)
sys.modules["rename_owner_in_facts"] = ro
_SPEC.loader.exec_module(ro)


@pytest.mark.parametrize(
    ("fact", "words"),
    [
        ("User owns a 2010 Mazda 3", ["User"]),
        ("The user has been unable to get over it", ["The user"]),
        ("Friend does not notice User's mood.", ["User's"]),
        ("It leaves user frozen and numbs the user.", ["user", "the user"]),
        ("the user\u2019s token is checked", ["the user\u2019s"]),
        # Part of a longer word, an identifier, a path, an address or a hyphenated compound.
        ("the admin panel cannot see what users are working on", []),
        ("USER_IDS holds user_id values; see USER.md", []),
        ("end-user docs and user-facing text", []),
        ("@user mentions and /home/user/ paths", []),
        ("UserPromptSubmit hook fires per prompt", []),
        # Some user, not this one: a name cannot follow an indefinite determiner.
        ("a user's token is checked and each user sees one user at a time", []),
        ("No user is shown another user's notes", []),
        # A part of a conversation or a system, unless it is somebody's.
        ("the user turn, a User Message, user input and the user account", []),
        ("the default urllib user agent and the user ID", []),
        ("User's account was locked and the user's role changed", ["User's", "the user's"]),
        ("per user request, the user data stays", ["user", "the user"]),
    ],
)
def test_which_words_are_marked(fact: str, words: list[str]) -> None:
    assert [m.group(0) for m in ro.marks(fact)] == words


def test_marked_numbers_each_word_in_order() -> None:
    assert (
        ro.marked("User's MR was held because user's boss asked, which user disliked")
        == "[[1: User's]] MR was held because [[2: user's]] boss asked, which [[3: user]] disliked"
    )


def test_rename_replaces_only_the_picked_words() -> None:
    fact = "A bot told the user what User's script does; the user agreed."
    assert (
        ro.rename(fact, [2, 3], "Sam") == "A bot told the user what Sam's script does; Sam agreed."
    )
    assert ro.rename(fact, [], "Sam") == fact
    # A number that is not a mark of this fact changes nothing.
    assert ro.rename(fact, [9], "Sam") == fact


def test_rename_keeps_the_possessive_and_drops_the_article() -> None:
    assert ro.rename("The user's sister visited the user", [1, 2], "Sam") == (
        "Sam's sister visited Sam"
    )


def test_rename_drops_a_word_that_stands_right_before_the_name() -> None:
    """A fact reading "User Sam asked" names the owner twice: the name already there is kept."""
    assert ro.rename("User Sam asked about User Sam's plan", [1, 2], "Sam") == (
        "Sam asked about Sam's plan"
    )
    assert ro.rename("logs in as user sam", [1], "Sam") == "logs in as sam"
    # Only the name itself: a longer word that starts with it is not the name.
    assert ro.rename("User Samuel asked", [1], "Sam") == "Sam Samuel asked"


def _rows(*facts: str) -> list[dict[str, Any]]:
    return [{"uuid": f"u{i}", "fact": f} for i, f in enumerate(facts, start=1)]


def _reply(*pairs: tuple[int, list[int]]) -> Any:
    return ro.OwnerMarksReply(facts=[ro.FactMarks(n=n, owner=o) for n, o in pairs])


def test_classify_batch_maps_line_numbers_back_to_facts(monkeypatch) -> None:
    seen: list[str] = []

    def fake(_llm, *, base_prompt, **_kw):
        seen.append(base_prompt)
        return _reply((1, [1]), (2, []))

    monkeypatch.setattr(ro, "structured_call", fake)
    rows = _rows("User owns a car", "the user who signs in sees the banner")
    assert ro.classify_batch(object(), rows, owner="Sam", model="m") == {"u1": [1], "u2": []}
    assert "The owner's name is Sam." in seen[0]
    assert "1|[[1: User]] owns a car" in seen[0]
    assert "2|[[1: the user]] who signs in sees the banner" in seen[0]


def test_a_reply_that_skips_a_line_is_split_until_each_fact_answers(monkeypatch) -> None:
    """A skipped or repeated number is the one sign that later verdicts belong to other
    facts, so the whole reply is dropped and the facts are asked again in halves."""
    sizes: list[int] = []

    def fake(_llm, *, base_prompt, **_kw):
        n = base_prompt.split("Facts:\n")[1].split("\n\nReply")[0].count("\n") + 1
        sizes.append(n)
        return _reply((1, [1])) if n == 1 else _reply((1, [1]), (1, [1]))

    monkeypatch.setattr(ro, "structured_call", fake)
    rows = _rows("User owns a car", "User plays piano")
    assert ro.classify_batch(object(), rows, owner="Sam", model="m") == {"u1": [1], "u2": [1]}
    assert sizes == [2, 1, 1]


def test_a_fact_that_never_classifies_is_left_out(monkeypatch) -> None:
    def fake(_llm, **_kw):
        raise RuntimeError("bad gateway")

    monkeypatch.setattr(ro, "structured_call", fake)
    assert ro.classify_batch(object(), _rows("User owns a car"), owner="Sam", model="m") == {}


def test_a_usage_limit_stops_the_run(monkeypatch) -> None:
    def fake(_llm, **_kw):
        raise UsageLimitError("limit")

    monkeypatch.setattr(ro, "structured_call", fake)
    with pytest.raises(UsageLimitError):
        ro.classify_batch(object(), _rows("User owns a car"), owner="Sam", model="m")


def test_propose_plans_only_facts_whose_text_changes() -> None:
    rows = _rows(
        "User owns a car",
        "a user who signs in sees the banner",
        "User's plan changed",
        "User asked something",
    )
    verdicts = {"u1": [1], "u2": [], "u3": [1]}  # u4: the classifier never answered

    def classify(chunk):
        return {r["uuid"]: verdicts[r["uuid"]] for r in chunk if r["uuid"] in verdicts}

    plan, counts = ro.propose(rows, classify, "Sam", batch=3)
    assert plan == [
        {"uuid": "u1", "old": "User owns a car", "new": "Sam owns a car"},
        {"uuid": "u3", "old": "User's plan changed", "new": "Sam's plan changed"},
    ]
    assert counts == {"facts": 4, "renamed": 2, "unchanged": 1, "unclassified": 1, "disagreed": 0}


def test_propose_renames_only_what_two_passes_in_different_orders_agree_on() -> None:
    """A verdict leans toward its neighbours' in the batch, so each fact is classified twice
    among different neighbours and one pass alone renames nothing."""
    rows = _rows(*(f"User did thing {i} for the user" for i in range(1, 9)))
    seen: list[list[str]] = []

    def classify(chunk):
        seen.append([r["uuid"] for r in chunk])
        first_pass = len(seen) <= 2
        # u2: mark 2 is named by the first pass only. u3: answered by the first pass only.
        return {
            r["uuid"]: [1, 2] if (first_pass and r["uuid"] == "u2") else [1]
            for r in chunk
            if first_pass or r["uuid"] != "u3"
        }

    plan, counts = ro.propose(rows, classify, "Sam", batch=4)
    first, second = seen[0] + seen[1], seen[2] + seen[3]
    assert first == [f"u{i}" for i in range(1, 9)]
    assert sorted(second) == sorted(first) and second != first
    assert {p["uuid"]: p["new"] for p in plan}["u2"] == "Sam did thing 2 for the user"
    assert "u3" not in {p["uuid"] for p in plan}
    assert counts == {"facts": 8, "renamed": 7, "unchanged": 0, "unclassified": 1, "disagreed": 1}


def test_backup_table_name_takes_only_a_date() -> None:
    assert ro.backup_table("20990101") == "kg_owner_rename_20990101"
    with pytest.raises(ValueError):
        ro.backup_table('x"; DROP TABLE kg_relationships; --')
