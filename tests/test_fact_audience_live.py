"""Live check of the fact-audience classifier on the configured provider.

Never part of CI (``integration``): it spends real LLM calls. Run it by hand, against
whatever ``stage_model("FACT_AUDIENCE")`` resolves to, before trusting the labels for
anything beyond phase 1::

    uv run pytest tests/test_fact_audience_live.py -m integration -p no:xdist

The facts are invented, of the two kinds a label has to catch even though they read as
ordinary technical or work text: a private tactic about how one's work looks to one's
employer, and a health mention inside a technical note. A failure here is a reason to amend
the prompt's rules and bump ``LABEL_VERSION``, not to edit the test.
"""

from __future__ import annotations

import pytest

from ingestion import fact_audience as fa
from ingestion.llm_client import create_llm_client

pytestmark = pytest.mark.integration

PRIVATE_FACTS = [
    "User suspects the team lead will blame him if the vendor query he did not write breaks in production",
    "User moved the ticket to Blocked so the delay shows against the other team",
    "The CI runner kept timing out the day the user was off sick with the flu",
    "User's migraine medication makes late on-call debugging harder, per the runbook notes",
]

WORK_SAFE_FACTS = [
    "The nightly build failed because a dependency dropped support for an old Python version",
    "Postgres VACUUM reclaims space held by dead rows",
]


def test_the_classifier_withholds_the_private_facts_and_passes_the_plain_ones():
    rows = [
        {"id": i, "fact": text}
        for i, text in enumerate([*PRIVATE_FACTS, *WORK_SAFE_FACTS], start=1)
    ]
    out = fa.classify_batch(create_llm_client(), rows)
    got = {r["fact"]: out[r["id"]].audience for r in rows}
    for text in PRIVATE_FACTS:
        assert got[text] == "personal", text
    for text in WORK_SAFE_FACTS:
        assert got[text] == "work-safe", text
