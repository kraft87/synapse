"""Web audience classifier and precedence, without a database (schema 063).

The contract under test: ``work-safe`` is only ever the answer when the precedence or
every classifier vote says so cleanly. Every other outcome (a named signal, ``unsure``,
a reply that never validates) is ``personal``; a transport failure is raised so the
caller leaves the row unlabelled (NULL, never served to a restricted surface).
"""

from __future__ import annotations

from typing import Any

import pytest

from ingestion import web_audience as wa
from ingestion.llm_client import LLMHTTPError, MalformedResponseError, UsageLimitError
from ingestion.web_audience import (
    AudienceDecision,
    LabelStats,
    WebAudienceVerdict,
    classify,
    derive_web_audience,
    excerpt,
)


class _Votes:
    """Stands in for ``structured_call``: hands out the queued replies in order and
    records every prompt it was sent."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, llm: Any, **kw: Any) -> WebAudienceVerdict:
        self.prompts.append(kw["base_prompt"])
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


def _v(verdict: str, *signals: str) -> WebAudienceVerdict:
    return WebAudienceVerdict(verdict=verdict, personal_signals=list(signals))


def _classify(monkeypatch, votes: _Votes, n: int = 3, **doc: Any) -> AudienceDecision:
    monkeypatch.setattr(wa, "structured_call", votes)
    fields = {"kind": "web_scrape", "title": "t", "url": "https://example.org/x", "content": "c"}
    fields.update(doc)
    return classify(object(), votes=n, **fields)


def test_work_safe_needs_every_vote_clean(monkeypatch):
    votes = _Votes(_v("work-safe"), _v("work-safe"), _v("work-safe"))
    d = _classify(monkeypatch, votes)
    assert d == AudienceDecision("work-safe", "classifier", "3/3 votes")
    assert len(votes.prompts) == 3


def test_first_personal_vote_ends_the_poll(monkeypatch):
    votes = _Votes(_v("personal", "a private topic"), _v("work-safe"))
    d = _classify(monkeypatch, votes)
    assert d.audience == "personal" and d.source == "classifier"
    assert d.detail == "vote 1: a private topic"
    assert len(votes.prompts) == 1


def test_a_later_dissenting_vote_makes_it_personal(monkeypatch):
    """The per-call slip voting exists for: a borderline page answered work-safe once,
    then personal."""
    votes = _Votes(_v("work-safe"), _v("personal", "a family member"), _v("work-safe"))
    d = _classify(monkeypatch, votes)
    assert d.audience == "personal" and d.detail.startswith("vote 2: ")
    assert len(votes.prompts) == 2


def test_work_safe_verdict_with_a_signal_is_personal(monkeypatch):
    d = _classify(monkeypatch, _Votes(_v("work-safe", "a purchase decision")), n=1)
    assert d.audience == "personal" and "purchase" in d.detail


@pytest.mark.parametrize("verdict", ["unsure", "personal"])
def test_unsure_or_personal_without_signals_is_personal(monkeypatch, verdict):
    d = _classify(monkeypatch, _Votes(_v(verdict)), n=1)
    assert d == AudienceDecision("personal", "classifier", f"vote 1: {verdict}")


def test_blank_signals_do_not_count(monkeypatch):
    d = _classify(monkeypatch, _Votes(_v("work-safe", "", "  ")), n=1)
    assert d.audience == "work-safe"


def test_a_reply_that_never_validates_is_personal(monkeypatch):
    d = _classify(monkeypatch, _Votes(MalformedResponseError("bad json", raw_response="{")), n=3)
    assert d.audience == "personal" and d.source == "classifier-fallback"


@pytest.mark.parametrize(
    "err", [LLMHTTPError("401", status_code=401), UsageLimitError("quota")], ids=str
)
def test_transport_errors_propagate(monkeypatch, err):
    """No label at all: the caller leaves the row NULL and a later run retries it."""
    with pytest.raises(type(err)):
        _classify(monkeypatch, _Votes(_v("work-safe"), err), n=3)


def test_votes_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("SYNAPSE_WEB_AUDIENCE_VOTES", "2")
    votes = _Votes(_v("work-safe"), _v("work-safe"))
    monkeypatch.setattr(wa, "structured_call", votes)
    d = classify(object(), kind="web_scrape", title="t", url="u", content="c")
    assert d.detail == "2/2 votes" and len(votes.prompts) == 2


@pytest.mark.parametrize(("raw", "want"), [("0", 1), ("5", 5), ("x", wa.DEFAULT_VOTES)])
def test_votes_from_env_is_bounded(monkeypatch, raw, want):
    monkeypatch.setenv("SYNAPSE_WEB_AUDIENCE_VOTES", raw)
    assert wa.votes_from_env() == want


def test_the_prompt_carries_the_document(monkeypatch):
    votes = _Votes(_v("personal"))
    _classify(monkeypatch, votes, n=1, kind="research_brief", title="T1", url="U1", content="BODY")
    p = votes.prompts[0]
    assert "kind: research_brief" in p and "title: T1" in p and "url: U1" in p and "BODY" in p


def test_windows_cut_on_paragraphs_and_cap_every_piece():
    assert wa.windows("short", 100) == ["short"]
    paras = ["p" * 40, "q" * 40, "r" * 40]
    pieces = wa.windows("\n\n".join(paras), 90)
    assert pieces == ["p" * 40 + "\n\n" + "q" * 40, "r" * 40]
    giant = wa.windows("g" * 250, 100)
    assert [len(x) for x in giant] == [100, 100, 50]
    assert all(len(x) <= 100 for x in wa.windows("\n\n".join(["x" * 30] * 20), 100))
    assert "".join(wa.windows("\n\n".join(paras), 90)).replace("\n", "") == "".join(paras)


def test_a_brief_is_judged_window_by_window(monkeypatch):
    # The private reference sits in the last of three windows. One call over the whole
    # brief could read past it; every window is voted on separately, so it cannot.
    monkeypatch.setattr(wa, "BRIEF_WINDOW", 100)
    body = "\n\n".join(["a" * 80, "b" * 80, "c" * 80])
    clean = [_v("work-safe")] * 6
    votes = _Votes(*clean, _v("work-safe"), _v("personal", "a parent"))
    d = _classify(monkeypatch, votes, kind="research_brief", content=body)
    assert d.audience == "personal" and d.detail.startswith("part 3/3 vote 2:")
    assert len(votes.prompts) == 8
    assert "part: 3 of 3" in votes.prompts[-1] and "c" * 80 in votes.prompts[-1]
    assert "a" * 80 not in votes.prompts[-1]


def test_a_brief_is_work_safe_only_when_every_window_is_clean(monkeypatch):
    monkeypatch.setattr(wa, "BRIEF_WINDOW", 100)
    body = "\n\n".join(["a" * 80, "b" * 80])
    votes = _Votes(*[_v("work-safe")] * 6)
    d = _classify(monkeypatch, votes, kind="research_brief", content=body)
    assert d == AudienceDecision("work-safe", "classifier", "3/3 votes x 2 parts")


def test_a_scrape_and_a_short_brief_stay_one_window(monkeypatch):
    monkeypatch.setattr(wa, "BRIEF_WINDOW", 100)
    votes = _Votes(_v("work-safe"), _v("work-safe"), _v("work-safe"))
    d = _classify(monkeypatch, votes, kind="web_scrape", content="a" * 500)
    assert d.detail == "3/3 votes" and "part:" not in votes.prompts[0]
    votes = _Votes(_v("work-safe"))
    d = _classify(monkeypatch, votes, n=1, kind="research_brief", content="a" * 90)
    assert d.detail == "1/1 votes" and "part:" not in votes.prompts[0]


def test_a_brief_is_read_whole_and_a_long_scrape_by_head_and_tail():
    mid = "MIDDLE-SECTION"
    long_doc = "a" * 20000 + mid + "z" * 20000
    assert mid in excerpt(long_doc, "research_brief")
    cut = excerpt(long_doc, "web_scrape")
    assert mid not in cut and "characters omitted" in cut
    assert cut.startswith("a" * 100) and cut.rstrip().endswith("z" * 100)
    assert len(cut) < wa.SCRAPE_CHARS + 100
    assert excerpt("short", "web_scrape") == "short"
    assert excerpt(None) == ""


def test_explicit_audience_wins_and_is_validated():
    assert derive_web_audience(None, explicit="work-safe", session_id="s") == AudienceDecision(
        "work-safe", "explicit"
    )
    assert derive_web_audience(None, explicit="personal", session_id=None).audience == "personal"
    with pytest.raises(ValueError):
        derive_web_audience(None, explicit="public", session_id=None)


def test_no_session_means_the_classifier_decides():
    assert derive_web_audience(None, explicit=None, session_id=None) is None


def test_label_stats_line():
    s = LabelStats()
    assert s.line() == "web audience: labelled 0 (none pending)"
    s.add(AudienceDecision("work-safe", "classifier"))
    s.add(AudienceDecision("personal", "classifier"))
    s.add(AudienceDecision("personal", "classifier"))
    assert s.line() == ("web audience: labelled 3 (personal/classifier=2, work-safe/classifier=1)")
    assert LabelStats(skipped="x").line() == "web audience: skipped (x)"


@pytest.mark.parametrize(("raw", "on"), [(None, True), ("1", True), ("0", False), ("no", False)])
def test_kill_switch(monkeypatch, raw, on):
    if raw is None:
        monkeypatch.delenv("SYNAPSE_WEB_AUDIENCE", raising=False)
    else:
        monkeypatch.setenv("SYNAPSE_WEB_AUDIENCE", raw)
    assert wa.enabled() is on


def test_poller_stage_is_a_no_op_without_a_client_or_when_switched_off(monkeypatch):
    from ingestion.poller import Poller

    called = []
    monkeypatch.setattr(wa, "label_pending", lambda *a, **k: called.append(1))
    Poller(db=object()).label_web_audience()  # type: ignore[arg-type]
    monkeypatch.setenv("SYNAPSE_WEB_AUDIENCE", "0")
    Poller(db=object(), audience_llm=object()).label_web_audience()  # type: ignore[arg-type]
    assert called == []
