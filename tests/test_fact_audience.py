"""The fact-audience labeller (schema 066): the parts that need no database.

The classifier is scripted (``structured_call`` is replaced), so these cover what the
stage does with a reply: how a fact becomes a line, how labels become an audience, what
happens to a reply that is short, wrong or never validates, and what the poller does with
the stage. The writes, the tiers and the probe are in test_fact_audience_db.py.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from ingestion import fact_audience as fa
from ingestion.llm_client import LLMHTTPError, MalformedResponseError, UsageLimitError
from ingestion.poller import _FACT_AUDIENCE_PAUSE_S, Poller

PRIVATE = "sensitivity:private"
SAFE = "sensitivity:work-safe"


def _rows(*texts: str, first_id: int = 100) -> list[dict[str, Any]]:
    return [{"id": first_id + i, "fact": t} for i, t in enumerate(texts)]


def _lines(prompt: str) -> list[str]:
    """The fact lines of a rendered prompt."""
    tail = prompt.split("Facts, one per line as <n>|<text>:\n", 1)[1]
    return tail.split("\n\nReply with ONLY", 1)[0].split("\n")


class _Judge:
    """Stands in for ``structured_call``: private when a fact says SECRET, else work-safe.
    ``raise_for`` maps a prompt to an exception (or None) so a reply can fail by content;
    ``reply_for`` maps a prompt's fact lines to a scripted reply (or None for the honest one)."""

    def __init__(self, raise_for=None, reply=None, reply_for=None) -> None:
        self.prompts: list[str] = []
        self.raise_for = raise_for
        self.reply = reply
        self.reply_for = reply_for

    def __call__(self, llm, **kw):
        assert kw["output_model"] is fa.FactAudienceReply
        self.prompts.append(kw["base_prompt"])
        if self.raise_for is not None:
            err = self.raise_for(kw["base_prompt"])
            if err is not None:
                raise err
        if self.reply is not None:
            return self.reply
        if self.reply_for is not None:
            made = self.reply_for(_lines(kw["base_prompt"]))
            if made is not None:
                return made
        facts = [
            fa.FactLabel(n=i, labels=[PRIVATE if "SECRET" in line else SAFE])
            for i, line in enumerate(_lines(kw["base_prompt"]), start=1)
        ]
        return fa.FactAudienceReply(facts=facts)

    @property
    def calls(self) -> int:
        return len(self.prompts)


@pytest.fixture()
def judge(monkeypatch):
    j = _Judge()
    monkeypatch.setattr(fa, "structured_call", j)
    return j


# ---------------------------------------------------------------------------
# Rendering and the label vocabulary
# ---------------------------------------------------------------------------


def test_render_numbers_from_one_folds_whitespace_and_caps_a_fact():
    rows = _rows("plain fact", "two\nlines  and\ttabs", "x" * (fa.FACT_CHARS + 50))
    lines = fa.render(rows).split("\n")
    assert lines[0] == "1|plain fact"
    assert lines[1] == "2|two lines and tabs"
    assert lines[2] == "3|" + "x" * fa.FACT_CHARS
    assert len(lines) == 3


@pytest.mark.parametrize(
    ("labels", "audience"),
    [
        ([SAFE], "work-safe"),
        ([PRIVATE], "personal"),
        ([SAFE, PRIVATE], "personal"),
        ([PRIVATE, SAFE], "personal"),
        ([SAFE, SAFE], "personal"),
        ([], "personal"),
    ],
)
def test_only_exactly_the_work_safe_label_is_work_safe(labels, audience):
    assert fa.to_audience(labels) == audience


def test_the_stored_vocabulary_is_the_surfaces_one():
    from ingestion.surfaces import AUDIENCES

    assert {fa.to_audience([SAFE]), fa.to_audience([PRIVATE])} == set(AUDIENCES)


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------


def test_the_prompt_has_one_slot_every_rule_and_a_json_object_reply():
    assert fa.PROMPT.count("{facts}") == 1
    for rule in ("P1.", "P2.", "P3.", "P4.", "P5.", "P6.", "W."):
        assert f"\n{rule} " in fa.PROMPT, rule
    rendered = fa.PROMPT.replace("{facts}", "1|a fact")
    assert "\n1|a fact\n" in rendered
    assert 'Reply with ONLY a JSON object {"facts": [...]}' in rendered
    assert '{"facts": [{"n": 1, "labels": ["sensitivity:private"]}' in rendered


def test_a_fact_that_looks_like_the_slot_is_not_substituted_again():
    rows = _rows("the literal {facts} placeholder")
    prompt = fa.PROMPT.replace("{facts}", fa.render(rows))
    assert "1|the literal {facts} placeholder" in prompt


def test_the_prompt_carries_no_private_address_range_or_owner_detail():
    """A public repo: the prompt names the private ranges as a class, and every example in
    it is invented."""
    assert not re.search(r"\b(?:10|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d", fa.PROMPT)
    assert "RFC 1918" in fa.PROMPT
    assert not re.search(r"\b[0-9a-f]{12,}\b|@(?!example\.com)\w+\.\w+", fa.PROMPT)


# ---------------------------------------------------------------------------
# classify_batch
# ---------------------------------------------------------------------------


def test_every_fact_is_labelled_from_its_own_line(judge):
    rows = _rows("a plain fact", "a SECRET fact", "another plain fact")
    out = fa.classify_batch(object(), rows)
    assert {i: v.audience for i, v in out.items()} == {
        100: "work-safe",
        101: "personal",
        102: "work-safe",
    }
    assert judge.calls == 1 and all(v.source == "model" for v in out.values())


def _entries(*pairs: tuple[int, str]) -> fa.FactAudienceReply:
    return fa.FactAudienceReply(facts=[fa.FactLabel(n=n, labels=[lab]) for n, lab in pairs])


def _wrong_when_long(wrong):
    """A reply script that is honest for one or two lines and ``wrong(lines)`` above that."""
    return lambda lines: wrong(lines) if len(lines) > 2 else None


@pytest.mark.parametrize(
    "wrong",
    [
        # the model drops line 1 and keeps counting: every later label shifts onto another fact
        lambda lines: _entries(*((i, SAFE) for i in range(1, len(lines)))),
        # one number repeated, the second entry private: not "the first one stands"
        lambda lines: _entries(
            (1, SAFE), (1, PRIVATE), *((i, SAFE) for i in range(3, len(lines) + 1))
        ),
        # an extra entry past the end
        lambda lines: _entries(*((i, SAFE) for i in range(1, len(lines) + 2))),
        # numbers that are not 1..n, though there are n of them
        lambda lines: _entries(*((i, SAFE) for i in range(2, len(lines) + 2))),
        # nothing at all
        lambda lines: _entries(),
    ],
    ids=["short", "repeat", "extra", "shifted", "empty"],
)
def test_a_reply_that_does_not_name_each_line_once_is_split_not_trusted(monkeypatch, wrong):
    """The model echoes only the line number, so a reply that skips, repeats or invents one
    cannot be told from one whose later labels belong to other facts. It is never read: the
    batch is split and each half asked again, so a private fact is not stored work-safe."""
    j = _Judge(reply_for=_wrong_when_long(wrong))
    monkeypatch.setattr(fa, "structured_call", j)
    rows = _rows("SECRET one", "plain two", "plain three", "plain four")
    out = fa.classify_batch(object(), rows)
    assert {i: v.audience for i, v in out.items()} == {
        100: "personal",
        101: "work-safe",
        102: "work-safe",
        103: "work-safe",
    }
    assert all(v.source == "model" for v in out.values())
    assert j.calls == 3, "the batch, then its two halves"


def test_a_repeated_number_with_conflicting_labels_never_stores_the_first_one(monkeypatch):
    """Work-safe first, private second, for the same line: not resolved by order. Asked
    alone, the fact gets its own answer."""
    reply = _entries((1, SAFE), (1, PRIVATE))
    j = _Judge(reply_for=lambda lines: reply if len(lines) == 2 else None)
    monkeypatch.setattr(fa, "structured_call", j)
    out = fa.classify_batch(object(), _rows("a SECRET fact", "a plain fact"))
    assert out[100] == fa.FactVerdict("personal")
    assert out[101] == fa.FactVerdict("work-safe")


def test_one_fact_whose_reply_never_lines_up_is_personal_and_costs_one_call(monkeypatch):
    j = _Judge(reply=_entries((2, SAFE)))
    monkeypatch.setattr(fa, "structured_call", j)
    out = fa.classify_batch(object(), _rows("only"))
    assert out == {100: fa.FactVerdict("personal", "error")}
    assert j.calls == 1


def test_a_reply_that_never_validates_is_split_until_only_the_bad_fact_fails(monkeypatch):
    def poison(prompt):
        return MalformedResponseError("not json", "x") if "POISON" in prompt else None

    j = _Judge(raise_for=poison)
    monkeypatch.setattr(fa, "structured_call", j)
    rows = _rows("fact one", "fact two", "POISON three", "fact four")
    out = fa.classify_batch(object(), rows)
    assert out[100] == fa.FactVerdict("work-safe")
    assert out[101] == fa.FactVerdict("work-safe")
    assert out[102] == fa.FactVerdict("personal", "error")
    assert out[103] == fa.FactVerdict("work-safe")
    # 1 call for the batch, 2 halves, and the half with the bad fact splits again: bounded
    assert j.calls == 1 + 2 + 2


def test_one_failing_single_fact_is_personal_not_an_exception(monkeypatch):
    monkeypatch.setattr(
        fa, "structured_call", _Judge(raise_for=lambda p: ValueError("schema says no"))
    )
    out = fa.classify_batch(object(), _rows("only"))
    assert out == {100: fa.FactVerdict("personal", "error")}


def test_a_usage_limit_propagates_at_any_depth(monkeypatch):
    """Even after a split: a single fact's call that hits the limit is not a verdict."""

    def fails_alone(prompt):
        lines = _lines(prompt)
        if len(lines) > 1:
            return MalformedResponseError("not json", "x")
        return UsageLimitError("limit") if "bad" in lines[0] else None

    monkeypatch.setattr(fa, "structured_call", _Judge(raise_for=fails_alone))
    with pytest.raises(UsageLimitError):
        fa.classify_batch(object(), _rows("good one", "bad one"))


#: What a dead backend raises. The first is what the Claude CLI path raises once its own
#: retries are spent (``agent_call``), which no transport-error tuple names.
_DEAD_BACKEND = [
    RuntimeError("agent_call(fact_audience): failed after retries: spawn failed"),
    LLMHTTPError("down", 503),
    httpx.ConnectError("refused"),
    ValueError("anything else"),
]


@pytest.mark.parametrize("err", _DEAD_BACKEND, ids=lambda e: type(e).__name__)
def test_a_backend_that_fails_every_call_ends_the_cycle_after_a_bounded_number_of_calls(
    monkeypatch, err
):
    """Not 99 failed calls and fifty ``personal`` verdicts: a run of failures stops the
    batch, and no verdict is returned."""
    j = _Judge(raise_for=lambda prompt: err)
    monkeypatch.setattr(fa, "structured_call", j)
    with pytest.raises(fa.ClassifierDown):
        fa.classify_batch(object(), _rows(*(f"fact {i}" for i in range(50))))
    assert j.calls == fa._fail_limit(50) == 8


def test_the_failure_limit_clears_the_split_depth_for_every_batch_size():
    """One bad fact costs about log2(N) failing calls in a row: the limit must sit above
    that for any batch the knob allows, or a single poison fact would read as a dead
    backend."""
    for n in range(1, fa.MAX_BATCH + 1):
        depth = 1
        size = n
        while size > 1:
            size = (size + 1) // 2  # the larger half
            depth += 1
        assert fa._fail_limit(n) > depth, n


@pytest.mark.parametrize("batch", [2, 3, 7, 50, 64, 200])
@pytest.mark.parametrize("where", ["first", "middle", "last"])
@pytest.mark.parametrize("err", [LLMHTTPError("moderation says no", 400), RuntimeError("boom")])
def test_one_fact_that_breaks_a_call_on_its_own_is_isolated_and_the_rest_are_labelled(
    monkeypatch, batch, where, err
):
    """A fact the provider refuses (a non-transient 4xx, an empty completion) must not stop
    the others being labelled, wherever it sits and however large the batch."""
    at = {"first": 0, "middle": batch // 2, "last": batch - 1}[where]
    texts = [f"plain fact {i}" for i in range(batch)]
    texts[at] = "POISON fact"
    j = _Judge(raise_for=lambda prompt: err if "POISON" in prompt else None)
    monkeypatch.setattr(fa, "structured_call", j)
    out = fa.classify_batch(object(), _rows(*texts), health=fa.CycleHealth(fa._fail_limit(batch)))
    assert out[100 + at] == fa.FactVerdict("personal", "error")
    assert all(v == fa.FactVerdict("work-safe") for i, v in out.items() if i != 100 + at)
    assert len(out) == batch


def test_no_rows_is_no_call(judge):
    assert fa.classify_batch(object(), []) == {}
    assert judge.calls == 0


def test_the_call_uses_the_stage_model_unless_one_is_given(monkeypatch):
    seen: list[str] = []

    def _record(llm, **kw):
        seen.append(kw["model"])
        return fa.FactAudienceReply(facts=[fa.FactLabel(n=1, labels=[SAFE])])

    monkeypatch.setattr(fa, "structured_call", _record)
    monkeypatch.setenv("SYNAPSE_FACT_AUDIENCE_MODEL", "stage-model")
    fa.classify_batch(object(), _rows("x"))
    fa.classify_batch(object(), _rows("x"), model="explicit")
    assert seen == ["stage-model", "explicit"]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["0", "false", "no", " 0 "])
def test_the_kill_switch(monkeypatch, value):
    monkeypatch.setenv("SYNAPSE_FACT_AUDIENCE", value)
    assert fa.enabled() is False


def test_the_stage_is_on_by_default(monkeypatch):
    monkeypatch.delenv("SYNAPSE_FACT_AUDIENCE", raising=False)
    assert fa.enabled() is True


@pytest.mark.parametrize(
    ("value", "batch"),
    [("30", 30), ("junk", fa.DEFAULT_BATCH), ("0", 1), ("-5", 1), ("9999", fa.MAX_BATCH)],
)
def test_the_batch_size_is_clamped_and_a_bad_value_is_the_default(monkeypatch, value, batch):
    monkeypatch.setenv("SYNAPSE_FACT_AUDIENCE_BATCH", value)
    assert fa.batch_from_env() == batch


@pytest.mark.parametrize(("value", "calls"), [("5", 5), ("junk", fa.DEFAULT_CALLS), ("0", 1)])
def test_the_call_count_has_a_floor_of_one(monkeypatch, value, calls):
    monkeypatch.setenv("SYNAPSE_FACT_AUDIENCE_CALLS", value)
    assert fa.calls_from_env() == calls


def test_the_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("SYNAPSE_FACT_AUDIENCE_BATCH", raising=False)
    monkeypatch.delenv("SYNAPSE_FACT_AUDIENCE_CALLS", raising=False)
    assert (fa.batch_from_env(), fa.calls_from_env()) == (50, 2)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def test_the_stats_line_names_what_was_labelled_skipped_or_stopped():
    stats = fa.LabelStats()
    assert stats.line() == "fact audience: labelled 0 (none pending)"
    for v in (
        fa.FactVerdict("work-safe"),
        fa.FactVerdict("personal"),
        fa.FactVerdict("personal"),
        fa.FactVerdict("personal", "error"),
    ):
        stats.add(v)
    assert stats.line() == ("fact audience: labelled 4 (personal=2, personal/error=1, work-safe=1)")
    stats.stopped = "usage limit"
    assert stats.line() == "fact audience: stopped: usage limit after 4"
    assert fa.LabelStats(skipped="schema 066 not applied").line() == (
        "fact audience: skipped (schema 066 not applied)"
    )


# ---------------------------------------------------------------------------
# The poller's stage
# ---------------------------------------------------------------------------


_LLM = object()


def _poller(llm: Any = _LLM) -> Poller:
    return Poller(db=SimpleNamespace(_conn=lambda: None), audience_llm=llm)  # type: ignore[arg-type]


@pytest.fixture()
def label_calls(monkeypatch):
    calls: list[dict[str, Any]] = []

    def _label(connect, llm, **kw):
        calls.append(kw)
        return fa.LabelStats(labelled={"work-safe": 3, "personal": 1})

    monkeypatch.setattr(fa, "label_pending", _label)
    return calls


def test_the_poller_stage_does_nothing_without_an_llm_client(label_calls):
    assert _poller(None).label_fact_audience() == 0
    assert label_calls == []


def test_the_poller_stage_obeys_the_kill_switch(label_calls, monkeypatch):
    monkeypatch.setenv("SYNAPSE_FACT_AUDIENCE", "0")
    assert _poller().label_fact_audience() == 0
    assert label_calls == []


def test_the_poller_stage_runs_with_the_env_sizes_and_reports_the_count(label_calls, monkeypatch):
    monkeypatch.setenv("SYNAPSE_FACT_AUDIENCE_BATCH", "25")
    monkeypatch.setenv("SYNAPSE_FACT_AUDIENCE_CALLS", "3")
    assert _poller().label_fact_audience() == 4
    assert label_calls == [{"batch": 25, "calls": 3}]


@pytest.mark.parametrize("reason", ["usage limit", "classifier failing"])
def test_a_stop_pauses_the_stage(monkeypatch, reason):
    stats = fa.LabelStats(stopped=reason, labelled={"work-safe": 2})
    calls: list[int] = []
    monkeypatch.setattr(fa, "label_pending", lambda *a, **k: calls.append(1) or stats)
    now = [1000.0]
    monkeypatch.setattr("ingestion.poller.time.monotonic", lambda: now[0])
    p = _poller()
    assert p.label_fact_audience() == 2
    assert len(calls) == 1
    now[0] += _FACT_AUDIENCE_PAUSE_S - 1
    assert p.label_fact_audience() == 0, "still paused"
    assert len(calls) == 1
    now[0] += 2
    assert p.label_fact_audience() == 2, "the pause lapsed"
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# The stage inside the maintenance loop
# ---------------------------------------------------------------------------

_FULL_CYCLE = [
    "rebuild_pending_chunks",
    "embed_pending",
    "label_web_audience",
    "label_fact_audience",
    "sweep_orphan_entities",
]


def _loop_poller(monkeypatch, calls: list[str], fact_error: BaseException | None = None) -> Poller:
    """A poller whose every stage records its call; the drain asks the loop to stop, so
    ``run_loop`` makes exactly one pass."""
    monkeypatch.setattr("signal.signal", lambda *a, **k: None)
    p = Poller(db=SimpleNamespace(release_stale_claims=lambda: 0), audience_llm=_LLM)  # type: ignore[arg-type]

    def stage(name: str, err: BaseException | None = None):
        def _run(*a: Any, **k: Any) -> int:
            calls.append(name)
            if err is not None:
                raise err
            return 0

        return _run

    for name in _FULL_CYCLE:
        monkeypatch.setattr(
            p, name, stage(name, fact_error if name == "label_fact_audience" else None)
        )

    def drain(batch_limit: int) -> int:
        calls.append("drain")
        p.request_stop()
        return 0

    monkeypatch.setattr(p, "drain_extraction_queue", drain)
    return p


def test_a_full_cycle_runs_the_fact_stage_once_between_the_web_stage_and_the_sweep(monkeypatch):
    monkeypatch.delenv("SYNAPSE_DRAIN_ONLY", raising=False)
    calls: list[str] = []
    _loop_poller(monkeypatch, calls).run_loop(interval_seconds=300)
    assert calls == [*_FULL_CYCLE, "drain"]


@pytest.mark.parametrize("value", ["1", "true", "yes"])
def test_a_drain_only_replica_never_runs_the_labelling_stages(monkeypatch, value):
    """Only the master labels: a scaled-out drain worker is an extraction-queue worker
    and nothing else."""
    monkeypatch.setenv("SYNAPSE_DRAIN_ONLY", value)
    calls: list[str] = []
    _loop_poller(monkeypatch, calls).run_loop(interval_seconds=300)
    assert calls == ["drain"]


def test_a_failing_fact_stage_does_not_stop_the_sweep_or_the_drain(monkeypatch):
    monkeypatch.delenv("SYNAPSE_DRAIN_ONLY", raising=False)
    calls: list[str] = []
    p = _loop_poller(monkeypatch, calls, fact_error=RuntimeError("the classifier fell over"))
    p.run_loop(interval_seconds=300)
    assert calls == [*_FULL_CYCLE, "drain"]
