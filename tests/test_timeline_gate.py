"""Unit tests for the timeline chat gate (ingestion/timeline_gate.py). Pure — the
LLM/db/embedder are stubs; the live Haiku + PG path was validated separately."""

from __future__ import annotations

import pytest

from ingestion.llm_client import MalformedResponseError
from ingestion.timeline_gate import TimelineGate, _parse_gate


def test_gate_prompt_ports_framing_and_date_anchoring():
    """v2 disciplines from the Mastra study: the framing preamble (only-record +
    authoritative user, question != happening) and in-text date anchoring."""
    from ingestion.timeline_gate import GATE_PROMPT

    # Framing preamble.
    assert "ONLY record" in GATE_PROMPT
    assert "authoritative" in GATE_PROMPT
    # In-text date anchoring for a further (non-event-timing) date.
    assert "(meaning 2026-01-31)" in GATE_PROMPT
    # Quoted-material suppression (issue #24): third-party happenings narrated in
    # pasted content are not the user's events; relayed-about-user stays eligible.
    # Probed 2026-07-05 (8 synthetic cases, DeepSeek): leaks 2->0, no legit-event loss.
    assert "QUOTED MATERIAL" in GATE_PROMPT
    assert "THEIR event" in GATE_PROMPT
    assert "still about the user" in GATE_PROMPT


def test_parse_gate_null_is_skip():
    assert _parse_gate('{"events": []}') == []
    assert _parse_gate('{"event": null}') == []  # legacy single-event shape


def test_parse_gate_event_and_salience():
    d = _parse_gate(
        'noise {"events": [{"event": "fixed the dating bug", "salience": 2, '
        '"event_type": "action"}]} trailing'
    )
    assert d == [
        {
            "event": "fixed the dating bug",
            "salience": 2,
            "event_type": "action",
            "domain": None,
            "date": None,
        }
    ]


def test_parse_gate_accepts_date():
    d = _parse_gate(
        '{"events": [{"event": "attended the conference", "salience": 1, '
        '"event_type": "action", "date": " 2026-06-20 "}]}'
    )
    assert d[0]["date"] == "2026-06-20"


def test_parse_gate_ignores_nonstring_or_empty_date():
    assert _parse_gate('{"event": "did a thing", "date": 20260620}')[0]["date"] is None
    assert _parse_gate('{"event": "did a thing", "date": ""}')[0]["date"] is None
    assert _parse_gate('{"event": "did a thing"}')[0]["date"] is None


def test_parse_gate_domain_field():
    ok = _parse_gate('{"event": "started a new gym routine", "domain": "personal"}')
    assert ok[0]["domain"] == "personal"
    # Invalid or missing -> None (unlabeled fails OPEN at read; a wrong default
    # would hide the event from personal-scope serving).
    assert _parse_gate('{"event": "did a thing", "domain": "work"}')[0]["domain"] is None
    assert _parse_gate('{"event": "did a thing"}')[0]["domain"] is None


def test_parse_gate_bad_event_type_nulls():
    d = _parse_gate('{"event": "did a thing", "salience": 1, "event_type": "vibe"}')
    assert d[0]["event_type"] is None


def test_parse_gate_bad_salience_clamps_to_med():
    d = _parse_gate('{"event": "ran the benchmark", "salience": 9}')
    assert d[0]["salience"] == 1


def test_parse_gate_caps_at_three_events():
    import json as _json

    raw = _json.dumps({"events": [{"event": f"did thing {i}", "salience": 1} for i in range(5)]})
    assert len(_parse_gate(raw)) == 3


def test_parse_gate_malformed_raises():
    with pytest.raises(MalformedResponseError):
        _parse_gate("no json here at all")


class _Boom:
    def __getattr__(self, _):  # any use of llm/db/embedder would explode
        raise AssertionError("should not be touched")


def _gate(enabled=True, monkeypatch=None):
    if monkeypatch:
        monkeypatch.setenv("SYNAPSE_TIMELINE_GATE", "1" if enabled else "0")
    return TimelineGate(db=_Boom(), llm_client=_Boom(), embedder=_Boom())


def test_disabled_gate_is_noop(monkeypatch):
    g = _gate(enabled=False, monkeypatch=monkeypatch)
    g.process({"id": 1, "episode_id": 5, "content": "x" * 500})  # would explode if it ran


def test_short_content_skips_llm(monkeypatch):
    g = _gate(enabled=True, monkeypatch=monkeypatch)
    g.process({"id": 1, "episode_id": 5, "content": "ok thanks"})  # under _MIN_CONTENT


def test_missing_episode_id_skips(monkeypatch):
    g = _gate(enabled=True, monkeypatch=monkeypatch)
    g.process({"id": 1, "content": "x" * 500})


def test_errors_are_swallowed(monkeypatch):
    # llm blows up -> process() logs and returns, never raises (KG work must not break)
    g = _gate(enabled=True, monkeypatch=monkeypatch)
    g.process({"id": 1, "episode_id": 5, "content": "x" * 500})


# ---- identifier dedup (write-time) ----


def test_extract_idents():
    from ingestion.timeline_gate import extract_idents

    f = "shipped PR #193 (commit 068074a8) and fixed #99"
    assert extract_idents(f) == ["#193", "068074a8", "#99"]
    assert extract_idents("decided to use halfvec everywhere") == []


class _Rec:
    """Records calls; configurable ident_exists answer."""

    def __init__(self, exists):
        self._exists = exists
        self.inserted = []

    def get_episodes_valid_at(self, ids):
        return "2026-07-02T10:00:00+00:00"

    def timeline_ident_exists(self, idents, project, t_valid, window_hours):
        return self._exists

    def insert_timeline_event(self, **kw):
        self.inserted.append(kw)
        return 1


class _StubEmb:
    def embed(self, texts, task):
        return [[0.0] * 4 for _ in texts]


def _gated(monkeypatch, event, exists):
    import ingestion.timeline_gate as tg

    monkeypatch.setenv("SYNAPSE_TIMELINE_GATE", "1")
    db = _Rec(exists)
    g = tg.TimelineGate(db=db, llm_client=object(), embedder=_StubEmb())
    monkeypatch.setattr(
        tg,
        "parse_with_retry",
        lambda *a, **k: [{"event": event, "salience": 1, "event_type": None, "date": None}],
    )
    g.process({"id": 1, "episode_id": 5, "content": "x" * 500, "project": "synapse"})
    return db.inserted


def test_gate_threads_domain_to_insert(monkeypatch):
    import ingestion.timeline_gate as tg

    monkeypatch.setenv("SYNAPSE_TIMELINE_GATE", "1")
    db = _Rec(exists=False)
    g = tg.TimelineGate(db=db, llm_client=object(), embedder=_StubEmb())
    monkeypatch.setattr(
        tg,
        "parse_with_retry",
        lambda *a, **k: [
            {
                "event": "booked a dentist appointment",
                "salience": 1,
                "event_type": "action",
                "domain": "personal",
                "date": None,
            }
        ],
    )
    g.process({"id": 1, "episode_id": 5, "content": "x" * 500, "project": "neuron"})
    assert db.inserted[0]["domain"] == "personal"


def test_commit_announcement_with_known_ident_skips(monkeypatch):
    assert _gated(monkeypatch, "committed PR #103 to main", exists=True) == []


def test_deploy_with_known_ident_still_writes(monkeypatch):
    assert len(_gated(monkeypatch, "deployed PR #103 to production", exists=True)) == 1


def test_commit_announcement_with_fresh_ident_writes(monkeypatch):
    assert len(_gated(monkeypatch, "committed PR #103 to main", exists=False)) == 1


# ---- date resolution (write-side) ----

TURN = "2026-07-02"


def test_resolve_event_date_none_falls_back_to_turn():
    from ingestion.timeline_gate import _resolve_event_date

    assert _resolve_event_date(None, TURN) == TURN


def test_resolve_event_date_malformed_falls_back():
    from ingestion.timeline_gate import _resolve_event_date

    assert _resolve_event_date("last Tuesday", TURN) == TURN
    assert _resolve_event_date("2026/06/20", TURN) == TURN
    assert _resolve_event_date("2026-13-40", TURN) == TURN  # ISO shape but not a real date


def test_resolve_event_date_future_falls_back():
    from ingestion.timeline_gate import _resolve_event_date

    assert _resolve_event_date("2026-08-01", TURN) == TURN


def test_resolve_event_date_too_old_falls_back():
    from ingestion.timeline_gate import _resolve_event_date

    assert _resolve_event_date("2024-01-01", TURN) == TURN  # >2 years back


def test_resolve_event_date_valid_past_kept():
    from ingestion.timeline_gate import _resolve_event_date

    assert _resolve_event_date("2026-06-20", TURN) == "2026-06-20"
    assert _resolve_event_date(" 2026-06-20 ", TURN) == "2026-06-20"


def _gated_full(monkeypatch, gate_ret, exists=False):
    """Drive _process with a fully-specified gate return (incl. optional date); no LLM."""
    import ingestion.timeline_gate as tg

    monkeypatch.setenv("SYNAPSE_TIMELINE_GATE", "1")
    db = _Rec(exists)  # get_episodes_valid_at -> "2026-07-02T10:00:00+00:00"
    g = tg.TimelineGate(db=db, llm_client=object(), embedder=_StubEmb())
    monkeypatch.setattr(
        tg,
        "parse_with_retry",
        lambda *a, **k: gate_ret if isinstance(gate_ret, list) else [gate_ret],
    )
    g.process({"id": 1, "episode_id": 5, "content": "x" * 500, "project": "synapse"})
    return db.inserted


def test_multi_event_turn_writes_suffixed_refs(monkeypatch):
    ins = _gated_full(
        monkeypatch,
        [
            {
                "event": "ran a 5K in 27 minutes 12 seconds",
                "salience": 1,
                "event_type": "action",
                "date": None,
            },
            {
                "event": "bought a road bike for $1,450",
                "salience": 1,
                "event_type": "action",
                "date": None,
            },
        ],
    )
    assert [i["source_ref"] for i in ins] == ["ep:5", "ep:5#2"]


def test_resolved_past_date_stamps_t_valid_at_noon(monkeypatch):
    ins = _gated_full(
        monkeypatch,
        {
            "event": "attended a conference (reported as 'last week')",
            "salience": 1,
            "event_type": "action",
            "date": "2026-06-20",
        },
    )
    assert len(ins) == 1
    assert ins[0]["t_valid"] == "2026-06-20T12:00:00+00:00"


def test_same_day_keeps_precise_turn_timestamp(monkeypatch):
    ins = _gated_full(
        monkeypatch,
        {"event": "fixed a bug", "salience": 1, "event_type": "action", "date": None},
    )
    assert len(ins) == 1
    assert ins[0]["t_valid"] == "2026-07-02T10:00:00+00:00"


def test_future_gate_date_falls_back_to_turn_timestamp(monkeypatch):
    ins = _gated_full(
        monkeypatch,
        {"event": "shipped it", "salience": 1, "event_type": "action", "date": "2027-01-01"},
    )
    assert ins[0]["t_valid"] == "2026-07-02T10:00:00+00:00"


# ---- dedup confirm-merge (write-time, schema 037) ----


def test_parse_verdict():
    from ingestion.timeline_gate import _parse_verdict

    assert _parse_verdict(" same\n") == "SAME"
    assert _parse_verdict("DISTINCT.") == "DISTINCT"
    with pytest.raises(MalformedResponseError):
        _parse_verdict("maybe?")


class _DedupDb(_Rec):
    """_Rec plus the dedup surface: a configurable candidate pool + bump recorder."""

    def __init__(self, cands):
        super().__init__(exists=False)
        self._cands = cands
        self.bumped = []

    def timeline_near_candidates(self, *a, **k):
        return self._cands

    def bump_timeline_reported(self, event_id, t_valid):
        self.bumped.append((event_id, t_valid))

    def get_episode(self, episode_id):
        return {"content": f"full source turn text for ep {episode_id}"}


_CAND = [
    {
        "id": 42,
        "fact": "tried a new bread recipe (reported as 'on Tuesday')",
        "t_valid": "2026-06-30T10:00:00+00:00",
        "source_ref": "ep:9",
        "dist": 0.04,
    }
]


def _dedup_gate(monkeypatch, cands, verdicts, dedup_env="1"):
    """Drive one gated event through the dedup path with scripted confirm verdicts."""
    import ingestion.timeline_gate as tg

    monkeypatch.setenv("SYNAPSE_TIMELINE_GATE", "1")
    monkeypatch.setenv("SYNAPSE_TIMELINE_DEDUP", dedup_env)
    db = _DedupDb(cands)
    g = tg.TimelineGate(db=db, llm_client=object(), embedder=_StubEmb())
    monkeypatch.setattr(
        tg,
        "parse_with_retry",
        lambda *a, **k: [
            {
                "event": "baked sourdough bread (reported as 'on Tuesday')",
                "salience": 1,
                "event_type": None,
                "date": None,
            }
        ],
    )
    seq = iter(verdicts)
    g._confirm_same = lambda a, b: next(seq)  # scripted; StopIteration = unexpected call
    g.process({"id": 1, "episode_id": 5, "content": "x" * 500, "project": "p"})
    return db


def test_both_orders_same_merges_instead_of_inserting(monkeypatch):
    db = _dedup_gate(monkeypatch, _CAND, [True, True])
    assert db.inserted == []
    assert db.bumped == [(42, "2026-07-02T10:00:00+00:00")]


def test_order_flip_keeps_the_event(monkeypatch):
    db = _dedup_gate(monkeypatch, _CAND, [True, False])
    assert len(db.inserted) == 1
    assert db.bumped == []


def test_distinct_keeps_the_event(monkeypatch):
    # first order says DISTINCT -> short-circuits, second call never made
    db = _dedup_gate(monkeypatch, _CAND, [False])
    assert len(db.inserted) == 1


def test_no_candidates_skips_confirm_entirely(monkeypatch):
    # empty verdict script: any confirm call would StopIteration -> swallowed -> no insert.
    db = _dedup_gate(monkeypatch, [], [])
    assert len(db.inserted) == 1


def test_dedup_kill_switch(monkeypatch):
    # candidates present and confirms would say SAME, but the env gate is off
    db = _dedup_gate(monkeypatch, _CAND, [True, True], dedup_env="0")
    assert len(db.inserted) == 1
    assert db.bumped == []


def test_dedup_db_error_fails_soft_to_insert(monkeypatch):
    # _Rec has no timeline_near_candidates at all -> AttributeError inside the
    # dedup path -> logged, event inserts normally (dup cheaper than lost event)
    ins = _gated(monkeypatch, "decided the failure mode is acceptable", exists=False)
    assert len(ins) == 1


def test_missing_candidate_turn_means_no_merge(monkeypatch):
    class _NoTurn(_DedupDb):
        def get_episode(self, episode_id):
            return None

    import ingestion.timeline_gate as tg

    monkeypatch.setenv("SYNAPSE_TIMELINE_GATE", "1")
    monkeypatch.setenv("SYNAPSE_TIMELINE_DEDUP", "1")
    db = _NoTurn(_CAND)
    g = tg.TimelineGate(db=db, llm_client=object(), embedder=_StubEmb())
    monkeypatch.setattr(
        tg,
        "parse_with_retry",
        lambda *a, **k: [{"event": "did a thing", "salience": 1, "event_type": None, "date": None}],
    )
    g._confirm_same = lambda a, b: (_ for _ in ()).throw(AssertionError("no turn -> no confirm"))
    g.process({"id": 1, "episode_id": 5, "content": "x" * 500, "project": "p"})
    assert len(db.inserted) == 1
    assert db.bumped == []
