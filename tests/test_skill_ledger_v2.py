"""skills-lane v2 ledger units (dream/skills/skill_ledger.py) — pure Python, no DB.

Covers the v2 evidence semantics: the (session_id, signal, class, scan_night) dedup key,
the derive recurrence unit (distinct sessions on distinct session days, never scan nights),
one judge vote per session in the rollup, and the observe->proposed gates (_passes_gate).
All fixtures are synthetic.
"""

from __future__ import annotations

from dream.skills.skill_ledger import (
    EVIDENCE_CAP,
    PROPOSE_SCORE,
    _passes_gate,
    _recurrence,
    _rollup,
    _session_dates,
    _union_evidence,
)


def _ev(sid="sess-1", signal="gap_scan", cls="judge", night=None, **extra):
    e = {"session_id": sid, "signal": signal, "class": cls}
    if night is not None:
        e["scan_night"] = night
    e.update(extra)
    return e


# ------------------------------------------------------------- _union_evidence


def test_union_same_night_duplicates_collapse():
    a = _ev(night="2026-07-17")
    b = _ev(night="2026-07-17")
    assert len(_union_evidence([a], [b])) == 1


def test_union_resighting_on_later_night_counts():
    a = _ev(night="2026-07-17")
    b = _ev(night="2026-07-18")
    assert len(_union_evidence([a], [b])) == 2


def test_union_legacy_entries_share_one_bucket():
    # entries without scan_night keep the old (session, signal, class) collapse
    merged = _union_evidence([_ev()], [_ev(), _ev()])
    assert len(merged) == 1


def test_union_legacy_and_dated_coexist():
    merged = _union_evidence([_ev()], [_ev(night="2026-07-18")])
    assert len(merged) == 2


def test_union_keeps_first_and_caps():
    old = [_ev(sid=f"sess-{i}") for i in range(EVIDENCE_CAP + 10)]
    assert len(_union_evidence(old, [])) == EVIDENCE_CAP


# ---------------------------------------------------------------- _passes_gate


def test_retune_proposes_on_one_quoted_instance():
    ev = [_ev(signal="post_fire_deviation", night="2026-07-18", quote="synthetic quote")]
    assert _passes_gate("retune", ev, 0.5, None)


def test_retune_without_quote_stays_observe():
    ev = [_ev(signal="post_fire_deviation", night="2026-07-18")]
    assert not _passes_gate("retune", ev, 0.5, None)


def test_retune_blank_quote_does_not_count():
    ev = [_ev(quote="   ")]
    assert not _passes_gate("retune", ev, 0.5, None)


def test_consolidate_stays_observe_even_with_quote():
    # overlap nominations carry no review-worthy moment; consolidate keeps the score path only
    assert not _passes_gate("consolidate", [_ev(quote="synthetic overlap moment")], 0.5, None)
    assert not _passes_gate(
        "consolidate", [_ev(night="2026-07-17"), _ev(night="2026-07-18")], 0.5, 5
    )


def test_derive_quote_alone_is_not_a_gate():
    # the one-strong-instance gate is retune/consolidate only
    assert not _passes_gate("derive", [_ev(quote="synthetic")], 0.5, None)


def test_derive_one_session_on_two_scan_nights_stays_observe():
    # one long session re-scanned the next night is not a repeat (the audit's point (e))
    ev = [
        _ev(night="2026-07-17", date="2026-07-16"),
        _ev(night="2026-07-18", date="2026-07-16"),
    ]
    assert not _passes_gate("derive", ev, 0.5, None)


def test_derive_proposes_on_two_sessions_on_two_days():
    ev = [_ev(sid="a", date="2026-10-23"), _ev(sid="b", date="2026-10-28")]
    assert _passes_gate("derive", ev, 1.0, None)


def test_derive_two_sessions_on_one_day_stays_observe():
    ev = [_ev(sid="a", date="2026-07-29"), _ev(sid="b", date="2026-07-29")]
    assert not _passes_gate("derive", ev, 1.0, None)


def test_derive_undated_sessions_do_not_make_up_a_day():
    ev = [_ev(sid="a", date="2026-07-29"), _ev(sid="b"), _ev(sid="c")]
    assert not _passes_gate("derive", ev, 1.0, None)


def test_derive_one_night_stays_observe():
    assert not _passes_gate("derive", [_ev(night="2026-07-18")], 0.5, None)


def test_session_dates_take_each_sessions_earliest_day():
    ev = [
        _ev(sid="a", date="2026-07-20"),
        _ev(sid="a", date="2026-07-19"),
        _ev(sid="b"),
        _ev(sid=None, date="2026-07-01"),
    ]
    assert _session_dates(ev) == {"a": "2026-07-19", "b": None}
    # a long session with dated entries on two days is still ONE session
    assert not _recurrence([_ev(sid="a", date="2026-07-19"), _ev(sid="a", date="2026-07-20")])


# --------------------------------------------------------------------- _rollup


def test_rollup_counts_a_rescanned_session_once():
    # entries are kept per scan night; three nights of the same session must not add up
    # to the 1.5 legacy score that three different sessions earn
    ev = [_ev(night=f"2026-07-1{i}") for i in range(3)]
    assert _rollup(ev) == (1, 0, 1.0, 0.0)
    ev += [_ev(sid="sess-2", night="2026-07-13"), _ev(signal="struggle_arc", night="2026-07-13")]
    assert _rollup(ev) == (2, 0, 3.0, 0.0)


def test_rollup_keeps_grounded_and_sessionless_entries_per_entry():
    ev = [
        _ev(sid=None, signal="overlap"),
        _ev(sid=None, signal="overlap"),
        _ev(sid=None, cls="grounded", signal="reject"),
        _ev(sid=None, cls="grounded", signal="accept"),
    ]
    assert _rollup(ev) == (0, 0, 2.0, 0.0)


def test_derive_proposes_on_salience():
    assert _passes_gate("derive", [_ev(night="2026-07-18")], 0.5, 4)
    assert not _passes_gate("derive", [_ev(night="2026-07-18")], 0.5, 3)


def test_a_standing_human_reject_turns_the_derive_looseners_off():
    reject = _ev(sid=None, cls="grounded", signal="reject")
    two_days = [_ev(sid="a", date="2026-07-19"), _ev(sid="b", date="2026-07-20")]
    # salience and recurrence earned before the reject don't re-propose it
    assert not _passes_gate("derive", [*two_days, reject], -2.0, 4)
    # a person's revive (or accept) after the reject lifts it
    revive = _ev(sid=None, cls="grounded", signal="revive")
    assert _passes_gate("derive", [*two_days, reject, revive], 1.0, None)
    # the score path, which carries the -3, still can
    assert _passes_gate("derive", [*two_days, reject], PROPOSE_SCORE, None)
    # retunes keep their quote gate
    assert _passes_gate("retune", [_ev(quote="synthetic"), reject], -2.5, None)


def test_union_keeps_every_review_act():
    # reject, revive, reject again: the second reject is a new act, not a duplicate
    # sighting, and dropping it at the next merge would leave the row looking revived
    acts = [_ev(sid=None, cls="grounded", signal=s) for s in ("reject", "revive", "reject")]
    merged = _union_evidence([_ev(), *acts], [_ev(sid="sess-2")])
    assert [e["signal"] for e in merged if e["class"] == "grounded"] == [
        "reject",
        "revive",
        "reject",
    ]
    assert not _passes_gate("derive", merged, -2.0, 4)


def test_legacy_score_gate_still_applies_to_all_kinds():
    for kind in ("derive", "retune", "consolidate"):
        assert _passes_gate(kind, [_ev()], PROPOSE_SCORE, None)


# --------------------------------------------------------- derive identity resolution
class _FakeCur:
    """Minimal cursor: _resolve_id only calls execute() + fetchall() on the derive path."""

    def __init__(self, rows):
        self._rows = rows

    def execute(self, *args):
        pass

    def fetchall(self):
        return self._rows


def _row(cid, session, sigkey, cooling=False):
    ev = [{"session_id": session, "class": "judge", "signal": "gap_scan"}]
    return (cid, ev, sigkey, cooling)


def test_a_cooling_rejected_derive_reports_its_cooldown():
    from dream.skills.skill_ledger import _resolve_id

    # a rejected row still inside rejected_until: the match says so, and merge_candidate
    # skips the sighting instead of opening a fresh row beside the rejected one
    cur = _FakeCur([_row(1, "s1", "media preferences subtitles", cooling=True)])
    got = _resolve_id(cur, "derive", "media-admin", None, None, "media preferences", {"s1"})
    assert got is not None and got[0] == 1 and got[2] is True


def test_shared_session_with_disagreeing_signatures_is_not_identity():
    from dream.skills.skill_ledger import _resolve_id

    # one long session routinely contains several distinct gaps: a shared session id
    # with an unrelated signature must not merge (and clobber) the existing row
    cur = _FakeCur([_row(1, "s1", "lan discovery recon nmap sweep")])
    got = _resolve_id(
        cur, "derive", "media-admin", None, None, "media preferences subtitles playback", {"s1"}
    )
    assert got is None


def test_shared_session_with_agreeing_signature_merges():
    from dream.skills.skill_ledger import _resolve_id

    cur = _FakeCur([_row(1, "s1", "media preferences subtitles playback config")])
    got = _resolve_id(
        cur, "derive", "media-admin", None, None, "media preferences subtitles playback", {"s1"}
    )
    assert got is not None and got[0] == 1


def test_shared_session_without_signatures_still_merges():
    from dream.skills.skill_ledger import _resolve_id

    # legacy rows carry no signature_key; session overlap alone still resolves there
    cur = _FakeCur([_row(1, "s1", "")])
    got = _resolve_id(cur, "derive", "media-admin", None, None, "media preferences", {"s1"})
    assert got is not None and got[0] == 1
