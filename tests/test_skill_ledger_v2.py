"""skills-lane v2 ledger units (dream/skills/skill_ledger.py) — pure Python, no DB.

Covers the v2 evidence semantics: the (session_id, signal, class, work date) dedup key
(scan_night only for entries without a date), derive recurrence counted in distinct evidence dates (never scan nights, never undated
entries), and the observe->proposed gates (_passes_gate). All fixtures are synthetic.
"""

from __future__ import annotations

from dream.skills.skill_ledger import (
    EVIDENCE_CAP,
    PROPOSE_SCORE,
    _passes_gate,
    _recurrence_days,
    _rollup,
    _union_evidence,
)


def _ev(sid="sess-1", signal="gap_scan", cls="judge", night=None, date=None, **extra):
    e = {"session_id": sid, "signal": signal, "class": cls}
    if night is not None:
        e["scan_night"] = night
    if date is not None:
        e["date"] = date
    e.update(extra)
    return e


# ------------------------------------------------------------- _union_evidence


def test_the_cap_trims_judge_entries_and_never_the_grounded_ones():
    # a promoted row's grounded 'accept' sat at the front and fell off after 60 more
    # sightings, so the rollup read grounded_weight 0 and grounded_required_to_apply failed
    accept = _ev(sid=None, signal="accept", cls="grounded")
    sightings = [
        _ev(sid=f"s{i}", signal="under_trigger", date="2026-07-18") for i in range(EVIDENCE_CAP + 1)
    ]
    merged = _union_evidence([_ev(sid="s-first", signal="under_trigger"), accept], sightings)
    assert accept in merged
    assert len(merged) == EVIDENCE_CAP
    assert _rollup(merged)[3] > 0
    assert merged[-1] == sightings[-1]  # the oldest judge entries went, not the newest


def test_union_same_night_duplicates_collapse():
    a = _ev(night="2026-07-17")
    b = _ev(night="2026-07-17")
    assert len(_union_evidence([a], [b])) == 1


def test_union_undated_resighting_on_later_night_counts():
    # no work date to key on (gap_scan, legacy rows): the scan night still separates sightings
    a = _ev(night="2026-07-17")
    b = _ev(night="2026-07-18")
    assert len(_union_evidence([a], [b])) == 2


def test_union_rescan_of_the_same_work_adds_nothing():
    # a held cursor re-runs the detectors over the same window: the arc is the same work, so
    # the second scan night must not add a second entry (or a second helping of weight)
    a = _ev(signal="struggle_arc", night="2026-07-17", date="2026-07-16")
    b = _ev(signal="struggle_arc", night="2026-07-18", date="2026-07-16")
    merged = _union_evidence([a], [b])
    assert merged == [a]
    assert _rollup(merged) == _rollup([a])
    assert not _passes_gate("derive", merged, 1.0, None)


def test_union_same_session_on_another_day_counts():
    a = _ev(signal="struggle_arc", night="2026-07-17", date="2026-07-16")
    b = _ev(signal="struggle_arc", night="2026-07-18", date="2026-07-17")
    assert len(_union_evidence([a], [b])) == 2


def test_union_collapses_rescans_already_in_the_ledger():
    # rows written before the date key hold one entry per scan night; the next merge heals them
    old = [_ev(night=f"2026-07-{d}", date="2026-07-16") for d in (17, 18, 19)]
    assert len(_union_evidence(old, [])) == 1


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


# ------------------------------------------------------------ _recurrence_days


def test_recurrence_days_empty():
    assert _recurrence_days([]) == 0


def test_recurrence_days_undated_entries_never_count():
    assert _recurrence_days([_ev(sid="a"), _ev(sid="b"), _ev(sid="c")]) == 0


def test_recurrence_days_distinct_dates():
    ev = [_ev(date="2026-07-17"), _ev(sid="x", date="2026-07-18"), _ev(date="2026-07-18")]
    assert _recurrence_days(ev) == 2


def test_recurrence_days_ignore_scan_nights():
    # the same work (one date) seen by two scans is one day, not two
    ev = [
        _ev(night="2026-07-17", date="2026-07-16"),
        _ev(night="2026-07-18", date="2026-07-16"),
    ]
    assert _recurrence_days(ev) == 1


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


def test_derive_quote_alone_is_not_a_gate():
    # the one-strong-instance gate is retune only
    assert not _passes_gate("derive", [_ev(quote="synthetic")], 0.5, None)


def test_derive_proposes_on_two_dates():
    ev = [_ev(night="2026-07-18", date="2026-07-16"), _ev(night="2026-07-18", date="2026-07-17")]
    assert _passes_gate("derive", ev, 1.0, None)


def test_derive_one_date_stays_observe():
    assert not _passes_gate("derive", [_ev(night="2026-07-18", date="2026-07-18")], 0.5, None)


def test_derive_rescan_of_the_same_work_stays_observe():
    # one arc seen on two scan nights (a re-scan) is not recurrence
    ev = [_ev(night="2026-07-17", date="2026-07-16"), _ev(night="2026-07-18", date="2026-07-16")]
    assert not _passes_gate("derive", ev, 1.0, None)


def test_derive_undated_entry_is_not_a_second_day():
    # a gap_scan entry (no date) plus one dated detector entry is one day, not two
    ev = [_ev(), _ev(signal="struggle_arc", night="2026-07-18", date="2026-07-18")]
    assert not _passes_gate("derive", ev, 1.0, None)


def test_derive_proposes_on_salience():
    assert _passes_gate("derive", [_ev(date="2026-07-18")], 0.5, 4)
    assert not _passes_gate("derive", [_ev(date="2026-07-18")], 0.5, 3)


def test_legacy_score_gate_still_applies_to_all_kinds():
    for kind in ("derive", "retune"):
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


def _row(cid, session, sigkey):
    ev = [{"session_id": session, "class": "judge", "signal": "gap_scan"}]
    return (cid, ev, sigkey, [], f"row-{cid}")


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


def test_a_signatureless_derive_matches_only_the_row_of_its_own_name():
    from dream.skills.skill_ledger import _resolve_id

    # an explicit "make this a skill" request names no topic: sharing a session with another
    # gap's candidate is not identity, only its own earlier row (same name) is
    cur = _FakeCur([_row(1, "s1", "media preferences subtitles playback config")])
    assert _resolve_id(cur, "derive", "explicit:s1", None, None, "", {"s1"}) is None
    got = _resolve_id(cur, "derive", "row-1", None, None, "", {"s1"})
    assert got is not None and got[0] == 1


def test_a_row_without_a_signature_is_matched_only_by_its_name():
    from dream.skills.skill_ledger import _resolve_id

    # a row with no signature_key (an explicit "make this a skill" request, a legacy row) has
    # no topic to compare: a shared session alone must not let a later sighting merge into it
    # (and overwrite its summary), only a sighting of the same name is that row
    cur = _FakeCur([_row(1, "s1", "")])
    assert (
        _resolve_id(cur, "derive", "media-admin", None, None, "media preferences", {"s1"}) is None
    )
    got = _resolve_id(cur, "derive", "row-1", None, None, "media preferences", {"s1"})
    assert got is not None and got[0] == 1
