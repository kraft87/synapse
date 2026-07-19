"""skills-lane v2 ledger units (dream/skills/skill_ledger.py) — pure Python, no DB.

Covers the v2 evidence semantics: the (session_id, signal, class, scan_night) dedup key,
distinct-scan-night counting with the single legacy bucket, and the observe->proposed
gates (_passes_gate). All fixtures are synthetic.
"""

from __future__ import annotations

from dream.skills.skill_ledger import (
    EVIDENCE_CAP,
    PROPOSE_SCORE,
    _passes_gate,
    _scan_nights,
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


# ---------------------------------------------------------------- _scan_nights


def test_scan_nights_empty():
    assert _scan_nights([]) == 0


def test_scan_nights_legacy_counts_once():
    assert _scan_nights([_ev(sid="a"), _ev(sid="b"), _ev(sid="c")]) == 1


def test_scan_nights_distinct_dates():
    ev = [_ev(night="2026-07-17"), _ev(sid="x", night="2026-07-18"), _ev(night="2026-07-18")]
    assert _scan_nights(ev) == 2


def test_scan_nights_legacy_plus_dated():
    assert _scan_nights([_ev(), _ev(night="2026-07-18")]) == 2


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


def test_derive_proposes_on_two_scan_nights():
    ev = [_ev(night="2026-07-17"), _ev(night="2026-07-18")]
    assert _passes_gate("derive", ev, 1.0, None)


def test_derive_one_night_stays_observe():
    assert not _passes_gate("derive", [_ev(night="2026-07-18")], 0.5, None)


def test_derive_proposes_on_salience():
    assert _passes_gate("derive", [_ev(night="2026-07-18")], 0.5, 4)
    assert not _passes_gate("derive", [_ev(night="2026-07-18")], 0.5, 3)


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


def _row(cid, session, sigkey):
    return (cid, [{"session_id": session, "class": "judge", "signal": "gap_scan"}], sigkey)


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
