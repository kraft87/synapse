"""Stage-6 gray-zone gate (issue #14): decision triage, enforcement shaping, and
the dedup_gate_shadow telemetry write.

Shadow mode must be behaviorally invisible (candidates_map reaches the LLM
untouched); enforcement resolves merge-zone facts without the LLM while still
firing the reinforce (assert-count bump) signal, drops auto-new candidates, and
never drops a candidate the gate has no embedding signal for.
"""

from __future__ import annotations

from ingestion.db import Database
from ingestion.extractor import (
    _apply_gate_enforce,
    _dedup_gate_mode,
    _gate_decisions,
    _gate_shadow_rows,
)
from ingestion.models import ExtractedFact

HIGH, LOW = 0.95, 0.70


def _c(uuid: str, sim: float | None) -> dict:
    return {"uuid": uuid, "fact": f"existing {uuid}", "_sim": sim}


def test_gate_decisions_thresholds():
    decisions = _gate_decisions(
        [_c("a", 0.97), _c("b", 0.95)],  # pair pool: >= high -> merge (boundary included)
        [_c("c", 0.70), _c("d", 0.80), _c("e", None)],  # <= low -> new; between -> gray
        HIGH,
        LOW,
    )
    by_uuid = {c["uuid"]: (pool, sim, d) for c, pool, sim, d in decisions}
    assert by_uuid["a"][2] == "merge" and by_uuid["b"][2] == "merge"
    assert by_uuid["c"][2] == "new"
    assert by_uuid["d"][2] == "gray"
    # BM25-only candidate (no embedding signal) must never be auto-decided
    assert by_uuid["e"][2] == "gray" and by_uuid["e"][1] is None
    assert by_uuid["a"][0] == "pair" and by_uuid["c"][0] == "semantic"


def test_enforce_merge_preskips_and_reinforces():
    gate_info = {
        0: _gate_decisions([_c("dup", 0.98)], [_c("other", 0.85)], HIGH, LOW),
    }
    gray_map, pre_skip, pre_reinforce = _apply_gate_enforce(gate_info)
    assert pre_skip == {0}
    assert pre_reinforce == {0: ["dup"]}  # assert-count bump still fires on auto-merge
    assert gray_map == {}  # fact resolved -> no LLM confirm for it


def test_enforce_shrinks_pools_to_gray_zone():
    gate_info = {
        0: _gate_decisions([_c("low", 0.60)], [_c("gray1", 0.85), _c("bm25", None)], HIGH, LOW),
        1: _gate_decisions([], [_c("low2", 0.10)], HIGH, LOW),
    }
    gray_map, pre_skip, pre_reinforce = _apply_gate_enforce(gate_info)
    assert pre_skip == set() and pre_reinforce == {}
    # fact 0: auto-new candidate dropped; gray + no-signal candidates survive
    pair, sem = gray_map[0]
    assert pair == []
    assert [c["uuid"] for c in sem] == ["gray1", "bm25"]
    # fact 1: every candidate auto-new -> no LLM confirm at all
    assert 1 not in gray_map


def test_gate_mode_default_and_validation(monkeypatch):
    monkeypatch.delenv("SYNAPSE_DEDUP_GATE", raising=False)
    assert _dedup_gate_mode() == "shadow"
    monkeypatch.setenv("SYNAPSE_DEDUP_GATE", "ENFORCE")
    assert _dedup_gate_mode() == "enforce"
    monkeypatch.setenv("SYNAPSE_DEDUP_GATE", "bogus")
    assert _dedup_gate_mode() == "shadow"


def _fact(i: int) -> ExtractedFact:
    return ExtractedFact(source=f"S{i}", target=f"T{i}", relationship="USES", fact=f"new fact {i}")


def test_shadow_rows_pair_verdicts_with_llm():
    facts = [_fact(0)]
    gate_info = {0: _gate_decisions([_c("dup", 0.98)], [_c("contra", 0.80)], HIGH, LOW)}
    llm_map = {0: ([_c("dup", 0.98)], [_c("contra", 0.80)])}  # shadow: full pools sent
    rows = _gate_shadow_rows(
        facts, gate_info, llm_map, "technical", {0: ["contra"]}, {0: ["dup"]}, llm_ok=True
    )
    by_uuid = {r[2]: r for r in rows}
    # (group, fact, cand_uuid, cand_fact, pool, sim, decision, llm_dup, llm_contra, llm_ran)
    assert by_uuid["dup"][6] == "merge" and by_uuid["dup"][7] is True and by_uuid["dup"][9] is True
    assert by_uuid["contra"][6] == "gray" and by_uuid["contra"][8] is True


def test_shadow_rows_null_verdicts_when_llm_failed_or_not_sent():
    facts = [_fact(0)]
    gate_info = {0: _gate_decisions([_c("a", 0.98)], [], HIGH, LOW)}
    # batch failed -> verdict columns NULL even though the candidate was sent
    rows = _gate_shadow_rows(
        facts, gate_info, {0: ([_c("a", 0.98)], [])}, "technical", {}, {}, llm_ok=False
    )
    assert rows[0][7] is None and rows[0][8] is None and rows[0][9] is False
    # enforcement dropped the candidate (not in llm_map) -> same NULL treatment,
    # even though enforce pre-seeded it into the reinforce map
    rows = _gate_shadow_rows(facts, gate_info, {}, "technical", {}, {0: ["a"]}, llm_ok=True)
    assert rows[0][7] is None and rows[0][9] is False


def test_log_dedup_gate_shadow_round_trip(conn, db_url):
    conn.execute("TRUNCATE dedup_gate_shadow RESTART IDENTITY")
    db = Database(db_url)
    db.log_dedup_gate_shadow(
        [
            (
                "technical",
                "new fact",
                "uuid-1",
                "old fact",
                "pair",
                0.97,
                "merge",
                True,
                False,
                True,
            ),
            (
                "technical",
                "new fact",
                "uuid-2",
                "old fact 2",
                "semantic",
                None,
                "gray",
                None,
                None,
                False,
            ),
        ]
    )
    rows = conn.execute(
        "SELECT candidate_uuid, sim, decision, llm_duplicate, llm_ran "
        "FROM dedup_gate_shadow ORDER BY id"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0] == ("uuid-1", 0.97, "merge", True, True)
    assert rows[1][1] is None and rows[1][3] is None and rows[1][4] is False
    db.log_dedup_gate_shadow([])  # empty batch is a no-op, not an error
