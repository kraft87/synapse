"""Stage 7 supersession lineage + evidence metadata (schema 028 links, schema 058 fields).

Pure-logic: the KG client is a MagicMock, the edge-date extractor and the writer-side
contradiction detector are stubbed. What is pinned:

* Every old edge a new fact retires carries ``invalidated_by`` -> the edge that replaced it.
  Stage 6 verdicts used to be flattened and written with NO superseder, so a correction
  found at extraction time was invisible to recall's successor lookup (which requires the
  link). The superseder is the new edge when written, the confirmed existing duplicate when
  the write is skipped (drop-in replacement), and NULL only when neither exists.
* Retirement is dated by the correcting fact's evidenced valid-from (in-text date, else the
  conversation timestamp), never by the write clock.
* ``ongoing`` / ``last_supported_at`` / ``last_supported_by`` ride the CREATE row, and only a
  USER-attributed fact counts as support — on create and on reinforcement alike.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ingestion.extractor import ExtractionPipeline
from ingestion.models import ExtractedFact

SEG = "2026-09-01T12:00:00+00:00"  # the segment (conversation) timestamp


def _pipe(n_facts: int, dates: list[tuple[str | None, str | None]] | None = None):
    pipe = ExtractionPipeline.__new__(ExtractionPipeline)
    pipe._kg = MagicMock()
    pipe._edge_date_extractor = MagicMock()
    pipe._edge_date_extractor.extract_batch.return_value = dates or [(None, None)] * n_facts
    pipe._contradiction_detector = MagicMock()
    pipe._contradiction_detector.detect_contradictions_batch.return_value = [
        [] for _ in range(n_facts)
    ]
    return pipe


def _fact(i: int, **over) -> ExtractedFact:
    kw = {"source": f"S{i}", "target": f"T{i}", "relationship": "DOES", "fact": f"fact-{i}"}
    kw.update(over)
    return ExtractedFact(**kw)


def _uuid_map(facts: list[ExtractedFact]) -> dict[str, str]:
    m: dict[str, str] = {}
    for f in facts:
        m[f.source] = f"u-{f.source}"
        m[f.target] = f"u-{f.target}"
    return m


def _invalidations(pipe) -> list[tuple[list[tuple[str, str | None]], str | None]]:
    """Every invalidate_edges_batch call as (items, invalidated_by)."""
    out = []
    for call in pipe._kg.invalidate_edges_batch.call_args_list:
        items = call.args[0]
        sup = call.kwargs.get("invalidated_by")
        out.append((items, sup))
    return out


def _created(pipe) -> list[dict]:
    calls = pipe._kg.create_edges_batch.call_args_list
    return calls[0].args[0] if calls else []


class TestSupersederLink:
    def test_stage6_contradiction_links_to_the_new_edge(self):
        facts = [_fact(0, attribution="user")]
        pipe = _pipe(1)
        pipe._stage7_write_edges(
            facts, _uuid_map(facts), [41, 42], "g", set(), {0: ["old-1"]}, default_valid_at=SEG
        )
        rows = _created(pipe)
        assert len(rows) == 1
        inv = _invalidations(pipe)
        assert len(inv) == 1
        items, sup = inv[0]
        assert sup == rows[0]["edge_uuid"]  # the link recall's successor lookup needs
        assert items == [("old-1", SEG)]  # retired as of the conversation, not the write clock

    def test_skipped_duplicate_links_to_the_confirmed_existing_edge(self):
        # Drop-in replacement: "now uses X" is a pure duplicate of the live X edge (skip the
        # write, reinforce it) while retiring the Y edge -> Y's superseder is the X edge.
        facts = [_fact(0, attribution="user")]
        pipe = _pipe(1)
        pipe._stage7_write_edges(
            facts,
            _uuid_map(facts),
            [7],
            "g",
            {0},
            {0: ["old-y"]},
            default_valid_at=SEG,
            reinforce={0: ["dup-x"]},
        )
        assert _created(pipe) == []
        assert _invalidations(pipe) == [([("old-y", SEG)], "dup-x")]

    def test_unresolved_endpoints_still_retire_without_a_link(self):
        facts = [_fact(0)]
        pipe = _pipe(1)
        pipe._stage7_write_edges(facts, {}, [1], "g", set(), {0: ["old-1"]}, default_valid_at=SEG)
        assert _created(pipe) == []
        assert _invalidations(pipe) == [([("old-1", SEG)], None)]

    def test_safety_net_contradictions_share_the_link_and_date(self):
        facts = [_fact(0)]
        pipe = _pipe(1, dates=[("2026-03-15T00:00:00Z", None)])
        pipe._contradiction_detector.detect_contradictions_batch.return_value = [["old-pair"]]
        pipe._stage7_write_edges(
            facts, _uuid_map(facts), [1], "g", set(), {0: ["old-sem"]}, default_valid_at=SEG
        )
        rows = _created(pipe)
        inv = _invalidations(pipe)
        assert len(inv) == 1
        items, sup = inv[0]
        assert sup == rows[0]["edge_uuid"]
        # the fact's own in-text date wins over the segment timestamp, for both verdict sources
        assert set(items) == {
            ("old-sem", "2026-03-15T00:00:00Z"),
            ("old-pair", "2026-03-15T00:00:00Z"),
        }
        assert rows[0]["t_valid"] == "2026-03-15T00:00:00Z"

    def test_same_old_edge_from_two_facts_is_retired_once_per_superseder(self):
        facts = [_fact(0), _fact(1)]
        pipe = _pipe(2)
        pipe._stage7_write_edges(
            facts,
            _uuid_map(facts),
            [1],
            "g",
            set(),
            {0: ["old-1"], 1: ["old-1"]},
            default_valid_at=SEG,
        )
        inv = _invalidations(pipe)
        assert len(inv) == 2  # one call per superseder, each carrying old-1 once
        assert all(items == [("old-1", SEG)] for items, _ in inv)
        assert len({sup for _, sup in inv}) == 2


class TestEvidenceMetadata:
    @pytest.mark.parametrize(
        ("attribution", "ongoing", "expect_support"),
        [
            ("user", True, True),
            ("assistant", True, False),
            ("third_party", False, False),
            ("unknown", False, False),
        ],
    )
    def test_create_row_carries_flags_and_user_only_support(
        self, attribution, ongoing, expect_support
    ):
        facts = [_fact(0, attribution=attribution, ongoing=ongoing)]
        pipe = _pipe(1, dates=[("2019-01-01T00:00:00Z", None)])
        pipe._stage7_write_edges(
            facts, _uuid_map(facts), [10, 12, 11], "g", set(), {}, default_valid_at=SEG
        )
        (row,) = _created(pipe)
        assert row["ongoing"] is ongoing
        if expect_support:
            # support time = when it was SAID (segment), not the extracted valid-from (2019)
            assert row["last_supported_at"] == SEG
            assert row["last_supported_by"] == 12  # latest source episode
        else:
            assert row["last_supported_at"] is None
            assert row["last_supported_by"] is None
        assert row["t_valid"] == "2019-01-01T00:00:00Z"

    def test_reinforcement_refreshes_support_only_for_user_restatements(self):
        facts = [_fact(0, attribution="user"), _fact(1, attribution="assistant")]
        pipe = _pipe(2)
        pipe._stage7_write_edges(
            facts,
            _uuid_map(facts),
            [5, 6],
            "g",
            {0, 1},
            {},
            default_valid_at=SEG,
            reinforce={0: ["e-user"], 1: ["e-assistant"]},
        )
        (call,) = pipe._kg.reinforce_edges.call_args_list
        items = {it[0]: it for it in call.args[0]}
        assert items["e-user"] == ("e-user", [5, 6], SEG, 6)
        assert items["e-assistant"] == ("e-assistant", [5, 6], None, None)

    def test_no_segment_timestamp_falls_back_to_now_for_support(self):
        facts = [_fact(0, attribution="user")]
        pipe = _pipe(1)
        pipe._stage7_write_edges(facts, _uuid_map(facts), [1], "g", set(), {})
        (row,) = _created(pipe)
        assert row["last_supported_at"] == row["t_created"]
