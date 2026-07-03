"""Tests for the write-time entity deduplicator.

The deduper runs four strategies in order: exact normalized-name short-
circuit, entropy gate, MinHash/LSH Jaccard, LLM confirm. These tests
cover each strategy independently and the summary-merge helper.

The graph client is mocked — ``NodeDeduper`` only touches it through
``entity_uuid_by_normalized_name`` (exact-name pass) and ``load_entities``
(initial LSH hydration), the KG-client methods used in
production (#67 PR 2).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from ingestion.dedup import (
    NodeDeduper,
    _jaccard,
    _name_entropy,
    _normalize_name,
    _shingles,
    has_high_entropy,
)

# ---------------------------------------------------------------------------
# Mock-KG helper
# ---------------------------------------------------------------------------


class _FakeQueryResult:
    def __init__(self, rows: list[list[Any]]) -> None:
        self.result_set = rows


class _FakeGraph:
    """Minimal Cypher mock.

    Routes queries by substring so the deduper's exact-name lookup and
    LSH hydration can be answered independently. Callers can preset:

    - ``entities`` — list of (uuid, name, summary) tuples that the LSH
      hydration query will return
    - ``exact_match`` — dict[normalized_name -> uuid] used by the
      indexed and fallback exact-name lookup queries
    """

    def __init__(
        self,
        entities: list[tuple[str, ...]] | None = None,
        exact_match: dict[str, str] | None = None,
        type_map: dict[str, str] | None = None,
    ) -> None:
        self._entities = entities or []
        self._exact_match = exact_match or {}
        self._type_map = type_map or {}

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> _FakeQueryResult:
        params = params or {}
        # Exact-name lookups (both indexed and fallback variants).
        if "normalized_name" in cypher or "toLower(trim" in cypher:
            uid = self._exact_match.get(params.get("name", ""))
            return _FakeQueryResult([[uid]] if uid else [])
        # LSH hydration query.
        if "RETURN e.uuid, e.name, e.summary" in cypher:
            return _FakeQueryResult([list(e) for e in self._entities])
        return _FakeQueryResult([])


class _FakeKGClient:
    def __init__(self, graph: _FakeGraph) -> None:
        self._graph_obj = graph

    def _graph(self, group_id: str) -> _FakeGraph:
        return self._graph_obj

    # The deduper now goes through the client-level read methods (which
    # serve from Postgres in production) rather than raw Cypher.
    def entity_uuid_by_normalized_name(self, normalized: str, group_id: str) -> str | None:
        return self._graph_obj._exact_match.get(normalized)

    def load_entities(self, group_id: str) -> list[tuple[str, ...]]:
        return list(self._graph_obj._entities)

    def load_type_map(self) -> dict[str, str]:
        return dict(self._graph_obj._type_map)


def _make_llm(answer: str) -> MagicMock:
    """Mock LLM client returning ``answer`` (one of yes/no/uncertain)."""
    client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(text=answer)]
    client.messages.create.return_value = msg
    return client


# ---------------------------------------------------------------------------
# Helpers — pure-function tests
# ---------------------------------------------------------------------------


class TestNormalizeName:
    def test_lowercases_and_strips_whitespace(self):
        assert _normalize_name("  FalkorDB  ") == "falkordb"

    def test_collapses_internal_whitespace(self):
        assert _normalize_name("Synapse   Poller") == "synapse poller"

    def test_strips_leading_trailing_punctuation(self):
        assert _normalize_name(".synapse.") == "synapse"
        assert _normalize_name("(falkordb)") == "falkordb"

    def test_empty_name(self):
        assert _normalize_name("") == ""
        assert _normalize_name("   ") == ""


class TestEntropy:
    def test_low_entropy_repeated_chars(self):
        # "aaaa" has near-zero entropy (single symbol).
        assert _name_entropy("aaaa") == pytest.approx(0.0)

    def test_high_entropy_varied_name(self):
        # A name with many distinct characters should clear 2.0.
        assert _name_entropy("falkordb") > 2.0

    def test_entropy_gate_rejects_short_single_token(self):
        assert has_high_entropy("api") is False

    def test_entropy_gate_accepts_long_specific_name(self):
        assert has_high_entropy("synapse-architecture.md") is True

    def test_entropy_gate_accepts_two_token_short_name(self):
        # 2 tokens — passes the length-OR-token check, entropy is high.
        assert has_high_entropy("ab cd ef") is True


class TestShinglesJaccard:
    def test_shingles_3gram(self):
        s = _shingles("falkor", n=3)
        # "falkor" → "fal", "alk", "lko", "kor"
        assert s == {"fal", "alk", "lko", "kor"}

    def test_jaccard_identical(self):
        a = _shingles("falkordb")
        assert _jaccard(a, a) == 1.0

    def test_jaccard_disjoint(self):
        assert _jaccard({"abc"}, {"xyz"}) == 0.0


# ---------------------------------------------------------------------------
# NodeDeduper — strategy 1 (exact normalized-name)
# ---------------------------------------------------------------------------


class TestExactNameMatch:
    def test_exact_name_match_with_trailing_space(self):
        graph = _FakeGraph(exact_match={"falkordb": "uuid-existing"})
        client = _FakeKGClient(graph)
        deduper = NodeDeduper(client, group_id="technical")

        # Two entities with names "FalkorDB" and "falkordb " (trailing space)
        # dedup to the same UUID via Strategy 1.
        first = deduper.find_or_none("FalkorDB", summary="graph db")
        second = deduper.find_or_none("falkordb ", summary="other summary")

        assert first == "uuid-existing"
        assert second == "uuid-existing"

    def test_no_match_returns_none(self):
        graph = _FakeGraph(exact_match={})
        client = _FakeKGClient(graph)
        deduper = NodeDeduper(client, group_id="technical")
        assert deduper.find_or_none("BrandNewName") is None

    def test_register_then_in_memory_hit(self):
        graph = _FakeGraph(exact_match={})
        client = _FakeKGClient(graph)
        deduper = NodeDeduper(client, group_id="technical")
        deduper.register("Synapse", "uuid-syn", summary="memory layer")
        # Next call to find_or_none on the same normalized name resolves
        # via the in-memory map, not Cypher.
        assert deduper.find_or_none("synapse") == "uuid-syn"


# ---------------------------------------------------------------------------
# Strategy 2 — entropy gate
# ---------------------------------------------------------------------------


class TestEntropyGateSkipsFuzzy:
    def test_low_entropy_name_below_threshold_skips_fuzzy(self):
        # Two distinct entities named "api" should NOT merge via fuzzy.
        # The exact-name pass would merge if both lived in the graph
        # already (legitimate), but here the second is brand new — the
        # entropy gate must prevent any fuzzy-collapse to the first.
        graph = _FakeGraph(
            entities=[("uuid-api-1", "api", "first api meaning")],
            exact_match={},  # exact lookup misses
        )
        client = _FakeKGClient(graph)
        deduper = NodeDeduper(client, group_id="technical")

        # "api" has length < 6 AND token_count < 2 → fails has_high_entropy.
        # No exact match → must return None.
        assert deduper.find_or_none("api", summary="second api meaning") is None


# ---------------------------------------------------------------------------
# Strategy 3 — MinHash/LSH Jaccard
# ---------------------------------------------------------------------------


class TestMinHashJaccardMatch:
    def test_minhash_match_with_no_llm(self):
        # "synapse-poller" and "synapse poller" share enough 3-gram
        # shingles to clear the LSH+Jaccard threshold of 0.7.
        graph = _FakeGraph(
            entities=[("uuid-poller", "synapse-poller", "watches sessions")],
            exact_match={},
        )
        client = _FakeKGClient(graph)
        # No LLM → deduper falls back to "Jaccard alone is enough."
        deduper = NodeDeduper(client, group_id="technical", llm_client=None)

        result = deduper.find_or_none("synapse poller", summary="...")
        assert result == "uuid-poller"

    def test_minhash_below_threshold_no_match(self):
        graph = _FakeGraph(
            entities=[("uuid-syn", "synapse", "memory layer")],
            exact_match={},
        )
        client = _FakeKGClient(graph)
        deduper = NodeDeduper(client, group_id="technical", llm_client=None)
        # Very different name shouldn't hit the LSH threshold.
        assert deduper.find_or_none("totally-unrelated-grandstream-router") is None


class TestTypeGate:
    """The taxonomy type-compatibility gate on FUZZY candidates (schema 020)."""

    def test_type_compatible_rules(self):
        from ingestion.dedup import _type_compatible

        assert _type_compatible("Service", "Service")  # equal
        assert _type_compatible("Service", None)  # untyped is permissive
        assert _type_compatible(None, "Database")
        assert _type_compatible("Tool", "Concept")  # Concept permissive
        assert _type_compatible("other", "Database")  # other permissive
        assert _type_compatible("Tool", "Service")  # allowlisted pair
        assert _type_compatible("Project", "Product")  # allowlisted pair
        assert not _type_compatible("Service", "Database")  # the false-merge case the gate prevents
        assert not _type_compatible("Person", "Tool")

    def test_gate_drops_cross_category_fuzzy_candidate(self, monkeypatch):
        monkeypatch.setattr("ingestion.dedup._TYPE_GATE_ON", True)
        graph = _FakeGraph(
            entities=[("uuid-poller", "synapse-poller", "the database", "Database")],
            exact_match={},
            type_map={"Database": "Database", "Service": "Service"},
        )
        deduper = NodeDeduper(_FakeKGClient(graph), group_id="technical", llm_client=None)
        # new entity is a Service; the only fuzzy candidate is a Database (not allowlisted) -> dropped
        kind, _payload = deduper.classify("synapse poller", "...", entity_type="Service")
        assert kind == "none"

    def test_gate_keeps_same_category_candidate(self, monkeypatch):
        monkeypatch.setattr("ingestion.dedup._TYPE_GATE_ON", True)
        graph = _FakeGraph(
            entities=[("uuid-poller", "synapse-poller", "a service", "Service")],
            exact_match={},
            type_map={"Service": "Service"},
        )
        deduper = NodeDeduper(_FakeKGClient(graph), group_id="technical", llm_client=None)
        kind, payload = deduper.classify("synapse poller", "...", entity_type="Service")
        assert kind == "candidates"
        assert payload[0][0] == "uuid-poller"

    def test_gate_untyped_new_entity_is_not_filtered(self, monkeypatch):
        # no entity_type -> gate no-ops -> current behavior preserved (candidate survives)
        monkeypatch.setattr("ingestion.dedup._TYPE_GATE_ON", True)
        graph = _FakeGraph(
            entities=[("uuid-poller", "synapse-poller", "the database", "Database")],
            exact_match={},
            type_map={"Database": "Database"},
        )
        deduper = NodeDeduper(_FakeKGClient(graph), group_id="technical", llm_client=None)
        kind, _payload = deduper.classify("synapse poller", "...")
        assert kind == "candidates"


# ---------------------------------------------------------------------------
# Strategy 4 — LLM confirm
# ---------------------------------------------------------------------------


class TestLLMConfirm:
    def test_llm_confirm_yes_returns_canonical_uuid(self):
        graph = _FakeGraph(
            entities=[("uuid-poller", "synapse-poller", "watches sessions")],
            exact_match={},
        )
        client = _FakeKGClient(graph)
        llm = _make_llm("yes")
        deduper = NodeDeduper(client, group_id="technical", llm_client=llm)

        result = deduper.find_or_none("synapse poller", summary="...")
        assert result == "uuid-poller"
        assert llm.messages.create.called

    def test_llm_confirm_no_returns_none(self):
        graph = _FakeGraph(
            entities=[("uuid-poller", "synapse-poller", "watches sessions")],
            exact_match={},
        )
        client = _FakeKGClient(graph)
        llm = _make_llm("no")
        deduper = NodeDeduper(client, group_id="technical", llm_client=llm)

        result = deduper.find_or_none("synapse poller", summary="...")
        assert result is None

    def test_llm_uncertain_returns_none(self):
        graph = _FakeGraph(
            entities=[("uuid-poller", "synapse-poller", "watches sessions")],
            exact_match={},
        )
        client = _FakeKGClient(graph)
        llm = _make_llm("uncertain")
        deduper = NodeDeduper(client, group_id="technical", llm_client=llm)
        # "uncertain" answers don't accept the merge — return None so the
        # caller writes a new node and the nightly pipeline can re-evaluate.
        assert deduper.find_or_none("synapse poller", summary="...") is None


# ---------------------------------------------------------------------------
# Summary merge
# ---------------------------------------------------------------------------


class TestSummaryMerge:
    def test_longer_summary_wins(self):
        short = "short"
        long_ = "this is a much longer summary with more detail"
        assert NodeDeduper.merge_summary(short, long_) == long_
        assert NodeDeduper.merge_summary(long_, short) == long_

    def test_empty_existing_keeps_incoming(self):
        assert NodeDeduper.merge_summary("", "new") == "new"

    def test_empty_incoming_keeps_existing(self):
        assert NodeDeduper.merge_summary("existing", "") == "existing"

    def test_both_empty_returns_empty(self):
        assert NodeDeduper.merge_summary("", "") == ""
        assert NodeDeduper.merge_summary(None, None) == ""


# ---------------------------------------------------------------------------
# NodeDeduper.classify — no-LLM candidate classification (batch resolver path)
# ---------------------------------------------------------------------------


class TestClassify:
    """``classify`` runs strategies 1-3 (no LLM) and tags the outcome.

    The batch resolver calls this instead of ``find_or_none`` so it can
    gather every entity needing a decision and confirm them in one LLM call.
    """

    def test_exact_hit_returns_exact(self):
        graph = _FakeGraph(exact_match={"falkordb": "uuid-existing"})
        deduper = NodeDeduper(_FakeKGClient(graph), group_id="technical")
        kind, payload = deduper.classify("FalkorDB", summary="graph db")
        assert kind == "exact"
        assert payload == "uuid-existing"

    def test_entropy_gated_returns_none(self):
        graph = _FakeGraph()  # no exact match
        deduper = NodeDeduper(_FakeKGClient(graph), group_id="technical")
        # "api" is short + single-token -> entropy gate -> none.
        kind, payload = deduper.classify("api")
        assert kind == "none"
        assert payload is None

    def test_lsh_candidate_returns_candidates(self):
        # An existing fuzzy-near node hydrated into the LSH index surfaces as
        # a candidate (uuid, name, summary, jaccard) needing confirmation.
        graph = _FakeGraph(
            entities=[("uuid-sp", "synapse poller", "the poller")],
        )
        deduper = NodeDeduper(_FakeKGClient(graph), group_id="technical")
        kind, payload = deduper.classify("synapse-poller", summary="...")
        assert kind == "candidates"
        assert isinstance(payload, list) and payload
        cand_uuid, cand_name, _cand_summary, jacc = payload[0]
        assert cand_uuid == "uuid-sp"
        assert cand_name == "synapse poller"
        assert 0.0 <= jacc <= 1.0
