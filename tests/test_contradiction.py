"""Tests for the Phase 3 ContradictionDetector.

Pure unit tests -- the detector accepts injectable falkordb / embedder /
llm dependencies so we can exercise the full pipeline without a live
FalkorDB.

Five cases mirror the spec:
  1. No candidates returned by the pair filter -- short-circuit empty list.
  2. Vector similarity below threshold -- never calls the LLM.
  3. LLM confirms contradiction -- detector returns the UUID; create_edge
     invalidates the old edge before the new one is written.
  4. LLM returns empty contradicted_facts -- no UUIDs returned.
  5. Already-invalidated edge is NOT a candidate (find_edges_by_pair only
     returns live edges, so the detector never sees them).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

from ingestion.contradiction import ContradictionDetector
from ingestion.models import ExtractedFact


def _fact() -> ExtractedFact:
    return ExtractedFact(
        source="Synapse",
        target="FalkorDB",
        relationship="USES",
        fact="Synapse uses FalkorDB on port 6379",
    )


def _llm_response(text: str) -> Any:
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


def _mock_embedder(vector: list[float]) -> MagicMock:
    """An embedder that returns the same vector on every embed call."""
    m = MagicMock()
    m.embed.return_value = [vector]
    return m


def _unit_vec(prefix: float, dims: int = 8) -> list[float]:
    """Tiny helper -- builds a unit vector seeded by `prefix` so we can
    control cosine similarity in the tests deterministically."""
    base = [prefix] + [0.01] * (dims - 1)
    norm = sum(x * x for x in base) ** 0.5
    return [x / norm for x in base]


class TestNoContradiction:
    def test_no_contradiction_returns_empty(self):
        """No same-pair candidates -> empty result, LLM never called."""
        falkordb = MagicMock()
        falkordb.find_edges_by_pair.return_value = []
        llm = MagicMock()
        embedder = _mock_embedder(_unit_vec(1.0))

        detector = ContradictionDetector(falkordb, embedder, llm)
        result = detector.detect_contradictions(_fact(), "src-uuid", "tgt-uuid", "technical")

        assert result == []
        # No LLM call when there are no candidates.
        llm.messages.create.assert_not_called()


class TestSimilarityGate:
    def test_low_similarity_skips_llm(self):
        """A candidate whose embedding sits below the 0.7 threshold gets
        dropped pre-LLM so no LLM call is made."""
        # Orthogonal embeddings -- cosine ~0
        new_emb = [1.0, 0.0, 0.0, 0.0]
        old_emb = [0.0, 1.0, 0.0, 0.0]
        falkordb = MagicMock()
        falkordb.find_edges_by_pair.return_value = [
            {
                "uuid": "old-edge",
                "fact": "Synapse uses Postgres",
                "valid_at": "2025-01-01",
                "fact_embedding": old_emb,
            }
        ]
        llm = MagicMock()
        embedder = _mock_embedder(new_emb)

        detector = ContradictionDetector(falkordb, embedder, llm, similarity_threshold=0.7)
        result = detector.detect_contradictions(_fact(), "src-uuid", "tgt-uuid", "technical")

        assert result == []
        llm.messages.create.assert_not_called()

    def test_caller_supplied_embedding_skips_embed_call(self):
        """When ``fact_embedding`` is passed in we never re-embed."""
        falkordb = MagicMock()
        falkordb.find_edges_by_pair.return_value = []
        embedder = MagicMock()  # would raise if .embed was called
        embedder.embed.side_effect = AssertionError("should not be called")
        llm = MagicMock()

        detector = ContradictionDetector(falkordb, embedder, llm)
        result = detector.detect_contradictions(
            _fact(),
            "src-uuid",
            "tgt-uuid",
            "technical",
            fact_embedding=_unit_vec(1.0),
        )

        assert result == []


class TestLLMConfirm:
    def test_llm_confirms_contradiction(self):
        """High-similarity candidate + LLM says idx 0 -> UUID returned."""
        new_emb = _unit_vec(1.0)
        old_emb = _unit_vec(1.0)  # identical, cosine=1.0 -> passes 0.7 gate
        falkordb = MagicMock()
        falkordb.find_edges_by_pair.return_value = [
            {
                "uuid": "old-edge-uuid",
                "fact": "Synapse uses FalkorDB on port 7777",
                "valid_at": "2025-01-01",
                "fact_embedding": old_emb,
            }
        ]
        llm = MagicMock()
        llm.messages.create.return_value = _llm_response(json.dumps({"contradicted_facts": [0]}))
        embedder = _mock_embedder(new_emb)

        detector = ContradictionDetector(falkordb, embedder, llm)
        result = detector.detect_contradictions(_fact(), "src-uuid", "tgt-uuid", "technical")

        assert result == ["old-edge-uuid"]
        llm.messages.create.assert_called_once()
        # The candidate's UUID must NOT appear in the prompt (only idx).
        # Phase 4: the verbatim Graphiti prompt is rendered as two messages
        # (system + user); concat both so the assertions check the whole
        # prompt without caring about message-list shape.
        call_args = llm.messages.create.call_args
        prompt = "\n".join(m["content"] for m in call_args.kwargs["messages"])
        assert "old-edge-uuid" not in prompt
        assert "[0]" in prompt
        # Haiku 4.5 per the PR #11 standard.
        assert call_args.kwargs["model"] == "claude-haiku-4-5"

    def test_llm_returns_no_contradiction(self):
        """High-similarity candidate but LLM says empty list -> no UUIDs."""
        new_emb = _unit_vec(1.0)
        old_emb = _unit_vec(1.0)
        falkordb = MagicMock()
        falkordb.find_edges_by_pair.return_value = [
            {
                "uuid": "old-edge-uuid",
                "fact": "Synapse uses FalkorDB",
                "valid_at": "2025-01-01",
                "fact_embedding": old_emb,
            }
        ]
        llm = MagicMock()
        llm.messages.create.return_value = _llm_response(json.dumps({"contradicted_facts": []}))
        embedder = _mock_embedder(new_emb)

        detector = ContradictionDetector(falkordb, embedder, llm)
        result = detector.detect_contradictions(_fact(), "src-uuid", "tgt-uuid", "technical")

        assert result == []
        llm.messages.create.assert_called_once()

    def test_llm_exception_returns_empty_no_raise(self):
        """An LLM blow-up must NOT raise -- contradiction misses are
        acceptable, blocking the new edge write is not."""
        new_emb = _unit_vec(1.0)
        old_emb = _unit_vec(1.0)
        falkordb = MagicMock()
        falkordb.find_edges_by_pair.return_value = [
            {
                "uuid": "old-edge-uuid",
                "fact": "Synapse uses FalkorDB",
                "valid_at": "2025-01-01",
                "fact_embedding": old_emb,
            }
        ]
        llm = MagicMock()
        llm.messages.create.side_effect = RuntimeError("upstream 500")
        embedder = _mock_embedder(new_emb)

        detector = ContradictionDetector(falkordb, embedder, llm)
        result = detector.detect_contradictions(_fact(), "src-uuid", "tgt-uuid", "technical")

        assert result == []  # graceful degradation


class TestCandidatePoolLiveness:
    def test_already_invalidated_edge_not_recontradicted(self):
        """``find_edges_by_pair`` is the only candidate source and it
        already filters on ``invalid_at IS NULL``. The detector therefore
        cannot re-contradict an already-invalidated edge."""
        # Simulate: invalidated edges exist in the graph, but the pair
        # filter returns only the LIVE one. The detector must never see
        # the invalidated one and never invalidate it twice.
        new_emb = _unit_vec(1.0)
        live_emb = _unit_vec(1.0)
        falkordb = MagicMock()
        # Pair filter only surfaces the live edge -- the (hidden, already
        # invalidated) edge is not in the list.
        falkordb.find_edges_by_pair.return_value = [
            {
                "uuid": "live-edge",
                "fact": "Synapse uses FalkorDB on port 6379",
                "valid_at": "2025-06-01",
                "fact_embedding": live_emb,
            }
        ]
        llm = MagicMock()
        llm.messages.create.return_value = _llm_response(json.dumps({"contradicted_facts": [0]}))
        embedder = _mock_embedder(new_emb)

        detector = ContradictionDetector(falkordb, embedder, llm)
        result = detector.detect_contradictions(_fact(), "src-uuid", "tgt-uuid", "technical")

        # Only the live edge UUID is returned -- invalidated edges
        # were never in the candidate pool to begin with.
        assert result == ["live-edge"]
        # And find_edges_by_pair was the only candidate-source called.
        falkordb.find_edges_by_pair.assert_called_once_with("src-uuid", "tgt-uuid", "technical")


class TestTopKCap:
    def test_top_k_bound_passed_to_llm(self):
        """When the pair pool has more than top_k candidates above
        threshold, only the top_k highest-similarity ones reach the LLM."""
        # 7 candidates, all above threshold, all with the same embedding
        # so they're tied -- top_k=3 means at most 3 reach the LLM.
        new_emb = _unit_vec(1.0)
        candidates = []
        for i in range(7):
            candidates.append(
                {
                    "uuid": f"old-{i}",
                    "fact": f"Synapse uses FalkorDB v{i}",
                    "valid_at": "2025-01-01",
                    "fact_embedding": _unit_vec(1.0),
                }
            )
        falkordb = MagicMock()
        falkordb.find_edges_by_pair.return_value = candidates
        llm = MagicMock()
        llm.messages.create.return_value = _llm_response(json.dumps({"contradicted_facts": []}))
        embedder = _mock_embedder(new_emb)

        detector = ContradictionDetector(falkordb, embedder, llm, top_k=3)
        detector.detect_contradictions(_fact(), "s", "t", "technical")

        # Phase 4 prompt is multi-message (system + user); concat both.
        prompt = "\n".join(m["content"] for m in llm.messages.create.call_args.kwargs["messages"])
        # Exactly indices 0, 1, 2 should appear.
        assert "[0]" in prompt and "[1]" in prompt and "[2]" in prompt
        assert "[3]" not in prompt


class TestMissingEmbeddingConservative:
    def test_missing_embedding_treated_as_high_similarity(self):
        """A candidate with no fact_embedding stored (legacy data) must
        survive the gate so the LLM gets a chance to evaluate it -- the
        alternative (silently dropping) would create blind spots on the
        pre-vecf32 backfill."""
        new_emb = _unit_vec(1.0)
        falkordb = MagicMock()
        falkordb.find_edges_by_pair.return_value = [
            {
                "uuid": "legacy-edge",
                "fact": "Synapse uses FalkorDB on port 9999",
                "valid_at": "2024-12-01",
                "fact_embedding": None,
            }
        ]
        llm = MagicMock()
        llm.messages.create.return_value = _llm_response(json.dumps({"contradicted_facts": [0]}))
        embedder = _mock_embedder(new_emb)

        detector = ContradictionDetector(falkordb, embedder, llm)
        result = detector.detect_contradictions(_fact(), "s", "t", "technical")

        assert result == ["legacy-edge"]


class TestTolerantJSONParse:
    """The claude-CLI subprocess wrapper occasionally returns JSON with
    trailing prose or wrapped in markdown code fences. raw_decode() parses
    the first valid JSON object and ignores everything after, so these
    don't fall to the except branch and silently return [] (which would
    let real contradictions slip into the graph)."""

    def _detector_with_candidate(self, llm_text: str):
        falkordb = MagicMock()
        falkordb.find_edges_by_pair.return_value = [
            {
                "uuid": "old-edge",
                "fact": "Synapse uses FalkorDB on port 9999",
                "valid_at": "2024-12-01",
                "fact_embedding": _unit_vec(1.0),
            }
        ]
        llm = MagicMock()
        llm.messages.create.return_value = _llm_response(llm_text)
        embedder = _mock_embedder(_unit_vec(1.0))
        return ContradictionDetector(falkordb, embedder, llm)

    def test_trailing_prose_after_json(self):
        text = '{"contradicted_facts": [0]}\n\nThis is the explanation the model added.'
        result = self._detector_with_candidate(text).detect_contradictions(
            _fact(), "s", "t", "technical"
        )
        assert result == ["old-edge"]

    def test_markdown_code_fence(self):
        text = '```json\n{"contradicted_facts": [0]}\n```'
        result = self._detector_with_candidate(text).detect_contradictions(
            _fact(), "s", "t", "technical"
        )
        assert result == ["old-edge"]

    def test_leading_prose_before_json(self):
        text = 'Here is my analysis:\n{"contradicted_facts": [0]}'
        result = self._detector_with_candidate(text).detect_contradictions(
            _fact(), "s", "t", "technical"
        )
        assert result == ["old-edge"]

    def test_no_json_object_returns_empty(self):
        text = "I could not determine if these contradict."
        result = self._detector_with_candidate(text).detect_contradictions(
            _fact(), "s", "t", "technical"
        )
        assert result == []


# ---------------------------------------------------------------------------
# create_edge integration: contradiction sets t_invalid on the OLD edge
# and still writes the new edge. Pure-unit -- uses a KGClient with a mocked
# writer that records the mutation calls instead of hitting a real DB.
# ---------------------------------------------------------------------------


class TestDetectContradictionsBatch:
    """Batched ContradictionDetector collapses N per-fact ~30s claude-CLI
    subprocesses into ONE Haiku call. Pre-LLM filters (pair match +
    similarity gate) drop facts that need no LLM check; the batched prompt
    only contains survivors with per-fact idx scoping.
    """

    def _detector_with_pair_lookup(self, lookups: dict, llm_text: str):
        falkordb = MagicMock()
        falkordb.find_edges_by_pair.side_effect = lambda src, tgt, g: lookups.get((src, tgt), [])
        llm = MagicMock()
        msg = MagicMock()
        msg.content = [MagicMock(text=llm_text)]
        llm.messages.create.return_value = msg
        embedder = _mock_embedder(_unit_vec(1.0))
        return ContradictionDetector(falkordb, embedder, llm), llm

    def test_one_call_for_many_facts(self):
        import json as _json

        from ingestion.models import ExtractedFact

        facts = [
            ExtractedFact(source="A", target="B", relationship="USES", fact="A uses B v2"),
            ExtractedFact(source="C", target="D", relationship="USES", fact="C uses D v2"),
        ]
        lookups = {
            ("a", "b"): [
                {"uuid": "old-ab", "fact": "A uses B v1", "fact_embedding": _unit_vec(1.0)}
            ],
            ("c", "d"): [
                {"uuid": "old-cd", "fact": "C uses D v1", "fact_embedding": _unit_vec(1.0)}
            ],
        }
        llm_payload = _json.dumps(
            {
                "results": [
                    {"id": 0, "contradicted_facts": [0]},
                    {"id": 1, "contradicted_facts": [0]},
                ]
            }
        )
        det, llm = self._detector_with_pair_lookup(lookups, llm_payload)
        result = det.detect_contradictions_batch(
            facts,
            ["a", "c"],
            ["b", "d"],
            "technical",
            fact_embeddings=[_unit_vec(1.0), _unit_vec(1.0)],
        )
        # ONE LLM call for both facts.
        assert llm.messages.create.call_count == 1
        assert result == [["old-ab"], ["old-cd"]]

    def test_empty_input(self):
        det, llm = self._detector_with_pair_lookup({}, "{}")
        assert det.detect_contradictions_batch([], [], [], "technical") == []
        llm.messages.create.assert_not_called()

    def test_pre_llm_gate_drops_pure_novel_facts(self):
        """Facts whose (src, tgt) pair has no live edges never enter the
        batched prompt. With ALL facts pure-novel, LLM is never called."""
        from ingestion.models import ExtractedFact

        facts = [
            ExtractedFact(source="A", target="B", relationship="USES", fact="A uses B"),
            ExtractedFact(source="C", target="D", relationship="USES", fact="C uses D"),
        ]
        det, llm = self._detector_with_pair_lookup({}, "{}")
        result = det.detect_contradictions_batch(
            facts,
            ["a", "c"],
            ["b", "d"],
            "technical",
            fact_embeddings=[_unit_vec(1.0), _unit_vec(1.0)],
        )
        assert result == [[], []]
        llm.messages.create.assert_not_called()

    def test_similarity_gate_drops_unrelated_pair_candidate(self):
        """A live edge sharing the pair but below the cosine threshold
        gets dropped before the LLM. No LLM call when nothing survives."""
        from ingestion.models import ExtractedFact

        new_emb = [1.0, 0.0, 0.0, 0.0]
        old_emb = [0.0, 1.0, 0.0, 0.0]
        lookups = {("a", "b"): [{"uuid": "u", "fact": "old", "fact_embedding": old_emb}]}
        facts = [ExtractedFact(source="A", target="B", relationship="USES", fact="new fact")]
        det, llm = self._detector_with_pair_lookup(lookups, "{}")
        result = det.detect_contradictions_batch(
            facts, ["a"], ["b"], "technical", fact_embeddings=[new_emb]
        )
        assert result == [[]]
        llm.messages.create.assert_not_called()

    def test_llm_exception_returns_all_empty(self):
        from ingestion.models import ExtractedFact

        lookups = {("a", "b"): [{"uuid": "u", "fact": "x", "fact_embedding": _unit_vec(1.0)}]}
        det, llm = self._detector_with_pair_lookup(lookups, "{}")
        llm.messages.create.side_effect = RuntimeError("upstream 500")
        facts = [ExtractedFact(source="A", target="B", relationship="USES", fact="y")]
        result = det.detect_contradictions_batch(
            facts, ["a"], ["b"], "technical", fact_embeddings=[_unit_vec(1.0)]
        )
        # Graceful degradation — never raise, never block writes.
        assert result == [[]]

    def test_missing_id_in_response_treated_as_empty(self):
        """If the LLM omits some fact ids in its results array, those
        facts get empty contradictions (fail-open)."""
        import json as _json

        from ingestion.models import ExtractedFact

        facts = [
            ExtractedFact(source="A", target="B", relationship="USES", fact="x"),
            ExtractedFact(source="C", target="D", relationship="USES", fact="y"),
        ]
        lookups = {
            ("a", "b"): [{"uuid": "u-ab", "fact": "f", "fact_embedding": _unit_vec(1.0)}],
            ("c", "d"): [{"uuid": "u-cd", "fact": "g", "fact_embedding": _unit_vec(1.0)}],
        }
        # LLM only returns id=0.
        llm_payload = _json.dumps({"results": [{"id": 0, "contradicted_facts": [0]}]})
        det, _ = self._detector_with_pair_lookup(lookups, llm_payload)
        result = det.detect_contradictions_batch(
            facts,
            ["a", "c"],
            ["b", "d"],
            "technical",
            fact_embeddings=[_unit_vec(1.0), _unit_vec(1.0)],
        )
        assert result == [["u-ab"], []]


class TestCreateEdgeBiTemporal:
    def _make_client_with_fake_writer(self):
        """Build a real KGClient whose writer is a MagicMock, so we can
        assert on the exact mutations fired by create_edge / invalidate_edge
        without needing a live Postgres."""
        from ingestion.kg_client import KGClient

        client = KGClient.__new__(KGClient)
        client._writer = MagicMock()
        client._reader = MagicMock()
        return client, client._writer

    def test_new_edge_sets_t_created_and_t_valid(self):
        client, writer = self._make_client_with_fake_writer()
        client.create_edge(
            source_uuid="src",
            target_uuid="tgt",
            relationship="USES",
            fact="A uses B",
            episode_ids=[1, 2],
            group_id="technical",
            fact_embedding=[0.1, 0.2, 0.3],
        )
        writer.create_edges.assert_called_once()
        rows, group = writer.create_edges.call_args.args
        assert group == "technical"
        (row,) = rows
        # Bi-temporal params populated, legacy fields mirrored.
        assert row["t_created"] == row["created_at"]
        assert row["t_valid"] == row["valid_at"]
        # And populated as ISO8601 strings, not None.
        assert isinstance(row["t_created"], str) and "T" in row["t_created"]
        assert row["emb"] == [0.1, 0.2, 0.3]
        assert row["episodes"] == [1, 2]

    def test_explicit_t_valid_overrides_now(self):
        client, writer = self._make_client_with_fake_writer()
        client.create_edge(
            source_uuid="src",
            target_uuid="tgt",
            relationship="USES",
            fact="A uses B since 2024",
            episode_ids=[],
            group_id="technical",
            t_valid="2024-06-15T00:00:00+00:00",
        )
        (row,) = writer.create_edges.call_args.args[0]
        # t_valid honors the caller-supplied date; valid_at mirrors.
        assert row["t_valid"] == "2024-06-15T00:00:00+00:00"
        assert row["valid_at"] == "2024-06-15T00:00:00+00:00"
        # t_created is still "now" (not the caller's t_valid).
        assert row["t_created"] != row["t_valid"]

    def test_contradiction_invalidates_old_edge_before_create(self):
        """When the detector confirms a contradiction, the OLD edge is
        invalidated BEFORE the new edge is written. The new edge is still
        written."""
        client, writer = self._make_client_with_fake_writer()
        parent = MagicMock()
        parent.attach_mock(writer, "writer")

        detector = MagicMock()
        detector.detect_contradictions.return_value = ["old-edge-uuid"]

        client.create_edge(
            source_uuid="src",
            target_uuid="tgt",
            relationship="USES",
            fact="A uses B v2",
            episode_ids=[],
            group_id="technical",
            detector=detector,
            extracted_fact=_fact(),
        )

        ops = [c[0] for c in parent.method_calls]
        assert ops == ["writer.invalidate_edges", "writer.create_edges"]
        (items, group) = writer.invalidate_edges.call_args_list[0].args
        assert group == "technical"
        assert items[0][0] == "old-edge-uuid"

    def test_contradiction_records_superseder(self):
        """The invalidated old edge records invalidated_by = the NEW edge's uuid (schema 028)."""
        client, writer = self._make_client_with_fake_writer()
        detector = MagicMock()
        detector.detect_contradictions.return_value = ["old-edge-uuid"]

        client.create_edge(
            source_uuid="src",
            target_uuid="tgt",
            relationship="USES",
            fact="A uses B v2",
            episode_ids=[],
            group_id="technical",
            detector=detector,
            extracted_fact=_fact(),
        )

        (row,) = writer.create_edges.call_args.args[0]
        kwargs = writer.invalidate_edges.call_args_list[0].kwargs
        assert kwargs.get("invalidated_by") == row["edge_uuid"]  # superseder = the new edge

    def test_no_contradictions_means_no_invalidate(self):
        client, writer = self._make_client_with_fake_writer()
        detector = MagicMock()
        detector.detect_contradictions.return_value = []

        client.create_edge(
            source_uuid="src",
            target_uuid="tgt",
            relationship="USES",
            fact="A uses B",
            episode_ids=[],
            group_id="technical",
            detector=detector,
            extracted_fact=_fact(),
        )

        writer.invalidate_edges.assert_not_called()
        writer.create_edges.assert_called_once()

    def test_pre_known_invalid_at_bookends_new_edge(self):
        """A fact that arrives already bookended ("from 2020 to 2022") is
        created and immediately invalidated with the extracted date."""
        client, writer = self._make_client_with_fake_writer()
        client.create_edge(
            source_uuid="src",
            target_uuid="tgt",
            relationship="WORKED_AT",
            fact="Kyle worked at X from 2020 to 2022",
            episode_ids=[],
            group_id="technical",
            t_valid="2020-01-01T00:00:00+00:00",
            t_invalid="2022-01-01T00:00:00+00:00",
        )
        (row,) = writer.create_edges.call_args.args[0]
        (items, _group) = writer.invalidate_edges.call_args.args
        assert items == [(row["edge_uuid"], "2022-01-01T00:00:00+00:00")]

    def test_invalidate_edge_passes_iso_timestamp(self):
        client, writer = self._make_client_with_fake_writer()
        client.invalidate_edge("edge-1", "technical")
        (items, group) = writer.invalidate_edges.call_args.args
        assert group == "technical"
        ((uuid_, ts),) = items
        assert uuid_ == "edge-1"
        assert isinstance(ts, str) and "T" in ts
