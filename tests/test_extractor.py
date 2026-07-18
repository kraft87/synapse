"""
Tests for Phase 2b extraction pipeline.

Unit tests run without any external services.
Integration tests (marked @pytest.mark.integration) require a live FalkorDB
and are skipped in CI.
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ingestion.extractor import (
    _SEMANTIC_POOL_LIMIT,
    DeterministicExtractor,
    EntityResolver,
    ExtractionPipeline,
    LLMExtractor,
    _cosine_similarity,
    _MalformedExtractionResponse,
)
from ingestion.models import (
    CombinedExtraction,
    ExtractedEntity,
    ExtractedFact,
    ExtractionResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entity(name: str, etype: str = "Tool", summary: str = "") -> ExtractedEntity:
    return ExtractedEntity(name=name, type=etype, summary=summary)


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


def test_cosine_similarity_identical():
    v = [1.0, 0.0, 0.0]
    assert _cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert _cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert _cosine_similarity(a, b) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# DeterministicExtractor
# ---------------------------------------------------------------------------


class TestDeterministicExtractor:
    extractor = DeterministicExtractor()

    def test_extracts_file_paths(self):
        episodes: list[dict[str, Any]] = [
            {"content": "Edited /home/user/services/synapse/ingestion/db.py", "metadata": {}},
        ]
        entities = self.extractor.extract(episodes)
        names = [e.name for e in entities]
        assert any("db.py" in n or "/ingestion/db.py" in n for n in names)

    def test_extracts_urls(self):
        episodes: list[dict[str, Any]] = [
            {
                "content": "See https://docs.voyageai.com/docs/embeddings for details",
                "metadata": {},
            },
        ]
        entities = self.extractor.extract(episodes)
        names = [e.name for e in entities]
        assert any("docs.voyageai.com" in n or "voyageai.com" in n for n in names)

    def test_extracts_error_types(self):
        episodes: list[dict[str, Any]] = [
            {
                "content": "Got UniqueViolation: duplicate key value violates unique constraint",
                "metadata": {},
            },
        ]
        entities = self.extractor.extract(episodes)
        types = [e.type for e in entities]
        assert "Issue" in types

    def test_extracts_tools_from_metadata(self):
        episodes: list[dict[str, Any]] = [
            {
                "content": "Used bash and read tools",
                "metadata": {"tools_used": ["bash", "read", "edit"]},
            }
        ]
        entities = self.extractor.extract(episodes)
        names = [e.name for e in entities]
        assert "bash" in names
        assert "read" in names
        assert "edit" in names

    def test_deduplicates_entities(self):
        episodes: list[dict[str, Any]] = [
            {"content": "Error: UniqueViolation in test A", "metadata": {}},
            {"content": "Error: UniqueViolation in test B", "metadata": {}},
        ]
        entities = self.extractor.extract(episodes)
        names = [e.name for e in entities]
        # Should not duplicate UniqueViolation
        assert names.count("UniqueViolation") <= 1

    def test_returns_empty_for_clean_content(self):
        episodes: list[dict[str, Any]] = [
            {"content": "The session went well.", "metadata": {}},
        ]
        entities = self.extractor.extract(episodes)
        # No errors/URLs/tools in clean text — may return empty or only generic
        assert isinstance(entities, list)

    def test_ignores_missing_metadata_key(self):
        episodes: list[dict[str, Any]] = [
            {"content": "Some content", "metadata": None},  # type: ignore[arg-type]
        ]
        # Should not raise
        entities = self.extractor.extract(episodes)
        assert isinstance(entities, list)


# ---------------------------------------------------------------------------
# LLMExtractor — parse_response
# ---------------------------------------------------------------------------


class TestLLMExtractorParsing:
    """``_parse_response`` returns a validated ``CombinedExtraction`` (Phase 1).

    Post-Phase-1 it raises ``_MalformedExtractionResponse`` on bad JSON so the
    tenacity retry loop in ``extract()`` can re-prompt with feedback. Tests
    here exercise the parser directly; retry behavior is covered in
    ``TestLLMExtractorRetry``.
    """

    extractor = LLMExtractor.__new__(LLMExtractor)

    def test_parse_valid_json(self):
        raw = """
        {
          "entities": [
            {"name": "FalkorDB", "type": "Tool", "summary": "Graph database"},
            {"name": "Synapse", "type": "Project", "summary": "Memory layer"}
          ],
          "facts": [{"source": "Synapse", "target": "FalkorDB", "relationship": "USES", "fact": "Synapse uses FalkorDB as its knowledge graph store"}]
        }
        """
        result = LLMExtractor._parse_response(raw)
        assert isinstance(result, CombinedExtraction)
        assert {e.name for e in result.entities} == {"FalkorDB", "Synapse"}
        assert len(result.facts) == 1
        assert result.facts[0].relationship == "USES"
        # Both endpoints declared → no facts dropped.
        assert result.dropped_facts == []

    def test_parse_json_in_markdown_block(self):
        raw = """
Here is the extraction:
```json
{"entities": [{"name": "voyage-4-large", "type": "Tool", "summary": "Embedding model"}], "facts": []}
```
"""
        result = LLMExtractor._parse_response(raw)
        assert result.entities[0].name == "voyage-4-large"

    def test_parse_invalid_json_raises(self):
        with pytest.raises(_MalformedExtractionResponse):
            LLMExtractor._parse_response("This is not JSON at all.")

    def test_parse_partial_json_missing_facts(self):
        raw = '{"entities": [{"name": "Synapse", "type": "Project", "summary": "Memory layer"}]}'
        result = LLMExtractor._parse_response(raw)
        assert result.entities[0].name == "Synapse"
        assert result.facts == []

    def test_parse_entities_missing_required_fields(self):
        raw = '{"entities": [{"name": "X"}], "facts": []}'
        result = LLMExtractor._parse_response(raw)
        # Missing type defaults gracefully (type has a fallback)
        assert len(result.entities) == 1
        assert result.entities[0].name == "X"

    def test_parse_empty_result(self):
        raw = '{"entities": [], "facts": []}'
        result = LLMExtractor._parse_response(raw)
        assert result.entities == []
        assert result.facts == []


# ---------------------------------------------------------------------------
# CombinedExtraction cross-reference validator (Phase 1)
# ---------------------------------------------------------------------------


class TestCombinedExtractionValidator:
    """The ``model_validator`` on CombinedExtraction enforces entity↔fact
    consistency:

    - Facts whose source or target doesn't normalize-exact-match a declared
      entity name are DROPPED into ``dropped_facts`` (not raised).
    - Empty-named entities are dropped into ``dropped_entities``.

    Entities without any fact reference are NOT dropped here — the orphan
    drop happens downstream in ``process_item`` because deterministic
    entities haven't been merged in yet.
    """

    def test_fact_with_known_endpoints_kept(self):
        ce = CombinedExtraction(
            entities=[
                ExtractedEntity(name="Synapse", type="Project"),
                ExtractedEntity(name="FalkorDB", type="Tool"),
            ],
            facts=[
                ExtractedFact(
                    source="Synapse",
                    target="FalkorDB",
                    relationship="USES",
                    fact="Synapse uses FalkorDB",
                )
            ],
        )
        assert len(ce.facts) == 1
        assert ce.dropped_facts == []

    def test_fact_with_unknown_source_dropped(self):
        ce = CombinedExtraction(
            entities=[ExtractedEntity(name="FalkorDB", type="Tool")],
            facts=[
                ExtractedFact(
                    source="Mystery",
                    target="FalkorDB",
                    relationship="USES",
                    fact="Mystery uses FalkorDB",
                )
            ],
        )
        assert ce.facts == []
        assert len(ce.dropped_facts) == 1
        assert ce.dropped_facts[0].source == "Mystery"

    def test_fact_with_unknown_target_dropped(self):
        ce = CombinedExtraction(
            entities=[ExtractedEntity(name="Synapse", type="Project")],
            facts=[
                ExtractedFact(
                    source="Synapse",
                    target="Mystery",
                    relationship="USES",
                    fact="Synapse uses Mystery",
                )
            ],
        )
        assert ce.facts == []
        assert len(ce.dropped_facts) == 1

    def test_cross_reference_is_case_and_whitespace_insensitive(self):
        ce = CombinedExtraction(
            entities=[ExtractedEntity(name="Synapse", type="Project")],
            facts=[
                # Self-referencing fact with name varying in casing and trailing space.
                ExtractedFact(
                    source="synapse",
                    target=" SYNAPSE ",
                    relationship="IS",
                    fact="Synapse is the memory layer",
                ),
            ],
        )
        assert len(ce.facts) == 1
        assert ce.dropped_facts == []

    def test_empty_entity_name_dropped(self):
        ce = CombinedExtraction(
            entities=[
                ExtractedEntity(name="Synapse", type="Project"),
                ExtractedEntity(name="   ", type="Other"),
            ],
            facts=[],
        )
        assert [e.name for e in ce.entities] == ["Synapse"]
        assert len(ce.dropped_entities) == 1

    def test_entity_without_fact_is_NOT_dropped_here(self):
        # Orphan-drop is downstream in process_item — at the model level we
        # preserve entities so deterministic-extractor entities can merge
        # before the final orphan pass.
        ce = CombinedExtraction(
            entities=[
                ExtractedEntity(name="Synapse", type="Project"),
                ExtractedEntity(name="Lonely", type="Topic"),
            ],
            facts=[
                ExtractedFact(
                    source="Synapse",
                    target="Synapse",
                    relationship="IS",
                    fact="Synapse is itself",
                )
            ],
        )
        assert {e.name for e in ce.entities} == {"Synapse", "Lonely"}


# ---------------------------------------------------------------------------
# LLMExtractor retry-on-malformed-JSON (Phase 1)
# ---------------------------------------------------------------------------


class TestLLMExtractorRetry:
    """``extract()`` retries up to 3 attempts on malformed responses, then
    logs a warning and returns an empty ``ExtractionResult`` rather than
    crashing the pipeline.
    """

    @staticmethod
    def _mock_response(text: str) -> Any:
        msg = MagicMock()
        msg.content = [MagicMock(text=text)]
        return msg

    def test_retries_then_succeeds_on_second_attempt(self):
        good = self._mock_response(
            '{"entities": [{"name": "Synapse", "type": "Project"}],'
            ' "facts": [{"source": "Synapse", "target": "Synapse",'
            ' "relationship": "IS", "fact": "Synapse is the memory layer"}]}'
        )
        bad = self._mock_response("not json at all")

        client = MagicMock()
        client.messages.create.side_effect = [bad, good]

        extractor = LLMExtractor(llm_client=client, model="claude-haiku-4-5")
        result = extractor.extract(summary="anything", context_entities=[])

        assert isinstance(result, ExtractionResult)
        assert [e.name for e in result.entities] == ["Synapse"]
        # Both attempts should have been called.
        assert client.messages.create.call_count == 2
        # Second call's prompt should include the feedback string verbatim.
        second_call_prompt = client.messages.create.call_args_list[1].kwargs["messages"][0][
            "content"
        ]
        assert "Your last response failed to parse" in second_call_prompt

    def test_returns_empty_after_all_attempts_fail(self):
        bad = self._mock_response("not json at all")
        client = MagicMock()
        client.messages.create.return_value = bad

        extractor = LLMExtractor(llm_client=client, model="claude-haiku-4-5")
        result = extractor.extract(summary="anything", context_entities=[])

        assert isinstance(result, ExtractionResult)
        assert result.entities == []
        assert result.facts == []
        # 3 total attempts (initial + 2 retries) per the spec.
        assert client.messages.create.call_count == 3


class TestExtractionPromptDisciplines:
    """The v2 prompt ports three Mastra observational-memory disciplines:
    authoritative-assertion framing, per-mention date anchoring, and
    supersession naming. Assert each survives in the rendered prompt and that
    the session date threads through the format call.
    """

    @staticmethod
    def _rendered_prompt(session_date: str | None = None) -> str:
        msg = MagicMock()
        msg.content = [MagicMock(text='{"entities": [], "facts": []}')]
        client = MagicMock()
        client.messages.create.return_value = msg
        LLMExtractor(llm_client=client, model="claude-haiku-4-5").extract(
            summary="anything", context_entities=[], session_date=session_date
        )
        return client.messages.create.call_args.kwargs["messages"][0]["content"]

    def test_session_date_threaded_into_prompt(self):
        assert "2026-01-15" in self._rendered_prompt(session_date="2026-01-15")

    def test_unknown_session_date_renders_placeholder(self):
        assert "Session date: unknown" in self._rendered_prompt(session_date=None)

    def test_framing_and_disciplines_present(self):
        prompt = self._rendered_prompt(session_date="2026-01-15")
        # Framing: capture-or-forget + user-is-authoritative.
        assert "ONLY memory" in prompt
        assert "authoritative" in prompt
        # Per-mention date anchoring.
        assert "ANCHOR EVERY DATE" in prompt
        assert "(meaning" in prompt
        # Supersession naming.
        assert "NAME WHAT CHANGED" in prompt


# ---------------------------------------------------------------------------
# EntityResolver
# ---------------------------------------------------------------------------


def _mock_falkordb(similar_nodes: list[dict] | None = None) -> MagicMock:
    """Mock KGClient.find_similar_nodes()."""
    client = MagicMock()
    client.find_similar_nodes.return_value = similar_nodes or []
    return client


class TestEntityResolver:
    def _make_embedding(self, val: float, dims: int = 4) -> list[float]:
        """Unit vector in a specific direction."""
        angle = val * math.pi / 2
        v = [math.cos(angle)] + [math.sin(angle) / math.sqrt(dims - 1)] * (dims - 1)
        norm = math.sqrt(sum(x**2 for x in v))
        return [x / norm for x in v]

    def _mock_embedder(self, *vectors):
        mock = MagicMock()
        mock.embed.side_effect = list(vectors)
        return mock

    def test_no_existing_nodes_assigns_new_uuid(self):
        v = self._make_embedding(0.1)
        embedder = self._mock_embedder([v])
        resolver = EntityResolver(embedder=embedder, llm_client=None)
        entities = [_entity("FalkorDB")]
        falkordb = _mock_falkordb(similar_nodes=[])
        mapping = resolver.resolve(entities, kg_client=falkordb, group_id="technical")
        assert "FalkorDB" in mapping
        assert mapping["FalkorDB"].startswith("new:")

    def test_dissimilar_node_no_merge(self):
        v_new = self._make_embedding(0.0)
        embedder = self._mock_embedder([v_new])
        # score=0.8 → similarity=0.2, below 0.85 threshold
        falkordb = _mock_falkordb(
            similar_nodes=[{"uuid": "abc123", "name": "PostgreSQL", "score": 0.8}]
        )
        resolver = EntityResolver(embedder=embedder, llm_client=None, similarity_threshold=0.85)
        mapping = resolver.resolve([_entity("FalkorDB")], kg_client=falkordb, group_id="technical")
        assert mapping["FalkorDB"].startswith("new:")

    def test_similar_node_triggers_llm_confirm_merge(self):
        same_vec = self._make_embedding(0.1)
        embedder = self._mock_embedder([same_vec])
        # score=0.12 → similarity=0.88, sits between threshold (0.85) and
        # autoconfirm (0.95) so the LLM confirm path is exercised.
        falkordb = _mock_falkordb(
            similar_nodes=[{"uuid": "abc123", "name": "FalkorDB graph DB", "score": 0.12}]
        )

        llm_client = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text='{"results": [{"id": 0, "duplicate_candidate_id": 0}]}')]
        llm_client.messages.create.return_value = mock_msg

        resolver = EntityResolver(
            embedder=embedder, llm_client=llm_client, similarity_threshold=0.85
        )
        mapping = resolver.resolve([_entity("FalkorDB")], kg_client=falkordb, group_id="technical")
        assert mapping["FalkorDB"] == "abc123"

    def test_similar_node_llm_denies_no_merge(self):
        same_vec = self._make_embedding(0.1)
        embedder = self._mock_embedder([same_vec])
        falkordb = _mock_falkordb(
            similar_nodes=[{"uuid": "xyz789", "name": "FlatDB", "score": 0.12}]
        )

        llm_client = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = [
            MagicMock(text='{"results": [{"id": 0, "duplicate_candidate_id": -1}]}')
        ]
        llm_client.messages.create.return_value = mock_msg

        resolver = EntityResolver(
            embedder=embedder, llm_client=llm_client, similarity_threshold=0.85
        )
        mapping = resolver.resolve([_entity("FalkorDB")], kg_client=falkordb, group_id="technical")
        assert mapping["FalkorDB"].startswith("new:")

    def test_autoconfirm_skips_llm(self):
        same_vec = self._make_embedding(0.1)
        embedder = self._mock_embedder([same_vec])
        # score=0.03 → similarity=0.97, above the 0.95 autoconfirm threshold.
        falkordb = _mock_falkordb(
            similar_nodes=[{"uuid": "abc123", "name": "FalkorDB", "score": 0.03}]
        )

        llm_client = MagicMock()
        # If the LLM is consulted at all, the test should fail.
        llm_client.messages.create.side_effect = AssertionError(
            "LLM should not be called above autoconfirm threshold"
        )

        resolver = EntityResolver(
            embedder=embedder,
            llm_client=llm_client,
            similarity_threshold=0.85,
            autoconfirm_threshold=0.95,
        )
        mapping = resolver.resolve([_entity("FalkorDB")], kg_client=falkordb, group_id="technical")
        assert mapping["FalkorDB"] == "abc123"
        llm_client.messages.create.assert_not_called()

    def test_multiple_entities_resolved_independently(self):
        v1 = self._make_embedding(0.0)
        v2 = self._make_embedding(0.5)
        embedder = self._mock_embedder([v1, v2])
        falkordb = _mock_falkordb(similar_nodes=[])
        resolver = EntityResolver(embedder=embedder, llm_client=None)
        entities = [_entity("Synapse"), _entity("FalkorDB")]
        mapping = resolver.resolve(entities, kg_client=falkordb, group_id="technical")
        assert "Synapse" in mapping
        assert "FalkorDB" in mapping
        assert mapping["Synapse"] != mapping["FalkorDB"]

    def test_batch_confirm_one_call_for_many_pending(self):
        # Two entities both land in the 0.85-0.95 confirm band -> they must be
        # resolved in a SINGLE batched LLM call, not one subprocess each.
        embedder = self._mock_embedder([self._make_embedding(0.1), self._make_embedding(0.2)])
        # score=0.12 -> similarity 0.88 (confirm band) for every lookup.
        falkordb = _mock_falkordb(
            similar_nodes=[{"uuid": "cand0", "name": "Candidate", "score": 0.12}]
        )
        llm_client = MagicMock()
        mock_msg = MagicMock()
        # id 0 merges with its candidate 0; id 1 is distinct (-1).
        mock_msg.content = [
            MagicMock(
                text='{"results": [{"id": 0, "duplicate_candidate_id": 0}, '
                '{"id": 1, "duplicate_candidate_id": -1}]}'
            )
        ]
        llm_client.messages.create.return_value = mock_msg

        resolver = EntityResolver(
            embedder=embedder, llm_client=llm_client, similarity_threshold=0.85
        )
        mapping = resolver.resolve(
            [_entity("Alpha"), _entity("Beta")], kg_client=falkordb, group_id="technical"
        )
        assert llm_client.messages.create.call_count == 1  # ONE batched call
        assert mapping["Alpha"] == "cand0"
        assert mapping["Beta"].startswith("new:")

    def test_batch_confirm_empty_response_defaults_distinct(self):
        # The old bug: an empty SDK body -> json.loads raises -> code defaulted
        # to SAME-entity (silent wrong-merge). Now the conservative default is
        # DISTINCT: every pending entity becomes a new node.
        embedder = self._mock_embedder([self._make_embedding(0.1)])
        falkordb = _mock_falkordb(
            similar_nodes=[{"uuid": "cand0", "name": "Candidate", "score": 0.12}]
        )
        llm_client = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="")]  # empty body -> parse failure
        llm_client.messages.create.return_value = mock_msg

        resolver = EntityResolver(
            embedder=embedder, llm_client=llm_client, similarity_threshold=0.85
        )
        mapping = resolver.resolve([_entity("Gamma")], kg_client=falkordb, group_id="technical")
        assert mapping["Gamma"].startswith("new:")  # NOT merged into cand0

    def test_batch_confirm_no_llm_trusts_top_candidate(self):
        # With no LLM client, a confirm-band match trusts the top candidate
        # (mirrors the prior no-LLM behaviour) without any LLM call.
        embedder = self._mock_embedder([self._make_embedding(0.1)])
        falkordb = _mock_falkordb(
            similar_nodes=[{"uuid": "cand0", "name": "Candidate", "score": 0.12}]
        )
        resolver = EntityResolver(embedder=embedder, llm_client=None, similarity_threshold=0.85)
        mapping = resolver.resolve([_entity("Delta")], kg_client=falkordb, group_id="technical")
        assert mapping["Delta"] == "cand0"


# ---------------------------------------------------------------------------
# Stage 6 — contradiction-detection pipeline (post-RRF + idx-based prompt)
# ---------------------------------------------------------------------------


class TestBuildResolutionPrompt:
    """The prompt fed to Haiku in Stage 6b must:
    - Use integer idx values, not raw UUIDs (token efficiency + hallucination guard)
    - Separate pair-matched (likely-duplicate) candidates from semantic
      (likely-contradiction) candidates into two labelled sections
    - Number indices continuously across both sections so the LLM can refer
      to either pool with one idx range
    """

    def test_prompt_uses_idx_not_uuids(self):
        from ingestion.extractor import build_resolution_prompt

        new_fact = "Synapse uses voyage-4-large for embeddings"
        dup_pool = [{"uuid": "uuid-a", "fact": "Synapse uses OpenAI text-embedding-3-large"}]
        contradiction_pool = [
            {"uuid": "uuid-b", "fact": "Synapse uses pgvector for vector storage"}
        ]
        prompt, idx_to_uuid = build_resolution_prompt(new_fact, dup_pool, contradiction_pool)

        # Idx-based references, never raw UUIDs
        assert "uuid-a" not in prompt
        assert "uuid-b" not in prompt
        assert "[0]" in prompt
        assert "[1]" in prompt
        # Map round-trips
        assert idx_to_uuid[0] == "uuid-a"
        assert idx_to_uuid[1] == "uuid-b"

    def test_prompt_has_two_labelled_sections(self):
        from ingestion.extractor import build_resolution_prompt

        prompt, _ = build_resolution_prompt(
            "new fact",
            [{"uuid": "a", "fact": "dup-fact"}],
            [{"uuid": "b", "fact": "contra-fact"}],
        )
        # Section labels (case-insensitive substring match)
        lower = prompt.lower()
        assert "existing facts" in lower or "duplicate" in lower
        assert "invalidation candidates" in lower or "contradiction" in lower
        # The two facts appear under different labels
        dup_section_idx = lower.find("dup-fact")
        contra_section_idx = lower.find("contra-fact")
        assert dup_section_idx >= 0 and contra_section_idx >= 0
        assert dup_section_idx != contra_section_idx

    def test_indices_are_continuous_across_pools(self):
        """idx values must NOT restart at 0 between sections — Graphiti uses one continuous idx range."""
        from ingestion.extractor import build_resolution_prompt

        _, idx_map = build_resolution_prompt(
            "x",
            [{"uuid": "a", "fact": "f1"}, {"uuid": "b", "fact": "f2"}],
            [{"uuid": "c", "fact": "f3"}],
        )
        assert set(idx_map.keys()) == {0, 1, 2}
        assert idx_map[0] == "a"
        assert idx_map[1] == "b"
        assert idx_map[2] == "c"

    def test_empty_pools_still_produce_a_prompt(self):
        from ingestion.extractor import build_resolution_prompt

        prompt, idx_map = build_resolution_prompt("new fact", [], [])
        assert "new fact" in prompt
        assert idx_map == {}


class TestBatchResolutionPrompt:
    """Batched stage-6b prompt (one LLM call for many facts). Each fact has
    its OWN candidate lists with idx scoped per-fact — never across facts.
    Mirrors the dedupe_nodes.build_batch_prompt design from PR #83.
    """

    def test_per_fact_idx_scoping(self):
        from ingestion.extractor import build_batch_resolution_prompt

        items = [
            {
                "id": 0,
                "new_fact": "Synapse uses voyage-4-large",
                "existing_pool": [{"uuid": "a", "fact": "Synapse uses voyage-3"}],
                "candidate_pool": [],
            },
            {
                "id": 1,
                "new_fact": "Synapse uses FalkorDB on port 6379",
                "existing_pool": [],
                "candidate_pool": [{"uuid": "b", "fact": "Synapse uses Neo4j on 7474"}],
            },
        ]
        prompt, per_item_maps = build_batch_resolution_prompt(items)

        # No UUIDs in prompt — token saving + hallucination guard.
        assert "uuid" not in prompt.lower().replace("uuids", "")
        # Fact 0 owns idx 0 -> uuid "a"; fact 1 owns idx 0 -> uuid "b". Same
        # local idx, different uuids. Per-fact scoping is the whole point.
        assert per_item_maps[0][0] == "a"
        assert per_item_maps[1][0] == "b"
        # Both facts' content appears in the rendered prompt.
        assert "voyage-3" in prompt
        assert "Neo4j on 7474" in prompt

    def test_indices_continuous_within_one_fact(self):
        from ingestion.extractor import build_batch_resolution_prompt

        items = [
            {
                "id": 0,
                "new_fact": "x",
                "existing_pool": [
                    {"uuid": "p0", "fact": "f1"},
                    {"uuid": "p1", "fact": "f2"},
                ],
                "candidate_pool": [{"uuid": "c0", "fact": "f3"}],
            }
        ]
        _, per_item_maps = build_batch_resolution_prompt(items)
        # Existing pool idx 0,1 then candidate pool continues at 2 — same
        # continuous scheme as build_resolution_prompt single-fact version.
        assert per_item_maps[0] == {0: "p0", 1: "p1", 2: "c0"}

    def test_empty_pools_yields_empty_map_for_id(self):
        from ingestion.extractor import build_batch_resolution_prompt

        items = [{"id": 7, "new_fact": "novel", "existing_pool": [], "candidate_pool": []}]
        prompt, per_item_maps = build_batch_resolution_prompt(items)
        assert per_item_maps[7] == {}
        assert "novel" in prompt


class TestStage6bBatchConfirm:
    """The batched _stage6b_batch_confirm collapses N per-fact 30s claude-CLI
    subprocess calls into ONE Haiku call. Each fact gets back its OWN dup +
    contradicted decisions in a single results array.
    """

    def _pipeline_with_llm(self, llm_text: str):
        """Build a minimal ExtractionPipeline whose only wiring is the LLM
        client that returns ``llm_text`` from messages.create. We never
        touch stage1-5 or stage7, so the rest of the pipeline can be left
        unset on the bare instance."""
        from ingestion.extractor import ExtractionPipeline

        pipe = ExtractionPipeline.__new__(ExtractionPipeline)
        pipe._contradiction_model = "claude-haiku-4-5"
        fake_msg = MagicMock()
        fake_msg.content = [MagicMock(text=llm_text)]
        fake_client = MagicMock()
        fake_client.messages.create.return_value = fake_msg
        # _stage6b_batch_confirm reaches self._llm._client.messages.create
        pipe._llm = MagicMock()
        pipe._llm._client = fake_client
        return pipe, fake_client

    def _facts(self, n: int):
        from ingestion.models import ExtractedFact

        return [
            ExtractedFact(source=f"S{i}", target=f"T{i}", relationship="USES", fact=f"fact-{i}")
            for i in range(n)
        ]

    def test_one_llm_call_for_many_facts(self):
        """Two facts, two pool entries — one batched LLM call total."""
        import json as _json

        llm_payload = _json.dumps(
            {
                "results": [
                    {"id": 0, "duplicate_facts": [0], "contradicted_facts": []},
                    {"id": 1, "duplicate_facts": [], "contradicted_facts": [0]},
                ]
            }
        )
        pipe, llm_client = self._pipeline_with_llm(llm_payload)
        facts = self._facts(2)
        candidates_map = {
            0: ([{"uuid": "dup-uuid", "fact": "old fact 0"}], []),
            1: ([], [{"uuid": "contra-uuid", "fact": "old fact 1"}]),
        }
        skip_indices, invalidate, reinforce, ok = pipe._stage6b_batch_confirm(facts, candidates_map)
        assert ok

        # ONE LLM call for both facts (the whole point of the PR)
        assert llm_client.messages.create.call_count == 1
        # Fact 0 is a pure duplicate -> skipped, no invalidations, reinforces its match
        assert 0 in skip_indices
        assert 0 not in invalidate
        assert reinforce[0] == ["dup-uuid"]  # the dedup-hit signal Stage 7 consumes
        # Fact 1 contradicts an existing edge -> NOT skipped, invalidate set
        assert 1 not in skip_indices
        assert invalidate[1] == ["contra-uuid"]

    def test_empty_candidates_map_short_circuits(self):
        """No candidates anywhere => no LLM call at all."""
        pipe, llm_client = self._pipeline_with_llm("{}")
        skip_indices, invalidate, reinforce, ok = pipe._stage6b_batch_confirm(self._facts(3), {})
        assert ok  # empty map is a clean no-op, not a failure
        assert skip_indices == set()
        assert invalidate == {}
        assert reinforce == {}
        llm_client.messages.create.assert_not_called()

    def test_tolerates_trailing_prose_in_response(self):
        """raw_decode parses the first JSON object — model can ramble after."""
        text = (
            '{"results": [{"id": 0, "duplicate_facts": [], '
            '"contradicted_facts": [0]}]}\n\nHere is my reasoning...'
        )
        pipe, _ = self._pipeline_with_llm(text)
        facts = self._facts(1)
        cmap = {0: ([], [{"uuid": "u", "fact": "old"}])}
        _, invalidate, _, _ = pipe._stage6b_batch_confirm(facts, cmap)
        assert invalidate == {0: ["u"]}

    def test_missing_fact_in_response_treated_as_no_op(self):
        """If the LLM omits some fact ids in its results array, treat those
        facts as (no skip, no contradictions). Fail-open on omission, same
        conservative posture as the per-fact except branch."""
        import json as _json

        llm_payload = _json.dumps(
            {"results": [{"id": 0, "duplicate_facts": [], "contradicted_facts": []}]}
        )
        pipe, _ = self._pipeline_with_llm(llm_payload)
        facts = self._facts(2)
        cmap = {
            0: ([{"uuid": "a", "fact": "f"}], []),
            1: ([{"uuid": "b", "fact": "g"}], []),
        }
        skip_indices, invalidate, _, _ = pipe._stage6b_batch_confirm(facts, cmap)
        # Fact 1 not in response -> not skipped, no contradictions
        assert 1 not in skip_indices
        assert 1 not in invalidate

    def test_llm_exception_returns_empty_no_raise(self):
        """An LLM transport failure must NOT raise: contradiction misses are
        recoverable on next extraction; blocking writes is not."""
        pipe, llm_client = self._pipeline_with_llm("{}")
        llm_client.messages.create.side_effect = RuntimeError("upstream 500")
        skip_indices, invalidate, reinforce, ok = pipe._stage6b_batch_confirm(
            self._facts(1),
            {0: ([{"uuid": "x", "fact": "y"}], [])},
        )
        assert ok is False  # the gate's shadow log must not treat this as a verdict
        assert skip_indices == set()
        assert invalidate == {}
        assert reinforce == {}


class TestStage6aDedup:
    """The pair-pool and semantic-pool overlap when an edge clears both checks.
    The pair entry wins (stronger prior — same endpoints implies duplicate
    semantics). Drop the duplicate from the semantic pool BEFORE prompting so
    the LLM doesn't see the same fact twice and get confused.
    """

    def test_dedup_pair_then_semantic(self):
        from ingestion.extractor import dedupe_pools

        pair = [{"uuid": "A", "fact": "shared"}, {"uuid": "B", "fact": "pair-only"}]
        semantic = [
            {"uuid": "A", "fact": "shared"},
            {"uuid": "C", "fact": "semantic-only"},
        ]
        pair_out, sem_out = dedupe_pools(pair, semantic)
        # Pair list untouched
        assert {p["uuid"] for p in pair_out} == {"A", "B"}
        # A removed from semantic since it already won the pair check
        assert {s["uuid"] for s in sem_out} == {"C"}


# ---------------------------------------------------------------------------
# Per-stage model resolution (issue #8)
# ---------------------------------------------------------------------------


class TestPerStageModels:
    """SYNAPSE_<STAGE>_MODEL env overrides reach each stage independently."""

    def _pipeline(self) -> ExtractionPipeline:
        embedder = MagicMock()
        embedder.embed.side_effect = lambda names, task=None: [[0.0, 0.0, 0.0, 0.0] for _ in names]
        db = MagicMock()
        return ExtractionPipeline(
            db=db,
            llm_client=MagicMock(),
            embedder=embedder,
            kg_client=_mock_falkordb(similar_nodes=[]),
        )

    def test_stage_envs_reach_their_stages(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_TIMELINE_MODEL", "stage/timeline")
        monkeypatch.setenv("SYNAPSE_CONTRADICTION_MODEL", "stage/contradiction")
        pipe = self._pipeline()
        assert pipe._timeline_gate._model == "stage/timeline"
        assert pipe._contradiction_model == "stage/contradiction"
        assert pipe._contradiction_detector._model == "stage/contradiction"
        # Untouched stages keep the code default.
        assert pipe._llm._model == "claude-haiku-4-5"
        assert pipe._preferences_gate._model == "claude-haiku-4-5"

    def test_global_env_covers_every_stage(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_LLM_MODEL", "global/model")
        pipe = self._pipeline()
        assert pipe._llm._model == "global/model"
        assert pipe._timeline_gate._model == "global/model"
        assert pipe._preferences_gate._model == "global/model"
        assert pipe._edge_date_extractor._model == "global/model"
        assert pipe._contradiction_model == "global/model"


# ---------------------------------------------------------------------------
# process_item — pre-resolve orphan filter
# ---------------------------------------------------------------------------


class TestProcessItemOrphanPrefilter:
    """``process_item`` must resolve ONLY entities referenced by a fact.

    A real corpus summary yields ~1300 deterministic entity mentions backing
    only ~6-12 facts. Resolving every one of them in Stage 4 (per-entity
    vector search + up to 4 LLM "same entity?" confirms, each a ~30s claude
    CLI subprocess) cost ~90 min/summary -- almost all of it on entities no
    fact references, which the downstream orphan-drop then discarded before
    write. The pre-resolve filter prunes to the fact-referenced set BEFORE
    Stage 4 so resolution touches dozens, not thousands.
    """

    def _pipeline(self) -> ExtractionPipeline:
        embedder = MagicMock()
        embedder.embed.side_effect = lambda names, task=None: [[0.0, 0.0, 0.0, 0.0] for _ in names]
        db = MagicMock()
        db.get_session_episodes.return_value = []
        db.get_synth_document_source_ids.return_value = []
        return ExtractionPipeline(
            db=db,
            llm_client=MagicMock(),
            embedder=embedder,
            kg_client=_mock_falkordb(similar_nodes=[]),
        )

    def test_only_fact_referenced_entities_resolved(self):
        pipe = self._pipeline()

        # "noise/path.py" is a deterministic orphan no fact references; it
        # must never reach Stage 4. Only the two fact endpoints should.
        det = [_entity("noise/path.py")]
        llm_entities = [_entity("FalkorDB"), _entity("Synapse")]
        facts = [
            ExtractedFact(
                source="Synapse",
                target="FalkorDB",
                relationship="USES",
                fact="Synapse uses FalkorDB",
            )
        ]
        pipe._stage2_deterministic = MagicMock(return_value=det)
        pipe._stage3_llm = MagicMock(
            return_value=ExtractionResult(entities=llm_entities, facts=facts)
        )

        seen: set[str] = set()

        def _fake_resolve(entities, group_id, deduper=None):
            for e in entities:
                seen.add(e.name)
            return {e.name: f"new:{e.name}" for e in entities}

        pipe._stage4_resolve = MagicMock(side_effect=_fake_resolve)
        pipe._stage5_write_nodes = MagicMock()
        pipe._process_facts_for_group = MagicMock()

        pipe.process_item(
            {
                "content_type": "summary",
                "content": "irrelevant",
                "project": "synapse",
                "session_id": "s1",
            }
        )

        assert "noise/path.py" not in seen
        assert {"FalkorDB", "Synapse"} <= seen

    def test_episode_with_no_facts_resolves_nothing(self):
        # Episodes produce no LLM facts -> referenced set empty -> Stage 4 is
        # never entered (previously it resolved every det entity then dropped
        # them all as orphans).
        pipe = self._pipeline()
        pipe._stage2_deterministic = MagicMock(
            return_value=[_entity("a/b.py"), _entity("https://x.test")]
        )
        pipe._stage4_resolve = MagicMock(return_value={})
        pipe._stage5_write_nodes = MagicMock()

        pipe.process_item(
            {
                "content_type": "episode",
                "content": "ran a/b.py against https://x.test",
                "project": "synapse",
                "metadata": "{}",
            }
        )

        pipe._stage4_resolve.assert_not_called()
        pipe._stage5_write_nodes.assert_not_called()


# ---------------------------------------------------------------------------
# Stage 6a candidate-pool sizing (token-bloat reduction)
# ---------------------------------------------------------------------------


class TestStage6aCandidateLimit:
    """The stage-6a hybrid search must cap the LLM-visible candidate pool at
    ``_SEMANTIC_POOL_LIMIT`` and pull 2x that from each source modality. This
    pins the token-bloat fix: each candidate is a full fact line in the
    stage-6b prompt, so reverting to a hardcoded 20 would re-inflate the
    dedup/contradiction call (~30K input tokens/call)."""

    def test_uses_semantic_pool_limit(self):
        # _stage6a_embedding_filter only reads self._embedder + self._kg,
        # so a bare instance is sufficient (skip the heavy __init__).
        pipe = ExtractionPipeline.__new__(ExtractionPipeline)
        pipe._embedder = MagicMock()
        pipe._embedder.embed.return_value = [[1.0, 0.0, 0.0, 0.0]]

        falkordb = MagicMock()
        falkordb.find_similar_edges.return_value = []
        falkordb.find_edges_by_fulltext.return_value = []
        falkordb.find_edges_by_pair.return_value = []
        pipe._kg = falkordb

        fact = ExtractedFact(source="A", target="B", relationship="USES", fact="A uses B")

        with patch("ingestion.extractor.rrf_merge", return_value=[]) as mock_rrf:
            pipe._stage6a_embedding_filter([fact], {"A": "ua", "B": "ub"}, "technical")

        assert falkordb.find_similar_edges.call_args.kwargs["limit"] == _SEMANTIC_POOL_LIMIT * 2
        assert falkordb.find_edges_by_fulltext.call_args.kwargs["limit"] == _SEMANTIC_POOL_LIMIT * 2
        assert mock_rrf.call_args.kwargs["limit"] == _SEMANTIC_POOL_LIMIT
        # Guard the intent: the cap is small (well under the old 20).
        assert _SEMANTIC_POOL_LIMIT < 20


class TestCanonicalAliases:
    """Task #49: identity aliases rewrite to the canonical hub pre-resolution.

    The hub name is deployment config (SYNAPSE_OWNER_NAME, issue #41) — the
    default install canonicalizes onto a neutral 'User' hub; a configured
    owner name and extra alias spellings resolve there instead.
    """

    def _ents(self, *names):
        return [ExtractedEntity(name=n, type="Person", summary=f"summary of {n}") for n in names]

    def test_alias_entity_renamed(self):
        from ingestion.extractor import _apply_canonical_aliases

        out = _apply_canonical_aliases(self._ents("the user"), [])
        assert [e.name for e in out] == ["User"]

    def test_alias_collapses_into_existing_canonical(self):
        from ingestion.extractor import _apply_canonical_aliases

        ents = [
            ExtractedEntity(name="User", type="Person", summary="short"),
            ExtractedEntity(
                name="The User", type="Person", summary="much longer summary wins here"
            ),
        ]
        out = _apply_canonical_aliases(ents, [])
        assert len(out) == 1
        assert out[0].name == "User"
        assert out[0].summary == "much longer summary wins here"

    def test_fact_endpoints_repointed(self):
        from ingestion.extractor import _apply_canonical_aliases

        facts = [
            ExtractedFact(
                source="the user", target="Synapse", relationship="USES", fact="User uses Synapse"
            )
        ]
        _apply_canonical_aliases(self._ents("the user", "Synapse"), facts)
        assert facts[0].source == "User"
        assert facts[0].target == "Synapse"

    def test_self_loop_dropped_after_rewrite(self):
        from ingestion.extractor import _apply_canonical_aliases

        facts = [
            ExtractedFact(
                source="User",
                target="The User",
                relationship="IS",
                fact="User is The User",
            ),
            ExtractedFact(
                source="User", target="Axon", relationship="BUILDS", fact="User builds Axon"
            ),
        ]
        _apply_canonical_aliases(self._ents("User", "The User", "Axon"), facts)
        assert len(facts) == 1
        assert facts[0].target == "Axon"

    def test_configured_owner_and_full_name_alias(self, monkeypatch):
        """SYNAPSE_OWNER_NAME + SYNAPSE_OWNER_ALIASES: a deployment's full-name
        spelling resolves onto its configured hub, not a hardcoded one."""
        from ingestion import dedup
        from ingestion.extractor import _apply_canonical_aliases

        monkeypatch.setenv("SYNAPSE_OWNER_NAME", "Sam")
        monkeypatch.setenv("SYNAPSE_OWNER_ALIASES", "Sam Smith, sammy")
        monkeypatch.setattr(dedup, "CANONICAL_ALIASES", dedup._build_canonical_aliases())
        out = _apply_canonical_aliases(self._ents("User", "Sam Smith", "Sammy"), [])
        assert [e.name for e in out] == ["Sam"]

    def test_assistant_cluster_untouched(self):
        from ingestion.extractor import _apply_canonical_aliases

        ents = self._ents("Claude", "neuron", "Neuron")
        out = _apply_canonical_aliases(ents, [])
        assert sorted(e.name for e in out) == ["Claude", "Neuron", "neuron"]

    def test_normalization_catches_case_and_whitespace(self):
        from ingestion.extractor import _apply_canonical_aliases

        out = _apply_canonical_aliases(self._ents("  THE USER "), [])
        assert [e.name for e in out] == ["User"]

    def test_non_alias_names_pass_through(self):
        from ingestion.extractor import _apply_canonical_aliases

        ents = self._ents("FalkorDB", "Postgres")
        out = _apply_canonical_aliases(ents, [])
        assert {e.name for e in out} == {"FalkorDB", "Postgres"}


# ---------------------------------------------------------------------------
# Per-worker NodeDeduper cache (LSH rebuilt per item regression)
# ---------------------------------------------------------------------------


class TestDeduperCache:
    """_deduper_for must reuse dedupers across items — the LSH build is O(all
    entities) and rebuilding it per process_item call dominated item latency."""

    def _pipeline(self) -> ExtractionPipeline:
        embedder = MagicMock()
        embedder.embed.side_effect = lambda names, task=None: [[0.0, 0.0, 0.0, 0.0] for _ in names]
        return ExtractionPipeline(
            db=MagicMock(),
            llm_client=MagicMock(),
            embedder=embedder,
            kg_client=_mock_falkordb(similar_nodes=[]),
        )

    def test_same_group_reuses_instance(self):
        pipe = self._pipeline()
        assert pipe._deduper_for("technical") is pipe._deduper_for("technical")

    def test_groups_get_distinct_dedupers(self):
        pipe = self._pipeline()
        assert pipe._deduper_for("technical") is not pipe._deduper_for("personal")

    def test_ttl_expiry_rebuilds(self, monkeypatch):
        monkeypatch.setenv("SYNAPSE_DEDUP_CACHE_TTL_SECONDS", "300")
        pipe = self._pipeline()
        first = pipe._deduper_for("technical")
        # Age the cache entry past the TTL; next call must build fresh.
        pipe._dedupers_built_at["technical"] -= 301
        assert pipe._deduper_for("technical") is not first
