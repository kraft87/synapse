"""Stage 6c restatement recheck and the deferred new-entity write (no DB, no LLM).

Pinned:

* A fact about to be created whose nearest live edge says the same thing (identical after
  case/whitespace folding, or a twin under the same-batch rule) is skipped, and that edge
  is reinforced through Stage 7's ordinary duplicate path. Near misses (a changed number,
  a substituted word) are written.
* Facts already skipped, or missing an endpoint, are never looked up; one query per group.
* An edge this batch retires is never the match, and a restated fact that also retires
  something links the retirement to the edge it was absorbed into.
* A new entity is written only when a fact Stage 7 creates links it; a name another
  worker wrote meanwhile is reused, not inserted twice.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from ingestion.extraction_nodes import PendingNodes
from ingestion.extraction_recheck import (
    _RECHECK_MAX_DIST,
    _RECHECK_NEIGHBOURS,
    recheck_restated,
    restated_edge,
)
from ingestion.extraction_twins import _TWIN_MIN_SIM
from ingestion.extractor import ExtractionPipeline
from ingestion.models import ExtractedEntity, ExtractedFact, ExtractionResult

SEG = "2026-09-01T12:00:00+00:00"
EMB = [1.0, 0.0, 0.0, 0.0]


def _fact(text: str, source: str = "S", target: str = "T", **over) -> ExtractedFact:
    return ExtractedFact(source=source, target=target, relationship="R", fact=text, **over)


def _edge(uuid: str, fact: str, score: float = 0.01) -> dict:
    return {"uuid": uuid, "fact": fact, "score": score}


def _kg(*per_fact: list[dict]) -> MagicMock:
    kg = MagicMock()
    kg.nearest_live_edges.return_value = list(per_fact)
    return kg


class TestRestatedEdge:
    def test_identical_after_case_and_whitespace_folding(self):
        edges = [_edge("e1", "the  Build cache lives on   the NAS")]
        assert restated_edge("The build cache lives on the NAS", edges) == "e1"

    def test_twin_reordered_list(self):
        edges = [_edge("e1", "Alice and Bob reviewed the release notes")]
        assert restated_edge("Bob and Alice reviewed the release notes", edges) == "e1"

    def test_twin_with_detail_on_one_side_either_way(self):
        short, long_ = "Build agents run on alpha", "Build agents run on alpha, beta and gamma"
        assert restated_edge(short, [_edge("e1", long_)]) == "e1"
        assert restated_edge(long_, [_edge("e1", short)]) == "e1"

    def test_near_misses_are_not_restatements(self):
        assert (
            restated_edge(
                "PR #12 was merged into main", [_edge("e1", "PR #14 was merged into main")]
            )
            is None
        )
        assert (
            restated_edge("The tea contains camphor", [_edge("e1", "The tea contains menthol")])
            is None
        )
        assert restated_edge("The job does not retry", [_edge("e1", "The job does retry")]) is None

    def test_first_matching_neighbour_wins_nearest_first(self):
        edges = [
            _edge("e-near-other", "The build cache lives on the SAN", 0.01),
            _edge("e-twin", "The build cache lives on the NAS", 0.03),
            _edge("e-twin-2", "the build cache lives on the NAS", 0.05),
        ]
        assert restated_edge("The build cache lives on the NAS", edges) == "e-twin"

    def test_no_neighbours(self):
        assert restated_edge("anything", []) is None


class TestRecheckRestated:
    def test_lookup_bound_is_the_twin_cosine_floor(self):
        assert _RECHECK_MAX_DIST == 1.0 - _TWIN_MIN_SIM

    def test_restated_fact_moves_to_skip_and_reinforce(self):
        facts = [_fact("The build cache lives on the NAS"), _fact("PR #7 was merged into main")]
        kg = _kg([_edge("e1", "the build cache lives on the NAS")], [])
        skip: set[int] = set()
        reinforce: dict[int, list[str]] = {}

        moved = recheck_restated(
            facts, [EMB, EMB], {"S": "u-s", "T": "u-t"}, "g", kg, skip, {}, reinforce
        )

        assert moved == 1
        assert skip == {0}
        assert reinforce == {0: ["e1"]}
        kg.nearest_live_edges.assert_called_once_with(
            [EMB, EMB], "g", max_distance=_RECHECK_MAX_DIST, limit=_RECHECK_NEIGHBOURS
        )

    def test_skipped_and_unresolved_facts_are_not_looked_up(self):
        facts = [
            _fact("already a 6b duplicate"),
            _fact("source never resolved", source="Ghost"),
            _fact("to be created"),
        ]
        kg = _kg([])
        skip = {0}
        reinforce = {0: ["judge-dup"]}
        recheck_restated(
            facts,
            [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]],
            {"S": "new:u-s", "T": "u-t"},
            "g",
            kg,
            skip,
            {},
            reinforce,
        )
        assert kg.nearest_live_edges.call_args.args[0] == [[0.0, 3.0]]
        assert skip == {0}
        assert reinforce == {0: ["judge-dup"]}  # the judge's verdict is untouched

    def test_nothing_to_create_means_no_query(self):
        kg = _kg()
        assert recheck_restated([_fact("x")], [EMB], {}, "g", kg, set(), {}, {}) == 0
        kg.nearest_live_edges.assert_not_called()

    def test_an_edge_this_batch_retires_is_never_the_match(self):
        facts = [_fact("The build cache lives on the NAS"), _fact("The cache moved to the SAN")]
        neighbours = [
            _edge("e-retiring", "The build cache lives on the NAS", 0.01),
            _edge("e-live", "the build cache lives on the NAS", 0.02),
        ]
        kg = _kg(neighbours, [])
        skip: set[int] = set()
        reinforce: dict[int, list[str]] = {}
        recheck_restated(
            facts, [EMB, EMB], {"S": "u", "T": "v"}, "g", kg, skip, {1: ["e-retiring"]}, reinforce
        )
        assert reinforce == {0: ["e-live"]}

    def test_a_fact_retiring_its_only_near_copy_is_written(self):
        facts = [_fact("The build cache lives on the NAS")]
        kg = _kg([_edge("e-old", "The build cache lives on the NAS")])
        skip: set[int] = set()
        recheck_restated(facts, [EMB], {"S": "u", "T": "v"}, "g", kg, skip, {0: ["e-old"]}, {})
        assert skip == set()


def _pipe(nearest: list[list[dict]] | None = None, judge=None) -> ExtractionPipeline:
    pipe = ExtractionPipeline.__new__(ExtractionPipeline)
    pipe._embedder = MagicMock()
    pipe._embedder.embed.side_effect = lambda texts, task=None: [EMB for _ in texts]
    pipe._kg = MagicMock()
    pipe._kg.find_edges_by_pair.return_value = []
    pipe._kg.find_similar_edges.return_value = []
    pipe._kg.find_edges_by_fulltext.return_value = []
    pipe._kg.nearest_live_edges.side_effect = lambda embs, *a, **k: (
        nearest if nearest is not None else [[] for _ in embs]
    )
    pipe._stage6b_batch_confirm = MagicMock(  # type: ignore[method-assign]
        side_effect=judge or (lambda facts, cands: (set(), {}, {}, True))
    )
    pipe._edge_date_extractor = MagicMock()
    pipe._edge_date_extractor.extract_batch.side_effect = lambda facts, reference_time=None: [
        (None, None) for _ in facts
    ]
    pipe._contradiction_detector = MagicMock()
    pipe._contradiction_detector.detect_contradictions_batch.side_effect = lambda facts, *a, **k: [
        [] for _ in facts
    ]
    return pipe


def _created(pipe) -> list[dict]:
    calls = pipe._kg.create_edges_batch.call_args_list
    return calls[0].args[0] if calls else []


class TestStage7Wiring:
    def test_restated_fact_reinforces_instead_of_creating(self):
        facts = [
            _fact("The build cache lives on the NAS", "Cache", "NAS", attribution="user"),
            _fact("PR #7 was merged into main", "PR #7", "main"),
        ]
        uuid_map = {n: f"u-{n}" for f in facts for n in (f.source, f.target)}
        pipe = _pipe(nearest=[[_edge("e-live", "The build cache lives on the NAS")], []])

        pipe._process_facts_for_group(facts, uuid_map, [4, 5], "g", default_valid_at=SEG)

        assert [r["fact"] for r in _created(pipe)] == ["PR #7 was merged into main"]
        pipe._kg.reinforce_edges.assert_called_once_with([("e-live", [4, 5], SEG, 5)], "g")

    def test_restated_fact_that_retires_an_edge_links_it_to_the_absorbing_edge(self):
        facts = [_fact("The build cache lives on the NAS", "Cache", "NAS")]
        uuid_map = {"Cache": "u-c", "NAS": "u-n"}
        pipe = _pipe(
            nearest=[[_edge("e-live", "The build cache lives on the NAS")]],
            judge=lambda fs, cands: (set(), {0: ["e-old-san"]}, {}, True),
        )

        pipe._process_facts_for_group(facts, uuid_map, [4], "g", default_valid_at=SEG)

        assert _created(pipe) == []
        pipe._kg.invalidate_edges_batch.assert_called_once_with(
            [("e-old-san", SEG)], "g", invalidated_by="e-live"
        )

    def test_writer_side_verdict_of_a_restated_fact_still_retires_linked(self):
        facts = [_fact("The build cache lives on the NAS", "Cache", "NAS")]
        uuid_map = {"Cache": "u-c", "NAS": "u-n"}
        pipe = _pipe(nearest=[[_edge("e-live", "The build cache lives on the NAS")]])
        pipe._contradiction_detector.detect_contradictions_batch.side_effect = lambda fs, *a, **k: [
            ["e-old-san"]
        ]

        pipe._process_facts_for_group(facts, uuid_map, [4], "g", default_valid_at=SEG)

        assert _created(pipe) == []
        pipe._kg.invalidate_edges_batch.assert_called_once_with(
            [("e-old-san", SEG)], "g", invalidated_by="e-live"
        )

    def test_writer_side_verdict_against_the_near_copy_writes_a_replacement(self):
        facts = [_fact("The build cache lives on the NAS", "Cache", "NAS")]
        uuid_map = {"Cache": "u-c", "NAS": "u-n"}
        pipe = _pipe(nearest=[[_edge("e-live", "The build cache lives on the NAS")]])
        pipe._contradiction_detector.detect_contradictions_batch.side_effect = lambda fs, *a, **k: [
            ["e-live"]
        ]

        pipe._process_facts_for_group(facts, uuid_map, [4], "g", default_valid_at=SEG)

        created = _created(pipe)
        assert [r["fact"] for r in created] == ["The build cache lives on the NAS"]
        pipe._kg.invalidate_edges_batch.assert_called_once_with(
            [("e-live", SEG)], "g", invalidated_by=created[0]["edge_uuid"]
        )


def _entity(name: str) -> ExtractedEntity:
    return ExtractedEntity(name=name, type="Concept", summary=f"{name} summary")


def _pending(*names: str, deduper=None) -> PendingNodes:
    return PendingNodes([_entity(n) for n in names], "proj", {}, deduper)


def _upserted(pipe) -> list[tuple[str, str]]:
    return [(c.kwargs["name"], c.kwargs["node_uuid"]) for c in pipe._kg.upsert_node.call_args_list]


class TestNewEntityDeferral:
    def test_only_entities_of_created_facts_are_written(self):
        facts = [
            _fact("The build cache lives on the NAS", "Cache", "NAS"),
            _fact("PR #7 was merged into main", "PR #7", "main"),
        ]
        uuid_map = {"Cache": "new:c", "NAS": "u-nas", "PR #7": "new:p", "main": "new:m"}
        pipe = _pipe(nearest=[[_edge("e-live", "The build cache lives on the NAS")], []])

        pipe._process_facts_for_group(
            facts,
            uuid_map,
            [4],
            "g",
            default_valid_at=SEG,
            new_nodes=_pending("Cache", "PR #7", "main"),
        )

        # Cache's only fact restated a live edge; NAS is an existing entity (not pending).
        assert sorted(_upserted(pipe)) == [("PR #7", "p"), ("main", "m")]

    def test_entity_of_a_fact_missing_its_other_endpoint_is_not_written(self):
        facts = [_fact("Ghost depends on the cache", "Ghost", "Cache")]
        pipe = _pipe()
        pipe._process_facts_for_group(
            facts, {"Cache": "new:c"}, [4], "g", default_valid_at=SEG, new_nodes=_pending("Cache")
        )
        assert _upserted(pipe) == []
        assert _created(pipe) == []

    def test_name_written_meanwhile_is_reused_and_the_edge_follows_it(self):
        facts = [_fact("The build cache lives on the NAS", "Cache", "NAS")]
        uuid_map = {"Cache": "new:c", "NAS": "new:n"}
        deduper = MagicMock()
        deduper.exact_match.side_effect = lambda name: (
            "e-cache-other-worker" if name == "Cache" else None
        )
        deduper.summary_of.return_value = "Cache summary from the other worker"
        deduper.type_map = {}
        pipe = _pipe()

        pipe._process_facts_for_group(
            facts,
            uuid_map,
            [4],
            "g",
            default_valid_at=SEG,
            new_nodes=_pending("Cache", "NAS", deduper=deduper),
        )

        assert sorted(_upserted(pipe)) == [("Cache", "e-cache-other-worker"), ("NAS", "n")]
        deduper.register.assert_called_once_with("NAS", "n", "NAS summary")
        assert uuid_map["Cache"] == "e-cache-other-worker"
        (row,) = _created(pipe)
        assert (row["src"], row["tgt"]) == ("e-cache-other-worker", "n")


class TestProcessItemSplit:
    def test_existing_entities_update_up_front_new_ones_wait_for_their_group(self):
        embedder = MagicMock()
        embedder.embed.side_effect = lambda names, task=None: [EMB for _ in names]
        db = MagicMock()
        db.get_session_episodes.return_value = []
        db.get_synth_document_source_ids.return_value = []
        pipe = ExtractionPipeline(
            db=db, llm_client=MagicMock(), embedder=embedder, kg_client=MagicMock()
        )
        facts = [_fact("Cache stores artifacts on the NAS", "Cache", "NAS")]
        pipe._stage2_deterministic = MagicMock(return_value=[])  # type: ignore[method-assign]
        pipe._stage3_llm = MagicMock(  # type: ignore[method-assign]
            return_value=ExtractionResult(entities=[_entity("Cache"), _entity("NAS")], facts=facts)
        )
        pipe._stage4_resolve = MagicMock(  # type: ignore[method-assign]
            return_value={"Cache": "new:c", "NAS": "u-nas"}
        )
        pipe._deduper_for = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
        pipe._stage5_write_nodes = MagicMock()  # type: ignore[method-assign]
        pipe._process_facts_for_group = MagicMock()  # type: ignore[method-assign]

        pipe.process_item(
            {"content_type": "summary", "content": "x", "project": None, "session_id": "s1"}
        )

        (call,) = pipe._stage5_write_nodes.call_args_list
        assert [e.name for e in call.args[0]] == ["NAS"]
        pending = pipe._process_facts_for_group.call_args.kwargs["new_nodes"]
        assert [e.name for e in pending.entities] == ["Cache"]
