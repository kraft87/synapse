"""Same-batch twin collapse (ingestion/extraction_twins.py).

Pure logic: no DB, no LLM. Embeddings are hand-built so the cosine floor is either
cleared (identical vectors) or not (orthogonal ones). What is pinned:

* A restatement in one batch is written once, including the common fan-out shape where
  each copy links a different entity pair.
* Near-identical facts that differ in a number, an identifier, a name or one swapped
  word are never merged, however close their embeddings.
* The most specific copy survives, the outcome does not depend on input order, and the
  survivor keeps every twin's user attribution. Stage 7 writes it with the batch's
  episodes, and the facts are embedded once per batch.
"""

from __future__ import annotations

import itertools
from unittest.mock import MagicMock

import pytest

from ingestion.extraction_twins import (
    _TWIN_MIN_SIM,
    collapse_batch_twins,
    fact_tokens,
    lexical_twins,
)
from ingestion.extractor import ExtractionPipeline
from ingestion.models import ExtractedFact

SEG = "2026-09-01T12:00:00+00:00"
NEAR = [1.0, 0.0, 0.0, 0.0]
FAR = [0.0, 1.0, 0.0, 0.0]


def _fact(text: str, source: str = "S", target: str = "T", **over) -> ExtractedFact:
    return ExtractedFact(source=source, target=target, relationship="R", fact=text, **over)


def _collapse(*facts: ExtractedFact, embeddings: list[list[float]] | None = None):
    embs = embeddings or [NEAR] * len(facts)
    return collapse_batch_twins(list(facts), embs)


def _texts(facts: list[ExtractedFact]) -> list[str]:
    return [f.fact for f in facts]


def _twins(a: str, b: str) -> bool:
    return lexical_twins(fact_tokens(a), fact_tokens(b))


class TestTwinsMerged:
    def test_identical_text_on_different_endpoints_collapses(self):
        # The fan-out shape: one sentence attached to each entity it names.
        text = "The vendor's two scanners were both flagged by the audit"
        out, embs, dropped = _collapse(
            _fact(text, "Vendor", "Scanner A"), _fact(text, "Vendor", "Scanner B")
        )
        assert dropped == 1
        assert _texts(out) == [text]
        assert len(embs) == 1

    def test_reordered_list_collapses(self):
        out, _, dropped = _collapse(
            _fact("Alice and Bob reviewed the release notes", "Alice", "Release notes"),
            _fact("Bob and Alice reviewed the release notes", "Bob", "Release notes"),
        )
        assert dropped == 1
        assert len(out) == 1

    def test_added_detail_keeps_the_more_specific_copy(self):
        out, _, dropped = _collapse(
            _fact("Build agents run on alpha", "Build agents", "alpha"),
            _fact("Build agents run on alpha, beta and gamma", "Build agents", "beta"),
        )
        assert dropped == 1
        assert _texts(out) == ["Build agents run on alpha, beta and gamma"]

    def test_inflection_and_function_words_do_not_block(self):
        assert _twins(
            "The cache relaxes the lock, worsening tail latency",
            "The cache worsens tail latency by relaxing the lock",
        )

    def test_case_punctuation_and_thousands_separators_normalised(self):
        assert _twins(
            "The job exported 3,291 rows to Sheet1.", "the job exported 3291 rows to SHEET1"
        )


class TestDifferentFactsKept:
    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("PR #221 was merged into main", "PR #223 was merged into main"),
            ("The service caps uploads at 10MB", "The service caps uploads at 20MB"),
            ("The patient takes 50 mg daily", "The patient takes 75 mg daily"),
            ("The job ran 3 times on 2026-01-05", "The job ran 3 times on 2026-01-06"),
            ("Release v1.2.3 fixed the crash", "Release v1.2.4 fixed the crash"),
        ],
    )
    def test_digit_differences(self, a, b):
        out, _, dropped = _collapse(_fact(a), _fact(b))
        assert dropped == 0
        assert _texts(out) == [a, b]

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("Ticket ABC-12 blocks the launch", "Ticket ABC-14 blocks the launch"),
            ("The hook lives in src/hooks/pre.py", "The hook lives in src/hooks/post.py"),
            ("Docs are at https://example.com/a", "Docs are at https://example.com/b"),
            ("The query reads user_id from the row", "The query reads group_id from the row"),
            ("The query reads aad.user from the row", "The query reads aad.group from the row"),
            ("Disk sda reports wear", "Disk sdb-1 reports wear"),
            ("The tool installs via Homebrew", "The tool installs via MacPorts"),
            ("Carol approved the budget", "Dave approved the budget"),
        ],
    )
    def test_identifier_and_name_differences(self, a, b):
        _, _, dropped = _collapse(_fact(a), _fact(b))
        assert dropped == 0

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("The ointment contains camphor", "The ointment contains menthol"),
            ("Lower thresholds are more generous", "Higher thresholds are more restrictive"),
            ("The exporter writes issue comments", "The exporter writes issue descriptions"),
            ("The package installs via yay", "The package installs via paru"),
        ],
    )
    def test_one_word_substitution(self, a, b):
        _, _, dropped = _collapse(_fact(a), _fact(b))
        assert dropped == 0

    def test_negation_is_part_of_the_signature(self):
        assert not _twins(
            "Compression is enabled by default", "Compression is not enabled by default"
        )
        assert not _twins("The probe finished", "The probe didn't finish")

    def test_cosine_floor(self):
        text = "Alice and Bob reviewed the release notes"
        _, _, dropped = _collapse(_fact(text, "Alice"), _fact(text, "Bob"), embeddings=[NEAR, FAR])
        assert dropped == 0
        just_under = [_TWIN_MIN_SIM - 0.01, (1 - (_TWIN_MIN_SIM - 0.01) ** 2) ** 0.5, 0.0, 0.0]
        _, _, dropped = _collapse(
            _fact(text, "Alice"), _fact(text, "Bob"), embeddings=[NEAR, just_under]
        )
        assert dropped == 0

    def test_two_different_elaborations_both_survive(self):
        # Each restates "X uses Y" but they add different details: only the bare one goes.
        out, _, dropped = _collapse(
            _fact("The importer uses the queue"),
            _fact("The importer uses the queue for retries"),
            _fact("The importer uses the queue for backfills"),
        )
        assert dropped == 1
        assert sorted(_texts(out)) == [
            "The importer uses the queue for backfills",
            "The importer uses the queue for retries",
        ]

    def test_same_endpoints_but_different_claim_kept(self):
        _, _, dropped = _collapse(
            _fact("The exporter writes issue comments", "Exporter", "Tracker"),
            _fact("The exporter writes issue attachments", "Exporter", "Tracker"),
        )
        assert dropped == 0


class TestSurvivor:
    def test_user_attribution_and_ongoing_carry_over(self):
        out, _, _ = _collapse(
            _fact("Alice and Bob reviewed the release notes", "Alice", attribution="assistant"),
            _fact(
                "Bob and Alice reviewed the release notes", "Bob", attribution="user", ongoing=True
            ),
        )
        assert out[0].attribution == "user"
        assert out[0].ongoing is True

    def test_input_order_never_changes_the_outcome(self):
        facts = [
            _fact("Build agents run on alpha", "Agents", "alpha"),
            _fact("Build agents run on alpha and beta", "Agents", "beta"),
            _fact("Build agents run on beta and alpha", "Agents", "alpha"),
            _fact("PR #7 was merged into main", "PR #7", "main"),
        ]
        outcomes = set()
        for perm in itertools.permutations(facts):
            out, _, dropped = collapse_batch_twins(list(perm), [NEAR] * len(perm))
            assert dropped == 2
            outcomes.add(tuple(sorted((f.fact, f.source, f.target) for f in out)))
        assert outcomes == {
            (
                ("Build agents run on alpha and beta", "Agents", "beta"),
                ("PR #7 was merged into main", "PR #7", "main"),
            )
        }

    def test_survivors_keep_input_order_and_their_own_embeddings(self):
        e0, e1, e2 = [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.99, 0.1, 0.0, 0.0]
        facts = [
            _fact("Disk sda reports wear"),
            _fact("PR #7 was merged into main"),
            _fact("Disk sda reports wear", target="T2"),
        ]
        out, embs, dropped = collapse_batch_twins(facts, [e0, e1, e2])
        assert dropped == 1
        assert _texts(out) == ["Disk sda reports wear", "PR #7 was merged into main"]
        assert embs == [e0, e1]

    def test_nothing_to_collapse_returns_inputs(self):
        facts = [_fact("PR #7 was merged into main")]
        assert collapse_batch_twins(facts, [NEAR]) == (facts, [NEAR], 0)


class TestPipelineWiring:
    """_process_facts_for_group: one embedding call, twins never reach Stage 6 or 7."""

    def _pipe(self):
        pipe = ExtractionPipeline.__new__(ExtractionPipeline)
        pipe._embedder = MagicMock()
        pipe._embedder.embed.side_effect = lambda texts, task=None: [NEAR for _ in texts]
        pipe._kg = MagicMock()
        pipe._kg.find_edges_by_pair.return_value = []
        pipe._kg.find_similar_edges.return_value = []
        pipe._kg.find_edges_by_fulltext.return_value = []
        pipe._kg.nearest_live_edges.side_effect = lambda embs, *a, **k: [[] for _ in embs]
        pipe._edge_date_extractor = MagicMock()
        pipe._edge_date_extractor.extract_batch.side_effect = lambda facts, reference_time=None: [
            (None, None) for _ in facts
        ]
        pipe._contradiction_detector = MagicMock()
        pipe._contradiction_detector.detect_contradictions_batch.side_effect = (
            lambda facts, *a, **k: [[] for _ in facts]
        )
        return pipe

    def test_twin_written_once_with_batch_provenance(self):
        pipe = self._pipe()
        facts = [
            _fact("Alice and Bob reviewed the release notes", "Alice", "Notes"),
            _fact("Bob and Alice reviewed the release notes", "Bob", "Notes", attribution="user"),
            _fact("PR #7 was merged into main", "PR #7", "main"),
        ]
        uuid_map = {n: f"u-{n}" for f in facts for n in (f.source, f.target)}
        pipe._process_facts_for_group(facts, uuid_map, [41, 42], "technical", default_valid_at=SEG)

        pipe._embedder.embed.assert_called_once()  # Stage 6a reuses the batch embeddings
        rows = pipe._kg.create_edges_batch.call_args.args[0]
        assert sorted(r["fact"] for r in rows) == [
            "Alice and Bob reviewed the release notes",
            "PR #7 was merged into main",
        ]
        twin = next(r for r in rows if r["fact"].startswith("Alice"))
        assert twin["episodes"] == [41, 42]
        # the absorbed copy was user-stated, so the survivor counts as user support
        assert twin["last_supported_at"] == SEG
        assert twin["last_supported_by"] == 42
        # Stage 6a searched once per surviving fact, never for the dropped twin
        assert pipe._kg.find_similar_edges.call_count == 2
