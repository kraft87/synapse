"""Two extractor defects seen on prod, reproduced end to end through process_item.

Real test DB and the real Stage 4-7 code; the LLM stages (extraction, the 6b judge, edge
dates, the writer-side contradiction check) are stubbed and embeddings are one-hot, so
texts that share a slot are identical vectors.

1. Restatement race. Overlapping chunk windows share a turn, so two batches extract the
   same claim from it. When a session is enqueued whole, the poller runs its adjacent
   windows on parallel worker threads: both pool and judge before either writes, and the
   claim landed twice. Run one after the other, it landed twice whenever the judge missed
   the pair. Pinned: one live edge, mention_count 2, the union of both batches' episodes.
2. Edge-less entities. Stage 5 wrote every new entity a fact named before Stage 6/7 decided
   which facts get written, so an entity whose only fact was then dropped (6b duplicate,
   6c restatement, same-batch twin, cross-group) was left with no edge. And two workers
   that resolved the same new name both inserted it.

Only symbols that predate the fix are imported, so the file also runs against the
pre-fix pipeline (where every reproduction fails on its assertion).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest

import ingestion.extractor as extractor_mod
from ingestion.dedup import _index_cache_reset
from ingestion.extractor import ExtractionPipeline
from ingestion.kg_client import KGClient
from ingestion.models import ExtractedEntity, ExtractedFact, ExtractionResult
from tests.helpers.embed import onehot

SEG = "2026-09-01T12:00:00+00:00"
ITEM = {"content_type": "chunk", "content": "window", "project": None, "session_id": "s1"}
WINDOW_A = [1, 2, 3, 4]
WINDOW_B = [4, 5, 6, 7]  # overlaps WINDOW_A on turn 4


class _Embedder:
    """One-hot embeddings. Texts listed in ``slots`` share that slot's vector; any other
    text (entity names, unrelated facts) gets its own slot."""

    def __init__(self, slots: dict[str, int] | None = None) -> None:
        self._slots = dict(slots or {})
        self._lock = threading.Lock()

    def embed(self, texts: list[str], task: str | None = None) -> list[list[float]]:
        with self._lock:
            return [onehot(self._slots.setdefault(t, 100 + len(self._slots))) for t in texts]


def _judge_misses(facts, candidates_map):
    return set(), {}, {}, True


def _judge_calls_every_candidate_a_duplicate(facts, candidates_map):
    skip, reinforce = set(), {}
    for idx, (pair_pool, semantic_pool) in candidates_map.items():
        pool = pair_pool + semantic_pool
        if pool:
            skip.add(idx)
            reinforce[idx] = [pool[0]["uuid"]]
    return skip, {}, reinforce, True


def _fact(text: str, source: str, target: str) -> ExtractedFact:
    return ExtractedFact(source=source, target=target, relationship="R", fact=text)


def _pipeline(
    embedder: _Embedder,
    facts: list[ExtractedFact],
    episodes: list[int],
    judge: Callable[..., Any] = _judge_misses,
) -> ExtractionPipeline:
    db = MagicMock()
    db.get_chunk_episode_ids.return_value = episodes
    db.get_episodes_valid_at.return_value = SEG
    pipe = ExtractionPipeline(
        db=db, llm_client=MagicMock(), embedder=embedder, kg_client=KGClient()
    )
    names = sorted({n for f in facts for n in (f.source, f.target)})
    entities = [ExtractedEntity(name=n, type="Concept", summary=f"{n} summary") for n in names]
    pipe._stage2_deterministic = MagicMock(return_value=[])  # type: ignore[method-assign]
    pipe._stage3_llm = MagicMock(  # type: ignore[method-assign]
        return_value=ExtractionResult(entities=entities, facts=facts)
    )
    pipe._stage6b_batch_confirm = MagicMock(side_effect=judge)  # type: ignore[method-assign]
    pipe._edge_date_extractor = MagicMock()
    pipe._edge_date_extractor.extract_batch.side_effect = lambda fs, reference_time=None: [
        (None, None) for _ in fs
    ]
    pipe._contradiction_detector = MagicMock()
    pipe._contradiction_detector.detect_contradictions_batch.side_effect = lambda fs, *a, **k: [
        [] for _ in fs
    ]
    return pipe


def _run_as_parallel_workers(
    pipe_a: ExtractionPipeline,
    pipe_b: ExtractionPipeline,
    hold: str,
    before_b: Callable[[], None] | None = None,
) -> None:
    """Run both items on their own threads, interleaved as two worker threads that
    claimed them together: neither gets past ``hold`` (a pipeline method, or a dotted path
    to one of its collaborators') until both have run it, then A runs to completion before
    B goes on. So B has done everything up to and including ``hold`` against the graph as
    it stood before any of A's writes."""
    barrier = threading.Barrier(2, timeout=30)
    a_done = threading.Event()
    errors: list[BaseException] = []
    owner, _, name = hold.rpartition(".")

    def gate(pipe: ExtractionPipeline, is_b: bool) -> None:
        target = getattr(pipe, owner) if owner else pipe
        inner = getattr(target, name)

        def held(*args: Any, **kwargs: Any) -> Any:
            out = inner(*args, **kwargs)
            barrier.wait()
            if is_b:
                assert a_done.wait(30), "worker A never finished"
                if before_b is not None:
                    before_b()
            return out

        setattr(target, name, held)

    def work(pipe: ExtractionPipeline, done: threading.Event | None) -> None:
        try:
            pipe.process_item(dict(ITEM))
        except BaseException as exc:
            errors.append(exc)
            barrier.abort()
        finally:
            if done is not None:
                done.set()

    gate(pipe_a, is_b=False)
    gate(pipe_b, is_b=True)
    threads = [
        threading.Thread(target=work, args=(pipe_a, a_done)),
        threading.Thread(target=work, args=(pipe_b, None)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    if errors:
        raise errors[0]


@pytest.fixture()
def graph(conn, monkeypatch, db_url):
    monkeypatch.setenv("SYNAPSE_DB_URL", db_url)
    conn.execute("TRUNCATE kg_entities, kg_relationships RESTART IDENTITY CASCADE")
    _index_cache_reset()  # the dedup index is process-wide; start each case empty
    yield conn
    _index_cache_reset()


def _live_edges(conn) -> list[tuple[str, str, int, list[int]]]:
    return conn.execute(
        "SELECT fact, src_uuid, mention_count, episodes FROM kg_relationships "
        "WHERE t_invalid IS NULL ORDER BY id"
    ).fetchall()


def _entity_names(conn) -> list[str]:
    return [r[0] for r in conn.execute("SELECT name FROM kg_entities ORDER BY name").fetchall()]


def _edgeless(conn) -> list[str]:
    return [
        r[0]
        for r in conn.execute(
            "SELECT name FROM kg_entities e WHERE NOT EXISTS ("
            "  SELECT 1 FROM kg_relationships r WHERE r.src_uuid = e.uuid OR r.tgt_uuid = e.uuid"
            ") ORDER BY name"
        ).fetchall()
    ]


# ---------------------------------------------------------------------------------------
# 1. Restatement race between overlapping windows
# ---------------------------------------------------------------------------------------

_SAME = (
    _fact("Orchard publishes the release calendar", "Orchard", "Calendar"),
    _fact("Orchard publishes the release calendar", "Orchard", "Calendar"),
)
# Fan-out twin: the shared turn's sentence, list reordered and hung on the other entity.
_TWIN = (
    _fact("Orchard and Lantern share the release calendar", "Orchard", "Calendar"),
    _fact("Lantern and Orchard share the release calendar", "Lantern", "Calendar"),
)


def _assert_one_reinforced_edge(conn, text: str) -> None:
    edges = _live_edges(conn)
    assert [e[0] for e in edges] == [text], "the shared claim must land once"
    _, _, mention_count, episodes = edges[0]
    assert mention_count == 2
    assert sorted(episodes) == [1, 2, 3, 4, 5, 6, 7]


@pytest.mark.parametrize("pair", [_SAME, _TWIN], ids=["identical", "twin"])
class TestRestatementRace:
    @pytest.mark.parametrize(
        "hold",
        [
            "_stage6b_batch_confirm",
            # Stage 7's own LLM pre-pass: B waited on its edge-date call while A wrote.
            "_edge_date_extractor.extract_batch",
        ],
        ids=["after-6b", "after-stage7-llm-prepass"],
    )
    def test_parallel_workers_both_judged_before_either_wrote(self, graph, pair, hold):
        fact_a, fact_b = pair
        emb = _Embedder({fact_a.fact: 0, fact_b.fact: 0})
        pipe_a = _pipeline(emb, [fact_a], WINDOW_A)
        pipe_b = _pipeline(emb, [fact_b], WINDOW_B)

        _run_as_parallel_workers(pipe_a, pipe_b, hold=hold)

        _assert_one_reinforced_edge(graph, fact_a.fact)
        assert _edgeless(graph) == []

    def test_sequential_batches_with_a_judge_that_misses(self, graph, pair):
        fact_a, fact_b = pair
        emb = _Embedder({fact_a.fact: 0, fact_b.fact: 0})
        _pipeline(emb, [fact_a], WINDOW_A).process_item(dict(ITEM))
        pipe_b = _pipeline(emb, [fact_b], WINDOW_B)
        pipe_b.process_item(dict(ITEM))

        assert pipe_b._stage6b_batch_confirm.call_args.args[1], "6a pooled A's edge for B"
        _assert_one_reinforced_edge(graph, fact_a.fact)
        assert _edgeless(graph) == []


# ---------------------------------------------------------------------------------------
# 2. Entities whose only fact is not written
# ---------------------------------------------------------------------------------------

_CONTROL = _fact("Meadow schedules the nightly backup", "Meadow", "Backup")


class TestNoEntityWithoutAWrittenFact:
    def test_fact_skipped_as_a_6b_duplicate(self, graph):
        seed = _fact("Orchard publishes the release calendar", "Orchard", "Calendar")
        # A paraphrase on a new entity: no twin of the seed, but the judge calls it one.
        para = _fact("Quartz keeps a copy of the release calendar", "Quartz", "Calendar")
        emb = _Embedder({seed.fact: 0, para.fact: 0})
        _pipeline(emb, [seed], WINDOW_A).process_item(dict(ITEM))

        _pipeline(
            emb, [para, _CONTROL], WINDOW_B, judge=_judge_calls_every_candidate_a_duplicate
        ).process_item(dict(ITEM))

        assert "Quartz" not in _entity_names(graph)
        assert {"Meadow", "Backup"} <= set(_entity_names(graph))  # written fact's entities
        assert [e[0] for e in _live_edges(graph)] == [seed.fact, _CONTROL.fact]
        assert _edgeless(graph) == []

    def test_fact_skipped_as_a_6c_restatement(self, graph):
        seed, twin = _TWIN
        emb = _Embedder({seed.fact: 0, twin.fact: 0})
        _pipeline(emb, [seed], WINDOW_A).process_item(dict(ITEM))

        _pipeline(emb, [twin, _CONTROL], WINDOW_B).process_item(dict(ITEM))

        # Lantern's only fact restated the seed; Orchard and Calendar came with the seed.
        assert "Lantern" not in _entity_names(graph)
        assert {"Meadow", "Backup"} <= set(_entity_names(graph))
        assert _edgeless(graph) == []

    def test_fact_collapsed_as_a_same_batch_twin(self, graph):
        kept, dropped = (
            _fact("Lantern and Orchard share the release calendar", "Lantern", "Calendar"),
            _fact("Orchard and Lantern share the release calendar", "Orchard", "Calendar"),
        )
        emb = _Embedder({kept.fact: 0, dropped.fact: 0})

        _pipeline(emb, [dropped, kept, _CONTROL], WINDOW_A).process_item(dict(ITEM))

        assert [e[0] for e in _live_edges(graph)] == [kept.fact, _CONTROL.fact]
        assert "Orchard" not in _entity_names(graph)  # named only by the dropped twin
        assert {"Lantern", "Calendar", "Meadow", "Backup"} <= set(_entity_names(graph))
        assert _edgeless(graph) == []

    def test_cross_group_fact(self, graph, monkeypatch):
        monkeypatch.setenv("SYNAPSE_PERSONAL_SCOPE", "1")
        monkeypatch.setattr(
            extractor_mod,
            "_classify_entity_group",
            lambda name, summary, default: "personal" if name == "Pebble" else default,
        )
        cross = _fact("Pebble is moored at the harbor", "Pebble", "Harbor")

        _pipeline(_Embedder(), [cross, _CONTROL], WINDOW_A).process_item(dict(ITEM))

        assert [e[0] for e in _live_edges(graph)] == [_CONTROL.fact]
        assert _entity_names(graph) == ["Backup", "Meadow"]
        assert _edgeless(graph) == []


# ---------------------------------------------------------------------------------------
# Same new name resolved by two workers at once
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "before_b",
    [None, _index_cache_reset],
    ids=["same-process", "other-process"],  # other process: only the live DB knows it
)
def test_same_new_name_from_parallel_workers_is_one_entity(graph, before_b):
    fact_a = _fact("Harbor hosts the build cache", "Harbor", "Cache")
    fact_b = _fact("Harbor mirrors the package index", "Harbor", "Index")
    emb = _Embedder()
    pipe_a = _pipeline(emb, [fact_a], WINDOW_A)
    pipe_b = _pipeline(emb, [fact_b], WINDOW_B)

    # Both resolve "Harbor" (to a new uuid each) before either writes anything.
    _run_as_parallel_workers(pipe_a, pipe_b, hold="_stage4_resolve", before_b=before_b)

    harbors = graph.execute(
        "SELECT uuid FROM kg_entities WHERE normalized_name = 'harbor'"
    ).fetchall()
    assert len(harbors) == 1
    edges = _live_edges(graph)
    assert sorted(e[0] for e in edges) == sorted([fact_a.fact, fact_b.fact])
    assert {e[1] for e in edges} == {harbors[0][0]}
    assert _edgeless(graph) == []
