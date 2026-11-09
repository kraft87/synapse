"""The fact-audience labeller against a real database (schema 066).

Covers the parts that write: which facts are picked and in what order, that a label is only
ever written where the fact is still unlabelled, the version and model stamps, the stop
conditions, and the poller's stage. The invariants: nothing is labelled before 066 or while
no restricted device could be served the facts, the personal pile is never touched, a backend
that fails leaves the rest NULL for the next cycle, and one fact that breaks calls on its own
neither blocks the queue nor stands in for a dead backend.
"""

from __future__ import annotations

import os
import uuid
from contextlib import nullcontext

import psycopg
import pytest
from psycopg.rows import dict_row

_DB_URL = os.environ.get(
    "SYNAPSE_TEST_URL", "postgresql://synapse:synapse@127.0.0.1:5432/synapse_test"
)

try:
    _probe = psycopg.connect(_DB_URL, connect_timeout=2)
    _probe.close()
except Exception:  # pragma: no cover - environment dependent
    pytest.skip("no test DB reachable", allow_module_level=True)

from ingestion import fact_audience as fa  # noqa: E402
from ingestion.db import Database  # noqa: E402
from ingestion.llm_client import LLMHTTPError, UsageLimitError  # noqa: E402
from ingestion.poller import Poller  # noqa: E402
from ingestion.surfaces import FACT_AUDIENCE_PROBE  # noqa: E402
from tests.helpers.embed import GROUP  # noqa: E402
from tests.helpers.surfaces import (  # noqa: E402
    clear_surfaces,
    register_device,
    register_full,
    register_restricted,
)

WORK = "work-proj"
ELSEWHERE = "other-proj"
PRIVATE = "sensitivity:private"
SAFE = "sensitivity:work-safe"


@pytest.fixture()
def clean(conn):
    def _wipe():
        conn.execute("TRUNCATE kg_relationships, kg_entities RESTART IDENTITY CASCADE")
        conn.execute("TRUNCATE episodes RESTART IDENTITY CASCADE")
        clear_surfaces(conn)

    _wipe()
    yield conn
    _wipe()


@pytest.fixture()
def restricted(clean):
    """A registered restricted device that reads WORK: the stage's precondition."""
    register_restricted(clean, [WORK])
    return clean


def _ep(conn, project: str = WORK) -> int:
    return conn.execute(
        "INSERT INTO episodes (session_id, sequence, project, content) "
        "VALUES (%s, 1, %s, 'a turn') RETURNING id",
        (f"fad-{uuid.uuid4().hex[:10]}", project),
    ).fetchone()[0]


def _fact(
    conn,
    text: str | None,
    *,
    ep: int | None = None,
    audience: str | None = None,
    web: int | None = None,
    retired: bool = False,
    group: str = GROUP,
) -> int:
    """One fact; its id is its insertion order. ``ep`` is its only source episode."""
    return conn.execute(
        "INSERT INTO kg_relationships (uuid, owner_id, group_id, src_uuid, tgt_uuid, name, fact, "
        "  episodes, t_valid, t_invalid, web_artifact_id, audience) "
        "VALUES (%s, 'default', %s, 'a', 'b', 'RELATES', %s, %s::jsonb, "
        "        '2026-01-01T00:00:00+00:00', %s, %s, %s) RETURNING id",
        (
            f"fad-{uuid.uuid4().hex}",
            group,
            text,
            None if ep is None else f"[{ep}]",
            "2026-02-01T00:00:00+00:00" if retired else None,
            web,
            audience,
        ),
    ).fetchone()[0]


def _label(conn, fid: int) -> tuple[str | None, int | None, str | None]:
    return conn.execute(
        "SELECT audience, label_version, label_model FROM kg_relationships WHERE id = %s", (fid,)
    ).fetchone()


def _audience(conn, fid: int) -> str | None:
    return _label(conn, fid)[0]


class _Judge:
    """Stands in for ``structured_call``: private when a fact says SECRET, else work-safe.
    Raises ``fail`` from the call after ``fail_after`` have succeeded. ``seen`` is every fact
    text it was shown, in order."""

    def __init__(self, fail: BaseException | None = None, fail_after: int = 0) -> None:
        self.calls = 0
        self.seen: list[str] = []
        self.fail, self.fail_after = fail, fail_after

    def __call__(self, llm, **kw):
        self.calls += 1
        if self.fail is not None and self.calls > self.fail_after:
            raise self.fail
        tail = kw["base_prompt"].split("Facts, one per line as <n>|<text>:\n", 1)[1]
        lines = tail.split("\n\nReply with ONLY", 1)[0].split("\n")
        self.seen += [ln.split("|", 1)[1] for ln in lines]
        return fa.FactAudienceReply(
            facts=[
                fa.FactLabel(n=i, labels=[PRIVATE if "SECRET" in ln else SAFE])
                for i, ln in enumerate(lines, start=1)
            ]
        )


@pytest.fixture()
def judge(monkeypatch):
    j = _Judge()
    monkeypatch.setattr(fa, "structured_call", j)
    return j


def _run(db_url: str, **kw):
    db = Database(db_url)
    try:
        return fa.label_pending(db._conn, object(), **kw)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# When the stage runs
# ---------------------------------------------------------------------------


def test_nothing_is_labelled_while_no_restricted_device_is_registered(clean, db_url, judge):
    fid = _fact(clean, "a fact", ep=_ep(clean))
    stats = _run(db_url)
    assert stats.skipped == "no restricted device can be served facts"
    assert judge.calls == 0 and _audience(clean, fid) is None


def test_a_full_trust_a_revoked_or_an_empty_allowlist_device_does_not_start_it(
    clean, db_url, judge
):
    fid = _fact(clean, "a fact", ep=_ep(clean))
    register_full(clean)
    register_restricted(clean, [], surface_id="no-projects")
    register_device(clean, "tok-revoked", trust="restricted", projects=[WORK], status="revoked")
    assert _run(db_url).skipped
    assert judge.calls == 0 and _audience(clean, fid) is None
    register_restricted(clean, [WORK])
    assert not _run(db_url).skipped
    assert _audience(clean, fid) == "work-safe"


def test_the_reachable_projects_are_the_union_over_approved_restricted_devices(clean):
    register_restricted(clean, [WORK, "b"], surface_id="one")
    register_restricted(clean, [WORK, "c"], surface_id="two")
    register_full(clean)
    with psycopg.connect(_DB_URL, row_factory=dict_row) as c:
        assert sorted(fa.restricted_projects(c)) == sorted([WORK, "b", "c"])


def test_nothing_is_labelled_before_066(restricted, db_url, judge):
    fid = _fact(restricted, "a fact", ep=_ep(restricted))
    tx = psycopg.connect(db_url, row_factory=dict_row)
    try:
        for col in ("audience", "label_version", "label_model"):
            tx.execute(f"ALTER TABLE kg_relationships DROP COLUMN {col} CASCADE")
        tx.execute("ALTER TABLE kg_entities DROP COLUMN is_self CASCADE")
        stats = fa.label_pending(lambda: nullcontext(tx), object())
        assert stats.skipped == "schema 066 not applied"
        assert FACT_AUDIENCE_PROBE.cached("") is False
    finally:
        tx.rollback()
        tx.close()
    assert judge.calls == 0 and _audience(restricted, fid) is None


# ---------------------------------------------------------------------------
# What it labels
# ---------------------------------------------------------------------------


def test_only_unlabelled_non_web_facts_with_text_are_labelled(restricted, db_url, judge):
    ep = _ep(restricted)
    plain = _fact(restricted, "a plain fact", ep=ep)
    private = _fact(restricted, "a SECRET fact", ep=ep)
    kept_safe = _fact(restricted, "already work-safe SECRET", ep=ep, audience="work-safe")
    kept_personal = _fact(restricted, "already personal", ep=ep, audience="personal")
    from_web = _fact(restricted, "a web page fact", ep=ep, web=7)
    empty = _fact(restricted, "", ep=ep)
    no_text = _fact(restricted, None, ep=ep)
    personal_group = _fact(restricted, "a personal-group fact", ep=ep, group="personal")
    stats = _run(db_url)
    assert _audience(restricted, plain) == "work-safe"
    assert _audience(restricted, private) == "personal"
    assert _audience(restricted, kept_safe) == "work-safe", "never overwritten"
    assert _audience(restricted, kept_personal) == "personal"
    for fid in (from_web, empty, no_text):
        assert _audience(restricted, fid) is None, fid
    # stamped by the schema's trigger, so the stage never saw it
    assert _label(restricted, personal_group)[:2] == ("personal", 0)
    assert judge.seen == ["a SECRET fact", "a plain fact"]
    assert stats.labelled == {"work-safe": 1, "personal": 1}


def test_each_label_carries_the_version_and_the_model(restricted, db_url, judge, monkeypatch):
    monkeypatch.setenv("SYNAPSE_FACT_AUDIENCE_MODEL", "stage-model")
    ep = _ep(restricted)
    safe, private = _fact(restricted, "plain", ep=ep), _fact(restricted, "SECRET", ep=ep)
    _run(db_url)
    assert _label(restricted, safe) == ("work-safe", fa.LABEL_VERSION, "stage-model")
    assert _label(restricted, private) == ("personal", fa.LABEL_VERSION, "stage-model")
    later = _fact(restricted, "plain again", ep=ep)
    _run(db_url, model="explicit-model")
    assert _label(restricted, later) == ("work-safe", fa.LABEL_VERSION, "explicit-model")


def test_a_second_run_resumes_with_what_is_left(restricted, db_url, judge):
    ep = _ep(restricted)
    ids = [_fact(restricted, f"fact {i}", ep=ep) for i in range(5)]
    _run(db_url, batch=2, calls=1)
    assert [_audience(restricted, i) for i in ids] == [None, None, None, "work-safe", "work-safe"]
    _run(db_url, batch=2, calls=1)
    _run(db_url, batch=2, calls=1)
    assert all(_audience(restricted, i) == "work-safe" for i in ids)
    assert _run(db_url).line() == "fact audience: labelled 0 (none pending)"


# ---------------------------------------------------------------------------
# The order
# ---------------------------------------------------------------------------


def _tiered(conn) -> dict[str, int]:
    """Insertion order is id order. Reachable = live, inside the allowlist."""
    work, elsewhere = _ep(conn), _ep(conn, ELSEWHERE)
    ids: dict[str, int] = {}
    ids["r1"] = _fact(conn, "reach 1", ep=work)
    ids["r2"] = _fact(conn, "reach 2", ep=work)
    ids["o1"] = _fact(conn, "other 1", ep=elsewhere)
    ids["o2"] = _fact(conn, "other 2 (no provenance)", ep=None)
    ids["t1"] = _fact(conn, "retired 1", ep=work, retired=True)
    ids["t2"] = _fact(conn, "retired 2", ep=elsewhere, retired=True)
    ids["r3"] = _fact(conn, "reach 3", ep=work)
    ids["o3"] = _fact(conn, "other 3", ep=elsewhere)
    return ids


def test_reachable_live_facts_come_first_then_other_live_then_retired_newest_first(
    restricted, db_url, judge
):
    _tiered(restricted)
    _run(db_url, batch=50, calls=1)
    assert judge.seen == [
        "reach 3",
        "reach 2",
        "reach 1",
        "other 3",
        "other 2 (no provenance)",
        "other 1",
        "retired 2",
        "retired 1",
    ]


def test_the_cycle_takes_batch_times_calls_facts_and_no_more(restricted, db_url, judge):
    ids = _tiered(restricted)
    stats = _run(db_url, batch=2, calls=2)
    assert judge.calls == 2
    taken = {k for k, v in ids.items() if _audience(restricted, v) is not None}
    assert taken == {"r3", "r2", "r1", "o3"}
    assert stats.labelled == {"work-safe": 4}


def test_a_short_tier_leaves_room_for_the_next_one(restricted, db_url, judge):
    ep = _ep(restricted)
    _fact(restricted, "reach", ep=ep)
    _fact(restricted, "retired", ep=ep, retired=True)
    _run(db_url, batch=10, calls=1)
    assert judge.seen == ["reach", "retired"]


# ---------------------------------------------------------------------------
# Races and stops
# ---------------------------------------------------------------------------


def test_a_label_set_while_the_judge_ran_is_never_replaced(restricted, db_url, monkeypatch):
    """The classifier runs with no connection held, so a label can land in between (a hand
    UPDATE, another labeller). The write is guarded by ``audience IS NULL``."""
    fid = _fact(restricted, "plain fact", ep=_ep(restricted))
    other = _fact(restricted, "second fact", ep=_ep(restricted))

    class _Racing(_Judge):
        def __call__(self, llm, **kw):
            restricted.execute(
                "UPDATE kg_relationships SET audience = 'personal', label_version = 9 "
                "WHERE id = %s",
                (fid,),
            )
            return super().__call__(llm, **kw)

    monkeypatch.setattr(fa, "structured_call", _Racing())
    _run(db_url)
    assert _label(restricted, fid)[:2] == ("personal", 9)
    assert _label(restricted, other)[:2] == ("work-safe", fa.LABEL_VERSION)


def test_a_usage_limit_stops_the_cycle_quietly_and_keeps_what_was_written(
    restricted, db_url, monkeypatch
):
    ep = _ep(restricted)
    ids = [_fact(restricted, f"fact {i}", ep=ep) for i in range(4)]
    monkeypatch.setattr(fa, "structured_call", _Judge(UsageLimitError("limit"), fail_after=1))
    stats = _run(db_url, batch=2, calls=2)
    assert stats.stopped == "usage limit"
    assert stats.line() == "fact audience: stopped: usage limit after 2"
    # newest first: the first call labelled the two newest, the second hit the limit
    assert [_audience(restricted, i) for i in ids] == [None, None, "work-safe", "work-safe"]


def test_a_backend_that_dies_mid_cycle_keeps_what_was_written_and_stops(
    restricted, db_url, monkeypatch
):
    ep = _ep(restricted)
    ids = [_fact(restricted, f"fact {i}", ep=ep) for i in range(4)]
    monkeypatch.setattr(fa, "structured_call", _Judge(LLMHTTPError("401", 401), fail_after=1))
    stats = _run(db_url, batch=2, calls=2)
    assert stats.stopped == "classifier failing"
    # the first call labelled the two newest; the second chunk's failures proved nothing, so
    # its facts are NULL (not a stored guess) for the next cycle
    assert [_audience(restricted, i) for i in ids] == [None, None, "work-safe", "work-safe"]
    assert restricted.execute(
        "SELECT count(*) FROM kg_relationships WHERE label_model LIKE %s",
        (fa.ERROR_MODEL_PREFIX + "%",),
    ).fetchone() == (0,)


def test_a_dead_backend_labels_nothing_and_stops_after_a_bounded_number_of_calls(
    restricted, db_url, monkeypatch
):
    """The default backend raises a bare RuntimeError once the CLI's own retries are spent:
    that is not a verdict on any fact. 60 facts, two chunks, and not one is written."""
    ep = _ep(restricted)
    ids = [_fact(restricted, f"fact {i}", ep=ep) for i in range(60)]
    dead = _Judge(RuntimeError("agent_call(fact_audience): failed after retries: spawn"))
    monkeypatch.setattr(fa, "structured_call", dead)
    stats = _run(db_url, batch=50, calls=2)
    assert stats.stopped == "classifier failing" and stats.labelled == {}
    assert dead.calls == fa._fail_limit(50)
    assert all(_label(restricted, i) == (None, None, None) for i in ids)


class _Poison(_Judge):
    """Fails any call whose prompt shows the fact POISON; honest otherwise."""

    def __init__(self, err: BaseException) -> None:
        super().__init__()
        self.err = err

    def __call__(self, llm, **kw):
        if "POISON" in kw["base_prompt"].split("Facts, one per line", 1)[1]:
            self.calls += 1
            raise self.err
        return super().__call__(llm, **kw)


def test_one_fact_that_breaks_calls_is_stamped_as_a_fallback_and_the_queue_moves_on(
    restricted, db_url, monkeypatch
):
    """A provider that refuses one fact's content must not wedge the newest slice for good.
    The fact is isolated by splitting, stored ``personal`` (never served restricted) with an
    ``error:`` model stamp that finds it again, and the next cycle has nothing pending."""
    monkeypatch.setenv("SYNAPSE_FACT_AUDIENCE_MODEL", "stage-model")
    ep = _ep(restricted)
    ids = [_fact(restricted, f"fact {i}", ep=ep) for i in range(4)]
    poison = _fact(restricted, "POISON fact", ep=ep)  # the newest: first in the slice
    monkeypatch.setattr(fa, "structured_call", _Poison(LLMHTTPError("400 refused", 400)))
    stats = _run(db_url, batch=5, calls=1)
    assert not stats.stopped
    assert stats.labelled == {"work-safe": 4, "personal/error": 1}
    assert _label(restricted, poison) == (
        "personal",
        fa.LABEL_VERSION,
        "error:stage-model",
    )
    assert all(_label(restricted, i) == ("work-safe", 1, "stage-model") for i in ids)
    assert _run(db_url).line() == "fact audience: labelled 0 (none pending)"


def test_a_fact_that_fails_last_in_the_cycle_waits_and_is_settled_by_the_next_one(
    restricted, db_url, monkeypatch
):
    """Nothing succeeded after it, so the failure could be the backend's: it stays NULL this
    cycle. The next cycle's slice starts at it, and the calls after it do succeed."""
    ep = _ep(restricted)
    ids = [_fact(restricted, "POISON fact" if i == 2 else f"fact {i}", ep=ep) for i in range(5)]
    monkeypatch.setattr(fa, "structured_call", _Poison(LLMHTTPError("400 refused", 400)))
    first = _run(db_url, batch=3, calls=1)  # the newest three: ids[4], ids[3], ids[2]
    assert first.stopped == "classifier failing"
    assert [_audience(restricted, i) for i in ids] == [None, None, None, "work-safe", "work-safe"]
    second = _run(db_url, batch=3, calls=1)  # ids[2] (first now), ids[1], ids[0]
    assert not second.stopped
    assert [_audience(restricted, i) for i in ids] == [
        "work-safe",
        "work-safe",
        "personal",
        "work-safe",
        "work-safe",
    ]
    assert _label(restricted, ids[2])[2].startswith(fa.ERROR_MODEL_PREFIX)


# ---------------------------------------------------------------------------
# The poller's stage
# ---------------------------------------------------------------------------


def test_the_poller_stage_labels_through_its_own_connection(restricted, db_url, judge):
    ep = _ep(restricted)
    safe, private = _fact(restricted, "plain", ep=ep), _fact(restricted, "SECRET", ep=ep)
    db = Database(db_url)
    try:
        assert Poller(db=db, audience_llm=object()).label_fact_audience() == 2
    finally:
        db.close()
    assert _audience(restricted, safe) == "work-safe"
    assert _audience(restricted, private) == "personal"
