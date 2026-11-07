"""The mention_count reinforcement prior on the fused KG fact score (kg_pg._mention_prior).

The fact leg fuses vector, BM25 and hop lists by RRF and cuts to the top ``limit``. With
``SYNAPSE_KG_MENTION_W`` > 0 (default 0.1) every fused score is first multiplied by
``1 + w * ln(mention_count)``, so a fact re-asserted across conversations (mention_count > 1,
schema 019) can displace a one-off of similar relevance in the cut. Sections:
  * the multiplier arithmetic, and that off (weight 0) sends no extra statement
  * the cut: a reinforced fact of lower rank takes the slot only with the prior on
  * the prior reorders what the legs returned and never widens it, including restricted callers
  * the knob parses from the environment
"""

from __future__ import annotations

import math
import subprocess
import sys
import uuid
from typing import Any

import psycopg
import pytest

import mcp_server.recall as recall_mod
from ingestion.surfaces import FULL_TRUST
from mcp_server import kg_pg
from mcp_server.recall import Recall
from tests.helpers.embed import onehot

TECH = "technical"
WORK = "work-proj"
OTHER = "other-proj"


class _Cur:
    """A cursor that answers the mention_count lookup from a dict and records statements."""

    def __init__(self, counts: dict[str, Any]):
        self.counts = counts
        self.sql: list[str] = []

    def execute(self, sql: str, params: Any = None) -> None:
        self.sql.append(sql)
        self._rows = [(u, self.counts[u]) for u in params[0] if u in self.counts]

    def fetchall(self) -> list[tuple[str, Any]]:
        return self._rows


def test_prior_multiplies_by_one_plus_w_ln_mention_count():
    cur = _Cur({"once": 1, "null": None, "zero": 0, "e": math.e, "ten": 10})
    fused = {"once": 0.02, "null": 0.02, "zero": 0.02, "e": 0.02, "ten": 0.02, "unknown": 0.02}
    out = kg_pg._mention_prior(cur, fused, 0.3)
    assert out["once"] == pytest.approx(0.02)  # a fact stated once is the neutral value
    assert out["null"] == pytest.approx(0.02)
    assert out["zero"] == pytest.approx(0.02)  # a sub-1 count never lowers a score
    assert out["unknown"] == pytest.approx(0.02)  # a uuid the lookup did not return
    assert out["e"] == pytest.approx(0.02 * 1.3)
    assert out["ten"] == pytest.approx(0.02 * (1 + 0.3 * math.log(10)))
    assert len(cur.sql) == 1 and "mention_count" in cur.sql[0]


@pytest.mark.parametrize("w", [0.0, -0.5])
def test_prior_off_sends_nothing(w):
    cur = _Cur({"a": 9})
    fused = {"a": 0.5}
    assert kg_pg._mention_prior(cur, fused, w) is fused
    assert kg_pg._mention_prior(cur, {}, 0.3) == {}
    assert cur.sql == []


# ---------------------------------------------------------------------------
# The cut
# ---------------------------------------------------------------------------


@pytest.fixture()
def kg(conn):
    def _wipe() -> None:
        conn.execute("TRUNCATE kg_relationships, kg_entities RESTART IDENTITY CASCADE")
        conn.execute("TRUNCATE episodes RESTART IDENTITY CASCADE")

    _wipe()
    yield conn
    _wipe()


def _ep(conn, project: str = WORK) -> int:
    return conn.execute(
        "INSERT INTO episodes (session_id, sequence, project, content) "
        "VALUES (%s, 1, %s, 'a turn') RETURNING id",
        (f"kmp-{uuid.uuid4().hex[:10]}", project),
    ).fetchone()[0]


def _at(theta: float) -> list[float]:
    """A unit vector at angle ``theta`` from the query vector (onehot(0)): the larger the
    angle, the larger the cosine distance."""
    v = onehot(0)
    v[0] = math.cos(theta)
    v[1] = math.sin(theta)
    return v


def _fact(
    conn,
    uid: str,
    ep: int,
    *,
    theta: float | None,
    mentions: int = 1,
    text: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO kg_relationships (uuid, owner_id, group_id, src_uuid, tgt_uuid, name, "
        "  fact, fact_embedding, episodes, t_valid, mention_count) "
        "VALUES (%s, 'default', %s, 'ent-x', 'ent-y', 'RELATES', %s, %s::vector, %s::jsonb, "
        "        '2026-01-01T00:00:00+00:00', %s)",
        (
            uid,
            TECH,
            text or f"fact text of {uid}",
            None if theta is None else kg_pg._vec_literal(_at(theta)),
            f"[{ep}]",
            mentions,
        ),
    )


def _search(db_url: str, limit: int, allowed: list[str] | None = None) -> list[str]:
    with psycopg.connect(db_url) as c, c.cursor() as cur:
        facts, _ = kg_pg.search_kg_postgres(
            cur, "zzqq", onehot(0), "default", TECH, [], limit, allowed
        )
    return [f["_uuid"] for f in facts]


def _two(conn) -> None:
    """The one-off is the nearer vector hit (rank 1); the reinforced fact is rank 2."""
    ep = _ep(conn)
    _fact(conn, "fresh", ep, theta=0.1, mentions=1)
    _fact(conn, "reinforced", ep, theta=0.3, mentions=6)


def test_weight_zero_serves_plain_rrf_order(kg, db_url, monkeypatch):
    _two(kg)
    monkeypatch.setattr(kg_pg, "_MENTION_W", 0.0)
    assert _search(db_url, 2) == ["fresh", "reinforced"]
    assert _search(db_url, 1) == ["fresh"]


def test_default_weight_lets_the_reinforced_fact_outrank_the_one_off(kg, db_url):
    # No monkeypatch: the shipped default (0.1) must already reorder, or the prior is dead.
    _two(kg)
    assert kg_pg._MENTION_W == 0.1
    assert _search(db_url, 1) == ["reinforced"]


def test_prior_lets_the_reinforced_fact_take_the_only_slot(kg, db_url, monkeypatch):
    """The failing-before case: with the prior off the nearer one-off keeps the slot."""
    _two(kg)
    monkeypatch.setattr(kg_pg, "_MENTION_W", 0.3)
    assert _search(db_url, 1) == ["reinforced"]
    assert _search(db_url, 2) == ["reinforced", "fresh"]


def test_prior_is_neutral_when_every_fact_was_stated_once(kg, db_url, monkeypatch):
    ep = _ep(kg)
    for i in range(5):
        _fact(kg, f"f{i}", ep, theta=0.1 * (i + 1), mentions=1)
    monkeypatch.setattr(kg_pg, "_MENTION_W", 0.0)
    plain = _search(db_url, 3)
    monkeypatch.setattr(kg_pg, "_MENTION_W", 0.6)
    assert _search(db_url, 3) == plain == ["f0", "f1", "f2"]


def test_weight_sets_how_far_down_the_list_a_reinforced_fact_climbs(kg, db_url, monkeypatch):
    """RRF scores are flat (rank 1 is only 1.16x rank 11), so the weight is a sharp knob: a
    fact stated three times and ranked eleventh stays out at w=0.05 and reaches first at 0.3."""
    ep = _ep(kg)
    for i in range(10):
        _fact(kg, f"n{i}", ep, theta=0.05 * (i + 1), mentions=1)
    _fact(kg, "far", ep, theta=1.2, mentions=3)
    monkeypatch.setattr(kg_pg, "_MENTION_W", 0.05)
    assert _search(db_url, 5) == ["n0", "n1", "n2", "n3", "n4"]
    monkeypatch.setattr(kg_pg, "_MENTION_W", 0.3)
    assert _search(db_url, 5)[0] == "far"


# ---------------------------------------------------------------------------
# The prior reorders the legs' candidates and never widens them
# ---------------------------------------------------------------------------


def test_prior_never_pulls_in_a_fact_no_leg_returned(kg, db_url, monkeypatch):
    """High mention_count, but no embedding, no matching text and no seed: no leg finds it."""
    ep = _ep(kg)
    _fact(kg, "near", ep, theta=0.1, mentions=1)
    _fact(kg, "invisible", ep, theta=None, mentions=500, text="nothing to match here")
    monkeypatch.setattr(kg_pg, "_MENTION_W", 0.6)
    assert _search(db_url, 12) == ["near"]


def test_restricted_scope_still_applies_with_the_prior_on(kg, db_url, monkeypatch):
    ep_ok, ep_out = _ep(kg, WORK), _ep(kg, OTHER)
    _fact(kg, "in-scope", ep_ok, theta=0.3, mentions=1)
    _fact(kg, "out-of-scope", ep_out, theta=0.1, mentions=40)
    monkeypatch.setattr(kg_pg, "_MENTION_W", 0.6)
    assert _search(db_url, 5, [WORK]) == ["in-scope"]
    assert _search(db_url, 5) == ["out-of-scope", "in-scope"]


class _Emb:
    def embed(self, texts, task):
        return [onehot(0) for _ in texts]


def test_recall_serves_the_reinforced_fact_past_the_cut(kg, db_url, monkeypatch):
    """End to end through recall(): thirteen candidates compete for the twelve fact slots;
    the thirteenth by distance is the one the owner re-stated, and only the prior serves it."""
    ep = _ep(kg)
    for i in range(12):
        _fact(kg, f"n{i:02d}", ep, theta=0.02 * (i + 1), mentions=1)
    _fact(kg, "restated", ep, theta=0.3, mentions=9)

    def served() -> set[str]:
        r = Recall(db_url, "")
        r._ensure_embedder = lambda: _Emb()
        r._rerank_pool_scored = lambda q, pool: [(i, 0.9) for i in range(len(pool))]
        r._compact_to_passages = lambda q, eps, n, **_k: [{"id": e["id"], "text": "t"} for e in eps]
        r._search_web_reranked = lambda q, emb: []
        r._search_notes = lambda *a, **k: []
        r._increment_retrieval_counts = lambda ids: None
        r._increment_fact_retrieval_counts = lambda *a, **k: None
        r._record_metrics = lambda m: None
        out = r.recall("zzqq", group_id=TECH, trust=FULL_TRUST)
        return {f["id"].removeprefix("f:") for f in out.get("facts") or []}

    assert recall_mod._FACT_LIMIT == 12
    monkeypatch.setattr(kg_pg, "_MENTION_W", 0.0)
    assert "restated" not in served()
    monkeypatch.setattr(kg_pg, "_MENTION_W", 0.3)
    got = served()
    assert "restated" in got and len(got) == 12


# ---------------------------------------------------------------------------
# The knob
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "want"), [(None, "0.1"), ("", "0.1"), ("0", "0.0"), ("0.3", "0.3")]
)
def test_weight_is_read_from_the_environment(env, want, monkeypatch):
    monkeypatch.delenv("SYNAPSE_KG_MENTION_W", raising=False)
    if env is not None:
        monkeypatch.setenv("SYNAPSE_KG_MENTION_W", env)
    out = subprocess.run(
        [sys.executable, "-c", "import mcp_server.kg_pg as k; print(k._MENTION_W)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == want
