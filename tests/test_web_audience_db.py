"""Web audience labelling against a real database (schema 063).

Covers the parts that write: the own-surface rule, the poller's ``label_pending`` (and
its stage), the backfill's propose/apply, and the brief script's ``--audience``. The
invariants: a label is only ever written where the row is still NULL (except an
explicit ``--audience``, which wins outright), nothing is labelled before 063, and a
transport failure leaves the row NULL so a later run retries it.
"""

from __future__ import annotations

import io
import os
import sys
import uuid
from pathlib import Path

import psycopg
import pytest

_DB_URL = os.environ.get(
    "SYNAPSE_TEST_URL", "postgresql://synapse:synapse@127.0.0.1:5432/synapse_test"
)

try:
    _probe = psycopg.connect(_DB_URL, connect_timeout=2)
    _probe.close()
except Exception:  # pragma: no cover - environment dependent
    pytest.skip("no test DB reachable", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import ingest_research_brief as brief_script  # noqa: E402
import web_audience_backfill as backfill  # noqa: E402

from ingestion import web_audience as wa  # noqa: E402
from ingestion.db import Database  # noqa: E402
from ingestion.llm_client import LLMHTTPError  # noqa: E402
from ingestion.poller import Poller  # noqa: E402
from ingestion.web_audience import WebAudienceVerdict  # noqa: E402
from tests.helpers.surfaces import (  # noqa: E402
    FULL_SURFACE,
    RESTRICTED_SURFACE,
    clear_surfaces,
    register_full,
    register_restricted,
)


@pytest.fixture()
def clean(conn):
    def _wipe():
        conn.execute("TRUNCATE web_artifacts RESTART IDENTITY CASCADE")
        conn.execute("TRUNCATE episodes RESTART IDENTITY CASCADE")
        clear_surfaces(conn)

    _wipe()
    yield conn
    _wipe()


def _artifact(conn, kind="web_scrape", content="widget notes", *, audience=None, session=None):
    return conn.execute(
        "INSERT INTO web_artifacts (kind, tool_name, tool_use_id, url, title, "
        "content_markdown, fetched_at, audience, session_id) "
        "VALUES (%s, 'test', %s, 'https://example.org/p', 'a page', %s, now(), %s, %s) "
        "RETURNING id",
        (kind, f"t:{uuid.uuid4().hex}", content, audience, session),
    ).fetchone()[0]


def _audience(conn, aid):
    return conn.execute("SELECT audience FROM web_artifacts WHERE id = %s", (aid,)).fetchone()[0]


def _turns(conn, session, *surfaces):
    for i, sid in enumerate(surfaces, start=1):
        conn.execute(
            "INSERT INTO episodes (session_id, sequence, project, content, surface_id) "
            "VALUES (%s, %s, 'p', 'a turn', %s)",
            (session, i, sid),
        )


class _Judge:
    """Stands in for ``structured_call``: work-safe unless the document says ``SECRET-TOPIC``;
    raises ``fail`` when set. Counts calls."""

    def __init__(self, fail: BaseException | None = None, fail_after: int = 0) -> None:
        self.calls = 0
        self.fail, self.fail_after = fail, fail_after

    def __call__(self, llm, **kw):
        self.calls += 1
        if self.fail is not None and self.calls > self.fail_after:
            raise self.fail
        if "SECRET-TOPIC" in kw["base_prompt"]:
            return WebAudienceVerdict(verdict="personal", personal_signals=["private life"])
        return WebAudienceVerdict(verdict="work-safe", personal_signals=[])


@pytest.fixture()
def judge(monkeypatch):
    j = _Judge()
    monkeypatch.setattr(wa, "structured_call", j)
    monkeypatch.setenv("SYNAPSE_WEB_AUDIENCE_VOTES", "1")
    return j


# ---------------------------------------------------------------------------
# Rule 2: own surface
# ---------------------------------------------------------------------------


def test_own_surface_needs_every_turn_from_one_approved_restricted_surface(clean):
    register_restricted(clean, [])
    register_restricted(clean, [], surface_id="test-restricted-two")
    register_full(clean)
    _turns(clean, "own", RESTRICTED_SURFACE, RESTRICTED_SURFACE)
    _turns(clean, "unstamped", RESTRICTED_SURFACE, None)
    _turns(clean, "trusted", FULL_SURFACE)
    _turns(clean, "two", RESTRICTED_SURFACE, "test-restricted-two")
    assert wa.own_surface_session(clean, "own")
    for session in ("unstamped", "trusted", "two", "no-turns", None):
        assert not wa.own_surface_session(clean, session), session
    clean.execute(
        "UPDATE surfaces SET status = 'revoked' WHERE surface_id = %s", (RESTRICTED_SURFACE,)
    )
    assert not wa.own_surface_session(clean, "own")


# ---------------------------------------------------------------------------
# label_pending: the poller's stage
# ---------------------------------------------------------------------------


def _connect(db_url):
    db = Database(db_url)
    return db, db._conn


def test_label_pending_labels_only_unlabelled_servable_rows(clean, db_url, judge):
    register_restricted(clean, [])
    _turns(clean, "own", RESTRICTED_SURFACE)
    safe = _artifact(clean, "research_brief", "kubernetes notes")
    private = _artifact(clean, "web_scrape", "SECRET-TOPIC notes")
    own = _artifact(clean, "web_scrape", "SECRET-TOPIC notes", session="own")
    kept = _artifact(clean, "web_scrape", "kubernetes notes", audience="personal")
    results = _artifact(clean, "search_result_set", "kubernetes notes")
    empty = _artifact(clean, "web_scrape", None)
    db, connect = _connect(db_url)
    try:
        stats = wa.label_pending(connect, object(), limit=10)
    finally:
        db.close()
    assert _audience(clean, safe) == "work-safe"
    assert _audience(clean, private) == "personal"
    assert _audience(clean, own) == "work-safe"  # rule 2, no classifier call
    assert _audience(clean, kept) == "personal"  # never overwritten
    assert _audience(clean, results) is None and _audience(clean, empty) is None
    assert judge.calls == 2
    assert stats.labelled == {
        "work-safe/classifier": 1,
        "personal/classifier": 1,
        "work-safe/own-surface": 1,
    }


def test_label_pending_takes_the_newest_first_and_respects_the_limit(clean, db_url, judge):
    ids = [_artifact(clean) for _ in range(3)]
    db, connect = _connect(db_url)
    try:
        wa.label_pending(connect, object(), limit=2)
    finally:
        db.close()
    assert [_audience(clean, i) for i in ids] == [None, "work-safe", "work-safe"]


def test_label_pending_never_overwrites_a_label_set_while_it_classified(clean, db_url, monkeypatch):
    """The classifier runs with no connection held, so a label can land in between (an
    explicit --audience, a hand UPDATE, the backfill). The poller's write is guarded by
    ``audience IS NULL`` and must not replace it, least of all with work-safe."""
    aid = _artifact(clean, content="kubernetes notes")

    class _Racing(_Judge):
        def __call__(self, llm, **kw):
            clean.execute("UPDATE web_artifacts SET audience = 'personal' WHERE id = %s", (aid,))
            return super().__call__(llm, **kw)

    monkeypatch.setattr(wa, "structured_call", _Racing())
    monkeypatch.setenv("SYNAPSE_WEB_AUDIENCE_VOTES", "1")
    db, connect = _connect(db_url)
    try:
        wa.label_pending(connect, object(), limit=10)
    finally:
        db.close()
    assert _audience(clean, aid) == "personal"


def test_a_transport_error_leaves_the_rest_unlabelled(clean, db_url, monkeypatch):
    monkeypatch.setattr(wa, "structured_call", _Judge(LLMHTTPError("401", 401), fail_after=1))
    monkeypatch.setenv("SYNAPSE_WEB_AUDIENCE_VOTES", "1")
    ids = [_artifact(clean) for _ in range(3)]
    db, connect = _connect(db_url)
    try:
        with pytest.raises(LLMHTTPError):
            wa.label_pending(connect, object(), limit=10)
    finally:
        db.close()
    # newest first: the first one was labelled and committed, the rest stay NULL
    assert [_audience(clean, i) for i in ids] == [None, None, "work-safe"]


def test_label_pending_does_nothing_before_063(clean, db_url, judge, monkeypatch):
    class _Absent:
        def ready(self, conn, key=""):
            return False

    monkeypatch.setattr(wa, "WEB_AUDIENCE_PROBE", _Absent())
    aid = _artifact(clean)
    db, connect = _connect(db_url)
    try:
        stats = wa.label_pending(connect, object())
    finally:
        db.close()
    assert stats.skipped == "schema 063 not applied" and judge.calls == 0
    assert _audience(clean, aid) is None


def test_the_poller_stage_labels_through_its_own_connection(clean, db_url, judge):
    aid = _artifact(clean, content="SECRET-TOPIC notes")
    db = Database(db_url)
    try:
        assert Poller(db=db, audience_llm=object()).label_web_audience() == 1
    finally:
        db.close()
    assert _audience(clean, aid) == "personal"


# ---------------------------------------------------------------------------
# scripts/web_audience_backfill.py
# ---------------------------------------------------------------------------


def test_propose_writes_a_review_file_and_no_labels(clean, judge):
    safe = _artifact(clean, content="kubernetes notes")
    private = _artifact(clean, content="SECRET-TOPIC notes")
    out = io.StringIO()
    counts = backfill.propose(clean, lambda: object(), out, workers=2)
    assert counts == {"work-safe": 1, "personal": 1}
    lines = out.getvalue().splitlines()
    assert lines[0].split("\t") == list(backfill.COLUMNS)
    proposed = {int(r.split("\t")[0]): r.split("\t")[2] for r in lines[1:]}
    assert proposed == {safe: "work-safe", private: "personal"}
    assert _audience(clean, safe) is None and _audience(clean, private) is None


def test_propose_records_a_transport_error_as_error(clean, monkeypatch):
    monkeypatch.setattr(wa, "structured_call", _Judge(LLMHTTPError("down", 503)))
    _artifact(clean)
    out = io.StringIO()
    assert backfill.propose(clean, lambda: object(), out) == {"error": 1}


def test_apply_writes_only_where_null_and_dry_run_rolls_back(clean, tmp_path):
    a = _artifact(clean)
    b = _artifact(clean, audience="personal")
    c = _artifact(clean)
    review = tmp_path / "review.tsv"
    review.write_text(
        "\t".join(backfill.COLUMNS)
        + "\n"
        + f"{a}\tweb_scrape\twork-safe\tclassifier\tt\t\n"
        + f"{b}\tweb_scrape\twork-safe\tclassifier\tt\t\n"
        + f"{c}\tweb_scrape\terror\ttransport\tt\tdown\n"
    )
    pairs = backfill.read_review(review)
    assert pairs == [(a, "work-safe"), (b, "work-safe")]
    with pytest.raises(backfill._Rollback) as rb:
        backfill.apply(clean, pairs, dry_run=True)
    assert rb.value.changed == 1 and _audience(clean, a) is None
    assert backfill.apply(clean, pairs) == 1
    assert [_audience(clean, x) for x in (a, b, c)] == ["work-safe", "personal", None]


def test_read_review_refuses_a_file_without_the_header(tmp_path):
    bad = tmp_path / "x.tsv"
    bad.write_text("1\tweb_scrape\twork-safe\n")
    with pytest.raises(SystemExit):
        backfill.read_review(bad)


# ---------------------------------------------------------------------------
# scripts/ingest_research_brief.py --audience
# ---------------------------------------------------------------------------


def _ingest(monkeypatch, path, *extra):
    monkeypatch.setattr(sys, "argv", ["ingest", "--file", str(path), "--slug", "s", *extra])
    assert brief_script.main() == 0
    return clean_audience()


def clean_audience():
    with psycopg.connect(_DB_URL) as c:
        return c.execute(
            "SELECT audience FROM web_artifacts WHERE kind = 'research_brief'"
        ).fetchall()


def test_brief_without_audience_stays_unlabelled_and_the_flag_wins(clean, tmp_path, monkeypatch):
    monkeypatch.setenv("SYNAPSE_DB_URL", _DB_URL)
    doc = tmp_path / "brief.md"
    doc.write_text("# A brief\n\nwidget notes\n")
    assert _ingest(monkeypatch, doc) == [(None,)]
    assert _ingest(monkeypatch, doc, "--audience", "work-safe") == [("work-safe",)]
    assert _ingest(monkeypatch, doc, "--audience", "personal") == [("personal",)]
