"""ColumnProbe: the cached "has this optional migration been applied?" answer.

Pure-logic: the connection is a fake whose probe statement the test controls."""

from __future__ import annotations

import threading
from contextlib import nullcontext

from ingestion.schema_check import ColumnProbe


class _Conn:
    def __init__(self, row=(1,), error: Exception | None = None):
        self.row, self.error, self.statements = row, error, 0

    def transaction(self):
        return nullcontext()

    def execute(self, sql):
        self.statements += 1
        if self.error:
            raise self.error
        return self

    def fetchone(self):
        return self.row


def test_a_positive_answer_is_cached_for_the_process():
    probe, conn = ColumnProbe("SELECT 1", "thing"), _Conn()
    assert probe.ready(conn, "db") and probe.ready(conn, "db")
    assert conn.statements == 1 and probe.cached("db") is True


def test_a_negative_answer_is_reprobed_only_after_the_ttl():
    probe, conn = ColumnProbe("SELECT 1", "thing", reprobe_s=60.0), _Conn(row=None)
    assert not probe.ready(conn, "db") and not probe.ready(conn, "db")
    assert conn.statements == 1 and probe.cached("db") is False
    probe.reprobe_s = 0.0
    conn.row = (1,)
    assert probe.ready(conn, "db") and conn.statements == 2


def test_a_failed_probe_answers_false_and_is_not_retried_at_once():
    probe, conn = ColumnProbe("SELECT 1", "thing"), _Conn(error=RuntimeError("boom"))
    assert not probe.ready(conn, "db") and not probe.ready(conn, "db")
    assert conn.statements == 1


def test_a_thread_arriving_mid_probe_waits_for_the_answer():
    inside, release = threading.Event(), threading.Event()
    probe = ColumnProbe("SELECT 1", "thing")

    class _Slow(_Conn):
        def execute(self, sql):
            inside.set()
            release.wait(3.0)
            return super().execute(sql)

    conn, answers = _Slow(), []
    first = threading.Thread(target=lambda: answers.append(probe.ready(conn, "db")))
    first.start()
    assert inside.wait(3.0)
    assert probe.cached("db") is None  # a probe in flight is not an answer
    second = threading.Thread(target=lambda: answers.append(probe.ready(conn, "db")))
    second.start()
    release.set()
    first.join(3.0)
    second.join(3.0)
    assert answers == [True, True] and conn.statements == 1
