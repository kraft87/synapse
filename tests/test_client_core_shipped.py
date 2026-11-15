"""The record of rows shipped (plugin/synapse_client/shipped.py)."""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

from tests.helpers.client_core import PLUGIN, use_core

use_core()
from synapse_client.shipped import Ledger  # noqa: E402


def _work(directory: str, plugin: str, k: int, n: int) -> None:
    import sys

    sys.path.insert(0, plugin)
    from synapse_client.shipped import Ledger as L

    led = L(directory)
    for i in range(n):
        led.add("sess", [f"p{k}-u{i}"], passed=[f"p{k}-x{i}"] if i % 7 == 0 else ())
        if i % 5 == 0:
            led.load("sess")


def test_many_processes_lose_no_addition(tmp_path: Path) -> None:
    """review-round3 ledger_mp.py, smaller: processes add to one session at once."""
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_work, args=(str(tmp_path), str(PLUGIN), k, 30)) for k in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
    got = Ledger(tmp_path).load("sess")
    assert got.uids == {f"p{k}-u{i}" for k in range(4) for i in range(30)}


def test_a_record_from_a_newer_client_is_read_and_never_rewritten(tmp_path: Path) -> None:
    """review-round3 ledger_mp.py: a newer client's record was rewritten as version 1."""
    led = Ledger(tmp_path)
    file = led._file("s2")
    tmp_path.mkdir(exist_ok=True)
    body = {"version": 7, "session_id": "s2", "uids": ["a"], "rows": {"b": 1}}
    file.write_text(json.dumps(body), encoding="utf-8")
    assert led.load("s2").uids == {"a"}
    assert led.add("s2", ["c"]).uids == {"a", "c"}
    assert json.loads(file.read_text("utf-8")) == body


def test_a_damaged_record_is_moved_aside(tmp_path: Path) -> None:
    led = Ledger(tmp_path)
    led.add("s3", ["x"])
    file = led._file("s3")
    file.write_text("{not json", encoding="utf-8")
    assert not led.load("s3").known
    assert file.with_name(file.name + ".damaged").read_text("utf-8") == "{not json"
    assert led.add("s3", ["z"]).uids == {"z"}


def test_forget_removes_one_session(tmp_path: Path) -> None:
    led = Ledger(tmp_path)
    led.add("a", ["1"])
    led.add("b", ["2"])
    led.forget("a")
    led.forget("missing")
    assert not led.load("a").known and led.load("b").uids == {"2"}
