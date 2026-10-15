"""compose.yml carries the contracts the code and the operator rely on."""

import signal
from pathlib import Path

import pytest
import yaml

from ingestion import poller as poller_mod
from ingestion.poller import Poller

_SERVICES = yaml.safe_load((Path(__file__).resolve().parent.parent / "compose.yml").read_text())[
    "services"
]


@pytest.mark.parametrize("name", ["poller", "poller-drain"])
def test_poller_healthcheck_reads_the_file_run_loop_touches(name):
    test = _SERVICES[name]["healthcheck"]["test"]
    assert poller_mod._HEARTBEAT.name in " ".join(test)


@pytest.mark.parametrize("name", ["poller", "poller-drain"])
def test_pollers_get_time_to_finish_their_in_flight_items(name):
    """An item takes minutes; a stop that kills sooner strands its claim for 45 minutes."""
    assert _SERVICES[name]["stop_grace_period"] == "5m"


@pytest.mark.parametrize(
    "threads,with_workers,claims", [("3", True, 3), ("1", True, 1), ("4", False, 1)]
)
def test_run_loop_claims_one_row_per_worker_thread(
    tmp_path, monkeypatch, threads, with_workers, claims
):
    """A claim that only waits for a free thread is held for nothing, and is stranded
    for 45 minutes if the worker is killed."""

    class _Db:
        def release_stale_claims(self) -> int:
            return 0

    monkeypatch.setattr(poller_mod, "_HEARTBEAT", tmp_path / "poller.alive")
    monkeypatch.setattr(signal, "signal", lambda *_: None)
    monkeypatch.setenv("SYNAPSE_DRAIN_ONLY", "1")
    monkeypatch.setenv("SYNAPSE_DRAIN_CONCURRENCY", threads)
    p = Poller(db=_Db(), worker_factory=(lambda: (None, None)) if with_workers else None)  # type: ignore[arg-type,return-value]
    claimed = []

    def drain(batch_limit: int) -> int:
        claimed.append(batch_limit)
        p.request_stop()
        return 0

    monkeypatch.setattr(p, "drain_extraction_queue", drain)
    p.run_loop(interval_seconds=3600)
    assert claimed == [claims]


def test_run_loop_touches_the_heartbeat(tmp_path, monkeypatch):
    class _Db:
        def release_stale_claims(self) -> int:
            return 0

    beat = tmp_path / "poller.alive"
    monkeypatch.setattr(poller_mod, "_HEARTBEAT", beat)
    monkeypatch.setattr(signal, "signal", lambda *_: None)  # not the test runner's handlers
    monkeypatch.setenv("SYNAPSE_DRAIN_ONLY", "1")
    p = Poller(db=_Db())  # type: ignore[arg-type]

    beat_when_drained = []

    def drain(batch_limit: int) -> int:
        beat_when_drained.append(beat.exists())
        p.request_stop()
        return 0

    monkeypatch.setattr(p, "drain_extraction_queue", drain)
    p.run_loop(interval_seconds=3600)
    assert beat_when_drained == [True]


@pytest.mark.parametrize(
    "name,port",
    [("postgres", "5432"), ("local-inference", "7997")],
)
def test_host_ports_that_are_not_the_product_bind_loopback(name, port):
    """Postgres carries only a password and the local-inference API has none; the
    app services reach both over the compose network, so the host port is for the
    operator's own machine unless they publish it wider on purpose."""
    published = [str(p) for p in _SERVICES[name]["ports"]]
    assert len(published) == 1
    assert published[0].startswith("127.0.0.1:") and published[0].endswith(f":{port}")
