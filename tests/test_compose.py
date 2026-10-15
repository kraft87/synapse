"""compose.yml carries the contracts the code and the operator rely on."""

from pathlib import Path

import pytest
import yaml

from ingestion import poller as poller_mod

_SERVICES = yaml.safe_load((Path(__file__).resolve().parent.parent / "compose.yml").read_text())[
    "services"
]


def test_poller_healthcheck_reads_the_files_the_threads_touch():
    """Both threads must be alive for the check to pass: it names each one's heartbeat."""
    test = " ".join(_SERVICES["poller"]["healthcheck"]["test"])
    assert poller_mod._EXTRACTION_HEARTBEAT.name in test
    assert poller_mod._MAINTENANCE_HEARTBEAT.name in test


def test_pollers_get_time_to_finish_their_in_flight_items():
    """An item takes minutes; a stop that kills sooner strands its claim for 45 minutes."""
    assert _SERVICES["poller"]["stop_grace_period"] == "5m"


def test_no_service_is_a_drain_only_poller():
    """The poller does both jobs; SYNAPSE_DRAIN_ONLY is honoured only for a compose that
    has not been re-applied, never set by this one."""
    assert "poller-drain" not in _SERVICES
    for service in _SERVICES.values():
        assert not any("SYNAPSE_DRAIN_ONLY" in str(e) for e in service.get("environment", []))


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
