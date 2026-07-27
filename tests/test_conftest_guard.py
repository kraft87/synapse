"""The scratch-database guard in conftest (`_refuse_production`).

pytest.ini loads `.env`, which carries the production `SYNAPSE_DB_URL` beside
`SYNAPSE_TEST_URL`, and the DB fixtures TRUNCATE episodes / kg_entities / notes. The
guard is the only thing standing between a mistyped DSN and wiping live memory, so it
gets a test of its own: scratch shapes run, everything else raises before a connection
is ever opened.
"""

from __future__ import annotations

import pytest

from tests.conftest import _refuse_production

_SCRATCH = [
    "postgresql://synapse:pw@127.0.0.1:5432/synapse_test",  # conftest's default
    "postgresql://synapse:pw@db.example.net:5432/ci_28563484663_1",  # CI, per ci.yml
    "postgresql://synapse:pw@127.0.0.1:5432/skillsv2f_test",
    "postgresql://synapse:pw@127.0.0.1:5432/synapse_test?sslmode=require",
]

_PRODUCTION = [
    "postgresql://synapse:pw@192.168.0.20:5432/synapse",  # the live database
    "postgresql://synapse:pw@127.0.0.1:5432/postgres",
    "postgresql://synapse:pw@127.0.0.1:5432/axon",
    "postgresql://synapse:pw@127.0.0.1:5432/",  # no database named at all
]


@pytest.mark.parametrize("url", _SCRATCH)
def test_scratch_databases_are_allowed(url):
    _refuse_production(url)


@pytest.mark.parametrize("url", _PRODUCTION)
def test_non_scratch_databases_are_refused(url):
    with pytest.raises(RuntimeError, match="refusing to run"):
        _refuse_production(url)
