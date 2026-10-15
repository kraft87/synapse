"""The scratch test database's DSN, in one place.

Test modules that talk to the database directly (instead of through the ``conn`` /
``db_url`` fixtures) import ``DB_URL`` from here and mark themselves
``pytestmark = pytest.mark.db``, so they run in the serial DB passes. An unreachable
database is a failure, never a skip: a skipped DB suite looks like a green one."""

import os

DB_URL = os.environ.get(
    "SYNAPSE_TEST_URL", "postgresql://synapse:synapse@127.0.0.1:5432/synapse_test"
)
