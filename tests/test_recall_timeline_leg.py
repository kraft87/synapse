"""recall()'s timeline leg — contract tests.

The leg fires on EVERY query (no temporal-intent gate): the old regex missed 41% of
dated questions on LongMemEval ("which came first, X or Y" carries no temporal
keyword), and the leg is one cheap parallel read serving <=8 events — an empty result
changes nothing in the payload. What needs pinning now is that the gate STAYS deleted
(re-adding a keyword gate is a measured regression) and the kill switch survives.
"""

from __future__ import annotations

import mcp_server.recall as recall_mod


def test_timeline_leg_is_ungated():
    # No temporal-intent regex: the leg must not consult query text to decide firing.
    assert not hasattr(recall_mod, "_TEMPORAL_RE")


def test_timeline_kill_switch_default_on():
    # SYNAPSE_RECALL_TIMELINE=0 is the only off-switch; default is on.
    assert recall_mod._TIMELINE_IN_RECALL is True
