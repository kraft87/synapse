"""Run the Synapse ingestion poller as a persistent process.

Usage:
    python -m ingestion
"""

import logging
import os
import sys

import logfire

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)

# Logfire spans for the extraction pipeline. Explicit spans on stage
# 6+7 + auto-instrumentation of every LLM call (claude_agent_sdk +
# anthropic) and HTTP request so the trace UI shows the full tree:
#
#   process_facts_for_group
#     stage6b_batch_confirm
#       claude_agent_sdk.messages.create  <-- per-LLM call latency, tokens, model
#     stage7_write_edges
#       create_edge (n times)
#         contradiction_detector  <-- only fires when similarity gate passes
#
# Each poller / drain replica registers under its own service so traces
# fan out per-worker in the UI.
logfire.configure(
    service_name=os.environ.get("LOGFIRE_SERVICE_NAME", "synapse-poller"),
    send_to_logfire="if-token-present",
)
logfire.instrument_claude_agent_sdk()
logfire.instrument_anthropic()
logfire.instrument_httpx()

from ingestion.config import get_settings  # noqa: E402
from ingestion.poller import make_poller  # noqa: E402

cfg = get_settings()

poller = make_poller(
    db_url=cfg.db_url,
    voyage_api_key=cfg.voyage_api_key,
)

poller.run_loop(interval_seconds=cfg.poll_interval_seconds)
