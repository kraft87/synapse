"""Server-side web capture: research out of the records ``POST /ingest`` already receives.

Every agent that runs the Stop hook ships its transcript records here, so capture is
tool-agnostic and needs nothing installed on the client. The episode writer
(``ingestion.episode_writer``) calls :func:`capture_calls` for the tool calls of the turns it
stored, after they committed:

* ``calls``: the tool_use / tool_result pairs of a turn's records (Claude Code, Codex, and
  sub-agent records the conversation parser drops);
* ``detect``: web research recognised by the SHAPE of the call, never by tool name, with the
  existing parsers as optional refinements (``ingestion.web_extractors``);
* ``ruleset``: the rules users add as files (``rules/`` shipped, ``SYNAPSE_CAPTURE_DIR`` for
  the operator's): sources the shapes cannot see, stored searchable or in the cold archive;
* ``guards``: the tools and URLs that are never captured;
* ``store``: the caps, the dedup and the insert (``ingestion.web_artifacts``);
* ``report``: the dry run's account of what a live capture would do, by tool.

Off by default: ``SYNAPSE_WEB_CAPTURE=on`` turns it on, once the dry run
(``POST /capture/dry-run``, ``synapse capture --dry-run``) has shown what it would store. A
capture failure is logged and counted; it never fails or delays an episode write. Generic
captures are searchable (recall's web bucket) and not KG-extracted; the poller chunks,
contextualises and embeds them (``Maintenance`` in ``ingestion.poller``).
"""

from ingestion.capture.calls import ToolCall, episode_tool_calls, sidechain_tool_calls
from ingestion.capture.detect import Capture, Skip, decide
from ingestion.capture.report import report
from ingestion.capture.store import CaptureResult, capture_calls, check_config, enabled

__all__ = [
    "Capture",
    "CaptureResult",
    "Skip",
    "ToolCall",
    "capture_calls",
    "check_config",
    "decide",
    "enabled",
    "episode_tool_calls",
    "report",
    "sidechain_tool_calls",
]
