#!/usr/bin/env python3
# mypy: ignore-errors
"""UserPromptSubmit hook — a one-line per-prompt reminder to use Synapse.

CLAUDE.md-style standing instructions sit at the top of a long context and get
ignored under pressure; a reminder injected WITH each prompt rides recency and
makes recall/remember use consistent. Deliberately NOT the retired auto-recall
push (which injected multi-second recall RESULTS with no relevance gate and
degraded reasoning): this is a static directive — zero latency, zero
API calls, the model still decides relevance.

Wording (v2, 2026-07-12): imperative + motivating rationale + conditional guard —
the shipped-system pattern (Anthropic's memory-tool directive pairs ALWAYS with
"ASSUME INTERRUPTION"; the openfang #583 fix made "recall first" conditional on
context). Soft pointers under-trigger; suppression adverbs are measured no-ops —
bias reads ON, guard with the already-in-context condition, not softeners.
About 70 tokens/turn with the timestamp line (v1 was ~22). Whether that earns its keep is
still open: the planned comparison (recall call rate of sessions with and without the
directive, from recall_metrics) has not been run; the result goes in HANDOFF. If it loses,
drop the directive and keep the timestamp line. It stays per prompt either way (see above),
not on the session-start board.

Also injects a wall-clock timestamp line ("[Wed 2026-08-26 15:55:01 EDT]",
host-local time): remember() and timeline references need the current moment,
and the model otherwise only knows the session-start date. The weekday is spelled
out because deriving it from a bare date is exactly the arithmetic models get wrong.

Switches: SYNAPSE_RECALL_NUDGE=0 and SYNAPSE_PROMPT_TIMESTAMP=0 (env, or `synapse config
recall_nudge off` / `prompt_timestamp off`). The Codex hook (plugin-codex/hooks/
user_prompt_submit.py) calls envelope() with its own wording.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import _cfg

# The tool is named differently on each harness, so the directive is the caller's.
DIRECTIVE = (
    "[Synapse] IMPORTANT: the user's cross-session history is searchable — assume "
    "it has context on any past work, project, device, or person they name. BEFORE "
    "answering anything that references the past: synapse:recall (skip only if the "
    "answer is already in this conversation). About to say 'noted' or 'I'll "
    "remember'? Call synapse:remember FIRST, then reply."
)


def envelope(directive: str) -> str | None:
    """The hook's JSON output for ``directive`` (timestamp line first), or None when both
    switches are off: nothing, not a blank line."""
    lines = []
    if _cfg("SYNAPSE_PROMPT_TIMESTAMP", "1") != "0":
        lines.append(time.strftime("[%a %Y-%m-%d %H:%M:%S %Z]"))
    if _cfg("SYNAPSE_RECALL_NUDGE", "1") != "0":
        lines.append(directive)
    if not lines:
        return None
    return json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": "\n".join(lines),
            }
        }
    )


def main() -> None:
    out = envelope(DIRECTIVE)
    if out:
        print(out)


if __name__ == "__main__":
    main()
