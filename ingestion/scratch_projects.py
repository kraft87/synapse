"""Drop turns from throwaway smoke-test workspaces at the ingest boundary.

Some Claude Code workspaces exist only to exercise a client, never to do work:
``axon-test/`` was the scratch dir for smoke-testing the Axon chat app, and bare
``tmp``/``test`` projects come from one-off `claude -p` probes run out of /tmp. Their
turns are ping/pong ("say the single word fresh", "reply with exactly {\"ok\":true}"),
arithmetic, and throwaway fiction prompts used to generate streaming text.

Nothing downstream can tell that content apart from real memory, and the KG extractor
reads it as biography. On 2026-06-05 the extractor mined a robot-gardener story from
``axon-test`` and wrote ``Elena is Marisol's daughter, living in the apartment
building...`` into the PERSONAL graph, where it sat for three months and was served by
recall as a fact about the operator's life. Fifty-two more facts came from the same
class of session: a beekeeper named Marta, a clockmaker named Idris, "why is the sky
blue" physics, and sample Terraform/S3 resources from a tool-call test.

Why a project denylist and not a content heuristic: the failure is not that the text
looks fake — good fiction reads exactly like biography, and a real session can discuss
a beekeeper. The failure is that the WORKSPACE is not a memory source. The project is
the only signal that is cheap, exact, and decided by the operator rather than inferred.

Wired exactly like ``ingestion.contamination`` and ``ingestion.private_sessions``: one
predicate applied at the chokepoints where a parsed turn becomes an episode
(``mcp_server.server.ingest_turns``, ``ingestion.backfill``, ``ingestion.codex_backfill``).
Dropping there means chunks, KG extraction, dream, and the timeline never see the turn.

Matching is EXACT on the normalized project name, never a substring: a substring rule on
"test" would swallow ``pytest-tuning``, ``test-harness-rewrite``, and any legitimate
project whose name contains it. Operators extend the set with
``SYNAPSE_SCRATCH_PROJECTS`` (comma-separated), which REPLACES the defaults so a
deployment that reuses one of these names for real work can opt out entirely.
"""

from __future__ import annotations

import os

# Workspaces that exist to exercise a client, never to do work. Exact match, lowercased.
_DEFAULT_SCRATCH = frozenset({"axon-test", "axon_test", "test", "tmp", "scratch"})

_ENV_VAR = "SYNAPSE_SCRATCH_PROJECTS"


def scratch_projects() -> frozenset[str]:
    """The active denylist. ``SYNAPSE_SCRATCH_PROJECTS`` replaces the defaults.

    An empty or whitespace-only value disables the filter entirely — that is the
    documented opt-out for a deployment where one of these names is a real project.
    """
    raw = os.environ.get(_ENV_VAR)
    if raw is None:
        return _DEFAULT_SCRATCH
    return frozenset(p.strip().lower() for p in raw.split(",") if p.strip())


def is_scratch_project(project: str | None) -> bool:
    """True if `project` is a throwaway test workspace (drop the turn, do not store it)."""
    if not project:
        return False
    return project.strip().lower() in scratch_projects()
