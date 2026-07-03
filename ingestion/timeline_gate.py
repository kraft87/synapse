"""Per-turn chat gate → timeline events (the timeline's conversation feeder).

Every turn already arrives server-side as an "episode" extraction-queue item (the
plugin's Stop hook → /ingest). This gate runs there: ONE small LLM call per turn
asking "did something actually HAPPEN?" — most turns are discussion and emit
nothing. On emit, one naked past-tense event lands in timeline_events, dated to
the turn (episodes.created_at), source_ref "ep:<id>" so a timeline hit can hydrate
the full turn.

The episode content the parser builds already interleaves the user text, the
assistant text, and the tool-activity lines ("[tool:Bash] ...") — and the tool
trace is where the happenings in coding sessions actually live, so the gate sees
it all in one block.

Naked events, per the 2026-07-01 design review: NO actor field — the VERB carries
decides-vs-executes ("decided/chose/approved" vs "committed/shipped/fixed/ran").
Facts must be self-contained (never "the bug above"). Salience 0/1/2 from the gate.

Fail-soft and env-gated: SYNAPSE_TIMELINE_GATE=0 kills it; any error is logged and
swallowed so KG extraction never breaks on the timeline's account.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from ingestion.llm_client import MalformedResponseError, parse_with_retry

logger = logging.getLogger(__name__)

# Identifiers that pin an event to a concrete artifact (PR/issue refs, commit SHAs).
# Used for exact-match dedup: embedding similarity measurably CANNOT separate "same
# happening restated" (0.63) from "related but distinct" (0.83) on real timeline rows,
# so dedup keys on shared identifiers instead — precise, no false kills.
_IDENT_RE = re.compile(r"#\d{2,6}\b|\b[0-9a-f]{7,40}\b")
_ANNOUNCE_VERBS = ("committed", "merged", "pushed")
_XDEDUP_WINDOW_HOURS = 72


def extract_idents(fact: str) -> list[str]:
    return _IDENT_RE.findall(fact.lower())


_MIN_CONTENT = 80  # turns shorter than this can't contain a happening worth keeping

GATE_PROMPT = """You are building a personal work TIMELINE from ONE turn of a coding/assistant session (the user directs; an AI agent executes). Decide if SOMETHING HAPPENED this turn worth a permanent dated timeline entry, and if so write it as ONE naked past-tense event.

EMIT for a concrete happening: a decision reached, an action carried out (usually visible in the [tool:...] activity lines — code written, a command run, a commit, a deploy, a bug fixed), a result/finding, or a milestone / state change.
DO NOT emit (return null) for: questions, requests or instructions with no outcome yet, opinions, brainstorming or design talk with NO decision reached, greetings, status checks, acknowledgements. Most turns are discussion — return null for them.

Write the `event` under two hard rules:
1. NAKED + VERB-HONEST. No leading name. Start with the past-tense verb, and let the VERB carry who-did-what: "decided / chose / approved / rejected" for a DECISION or direction; "committed / shipped / deployed / fixed / built / wrote / ran / added" for an ACTION the agent carried out. (e.g. "decided episodic and semantic memory live in separate stores"; "fixed the ingest dating bug so fact valid-time inherits the segment timestamp".)
2. SELF-CONTAINED. It must make sense a month from now with NO other context. Name the concrete thing. NEVER "the bug above", "that approach" — spell it out.
3. Start with a LOWERCASE verb (it's a log line, not a sentence).
4. NO third-party personal names (clients, transcript subjects, other people's data) — describe generically ("8 client transcripts", not the names). Project/tool/service names are fine.

salience: 2 = milestone / shipped to prod / major decision; 1 = a normal action or decision; 0 = minor or routine.
event_type: "decision" (a choice/direction was reached), "action" (something was executed: code, command, deploy, fix), "finding" (a result/diagnosis/measurement was learned), or "milestone" (a phase completed / shipped).

Output ONLY JSON: {"event": "<naked past-tense fact>", "salience": 0|1|2, "event_type": "decision"|"action"|"finding"|"milestone"}  OR  {"event": null}

THE TURN:
"""


def _parse_gate(text: str) -> dict[str, Any] | None:
    """Parser for parse_with_retry: {"event": null} -> None, else validated dict."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise MalformedResponseError("no JSON object in response", text[:200])
    d = json.loads(text[start : end + 1])
    ev = d.get("event")
    if not ev:
        return None
    sal = d.get("salience", 1)
    et = d.get("event_type")
    return {
        "event": str(ev).strip(),
        "salience": sal if isinstance(sal, int) and 0 <= sal <= 2 else 1,
        "event_type": et if et in ("decision", "action", "finding", "milestone") else None,
    }


class TimelineGate:
    """One gate call per episode-type extraction item; writes timeline_events rows."""

    def __init__(
        self,
        db: Any,
        llm_client: Any,
        embedder: Any,
        model: str = "claude-haiku-4-5",
    ) -> None:
        self._db = db
        self._llm_client = llm_client
        self._embedder = embedder
        self._model = model
        self.enabled = os.environ.get("SYNAPSE_TIMELINE_GATE", "1") != "0"

    def process(self, item: dict[str, Any]) -> None:
        """Gate one turn. Fail-soft: errors are logged, never raised."""
        if not self.enabled:
            return
        try:
            self._process(item)
        except Exception as e:
            logger.warning("timeline gate failed for item %s: %s", item.get("id"), e)

    def _process(self, item: dict[str, Any]) -> None:
        content = (item.get("content") or "").strip()
        episode_id = item.get("episode_id")
        if not episode_id or len(content) < _MIN_CONTENT:
            return

        gate = parse_with_retry(
            self._llm_client,
            base_prompt=GATE_PROMPT + content[:6000],
            parser=_parse_gate,
            model=self._model,
            max_tokens=256,
        )
        if gate is None:
            return

        # Dated to the turn itself (episodes.created_at) — never ingest wall-clock.
        t_valid = self._db.get_episodes_valid_at([int(episode_id)])
        if not t_valid:
            return

        # Write-time cross-source dedup: a bare commit/merge ANNOUNCEMENT whose PR-ref/SHA
        # is already on the timeline (git is canonical for those) adds nothing — skip it.
        # Deploys/fixes/decisions about the same ref are distinct happenings and still write.
        # Ordering caveat: chat often ingests BEFORE the git feeder pushes, so this catches
        # only chat-after-git; the read-time identifier-collapse in serving covers the rest.
        verb = gate["event"].split(None, 1)[0].lower().rstrip(":,")
        idents = extract_idents(gate["event"])
        if idents and verb in _ANNOUNCE_VERBS:
            if self._db.timeline_ident_exists(
                idents, item.get("project"), t_valid, _XDEDUP_WINDOW_HOURS
            ):
                logger.info("timeline gate skip (ident dup %s) ep:%s", idents, episode_id)
                return

        project = item.get("project")
        vec = self._embedder.embed(
            [f"Project: {project or '-'} | {gate['event']}"], task="document"
        )[0]
        self._db.insert_timeline_event(
            t_valid=t_valid,
            fact=gate["event"],
            source="chat",
            source_ref=f"ep:{episode_id}",
            project=project,
            salience=gate["salience"],
            embedding=vec,
            # model_name is set by every factory-built embedder; the getattr
            # fallback covers test stubs that only implement embed().
            embed_model=getattr(self._embedder, "model_name", None) or "voyage-4-large",
            event_type=gate.get("event_type"),
        )
        logger.info(
            "timeline event (s%d) from ep:%s: %s", gate["salience"], episode_id, gate["event"][:80]
        )
