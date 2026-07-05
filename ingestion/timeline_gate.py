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
from datetime import datetime, timedelta
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

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _resolve_event_date(date_str: str | None, turn_date: str) -> str:
    """Validate the gate's resolved event date against the turn date; fall back to the
    turn date on any failure. ``turn_date`` and the return value are YYYY-MM-DD.

    Mirrors event_extract_v2.resolve() (the write-side fix validated on LongMemEval):
    the event date must be a parseable ISO date, not in the future (a 1-day skew is
    tolerated), and within 2 years back of the turn. A wrong resolution outside that
    window collapses to the turn date rather than poisoning the timeline order."""
    if not date_str or not _ISO_DATE_RE.match(date_str.strip()):
        return turn_date
    ds = date_str.strip()
    try:
        d = datetime.strptime(ds, "%Y-%m-%d")
        t = datetime.strptime(turn_date, "%Y-%m-%d")
    except ValueError:
        return turn_date
    # events lie in the past, within 2 years of the turn
    if d > t + timedelta(days=1) or d < t - timedelta(days=730):
        return turn_date
    return ds


def extract_idents(fact: str) -> list[str]:
    return _IDENT_RE.findall(fact.lower())


_MIN_CONTENT = 80  # turns shorter than this can't contain a happening worth keeping

GATE_PROMPT = """You are building a personal TIMELINE from ONE turn of a chat/assistant session (the user directs; an AI agent may execute work). Decide if anything HAPPENED this turn worth a permanent dated timeline entry, and if so write each happening as a naked past-tense event (up to 3).

These events are the ONLY record the timeline keeps of this turn — a happening you don't capture is forgotten. The user directs and is authoritative about what they decided or want, so a user assertion counts even when phrased casually; but a QUESTION, request, or instruction with no outcome yet is NOT a happening.

EMIT one event per concrete happening:
- WORK: a decision reached, an action carried out (usually visible in the [tool:...] activity lines — code written, a command run, a commit, a deploy, a bug fixed), a result/finding, or a milestone / state change.
- LIFE: something from the user's OWN life the user reports as already having happened — did, attended, visited, bought, sold, received, started, finished, achieved, experienced. Including things that happened on an earlier day than this turn.
DO NOT emit for: questions, requests or instructions with no outcome yet, opinions, brainstorming or design talk with NO decision reached, future plans or intentions, greetings, status checks, acknowledgements, or anything the assistant (not the user) relates. Most turns are discussion — return an empty list for them.

Write each `event` under these hard rules:
1. NAKED + VERB-HONEST. No leading name. Start with the past-tense verb, and let the VERB carry who-did-what: "decided / chose / approved / rejected" for a DECISION or direction; "committed / shipped / deployed / fixed / built / wrote / ran / added" for an ACTION carried out. (e.g. "decided episodic and semantic memory live in separate stores"; "finished a 5K run in 27 minutes 12 seconds".)
2. SELF-CONTAINED. It must make sense a month from now with NO other context. Name the concrete thing. NEVER "the bug above", "that approach" — spell it out.
3. Start with a LOWERCASE verb (it's a log line, not a sentence).
4. NO third-party personal names (clients, transcript subjects, other people's data) — describe generically ("8 client transcripts", not the names). Project/tool/service names are fine.
5. EXACT NUMBERS VERBATIM. Keep quantities, prices, times, distances, and scores exactly as stated ("27 minutes 12 seconds", "$1,450", "20%") — never round, convert, or paraphrase them.

DATE. This turn's date is given below. Most events happen the day they're discussed — leave `date` OUT for those. Only when the user reports something that ALREADY happened on a DIFFERENT day set `date` to the resolved calendar date (YYYY-MM-DD): resolve an absolute mention ("on May 3rd" -> that date, this turn's year unless stated) or a relative one ("last Tuesday", "two weeks ago", "this past weekend") against this turn's date. When the date you set differs from this turn's date, KEEP the user's original timing phrase in the event text verbatim so a wrong resolution stays auditable (e.g. "deployed the search reindex (reported as 'last Tuesday')"). Omit `date` when the event happened this turn or the timing can't be inferred.
If the event text itself names a FURTHER date that is NOT the event's own timing (a deadline, a scheduled-for day), anchor it inline with its resolved absolute date in parens: "scheduled the cutover for January 31 (meaning 2026-01-31)" (this turn's year unless stated).

salience: 2 = milestone / shipped to prod / major decision or life event; 1 = a normal action, decision, or happening; 0 = minor or routine.
event_type: "decision" (a choice/direction was reached), "action" (something was executed or done), "finding" (a result/diagnosis/measurement was learned), or "milestone" (a phase completed / shipped / achieved).
domain: "personal" = the user's OWN life outside engineering work — health, medication, family, appointments, purchases, home, travel, mood, errands, job applications and career moves. "technical" = code, infrastructure, homelab, deployments, benchmarks, research, tooling. Judge by what the event is ABOUT, not who executed it (an agent booking the user's appointment is still personal).

Output ONLY JSON: {"events": [{"event": "<naked past-tense fact>", "salience": 0|1|2, "event_type": "decision"|"action"|"finding"|"milestone", "domain": "personal"|"technical", "date": "YYYY-MM-DD" (optional — omit unless the event happened on a different day)}]} — at most 3 events, ordered by importance. Most turns: {"events": []}
"""


_MAX_EVENTS_PER_TURN = 3

# ---------------------------------------------------------------------------
# Write-time dedup (schema 037): a re-told happening merges into its existing row.
#
# Validated 2026-07-04 on 120 stratified real pairs vs a full-context referee:
# snippet-only judging is unreliable (79%) because the discriminating info lives
# in the SOURCE TURNS ("cleaned up segments" = N distinct jobs with identical
# event text). The surviving form — both full turns in the prompt + both-orders
# SAME consensus — leaves ~2% true false-merge, all recoverable (non-destructive
# reported_count bump, no deletes). See workspace/timeline-dedup-decision.md.
# ---------------------------------------------------------------------------

_DEDUP_WINDOW_DAYS = 14
_DEDUP_MAX_DIST = 0.20
_DEDUP_TURN_CHARS = 2500

TIMELINE_DEDUP_PROMPT = """Two dated event rows were extracted (by an automated gate) from two conversation turns for a personal timeline. Decide if they record the SAME single real-world happening (one occurrence, told or restated twice) or two DISTINCT happenings (the same kind of thing genuinely occurring again, or different things).

Rules:
- Recorded dates are UNRELIABLE: date resolution is per-turn and error-prone, so the SAME happening frequently appears under two different dates. Never treat differing dates alone as evidence of DISTINCT.
- Arc evolution is DISTINCT: a diagnosis and its later fix, a decision and its later revision, an implementation and its replacement are separate happenings even when they concern the same object.
- Recurrence is DISTINCT: the same chore or action genuinely done again on a different occasion (e.g. after two different restarts) is a new happening.
- The full source turn of each event is included — use it as ground truth for what actually happened.

EVENT A (dated {da}): {a}
SOURCE TURN A:
{ea}

EVENT B (dated {db}): {b}
SOURCE TURN B:
{eb}

Answer with exactly one word: SAME or DISTINCT."""


def _parse_verdict(text: str) -> str:
    t = text.strip().upper()
    if t.startswith("SAME"):
        return "SAME"
    if t.startswith("DISTINCT"):
        return "DISTINCT"
    raise MalformedResponseError("expected SAME or DISTINCT", text[:100])


def _parse_gate(text: str) -> list[dict[str, Any]]:
    """Parser for parse_with_retry: {"events": []} -> [], else validated dicts (capped).

    Also accepts the legacy single-event shape ({"event": ...}) so a model that
    regresses to the old contract degrades to one event instead of a retry loop."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise MalformedResponseError("no JSON object in response", text[:200])
    d = json.loads(text[start : end + 1])
    raw = d.get("events")
    if raw is None:
        raw = [d] if d.get("event") else []
    if not isinstance(raw, list):
        raise MalformedResponseError("'events' is not a list", text[:200])
    out: list[dict[str, Any]] = []
    for item in raw[:_MAX_EVENTS_PER_TURN]:
        if not isinstance(item, dict):
            continue
        ev = item.get("event")
        if not ev:
            continue
        sal = item.get("salience", 1)
        et = item.get("event_type")
        raw_date = item.get("date")
        # Shape only here (a non-empty string); range/parse validation is _resolve_event_date's job.
        date = raw_date.strip() if isinstance(raw_date, str) and raw_date.strip() else None
        dom = item.get("domain")
        out.append(
            {
                "event": str(ev).strip(),
                "salience": sal if isinstance(sal, int) and 0 <= sal <= 2 else 1,
                "event_type": et if et in ("decision", "action", "finding", "milestone") else None,
                # Invalid/missing -> None (unlabeled fails OPEN at read; a wrong default
                # would fail closed and hide the event from personal-scope serving).
                "domain": dom if dom in ("personal", "technical") else None,
                "date": date,
            }
        )
    return out


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
        self.dedup_enabled = os.environ.get("SYNAPSE_TIMELINE_DEDUP", "1") != "0"

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

        # The turn's own date (episodes.created_at, MAX over the window) — the anchor
        # the gate resolves relative timing against, and the default t_valid. Fetched
        # BEFORE the gate call so the prompt can carry it; never ingest wall-clock.
        turn_ts = self._db.get_episodes_valid_at([int(episode_id)])
        if not turn_ts:
            return
        turn_date = turn_ts[:10]  # YYYY-MM-DD

        events = parse_with_retry(
            self._llm_client,
            base_prompt=(
                f"{GATE_PROMPT}\nThis turn happened on {turn_date}.\n\nTHE TURN:\n{content[:6000]}"
            ),
            parser=_parse_gate,
            model=self._model,
            max_tokens=512,
        )
        if not events:
            return

        project = item.get("project")
        # One embed round-trip for the whole turn's events.
        vecs = self._embedder.embed(
            [f"Project: {project or '-'} | {e['event']}" for e in events], task="document"
        )
        for k, (gate, vec) in enumerate(zip(events, vecs, strict=True), start=1):
            # When the gate resolved the event to a DIFFERENT past day (something the user
            # reports as already-happened), stamp t_valid to that day at noon UTC; otherwise
            # keep the precise turn timestamp. _resolve_event_date validates + clamps first.
            resolved_date = _resolve_event_date(gate.get("date"), turn_date)
            t_valid = turn_ts if resolved_date == turn_date else f"{resolved_date}T12:00:00+00:00"

            # Write-time cross-source dedup: a bare commit/merge ANNOUNCEMENT whose PR-ref/SHA
            # is already on the timeline (git is canonical for those) adds nothing — skip it.
            # Deploys/fixes/decisions about the same ref are distinct happenings and still write.
            # Ordering caveat: chat often ingests BEFORE the git feeder pushes, so this catches
            # only chat-after-git; the read-time identifier-collapse in serving covers the rest.
            verb = gate["event"].split(None, 1)[0].lower().rstrip(":,")
            idents = extract_idents(gate["event"])
            if idents and verb in _ANNOUNCE_VERBS:
                if self._db.timeline_ident_exists(idents, project, t_valid, _XDEDUP_WINDOW_HOURS):
                    logger.info("timeline gate skip (ident dup %s) ep:%s", idents, episode_id)
                    continue

            if self.dedup_enabled and self._merge_if_retold(
                gate["event"], vec, project, t_valid, episode_id, content, turn_date
            ):
                continue

            self._db.insert_timeline_event(
                t_valid=t_valid,
                fact=gate["event"],
                source="chat",
                # "ep:<id>" for the first event keeps the historic key shape (idempotent
                # re-processing); "#k" suffixes let 2nd/3rd events coexist under the
                # UNIQUE(source, source_ref) constraint. Nothing parses source_ref at serve.
                source_ref=f"ep:{episode_id}" if k == 1 else f"ep:{episode_id}#{k}",
                project=project,
                salience=gate["salience"],
                embedding=vec,
                # model_name is set by every factory-built embedder; the getattr
                # fallback covers test stubs that only implement embed().
                embed_model=getattr(self._embedder, "model_name", None) or "voyage-4-large",
                event_type=gate.get("event_type"),
                domain=gate.get("domain"),
            )
            logger.info(
                "timeline event (s%d) from ep:%s: %s",
                gate["salience"],
                episode_id,
                gate["event"][:80],
            )

    # ------------------------------------------------------------------
    # Dedup confirm-merge
    # ------------------------------------------------------------------

    def _merge_if_retold(
        self,
        event: str,
        vec: list[float],
        project: str | None,
        t_valid: str,
        episode_id: Any,
        turn_content: str,
        turn_date: str,
    ) -> bool:
        """True if ``event`` re-tells an existing timeline row (that row is bumped).

        Fail-soft in the opposite direction from the gate itself: ANY error here
        means no merge and a normal insert — a surviving duplicate is cheaper than
        a lost happening. Merge requires SAME from the confirm call in BOTH
        presentation orders; a flip means uncertainty, which keeps the event."""
        try:
            cands = self._db.timeline_near_candidates(
                vec,
                project,
                t_valid,
                exclude_episode_ref=f"ep:{episode_id}",
                window_days=_DEDUP_WINDOW_DAYS,
                max_dist=_DEDUP_MAX_DIST,
            )
            if not cands:
                return False
            cand = cands[0]
            cand_turn = self._hydrate_turn(cand["source_ref"])
            if not cand_turn:
                return False
            a = (str(cand["t_valid"])[:10], cand["fact"], cand_turn[:_DEDUP_TURN_CHARS])
            b = (turn_date, event, turn_content[:_DEDUP_TURN_CHARS])
            if self._confirm_same(a, b) and self._confirm_same(b, a):
                self._db.bump_timeline_reported(cand["id"], t_valid)
                logger.info(
                    "timeline dedup merge: ep:%s '%s' -> event %s (reported_count+1)",
                    episode_id,
                    event[:60],
                    cand["id"],
                )
                return True
            return False
        except Exception as e:
            logger.warning(
                "timeline dedup check failed for ep:%s (inserting anyway): %s", episode_id, e
            )
            return False

    def _hydrate_turn(self, source_ref: str | None) -> str | None:
        """Full source-turn text behind a chat event's ``ep:<id>`` ref. The confirm
        call is only trustworthy with both turns in view — snippet-only judging was
        measured unreliable — so no turn means no merge."""
        m = re.match(r"ep:(\d+)", source_ref or "")
        if not m:
            return None
        row = self._db.get_episode(int(m.group(1)))
        return (row or {}).get("content")

    def _confirm_same(self, a: tuple[str, str, str], b: tuple[str, str, str]) -> bool:
        """One confirm call: (date, fact, turn) pair A vs B -> SAME?"""
        verdict = parse_with_retry(
            self._llm_client,
            base_prompt=TIMELINE_DEDUP_PROMPT.format(
                da=a[0], a=a[1], ea=a[2], db=b[0], b=b[1], eb=b[2]
            ),
            parser=_parse_verdict,
            model=self._model,
            max_tokens=8,
        )
        return verdict == "SAME"
