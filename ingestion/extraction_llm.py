from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ingestion.llm_client import MalformedResponseError
from ingestion.models import (
    CombinedExtraction,
    ExtractedEntity,
    ExtractedFact,
    ExtractionResult,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


_EXTRACTION_PROMPT = """\
These facts are the ONLY memory the system keeps from this conversation — anything you \
do not capture here is forgotten, so extract completely. The user is the authoritative \
source about their own work, life, and preferences: treat a user assertion as a fact even \
when phrased casually ("I'll just go with Postgres" -> a decision), but a user QUESTION is \
NOT an assertion — never mint a fact from something the user only asked about.

Session date: {session_date}. Resolve every dated mention in the facts against it.

Given the session summary and pre-identified entities below, extract any additional \
entities and the relationships between all entities.

Pre-identified entities:
{context_entities}

Session summary:
{summary}

Output ONLY valid JSON (no explanation, no markdown fence):
{{
  "entities": [{{"name": "...", "type": "...", "summary": "..."}}],
  "facts": [{{"source": "...", "target": "...", "relationship": "...", "fact": "..."}}]
}}

Rules:
- Entity types are open-ended — use whatever fits (Tool, Project, Decision, Issue, Config, etc.)
- fact text must be phrased as a self-contained searchable statement ("X uses Y for Z")
- Include rationale for decisions ("chose X because Y")
- Extract every distinct, durable, self-contained fact the summary genuinely
  contains — there is NO fixed number. Do not pad to reach a count, and do not omit a
  substantive fact to stay small. Let the content set the count: a thin turn yields a
  few, a dense one yields many.
- NEVER record transient or operational actions as facts. A fact must be reusable
  knowledge that stays true beyond this session — a decision and its rationale, a
  config value, a root cause, a relationship between components. BAD (never emit):
  "the inbox was checked and found empty", "discord-send.sh was used to post the
  briefing", "the calendar was clear", "X was run". GOOD: "the briefing pipeline posts
  to Discord #briefings via discord-send.sh", "Voyage embeddings are 2048-dim".
- DATED HAPPENINGS ARE FACTS, not operational chatter. The ban above is for
  routine tool churn, NOT for events someone would later ask "when did..." about.
  DO emit first-person life events ("User attended their cousin's wedding as a
  bridesmaid", "User repotted the spider plant"), shipped milestones ("the notes
  store shipped to production"), and one-time occurrences with a date anchor
  ("User received a crystal chandelier from their aunt"). The line: "checked the
  inbox" recurs and teaches nothing; "gave spider plant cuttings to Mrs. Johnson"
  happened once and is the answer to a future when-question.
- PLANNED IS NOT DONE. Never assert completion of an action the text only plans,
  books, schedules, requests, intends, or lists as a component of a plan, workup,
  agenda, or checklist. A noun phrase naming a plan's parts ("the 2026 workup
  (MRI, blood panel, referral)") describes INTENDED work: extract the plan AS a
  plan, keeping the pending status in the fact text. Emit "was performed / was
  done / was sent / underwent" ONLY when the text states the action already
  happened. The meta-act is separate and may itself be complete: booking an MRI
  is a completed BOOKING ("the MRI was booked for June 16"), not a completed MRI.
  BAD:  fact "An MRI was performed as part of User's mother's memory workup"
        (source only listed the MRI as a planned workup component)
  GOOD: fact "An MRI is a planned component of User's mother's memory workup,
        booked for June 16 (meaning 2026-06-16)"
  When completion status is genuinely ambiguous in the text, phrase the fact with
  the weaker claim (planned/underway), never the stronger one.
- RECOMMENDED IS NOT ADOPTED. A recommendation, an option still being weighed, or a
  proposal explicitly parked ("separate issue, can park", "could do X later") is NOT
  a decision and NOT an implementation. Mint an adoption or shipped fact only when
  the text says the choice was made or the change landed. If the text floats
  mechanism X but never confirms X shipped, never state X as the implemented
  mechanism — what actually shipped may use something else entirely.
  BAD:  fact "The Dependency entity type was merged into Tool because the
        library/tool distinction is fuzzy"
        (the merge was only RECOMMENDED; the final config chose differently)
  GOOD: fact "Merging the Dependency entity type into Tool was recommended because
        the library/tool distinction is fuzzy in practice"
  Proposal facts ARE worth keeping — phrase them as proposals ("was proposed", "was
  recommended", "was parked as a follow-up"), never as the stronger claim.
- POLARITY AND BINDING ARE SACRED. Never mint a fact whose polarity contradicts an
  explicit negation or refusal in the text. "Don't change the chunk sizes" can only
  yield a fact that the change was REFUSED, never a chunk-size-change fact — a topic
  being discussed is not evidence the action happened. And bind every outcome to the
  exact subject the text ties it to: with several PRs, branches, or components in
  play, do not attach a merge, deploy, or fix to a nearby but wrong one. If the text
  does not clearly bind the outcome, drop the identifier rather than guess.
  BAD:  fact "Chunk size was reduced to resolve MAX_TOKENS truncation errors"
        (the user said "don't change chunk sizes")
  GOOD: fact "User ruled out changing chunk sizes while fixing MAX_TOKENS truncation
        errors"
  BAD:  fact "PR #12 was deployed to production"
        (PR #12 only changed a client-side plugin file; the deploy was other work)
  GOOD: fact "PR #12 changed a client-side plugin file"
- EDGES CONNECT THE REAL PARTICIPANTS — the user is NOT the default source. The user
  narrates nearly every turn, so routing every happening through a "User" node collapses
  the graph into one hub that no longer discriminates. When an event or statement has a
  real counterparty or object (a person, org, place, thing), the EDGE joins THOSE, and
  the user survives in the `fact` TEXT.
  BAD:  source "User", target "Mrs. Johnson", GAVE
        fact "User gave Mrs. Johnson spider plant cuttings on 2023-03-18 (meaning 2023-03-18)"
  GOOD: source "spider plant cuttings", target "Mrs. Johnson", GIVEN_TO
        fact "User gave Mrs. Johnson spider plant cuttings on 2023-03-18 (meaning 2023-03-18)"
  BAD:  source "User", target "voyage-4-large", SWITCHED_TO
        fact "User switched the embedder to voyage-4-large, replacing voyage-3"
  GOOD: source "synapse", target "voyage-4-large", USES_EMBEDDER
        fact "User switched synapse's embedder to voyage-4-large, replacing voyage-3"
- A DECISION, APPROVAL, REQUEST, PURCHASE, or CLAIM IS A DOING — same rule. Connect
  WHAT WAS CHOSEN to WHAT IT AFFECTS (or what it replaced), and never invent an
  abstraction node ("the X decision", "the plan", "the approach") to receive the edge.
  BAD:  source "User", target "ship decision", DECIDED
        fact "User reversed and greenlit shipping episode rerank to production on 2026-06-03"
  GOOD: source "episode rerank", target "production recall()", SHIPPED_TO
        fact "User greenlit shipping episode rerank into production recall() on 2026-06-03
        (meaning 2026-06-03), reversing an earlier hold-as-findings-only decision"
  BAD:  source "User", target "MokerLink 2G05110GSM switch", BUYING
        fact "User is buying the MokerLink 2G05110GSM switch for $40 CAD"
  GOOD: source "MokerLink 2G05110GSM switch", target "homelab network", PURCHASED_FOR
        fact "User is buying the MokerLink 2G05110GSM switch for $40 CAD for the homelab network"
- REIFY a participant-less happening as its own event entity. When the happening has no
  real counterparty to connect (a trip, a wedding, an appointment, a milestone), mint a
  short noun-phrase entity for the EVENT itself and hang its participants, roles, dates,
  and attributes off THAT node — one fact per attribute.
  BAD:  source "User", target "cousin's wedding", ATTENDED
        fact "User attended their cousin's wedding as a bridesmaid on 2025-06-14"
  GOOD: source "cousin's wedding", target "bridesmaid", HAD_ROLE
        fact "User was a bridesmaid at their cousin's wedding on 2025-06-14"
        plus e.g. source "cousin's wedding", target "Niagara-on-the-Lake", HELD_AT
        fact "User's cousin's wedding was held in Niagara-on-the-Lake on 2025-06-14"
- THE FACT TEXT LOSES NOTHING. Rerouting an edge NEVER removes the user or the date from
  the sentence — the text is what retrieval searches. Every fact about something the user
  did, owns, received, or attended still SAYS so ("User ...", "User's ..."), and still
  carries its date exactly as the date rule requires. A fact that reads "spider plant
  cuttings were given to Mrs. Johnson" with the user erased is a FAILED extraction.
- The user IS still a legitimate `source`/`target` for a STANDING PROPERTY of the user —
  something that is true of them between sessions, with no date and no event behind it:
  a preference, trait, condition, skill, employment, or human relationship ("User has
  ADHD", "User prefers Postgres over MySQL", "User is Mrs. Johnson's neighbour", "User
  works as a Security Playbook Engineer"). Do not contort those into object-first edges.
  This carve-out does NOT cover anything that HAPPENED — a decision, approval, request,
  purchase, report, complaint, assertion, action item, or a status the user is merely
  waiting on. If the fact would carry a date, it is a happening: reroute it.
  BAD:  source "Coinbase application", target "User", PENDING
  GOOD: source "Coinbase application", target "Coinbase", AWAITING_REPLY_FROM
        fact "User's Coinbase application was ~4 days old with no reply as of 2026-05-01
        (meaning 2026-05-01)"
  ACQUIRING something is not the same as HAVING it: a purchase, gift, delivery, or any
  one-time acquisition is a DOING, even though it leaves the user owning a thing. Same
  for a complaint or a symptom report — the user reporting it is an event, and the
  standing condition (if there is one) is a separate fact.
  BAD:  source "User", target "Carex Day-Light Elite", PURCHASED
        fact "User purchased a Carex Day-Light Elite lamp on 2026-01-08"
  GOOD: source "Carex Day-Light Elite", target "seasonal affective disorder", BOUGHT_FOR
        fact "User purchased a Carex Day-Light Elite light therapy lamp on 2026-01-08
        (meaning 2026-01-08) to treat seasonal affective disorder"
  BAD:  source "User", target "left knee pain", REPORTED
  GOOD: source "left knee pain", target "half marathon", STARTED_AFTER
        fact "User reported left knee pain starting after the half marathon on 2026-03-02
        (meaning 2026-03-02)"
  The rule bans the user as a HUB for doings and happenings, not as an entity.
- PRESERVE EXACT QUANTITIES. A fact carrying a number, price, percentage, count, or
  duration MUST keep the figure VERBATIM in the fact text ("Women hold 20% of
  leadership positions at User's company", "User spent $120 on a helmet"). Never
  paraphrase a figure and never convert it — "20%" must NOT become "20 people" or
  "20 women". If one statement carries several distinct quantities, emit one fact
  per figure rather than folding them together.
- NAME WHAT CHANGED. When a fact records a switch, upgrade, migration, or a replaced
  choice/config/state, name what it REPLACED in the fact text — "switched the embedder to
  voyage-4-large, replacing voyage-3", not "uses voyage-4-large". Same for moves and
  cancellations ("moved the prod compose to /opt/docker/synapse, no longer at
  ~/synapse"). Naming the superseded value is what lets a later reader tell
  current state from stale.
- ANCHOR EVERY DATE. When a fact's text carries a dated reference, append its resolved
  absolute date in parentheses right after the original wording. Resolve an ABSOLUTE
  mention against the session date's year unless the text states one ("my flight is
  January 31" -> "... January 31 (meaning 2026-01-31)"); resolve a RELATIVE mention against
  the session date ("two days from now", "last Tuesday" -> that calendar date, e.g.
  "(meaning 2026-01-17)"). Keep the user's original wording and add "(meaning YYYY-MM-DD)"
  after it. If the session date is unknown or the reference is too vague to pin
  ("recently", "a while ago"), leave it unanchored. When one statement carries SEVERAL
  dated events, SPLIT it into one fact per event so each fact carries exactly one date.

ENTITY NAMES are short noun phrases (5 words or fewer) naming a discrete thing in the
user's world — never a full sentence, goal statement, action item, or quoted phrase.
For "decided to expand the recall bench to 200 questions", the entity is "recall
bench", not the whole proposition. NEVER mint an entity from:
- a clock time or bare date ("5:54 am", "January 15") — dates ride inside fact text
- a bare quantity, price, percentage, or duration ("7-8 hours", "$150/hour", "20%")
  — figures are properties of the thing they measure, keep them in the fact
- an imperative tip-list header ("Buy in bulk", "Plan your meals") — name the topical
  noun ("bulk buying") if it's worth keeping at all
- a quoted slogan, idiom, or definitional phrase — not retrievable referents
CONNECT FACTS DIRECTLY between the people, projects, tools, and events involved — never
route a relationship through scenery or observation nouns. GOOD: "synapse -> MIGRATED_TO
-> Postgres 17". BAD: "the migration log -> MENTIONS -> synapse". Descriptive detail
belongs inside the fact text of the direct edge.

OUTPUT DISCIPLINE:
- `fact` and `summary` are plain declarative statements. NEVER include hedging
  ("appears to", "possibly"), your own reasoning, alternatives you considered, or
  text echoed from these instructions or the output schema.
- If you have nothing real for a `summary`, use "" — never a sentence explaining
  the absence ("no additional information available").
- `relationship` is a short UPPER_SNAKE_CASE label ("USES", "DECIDED_ON"), never a
  sentence.

ENTITY/FACT CONSISTENCY (strict — orphan entities are dropped on the server):
- Every entity you list in `entities` MUST appear as the `source` or `target` of at
  least one fact. Every fact's `source` and `target` MUST exactly match the `name`
  of an entity declared in `entities`. If you can't write a meaningful fact about
  an entity, don't list it.
"""


_WEB_EXTRACTION_PROMPT = """\
The text below is an excerpt from {source_kind}. Extract entities and the \
relationships between them.

Source: {source_desc}
Pre-identified entities:
{context_entities}

Excerpt:
{summary}

Output ONLY valid JSON (no explanation, no markdown fence):
{{
  "entities": [{{"name": "...", "type": "...", "summary": "..."}}],
  "facts": [{{"source": "...", "target": "...", "relationship": "...", "fact": "..."}}]
}}

Rules:
- ATTRIBUTION FIREWALL (strict): this is third-party content. NEVER phrase a fact
  as something the user said, did, decided, owns, or prefers. The source
  makes claims; the user does not appear in them. BAD: "the user uses Trafilatura",
  "the chosen approach is X". GOOD: "Trafilatura achieves F1 0.93 on boilerplate
  removal benchmarks", "Mem0 removed its graph backend in its v3 rewrite".
- Entity types: use ONLY these — Person, Organization, Product, Technology,
  Technique, Benchmark, Publication, Event, Location. Pick the closest fit;
  do not invent new types.
- SALIENCE BAR: extract only entities that would plausibly matter beyond this one
  page — named tools, products, techniques, organizations, people, measured results.
  Skip page furniture (bylines, categories, related-article titles), code variable
  names, and generic concepts ("performance", "users", "the team"). Bare quantities,
  clock times, and coordinates are never entities — a figure like "F1 0.93" or
  "7.5 hours" stays inside the fact text of the thing it measures.
- fact text must be a self-contained searchable claim ("X achieves Y on Z",
  "X replaced Y because Z"). Include numbers, versions, and dates verbatim when
  the source states them.
- Extract every distinct, durable claim the excerpt genuinely contains — no fixed
  count. A navigation-heavy excerpt may yield zero; return empty lists rather than
  padding.
- NEVER record transient page events ("the article lists 10 tools", "the post was
  updated") — only reusable knowledge claims.

ENTITY/FACT CONSISTENCY (strict — orphan entities are dropped on the server):
- Every entity in `entities` MUST appear as the `source` or `target` of at least
  one fact, and every fact's `source`/`target` MUST exactly match a declared
  entity name.
"""


class LLMExtractor:
    """Extract entities + facts from session summaries via LLM call."""

    # Up to 2 retries (3 total attempts) on malformed JSON, matching the spec.
    _MAX_ATTEMPTS = 3

    def __init__(self, llm_client: Any, model: str = "claude-haiku-4-5") -> None:
        self._client = llm_client
        self._model = model

    def extract(
        self,
        summary: str,
        context_entities: list[ExtractedEntity],
        session_date: str | None = None,
    ) -> ExtractionResult:
        """Run the extractor LLM, then validate + cross-ref the response.

        ``session_date`` (YYYY-MM-DD, the segment's conversation date) is the
        anchor the prompt resolves in-text date mentions against; None renders
        "unknown" and the prompt leaves relative dates unanchored.

        Two retry concerns compose at separate layers inside
        ``structured_call``:

        * Transient wire errors (rate-limit, timeout) → tenacity around the
          pydantic-ai run (see ``ingestion.llm_client._run_agent_sync``).
        * Malformed / schema-violating output → pydantic-ai's
          retry-with-validation-feedback loop (``max_attempts`` total tries).

        The load-bearing Phase 1 logic — the Pydantic ``CombinedExtraction``
        cross-reference validator that drops facts referencing undeclared
        entities — survives untouched in ``_run`` below.
        """
        context_str = (
            ", ".join(f"{e.name} ({e.type})" for e in context_entities)
            if context_entities
            else "none"
        )
        base_prompt = _EXTRACTION_PROMPT.format(
            context_entities=context_str,
            summary=summary,
            session_date=session_date or "unknown",
        )
        return self._run(base_prompt)

    def extract_web(
        self,
        content: str,
        context_entities: list[ExtractedEntity],
        provenance: dict[str, Any],
    ) -> ExtractionResult:
        """Extract from third-party web content (task #68).

        Same parse/retry machinery as ``extract``, different prompt: the web
        variant carries the attribution firewall (claims belong to the source,
        never to the user), a closed entity-type vocabulary, and a salience
        bar — third-party pages are noisier and less trustworthy substrate
        than the user's own conversation chunks.
        """
        context_str = (
            ", ".join(f"{e.name} ({e.type})" for e in context_entities)
            if context_entities
            else "none"
        )
        synthesized = bool(provenance.get("synthesized"))
        if provenance.get("kind") == "research_brief":
            source_kind = "a multi-source research brief compiled by an AI assistant"
        elif synthesized:
            source_kind = "an AI-generated answer about a web page (secondhand, not the raw page)"
        else:
            source_kind = "a scraped web page"
        source_desc = " | ".join(
            str(v)
            for v in (
                provenance.get("title"),
                provenance.get("url"),
                provenance.get("published_at") or provenance.get("fetched_at"),
            )
            if v
        )
        base_prompt = _WEB_EXTRACTION_PROMPT.format(
            source_kind=source_kind,
            source_desc=source_desc or "unknown",
            context_entities=context_str,
            summary=content,
        )
        return self._run(base_prompt)

    def _run(self, base_prompt: str) -> ExtractionResult:
        from ingestion.llm_client import structured_call
        from ingestion.llm_schemas import ExtractionOutput

        try:
            output = structured_call(
                self._client,
                output_model=ExtractionOutput,
                base_prompt=base_prompt,
                model=self._model,
                # No fixed fact cap (content sets the count) + the anti-filler
                # quality bar; dense turns can run 15-25 facts. Dense infra
                # chunks proved 3072 too small — outputs truncated mid-array
                # at ~12K chars and failed all parse retries (2026-07-17), so
                # the headroom is 8192; a truncated tail silently drops the
                # last facts.
                max_tokens=8192,
                max_attempts=self._MAX_ATTEMPTS,
            )
        except MalformedResponseError as exc:
            # The model ANSWERED and the answer never validated — a content failure, so
            # an empty result is a fair reading of it. The other case (no output at all:
            # unrecognised model id, dead auth) arrives as LLMUnavailableError and is NOT
            # caught here: nothing was decided, so the item must fail and be retried.
            logger.warning(
                "LLM extraction failed after %d attempts (malformed JSON): %s",
                self._MAX_ATTEMPTS,
                exc,
            )
            return ExtractionResult()

        # Cross-reference validation (facts pointing at undeclared entities
        # are dropped, never raised) — the load-bearing Phase 1 invariant.
        combined = CombinedExtraction(
            entities=[
                ExtractedEntity(name=e.name, type=e.type, summary=e.summary)
                for e in output.entities
            ],
            facts=[
                ExtractedFact(
                    source=f.source,
                    target=f.target,
                    relationship=f.relationship,
                    fact=f.fact,
                )
                for f in output.facts
            ],
        )

        if combined.dropped_facts:
            logger.info(
                "Dropped %d fact(s) referencing entities not in extractor output",
                len(combined.dropped_facts),
            )
        if combined.dropped_entities:
            logger.debug(
                "Dropped %d entity row(s) during validation (empty, over-cap, or "
                "policy-banned name)",
                len(combined.dropped_entities),
            )

        return ExtractionResult(entities=combined.entities, facts=combined.facts)


# ---------------------------------------------------------------------------
# Stage 4 — Entity resolver
# ---------------------------------------------------------------------------
