"""Who may read a KG fact: the audience label on ``kg_relationships`` (schema 066).

A restricted surface (a work laptop) is served a KG fact only when it is labelled
``work-safe`` (or, until the backfill reaches it, unlabelled and older than 066: the legacy
clause in :func:`mcp_server.kg_pg.scope_predicate`). The label is about SENSITIVITY, not
project: the project check (schema 056) still gates every restricted read, and this is a
second, independent key on top of it. The vocabulary is the notes and web one
(:data:`ingestion.surfaces.AUDIENCES`); the prompt says "private" and the column stores
``personal``, and :func:`to_audience` is the only place that maps one to the other.

The poller's maintenance cycle runs :func:`label_pending`, a clone of
:func:`ingestion.web_audience.label_pending` for facts rather than documents:

* **Only while it matters.** It runs only while an approved restricted device is registered
  that can be served KG facts (a non-empty allowlist), so a one-device install pays nothing.
* **Newest first, in tiers.** (1) live facts a registered restricted device can reach today
  (the privacy win: those are the ones a work laptop is served unlabelled); (2) every other
  live fact; (3) retired facts. Retired facts matter because the supersession legs need both
  sides of a pair to pass the rule, and serve the retired side's text.
* **Batched.** Many facts per call, one line each, on :func:`stage_model` ``FACT_AUDIENCE``.
  The reply is a list of labels in input order, and it counts only if it names every line
  exactly once: a line the model skipped would shift every later label onto the wrong fact,
  so a reply that is short, repeats a number or does not validate is split in halves and
  retried, down to single facts. One bad reply neither mislabels a batch nor blocks the queue.
* **A dead backend labels nothing.** Any other failure of a call (a spawn or connection
  error, a 4xx, an empty completion) is split the same way, because one fact's content can
  cause it. A fact that still fails alone is written ``personal`` with ``label_model``
  ``error:<model>`` (so it can be found and relabelled), but only once a later call in the
  same cycle has succeeded: a run of failures with none after it is the backend's, and ends
  the cycle with those rows still NULL.
* **Phase 1 is one vote.** ``work-safe`` is any single call that names the fact work-safe and
  nothing else; the provenance rule still gates, so this is a cut, not the only wall. Each
  label is stamped with :data:`LABEL_VERSION` and the model, so a later phase can ask for
  unanimous votes, or relabel a version, without a migration. One vote is the weak point
  for private thoughts about the user's own employer written in technical words: on a
  held-out sample at batch 50 one call on the default stage model caught them about half
  the time (health mentions every time), one call on a stronger model every time.
  ``SYNAPSE_FACT_AUDIENCE_MODEL`` is the dial; a missed fact is exactly as visible as it
  was before 066.

NULL means "not labelled yet". The write never replaces a label (``audience IS NULL``): a
concurrent labeller or a human decision wins, and so does the ``personal`` v0 label every fact
from the old personal pile carries (schema 066 stamped it at insert, 067 kept it when it merged
the piles), so this stage never sees those. A usage-limit stop, or a backend that keeps failing, ends the cycle quietly (what was
written stays written, the rest stays NULL) and the poller sits the stage out for a while.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Literal

from psycopg.rows import dict_row
from pydantic import BaseModel

from ingestion.llm_client import UsageLimitError, stage_model, structured_call
from ingestion.surfaces import DEFAULT_AUDIENCE, FACT_AUDIENCE_PROBE, RESTRICTED_AUDIENCE

logger = logging.getLogger(__name__)

#: Pipeline stage name for :func:`ingestion.llm_client.stage_model`
#: (``SYNAPSE_FACT_AUDIENCE_MODEL`` beats ``SYNAPSE_LLM_MODEL``).
STAGE = "FACT_AUDIENCE"

#: Stamped into ``label_version`` with every label this stage writes. Tied to the wording of
#: :data:`PROMPT`: bump it on any change to the rules, so labels written under the old
#: wording can be told apart (schema 066 reserves 0 for its own seed and trigger).
LABEL_VERSION = 1

#: Facts per classifier call (``SYNAPSE_FACT_AUDIENCE_BATCH``, clamped to 1..:data:`MAX_BATCH`).
#: Agreement between runs was measured on short batches; a longer one lets a fact's text
#: steer its neighbours, so this is the knob to lower if the labels drift.
DEFAULT_BATCH = 50
MAX_BATCH = 200

#: Calls per full maintenance cycle (``SYNAPSE_FACT_AUDIENCE_CALLS``, at least 1). Each takes
#: tens of seconds, and the cycle that runs this stage also owns the drain's chunk and embed
#: work, so the stage takes a bounded slice (``batch * calls`` facts) and leaves the rest.
DEFAULT_CALLS = 2

#: A fact is one sentence; the cap only guards a malformed row from swamping a batch.
FACT_CHARS = 1000

#: ``label_model`` prefix of a ``personal`` fallback written for a fact no verdict could be
#: had for (schema 066): ``error:<model>``.
ERROR_MODEL_PREFIX = "error:"

Connect = Callable[[], AbstractContextManager[Any]]

# The classifier's rules, one fact per line, no other context. ``{facts}`` is replaced (not
# ``str.format``: the reply example's braces would break it). P5 names the private address
# ranges as a class rather than listing them. The reply is a JSON object, which strict
# structured output needs for its root. Any wording change is a new LABEL_VERSION.
PROMPT = """You label facts stored in a personal AI assistant's long-term memory. Each fact was \
extracted from conversations between ONE user and their assistant. "The user" may appear \
as "User", "the user", or a first name. You see one fact per line with no other context: \
judge only what the line says, and do not guess hidden context.

Give every fact exactly ONE label on ONE axis, "sensitivity". It answers one question: may \
this fact be shown on a RESTRICTED device? A restricted device is a machine tied to the \
user's job or school that the user does not fully control (an employer laptop, a hospital \
or school workstation, a screen colleagues or classmates can see). Topic does not decide \
the label; who may see the fact does.

- sensitivity:private = the user would not want an employer, colleague, teacher or \
classmate to see it.
- sensitivity:work-safe = harmless on that screen.

Check rules P1 to P6 in order. If ANY of them matches, the label is private. If none \
matches, the label is work-safe (W). If you are unsure whether a rule matches, answer \
private.

P1. Body, health and mind. Anything about the health, body, symptoms, diagnoses, \
medications, supplements, therapy, mental state, moods, sleep, diet, weight, fitness, \
alcohol or drug use, sex or sexuality of the user OR of a person in their private life.
  "User's knee has been sore since the hike" | "User sleeps about six hours on weeknights" \
| "User's father started a new blood-pressure medication"

P2. Private relationships. Any fact that mentions a person from the user's private life \
(partner, ex, date, child, parent, sibling, other relative, friend, neighbour, roommate) \
together with anything about that person, or the user's dealings with them: family events, \
conflicts, plans, what the person does, owns, uses, says or needs. Also the user's dating \
or romantic life.
  "User's sister is moving to Lisbon in spring" | "User and his partner argued about \
holiday plans" | "User's mother uses the tablet app the user set up to read the news"

P3. Career moves, as opposed to the work itself. Job searching, applications, recruiters, \
interviews with any employer (including technical interviews and coding tests), offers, \
salary or pay, raises, performance reviews, promotion cases, plans to leave a job, \
opinions or complaints about their own employer, manager or colleagues, private worries, \
suspicions or tactics about how their own work, mistakes or delays will look to their \
employer or manager (who will be blamed, how to steer the blame), a resume or \
portfolio, paid work outside their main job (freelancing, a side business), facts that \
size up any organisation as a place to work (its pay, on-call duties, work arrangements, \
culture, hiring process, job postings, employee ratings), and facts about the job market \
(which roles or skills are in demand, typical pay, hiring trends).
  "User has a panel interview at another clinic on Thursday" | "User's resume lists five \
years of litigation support" | "User feels her department head ignores her ideas" | \
"User is quietly logging which teammate caused each missed deadline" | \
"Northgate Bank's analyst roles require a take-home case study" | "Hospitals in the region \
are short of night-shift pharmacists"

P4. Personal life and money. The user's own leisure and private activities: hobbies, \
games, sport, music, books, films and shows they watch or make, travel and holidays, \
social plans, food they cook or like; their home, address, neighbourhood, car, pets, \
errands and household; their money: income, debts, budgets, bills, taxes, insurance, what \
they bought, are shopping for, comparing or considering buying. This includes technology \
whose purpose is one of these (a game, game-streaming or media setup, a fitness or sleep \
tracker, a budgeting app, a smart-home device that reveals the household's routine) and \
facts saying which products the user owns or uses for private life.
  "User is comparing two used e-readers under $150" | "User's game console downloads \
updates overnight" | "User rents a two-bedroom flat near the river" | "User's landlord is \
replacing the kitchen windows" | "User is learning to play the cello"

P5. Secrets and contact or account numbers. A password, access token, API key, private key \
or recovery code; a personal email address or phone number; a bank, card, licence, \
passport, tax or other government or financial account number; the public IP address of \
the user's home or a device's hardware (MAC) address. NOT P5: usernames, handles, \
repository or file paths, host and domain names, VPN or private network addresses \
(10.x.x.x and the other RFC 1918 private ranges, and 100.64-100.127.x), and descriptions \
of how a system stores or scopes credentials without stating the secret.
  "The router admin password is Tulip-4471" | "User's personal email is \
jo.reyes@example.com" | "The home connection's public IP is 203.0.113.7"

P6. General knowledge on a private subject. A fact about the world (not about the user) \
whose subject is medicine, drugs, supplements, physical or mental health, health-care \
providers and clinics (doctors, dentists, therapists, pharmacies), sex, dating or \
relationships, parenting, personal finance or debt, religion, politics or legal trouble, \
unless the fact itself shows it is part of the user's job or course of study. Keeping it \
in memory reveals what the user was looking into.
  "Sertraline can cause insomnia in the first weeks" | "Debt consolidation loans usually \
carry fees of 1-5%" | "Attachment styles shape how couples handle conflict"

W. Everything else is work-safe:
- the user's job or studies in any profession: tasks, cases, clients, patients, students, \
customers, colleagues' work roles, meetings, documents, deadlines and decisions, including \
the private details of the people the user serves in that job (a nurse's patient, a \
lawyer's client);
- technology of any kind: how software, computers, networks, servers, storage, AI tools \
and code are built, configured, behave, fail or get fixed, wherever they run (a home \
server, a personal laptop, a hobby coding project), as long as no P rule matches;
- organisations, products, public prices and offers (unless the fact says the user is \
considering them), places and buildings, science, history, nature, cooking and craft \
techniques, and other general knowledge not covered by P6;
- how the user wants their AI assistant to behave.
  "User charts patient vitals at the end of each shift" | "The client's custody hearing \
moved to June 2" | "User teaches two sections of Grade 9 algebra" | "User sent the \
quarterly variance report to her manager" | "M10 bolts on this mount are torqued to 45 Nm" \
| "The home server backs up the shared drive to an external disk every night" | "The \
nightly build failed because a dependency dropped support for an old Python version" | \
"Postgres VACUUM reclaims space held by dead rows" | "Bread dough rises faster in a warm \
kitchen" | "User wants answers without bullet points"

Do not let single words decide: "sleep" inside a program's retry loop is technology (W); a \
doctor's appointment booked through an app is P1. A fact that matches both a P rule and W \
is private.

Facts, one per line as <n>|<text>:
{facts}

Reply with ONLY a JSON object {"facts": [...]}, one entry per fact, in input order, \
nothing else:
{"facts": [{"n": 1, "labels": ["sensitivity:private"]}, {"n": 2, "labels": \
["sensitivity:work-safe"]}, ...]}"""


class FactLabel(BaseModel):
    """One fact's verdict, by its 1-based line number in the prompt."""

    n: int
    labels: list[Literal["sensitivity:private", "sensitivity:work-safe"]]


class FactAudienceReply(BaseModel):
    """The classifier's reply. An object root: strict structured output needs one."""

    facts: list[FactLabel]


@dataclass(frozen=True)
class FactVerdict:
    audience: str
    #: model | error (a single fact no call could label: ``personal`` by default)
    source: str = "model"


class ClassifierDown(Exception):
    """A run of classifier calls all failed: the backend, not one fact's content, is the
    problem. The cycle stops with nothing written for the chunk in hand."""


@dataclass
class CycleHealth:
    """What one cycle's classifier calls have shown about the backend. A call is a success
    when its reply lines up with the facts (:func:`_names_each_line_once`); anything else,
    an exception or a reply that does not, is a failure."""

    #: Failures in a row that end the cycle. Isolating one fact in a batch of N takes about
    #: log2(N) failing calls down the split, so this sits just above that; a backend that
    #: is down fails every call and reaches it.
    limit: int
    streak: int = 0
    #: Successful calls so far.
    answered: int = 0
    #: Each single fact that failed alone, with the success count at that moment. A fallback
    #: is only trusted once a call has succeeded since: then the failure was the fact's, not
    #: the backend's.
    failed_at: dict[int, int] = field(default_factory=dict)


def _fail_limit(batch: int) -> int:
    return max(1, batch).bit_length() + 2


def to_audience(labels: Sequence[str]) -> str:
    """The stored audience for a fact's labels: ``work-safe`` only for exactly the one
    work-safe label. A private label, none, or both is ``personal``: unsure is private."""
    return RESTRICTED_AUDIENCE if list(labels) == ["sensitivity:work-safe"] else DEFAULT_AUDIENCE


def render(rows: list[dict[str, Any]]) -> str:
    """The prompt's fact lines, numbered from 1. Whitespace is folded so a fact stays one
    line, and capped at :data:`FACT_CHARS`."""
    return "\n".join(
        f"{i}|{' '.join(str(r['fact']).split())[:FACT_CHARS]}" for i, r in enumerate(rows, start=1)
    )


def _names_each_line_once(reply: FactAudienceReply, n: int) -> bool:
    """True when the reply's numbers are exactly 1..``n``, each once. Anything else cannot
    be trusted to line up with the facts: the model echoes only ``n``, so a skipped or
    repeated number is the one sign that later labels belong to different facts."""
    return sorted(e.n for e in reply.facts) == list(range(1, n + 1))


def classify_batch(
    llm: Any,
    rows: list[dict[str, Any]],
    *,
    model: str | None = None,
    health: CycleHealth | None = None,
) -> dict[int, FactVerdict]:
    """Verdicts for ``rows`` (``id`` and ``fact``), by row id. A usage limit propagates (the
    caller stops there). A call that fails, or a reply that does not validate or does not
    name each line exactly once (:func:`_names_each_line_once`), is split in halves and each
    half classified on its own, down to single facts, so one bad reply or one fact that
    breaks a call costs a few calls, not 50 labels; a single fact that still fails is
    ``personal`` (source ``error``). ``health`` carries the cycle's failure streak: when
    :class:`ClassifierDown` is raised the backend is failing and nothing is returned."""
    if not rows:
        return {}
    health = health or CycleHealth(_fail_limit(len(rows)))
    why = "reply did not name each fact once"
    reply: FactAudienceReply | None
    try:
        reply = structured_call(
            llm,
            output_model=FactAudienceReply,
            base_prompt=PROMPT.replace("{facts}", render(rows)),
            model=model or stage_model(STAGE),
            max_tokens=4096,
            max_attempts=2,
        )
    except UsageLimitError:
        raise
    except Exception as e:
        reply, why = None, f"call failed ({str(e)[:120]})"
    if reply is not None and _names_each_line_once(reply, len(rows)):
        health.streak = 0
        health.answered += 1
        return {rows[e.n - 1]["id"]: FactVerdict(to_audience(e.labels)) for e in reply.facts}
    health.streak += 1
    if health.streak >= health.limit:
        raise ClassifierDown(f"{health.streak} classifier calls in a row failed: {why}")
    return _split(llm, rows, model, health, why)


def _split(
    llm: Any, rows: list[dict[str, Any]], model: str | None, health: CycleHealth, why: str
) -> dict[int, FactVerdict]:
    """The verdicts for ``rows`` after a call over them failed: each half on its own, or,
    for one fact, ``personal``."""
    if len(rows) == 1:
        logger.warning("fact %s did not classify (%s): personal", rows[0]["id"], why)
        health.failed_at[rows[0]["id"]] = health.answered
        return {rows[0]["id"]: FactVerdict(DEFAULT_AUDIENCE, "error")}
    half = len(rows) // 2
    return {
        **classify_batch(llm, rows[:half], model=model, health=health),
        **classify_batch(llm, rows[half:], model=model, health=health),
    }


# Every restricted device that can be served KG facts: the same rule as
# ingestion.db_notes.restricted_surface_projects, on the stage's own connection. An empty
# result means recall serves no KG facts to any restricted caller, so there is nothing to
# label for.
_RESTRICTED_PROJECTS_SQL = """
SELECT DISTINCT unnest(allowed_projects) AS project FROM surfaces
WHERE trust = 'restricted' AND status = 'approved'
"""

# Tier 1: live facts inside the union of the registered allowlists, newest first.
_TIER_REACHABLE_SQL = """
SELECT id, fact FROM kg_relationships
WHERE audience IS NULL AND web_artifact_id IS NULL AND t_invalid IS NULL
  AND fact IS NOT NULL AND fact <> ''
  AND source_projects IS NOT NULL AND cardinality(source_projects) > 0
  AND source_projects <@ %s::text[]
ORDER BY id DESC
LIMIT %s
"""

# Tier 2: the other live facts (not the ids tier 1 took).
_TIER_LIVE_SQL = """
SELECT id, fact FROM kg_relationships
WHERE audience IS NULL AND web_artifact_id IS NULL AND t_invalid IS NULL
  AND fact IS NOT NULL AND fact <> ''
  AND id <> ALL(%s::bigint[])
ORDER BY id DESC
LIMIT %s
"""

# Tier 3: retired facts. Their text is served in superseded pairs, and a pair needs both
# sides labelled.
_TIER_RETIRED_SQL = """
SELECT id, fact FROM kg_relationships
WHERE audience IS NULL AND web_artifact_id IS NULL AND t_invalid IS NOT NULL
  AND fact IS NOT NULL AND fact <> ''
ORDER BY id DESC
LIMIT %s
"""

# Never overwrites a label: a re-run, a concurrent labeller, a personal v0 stamp from the old
# personal pile or a human decision wins. The columns are out of every index and partial-index predicate, so
# the update can be a heap-only tuple.
_SET_SQL = """
UPDATE kg_relationships r
   SET audience = v.audience, label_version = %s, label_model = v.model
  FROM unnest(%s::bigint[], %s::text[], %s::text[]) AS v(id, audience, model)
 WHERE r.id = v.id AND r.audience IS NULL
"""


@dataclass
class LabelStats:
    skipped: str = ""
    stopped: str = ""
    labelled: dict[str, int] = field(default_factory=dict)

    def add(self, v: FactVerdict) -> None:
        k = v.audience if v.source == "model" else f"{v.audience}/{v.source}"
        self.labelled[k] = self.labelled.get(k, 0) + 1

    def line(self) -> str:
        n = sum(self.labelled.values())
        if self.skipped:
            return f"fact audience: skipped ({self.skipped})"
        if self.stopped:
            return f"fact audience: stopped: {self.stopped} after {n}"
        parts = ", ".join(f"{k}={v}" for k, v in sorted(self.labelled.items()))
        return f"fact audience: labelled {n} ({parts or 'none pending'})"


def enabled() -> bool:
    """Kill switch for the poller stage: ``SYNAPSE_FACT_AUDIENCE=0``."""
    return os.environ.get("SYNAPSE_FACT_AUDIENCE", "1").strip() not in ("0", "false", "no")


def batch_from_env() -> int:
    """``SYNAPSE_FACT_AUDIENCE_BATCH`` (default :data:`DEFAULT_BATCH`, 1..:data:`MAX_BATCH`)."""
    try:
        n = int(os.environ.get("SYNAPSE_FACT_AUDIENCE_BATCH", DEFAULT_BATCH))
    except ValueError:
        return DEFAULT_BATCH
    return min(MAX_BATCH, max(1, n))


def calls_from_env() -> int:
    """``SYNAPSE_FACT_AUDIENCE_CALLS`` (default :data:`DEFAULT_CALLS`, at least 1)."""
    try:
        return max(1, int(os.environ.get("SYNAPSE_FACT_AUDIENCE_CALLS", DEFAULT_CALLS)))
    except ValueError:
        return DEFAULT_CALLS


def restricted_projects(conn: Any) -> list[str]:
    """Every project any approved restricted surface may read. Empty: no restricted device
    can be served KG facts, so no label is wanted."""
    rows = conn.cursor(row_factory=dict_row).execute(_RESTRICTED_PROJECTS_SQL).fetchall()
    return [r["project"] for r in rows if r["project"]]


def pending(conn: Any, limit: int, reachable: list[str]) -> list[dict[str, Any]]:
    """Up to ``limit`` unlabelled facts, in the tiers the module docstring gives, newest
    first within each. ``reachable`` is :func:`restricted_projects`; tier 1 uses the
    allowlists as stored, so a fact whose projects differ only in case waits for tier 2."""
    cur = conn.cursor(row_factory=dict_row)
    rows: list[dict[str, Any]] = list(cur.execute(_TIER_REACHABLE_SQL, (reachable, limit)))
    if len(rows) < limit:
        taken = [r["id"] for r in rows]
        rows += cur.execute(_TIER_LIVE_SQL, (taken, limit - len(rows)))
    if len(rows) < limit:
        rows += cur.execute(_TIER_RETIRED_SQL, (limit - len(rows),))
    return rows


def write_labels(connect: Connect, verdicts: dict[int, FactVerdict], stamp: str) -> None:
    """Commit ``verdicts`` (fact id to verdict) in one statement. A model verdict carries
    ``stamp`` as its ``label_model``; an ``error`` fallback carries
    :data:`ERROR_MODEL_PREFIX` + ``stamp``, so the two can be told apart afterwards."""
    if not verdicts:
        return
    ids = list(verdicts)
    with connect() as conn:
        conn.execute(
            _SET_SQL,
            (
                LABEL_VERSION,
                ids,
                [verdicts[i].audience for i in ids],
                [
                    stamp if verdicts[i].source == "model" else ERROR_MODEL_PREFIX + stamp
                    for i in ids
                ],
            ),
        )


def label_pending(
    connect: Connect,
    llm: Any,
    *,
    batch: int = DEFAULT_BATCH,
    calls: int = DEFAULT_CALLS,
    model: str | None = None,
    key: str = "",
) -> LabelStats:
    """Label up to ``batch * calls`` unlabelled facts, ``batch`` per classifier call. Each
    call runs with no connection held, and each call's labels are committed together.

    A usage limit, or a backend that fails call after call (:class:`ClassifierDown`), ends
    the cycle with what was written kept and the rest NULL (``stats.stopped``; the poller
    sits the stage out for a while). A fact no call could label (``personal``, source
    ``error``) is held back to the cycle's end and written only if a call succeeded after it
    failed: one fact that breaks calls on its own must not block the queue, but a backend
    that went away must not label anything."""
    stats = LabelStats()
    with connect() as conn:
        if not FACT_AUDIENCE_PROBE.ready(conn, key):
            stats.skipped = "schema 066 not applied"
            return stats
        reachable = restricted_projects(conn)
        if not reachable:
            stats.skipped = "no restricted device can be served facts"
            return stats
        rows = pending(conn, batch * calls, reachable)
    stamp = model or stage_model(STAGE)
    health = CycleHealth(limit=_fail_limit(batch))
    fallbacks: dict[int, FactVerdict] = {}
    for i in range(0, len(rows), batch):
        try:
            verdicts = classify_batch(llm, rows[i : i + batch], model=stamp, health=health)
        except UsageLimitError:
            stats.stopped = "usage limit"
            return stats
        except ClassifierDown as e:
            logger.warning("fact audience: %s", e)
            stats.stopped = "classifier failing"
            return stats
        answered = {k: v for k, v in verdicts.items() if v.source == "model"}
        fallbacks.update({k: v for k, v in verdicts.items() if v.source != "model"})
        write_labels(connect, answered, stamp)
        for v in answered.values():
            stats.add(v)
    trusted = {k: v for k, v in fallbacks.items() if health.answered > health.failed_at[k]}
    write_labels(connect, trusted, stamp)
    for v in trusted.values():
        stats.add(v)
    if len(trusted) < len(fallbacks):
        stats.stopped = "classifier failing"
    return stats
