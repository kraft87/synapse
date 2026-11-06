"""Who may read a piece of web research: the audience label on ``web_artifacts`` (schema 063).

A restricted surface (a work laptop) is served web chunks only from artifacts labelled
``work-safe``. The label is about SENSITIVITY, not project: research one project paid for
is useful to a similar one, so a project allowlist is the wrong key. What must never reach
a restricted host is personal research, whichever project asked for it.

The vocabulary is the notes one (:data:`ingestion.surfaces.AUDIENCES`), and so is the
shape of the precedence (:func:`ingestion.surfaces.derive_audience`):

1. **explicit** — ``scripts/ingest_research_brief.py --audience`` wins outright.
2. **own surface** — an artifact captured in a session whose turns were uploaded by one
   approved restricted surface (057 provenance on ``episodes.surface_id``) is work-safe:
   the content is already on that machine.
3. **classifier** — a cheap structured LLM call on the pipeline's own provider. It is
   conservative by construction: ``work-safe`` needs the model to answer work-safe AND
   to name no personal signal. ``unsure``, mixed content, any named signal, or a reply
   that never validates all land on ``personal``.

Rule 3 runs server-side, in the poller's maintenance cycle (:func:`label_pending`), not
in the brief script: the script runs on a client machine whose environment has no
extraction provider configured, and a classifier there would fall back to the Claude CLI.

NULL means "not classified yet" and is never served to a restricted caller, the same as
``personal``. A transport failure (provider down, bad key, quota) therefore leaves the row
NULL so the next cycle retries it, instead of writing a ``personal`` nobody would revisit.
Misclassifying the other way is the expensive mistake: a personal page labelled
work-safe is served to every work session from then on (see scripts/audience_backfill.py).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Literal

from psycopg.rows import dict_row
from pydantic import BaseModel

from ingestion.llm_client import LLM_TRANSPORT_ERRORS, stage_model, structured_call
from ingestion.surfaces import (
    AUDIENCES,
    DEFAULT_AUDIENCE,
    OWN_SURFACE_PROBE,
    RESTRICTED_AUDIENCE,
    WEB_AUDIENCE_PROBE,
)

logger = logging.getLogger(__name__)

#: Pipeline stage name for :func:`ingestion.llm_client.stage_model`
#: (``SYNAPSE_WEB_AUDIENCE_MODEL`` beats ``SYNAPSE_LLM_MODEL``).
STAGE = "WEB_AUDIENCE"

#: Artifact kinds whose content is chunked and can be served. Search-result sets and
#: research-job refs have no chunks, so they are never labelled and never served.
SERVED_KINDS = ("research_brief", "web_scrape")

#: How much of a document the classifier reads: the head, plus the tail of a long one
#: (a research brief often ends with a section addressed to its reader).
HEAD_CHARS = 9000
TAIL_CHARS = 3000

Connect = Callable[[], AbstractContextManager[Any]]


class WebAudienceVerdict(BaseModel):
    """The classifier's reply. Signals come first so the model looks before it decides."""

    personal_signals: list[str] = []
    verdict: Literal["work-safe", "personal", "unsure"]


@dataclass(frozen=True)
class AudienceDecision:
    audience: str
    #: explicit | own-surface | classifier | classifier-fallback
    source: str
    detail: str = ""


PROMPT = """You decide whether a saved web document may be shown on its owner's WORK computer \
(an employer-managed laptop). Nothing about the owner's private life may ever reach that \
machine, so you are the last check before it does.

Answer "work-safe" ONLY when you are confident that BOTH hold:
1. The document is technical, professional or general-knowledge material: software, \
infrastructure, security, engineering, science, tools and products as used for work, \
industry news, general how-to.
2. Neither its content nor its TOPIC says or implies anything about a particular person's \
health, body, medical conditions, medications, diet or fitness; money, debts, taxes, \
insurance, purchases or shopping decisions; romantic relationships or dating; family or \
children; legal matters; identity or personal details; job search, resumes, interviews, \
compensation or plans to change employer; home, housing or real estate; religion, \
politics or sexuality; hobbies, entertainment, travel or other private life.

The topic alone is enough to make a document personal: a generic article about a medical \
condition, a medication, personal finance, dating, parenting, a legal dispute, buying a \
home or a car, job hunting or a leisure pursuit is personal, because having researched it \
reveals something private about the reader.

Answer "personal" when any part of the document is personal, even if the rest is \
technical. Answer "unsure" when you cannot tell.

In personal_signals, list every topic or phrase that is personal or made you hesitate. \
Leave it empty only when there is none.

DOCUMENT
kind: {kind}
title: {title}
url: {url}
---
{text}
---"""


def excerpt(content: str | None) -> str:
    """What the classifier reads: the whole document, or its head and tail when long."""
    text = (content or "").strip()
    if len(text) <= HEAD_CHARS + TAIL_CHARS:
        return text
    omitted = len(text) - HEAD_CHARS - TAIL_CHARS
    return f"{text[:HEAD_CHARS]}\n\n[... {omitted} characters omitted ...]\n\n{text[-TAIL_CHARS:]}"


def classify(
    llm: Any,
    *,
    kind: str | None,
    title: str | None,
    url: str | None,
    content: str | None,
    model: str | None = None,
) -> AudienceDecision:
    """Rule 3: the conservative classifier. Transport errors propagate (the caller leaves
    the row unlabelled and retries later); every other failure answers ``personal``."""
    prompt = PROMPT.format(
        kind=kind or "-", title=(title or "-").strip(), url=url or "-", text=excerpt(content)
    )
    try:
        verdict = structured_call(
            llm,
            output_model=WebAudienceVerdict,
            base_prompt=prompt,
            model=model or stage_model(STAGE),
            max_tokens=400,
            max_attempts=2,
        )
    except LLM_TRANSPORT_ERRORS:
        raise
    except Exception as e:
        return AudienceDecision(DEFAULT_AUDIENCE, "classifier-fallback", f"error: {e}"[:200])
    signals = [s.strip() for s in verdict.personal_signals if s and s.strip()]
    if verdict.verdict == "work-safe" and not signals:
        return AudienceDecision(RESTRICTED_AUDIENCE, "classifier")
    return AudienceDecision(
        DEFAULT_AUDIENCE, "classifier", "; ".join(signals)[:300] or verdict.verdict
    )


_OWN_SURFACE_SQL = """
SELECT count(*) AS n,
       count(s.surface_id) AS restricted,
       count(DISTINCT e.surface_id) AS surfaces
FROM episodes e
LEFT JOIN surfaces s
       ON s.surface_id = e.surface_id AND s.trust = 'restricted' AND s.status = 'approved'
WHERE e.session_id = %s
"""


def own_surface_session(conn: Any, session_id: str | None, key: str = "") -> bool:
    """Rule 2: True when every turn of ``session_id`` was uploaded by the SAME approved
    restricted surface. A session with no turns, an unstamped turn (pre-057, root token)
    or a turn from a trusted host does not qualify. False before schema 057."""
    if not session_id or not OWN_SURFACE_PROBE.ready(conn, key):
        return False
    row = conn.cursor(row_factory=dict_row).execute(_OWN_SURFACE_SQL, (session_id,)).fetchone()
    return bool(row and row["n"] > 0 and row["restricted"] == row["n"] and row["surfaces"] == 1)


def derive_web_audience(
    conn: Any, *, explicit: str | None, session_id: str | None, key: str = ""
) -> AudienceDecision | None:
    """Rules 1 and 2. None means neither applies and the classifier decides."""
    if explicit:
        if explicit not in AUDIENCES:
            raise ValueError(f"invalid audience {explicit!r} — expected one of {AUDIENCES}")
        return AudienceDecision(explicit, "explicit")
    if own_surface_session(conn, session_id, key):
        return AudienceDecision(RESTRICTED_AUDIENCE, "own-surface")
    return None


_PENDING_SQL = """
SELECT id, kind, title, url, session_id, content_markdown
FROM web_artifacts
WHERE audience IS NULL AND kind = ANY(%s) AND content_markdown IS NOT NULL
ORDER BY id DESC
LIMIT %s
"""

# Never overwrites a label: a re-run, a concurrent labeller or a human decision wins.
_SET_SQL = "UPDATE web_artifacts SET audience = %s WHERE id = %s AND audience IS NULL"


@dataclass
class LabelStats:
    skipped: str = ""
    labelled: dict[str, int] = field(default_factory=dict)

    def add(self, d: AudienceDecision) -> None:
        k = f"{d.audience}/{d.source}"
        self.labelled[k] = self.labelled.get(k, 0) + 1

    def line(self) -> str:
        if self.skipped:
            return f"web audience: skipped ({self.skipped})"
        parts = ", ".join(f"{k}={v}" for k, v in sorted(self.labelled.items()))
        return f"web audience: labelled {sum(self.labelled.values())} ({parts or 'none pending'})"


def enabled() -> bool:
    """Kill switch for the poller stage: ``SYNAPSE_WEB_AUDIENCE=0``."""
    return os.environ.get("SYNAPSE_WEB_AUDIENCE", "1").strip() not in ("0", "false", "no")


def pending(conn: Any, limit: int) -> list[dict[str, Any]]:
    """Unlabelled, servable artifacts, newest first (a new brief should not wait behind a
    backlog)."""
    return list(
        conn.cursor(row_factory=dict_row).execute(_PENDING_SQL, (list(SERVED_KINDS), limit))
    )


def classify_row(llm: Any, row: dict[str, Any], *, model: str | None = None) -> AudienceDecision:
    """:func:`classify` on a :func:`pending` row."""
    return classify(
        llm,
        kind=row.get("kind"),
        title=row.get("title"),
        url=row.get("url"),
        content=row.get("content_markdown"),
        model=model,
    )


def label_pending(
    connect: Connect, llm: Any, *, limit: int = 10, model: str | None = None, key: str = ""
) -> LabelStats:
    """Label up to ``limit`` unlabelled artifacts. Each LLM call runs with no connection
    held, and each label is committed on its own. A transport error stops the batch and
    propagates; what was already labelled stays labelled, the rest stays NULL."""
    stats = LabelStats()
    with connect() as conn:
        if not WEB_AUDIENCE_PROBE.ready(conn, key):
            stats.skipped = "schema 063 not applied"
            return stats
        rows = pending(conn, limit)
        ruled = {
            r["id"]: derive_web_audience(
                conn, explicit=None, session_id=r.get("session_id"), key=key
            )
            for r in rows
        }
    for r in rows:
        d = ruled[r["id"]] or classify_row(llm, r, model=model)
        with connect() as conn:
            conn.execute(_SET_SQL, (d.audience, r["id"]))
        stats.add(d)
    return stats
