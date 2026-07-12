"""Per-turn chat gate → preferences (the standing-preference store's feeder).

Every turn already arrives server-side as an "episode" extraction-queue item (the
plugin's Stop hook → /ingest). This gate rides that same item, exactly like the
timeline gate: ONE small LLM call per turn asking "did the USER assert a durable
preference?" — most turns don't, and emit nothing. On emit, one or more self-contained
preference rows are reconciled into the ``preferences`` table (schema 035).

Why a SEPARATE store and not KG edges: every preference is about the single User
entity, so modelling them as graph edges rebuilds the exact User-supernode the
timeline store was built to kill (a "User prefers everything" star node has nothing
to traverse and drowns entity resolution). Preferences are Synapse's worst
LongMemEval category (56.7% vs Mastra 73.3), and a June A/B showed injecting typed
preference facts is worth +13 pts there — so they get first-class, deduplicated
storage rather than being diluted into the graph.

Reconciliation (write-time), keyed on cosine similarity to the live set:
  - > _REASSERT_SIM               → RE-ASSERTION: bump assert_count, keep older text
  - _SUPERSEDE_SIM..._REASSERT_SIM AND a polarity flip → SUPERSESSION: retire old, insert new
  - else                          → INSERT a new live preference
The pure decision lives in ``decide_pref_action`` so it's unit-testable without a DB.

Fail-soft and env-gated: SYNAPSE_PREFS_GATE=0 kills it; any error is logged and
swallowed so KG extraction never breaks on the preference gate's account.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from ingestion.llm_client import MalformedResponseError, parse_with_retry

logger = logging.getLogger(__name__)

# Single-owner constant for the live store today, mirroring kg_pg_write.OWNER — one axis
# so a future multi-tenant split threads the same env var through preferences too.
_OWNER = os.environ.get("SYNAPSE_KG_OWNER_ID", "default")

_MIN_CONTENT = 40  # turns shorter than this can't carry a stated preference worth keeping

# Dedup / supersession thresholds against the live-set cosine similarity. INITIAL VALUES,
# pending calibration on real preference pairs — cosine can't be trusted to separate
# "same preference restated" from "adjacent but distinct" as finely as these bands imply,
# so the supersession band ALSO requires a polarity flip before it retires a row.
_REASSERT_SIM = 0.90  # >= this: near-duplicate, treat as a restatement
_SUPERSEDE_SIM = 0.78  # [this, _REASSERT_SIM) + opposite polarity: a stance flip → supersede

_VALID_POLARITY = ("like", "dislike", "rule")


GATE_PROMPT = """You are watching ONE turn of a coding/assistant session (the user directs; an AI agent executes) for DURABLE USER PREFERENCES worth remembering permanently — how THE USER wants things done, across sessions.

EMIT a preference when the user asserts, explicitly OR in passing, any of:
- a LIKE / preference ("I prefer bullet lists", "I like short answers", "always use tabs")
- a DISLIKE / aversion ("I hate tables", "don't use emoji", "I can't stand long preambles")
- a standing RULE / instruction ("never suggest contract roles", "always confirm before sending email")
- a durable STYLE preference (tone, formatting, verbosity, naming).

DO NOT emit for: questions ("should I use tables?"), task-specific one-offs ("make THIS one shorter", "reword this paragraph"), one-time requests with no standing intent, the assistant's own statements, or facts about the world. Only the USER's own assertions count — a user QUESTION is never an assertion.

Two subject traps — both are NEVER preferences:
- The user DESCRIBING or PREDICTING the assistant's behavior ("you probably default to local", "you always over-explain"). That is an observation about the agent, not a want of the user's. Emit only if the user turns it into a directive ("stop doing X", "I want you to...").
- DESIGN DISCUSSION about how a system/feature under construction should behave (defaults, storage, architecture: "the server should render it", "direct cut, we can revert"). Stances about the system being built are not personal standing preferences unless the user states one explicitly about their own workflow.

Most turns contain no durable preference — return an empty list for them.

Write each `pref` under these hard rules:
1. SELF-CONTAINED and THIRD-PERSON-ABOUT-THE-USER. It must make sense a month from now with no other context. Start with "User " and name the concrete thing: "User prefers bullet lists over tables", "User dislikes em-dashes in written drafts", "User never wants contract or temp roles surfaced". NEVER "this", "that approach", "the above".
2. ONE preference per entry. Split a compound statement into separate entries.
3. GENERALIZE the standing intent, not the momentary task. "make this shorter" is a one-off (skip); "I always want concise answers" is a preference (emit).

polarity: "like" for a preference/liking, "dislike" for an aversion, "rule" for a standing must/must-not instruction.

Output ONLY JSON: {"preferences": [{"pref": "User ...", "polarity": "like"|"dislike"|"rule"}, ...]}  OR  {"preferences": []} when the turn states no durable preference.
"""


def _parse_prefs(text: str) -> list[dict[str, Any]]:
    """Parser for parse_with_retry: extract the {"preferences": [...]} list. Returns a
    (possibly empty) list of validated {pref, polarity} dicts; raises on malformed JSON."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise MalformedResponseError("no JSON object in response", text[:200])
    d = json.loads(text[start : end + 1])
    prefs = d.get("preferences")
    if prefs is None:
        return []
    if not isinstance(prefs, list):
        raise MalformedResponseError("'preferences' is not a list", text[:200])
    out: list[dict[str, Any]] = []
    for p in prefs:
        if not isinstance(p, dict):
            continue
        pref = p.get("pref")
        pol = p.get("polarity")
        if not isinstance(pref, str) or not pref.strip():
            continue
        if pol not in _VALID_POLARITY:
            continue
        out.append({"pref": pref.strip(), "polarity": pol})
    return out


def _contradicts(new_polarity: str, existing_polarity: str) -> bool:
    """A detectable stance flip on the same topic: like <-> dislike. (A 'rule' change
    isn't a polarity contradiction — those are conservatively inserted as new.)"""
    return {new_polarity, existing_polarity} == {"like", "dislike"}


def decide_pref_action(new_polarity: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Pure reconciliation decision for one new preference against the live set.

    ``candidates`` are the nearest live prefs, each ``{id, polarity, sim}``, DESCENDING
    by ``sim`` (cosine similarity in [-1, 1]; ~1 = near-identical text). Returns
    ``{"action": "reassert"|"supersede"|"insert", "target_id": int|None}``.

    Kept DB-free and side-effect-free so the thresholds can be unit-tested directly."""
    if not candidates:
        return {"action": "insert", "target_id": None}
    top = candidates[0]
    if top["sim"] >= _REASSERT_SIM:
        return {"action": "reassert", "target_id": top["id"]}
    if top["sim"] >= _SUPERSEDE_SIM and _contradicts(new_polarity, top["polarity"]):
        return {"action": "supersede", "target_id": top["id"]}
    return {"action": "insert", "target_id": None}


def _group_for(project: str | None) -> str:
    """Group routing derived from the project tag — same axis as the KG. Imported lazily
    to avoid an import cycle (the extractor imports this gate)."""
    try:
        from ingestion.extractor import _default_group_for_project

        return _default_group_for_project(project)
    except Exception:
        return "technical"


class PreferencesGate:
    """One gate call per episode-type extraction item; reconciles preferences rows."""

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
        self.enabled = os.environ.get("SYNAPSE_PREFS_GATE", "1") != "0"

    def process(self, item: dict[str, Any]) -> None:
        """Gate one turn. Fail-soft: errors are logged, never raised."""
        if not self.enabled:
            return
        try:
            self._process(item)
        except Exception as e:
            logger.warning("preferences gate failed for item %s: %s", item.get("id"), e)

    def _process(self, item: dict[str, Any]) -> None:
        content = (item.get("content") or "").strip()
        episode_id = item.get("episode_id")
        if not episode_id or len(content) < _MIN_CONTENT:
            return

        prefs = parse_with_retry(
            self._llm_client,
            base_prompt=f"{GATE_PROMPT}\n\nTHE TURN:\n{content[:6000]}",
            parser=_parse_prefs,
            model=self._model,
            max_tokens=512,
        )
        if not prefs:
            return

        project = item.get("project")
        group_id = _group_for(project)
        source_ref = f"ep:{episode_id}"
        for p in prefs:
            self._store_one(p, project, group_id, source_ref)

    def _store_one(
        self, p: dict[str, Any], project: str | None, group_id: str, source_ref: str
    ) -> None:
        # Embed the BARE pref text (no metadata prefix): the recall serving leg ranks live
        # prefs against the plain query embedding, so the store must share that space, and
        # the gate's own dedup KNN then compares like-with-like.
        vec = self._embedder.embed([p["pref"]], task="document")[0]
        candidates = self._db.find_live_preferences(_OWNER, group_id, vec, limit=5)
        action = decide_pref_action(p["polarity"], candidates)
        embed_model = getattr(self._embedder, "model_name", None) or "voyage-4-large"

        if action["action"] == "reassert":
            self._db.reassert_preference(action["target_id"])
            logger.info("preference re-asserted (#%s): %s", action["target_id"], p["pref"][:80])
            return

        new_id = self._db.insert_preference(
            owner_id=_OWNER,
            group_id=group_id,
            project=project,
            pref=p["pref"],
            polarity=p["polarity"],
            embedding=vec,
            embed_model=embed_model,
            source_ref=source_ref,
        )
        if action["action"] == "supersede":
            self._db.supersede_preference(action["target_id"], new_id)
            logger.info(
                "preference superseded #%s -> #%s: %s",
                action["target_id"],
                new_id,
                p["pref"][:80],
            )
        else:
            logger.info("preference stored (#%s, %s): %s", new_id, p["polarity"], p["pref"][:80])
