#!/usr/bin/env python3
"""A/B the PLANNED-IS-NOT-DONE rule in the TIMELINE GATE prompt.

Sibling of scripts/ab_planned_not_done.py, which A/Bs the same rule in the KG
extractor. Trigger case (2026-08-27): a turn where a recruiter stated a pay
ceiling for a PROSPECTIVE role and the user said they hoped to negotiate higher
produced the timeline event "received a job offer of $115K and planned to
negotiate toward the posted band max" (salience 2) — a completed happening that
never happened. The gate's only guard was a weak "future plans or intentions"
bullet; the extractor's PLANNED IS NOT DONE rule had never been ported over.

Dev slice: 6 planning/intent turns (a completed-outcome event here = fabrication)
and 4 completed-action controls (those events are REAL and must survive). Each
text runs N times per arm (the gate is stochastic at temperature).

Metrics per arm:
  fabricated  — completion-verb events minted from PLANNED turns  (want 0)
  preserved   — completion-verb events minted from DONE controls  (want == baseline)

WRITE-FREE: drives the gate's own structured_call/TimelineGateEvents path with
the prompt as the only variable; never touches Postgres, the embedder, or dedup.

    .venv/bin/python scripts/ab_planned_not_done_timeline.py --runs 3 --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import ingestion.timeline_gate as gate_mod  # noqa: E402
from ingestion.llm_schemas import TimelineGateEvents  # noqa: E402

TURN_DATE = "2026-06-10"


def _load_env() -> None:
    """Load .env so create_llm_client builds the SAME provider/model prod uses
    (SYNAPSE_LLM_PROVIDER/_MODEL/_API_KEY) — validating a prompt rule on a
    different model than the one that minted the defect proves nothing."""
    envf = REPO / ".env"
    if not envf.exists():
        return
    for line in envf.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


# The rule under test — sliced back out of the live prompt so the OLD arm is
# exactly prod-minus-rule even as the surrounding prompt evolves.
_RULE_MARKER = "PLANNED IS NOT DONE."
_RULE_END_MARKER = "Write each `event` under these hard rules:"

# Gate events are naked lowercase past-tense verbs ("received a job offer...",
# "shipped the auth refactor"), so match the verb forms directly rather than the
# extractor's passive "was performed" shapes. Meta-act verbs the rule explicitly
# ALLOWS (booked / scheduled / decided / chose / drafted / asked / listed) are
# deliberately absent — emitting those is correct behaviour, not preservation.
_COMPLETION_RE = re.compile(
    r"\b(?:received|accepted|got|obtained|landed|secured|negotiated|signed|"
    r"underwent|performed|completed|finished|attended|shipped|deployed|merged|"
    r"restarted|applied|filed|submitted|sent|paid|bought|purchased|replaced|"
    r"delivered|installed|renewed|hired|ran)\b",
    re.I,
)

# --- dev slice -------------------------------------------------------------
# Planning/intent turns, each paired with the pattern that catches ITS planned
# thing asserted as done. A generic completion-verb sweep is useless here: the
# gate legitimately writes "completed a recruiter call", "booked the MRI",
# "received a $940 quote" off these turns — the meta-act and the receipt of
# information really did happen. Only the planned WORK coming back as done is
# the defect, and what that looks like is per-text.
# Phrased the way the trigger turn actually read: terse, no "nothing started
# yet" disclaimer to lean on. A slice that spells the answer out doesn't
# reproduce the defect and so can't measure the fix.
PLANNED: list[tuple[str, str]] = [
    # fictional analog of the real trigger (public repo — never the real case):
    # a stated pay ceiling for a PROSPECTIVE role + intent to negotiate. Any
    # event asserting an offer exists, was received, or was negotiated is minted.
    (
        "recruiter call for the Vantor Systems platform-security role went well. they "
        "will pay $118,000 and i am hoping to negotiate higher. their max is $141,000. "
        "she wants to move me onto the hiring manager loop next.",
        r"\b(?:received|got|accepted|landed|secured|fielded)\b[^,;.]{0,40}\boffer\b"
        r"|\bwas offered\b|\b(?:they|vantor\w*|the recruiter)\s+offered\b"
        r"|\b(?:a|an|the)\s+(?:job\s+|written\s+|verbal\s+)?offer\b"
        r"|\bnegotiated\b|\baccepted\b",
    ),
    (
        "mom's memory workup: MRI, blood biomarker panel, geriatric referral. the MRI "
        "is Sunday June 16 at the imaging place on Kipling. the lab requisition for the "
        "panel has to come from the clinic.",
        r"\b(?:underwent|had|completed|did|performed|received|got)\b[^,;.]{0,40}"
        r"\b(?:MRI|scan|panel|referral|workup)\b"
        r"|\b(?:MRI|panel|workup)\b[^,;.]{0,30}\bwas (?:performed|done|completed)\b"
        r"|\b(?:results|report)\b[^,;.]{0,30}\b(?:came back|arrived)\b",
    ),
    (
        "migration plan: dump the old schema, apply migration 051, restart the poller. "
        "going with next weekend for the window since nothing else is running then.",
        r"\b(?:applied|ran|executed|dumped|restarted|completed|finished)\b[^,;.]{0,40}"
        r"\b(?:migration|schema|poller|051)\b|\bmigrated\b",
    ),
    (
        "the shop says $940 for the rear struts and they can take the car in on the "
        "9th. getting a second quote from the place on Dundas first.",
        r"\b(?:paid|replaced|installed|completed|finished|approved)\b[^,;.]{0,40}"
        r"\b(?:\$?940|strut|repair|work)\b|\bdropped the car off\b"
        r"|\bhad the (?:struts|car)\b",
    ),
    (
        "sprint plan for next week: ship the auth refactor, file the SOC2 evidence "
        "request, schedule the pen test for Q3. that plus whatever the ingest backlog "
        "throws up.",
        r"\b(?:shipped|deployed|merged|completed|finished|landed)\b[^,;.]{0,40}"
        r"\b(?:auth refactor|refactor)\b"
        r"|\b(?:filed|submitted|sent)\b[^,;.]{0,40}\b(?:SOC2|evidence request)\b"
        r"|\b(?:ran|completed|conducted)\b[^,;.]{0,30}\bpen test\b",
    ),
    (
        "tomorrow: signed lease back to the landlord, and book movers for the first "
        "week of July. storage locker too if the closing date slips.",
        # "(the) signed lease" is the NOUN — only a verb use of signed/sent counts.
        r"(?<!the )\b(?:sent|signed|returned|mailed|delivered|handed)\b[^,;.]{0,40}"
        r"\blease\b|\b(?:booked|hired|paid)\b[^,;.]{0,30}\bmovers\b"
        r"|\b(?:rented|booked)\b[^,;.]{0,30}\bstorage\b",
    ),
]
# Completed-action controls: these completion events are REAL and must survive.
DONE = [
    "signed and accepted the Vantor Systems offer this morning — $126,500 base, start "
    "date September 8. emailed the other recruiter to tell them i'm out of their "
    "process.",
    "merged PR #150 and deployed it to prod. restarted the mcp-server container "
    "afterwards and it came back healthy on the first health check.",
    "attended the dental appointment this morning — cleaning plus two x-rays, no "
    "cavities. paid $210 out of pocket because the plan only covers one cleaning a "
    "year.",
    "the lab work got done on May 3 and the results were sent to the ordering "
    "physician. everything came back in range except ferritin, which was low.",
]


def _strip_rule(prompt: str) -> str:
    start = prompt.index(_RULE_MARKER)
    end = prompt.index(_RULE_END_MARKER)
    return prompt[:start] + prompt[end:]


_FAB_RES = [re.compile(pat, re.I) for _, pat in PLANNED]


def _score(events: list[dict], kind: str, text_idx: int) -> list[str]:
    """Events that count for this text's metric.

    PLANNED → the per-text fabrication pattern (the planned thing asserted done).
    DONE    → any completion verb (these completions are real; they must survive).
    """
    rx = _FAB_RES[text_idx] if kind == "planned" else _COMPLETION_RE
    return [e["event"] for e in events if rx.search(e["event"])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3, help="gate calls per text per arm")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default=None, help="override; default = prod SYNAPSE_LLM_MODEL")
    args = ap.parse_args()

    _load_env()
    from ingestion.llm_client import create_llm_client, structured_call

    model = args.model or os.environ.get("SYNAPSE_LLM_MODEL") or "claude-haiku-4-5"
    client = create_llm_client(model=model)
    args.model = model
    new_prompt = gate_mod.GATE_PROMPT
    if _RULE_MARKER not in new_prompt:
        print("FATAL: rule marker not found in GATE_PROMPT", file=sys.stderr)
        return 2
    old_prompt = _strip_rule(new_prompt)

    jobs = []  # (arm, kind, text_idx, run_idx)
    for arm in ("old", "new"):
        for kind, texts in (("planned", PLANNED), ("done", DONE)):
            for ti, _ in enumerate(texts):
                for ri in range(args.runs):
                    jobs.append((arm, kind, ti, ri))

    def run_one(job):
        arm, kind, ti, ri = job
        text = PLANNED[ti][0] if kind == "planned" else DONE[ti]
        prompt = old_prompt if arm == "old" else new_prompt
        try:
            gated = structured_call(
                client,
                output_model=TimelineGateEvents,
                base_prompt=(
                    f"{prompt}\nThis turn happened on {TURN_DATE}.\n\nTHE TURN:\n{text[:6000]}"
                ),
                model=args.model,
                max_tokens=512,
                max_attempts=3,
            )
        except Exception as e:  # transient API noise — count as no data
            return (arm, kind, ti, ri, None, None, str(e)[:100])
        events = gate_mod._events_to_dicts(gated)
        return (arm, kind, ti, ri, _score(events, kind, ti), [e["event"] for e in events], None)

    results = []
    for arm in ("old", "new"):
        arm_jobs = [j for j in jobs if j[0] == arm]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results.extend(pool.map(run_one, arm_jobs))

    stats: dict = {a: {"planned": [], "done": []} for a in ("old", "new")}
    errors = 0
    for arm, kind, ti, ri, completions, events, err in results:
        if err:
            errors += 1
            continue
        stats[arm][kind].append((ti, ri, completions, events))

    print(f"\n=== timeline-gate planned-is-not-done A/B (runs={args.runs}, model={args.model}) ===")
    for arm in ("old", "new"):
        planned_fab = sum(len(c) for _, _, c, _ in stats[arm]["planned"])
        planned_n = len(stats[arm]["planned"])
        planned_ev = sum(len(e) for _, _, _, e in stats[arm]["planned"])
        done_pres = sum(len(c) for _, _, c, _ in stats[arm]["done"])
        done_n = len(stats[arm]["done"])
        done_ev = sum(len(e) for _, _, _, e in stats[arm]["done"])
        print(
            f"{arm:>4}: fabricated completion events on PLANNED turns: {planned_fab} "
            f"across {planned_n} gate calls ({planned_ev} events total) | completion "
            f"events on DONE controls: {done_pres} across {done_n} gate calls "
            f"({done_ev} events total)"
        )
    if errors:
        print(f"(transient errors: {errors})")

    out = REPO / "kg_eval_runs" / "planned_not_done_timeline_ab.json"
    out.parent.mkdir(exist_ok=True)
    dump = {
        arm: {
            kind: [
                {"text_idx": ti, "run": ri, "flagged": c, "events": e}
                for ti, ri, c, e in stats[arm][kind]
            ]
            for kind in ("planned", "done")
        }
        for arm in ("old", "new")
    }
    out.write_text(json.dumps(dump, indent=1))
    print(f"event dump: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
