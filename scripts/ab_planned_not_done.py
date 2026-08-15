#!/usr/bin/env python3
"""A/B the PLANNED-IS-NOT-DONE extraction rule on a targeted dev slice.

Trigger case (2026-08-14, recall_feedback 333/334): a manual note describing a
medical workup's PLANNED components ("the workup (MRI, p-tau217 panel,
geriatric referral)") yielded three completed-action facts ("An MRI was
performed..."). This harness measures whether the new prompt rule stops the
tense jump WITHOUT suppressing legitimate completed-action facts.

Dev slice: 6 planning-language texts (completion facts here = defect) and
4 completed-action controls (completion facts here = required). Each text runs
N times per arm (extraction is stochastic at temperature).

Metrics per arm:
  fabricated  — completion-verb facts minted from PLANNED texts  (want 0)
  preserved   — completion-verb facts minted from DONE controls  (want == baseline)

WRITE-FREE: counts LLMExtractor.extract() output; never touches Postgres.
Uses the direct anthropic client like scripts/ab_fact_density.py — eval only.

    .venv/bin/python scripts/ab_planned_not_done.py --runs 3 --workers 4
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

import ingestion.extractor as extractor_mod  # noqa: E402
from ingestion.extractor import LLMExtractor  # noqa: E402


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


# The rule under test — must match extractor.py verbatim (sliced back out so the
# OLD arm is exactly prod-minus-rule even as the surrounding prompt evolves).
_RULE_MARKER = "- PLANNED IS NOT DONE."
_RULE_END_MARKER = "- EDGES CONNECT THE REAL PARTICIPANTS"

_COMPLETION_RE = re.compile(
    r"\b(?:was|were|has been|have been) "
    r"(?:performed|completed|done|made|conducted|carried out|administered|run|"
    r"executed|obtained|received|sent|filed|submitted|signed)\b|\bunderwent\b",
    re.I,
)

# --- dev slice -------------------------------------------------------------
# Planning-language texts: any completion-verb fact from these is a fabrication.
PLANNED = [
    # the real trigger, sanitized
    "User's mother is the subject of the 2026 memory-decline workup (MRI, "
    "blood biomarker panel, geriatric referral). User prefers outbound messages "
    "to clinics not mention the family doctor's refusal to order the test.",
    "Agenda for Friday's appointment: (1) memory clinic referral, (2) lab "
    "requisition for the blood panel, (3) ask about a cognitive baseline score. "
    "The MRI is booked for Sunday June 16.",
    "The migration plan covers three steps: dump the old schema, apply "
    "migration 051, and restart the poller. User wants this run next weekend.",
    "User is preparing a renewal package for the insurance claim: photos of the "
    "damage, two contractor quotes, and the adjuster's form.",
    "Sprint checklist: ship the auth refactor, file the SOC2 evidence request, "
    "and schedule the pen test for Q3.",
    "User plans to send the signed lease back to the landlord tomorrow and "
    "book movers for the first week of July.",
]
# Completed-action controls: these completion facts are REAL and must survive.
DONE = [
    "User confirmed the MRI was performed on Tuesday and the radiologist's report came back clean.",
    "PR #150 was merged and deployed to prod; the mcp-server container was "
    "restarted and came back healthy.",
    "User sent the signed lease back to the landlord yesterday and received "
    "confirmation it was filed.",
    "The blood panel was administered at the lab on May 3 and results were "
    "sent to the ordering physician.",
]


def _strip_rule(prompt: str) -> str:
    start = prompt.index(_RULE_MARKER)
    end = prompt.index(_RULE_END_MARKER)
    return prompt[:start] + prompt[end:]


def _count_completions(facts: list) -> list[str]:
    return [f.fact for f in facts if _COMPLETION_RE.search(f.fact)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3, help="extractions per text per arm")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default=None, help="override; default = prod SYNAPSE_LLM_MODEL")
    args = ap.parse_args()

    _load_env()
    from ingestion.llm_client import create_llm_client

    model = args.model or os.environ.get("SYNAPSE_LLM_MODEL") or "claude-haiku-4-5"
    client = create_llm_client(model=model)
    args.model = model
    new_prompt = extractor_mod._EXTRACTION_PROMPT
    if _RULE_MARKER not in new_prompt:
        print("FATAL: rule marker not found in _EXTRACTION_PROMPT", file=sys.stderr)
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
        text = (PLANNED if kind == "planned" else DONE)[ti]
        extractor_mod._EXTRACTION_PROMPT = old_prompt if arm == "old" else new_prompt
        ex = LLMExtractor(client, model=args.model)
        try:
            result = ex.extract(text, [], session_date="2026-06-10")
        except Exception as e:  # transient API noise — count as no data
            return (arm, kind, ti, ri, None, str(e)[:100])
        return (arm, kind, ti, ri, _count_completions(result.facts), None)

    # NOTE: mutating the module global per-job is racy across threads; keep arms
    # serialized instead — run old fully, then new.
    results = []
    for arm in ("old", "new"):
        arm_jobs = [j for j in jobs if j[0] == arm]
        extractor_mod._EXTRACTION_PROMPT = old_prompt if arm == "old" else new_prompt
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results.extend(pool.map(run_one, arm_jobs))

    stats = {"old": {"planned": [], "done": []}, "new": {"planned": [], "done": []}}
    errors = 0
    for arm, kind, ti, ri, completions, err in results:
        if err:
            errors += 1
            continue
        stats[arm][kind].append((ti, ri, completions))

    print(f"\n=== planned-is-not-done A/B (runs={args.runs}, model={args.model}) ===")
    for arm in ("old", "new"):
        planned_fab = sum(len(c) for _, _, c in stats[arm]["planned"])
        planned_n = len(stats[arm]["planned"])
        done_pres = sum(len(c) for _, _, c in stats[arm]["done"])
        done_n = len(stats[arm]["done"])
        print(
            f"{arm:>4}: fabricated completion facts on PLANNED texts: {planned_fab} "
            f"across {planned_n} extractions | completion facts on DONE controls: "
            f"{done_pres} across {done_n} extractions"
        )
    if errors:
        print(f"(transient errors: {errors})")

    out = REPO / "kg_eval_runs" / "planned_not_done_ab.json"
    out.parent.mkdir(exist_ok=True)
    dump = {
        arm: {
            kind: [
                {"text_idx": ti, "run": ri, "completion_facts": c} for ti, ri, c in stats[arm][kind]
            ]
            for kind in ("planned", "done")
        }
        for arm in ("old", "new")
    }
    out.write_text(json.dumps(dump, indent=1))
    print(f"fact dump: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
