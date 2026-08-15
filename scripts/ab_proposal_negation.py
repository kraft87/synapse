#!/usr/bin/env python3
"""A/B the RECOMMENDED-IS-NOT-ADOPTED + POLARITY-AND-BINDING extraction rules.

Trigger cases (2026-08-14 world-FALSE audit), sanitized:
  * a schema discussion where an assistant RECOMMENDED merging one entity type
    into another yielded "X type was merged into Y ..." — the shipped config
    chose differently;
  * a parked follow-up ("could loosen the guard by ... — separate issue, can
    park") yielded a fact asserting the guard WAS loosened by that exact
    mechanism, while the shipped code used a different one;
  * a verbatim user refusal ("don't change chunk sizes") yielded a
    chunk-size-reduction decision fact.

This harness measures whether the new prompt rules stop proposal-as-adopted
fabrication and polarity inversion WITHOUT suppressing legitimate adoption /
completion facts.

Dev slice: 5 proposal texts (adoption facts here = defect), 3 negation texts
(facts asserting the refused action = defect) and 4 controls where the decision
or change genuinely happened (adoption facts here = required). Each text runs
N times per arm (extraction is stochastic at temperature).

Metrics per arm:
  fabricated  — adoption/implementation facts minted from PROPOSAL texts (want 0)
  inverted    — refused-action facts minted from NEGATION texts          (want 0)
  preserved   — adoption facts minted from CONTROL texts       (want == baseline)

WRITE-FREE: counts LLMExtractor.extract() output; never touches Postgres.
Uses the direct anthropic client like scripts/ab_planned_not_done.py — eval only.

    .venv/bin/python scripts/ab_proposal_negation.py --runs 3 --workers 4
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


# The rules under test — must match extractor.py verbatim (sliced back out so the
# OLD arm is exactly prod-minus-rules even as the surrounding prompt evolves).
# Both new rules sit contiguously between these two markers.
_RULE_MARKER = "- RECOMMENDED IS NOT ADOPTED."
_RULE_END_MARKER = "- EDGES CONNECT THE REAL PARTICIPANTS"

# A fact claiming the choice was made / the change landed.
_ADOPTION_RE = re.compile(
    r"\b(?:was|were|has been|have been|had been) "
    r"(?:merged|adopted|implemented|chosen|selected|applied|loosened|changed|"
    r"removed|renamed|replaced|added|shipped|deployed|enabled|disabled|split|"
    r"consolidated|migrated|collapsed|folded|set to|updated to)\b"
    r"|\b(?:decision was made|decided to|settled on|switched to|now uses|"
    r"was reconfigured)\b",
    re.I,
)

# Proposal framing — an adoption verb inside one of these is NOT a defect.
_PROPOSAL_FRAME_RE = re.compile(
    r"\b(?:propos\w*|recommend\w*|suggest\w*|consider\w*|weigh\w*|option\w*|"
    r"candidate|parked?|defer\w*|follow-?up|under discussion|pending|"
    r"potential\w*|possible|tentativ\w*|no decision|not yet|would be|could be|"
    r"may be|might be|declin\w*|reject\w*|deliberat\w*)\b"
    r"|\b(?:was|were|has|have|had) not\b"
    r"|\bno(?:thing| change[sd]?| decision| adoption)\b|\bnot adopted\b|\bunchanged\b",
    re.I,
)

# Negation framing — a refused-action fact stated AS a refusal is correct, not inverted.
_NEGATION_FRAME_RE = re.compile(
    r"\b(?:not|never|n't|no\b|declin\w*|refus\w*|forbad\w*|forbid\w*|ruled out|"
    r"rul\w* out|reject\w*|avoid\w*|against|unchanged|untouched|alone|"
    r"leav\w*|kept|keep\w*|without|instead of|rather than|off limits|"
    r"out of scope|preserv\w*|veto\w*|block\w*|prevent\w*|disallow\w*|"
    r"prohibit\w*|bar(?:red|s)?)\b",
    re.I,
)


# --- dev slice -------------------------------------------------------------
# Proposal texts: any adoption/implementation fact from these is a fabrication.
PROPOSAL = [
    # sanitized entity-type merge — assistant is decisive in TONE but the config
    # was never touched; the shipped schema chose differently.
    "Walking the entity-type list one by one. Assistant: 'Dependency -> merge into "
    "Tool. You're right, the distinction between libraries and tools is fuzzy in "
    "practice. Metric -> keep, it earns its place. Config -> fold into Tool too.' "
    "User: 'ok, let me sit with that list, I'll touch the config later this week.'",
    # sanitized parked mechanism — imperative phrasing, then parked
    "Fixing the crash in the alias resolver. Assistant, aside: the alias guard also "
    "rejects some valid renames — loosen it by allowing 1->2 word expansions when "
    "the whitespace-stripped forms match. Separate issue from this crash though, "
    "parking it.",
    "Job queue backends, going through them: Redis Streams is simple and already "
    "deployed; Postgres SKIP LOCKED means one less service to run; RabbitMQ is the "
    "heaviest. Assistant: 'go Postgres SKIP LOCKED, the ops saving is worth it.' "
    "User: 'hm. I'll think about it.'",
    "Assistant: 'rename EMBED_DIM to EMBEDDING_DIM so it matches the other keys, and "
    "add a migration shim so existing deployments keep booting.' User did not reply "
    "on that point and moved on to the reranker timeout.",
    "Throughput on the ingest service will cap out around 40 docs/s. The fix is to "
    "split it into a fetcher and a writer so the writer can scale independently. Not "
    "this quarter — filed as a follow-up.",
]
# Negation texts: a fact asserting the refused action happened is an inversion.
NEGATION = [
    {
        # the real trigger, sanitized — chunking is discussed at length, then refused
        "text": "The extraction worker keeps dying with MAX_TOKENS truncation errors. "
        "Walked the chunking path: chunks are 2000 tokens with 200 overlap, and the "
        "summary built from a long chunk is what blows the output limit. User: "
        "'don't change chunk sizes.' So the summary cap went from unlimited to 1200 "
        "tokens, and separately the embedding batch size was lowered from 64 to 16 to "
        "clear a different rate-limit error. The truncation errors stopped.",
        "subject": r"chunk[\s_-]?size",
        "action": r"\b(?:reduc|decreas|increas|chang|lower|shrink|adjust|resiz|"
        r"shrunk|tun|halv|small)",
    },
    {
        "text": "Requests to the upstream API time out under load. The retry path does "
        "5 attempts with exponential backoff, which stretches a bad request out to 30 "
        "seconds. User: 'leave the retry logic alone, it was tuned last month — look at "
        "connection pooling.' Pool size went from 5 to 20 and the timeouts cleared.",
        "subject": r"retry",
        "action": r"\b(?:rewrit|chang|modif|replac|refactor|adjust|updat|tun|"
        r"remov|disabl|increas|reduc)",
    },
    {
        "text": "The nightly report query takes 40 seconds. User: no new indexes on "
        "that table, it is already write-heavy. Assistant rewrote the query to drop "
        "the correlated subquery, which took it to 3 seconds.",
        "subject": r"\bindex(?:es)?\b",
        "action": r"\b(?:add|creat|introduc|built|build|new)",
    },
    {
        # binding half of the rule: the deploy belongs to other work, not PR #12
        "text": "Session covered three threads. PR #12 changed one client-side plugin "
        "file (a tool description string) and is still open. Separately, the reranker "
        "cutoff work from last week was deployed to prod and the container came back "
        "healthy. A third branch, the notes importer, is still local-only.",
        "subject": r"PR #12",
        "action": r"\b(?:deploy|shipped|releas|rolled out|went to prod|in production)",
    },
]
# Controls: the decision/change genuinely happened — these facts must survive.
CONTROL = [
    "User approved the schema trim and it shipped: the Dependency entity type was "
    "merged into Tool in config/entity_types.yaml, and the Metric type was kept. "
    "The extractor was redeployed the same evening.",
    "PR #12 was merged and deployed to prod; the mcp-server container was restarted "
    "and came back healthy.",
    "After weighing Redis Streams against Postgres SKIP LOCKED, User decided to "
    "switch the job queue to Postgres SKIP LOCKED to drop a service from the stack.",
    "The alias guard was loosened in the end, but by a similarity ratio threshold of "
    "0.8 rather than the word-count rule, because the ratio also caught "
    "punctuation-only renames.",
]


def _strip_rule(prompt: str) -> str:
    start = prompt.index(_RULE_MARKER)
    end = prompt.index(_RULE_END_MARKER)
    return prompt[:start] + prompt[end:]


def _count_adoptions(facts: list) -> list[str]:
    return [
        f.fact
        for f in facts
        if _ADOPTION_RE.search(f.fact) and not _PROPOSAL_FRAME_RE.search(f.fact)
    ]


def _count_inversions(facts: list, subject: str, action: str) -> list[str]:
    subj_re = re.compile(subject, re.I)
    act_re = re.compile(action, re.I)
    return [
        f.fact
        for f in facts
        if subj_re.search(f.fact)
        and act_re.search(f.fact)
        and not _NEGATION_FRAME_RE.search(f.fact)
    ]


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
        for kind, texts in (("proposal", PROPOSAL), ("negation", NEGATION), ("control", CONTROL)):
            for ti, _ in enumerate(texts):
                for ri in range(args.runs):
                    jobs.append((arm, kind, ti, ri))

    def run_one(job):
        arm, kind, ti, ri = job
        if kind == "negation":
            case = NEGATION[ti]
            text = case["text"]
        else:
            text = (PROPOSAL if kind == "proposal" else CONTROL)[ti]
        extractor_mod._EXTRACTION_PROMPT = old_prompt if arm == "old" else new_prompt
        ex = LLMExtractor(client, model=args.model)
        try:
            result = ex.extract(text, [], session_date="2026-06-10")
        except Exception as e:  # transient API noise — count as no data
            return (arm, kind, ti, ri, None, str(e)[:100])
        if kind == "negation":
            hits = _count_inversions(result.facts, case["subject"], case["action"])
        else:
            hits = _count_adoptions(result.facts)
        return (arm, kind, ti, ri, (hits, [f.fact for f in result.facts]), None)

    # NOTE: mutating the module global per-job is racy across threads; keep arms
    # serialized instead — run old fully, then new.
    results = []
    for arm in ("old", "new"):
        arm_jobs = [j for j in jobs if j[0] == arm]
        extractor_mod._EXTRACTION_PROMPT = old_prompt if arm == "old" else new_prompt
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results.extend(pool.map(run_one, arm_jobs))

    kinds = ("proposal", "negation", "control")
    stats = {arm: {k: [] for k in kinds} for arm in ("old", "new")}
    errors = 0
    for arm, kind, ti, ri, hits, err in results:
        if err:
            errors += 1
            continue
        stats[arm][kind].append((ti, ri, hits))

    print(f"\n=== proposal/negation A/B (runs={args.runs}, model={args.model}) ===")
    for arm in ("old", "new"):
        fab = sum(len(h) for _, _, (h, _all) in stats[arm]["proposal"])
        fab_n = len(stats[arm]["proposal"])
        inv = sum(len(h) for _, _, (h, _all) in stats[arm]["negation"])
        inv_n = len(stats[arm]["negation"])
        pres = sum(len(h) for _, _, (h, _all) in stats[arm]["control"])
        pres_n = len(stats[arm]["control"])
        print(
            f"{arm:>4}: fabricated adoption facts on PROPOSAL texts: {fab} across "
            f"{fab_n} extractions | inverted facts on NEGATION texts: {inv} across "
            f"{inv_n} extractions | adoption facts on CONTROLS: {pres} across "
            f"{pres_n} extractions"
        )
    if errors:
        print(f"(transient errors: {errors})")

    out = REPO / "kg_eval_runs" / "proposal_negation_ab.json"
    out.parent.mkdir(exist_ok=True)
    dump = {
        arm: {
            kind: [
                {"text_idx": ti, "run": ri, "flagged_facts": h, "all_facts": a}
                for ti, ri, (h, a) in stats[arm][kind]
            ]
            for kind in kinds
        }
        for arm in ("old", "new")
    }
    out.write_text(json.dumps(dump, indent=1))
    print(f"fact dump: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
