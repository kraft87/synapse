#!/usr/bin/env python3
"""De-hub extraction-prompt A/B — 2026-08-07.

ONE model (z-ai/glm-5.2 via OpenRouter), TWO arms that differ ONLY in the
extraction prompt:

  control — ``_EXTRACTION_PROMPT`` as it stands on ``origin/main`` (read out of
            git with ``ast.literal_eval``, so the arm can never drift from what
            prod actually ships)
  dehub   — ``_EXTRACTION_PROMPT`` from the working tree (branch
            ``feat/dehub-extraction``): edges connect the real participants,
            participant-less happenings get reified as event nodes, and the
            fact sentence always keeps the user mention + the date.

Everything else is the Jun-16 frozen-chunk methodology, unchanged: seed=7
PII-free chunks, prod ``LLMExtractor``/``structured_call`` path,
``DeterministicExtractor`` entity context, Voyage dedup, blind Haiku judge
(judge sees fact TEXT ONLY, never the arm or the edge endpoints).

On top of the quality metrics this run adds STRUCTURAL metrics computed from the
raw extractions — the thing the revision is actually trying to move:

  user-edge rate      share of facts whose source or target resolves to the owner
                      node (User/Kyle/the user/me) — the supernode driver
  event nodes         entities typed as an event (reification target), total and
                      the subset carrying >=2 facts (a real hub, not a leaf)
  de-hubbed rate      share of facts whose TEXT mentions the user while the EDGE
                      touches no user node — the target behaviour
  user-mention rate   share of facts whose text mentions the user at all — the
                      guardrail: rerouting the edge must not delete the user
  date-anchor rate    share of facts carrying a date — the other guardrail

WRITE-FREE: never touches the KG or any prod table (one read-only SELECT for the
frozen chunks). Sequential arms (shared drop-counter), parallel chunks.

  usage: .venv/bin/python scripts/ab_dehub_prompt.py [--n 50] [--workers 8]
"""

from __future__ import annotations

import argparse
import ast
import logging
import os
import random
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import orjson
import psycopg
from psycopg.rows import dict_row

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ingestion.extractor import (  # noqa: E402
    _EXTRACTION_PROMPT,
    DeterministicExtractor,
    LLMExtractor,
)
from ingestion.llm_client import OpenAIChatClient  # noqa: E402
from ingestion.models import ExtractionResult  # noqa: E402

OR_BASE = "https://openrouter.ai/api/v1"
JUDGE_MODEL = "anthropic/claude-haiku-4.5"
MODEL = "z-ai/glm-5.2"
MODEL_PRICE = (0.25, 0.79)  # $/Mtok in, out — for the est-cost column only
PII_FREE = ("synapse", "neuron", "cursor-logfire-plugin", "axon-backend")
CONTROL_REF = "origin/main"

# --- owner-node / user-mention detection -----------------------------------
# The owner hub is deployment config (SYNAPSE_OWNER_NAME, see ingestion/dedup.py);
# on this corpus it is the default "User", with "Kyle" as the in-text spelling.
_OWNER = (os.environ.get("SYNAPSE_OWNER_NAME") or "User").strip() or "User"
_OWNER_ALIASES = {
    a.strip().lower()
    for a in (os.environ.get("SYNAPSE_OWNER_ALIASES") or "").split(",")
    if a.strip()
}
_USER_NODE_NAMES = {
    _OWNER.lower(),
    "user",
    "the user",
    "kyle",
    "kyle doucette",
    "i",
    "me",
    "myself",
    "owner",
    "the owner",
} | _OWNER_ALIASES
_USER_MENTION_RE = re.compile(r"\b(user|kyle)('s)?\b", re.I)
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_MEANING_RE = re.compile(r"\(meaning \d{4}-\d{2}-\d{2}\)")
_LOOSE_DATE_RE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|aug(ust)?|"
    r"sep(t|tember)?|oct(ober)?|nov(ember)?|dec(ember)?)\b",
    re.I,
)
# Entity TYPE strings that mark a reified happening (the node the de-hub rule
# asks the model to mint when an event has no real counterparty).
_EVENT_TYPE_RE = re.compile(
    r"\b(event|wedding|trip|travel|meeting|milestone|occasion|ceremony|"
    r"appointment|incident|outage|launch|release|holiday|birthday|"
    r"conference|celebration|activity|session)\b",
    re.I,
)


def _norm_node(name: str) -> str:
    n = (name or "").strip().lower().strip("\"'")
    n = re.sub(r"\s+", " ", n)
    return n


def is_user_node(name: str) -> bool:
    return _norm_node(name) in _USER_NODE_NAMES


def mentions_user(text: str) -> bool:
    return bool(_USER_MENTION_RE.search(text or ""))


def has_date(text: str) -> bool:
    return bool(_LOOSE_DATE_RE.search(text or ""))


class PromptExtractor(LLMExtractor):
    """LLMExtractor with the extraction prompt TEMPLATE injected.

    Identical code path to prod (``LLMExtractor._run`` -> ``structured_call``
    -> the ``CombinedExtraction`` cross-reference validator); the ONLY
    difference between arms is the template string rendered here.
    """

    def __init__(self, *a, template: str, **k) -> None:
        super().__init__(*a, **k)
        self._template = template

    def extract(
        self,
        summary: str,
        context_entities: list,
        session_date: str | None = None,
    ) -> ExtractionResult:
        context_str = (
            ", ".join(f"{e.name} ({e.type})" for e in context_entities)
            if context_entities
            else "none"
        )
        return self._run(
            self._template.format(
                context_entities=context_str,
                summary=summary,
                session_date=session_date or "unknown",
            )
        )


def prompt_at_ref(ref: str = CONTROL_REF) -> str:
    """Read ``_EXTRACTION_PROMPT`` out of ``ingestion/extractor.py`` at a git ref.

    ast-parses rather than imports: the ref's module has its own imports and
    module-level env reads, and we only want one string constant out of it.
    """
    src = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{ref}:ingestion/extractor.py"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "_EXTRACTION_PROMPT" for t in node.targets
        ):
            return str(ast.literal_eval(node.value))
    raise SystemExit(f"_EXTRACTION_PROMPT not found at {ref}")


def _load_env() -> None:
    for envf in (REPO / ".env", Path.home() / ".config" / "openrouter.env"):
        if not envf.exists():
            continue
        for line in envf.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))


class DropCounter(logging.Handler):
    """Counts the extractor's dropped-fact / failed-extraction log lines."""

    def __init__(self) -> None:
        super().__init__()
        self.dropped_facts = 0
        self.full_failures = 0
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.msg or ""
        with self._lock:
            if msg.startswith("Dropped %d fact"):
                self.dropped_facts += int(record.args[0])
            elif msg.startswith("LLM extraction failed"):
                self.full_failures += 1


def _dedup_count(vecs, thr) -> int:
    reps: list = []
    for v in vecs:
        nv = np.asarray(v, dtype=np.float32)
        norm = np.linalg.norm(nv)
        if norm == 0:
            continue
        nv = nv / norm
        if not any(float(nv.dot(rep)) >= thr for rep in reps):
            reps.append(nv)
    return len(reps)


def judge_quality(client, facts) -> list[int]:
    """Blind quality judge — VERBATIM from scripts/ab_or_extract.py.

    The judge sees fact TEXT only, in arm-agnostic order; it never learns which
    prompt produced a fact, so it cannot reward the structural change directly.
    """
    numbered = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(facts))
    p = (
        "Below are candidate facts extracted for a long-term knowledge graph. Rate EACH:\n"
        "1 = HIGH-VALUE: durable AND specific AND non-obvious — a decision+rationale, a config/value, "
        "a file/path/service/address, a credential, a root cause, or a concrete non-trivial relationship.\n"
        "0 = LOW-VALUE/JUNK: a transient action (something was run/checked/posted), vague or generic, "
        "obvious boilerplate, restates ambient/system context, or not really a fact.\n\n"
        f"{numbered}\n\n"
        "Return ONLY a JSON array of 0/1 integers, one per fact in order."
    )
    r = client.messages.create(
        model=JUDGE_MODEL, max_tokens=600, messages=[{"role": "user", "content": p}]
    )
    txt = str(r.content[0].text)
    a, b = txt.find("["), txt.rfind("]")
    arr = orjson.loads(txt[a : b + 1])
    return [1 if int(x) else 0 for x in arr][: len(facts)]


def structural_metrics(recs: list[dict], ents: list[dict]) -> dict:
    """Per-arm structure, computed from the RAW extraction (pre-dedup)."""
    n_f = len(recs)
    if not n_f:
        return {}
    user_edge = sum(1 for f in recs if is_user_node(f["source"]) or is_user_node(f["target"]))
    user_src = sum(1 for f in recs if is_user_node(f["source"]))
    txt_user = sum(1 for f in recs if mentions_user(f["fact"]))
    dehubbed = sum(
        1
        for f in recs
        if mentions_user(f["fact"])
        and not is_user_node(f["source"])
        and not is_user_node(f["target"])
    )
    dated = sum(1 for f in recs if has_date(f["fact"]))
    anchored = sum(1 for f in recs if _MEANING_RE.search(f["fact"]))

    # reified event nodes: entities the model TYPED as a happening, counted once
    # per (chunk, name) and cross-checked against edge participation.
    ev_names = {
        (e["chunk"], _norm_node(e["name"])) for e in ents if _EVENT_TYPE_RE.search(e["type"] or "")
    }
    deg: Counter = Counter()
    for f in recs:
        for nd in (f["source"], f["target"]):
            key = (f["chunk"], _norm_node(nd))
            if key in ev_names:
                deg[key] += 1
    return {
        "user_edge_rate": user_edge / n_f,
        "user_source_rate": user_src / n_f,
        "user_edges": user_edge,
        "text_user_rate": txt_user / n_f,
        "dehubbed_rate": dehubbed / n_f,
        "dated_rate": dated / n_f,
        "anchored_rate": anchored / n_f,
        "event_nodes": len(ev_names),
        "event_nodes_linked": sum(1 for k in ev_names if deg[k] >= 1),
        "event_hubs_2plus": sum(1 for k in ev_names if deg[k] >= 2),
        "event_edge_share": sum(deg.values()) / n_f,
    }


_WORD_RE = re.compile(r"[a-z0-9][a-z0-9._/-]+")
_STOP = frozenset(
    "the a an of to for and with in on at is was were by from that this it as".split()
)


def _bag(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall((text or "").lower()) if w not in _STOP}


def side_by_side(results: dict, chunk_ids: list[str], limit: int = 10) -> list[tuple]:
    """Control facts that route through a User node, paired with the dehub fact
    covering the same content (max word-overlap inside the same chunk).

    Content-matched rather than index-matched: the two arms emit different fact
    counts per chunk, so position carries no meaning.
    """
    per_arm = {
        arm: {c: [f for f in r["facts"] if f["chunk"] == c] for c in chunk_ids}
        for arm, r in results.items()
    }
    out: list[tuple] = []
    for cid in chunk_ids:
        picked = 0
        for a in per_arm["control"].get(cid, []):
            if not (is_user_node(a["source"]) or is_user_node(a["target"])):
                continue
            ba = _bag(a["fact"])
            best, best_ov = None, 0.0
            for b in per_arm["dehub"].get(cid, []):
                bb = _bag(b["fact"])
                ov = len(ba & bb) / max(1, len(ba | bb))
                if ov > best_ov:
                    best, best_ov = b, ov
            out.append((cid, a, best if best_ov >= 0.15 else None, best_ov))
            picked += 1
            if picked >= 2:
                break
    out.sort(key=lambda t: -t[3])
    return out[:limit]


def build_table(results: dict, n: int, seed: int) -> str:
    arms = list(results)
    lines = [
        f"\n===== de-hub extraction-prompt A/B — {MODEL} (n={n} chunks, seed={seed}) =====",
        "arms differ ONLY in _EXTRACTION_PROMPT (control = origin/main)",
    ]
    hdr = f"{'metric':<26} " + " ".join(f"{a:>12}" for a in arms) + f" {'delta':>12}"
    lines += [hdr, "-" * len(hdr)]

    def row(label, fn, fmt="{:.3f}", delta=True):
        vals, raw = [], []
        for a in arms:
            v = fn(results[a])
            raw.append(v)
            vals.append("-" if v is None else (fmt.format(v) if isinstance(v, float) else str(v)))
        d = ""
        if delta and len(raw) == 2 and all(isinstance(x, (int, float)) for x in raw):
            if raw[0]:
                d = f"{(raw[1] - raw[0]) / abs(raw[0]) * 100:+.0f}%"
            else:
                d = f"{raw[1] - raw[0]:+g}"
        lines.append(f"{label:<26} " + " ".join(f"{v:>12}" for v in vals) + f" {d:>12}")

    def s(r, k):
        return r.get("struct", {}).get(k)

    lines.append("-- volume / quality --")
    row("units ok", lambda r: r["units"], "{}", delta=False)
    row("full failures", lambda r: r["full_failures"], "{}", delta=False)
    row("raw facts/chunk", lambda r: r["n_facts"] / n, "{:.2f}")
    row(
        "dropped-fact rate",
        lambda r: r["dropped_facts"] / max(1, r["n_facts"] + r["dropped_facts"]),
    )
    row("unique facts/chunk", lambda r: r["uniq"] / n, "{:.2f}")
    row("high-value rate", lambda r: r["high_value_rate"])
    row(
        "QUALITY-unique/chunk",
        lambda r: (r["uniq"] / n * r["high_value_rate"]) if r["high_value_rate"] else None,
        "{:.2f}",
    )
    row("entities/chunk", lambda r: r["ents"] / n, "{:.2f}")
    lines.append("-- structure (the point) --")
    row("USER-EDGE rate", lambda r: s(r, "user_edge_rate"))
    row("user-as-source rate", lambda r: s(r, "user_source_rate"))
    row("de-hubbed rate", lambda r: s(r, "dehubbed_rate"))
    row("event nodes (total)", lambda r: s(r, "event_nodes"), "{}")
    row("event hubs (>=2 edges)", lambda r: s(r, "event_hubs_2plus"), "{}")
    row("event-edge share", lambda r: s(r, "event_edge_share"))
    lines.append("-- retrieval guardrails --")
    row("text mentions user", lambda r: s(r, "text_user_rate"))
    row("fact carries a date", lambda r: s(r, "dated_rate"))
    row("date '(meaning …)' rate", lambda r: s(r, "anchored_rate"))
    lines.append("-- cost / latency --")
    row("latency p50 (s)", lambda r: r["lat_p50"], "{:.1f}")
    row("latency p90 (s)", lambda r: r["lat_p90"], "{:.1f}")
    row("est $/1K chunks", lambda r: r["cost_1k"], "{:.2f}")
    return "\n".join(lines)


def paired_recompute(path: Path, dup_thr: float) -> str:
    """Offline salvage: re-score a run on the chunks BOTH arms completed.

    Needed when an arm dies part-way (run 2 hit OpenRouter HTTP 402 — the
    account ran out of credits at chunk ~22/50), which makes the raw per-chunk
    columns incomparable. Recomputes volume + structure on the intersection of
    fact-bearing chunks and re-runs the Voyage dedup on that subset. NO LLM
    calls — the blind judge cannot be redone offline, so ``high-value rate``
    stays whatever the original run measured (None if the judge also 402'd).
    """
    d = orjson.loads(path.read_bytes())
    res = d["results"]
    arms = list(res)
    chunk_sets = [{f["chunk"] for f in res[a]["facts"]} for a in arms]
    paired = sorted(set.intersection(*chunk_sets))
    np_ = len(paired)
    out = {}
    from ingestion.embedding import VoyageEmbeddingModel

    embedder = VoyageEmbeddingModel(api_key=os.environ["VOYAGE_API_KEY"])
    for a in arms:
        facts = [f for f in res[a]["facts"] if f["chunk"] in paired]
        ents = [e for e in res[a]["entities"] if e["chunk"] in paired]
        texts = [f["fact"] for f in facts]
        out[a] = {
            "units": np_,
            "full_failures": res[a]["full_failures"],
            "n_facts": len(facts),
            "ents": len(ents),
            "dropped_facts": res[a]["dropped_facts"],
            "uniq": _dedup_count(embedder.embed(texts, task="document"), dup_thr) if texts else 0,
            "high_value_rate": res[a].get("high_value_rate"),
            "lat_p50": res[a]["lat_p50"],
            "lat_p90": res[a]["lat_p90"],
            "cost_1k": res[a]["cost_1k"],
            "struct": structural_metrics(facts, ents),
        }
    d["paired"] = {"chunks": paired, "n_paired": np_, "results": out}
    path.write_text(orjson.dumps(d, option=orjson.OPT_INDENT_2).decode())
    return build_table(out, np_, int(d["params"].get("seed", 0)))


def or_usage(key: str) -> float | None:
    """Lifetime OpenRouter spend for the key — snapshotted around the run to
    report MEASURED cost instead of a chars/4 estimate."""
    try:
        import httpx

        r = httpx.get(f"{OR_BASE}/auth/key", headers={"Authorization": f"Bearer {key}"}, timeout=20)
        return float(r.json()["data"]["usage"])
    except Exception as e:
        print(f"  (usage snapshot failed: {type(e).__name__}: {str(e)[:60]})")
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--judge-n", type=int, default=40, help="facts/arm to judge (0=skip)")
    ap.add_argument("--dup-thr", type=float, default=0.84)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--control-ref", default=CONTROL_REF)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--paired",
        default=None,
        help="offline: re-score an existing run json on the chunks both arms completed",
    )
    args = ap.parse_args()
    _load_env()

    if args.paired:
        print(paired_recompute(Path(args.paired), args.dup_thr))
        return

    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("SYNAPSE_LLM_API_KEY", "")
    if not key:
        raise SystemExit("need OPENROUTER_API_KEY (or SYNAPSE_LLM_API_KEY)")
    os.environ.pop("SYNAPSE_OPENROUTER_PROVIDERS", None)

    prompts = {"control": prompt_at_ref(args.control_ref), "dehub": _EXTRACTION_PROMPT}
    if prompts["control"] == prompts["dehub"]:
        raise SystemExit("control and dehub prompts are identical — nothing to A/B")
    print(
        f"prompts: control={len(prompts['control'])} chars ({args.control_ref}), "
        f"dehub={len(prompts['dehub'])} chars (working tree)",
        flush=True,
    )

    # --- frozen chunk sample (read-only; identical query to ab_or_extract.py) ---
    conn = psycopg.connect(os.environ["SYNAPSE_DB_URL"], row_factory=dict_row)
    conn.execute("SET statement_timeout = 0")
    conn.execute("SET default_transaction_read_only = on")
    chunks = conn.execute(
        """SELECT c.id, c.session_id, c.start_sequence, c.end_sequence, c.content
           FROM chunks c
           WHERE c.session_id IN (SELECT DISTINCT session_id FROM episodes
                                  WHERE project = ANY(%s))
           ORDER BY md5(c.id::text || %s) LIMIT %s""",
        (list(PII_FREE), str(args.seed), args.n),
    ).fetchall()
    for ch in chunks:
        ch["eps"] = conn.execute(
            "SELECT sequence, content, metadata, created_at FROM episodes "
            "WHERE session_id=%s AND sequence BETWEEN %s AND %s ORDER BY sequence",
            (ch["session_id"], ch["start_sequence"], ch["end_sequence"]),
        ).fetchall()
        # prod anchors the date rules on the segment's latest episode time
        ts = [e["created_at"] for e in ch["eps"] if e.get("created_at")]
        ch["date"] = max(ts).date().isoformat() if ts else None
    conn.close()
    n = len(chunks)
    print(f"{n} chunks sampled (seed={args.seed})", flush=True)

    det = DeterministicExtractor()
    for ch in chunks:
        ch["det"] = det.extract(ch["eps"]) if ch["eps"] else []

    drop_counter = DropCounter()
    exlog = logging.getLogger("ingestion.extractor")
    exlog.addHandler(drop_counter)
    exlog.setLevel(logging.INFO)

    usage_start = or_usage(key)
    results: dict[str, dict] = {}
    for arm, template in prompts.items():
        client = OpenAIChatClient(base_url=OR_BASE, api_key=key, model=args.model)
        ex = PromptExtractor(llm_client=client, model=args.model, template=template)
        d0, f0 = drop_counter.dropped_facts, drop_counter.full_failures
        lats: list[float] = []
        lat_lock = threading.Lock()

        def run(ch, _ex=ex, _lats=lats, _lock=lat_lock):
            t = time.monotonic()
            try:
                r = _ex.extract(
                    summary=ch["content"], context_entities=ch["det"], session_date=ch["date"]
                )
                dt = time.monotonic() - t
                with _lock:
                    _lats.append(dt)
                return {
                    "ok": True,
                    "chunk": str(ch["id"]),
                    "facts": [
                        {
                            "chunk": str(ch["id"]),
                            "source": f.source,
                            "target": f.target,
                            "relationship": f.relationship,
                            "fact": f.fact,
                        }
                        for f in (r.facts or [])
                    ],
                    "entities": [
                        {"chunk": str(ch["id"]), "name": e.name, "type": e.type}
                        for e in (r.entities or [])
                    ],
                    "in_ch": len(ch["content"]),
                    "out_ch": sum(len(f.fact) for f in (r.facts or []))
                    + sum(len(e.summary or "") for e in (r.entities or [])),
                }
            except Exception as e:
                return {
                    "ok": False,
                    "chunk": str(ch["id"]),
                    "err": f"{type(e).__name__}: {str(e)[:140]}",
                }

        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            out = list(pool.map(run, chunks))
        wall = time.monotonic() - t0

        recs = [f for r in out if r.get("ok") for f in r["facts"]]
        ents = [e for r in out if r.get("ok") for e in r["entities"]]
        errs = [r["err"] for r in out if not r.get("ok")]
        in_ch = sum(r.get("in_ch", 0) for r in out if r.get("ok"))
        out_ch = sum(r.get("out_ch", 0) for r in out if r.get("ok"))
        units = sum(1 for r in out if r.get("ok"))
        est_in = in_ch / 4 + units * (len(template) / 4)
        est_out = out_ch / 4
        pi, po = MODEL_PRICE
        results[arm] = {
            "model": args.model,
            "prompt_chars": len(template),
            "units": units,
            "unit_errors": errs,
            "facts": recs,
            "entities": ents,
            "n_facts": len(recs),
            "ents": len(ents),
            "dropped_facts": drop_counter.dropped_facts - d0,
            "full_failures": drop_counter.full_failures - f0,
            "est_in_tok": est_in,
            "est_out_tok": est_out,
            "cost_1k": (est_in / max(1, units) * pi + est_out / max(1, units) * po) / 1e6 * 1000,
            "lat_p50": float(np.percentile(lats, 50)) if lats else 0.0,
            "lat_p90": float(np.percentile(lats, 90)) if lats else 0.0,
            "wall_s": wall,
            "struct": structural_metrics(recs, ents),
        }
        st = results[arm]["struct"]
        print(
            f"[{arm}] {units}/{n} units, {len(recs)} facts, "
            f"{results[arm]['full_failures']} fails, user-edge {st.get('user_edge_rate', 0):.3f}, "
            f"{wall:.0f}s",
            flush=True,
        )

    # --- Voyage dedup on fact TEXT (same threshold as every prior run) ---
    from ingestion.embedding import VoyageEmbeddingModel

    embedder = VoyageEmbeddingModel(api_key=os.environ["VOYAGE_API_KEY"])
    for r in results.values():
        texts = [f["fact"] for f in r["facts"]]
        r["uniq"] = (
            _dedup_count(embedder.embed(texts, task="document"), args.dup_thr) if texts else 0
        )

    # --- blind Haiku judge (fact text only, same model + prompt per arm) ---
    judge = OpenAIChatClient(base_url=OR_BASE, api_key=key, model=JUDGE_MODEL)
    judge.openrouter_providers = []
    rng = random.Random(args.seed)
    for arm, r in results.items():
        r["high_value_rate"] = None
        texts = [f["fact"] for f in r["facts"]]
        if args.judge_n <= 0 or not texts:
            continue
        sample = texts if len(texts) <= args.judge_n else rng.sample(texts, args.judge_n)
        ratings: list[int] = []
        for i in range(0, len(sample), 15):
            try:
                ratings += judge_quality(judge, sample[i : i + 15])
            except Exception as e:
                print(f"  judge error [{arm}]: {type(e).__name__}: {str(e)[:80]}", flush=True)
        r["high_value_rate"] = (sum(ratings) / len(ratings)) if ratings else None
        r["judged_n"] = len(ratings)

    usage_end = or_usage(key)
    measured = None if (usage_start is None or usage_end is None) else usage_end - usage_start

    print(build_table(results, n, args.seed))
    if measured is not None:
        print(f"\nMEASURED OpenRouter spend for this run: ${measured:.4f} (both arms + judge)")

    # --- side-by-side examples: same chunk, both arms ---
    print("\n--- side-by-side (same chunk, control vs dehub) ---")
    print("control facts that touch a User node, matched to the dehub fact about")
    print("the same content (best word-overlap within the chunk):")
    examples = side_by_side(results, [str(c["id"]) for c in chunks], limit=10)
    for i, (cid, a, b, ov) in enumerate(examples, 1):
        print(f"\n[{i}] chunk {cid[:8]} (overlap {ov:.2f})")
        print(f"  CONTROL ({a['source']}) -[{a['relationship']}]-> ({a['target']})")
        print(f"          {a['fact'][:200]}")
        if b:
            print(f"  DEHUB   ({b['source']}) -[{b['relationship']}]-> ({b['target']})")
            print(f"          {b['fact'][:200]}")
        else:
            print("  DEHUB   (no comparable fact in this chunk)")

    outdir = REPO / "scripts" / "kg_eval_runs"
    outdir.mkdir(exist_ok=True)
    out_path = Path(args.out) if args.out else outdir / f"_ab_dehub_seed{args.seed}_n{n}.json"
    dump = {
        "params": vars(args) | {"n_sampled": n, "control_ref": args.control_ref},
        "measured_cost_usd": measured,
        "prompt_chars": {a: len(p) for a, p in prompts.items()},
        "results": results,
        "examples": [
            {"chunk": cid, "control": a, "dehub": b, "overlap": ov} for cid, a, b, ov in examples
        ],
    }
    out_path.write_text(orjson.dumps(dump, option=orjson.OPT_INDENT_2).decode())
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
