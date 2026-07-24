#!/usr/bin/env python3
# mypy: ignore-errors
"""dream->skills v2 — WEEKLY procedure-fingerprint miner (deterministic detector).

The other detectors judge transcripts with an LLM; this one counts. It normalizes shell
commands ([tool:Bash] markers) and MCP tool calls from the chunk+episode union into
FINGERPRINTS, counts DISTINCT sessions per fingerprint, clusters fingerprints that
repeatedly co-occur (same session, nearby in the episode stream), and only then spends a
small LLM budget confirming the top clusters as coherent repeatable procedures. Confirmed
clusters become 'derive' candidates in the skills_lane ledger (signal='procedure_freq');
clusters that extend an existing skill become 'retune' candidates instead (update-first).
The confirm pass also labels each procedure's disposition: fixed sequences with no
branching judgment are flagged "[script-candidate]" in the summary (wrapper-script
material, routed by human review — no script generation happens here).

Normalization rules (validated against the live corpus, 2026-07):
  - ssh <host> "<remote>"      -> ssh:<host>><remote fingerprint or command head>
  - docker exec <ctr> <bin>    -> docker-exec:<ctr>:<bin> [subcommand, flags dropped]
  - docker <verb> [<target>]   -> docker:<verb>:<target>
  - <user scripts dir>/x.sh    -> script:<basename>
  - git/gh/himalaya/systemctl/journalctl/crontab -> <cmd>:<subcommand>
  - curl <url>                 -> curl:<url normalized: query stripped, ids collapsed>
  - MCP tool markers           -> tool:mcp:<server>:<tool>

Exclusion rules (the rejected-procedure taxonomy, encoded):
  - single-command "procedures" never survive (MIN_STEPS >= 2): one script call is already
    covered by the script itself.
  - MCP-native tool calls (tool:* fingerprints) never cluster: the MCP tool IS the
    procedure; a skill would only restate its description.
  - generic dev-workflow fingerprints (git/gh basics, venv activation, python -c
    one-liners, crontab list ritual) are stoplisted — universal background noise.
  - clusters an existing skill already covers are skipped (deterministic registry
    prescreen + LLM covered_by), or routed to a retune when they clearly extend one.

Cadence is the CALLER's job: the nightly integration calls run() only when due() says the
last pass is >= 7 days old (watermark lives in skill_scan_cursor.config['procedure_miner']).
run() is idempotent within a scan night — evidence entries carry scan_night and the ledger
dedups on (session_id, signal, class, scan_night), so a re-run cannot double evidence.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import date

from . import config, skill_ledger, skill_measure

# ------------------------------------------------------------------------- knobs
WINDOW_DAYS = 35  # look-back for the chunk+episode union
MIN_SESSIONS = 4  # distinct-session floor for a cluster (design §8)
MIN_STEPS = 2  # single-command procedures are script-covered — never propose them
EDGE_MIN_SESSIONS = 2  # a fingerprint pair must co-occur in this many sessions to link
PROXIMITY = 8  # max episode-sequence distance for two fingerprints to "co-occur"
MAX_STEPS = 10  # steps kept per cluster (ranked by distinct-session count)
MAX_CONFIRM = 8  # LLM confirm budget per run
EVIDENCE_SESSION_CAP = 10  # evidence entries per candidate (one per distinct session)
HIGH_SALIENCE_SESSIONS = 8  # >= this many distinct sessions -> salience 4 (else 3)
COVERAGE_RATIO = 0.6  # registry-prescreen: fraction of head tokens one skill must mention

SIGNAL = "procedure_freq"
DETECTOR = "procedure_miner"
EXCLUDED_PROJECTS = ("transcribe_ai",)  # foreign skill environment (has its own catalog)

# Generic dev-workflow stoplist: these fingerprints appear in most sessions and are not a
# nameable procedure on their own (git/gh basics, venv activation, one-liners, cron ritual).
GENERIC_PREFIXES = ("git:", "bin:gh:")
GENERIC_FINGERPRINTS = frozenset(
    {"venv:use", "python:-c", "python3:-c", "crontab:-l", "crontab:-e"}
)

_GIT_SUBCOMMANDS = frozenset(
    {
        "status",
        "log",
        "diff",
        "add",
        "commit",
        "push",
        "pull",
        "stash",
        "checkout",
        "branch",
        "fetch",
        "rebase",
        "worktree",
        "show",
    }
)
_KNOWN_BINARIES = frozenset(
    {
        "yt-dlp",
        "ffmpeg",
        "sqlite3",
        "jq",
        "tailscale",
        "wg",
        "smartctl",
        "zpool",
        "zfs",
        "apt",
        "apt-get",
        "pip",
        "pip3",
        "gh",
        "pandoc",
        "free",
        "df",
        "nvidia-smi",
        "sensors",
        "uptime",
    }
)
_HARNESS_TOOLS = frozenset({"Agent", "CronCreate", "CronList", "SendMessage"})
_HOME = r"(?:~|\$HOME|/home/\w+)"
_TOOL_MARKER = re.compile(r"\[tool:(\w+)\]\s*([^\n]*(?:\n(?!\[)[^\n]*)*)")
_SHELL_BREAK = frozenset({"&&", ";", "|", "||"})


def eligible(fp: str) -> bool:
    """Fingerprints allowed to participate in clustering (exclusion rules above)."""
    if fp.startswith("tool:"):  # MCP-native + harness tools
        return False
    if fp.startswith(GENERIC_PREFIXES):
        return False
    return fp not in GENERIC_FINGERPRINTS


# ---------------------------------------------------------------- fingerprinting
def _norm_url(u: str) -> str:
    u = re.sub(r"\?.*$", "", u)
    u = re.sub(r"/\d+", "/N", u)
    u = re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f-]{20,}", "UUID", u)
    return u[:80]


def _first_target(tokens: list[str]) -> str:
    """First non-flag token, stopping at a shell operator (&&, ;, |)."""
    for t in tokens:
        if t in _SHELL_BREAK:
            return ""
        if not t.startswith("-") and re.fullmatch(r"[\w./:-]+", t):
            return t
    return ""


def normalize_command(cmd: str, depth: int = 0) -> list[str]:
    """Turn one shell command atom into zero or more fingerprints."""
    out: list[str] = []
    cmd = cmd.strip()
    if not cmd or len(cmd) < 3:
        return out
    # ssh <host> <remote...> -> recurse into the remote command
    m = re.match(r"ssh\s+(?:-\S+\s+)*([\w.@-]+)\s+(.*)", cmd, re.S)
    if m and depth < 2:
        host, remote = m.group(1), m.group(2).strip().strip("\"'")
        sub = normalize_command(remote, depth + 1)
        if sub:
            return [f"ssh:{host}>{s}" for s in sub[:2]]
        head = remote.split()[0] if remote.split() else "?"
        return [f"ssh:{host}>{head}"]
    # docker exec <container> <binary> [<subcommand>] (flags dropped)
    m = re.match(r"docker\s+exec\s+(?:-\S+\s+)*([\w-]+)\s+(\S+)(?:\s+(\S+))?", cmd)
    if m:
        nxt = m.group(3) or ""
        sub = "" if (not nxt or nxt.startswith("-")) else f" {nxt}"
        return [f"docker-exec:{m.group(1)}:{m.group(2)}{sub}"]
    # docker <verb> [<target>]
    m = re.match(
        r"docker\s+(compose\s+\S+|logs|restart|ps|stats|inspect|start|stop|build|image|images|network|cp)\b",
        cmd,
    )
    if m:
        toks = cmd.split()
        skip = 3 if cmd.startswith("docker compose") else 2
        return [f"docker:{m.group(1)}:{_first_target(toks[skip:])}"[:70].rstrip(":")]
    # user script by basename
    m = re.match(rf"(?:python3?\s+|bash\s+|sh\s+|\.\s+|source\s+)?{_HOME}/scripts/([\w.-]+)", cmd)
    if m:
        return [f"script:{m.group(1)}"]
    m = re.match(r"himalaya\s+(?:-\S+\s+)*(\S+)", cmd)
    if m:
        return [f"himalaya:{m.group(1)}"]
    m = re.match(r"rclone\s+(\S+)\s*(\S*)", cmd)
    if m:
        return [f"rclone:{m.group(1)}:{m.group(2)[:30]}".rstrip(":")]
    # piped-or-direct psql
    if re.search(r"\bpsql\b", cmd) and "psql" in cmd.split("|")[-1][:30]:
        return ["psql:direct"]
    m = re.match(r"curl\s+(?:-\S+\s+|--\S+(?:=\S+)?\s+)*[\"']?(https?://[^\s\"']+)", cmd)
    if m:
        return [f"curl:{_norm_url(m.group(1))}"]
    m = re.match(r"(?:sudo\s+)?systemctl\s+(?:--user\s+)?(\S+)\s*([\w@.-]*)", cmd)
    if m:
        return [f"systemctl:{m.group(1)}:{m.group(2)}".rstrip(":")]
    if re.match(r"(?:sudo\s+)?journalctl\b", cmd):
        return ["journalctl"]
    m = re.match(r"crontab\s+(\S+)", cmd)
    if m:
        return [f"crontab:{m.group(1)}"]
    m = re.match(r"git\s+(?:-C\s+\S+\s+)?(\S+)", cmd)
    if m and m.group(1) in _GIT_SUBCOMMANDS:
        return [f"git:{m.group(1)}"]
    # project venv use (activate / venv python / venv pip)
    if re.match(rf"(?:source\s+)?{_HOME}/venv/bin/(?:activate|python3?|pip3?)\b", cmd):
        return ["venv:use"]
    # runtime + first-arg basename
    m = re.match(r"(python3?|node|npx|npm|flutter|dart|cargo|go)\s+(\S+)", cmd)
    if m:
        return [f"{m.group(1)}:{re.sub(r'.*/', '', m.group(2))[:40]}"]
    m = re.match(r"tmux\s+(\S+)", cmd)
    if m:
        return [f"tmux:{m.group(1)}"]
    # infra CLIs with verb + target
    m = re.match(r"(kubectl|pvesh|pct|qm|pveam|vzdump)\s+(\S+)\s*(\S*)", cmd)
    if m:
        return [f"{m.group(1)}:{m.group(2)}:{m.group(3)[:30]}".rstrip(":")]
    # known-interesting bare binaries
    toks = cmd.split()
    if toks and toks[0] in _KNOWN_BINARIES:
        return [f"bin:{toks[0]}:{_first_target(toks[1:])}".rstrip(":")]
    return out


def split_atoms(bashtext: str) -> list[str]:
    """Split a Bash tool invocation into command atoms (&& / ; / newlines).
    ssh lines stay whole so the remote command survives for the ssh rule."""
    atoms: list[str] = []
    for line in bashtext.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("ssh "):
            atoms.append(line)
            continue
        atoms.extend(p.strip() for p in re.split(r"\s*(?:&&|;)\s*", line) if p.strip())
    return atoms


def content_fingerprints(content: str) -> list[tuple[str, str | None]]:
    """(fingerprint, example_atom) pairs from one episode/chunk content blob."""
    out: list[tuple[str, str | None]] = []
    for m in _TOOL_MARKER.finditer(content or ""):
        tool, args = m.group(1), m.group(2)
        if tool == "Bash":
            for atom in split_atoms(args[:3000]):
                out.extend((fp, atom) for fp in normalize_command(atom))
        elif tool.startswith("mcp__"):
            mm = re.match(r"mcp__(\w+?)__(\w+)$", tool)
            out.append((f"tool:mcp:{mm.group(1)}:{mm.group(2)}" if mm else f"tool:{tool}", None))
        elif tool in _HARNESS_TOOLS:
            out.append((f"tool:{tool}", None))
    return out


# ----------------------------------------------------------------- corpus (SQL seam)
def _fetch_rows(conn, window_days: float) -> list[tuple[str, int, str, str]]:
    """(session_id, position, date_iso, content) rows from the episode+chunk union.
    Chunks are sliding windows OVER episodes — the union is deduped downstream by
    counting DISTINCT sessions / positions, never raw occurrences."""
    excl = list(dict.fromkeys(EXCLUDED_PROJECTS + tuple(config.EXCLUDE_PROJECTS)))
    cur = conn.cursor()
    rows: list[tuple[str, int, str, str]] = []
    cur.execute(
        """SELECT session_id, sequence, created_at::date::text, content
             FROM episodes
            WHERE platform = 'claude_code'
              AND created_at > now() - (%s || ' days')::interval
              AND (project IS NULL OR project <> ALL(%s))""",
        (str(window_days), excl),
    )
    rows += [(str(s), int(q), d, c or "") for s, q, d, c in cur.fetchall()]
    cur.execute(
        """SELECT session_id, start_sequence, created_at::date::text, content
             FROM chunks
            WHERE created_at > now() - (%s || ' days')::interval
              AND (project IS NULL OR project <> ALL(%s))""",
        (str(window_days), excl),
    )
    rows += [(str(s), int(q), d, c or "") for s, q, d, c in cur.fetchall()]
    return rows


def _occurrences(rows):
    """rows -> (session->fp->positions, session->earliest date, fp->example atoms).
    Position = episode sequence (chunks use start_sequence — same coordinate space), so
    a chunk re-serving an episode's command collapses into the same (session, fp, pos)."""
    sess_fp_pos: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    sess_date: dict[str, str] = {}
    examples: dict[str, list[str]] = defaultdict(list)
    for sid, pos, d, content in rows:
        if d and (sid not in sess_date or d < sess_date[sid]):
            sess_date[sid] = d
        for fp, atom in content_fingerprints(content):
            sess_fp_pos[sid][fp].add(pos)
            if atom and len(examples[fp]) < 2 and atom[:160] not in examples[fp]:
                examples[fp].append(atom[:160])
    return sess_fp_pos, sess_date, examples


def _fp_sessions(sess_fp_pos) -> dict[str, set[str]]:
    """fingerprint -> DISTINCT session ids (the frequency unit)."""
    out: dict[str, set[str]] = defaultdict(set)
    for sid, fps in sess_fp_pos.items():
        for fp in fps:
            out[fp].add(sid)
    return out


# --------------------------------------------------------------------- clustering
def _pairs(sess_fp_pos, proximity: int = PROXIMITY) -> dict[frozenset, set[str]]:
    """Co-occurrence: eligible fingerprint pairs seen within `proximity` positions in the
    same session -> the set of sessions where that happened."""
    pair_sessions: dict[frozenset, set[str]] = defaultdict(set)
    for sid, fps in sess_fp_pos.items():
        items = sorted((pos, fp) for fp, poss in fps.items() if eligible(fp) for pos in poss)
        for i, (p, f) in enumerate(items):
            for q, g in items[i + 1 :]:
                if q - p > proximity:
                    break
                if g != f:
                    pair_sessions[frozenset((f, g))].add(sid)
    return pair_sessions


def _cluster(pair_sessions, fp_sessions, min_sessions: int) -> list[dict]:
    """Union-find over pairs that co-occur in >= EDGE_MIN_SESSIONS sessions. A cluster
    survives with >= MIN_STEPS distinct steps AND >= min_sessions distinct sessions in
    which at least two of its steps co-occurred."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])  # path halving
            x = parent[x]
        return x

    for pair, sids in pair_sessions.items():
        if len(sids) < EDGE_MIN_SESSIONS:
            continue
        a, b = tuple(pair)
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    comps: dict[str, set[str]] = defaultdict(set)
    for fp in list(parent):
        comps[find(fp)].add(fp)

    clusters = []
    for members in comps.values():
        if len(members) < MIN_STEPS:
            continue
        support: set[str] = set()
        for pair, sids in pair_sessions.items():
            if pair <= members:
                support |= sids
        if len(support) < min_sessions:
            continue
        steps = sorted(members, key=lambda f: (-len(fp_sessions.get(f, ())), f))[:MAX_STEPS]
        clusters.append({"steps": steps, "sessions": support})
    clusters.sort(key=lambda c: (-len(c["sessions"]), c["steps"]))
    return clusters


def _session_steps(cluster: dict, fp_pos: dict[str, set[int]]) -> list[str]:
    """The cluster's steps as observed in one session, in first-occurrence order."""
    members = set(cluster["steps"])
    hits = sorted((min(ps), fp) for fp, ps in fp_pos.items() if fp in members and ps)
    return [fp for _, fp in hits]


# ------------------------------------------------------- existing-skill coverage
_STRUCTURAL_TOKENS = frozenset(
    {"ssh", "script", "bin", "tool", "docker", "exec", "mcp", "sh", "py", "bash", "direct"}
)


def _load_registry(conn) -> dict[str, str]:
    cur = conn.cursor()
    cur.execute(
        "SELECT name, COALESCE(description, '') FROM skills_lane.skill_registry "
        "WHERE status = 'active'"
    )
    return {name: desc for name, desc in cur.fetchall()}


def _covered_by_registry(steps: list[str], registry: dict[str, str]) -> str | None:
    """Deterministic prescreen: if one existing skill's name+description mentions most of a
    cluster's distinctive tokens, the procedure is already skill-covered — don't spend LLM
    budget on it. Precision-first; the LLM confirm's covered_by is the backstop."""
    heads = {
        t
        for fp in steps
        for t in re.findall(r"[a-z0-9]+", fp.lower())
        if t not in _STRUCTURAL_TOKENS and not t.isdigit() and len(t) > 1
    }
    if not heads:
        return None
    best_name, best_ratio = None, 0.0
    for name, desc in registry.items():
        hay = f"{name} {desc}".lower()
        ratio = sum(1 for t in heads if t in hay) / len(heads)
        if ratio > best_ratio:
            best_name, best_ratio = name, ratio
    return best_name if best_ratio >= COVERAGE_RATIO else None


# ------------------------------------------------------------------ LLM confirm
_CONFIRM_PROMPT = """You audit an AI coding agent's RECURRING SHELL PROCEDURES, mined deterministically
from {n_sessions} distinct sessions over the last {window_days} days. The cluster below is a set of
normalized command fingerprints that repeatedly co-occur across sessions.

CLUSTER STEPS (fingerprint — distinct sessions seen in):
{steps}

EXAMPLE COMMANDS (verbatim, truncated):
{examples}

PER-SESSION STEP ORDER (one line per session):
{sequences}

EXISTING SKILL CATALOG (name: description):
{catalog}

Do NOT capture: environment-dependent failures; negative claims about tools ("X is broken"
hardens into self-citing refusals); transient errors that resolved; one-off task narratives;
novel debugging of a new problem (not a repeatable procedure); behavioral/preference corrections
(those belong to the config lane, not skills); one-off architecture decisions.

Judge STRICTLY:
1. Is this a COHERENT, REPEATABLE, multi-step procedure — or coincidental co-occurrence?
2. Update-first: if an existing catalog skill ALREADY covers it, put its name in covered_by.
   If the procedure is clearly an EXTENSION of an existing skill (same job, extra steps),
   put that skill's name in extends instead.
3. If it stands alone: short kebab-case name; summarize the STANDARDIZED steps (numbered,
   one line each, concrete commands); note what VARIES between repetitions.
4. Disposition: "script" when the steps are a FIXED sequence with no branching judgment
   (better served by a wrapper script than a skill); "skill" when the procedure carries
   real judgment (symptom-dependent branching, interpretation of output, recovery choices).

Output ONLY a JSON object, no prose:
{{"procedure": true, "name": "kebab-name", "summary": "1. ...\\n2. ...",
"varies": "one sentence", "disposition": "script|skill", "covered_by": null, "extends": null}}"""


def _confirm_llm(prompt: str, model: str | None) -> str:
    """Thin delegator to the lane's shared judge dispatch (skill_measure.run_judge).
    Kept as a named seam so tests can stub the LLM out."""
    return skill_measure.run_judge(prompt, model)


def _confirm(cluster, sess_fp_pos, fp_sessions, examples, catalog, window_days, model):
    step_lines = "\n".join(
        f"- {fp}  ({len(fp_sessions.get(fp, ()))} sessions)" for fp in cluster["steps"]
    )
    ex_lines = (
        "\n".join(f"- {ex}" for fp in cluster["steps"] for ex in examples.get(fp, [])[:2])
        or "(none)"
    )
    seq_lines = []
    for sid in sorted(cluster["sessions"])[:5]:
        steps = _session_steps(cluster, sess_fp_pos.get(sid, {}))
        if steps:
            seq_lines.append("- " + " -> ".join(steps))
    prompt = _CONFIRM_PROMPT.format(
        n_sessions=len(cluster["sessions"]),
        window_days=window_days,
        steps=step_lines,
        examples=ex_lines,
        sequences="\n".join(seq_lines) or "(none)",
        catalog=catalog,
    )
    try:
        raw = _confirm_llm(prompt, model)
    except Exception:
        return None
    return skill_measure._extract_json(raw)


# --------------------------------------------------------------------- evidence
def _evidence(cluster, sess_fp_pos, sess_date, scan_night: str) -> list[dict]:
    """One judge entry per distinct session (sorted for idempotency, capped), the observed
    step sequence as the quote. The ledger dedups on (session_id, signal, class, scan_night)."""
    entries = []
    for sid in sorted(cluster["sessions"])[:EVIDENCE_SESSION_CAP]:
        steps = _session_steps(cluster, sess_fp_pos.get(sid, {}))
        entries.append(
            {
                "class": "judge",
                "signal": SIGNAL,
                "session_id": sid,
                "quote": " -> ".join(steps or cluster["steps"][:4]),
                "scan_night": scan_night,
                "date": sess_date.get(sid),
            }
        )
    return entries


# -------------------------------------------------------------------- watermark
def last_run(conn) -> str | None:
    """ISO date of the last completed pass (None = never ran)."""
    cur = conn.cursor()
    cur.execute(
        "SELECT config->'procedure_miner'->>'last_run' "
        "FROM skills_lane.skill_scan_cursor WHERE id = 1"
    )
    row = cur.fetchone()
    return row[0] if row else None


def due(conn, every_days: int = 7) -> bool:
    """Weekly cadence check for the caller: True when the last pass is >= every_days old."""
    lr = last_run(conn)
    if not lr:
        return True
    return (date.today() - date.fromisoformat(lr)).days >= every_days


def _stamp_watermark(conn, scan_night: str) -> None:
    """Merge (not replace) our watermark into skill_scan_cursor.config — the nightly owns
    the rest of that JSONB and last_scan_at is never touched here."""
    cur = conn.cursor()
    cur.execute(
        """UPDATE skills_lane.skill_scan_cursor
              SET config = COALESCE(config, '{}'::jsonb)
                         || jsonb_build_object('procedure_miner',
                                               jsonb_build_object('last_run', %s::text)),
                  updated_at = now()
            WHERE id = 1""",
        (scan_night,),
    )
    conn.commit()


# -------------------------------------------------------------------------- run
def run(conn, *, window_days=WINDOW_DAYS, min_sessions=MIN_SESSIONS, model=None) -> dict:
    """One weekly pass: fingerprint -> cluster -> confirm -> merge_candidate. Returns stats."""
    scan_night = skill_measure.scan_night()
    rows = _fetch_rows(conn, window_days)
    sess_fp_pos, sess_date, examples = _occurrences(rows)
    fp_sessions = _fp_sessions(sess_fp_pos)
    clusters = _cluster(_pairs(sess_fp_pos), fp_sessions, min_sessions)
    registry = _load_registry(conn)
    catalog = "\n".join(f"- {n}: {d}" for n, d in sorted(registry.items())) or "(none)"

    stats = {
        "scan_night": scan_night,
        "window_days": window_days,
        "sessions": len(sess_fp_pos),
        "fingerprints": len(fp_sessions),
        "clusters": len(clusters),
        "covered": 0,
        "llm_calls": 0,
        "rejected": 0,
        "derived": 0,
        "retuned": 0,
        "script_candidates": 0,
        "candidate_ids": [],
    }

    # deterministic coverage prescreen — don't spend LLM budget on skill-covered clusters
    to_confirm = []
    for c in clusters:
        if _covered_by_registry(c["steps"], registry):
            stats["covered"] += 1
        else:
            to_confirm.append(c)

    for c in to_confirm[:MAX_CONFIRM]:
        verdict = _confirm(c, sess_fp_pos, fp_sessions, examples, catalog, window_days, model)
        stats["llm_calls"] += 1
        if not verdict or not verdict.get("procedure"):
            stats["rejected"] += 1
            continue
        covered = (verdict.get("covered_by") or "").strip()
        if covered and covered in registry:
            stats["covered"] += 1
            continue

        n = len(c["sessions"])
        salience = 4 if n >= HIGH_SALIENCE_SESSIONS else 3
        evidence = _evidence(c, sess_fp_pos, sess_date, scan_night)
        name = re.sub(r"[^a-z0-9-]", "", (verdict.get("name") or "").lower()) or "unnamed-procedure"
        summary = (verdict.get("summary") or "").strip()
        varies = (verdict.get("varies") or "").strip()
        if varies:
            summary = f"{summary}\nVaries between runs: {varies}"
        # script-vs-skill disposition: fixed sequences with no branching judgment belong in a
        # wrapper script, not a SKILL.md — flag them for review routing (no generation here).
        if (verdict.get("disposition") or "skill").strip().lower() == "script":
            summary = f"[script-candidate] {summary}"
            stats["script_candidates"] += 1

        # STABLE signature on every derive: the sorted fingerprint set is the cluster's
        # identity key. Never emit a derive without one — the ledger resolver's signature
        # leg wildcard-merges empty-signature candidates (empty-vs-empty jaccard == 1.0).
        signature = " ".join(sorted(c["steps"]))

        extends = (verdict.get("extends") or "").strip()
        if not extends and name in registry:
            extends = name  # name collision with an existing skill = extension, not a new skill
        if extends and extends in registry:
            res = skill_ledger.merge_candidate(
                conn,
                "retune",
                extends,
                evidence,
                direction="extend",
                target_skills=[extends],
                signature=signature,
                summary=f"extend with recurring procedure ({n} sessions): {summary}",
                salience=salience,
                source_detector=DETECTOR,
                do_embed=False,
            )
            stats["retuned"] += 1
        else:
            res = skill_ledger.merge_candidate(
                conn,
                "derive",
                name,
                evidence,
                signature=signature,
                tools=sorted(c["steps"]),
                summary=summary,
                salience=salience,
                source_detector=DETECTOR,
            )
            stats["derived"] += 1
        stats["candidate_ids"].append(res["id"])

    _stamp_watermark(conn, scan_night)
    return stats


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="weekly procedure-fingerprint miner (one pass)")
    ap.add_argument("--window-days", type=int, default=WINDOW_DAYS)
    ap.add_argument("--min-sessions", type=int, default=MIN_SESSIONS)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    c = skill_ledger.connect()
    try:
        print(
            json.dumps(
                run(
                    c,
                    window_days=args.window_days,
                    min_sessions=args.min_sessions,
                    model=args.model,
                ),
                indent=2,
            )
        )
    finally:
        c.close()
